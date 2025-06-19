import datetime
import logging
import pandas as pd
import numpy as np
import time

from backtest.broker import Broker
from util.strategies_util import calculate_performance_metrics, get_next_weekday 
from strategies.strategy import DailyStrategy, MinuteStrategy 
from manager.db_manager import DBManager 
from manager.data_manager import DataManager

# --- 로거 설정 (스크립트 최상단에서 설정하여 항상 보이도록 함) ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # 테스트 시 DEBUG로 설정하여 모든 로그 출력
class Backtester:
    def __init__(self, api_client, initial_cash):
        self.api_client = api_client
        self.broker = Broker(initial_cash, commission_rate=0.0016, slippage_rate=0.0004) # 커미션 0.16% + 슬리피지 0.04% = 총 0.2%
        self.data_store = {'daily': {}, 'minute': {}} # {stock_code: DataFrame}
        self.portfolio_values = [] # (datetime, value) 튜플 저장
        self.initial_cash = initial_cash
        
        self.daily_strategy: DailyStrategy = None
        self.minute_strategy: MinuteStrategy = None

        self.db_manager = DBManager() # DBManager 인스턴스 생성
        self.data_manager = DataManager() # DBManager 인스턴스 생성
        self.pending_daily_signals = {} # 일봉 전략이 다음 날 실행을 위해 생성한 신호들을 저장

        logging.info(f"백테스터 초기화 완료. 초기 현금: {self.initial_cash:,.0f}원")

    def set_strategies(self, daily_strategy: DailyStrategy = None, minute_strategy: MinuteStrategy = None):
        """
        백테스팅에 사용할 일봉 및 분봉 전략 인스턴스를 설정합니다.
        각 전략 인스턴스는 이미 data_store, broker 등을 주입받은 상태여야 합니다.
        """
        if daily_strategy:
            if not isinstance(daily_strategy, DailyStrategy):
                raise TypeError("daily_strategy는 DailyStrategy 타입의 인스턴스여야 합니다.")
            self.daily_strategy = daily_strategy
            self.daily_strategy._initialize_signals_for_all_stocks() # 새로운 종목 추가 시마다 호출
            logging.info(f"일봉 전략 '{daily_strategy.__class__.__name__}'이(가) 설정되었습니다.")

        if minute_strategy:
            if not isinstance(minute_strategy, MinuteStrategy):
                raise TypeError("minute_strategy는 MinuteStrategy 타입의 인스턴스여야 합니다.")
            self.minute_strategy = minute_strategy
            logging.info(f"분봉 전략 '{minute_strategy.__class__.__name__}'이(가) 설정되었습니다.")

        if not self.daily_strategy and not self.minute_strategy:
            logging.warning("설정된 일봉 또는 분봉 전략이 없습니다. 백테스트가 제대로 동작하지 않을 수 있습니다.")

    def set_broker_stop_loss_params(self, params):
        """Broker의 손절매 파라미터를 설정합니다."""
        if self.broker:
            self.broker.set_stop_loss_params(params)
        else:
            logging.warning("Broker가 초기화되지 않아 손절매 파라미터를 설정할 수 없습니다.")
        
    
    def add_daily_data(self, stock_code, daily_df):
        """백테스터에 종목별 일봉 데이터를 추가합니다."""
        self.data_store['daily'][stock_code] = daily_df
        if self.daily_strategy:
            self.daily_strategy._initialize_signals_for_all_stocks() # 이 부분을 다시 추가

    def get_next_business_day(self, date):
        """일봉 데이터를 기반으로 다음 거래일을 찾습니다."""
        next_day = date + datetime.timedelta(days=1)
        max_attempts = 10 # 최대 10일까지 다음 거래일을 찾아봄 (주말, 공휴일 등 고려)
        
        while max_attempts > 0:
            has_data = False
            for stock_code in self.data_store['daily']:
                daily_df = self.data_store['daily'][stock_code]
                if not daily_df.empty:
                    next_day_normalized = pd.Timestamp(next_day).normalize()
                    if next_day_normalized in daily_df.index:
                        has_data = True
                        break
            
            if has_data:
                return next_day
            
            next_day += datetime.timedelta(days=1)
            max_attempts -= 1
        
        logging.warning(f"{date.strftime('%Y-%m-%d')} 이후 {10}일 이내에 거래일을 찾을 수 없습니다.")
        return None

    def _get_minute_data_for_signal_dates(self, stock_code, signal_date, execution_date):
        """
        매수/매도 시그널이 발생한 날짜와 실행될 거래일의 분봉 데이터를 조회합니다.
        이미 로드된 데이터가 있으면 재사용하고, 없으면 API를 통해 가져옵니다.
        """
        # 이미 로드된 데이터가 있는지 확인
        if (stock_code in self.data_store['minute'] and 
            signal_date in self.data_store['minute'][stock_code] and 
            execution_date in self.data_store['minute'][stock_code]):
            return pd.concat([
                self.data_store['minute'][stock_code][signal_date],
                self.data_store['minute'][stock_code][execution_date]
            ]).sort_index()
        
        # 데이터가 없으면 API에서 가져오기
        minute_df = self.data_manager.cache_minute_ohlcv(
            stock_code, 
            min(signal_date, execution_date), 
            max(signal_date, execution_date)
        )
        
        # 데이터 저장 및 반환
        if not minute_df.empty:
            if stock_code not in self.data_store['minute']:
                self.data_store['minute'][stock_code] = {}
            for date in [signal_date, execution_date]:
                date_data = minute_df[minute_df.index.normalize() == pd.Timestamp(date).normalize()]
                if not date_data.empty:
                    self.data_store['minute'][stock_code][date] = date_data
                    logging.debug(f"{stock_code} 종목의 {date} 분봉 데이터 로드 완료. 데이터 수: {len(date_data)}행")
        
        return minute_df

    def run(self, start_date, end_date):
        portfolio_values = []
        dates = []
        
        # ----------------------------------------------------------------------
        # DB 스키마 생성 및 초기화 (최초 실행 시에만 필요)
        #self.db_manager.execute_sql_file('create_backtesting_schema')
        # ----------------------------------------------------------------------

        all_daily_dates = pd.DatetimeIndex([])
        for stock_code, daily_df in self.data_store['daily'].items():
            if not daily_df.empty:
                all_daily_dates = all_daily_dates.union(pd.DatetimeIndex(daily_df.index).normalize())

        daily_dates_to_process = all_daily_dates[
            (all_daily_dates >= pd.Timestamp(start_date).normalize()) & \
            (all_daily_dates <= pd.Timestamp(end_date).normalize())
        ].sort_values()

        if daily_dates_to_process.empty:
            logging.error("지정된 백테스트 기간 내에 일봉 데이터가 없습니다. 종료합니다.")
            return pd.Series(), {} 

        # --- 백테스트 실행 정보 (run_id)를 먼저 DB에 저장 ---
        run_data = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_cash,
            'final_capital': None, 
            'total_profit_loss': None,
            'cumulative_return': None,
            'max_drawdown': None,
            'strategy_daily': self.daily_strategy.__class__.__name__ if self.daily_strategy else None,
            'strategy_minute': self.minute_strategy.__class__.__name__ if self.minute_strategy else None,
            'params_json_daily': self.daily_strategy.strategy_params if self.daily_strategy else None,
            'params_json_minute': self.minute_strategy.strategy_params if self.minute_strategy else None,
        }
        run_id = self.db_manager.save_backtest_run(run_data)
        if run_id is None:
            logging.error("백테스트 실행 정보를 DB에 저장하는데 실패했습니다. 백테스트 결과가 저장되지 않을 수 있습니다.")
            return pd.Series(), {} 
        # ---------------------------------------------------
        
        # 일별 성능 데이터 저장을 위한 리스트
        performance_records = []
        # 백테스트 시작 시점의 포트폴리오 가치 (initial_cash)를 초기값으로 설정
        # MDD 및 수익률 계산의 기준점이 됩니다.
        previous_day_portfolio_value = self.initial_cash

        for i, current_daily_date_full in enumerate(daily_dates_to_process):
            current_daily_date = current_daily_date_full.date()
            logging.info(f"\n--- 처리 중인 날짜: {current_daily_date.isoformat()} ---")

            # 1. 전날 일봉 전략에서 생성된 '오늘 실행할' 신호들을 처리 (분봉 로직)
            if self.minute_strategy:
                # 전날 생성된 pending_daily_signals를 현재 날짜의 실행 신호로 사용
                signals_to_execute_today = self.pending_daily_signals.copy()
                self.pending_daily_signals = {}  # 실행 후 초기화

                # 매수/매도 신호가 없고 stop_loss_params가 None이면 분봉 로직을 건너뜁니다.
                has_trading_signals = any(signal_info['signal'] in ['buy', 'sell'] for signal_info in signals_to_execute_today.values())
                has_stop_loss = self.broker.stop_loss_params is not None

                if not (has_trading_signals or has_stop_loss):
                    logging.debug(f"[{current_daily_date.isoformat()}] 매수/매도 신호가 없고 손절매가 비활성화되어 있어 분봉 로직을 건너뜁니다.")
                else:
                    # 1-1. 먼저 당일 실행할 신호가 있는 종목들의 분봉 데이터를 모두 로드
                    stocks_to_load = [
                        stock_code for stock_code, signal_info in signals_to_execute_today.items()
                        if signal_info['signal'] in ['buy', 'sell', 'hold'] or has_stop_loss
                    ]
                    
                    logging.info(f"[{current_daily_date.isoformat()}] 분봉 데이터 로드 시작: {len(stocks_to_load)}개 종목")
                    
                    # 모든 필요한 종목의 분봉 데이터를 한 번에 로드
                    for stock_code in stocks_to_load:
                        signal_info = signals_to_execute_today[stock_code]
                        # 매도 신호인데 현재 포지션이 없으면 건너뜁니다.
                        if signal_info['signal'] == 'sell' and self.broker.get_position_size(stock_code) <= 0:
                            logging.debug(f"[{current_daily_date.isoformat()}] {stock_code}: 매도 신호가 있지만 보유 포지션이 없어 분봉 데이터 로드를 건너뜁니다.")
                            continue
                            
                        self._get_minute_data_for_signal_dates(
                            stock_code, 
                            signal_info['signal_date'], 
                            current_daily_date
                        )
                    
                    # 1-2. 모든 시그널을 분봉 전략에 한 번에 업데이트
                    self.minute_strategy.update_signals(signals_to_execute_today)
                    logging.debug(f"[{current_daily_date.isoformat()}] 분봉 전략에 {len(signals_to_execute_today)}개의 시그널 업데이트 완료.")

                    # 1-3. 분봉 매매 로직 실행
                    for stock_code, signal_info in signals_to_execute_today.items():
                        if signal_info['signal'] in ['buy', 'sell', 'hold'] or has_stop_loss:
                            # 매도 신호인데 현재 포지션이 없으면 건너뜁니다.
                            if signal_info['signal'] == 'sell' and self.broker.get_position_size(stock_code) <= 0:
                                logging.debug(f"[{current_daily_date.isoformat()}] {stock_code}: 매도 신호가 있지만 보유 포지션이 없어 매매를 건너뜁니다.")
                                continue

                            # 이미 로드된 분봉 데이터 사용
                            if stock_code in self.data_store['minute'] and current_daily_date in self.data_store['minute'][stock_code]:
                                minute_data_today = self.data_store['minute'][stock_code][current_daily_date]
                                
                                if not minute_data_today.empty:
                                    logging.debug(f"[{current_daily_date.isoformat()}] {stock_code}: {len(minute_data_today)}개의 분봉 데이터로 매매 시도.")
                                    
                                    # 손절매가 활성화된 경우: 모든 분봉에서 RSI 매매 및 손절매 체크
                                    if has_stop_loss:
                                        for minute_dt in minute_data_today.index:
                                            if minute_dt.date() > end_date:
                                                logging.info(f"[{current_daily_date.isoformat()}] 백테스트 종료일 {end_date}를 넘어섰습니다. 백테스트 종료.")
                                                break
                                            
                                            self.minute_strategy.run_minute_logic(stock_code, minute_dt)
                                            
                                            if self.minute_strategy.signals.get(stock_code, {}).get('traded_today', False):
                                                logging.debug(f"[{current_daily_date.isoformat()}] {stock_code}: 분봉 매매 완료 (traded_today=True), 다음 분봉 틱 건너뜁니다.")
                                                break
                                else:
                                    logging.warning(f"[{current_daily_date.isoformat()}] {stock_code}: 시그널({signal_info['signal']})이 있으나 현재 날짜의 분봉 데이터가 없어 매매를 시도할 수 없습니다.")
                            else:
                                logging.warning(f"[{current_daily_date.isoformat()}] {stock_code}: 시그널({signal_info['signal']})이 있으나 분봉 데이터가 로드되지 않았습니다.")
                        else:  # 'hold' 또는 None 시그널인 경우
                            logging.debug(f"[{current_daily_date.isoformat()}] {stock_code}: 시그널이 '{signal_info['signal']}'이므로 분봉 매매 로직을 건너뜁니다.")
            else:
                logging.debug(f"분봉 전략이 설정되지 않아 분봉 로직을 건너뜁니다.")

            # 2. 오늘 일봉 데이터를 기반으로 '내일 실행할' 신호를 생성
            if self.daily_strategy:
                # 매일 시작 시 모든 종목의 'traded_today' 플래그 초기화는 daily_strategy에서 이미 수행됩니다.
                # run_daily_logic이 실행되면 self.daily_strategy.signals가 업데이트됩니다.
                self.daily_strategy.run_daily_logic(current_daily_date)

                # 생성된 신호 중 'buy' 또는 'sell' 신호를 pending_daily_signals에 저장
                for stock_code, signal_info in self.daily_strategy.signals.items():
                    if signal_info['signal'] in ['buy', 'sell', 'hold']: # 'hold'도 포함하여 다음 날에도 계속 감시할 수 있도록 합니다.
                        self.pending_daily_signals[stock_code] = signal_info
                        # 'traded_today' 플래그는 매일 초기화되므로 여기서 특별히 건드릴 필요는 없습니다.
                        # 다음 날 이 신호가 사용될 때, 해당 플래그는 다시 False로 시작해야 합니다.
                        self.pending_daily_signals[stock_code]['traded_today'] = False 
                        # signal_date는 신호가 발생한 current_daily_date로 설정됩니다.
                        self.pending_daily_signals[stock_code]['signal_date'] = current_daily_date


            # 3. 일별 종료 시 포트폴리오 가치 계산 및 기록 (기존 로직 유지)
            # ... (기존 포트폴리오 가치 계산 및 성능 지표 기록 로직)
            current_prices = {}
            for stock_code in self.data_store['daily']:
                daily_bar = self.data_store['daily'][stock_code].loc[self.data_store['daily'][stock_code].index.normalize() == current_daily_date_full.normalize()]
                if not daily_bar.empty:
                    current_prices[stock_code] = daily_bar['close'].iloc[0]
                else:
                    last_valid_idx = self.data_store['daily'][stock_code].index.normalize() <= current_daily_date_full.normalize()
                    if last_valid_idx.any():
                        current_prices[stock_code] = self.data_store['daily'][stock_code].loc[last_valid_idx]['close'].iloc[-1]
                    else:
                        current_prices[stock_code] = 0

            portfolio_value = self.broker.get_portfolio_value(current_prices)
            portfolio_values.append(portfolio_value)
            dates.append(current_daily_date_full)

            daily_profit_loss = portfolio_value - previous_day_portfolio_value
            daily_return = daily_profit_loss / previous_day_portfolio_value if previous_day_portfolio_value != 0 else 0

            current_portfolio_series = pd.Series(portfolio_values, index=dates)
            temp_metrics = calculate_performance_metrics(current_portfolio_series, risk_free_rate=0.03) 

            performance_records.append({
                'run_id': run_id,
                'date': current_daily_date,
                'end_capital': portfolio_value,
                'daily_return': daily_return, 
                'daily_profit_loss': daily_profit_loss,
                'cumulative_return': temp_metrics['total_return'], 
                'drawdown': temp_metrics['mdd'] 
            })

            previous_day_portfolio_value = portfolio_value
            # ----------------------------------

        portfolio_value_series = pd.Series(portfolio_values, index=dates)
        metrics = calculate_performance_metrics(portfolio_value_series, risk_free_rate=0.03)
        
        logging.info("\n=== 백테스트 결과 ===")
        logging.info(f"시작일: {start_date.isoformat()}")
        logging.info(f"종료일: {end_date.isoformat()}")
        logging.info(f"초기자금: {self.initial_cash:,.0f}원")
        logging.info(f"최종 포트폴리오 가치: {portfolio_values[-1]:,.0f}원")
        logging.info(f"총 수익률: {metrics['total_return']*100:.2f}%")
        logging.info(f"연간 수익률: {metrics['annual_return']*100:.2f}%")
        logging.info(f"연간 변동성: {metrics['annual_volatility']*100:.2f}%")
        logging.info(f"샤프 지수: {metrics['sharpe_ratio']:.2f}")
        logging.info(f"최대 낙폭 (MDD): {metrics['mdd']*100:.2f}%")
        logging.info(f"승률: {metrics['win_rate']*100:.2f}%")
        logging.info(f"수익비 (Profit Factor): {metrics['profit_factor']:.2f}")
        
        logging.info("\n--- 최종 포지션 현황 ---")
        if self.broker.positions:
            for stock_code, pos_info in self.broker.positions.items():
                logging.info(f"{stock_code}: 보유수량 {pos_info['size']}주, 평균단가 {pos_info['avg_price']:,.0f}원")
        else:
            logging.info("보유 중인 종목 없음")

        # --- 백테스트 결과 (run_id) 최종 업데이트 ---
        final_run_data = {
            'run_id': run_id,
            'final_capital': portfolio_values[-1],
            'total_profit_loss': portfolio_values[-1] - self.initial_cash,
            'cumulative_return': metrics['total_return'],
            'max_drawdown': metrics['mdd']
        }
        update_sql = """
            UPDATE backtest_run
            SET final_capital = %s,
                total_profit_loss = %s,
                cumulative_return = %s,
                max_drawdown = %s
            WHERE run_id = %s
            """
        self.db_manager.execute_sql(update_sql, (
            final_run_data['final_capital'],
            final_run_data['total_profit_loss'],
            final_run_data['cumulative_return'],
            final_run_data['max_drawdown'],
            run_id
        ))
        logging.info(f"백테스트 실행 요약 (run_id: {run_id}) 최종 결과 업데이트 완료.")
        # ---------------------------------------------

        # --- 거래 내역 (transaction_log) DB에 저장 ---
        trade_records = []
        # 'entry_trade_id'를 정확히 매칭하기 위해 매수 거래의 DB ID를 추적해야 함
        # 여기서는 단순화를 위해 매수 거래의 DB ID를 직접 매칭하지 않고 None으로 둠.
        # 실제 구현에서는 매수 트랜잭션 저장 후 반환되는 ID를 저장하고, 매도 시 해당 ID를 사용해야 합니다.
        # 예시: self.broker.transaction_log 대신 매수/매도 시마다 trade_records에 추가하도록 변경 고려
        
        # 현재는 broker.transaction_log의 튜플 형태를 딕셔너리로 변환하여 사용
        for trade in self.broker.transaction_log:
            trade_type_str = trade[2].upper() 
            trade_amount = trade[4] * trade[3] 
            
            realized_profit_loss = 0
            entry_trade_id = None 

            trade_records.append({
                'run_id': run_id,
                'stock_code': trade[1],
                'trade_type': trade_type_str,
                'trade_price': trade[3],
                'trade_quantity': trade[4],
                'trade_amount': trade_amount, 
                'trade_datetime': trade[0],
                'commission': trade[5],
                'tax': 0, 
                'realized_profit_loss': realized_profit_loss,
                'entry_trade_id': entry_trade_id
            })
        # 거래 내역이 있을 때만 DB 저장 실행
        if trade_records:
            self.db_manager.save_backtest_trade(trade_records)
            logging.info(f"{len(trade_records)}개의 거래 내역을 DB에 저장했습니다.")
        else:
            logging.info("저장할 거래 내역이 없습니다.")
        # --------------------------------------------------

        # --- 일별 성능 지표 (performance_records) DB에 저장 ---
        self.db_manager.save_backtest_performance(performance_records)
        logging.info(f"{len(performance_records)}개의 일별 성능 지표를 DB에 저장했습니다.")
        # -------------------------------------------------------
        
        # DBManager 연결 닫기 (백테스터 실행 종료 시)
        self.db_manager.close()

        return portfolio_value_series, metrics