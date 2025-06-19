# backtest/backtester.py
import datetime
from datetime import time
import logging
import pandas as pd
import numpy as np
import time as time_module
import sys
import os

from backtest.broker import Broker
from backtest.reporter import Reporter # Reporter 타입 힌트를 위해 남겨둠
from manager.data_manager import DataManager # DataManager 타입 힌트를 위해 남겨둠
from util.strategies_util import calculate_performance_metrics, get_next_weekday 
from strategies.strategy import DailyStrategy, MinuteStrategy 
from manager.data_manager import DataManager # DataManager 타입 힌트를 위해 남겨둠
from selector.stock_selector import StockSelector # StockSelector 타입 힌트를 위해 남겨둠
from api.creon_api import CreonAPIClient

# 현재 스크립트의 경로를 sys.path에 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- 로거 설정 (스크립트 최상단에서 설정하여 항상 보이도록 함) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 테스트 시 DEBUG로 설정하여 모든 로그 출력 - 제거

class Backtester:
    # __init__ 메서드를 외부에서 필요한 인스턴스를 주입받도록 변경
    def __init__(self, api_client: CreonAPIClient, initial_cash: float, 
                 data_manager: DataManager, reporter: Reporter, stock_selector: StockSelector,
                 save_to_db: bool = True):  # DB 저장 여부 파라미터 추가
        self.api_client = api_client
        self.broker = Broker(initial_cash, commission_rate=0.0016, slippage_rate=0.0004) # 커미션 0.16% + 슬리피지 0.04% = 총 0.2%
        self.data_store = {'daily': {}, 'minute': {}} # {stock_code: DataFrame}
        self.portfolio_values = [] # (datetime, value) 튜플 저장
        self.initial_cash = initial_cash
        self.save_to_db = save_to_db  # DB 저장 여부 저장
        
        self.daily_strategy: DailyStrategy = None
        self.minute_strategy: MinuteStrategy = None

        # 외부에서 주입받은 인스턴스를 사용
        self.data_manager = data_manager
        self.reporter = reporter
        self.stock_selector = stock_selector
        
        self.pending_daily_signals = {} # 일봉 전략이 다음 날 실행을 위해 생성한 신호들을 저장

        logging.info(f"백테스터 초기화 완료. 초기 현금: {self.initial_cash:,.0f}원, DB저장: {self.save_to_db}")

    def set_strategies(self, daily_strategy: DailyStrategy = None, minute_strategy: MinuteStrategy = None):
        if daily_strategy:
            self.daily_strategy = daily_strategy
            logging.info(f"일봉 전략 '{daily_strategy.__class__.__name__}' 설정 완료.")
        if minute_strategy:
            self.minute_strategy = minute_strategy
            logging.info(f"분봉 전략 '{minute_strategy.__class__.__name__}' 설정 완료.")

    def set_broker_stop_loss_params(self, params: dict):
        self.broker.set_stop_loss_params(params)
        logging.info("브로커 손절매 파라미터 설정 완료.")

    def add_daily_data(self, stock_code: str, df: pd.DataFrame):
        """백테스트를 위한 일봉 데이터를 추가합니다."""
        if not df.empty:
            self.data_store['daily'][stock_code] = df
            logging.debug(f"일봉 데이터 추가: {stock_code}, {len(df)}행")
        else:
            logging.warning(f"빈 데이터프레임이므로 {stock_code}의 일봉 데이터를 추가하지 않습니다.")

    def add_minute_data(self, stock_code: str, df: pd.DataFrame):
        """백테스트를 위한 분봉 데이터를 추가합니다."""
        if not df.empty:
            self.data_store['minute'][stock_code] = df
            logging.debug(f"분봉 데이터 추가: {stock_code}, {len(df)}행")
        else:
            logging.warning(f"빈 데이터프레임이므로 {stock_code}의 분봉 데이터를 추가하지 않습니다.")

    def run(self, start_date: datetime.date, end_date: datetime.date):
        logging.info(f"백테스트 시작: {start_date} 부터 {end_date} 까지")
        current_date = start_date
        
        # 시작일 이전의 초기 포트폴리오 가치 기록
        initial_portfolio_value = self.broker.get_portfolio_value({}) # 초기 현금만 반영
        self.portfolio_values.append((current_date - datetime.timedelta(days=1), initial_portfolio_value))
        
        while current_date <= end_date:
            # 주말 및 공휴일 (데이터가 없는 날) 건너뛰기
            if current_date.weekday() >= 5: # 토요일(5), 일요일(6)
                logging.debug(f"{current_date.isoformat()}는 주말이므로 건너뜁니다.")
                current_date += datetime.timedelta(days=1)
                continue
            
            # 해당 날짜의 일봉 데이터 확인 (데이터가 없으면 건너뛰기)
            # 모든 종목의 일봉 데이터가 없으면 해당 날짜 스킵 (장개장일이 아닌 경우)
            daily_data_available = False
            for stock_code in self.data_store['daily']:
                if self.daily_strategy and stock_code == self.daily_strategy.strategy_params.get('safe_asset_code'):
                    # 안전자산 코드는 항상 데이터가 있다고 가정하거나 별도 처리
                    daily_data_available = True
                    break
                # Ensure the index is date-only for comparison
                if not self.data_store['daily'][stock_code].empty and \
                   current_date in self.data_store['daily'][stock_code].index.date:
                    daily_data_available = True
                    break
            
            if not daily_data_available:
                logging.info(f"{current_date.isoformat()}는 공휴일이므로 건너뜁니다. (일봉 데이터 없음)")
                current_date += datetime.timedelta(days=1)
                continue

            logging.info(f"\n--- 현재 백테스트 날짜: {current_date.isoformat()} ---")

            # 1. 전날 일봉 전략에서 생성된 '오늘 실행할' 신호들을 처리 (분봉 로직)
            if self.minute_strategy:
                # OpenMinute 전략인지 확인
                is_open_minute_strategy = hasattr(self.minute_strategy, 'strategy_name') and self.minute_strategy.strategy_name == "OpenMinute"
                
                # 전날 생성된 pending_daily_signals를 현재 날짜의 실행 신호로 사용
                signals_to_execute_today = self.pending_daily_signals.copy()
                self.pending_daily_signals = {}  # 실행 후 초기화

                # 매수/매도 신호가 없고 stop_loss_params가 None이면 분봉 로직을 건너뜁니다.
                has_trading_signals = any(signal_info['signal'] in ['buy', 'sell'] for signal_info in signals_to_execute_today.values())
                has_stop_loss = self.broker.stop_loss_params is not None

                if not (has_trading_signals or has_stop_loss):
                    logging.debug(f"[{current_date.isoformat()}] 매수/매도 신호가 없고 손절매가 비활성화되어 있어 분봉 로직을 건너뜁니다.")
                else:
                    if is_open_minute_strategy:
                        # OpenMinute 전략: 분봉 데이터 로딩 없이 9:01에만 매매 실행
                        logging.info(f"[{current_date.isoformat()}] OpenMinute 전략: 분봉 데이터 로딩 없이 9:01에 매매 실행")
                        
                        # 9:01 시간 생성 (첫 분봉 완성 후)
                        trade_time = datetime.datetime.combine(current_date, time(9, 1))
                        
                        # OpenMinute 전략에 신호 업데이트 (target_quantity 정보 포함)
                        self.minute_strategy.update_signals(signals_to_execute_today)
                        
                        # 1-3. 9:01에만 매매 로직 실행
                        stocks_to_trade = set()
                        
                        # 매수/매도 신호가 있는 종목들 추가
                        for stock_code, signal_info in signals_to_execute_today.items():
                            if signal_info.get('signal') in ['buy', 'sell']:
                                stocks_to_trade.add(stock_code)
                        
                        # 9:01에 매매 실행
                        for stock_code in stocks_to_trade:
                            self.minute_strategy.run_minute_logic(trade_time, stock_code)
                    
                    else:
                        # 기존 분봉 전략: 분봉 데이터 로딩 후 매매 실행
                        # 1-1. 먼저 당일 실행할 신호가 있는 종목들의 분봉 데이터를 모두 로드
                        # 실제로 분봉 데이터가 필요한 종목들만 선별
                        stocks_to_load = set()
                        
                        # 매수/매도 신호가 있는 종목들 추가
                        for stock_code, signal_info in signals_to_execute_today.items():
                            if signal_info['signal'] in ['buy', 'sell']:
                                stocks_to_load.add(stock_code)
                        
                        # 현재 보유 중인 종목들 추가 (손절매 체크용)
                        if has_stop_loss:
                            current_positions = set(self.broker.positions.keys())
                            stocks_to_load.update(current_positions)

                        logging.info(f"[{current_date.isoformat()}] 분봉 데이터 로드 시작: {len(stocks_to_load)}개 종목")

                        # 필요한 종목들의 분봉 데이터를 로드
                        for stock_code in stocks_to_load:
                            # signals_to_execute_today에 있는 종목은 해당 신호의 signal_date 사용
                            if stock_code in signals_to_execute_today:
                                signal_info = signals_to_execute_today[stock_code]
                                # 매도 신호인데 현재 포지션이 없으면 건너뜁니다.
                                if signal_info['signal'] == 'sell' and self.broker.get_position_size(stock_code) <= 0:
                                    logging.debug(f"[{current_date.isoformat()}] {stock_code}: 매도 신호가 있지만 보유 포지션이 없어 분봉 데이터 로드를 건너뜁니다.")
                                    continue
                                signal_date = signal_info['signal_date']
                            else:
                                # 현재 보유 중인 종목이지만 신호가 없는 경우 (손절매 체크용)
                                signal_date = current_date

                            # DataManager를 사용하여 분봉 데이터 로드
                            minute_df = self.data_manager.cache_minute_ohlcv(
                                stock_code,
                                signal_date,
                                current_date
                            )

                            # 기존 백테스터와 동일하게 날짜별로 분봉 데이터 저장
                            if not minute_df.empty:
                                if stock_code not in self.data_store['minute']:
                                    self.data_store['minute'][stock_code] = {}
                                for date in [signal_date, current_date]:
                                    date_data = minute_df[minute_df.index.normalize() == pd.Timestamp(date).normalize()]
                                    if not date_data.empty:
                                        self.data_store['minute'][stock_code][date] = date_data
                                        logging.debug(f"{stock_code} 종목의 {date} 분봉 데이터 로드 완료. 데이터 수: {len(date_data)}행")

                        # 1-2. 모든 시그널을 분봉 전략에 한 번에 업데이트
                        self.minute_strategy.update_signals(signals_to_execute_today)
                        logging.debug(f"[{current_date.isoformat()}] 분봉 전략에 {len(signals_to_execute_today)}개의 시그널 업데이트 완료.")

                        # 1-3. 분봉 매매 로직 실행
                        # 실제로 매매가 필요한 종목들만 선별
                        stocks_to_trade = set()
                        # 매수/매도 신호가 있는 종목들 추가
                        for stock_code, signal_info in signals_to_execute_today.items():
                            if signal_info['signal'] in ['buy', 'sell']:
                                stocks_to_trade.add(stock_code)
                        # 현재 보유 중인 종목들 추가 (손절매 체크용)
                        if has_stop_loss:
                            current_positions = set(self.broker.positions.keys())
                            stocks_to_trade.update(current_positions)

                        for stock_code in stocks_to_trade:
                            # signals_to_execute_today에 있는 종목은 해당 신호 정보 사용
                            if stock_code in signals_to_execute_today:
                                signal_info = signals_to_execute_today[stock_code]
                                # 매도 신호인데 현재 포지션이 없으면 건너뜁니다.
                                if signal_info['signal'] == 'sell' and self.broker.get_position_size(stock_code) <= 0:
                                    logging.debug(f"[{current_date.isoformat()}] {stock_code}: 매도 신호가 있지만 보유 포지션이 없어 매매를 건너뜁니다.")
                                    continue
                            else:
                                # 현재 보유 중인 종목이지만 신호가 없는 경우 (손절매 체크용)
                                signal_info = {'signal': 'hold'}

                            # 기존 백테스터와 동일하게 날짜별로 저장된 분봉 데이터 사용
                            if stock_code in self.data_store['minute'] and current_date in self.data_store['minute'][stock_code]:
                                minute_data_today = self.data_store['minute'][stock_code][current_date]
                                if not minute_data_today.empty:
                                    logging.debug(f"[{current_date.isoformat()}] {stock_code}: {len(minute_data_today)}개의 분봉 데이터로 매매 시도.")
                                    # 모든 분봉에서 매매 및 손절매 체크
                                    for minute_dt in minute_data_today.index:
                                        if minute_dt.date() > end_date:
                                            logging.info(f"[{current_date.isoformat()}] 백테스트 종료일 {end_date}를 넘어섰습니다. 백테스트 종료.")
                                            break
                                        self.minute_strategy.run_minute_logic(stock_code, minute_dt)
                                        if self.minute_strategy.signals.get(stock_code, {}).get('traded_today', False):
                                            logging.debug(f"[{current_date.isoformat()}] {stock_code}: 분봉 매매 완료 (traded_today=True), 다음 분봉 틱 건너뜁니다.")
                                            break
                                else:
                                    logging.warning(f"[{current_date.isoformat()}] {stock_code}: 시그널({signal_info['signal']})이 있으나 현재 날짜의 분봉 데이터가 없어 매매를 시도할 수 없습니다.")
                            else:
                                logging.warning(f"[{current_date.isoformat()}] {stock_code}: 시그널({signal_info['signal']})이 있으나 분봉 데이터가 로드되지 않았습니다.")
            else:
                logging.debug(f"분봉 전략이 설정되지 않아 분봉 로직을 건너뜁니다.")

            # 2. 오늘 일봉 데이터를 기반으로 '내일 실행할' 신호를 생성
            if self.daily_strategy:
                # 매일 시작 시 모든 종목의 'traded_today' 플래그 초기화는 daily_strategy에서 이미 수행됩니다.
                # run_daily_logic이 실행되면 self.daily_strategy.signals가 업데이트됩니다.
                self.daily_strategy.run_daily_logic(current_date)

                # 생성된 신호 중 'buy' 또는 'sell' 신호를 pending_daily_signals에 저장
                for stock_code, signal_info in self.daily_strategy.signals.items():
                    if signal_info['signal'] in ['buy', 'sell', 'hold']: # 'hold'도 포함하여 다음 날에도 계속 감시할 수 있도록 합니다.
                        self.pending_daily_signals[stock_code] = signal_info
                        # 'traded_today' 플래그는 매일 초기화되므로 여기서 특별히 건드릴 필요는 없습니다.
                        # 다음 날 이 신호가 사용될 때, 해당 플래그는 다시 False로 시작해야 합니다.
                        self.pending_daily_signals[stock_code]['traded_today'] = False 
                        # signal_date는 신호가 발생한 current_date로 설정됩니다.
                        self.pending_daily_signals[stock_code]['signal_date'] = current_date

            # 3. 브로커 일일 초기화 (예: 당일 거래 가능 여부 초기화)
            self.broker.reset_daily_transactions()
            
            # 4. 일일 포트폴리오 가치 업데이트 (장 마감 기준)
            # 당일 종가를 기준으로 포트폴리오 가치 계산
            current_prices = {}
            for stock_code in self.data_store['daily']:
                daily_bar = self.data_store['daily'][stock_code].loc[self.data_store['daily'][stock_code].index.date == current_date]
                if not daily_bar.empty:
                    current_prices[stock_code] = daily_bar['close'].iloc[0]
                else:
                    last_valid_idx = self.data_store['daily'][stock_code].index.date <= current_date
                    if last_valid_idx.any():
                        current_prices[stock_code] = self.data_store['daily'][stock_code].loc[last_valid_idx]['close'].iloc[-1]
                    else:
                        current_prices[stock_code] = 0
            
            current_portfolio_value = self.broker.get_portfolio_value(current_prices)
            self.portfolio_values.append((current_date, current_portfolio_value))
            logging.info(f"날짜: {current_date.isoformat()}, 포트폴리오 가치: {current_portfolio_value:,.0f}원, 현금: {self.broker.cash:,.0f}원")

            current_date += datetime.timedelta(days=1)
        
        logging.info("백테스트 완료.")
        
        # 최종 결과 및 지표 계산 및 저장 로직을 Reporter로 위임
        self._save_results_to_db(start_date, end_date) # start_date와 end_date를 인자로 전달

        # portfolio_values를 pd.DataFrame으로 변환하여 반환
        portfolio_df = pd.DataFrame(self.portfolio_values, columns=['Date', 'PortfolioValue'])
        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
        
        # 첫 번째 더미 데이터를 제외하고 성능 지표 계산에 사용할 Series 생성
        actual_portfolio_value_series = portfolio_df[portfolio_df['Date'] >= pd.Timestamp(start_date)].set_index('Date')['PortfolioValue']

        final_metrics = calculate_performance_metrics(actual_portfolio_value_series, risk_free_rate=0.03)

        return portfolio_df, final_metrics


    def _save_results_to_db(self, start_date: datetime.date, end_date: datetime.date):
        """백테스트 최종 결과를 DB에 저장합니다."""
        if not self.save_to_db:
            logging.debug("DB 저장이 비활성화되어 있어 결과를 저장하지 않습니다.")
            return
            
        logging.info("백테스트 최종 결과를 DB에 저장 중...")

        # self.portfolio_values를 pandas Series로 변환
        # 첫 번째 항목 (시작일 이전의 초기 자본)은 제외하고 실제 백테스트 기간 데이터만 사용
        if len(self.portfolio_values) > 1:
            portfolio_df = pd.DataFrame(self.portfolio_values[1:], columns=['Date', 'PortfolioValue'])
            portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
            portfolio_value_series = portfolio_df.set_index('Date')['PortfolioValue']
        else:
            # 백테스트 기간이 너무 짧아 포트폴리오 가치 데이터가 없는 경우
            portfolio_value_series = pd.Series(dtype=float)


        # 전략 이름과 파라미터 추출
        daily_strategy_name = self.daily_strategy.__class__.__name__ if self.daily_strategy else "N/A"
        minute_strategy_name = self.minute_strategy.__class__.__name__ if self.minute_strategy else "N/A"
        daily_strategy_params = self.daily_strategy.strategy_params if self.daily_strategy else {}
        minute_strategy_params = self.minute_strategy.strategy_params if self.minute_strategy else {}

        # 모든 저장 로직을 self.reporter에게 위임
        self.reporter.generate_and_save_report(
            start_date=start_date,
            end_date=end_date,
            initial_cash=self.initial_cash,
            portfolio_value_series=portfolio_value_series,
            transaction_log=self.broker.transaction_log,
            daily_strategy_name=daily_strategy_name,
            minute_strategy_name=minute_strategy_name,
            daily_strategy_params=daily_strategy_params,
            minute_strategy_params=minute_strategy_params
        )
        logging.info("백테스트 최종 결과 DB 저장 완료.")