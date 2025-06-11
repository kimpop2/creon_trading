import datetime
import logging
import pandas as pd
import numpy as np
import time

from backtest.broker_losscut import Broker
from util.utils import calculate_performance_metrics, get_next_weekday 
from strategies.strategy import DailyStrategy, MinuteStrategy 

class Backtester:
    def __init__(self, api_client, initial_cash):
        self.api_client = api_client
        self.broker = Broker(initial_cash, commission_rate=0.0003) # 수수료 0.03%
        self.data_store = {'daily': {}, 'minute': {}} # {stock_code: DataFrame}
        self.portfolio_values = [] # (datetime, value) 튜플 저장
        self.initial_cash = initial_cash
        
        self.daily_strategy: DailyStrategy = None
        self.minute_strategy: MinuteStrategy = None

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

    def _get_minute_data_for_signal_dates(self, stock_code, signal_date):
        """
        매수/매도 시그널이 발생한 날짜와 다음 거래일의 분봉 데이터를 조회합니다.
        (손절이 일봉 기준이므로, 보유 종목의 분봉은 여기서 로드하지 않음)
        """
        # 다음 거래일 찾기
        next_trading_day = self.get_next_business_day(signal_date)
        if next_trading_day is None:
            logging.warning(f"{signal_date} 이후의 다음 거래일을 찾을 수 없습니다 - {stock_code}")
            return pd.DataFrame()
            
        # 시그널 발생일과 다음 거래일의 분봉 데이터 로드
        dates_to_load = [signal_date, next_trading_day]
        
        dfs_to_concat = []
        for date in dates_to_load:
            date_str = date.strftime('%Y%m%d')
            
            # 해당 날짜의 분봉 데이터가 이미 있는지 확인
            if stock_code in self.data_store['minute'] and date in self.data_store['minute'][stock_code]:
                dfs_to_concat.append(self.data_store['minute'][stock_code][date])
                continue
            
            # 해당 날짜가 거래일인지 확인
            daily_df = self.data_store['daily'].get(stock_code)
            if daily_df is not None and not daily_df.empty and pd.Timestamp(date).normalize() in daily_df.index:
                minute_df_day = self.api_client.get_minute_ohlcv(stock_code, date_str, date_str, interval=1)
                time.sleep(0.3)   # API 호출 제한 방지를 위한 대기
                
                if not minute_df_day.empty:
                    if stock_code not in self.data_store['minute']:
                        self.data_store['minute'][stock_code] = {}
                    self.data_store['minute'][stock_code][date] = minute_df_day
                    dfs_to_concat.append(minute_df_day)
                    logging.info(f"{stock_code} 종목의 {date_str} 분봉 데이터 로드 완료. 데이터 수: {len(minute_df_day)}행")
                else:
                    logging.warning(f"{stock_code} 종목의 {date_str} 분봉 데이터가 없습니다 (거래일임에도 불구하고).")
        
        if dfs_to_concat:
            full_df = pd.concat(dfs_to_concat).sort_index()
            return full_df
        return pd.DataFrame()

    def run(self, start_date, end_date):
        portfolio_values = []
        dates = []
        
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

        # Broker의 초기 포트폴리오 가치 설정 (포트폴리오 손절을 위해)
        self.broker.initial_portfolio_value = self.initial_cash 

        for current_daily_date_full in daily_dates_to_process:
            current_daily_date = current_daily_date_full.date()
            logging.info(f"\n--- 처리 중인 날짜: {current_daily_date.isoformat()} ---")

            # 매일 시작 시 모든 종목의 'traded_today' 플래그 초기화
            if self.daily_strategy:
                for stock_code in list(self.daily_strategy.signals.keys()):
                    self.daily_strategy.signals[stock_code]['traded_today'] = False

            # 일봉 전략 로직 실행
            if self.daily_strategy:
                self.daily_strategy.run_daily_logic(current_daily_date)
            
            # 분봉 전략에 최신 시그널 업데이트
            if self.daily_strategy and self.minute_strategy:
                self.minute_strategy.update_signals(self.daily_strategy.signals)

            # 매수/매도 시그널이 있는 종목들에 대해서만 분봉 데이터 처리
            if self.daily_strategy and self.minute_strategy:
                # self.daily_strategy.signals가 비어있지 않은지 확인
                if not self.daily_strategy.signals:
                    logging.debug(f"[{current_daily_date.isoformat()}] 일봉 전략에서 생성된 시그널이 없어 분봉 로직을 건너뜜.")
                
                for stock_code, signal_info in self.daily_strategy.signals.items():
                    if signal_info.get('traded_today', False):
                        continue

                    if signal_info['signal'] in ['buy', 'sell']:
                        signal_date = signal_info['signal_date']
                        
                        next_trading_day_for_signal = self.get_next_business_day(signal_date)
                        if next_trading_day_for_signal and next_trading_day_for_signal == current_daily_date:
                            if signal_info['signal'] == 'sell' and self.broker.get_position_size(stock_code) <= 0:
                                continue

                            minute_data = self._get_minute_data_for_signal_dates(stock_code, signal_date) # API 호출로 분봉 데이터 로드
                            if not minute_data.empty:
                                minute_data_today = minute_data.loc[minute_data.index.normalize() == pd.Timestamp(current_daily_date).normalize()]
                                for minute_dt in minute_data_today.index:
                                    if minute_dt > end_date: 
                                        break
                                    # RSI 분봉 트레이더를 통해 실제 매수/매도 실행
                                    self.minute_strategy.run_minute_logic(stock_code, minute_dt)
                                    # 매수/매도가 완료되면 해당 종목의 분봉 루프는 중단
                                    # (traded_today 플래그가 True가 되면 다음 종목으로 넘어감)
                                    if self.daily_strategy.signals[stock_code]['traded_today']:
                                        break
                            else:
                                logging.warning(f"[{current_daily_date.isoformat()}] {stock_code}: 시그널({signal_info['signal']})이 있으나 분봉 데이터가 없어 매매를 시도할 수 없습니다.")

                        
            # --- 일별 종료 시점: 일봉 종가 기반 손절매 로직 실행 ---
            current_daily_close_prices = {} # 그날의 모든 종목의 일봉 종가
            for stock_code in self.data_store['daily']:
                daily_bar = self.data_store['daily'][stock_code].loc[self.data_store['daily'][stock_code].index.normalize() == current_daily_date_full.normalize()]
                if not daily_bar.empty:
                    current_daily_close_prices[stock_code] = daily_bar['close'].iloc[0]
                else:
                    # 해당 날짜에 데이터가 없으면, 전날 마지막 가격으로 대체 시도
                    last_valid_idx = self.data_store['daily'][stock_code].index.normalize() <= current_daily_date_full.normalize()
                    if last_valid_idx.any():
                        current_daily_close_prices[stock_code] = self.data_store['daily'][stock_code].loc[last_valid_idx]['close'].iloc[-1]
                    else:
                        current_daily_close_prices[stock_code] = 0 # 데이터 없으면 0으로 간주

            # 1. 포트폴리오 최고가 업데이트 (일봉 종가 기준)
            for stock_code, pos_info in list(self.broker.positions.items()):
                if stock_code in current_daily_close_prices and pos_info['size'] > 0:
                    self.broker.update_daily_highest_price(stock_code, current_daily_close_prices[stock_code])

            # 2. 포트폴리오 전체 손절 체크 (일봉 종가 기준)
            # 포트폴리오 손절이 발생하면 모든 포지션 청산 후 다음 날짜로 넘어감
            if self.broker.check_and_execute_portfolio_stop_loss(current_daily_close_prices, current_daily_date_full):
                # 모든 포지션이 청산되었으므로, 해당 날짜의 남은 개별 종목 손절은 건너뜀
                logging.info(f"[{current_daily_date.isoformat()}] 포트폴리오 전체 손절매 발생으로 남은 개별 종목 손절 체크 건너뜜.")
            else:
                # 3. 개별 종목 손절 체크 (일봉 종가 기준)
                # 현재 보유 중인 모든 종목에 대해 손절 체크
                for stock_code, pos_info in list(self.broker.positions.items()):
                    if stock_code in current_daily_close_prices and pos_info['size'] > 0:
                        daily_close = current_daily_close_prices[stock_code]
                        self.broker.check_and_execute_stop_loss(stock_code, daily_close, current_daily_date_full)


            # 일별 종료 시 포트폴리오 가치 계산 및 기록
            portfolio_value = self.broker.get_portfolio_value(current_daily_close_prices)
            portfolio_values.append(portfolio_value)
            dates.append(current_daily_date_full)

        portfolio_value_series = pd.Series(portfolio_values, index=dates)
        metrics = calculate_performance_metrics(portfolio_value_series, risk_free_rate=0.03)
        
        logging.info("\n=== 백테스트 결과 ===")
        logging.info(f"시작일: {start_date.date().isoformat()}")
        logging.info(f"종료일: {end_date.date().isoformat()}")
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
        
        return portfolio_value_series, metrics