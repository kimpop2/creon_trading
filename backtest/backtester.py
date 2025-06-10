import datetime
import logging
import pandas as pd
import numpy as np
import time

from backtest.broker import Broker
from util.utils import calculate_performance_metrics, get_next_weekday 
# 전략 추상 클래스 임포트
from strategies.strategy_base import DailyStrategy, MinuteStrategy 

class Backtester:
    def __init__(self, api_client, initial_cash):
        self.api_client = api_client
        self.broker = Broker(initial_cash, commission_rate=0.0003) # 수수료 0.03%
        self.data_store = {'daily': {}, 'minute': {}} # {stock_code: DataFrame}
        self.position_info = {} # {stock_code: {'highest_price': float, 'entry_date': datetime.date}}
        self.portfolio_values = [] # (datetime, value) 튜플 저장
        self.initial_cash = initial_cash

        self.strategy_params = {
            'momentum_period': 60, # 60일 모멘텀
            'rebalance_weekday': 4, # 0:월, 4:금 (주간 리밸런싱)
            'num_top_stocks': 5, # 상위 5개 종목
            'safe_asset_code': 'A439870',  # 안전자산 코드 (국고채 ETF) # KODEX 200 (코스피200 ETF) 또는 KODEX 인버스 (경기 침체 시)
            'equal_weight_amount': 2_000_000, # 종목당 2백만원 균등 배분

            'minute_rsi_period': 14, # 분봉 RSI 기간
            'minute_rsi_overbought': 70, # 과매수 기준
            'minute_rsi_oversold': 30, # 과매도 기준
            'stop_loss_ratio': -5, # 일반 손절 -5%
            'trailing_stop_ratio': -3, # 트레일링 스탑 -3% (고점 대비)
            'early_stop_loss': -2, # 매수 후 5거래일 이내 -2% 손절
            'max_losing_positions': 3, # 최대 손실 허용 포지션 개수 (미사용)
            'initial_cash': initial_cash # RSI 전략에서 포트폴리오 손절 계산을 위함 (현재는 사용 안함)
        }
        
        # 단일 전략 인스턴스를 저장하도록 변경
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
            logging.info(f"일봉 전략 '{daily_strategy.__class__.__name__}'이(가) 설정되었습니다.")

        if minute_strategy:
            if not isinstance(minute_strategy, MinuteStrategy):
                raise TypeError("minute_strategy는 MinuteStrategy 타입의 인스턴스여야 합니다.")
            self.minute_strategy = minute_strategy
            logging.info(f"분봉 전략 '{minute_strategy.__class__.__name__}'이(가) 설정되었습니다.")

        if not self.daily_strategy and not self.minute_strategy:
            logging.warning("설정된 일봉 또는 분봉 전략이 없습니다. 백테스트가 제대로 동작하지 않을 수 있습니다.")


    def add_daily_data(self, stock_code, daily_df):
        """
        백테스터에 일봉 데이터를 추가합니다.
        """
        if daily_df.empty:
            logging.warning(f"데이터가 없어 {stock_code}의 일봉 데이터를 추가하지 않습니다.")
            return

        self.data_store['daily'][stock_code] = daily_df
        logging.debug(f"'{stock_code}' 일봉 데이터 {len(daily_df)}행 추가 완료.")


    def _get_minute_data_for_signal_dates(self, signal_dates_and_codes, end_date):
        """
        일봉 전략 시그널 발생일의 다음 거래일 분봉 데이터를 미리 로드합니다.
        trade_date에 해당하는 분봉 데이터만 로드하여 self.data_store['minute']에 저장.
        signal_dates_and_codes: {stock_code: [signal_date1, signal_date2, ...]}
        """
        logging.info("분봉 데이터 로드 시작...")
        self.data_store['minute'] = {} # 기존 분봉 데이터 초기화
        loaded_count = 0
        # total_requests = sum(len(dates) for dates in signal_dates_and_codes.values()) # 더 이상 사용 안함 (단일 전략)

        for stock_code, dates in signal_dates_and_codes.items():
            self.data_store['minute'][stock_code] = {}
            for trade_date in sorted(list(set(dates))): # 중복 날짜 제거 및 정렬
                trade_dt = datetime.datetime.combine(trade_date, datetime.time(9, 0, 0))
                if trade_dt.date() > end_date.date():
                    continue

                if stock_code not in self.data_store['daily']:
                    logging.debug(f"{stock_code}: 일봉 데이터가 없어 분봉 데이터 로드 건너뜀.")
                    continue
                
                if trade_date in self.data_store['minute'][stock_code]:
                    logging.debug(f"{stock_code} {trade_date.isoformat()}: 분봉 데이터가 이미 로드됨.")
                    continue

                from_date_str = trade_date.strftime('%Y%m%d')
                to_date_str = trade_date.strftime('%Y%m%d')

                logging.debug(f"로딩 중: {self.api_client.get_stock_name(stock_code)} ({stock_code}) - {from_date_str}")
                minute_df = self.api_client.get_minute_ohlcv(stock_code, from_date_str, to_date_str, interval=1)
                time.sleep(0.3)

                if not minute_df.empty:
                    self.data_store['minute'][stock_code][trade_date] = minute_df
                    loaded_count += 1
                else:
                    logging.warning(f"'{self.api_client.get_stock_name(stock_code)}' ({stock_code}) {from_date_str} 분봉 데이터 로드 실패 또는 데이터 없음.")
        
        logging.info(f"분봉 데이터 로드 완료. 총 {loaded_count}개 일자/종목 데이터 로드됨.")


    def run(self, start_date, end_date):
        """
        백테스팅을 실행합니다.
        """
        logging.info(f"백테스트 시작: {start_date.isoformat()} ~ {end_date.isoformat()}")
        
        current_date_iter = start_date # 현재 반복 중인 날짜 (datetime 객체)

        # 일봉 전략으로부터의 매매 시그널을 저장할 변수
        dual_momentum_signals = {}
        
        # 일봉 전략 초기화 관련 (전략 내부에서 관리하도록)
        if self.daily_strategy:
            if hasattr(self.daily_strategy, '_initialize_momentum_signals_for_all_stocks'):
                self.daily_strategy._initialize_momentum_signals_for_all_stocks()

        while current_date_iter <= end_date:
            if current_date_iter.weekday() >= 5: # 토요일(5) 또는 일요일(6)
                current_date_iter += datetime.timedelta(days=1)
                continue
            
            current_daily_date = current_date_iter.date() # 현재 일자 (datetime.date 객체)

            # 1. 일봉 전략 실행 (단일 일봉 전략 호출)
            if self.daily_strategy:
                self.daily_strategy.run_daily_logic(current_daily_date)
                # 일봉 전략이 시그널을 생성했다면, 그 시그널을 분봉 전략에 전달하기 위해 저장
                if hasattr(self.daily_strategy, 'momentum_signals'):
                    dual_momentum_signals = self.daily_strategy.momentum_signals 
            else:
                logging.debug(f"[{current_daily_date.isoformat()}] 설정된 일봉 전략이 없습니다.")


            # 2. 분봉 데이터 로드 (매매 시그널이 발생한 날만)
            signal_dates_and_codes = {} # {stock_code: [datetime.date, ...]}
            
            for stock_code, info in dual_momentum_signals.items():
                # 시그널의 날짜가 현재 일봉 날짜와 같고, 매수/매도 시그널인 경우에만 분봉 데이터 로드 대상
                if info['signal'] in ['buy', 'sell'] and info['signal_date'] == current_daily_date:
                    if stock_code not in signal_dates_and_codes:
                        signal_dates_and_codes[stock_code] = []
                    signal_dates_and_codes[stock_code].append(current_daily_date)
            
            # 현재 보유 중인 종목의 분봉 데이터도 로드 (손절/트레일링 스탑을 위해)
            for stock_code in self.broker.positions.keys():
                if self.broker.get_position_size(stock_code) > 0:
                    if stock_code not in signal_dates_and_codes:
                        signal_dates_and_codes[stock_code] = []
                    signal_dates_and_codes[stock_code].append(current_daily_date)

            if signal_dates_and_codes:
                self._get_minute_data_for_signal_dates(signal_dates_and_codes, end_date)
            else:
                logging.debug(f"[{current_daily_date.isoformat()}] 매매 시그널 또는 보유 종목이 없어 분봉 데이터 로드 건너뜀.")

            # 3. 분봉 전략 실행 (오전 9시부터 오후 3시 30분까지)
            market_open_time = datetime.time(9, 0, 0)
            market_close_time = datetime.time(15, 30, 0)
            
            all_minute_times = set()
            # 시그널 종목과 보유 종목의 분봉 데이터를 모두 확인하여 시간을 모음
            target_stock_codes_for_minute = set(signal_dates_and_codes.keys()) | set(self.broker.positions.keys())

            for stock_code in target_stock_codes_for_minute:
                if stock_code in self.data_store['minute'] and current_daily_date in self.data_store['minute'][stock_code]:
                    minute_df = self.data_store['minute'][stock_code][current_daily_date]
                    for dt_index in minute_df.index:
                        if market_open_time <= dt_index.time() <= market_close_time:
                            all_minute_times.add(dt_index)

            sorted_minute_times = sorted(list(all_minute_times))
            
            if not sorted_minute_times:
                logging.debug(f"[{current_daily_date.isoformat()}] 현재 날짜에 처리할 분봉 데이터가 없습니다. 다음 날로 이동.")
                # 시그널이 발생했는데 분봉 데이터가 없으면 'traded_today'를 False로 초기화
                for stock_code in dual_momentum_signals:
                    dual_momentum_signals[stock_code]['traded_today'] = False
                current_date_iter += datetime.timedelta(days=1)
                continue

            for current_minute_dt in sorted_minute_times:
                if not (market_open_time <= current_minute_dt.time() <= market_close_time):
                    continue 
                
                # 단일 분봉 전략에 일봉 시그널 전달
                if self.minute_strategy:
                    self.minute_strategy.update_momentum_signals(dual_momentum_signals)
                
                # 시그널 발생한 종목 및 보유 종목에 대해 단일 분봉 전략 실행
                for stock_code in target_stock_codes_for_minute:
                    if stock_code in self.data_store['minute'] and \
                       current_daily_date in self.data_store['minute'][stock_code] and \
                       current_minute_dt in self.data_store['minute'][stock_code][current_daily_date].index:
                        
                        if self.minute_strategy:
                            self.minute_strategy.run_minute_logic(stock_code, current_minute_dt)
                    else:
                        logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: 현재 시간({current_minute_dt.time()})에 해당하는 분봉 데이터가 없어 분봉 로직을 건너뜀.")


            # 4. 일일 포트폴리오 가치 기록 (장 마감 시점)
            current_prices = {}
            for stock_code, daily_df in self.data_store['daily'].items():
                daily_data_for_date = daily_df.loc[daily_df.index.normalize() == pd.Timestamp(current_daily_date).normalize()]
                if not daily_data_for_date.empty:
                    current_prices[stock_code] = daily_data_for_date['close'].iloc[-1]
                else: 
                    if not daily_df.empty and daily_df.index.normalize().max() < current_daily_date:
                         current_prices[stock_code] = daily_df['close'].iloc[-1]
                    else:
                        current_prices[stock_code] = 0

            portfolio_value = self.broker.get_portfolio_value(current_prices)
            self.portfolio_values.append((current_date_iter, portfolio_value))
            logging.info(f"[{current_date_iter.isoformat()}] 일일 포트폴리오 가치: {portfolio_value:,.0f}원, 현금 잔고: {self.broker.cash:,.0f}원")

            # 5. 다음 날짜로 이동
            current_date_iter += datetime.timedelta(days=1)
        
        logging.info("백테스트 완료.")
        
        if self.portfolio_values:
            portfolio_df = pd.DataFrame(self.portfolio_values, columns=['datetime', 'value'])
            portfolio_df.set_index('datetime', inplace=True)
            portfolio_series = portfolio_df['value']
        else:
            portfolio_series = pd.Series([], dtype='float64')

        metrics = calculate_performance_metrics(portfolio_series, risk_free_rate=0.03) 
        logging.info("--- 백테스팅 최종 결과 ---")
        logging.info(f"총 수익률: {metrics['total_return']:.2%}")
        logging.info(f"연간 수익률: {metrics['annual_return']:.2%}")
        logging.info(f"연간 변동성: {metrics['annual_volatility']:.2%}")
        logging.info(f"샤프 지수: {metrics['sharpe_ratio']:.2f}")
        logging.info(f"최대 낙폭 (MDD): {metrics['mdd']:.2%}")
        logging.info(f"승률: {metrics['win_rate']:.2%}")
        logging.info(f"수익 요인 (Profit Factor): {metrics['profit_factor']:.2f}")
        logging.info("------------------------")
        
        return portfolio_series, metrics