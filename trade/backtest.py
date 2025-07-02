# backtest/backtester.py
import datetime
from datetime import time
import logging
import pandas as pd
import numpy as np
import time as time_module
import sys
import os

from trade.broker import Broker
from trade.backtest_report import BacktestReport 
from manager.backtest_manager import BacktestManager # BacktestManager 타입 힌트를 위해 남겨둠
from util.strategies_util import calculate_performance_metrics, get_next_weekday 
from strategies.strategy import DailyStrategy, MinuteStrategy 
from manager.db_manager import DBManager # BacktestManager 타입 힌트를 위해 남겨둠
from api.creon_api import CreonAPIClient

# --- 로거 설정 (스크립트 최상단에서 설정하여 항상 보이도록 함) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 테스트 시 DEBUG로 설정하여 모든 로그 출력 - 제거

class Backtest:
    # __init__ 메서드를 외부에서 필요한 인스턴스를 주입받도록 변경
    def __init__(self, api_client: CreonAPIClient, initial_cash: float, 
                 backtest_manager: BacktestManager, backtest_report: BacktestReport, db_manager: DBManager,
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
        self.manager = backtest_manager
        self.reporter = backtest_report
        self.db_manager = db_manager
        
        self.current_day_signals = {}

        # NEW: 현재 날짜의 분봉 매매를 위해 사용될 신호들을 저장
        self.signals_for_minute_trading = {}
        # NEW: 포트폴리오 손절 체크 시간을 추적하기 위한 변수
        self.last_portfolio_check = None
        # 포트폴리오 손절 발생 시 당일 매매 중단 플래그
        self.portfolio_stop_flag = False

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


    # 필수 : 포트폴리오 손절시각 체크, 없으면 매분마다 보유종목 손절 체크로 비효율적
    def _should_check_portfolio(self, current_dt):
        """포트폴리오 체크가 필요한 시점인지 확인합니다."""
        if self.last_portfolio_check is None:
            return True
        
        current_time = current_dt.time()
        # 시간 비교를 정수로 변환하여 효율적으로 비교
        current_minutes = current_time.hour * 60 + current_time.minute
        #check_minutes = [9 * 60, 15 * 60 + 20]  # 9:00, 15:20
        check_minutes = [15 * 60 + 20]  # 9:00, 15:20
        
        if current_minutes in check_minutes and (self.last_portfolio_check.date() != current_dt.date() or 
                                               (self.last_portfolio_check.hour * 60 + self.last_portfolio_check.minute) not in check_minutes):
            self.last_portfolio_check = current_dt
            return True
            
        return False
    

    def run(self, start_date: datetime.date, end_date: datetime.date):
        logging.info(f"백테스트 시작: {start_date} 부터 {end_date} 까지")
        current_date = start_date
        
        # 시작일 이전의 초기 포트폴리오 가치 기록
        initial_portfolio_value = self.broker.get_portfolio_value({}) # 초기 현금만 반영
        self.portfolio_values.append((current_date - datetime.timedelta(days=1), initial_portfolio_value))
        
        # 시장 캘린더 데이터 로드
        market_calendar_df = self.db_manager.fetch_market_calendar(start_date, end_date)
        if market_calendar_df.empty:
            logger.error("시장 캘린더 데이터를 가져올 수 없습니다. 백테스트를 중단합니다.")
            return
        # 영업일만 필터링하고 날짜를 리스트로 변환하여 정렬
        trading_dates = market_calendar_df[market_calendar_df['is_holiday'] == 0]['date'].dt.date.tolist()
        trading_dates.sort() # 날짜가 오름차순으로 정렬되도록 보장
        if not trading_dates:
            logger.warning("지정된 기간 내 영업일이 없습니다. daily_universe 채우기를 건너뜁니다.")
            return

        # 영업일을 순회하며 SetupManager 실행
        for current_date in trading_dates:  
            logging.info(f"\n--- 현재 백테스트 날짜: {current_date.isoformat()} ---")

            # 전일 날짜 계산
            current_date_index = trading_dates.index(current_date)
            prev_date = None
            if current_date_index == 0 :
                prev_date = start_date - datetime.timedelta(days=1)
            else:
                prev_date = trading_dates[current_date_index - 1]            # self.data_store 
            
            fetch_date = current_date - datetime.timedelta(days=180)
            self.data_store['daily'] = {} # 보유종목과 유니버스 종목을 다시 담기 위해 초기화
            
            # 보유종목 구하기
            #self.positions = {}  # {stock_code: {'size': int, 'avg_price': float, 'entry_date': datetime.date, 'highest_price': float}}
            for stock_code, position_info in self.broker.positions.items(): # .items()를 사용하여 키와 값을 모두 가져옵니다.
                if position_info['size'] > 0: # position_info['size']로 직접 접근합니다.
                    daily_df = self.manager.cache_daily_ohlcv(stock_code, fetch_date, prev_date)
                    self.data_store['daily'][stock_code] = daily_df
                    # self.data_store['daily']가 초기화되지 않았을 가능성을 대비하여 확인 후 할당합니다.
                    # if 'daily' not in self.data_store:
                    #     self.data_store['daily'] = {}
                    #     self.data_store['daily'][stock_code] = daily_df

            # 유니버스 종목 하루전
            stocks = self.db_manager.fetch_daily_theme_stock(prev_date, prev_date)
            #print(stocks)
            for i, (stock_code, stock_name) in enumerate(stocks):
                daily_df = self.manager.cache_daily_ohlcv(stock_code, fetch_date, prev_date)
                self.data_store['daily'][stock_code] = daily_df

            # 2. 전일 일봉 데이터를 기반으로 '오늘 실행할' 신호를 생성합니다.
            if self.daily_strategy:
                
                # 직전영업일이 있다. (지표계산을 위해 백테스트 시작일 이전 1달치 일봉데이터도 저장되어 있음)
                if prev_date:
                    ##############################
                    # 전일 데이터로 일봉 전략 실행 - 오늘은 아직 장이 시작하지 않았으므로(전일 종가까지 데이터로)
                    self.daily_strategy.run_daily_logic(prev_date)
                    # self.daily_strategy.signals 에 전략 실행결과가 보관 됨
                    ##############################

                    # 생성된 신호 중 'buy', 'sell', 'hold' 신호를 current_day_signals에 저장합니다.
                    # 종목별로 일봉전략 결과로 신호를 생성
                    for stock_code, signal_info in self.daily_strategy.signals.items():
                        if signal_info['signal'] in ['buy', 'sell', 'hold']:
                            # 분봉에 전달해 줄 일봉전략 처리 결과를 current_day_signals 에 저장  
                            self.current_day_signals[stock_code] = {
                                **signal_info,
                                'traded_today': False, # 초기화된 상태로 전달
                                # 신호발생일은 전영업일: 신호는 전영업일 종가로 생성 (전영업일을 구하는 데 사용해도 됨)
                                'signal_date': prev_date 
                            }

                    logging.debug(f"[{current_date.isoformat()}] 전일({prev_date.isoformat()}) 일봉 전략 실행 완료: {len(self.current_day_signals)}개의 당일 매매 신호 생성.")

            # 3. 오늘 분봉 매매 로직을 실행합니다.
                # OpenMinute는 최적화 전용 분봉전략으로, 처리속도를 빠르게 하기위해 분봉데이터를 사용하지 않고, 분봉 가상화 처리
                is_open_minute_strategy = hasattr(self.minute_strategy, 'strategy_name') and self.minute_strategy.strategy_name == "OpenMinute"
                
                # OpenMinute 전략이면 매일 분봉 가상화 캐시 초기화
                if is_open_minute_strategy and hasattr(self.minute_strategy, 'reset_virtual_range_cache'):
                    self.minute_strategy.reset_virtual_range_cache()
                # 오늘 분봉은 일봉전략 처리결과를 signals_for_minute_trading 에 담아 사용
                self.signals_for_minute_trading = self.current_day_signals.copy()
                
                # 일봉전략 처리 결과에
                # 매수/매도 처리 유무, 손절 처리 유무 확인
                has_trading_signals = any(signal_info['signal'] in ['buy', 'sell'] for signal_info in self.signals_for_minute_trading.values())
                has_stop_loss = self.broker.stop_loss_params is not None # 손절처리 유무
                
                # 매수/매도 처리, 손절처리가 없다면, 분봉에서 매매 할 필요가 없으므로
                if not (has_trading_signals or has_stop_loss):
                    logging.debug(f"[{current_date.isoformat()}] 매수/매도 신호가 없고 손절매가 비활성화되어 있어 분봉 로직을 건너킵니다.")

                else:
                    logging.info(f"-------- {current_date.isoformat()} 매매 시작 --------")
                    
                    # 최적화에서 사용하는 분봉 가상화 전략
                    if is_open_minute_strategy:
                        # OpenMinute 전략: 분봉 데이터 로딩 없이 9:01에만 매매 실행
                        logging.info(f"[{current_date.isoformat()}] OpenMinute 전략: 분봉 데이터 로딩 없이 9:01에 매매 실행")
                        
                        # 9:01 시간 생성 (첫 분봉 완성 후)
                        trade_time = datetime.datetime.combine(current_date, time(9, 1))
                        
                        # OpenMinute 전략에 신호 업데이트 (target_quantity 정보 포함)
                        # 신호 전달 브릿지 로그 추가
                        num_signals = len(self.signals_for_minute_trading)
                        num_buy = sum(1 for s in self.signals_for_minute_trading.values() if s.get('signal') == 'buy')
                        num_sell = sum(1 for s in self.signals_for_minute_trading.values() if s.get('signal') == 'sell')
                        num_hold = sum(1 for s in self.signals_for_minute_trading.values() if s.get('signal') == 'hold')
                        logging.info(f"[신호전달] 분봉전략에 신호 전달: 총 {num_signals}개 (매수: {num_buy}, 매도: {num_sell}, 홀딩: {num_hold})")
                        self.minute_strategy.update_signals(self.signals_for_minute_trading)
                        
                        # 9:01에 매매 실행
                        stocks_to_trade = set()
                        # 매수/매도 신호가 있는 종목들 추가
                        for stock_code, signal_info in self.signals_for_minute_trading.items():
                            if signal_info.get('signal') in ['buy', 'sell']:
                                stocks_to_trade.add(stock_code)
                        
                        # OpenMinute 전략은 하루 한번 매매로 분봉매매를 시뮬레이션 한다.
                        # 그러므로 손절은 종가로만 처리한다. 이렇게 하지 않으면 과최적화 된다.
                        # 시가 손절도 있지만, 주가의 우상향 성질을 이용해서 종가가 시가 보다 나을 듯 하다.
                        # 포트폴리오 손절은 없음
                        for stock_code in stocks_to_trade:

                            ###################################
                            # 분봉 전략 매매 실행
                            self.minute_strategy.run_minute_logic(trade_time, stock_code)
                            ###################################
                    
                    # 일반 분봉 전략 실행
                    else:
                        # 3-1. 먼저 당일 실행할 신호(매수/매도/보유)가 있는 종목들의 분봉 데이터를 모두 로드
                        stocks_to_load = set()  # 분봉 데이터가 필요한 종목들

                        # 매수/매도 신호가 있는 종목들 추가
                        for stock_code, signal_info in self.signals_for_minute_trading.items():
                            if signal_info['signal'] in ['buy', 'sell']:
                                stocks_to_load.add(stock_code)
                        
                        current_positions = set(self.broker.positions.keys()) #### 보유종목 구하기
                        stocks_to_load.update(current_positions)                        

                        # 필요한 종목들의 분봉 데이터를 로드
                        for stock_code in stocks_to_load:
                            signal_info = self.current_day_signals.get(stock_code)
                            prev_trading_day = signal_info.get('signal_date', current_date) if signal_info else current_date

                            if prev_trading_day:
                                # TraderManager를 사용하여 분봉 데이터 로드 (전일~당일)
                                minute_df = self.manager.cache_minute_ohlcv(
                                    stock_code,
                                    prev_trading_day,  # 전일 영업일부터
                                    current_date       # 당일까지
                                )

                                # 기존 백테스터와 동일하게 날짜별로 분봉 데이터 저장
                                if not minute_df.empty:
                                    if stock_code not in self.data_store['minute']:
                                        self.data_store['minute'][stock_code] = {}
                                    for date in [prev_trading_day, current_date]:
                                        date_data = minute_df[minute_df.index.normalize() == pd.Timestamp(date).normalize()]
                                        if not date_data.empty:
                                            self.data_store['minute'][stock_code][date] = date_data
                                            logging.debug(f"{stock_code} 종목의 {date} 분봉 데이터 로드 완료. 데이터 수: {len(date_data)}행")
                        # for 분봉로드 끝

                        # 3-2. 모든 시그널을 분봉 전략에 한 번에 업데이트
                        # 로그출력을 위한 처리
                        num_signals = len(self.signals_for_minute_trading)
                        num_buy = sum(1 for s in self.signals_for_minute_trading.values() if s.get('signal') == 'buy')
                        num_sell = sum(1 for s in self.signals_for_minute_trading.values() if s.get('signal') == 'sell')
                        num_hold = sum(1 for s in self.signals_for_minute_trading.values() if s.get('signal') == 'hold')
                        logging.info(f"[신호 전달] 분봉전략에 총 {num_signals}개 (매수: {num_buy}, 매도: {num_sell}, 홀딩: {num_hold})")
                        # 분봉전략 내의 signals 변수에 전달
                        self.minute_strategy.update_signals(self.signals_for_minute_trading)
                        logging.debug(f"[{current_date.isoformat()}] 분봉 전략에 {len(self.signals_for_minute_trading)}개의 시그널 업데이트 완료.")
                        
                        # 3-3. 분봉 매매 대상은 신호(매수/매도) 손절 파라미터 설정시 보유 종목
                        stocks_to_trade = set() # 매매대상
                        
                        # 매수/매도 신호가 있는 종목들 추가
                        for stock_code, signal_info in self.signals_for_minute_trading.items():
                            if signal_info['signal'] in ['buy', 'sell']:
                                stocks_to_trade.add(stock_code)
                        # 손절매 기능이 있다면, 보유 중인 종목들 추가 (손절매 체크용)
                        if has_stop_loss:
                            current_positions = set(self.broker.positions.keys())
                            stocks_to_trade.update(current_positions)

                        ###################################
                        # 분봉 전략 매매 오늘 장이 끝날때 까지 매분 반복 실행
                        ###################################
                        # 장 시작 시간부터 장 마감 시간까지 1분 단위로 반복하며 분봉 전략 실행
                        start_time = time(9, 0) # 9시 정각
                        end_time = time(15, 30) # 3시 30분 (장 마감)
                        trade_time = datetime.datetime.combine(current_date, start_time) # 날짜+분

                        while trade_time <= datetime.datetime.combine(current_date, end_time):
                            
                            ##############################
                            # 분봉 전략 실행 및 매매 처리
                            ##############################                                            
                            for stock_code in stocks_to_trade:
                                # 종목에 해당 시간의 분봉 데이터가 없으면, 이 종목 스킵
                                if stock_code not in self.data_store['minute'] or \
                                   current_date not in self.data_store['minute'][stock_code] or \
                                   trade_time not in self.data_store['minute'][stock_code][current_date].index:
                                    logging.debug(f"[{trade_time.isoformat()}] {stock_code}: 해당 시간의 분봉 데이터 없음. 스킵.")
                                    continue
                                
                                # 분봉전략 실행
                                self.minute_strategy.run_minute_logic(trade_time, stock_code)
 
                            # ########################
                            # 포토폴리오 손절
                            # ------------------------
                            # 포트폴리오 손절을 위한 9:00, 15:20 시간 체크, 분봉마다하는 것이 정확하겠지만 속도상 
                            if self.broker.stop_loss_params is not None and self._should_check_portfolio(trade_time):
                                
                                current_prices = {}
                                for code in list(self.broker.positions.keys()):
                                    # 캐시된 가격 사용
                                    if code in self.minute_strategy.last_prices:
                                        current_prices[code] = self.minute_strategy.last_prices[code]
                                    else:
                                        price_data = self._get_bar_at_time('minute', code, trade_time)
                                        if price_data is not None:
                                            current_prices[code] = price_data['close']
                                            self.minute_strategy.last_prices[code] = price_data['close']
                            
                                    # 현재 가격이 없는 종목은 제외
                                    current_prices = {k: v for k, v in current_prices.items() if not np.isnan(v)}                                
                               
                                # 포트폴리오 손절은 Broker에 위임처리
                                if self.broker.check_and_execute_portfolio_stop_loss(current_prices, trade_time):
                                    # 매도처리
                                    for code in list(self.minute_strategy.signals.keys()):
                                        # 매도된 것 신호 정리
                                        if code in self.broker.positions and self.broker.positions[code]['size'] == 0: # 매도 후 == 수량 0
                                            self.minute_strategy.reset_signal(stock_code)

                                    logging.info(f"[{trade_time.isoformat()}] 포트폴리오 손절매 실행 완료. 오늘의 매매 종료.")
                                    #self.portfolio_stop_flag = True 불필요 break
                                    break # 분봉 루프를 종료, 일일 포트폴리오 처리 후 다음 "영업일"로 넘어감
                            # 포트폴리오 손절 ------------------------
                            
                            # 다음 분으로 이동
                            trade_time += datetime.timedelta(minutes=1)
                        # while -------------------------------------
                # end if 분봉매매

            # end if 분봉 전략 유무
                            
            # 일일 포트폴리오 가치 기록 (장 마감 시점)
            # 장 마감 종가를 가져오기 위해 일봉 데이터 사용
            current_day_close_prices = {}
            #for stock_code, df in self.data_store['daily'].items():
            for stock_code, position_info in self.broker.positions.items(): # .items()를 사용하여 키와 값을 모두 가져옵니다.
                # 오늘 일봉데이터 다시 가져오기
                df = self.manager.cache_daily_ohlcv(stock_code, fetch_date, current_date)
                self.data_store['daily'][stock_code] = df
                current_day_close_prices[stock_code] = df['close'].iloc[0]

            daily_portfolio_value = self.broker.get_portfolio_value(current_day_close_prices)
            self.portfolio_values.append((current_date, daily_portfolio_value))
            logging.info(f"[{current_date.isoformat()}] 장 마감 포트폴리오 가치: {daily_portfolio_value:,.0f}원, 현금: {self.broker.cash:,.0f}원")

            # 일일 거래 기록 초기화 (broker 내부)
            self.broker.reset_daily_transactions() # 현재 pass 상태, 아래 로직으로 채워야 함
            
            # 수정: 다음날을 위해 모든 신호 초기화
            # 1. 일봉 전략의 신호 완전 초기화
            self.daily_strategy._reset_all_signals()  # 모든 신호를 완전히 삭제
            
            # 2. 백테스터의 신호 저장소 초기화 (다음날을 위해)
            self.current_day_signals = {}  # 다음날 일봉 전략이 새로운 신호를 생성할 수 있도록 초기화
            self.signals_for_minute_trading = {}  # 분봉 매매용 신호도 초기화
            self.last_portfolio_check_time = None # 다음 날을 위해 손절 체크 시간 초기화
            
            logging.debug(f"[{current_date.isoformat()}] 일일 신호 초기화 완료 - 다음날을 위해 모든 신호 저장소 비움")

            # 다음날로 이동
            current_date += datetime.timedelta(days=1)
            # 다음 장 개장일까지 스킵: get_next_weekday()의 'target_weekday' 인자 누락 오류를 해결하기 위해 인라인 로직으로 대체
            temp_date = current_date
            while temp_date.weekday() >= 5: # 토요일(5), 일요일(6)
                temp_date += datetime.timedelta(days=1)
            current_date = temp_date

            self.portfolio_stop_flag = False  # 새로운 날짜마다 플래그 초기화

        logging.info("백테스트 완료.")
        
        # 백테스트 결과 보고서 생성 및 저장
        # 초기 포트폴리오 항목 (시작일 이전의 초기 자본)은 제외하고 실제 백테스트 기간 데이터만 사용
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
        final_metrics = calculate_performance_metrics(portfolio_value_series)
        return portfolio_value_series, final_metrics