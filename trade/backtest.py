# backtest/backtester.py
from datetime import datetime, time, timedelta
import logging
import pandas as pd
import numpy as np
import sys
import os
# 프로젝트 루트 경로를 sys.path에 추가 (manager 디렉토리에서 실행 가능하도록)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trade.broker import Broker
from trade.backtest_report import BacktestReport # Reporter 타입 힌트를 위해 남겨둠
from api.creon_api import CreonAPIClient
from manager.backtest_manager import BacktestManager # BacktestManager 타입 힌트를 위해 남겨둠
from manager.db_manager import DBManager    
from strategies.strategy import DailyStrategy, MinuteStrategy 
from util.strategies_util import *
from strategies.sma_daily import SMADaily
from strategies.contrarian_daily import ContrarianDaily
from strategies.rsi_minute import RSIMinute
from strategies.open_minute import OpenMinute

logger = logging.getLogger(__name__)

class Backtest:
    # __init__ 메서드를 외부에서 필요한 인스턴스를 주입받도록 변경
    def __init__(self, 
                 api_client: CreonAPIClient, 
                 manager: BacktestManager, 
                 report: BacktestReport, 
                 db_manager: DBManager,
                 initial_cash: float, 
                 save_to_db: bool = True
        ):  # DB 저장 여부 파라미터 추가
        
        self.api_client = api_client
        self.broker = Broker(initial_cash, commission_rate=0.0016, slippage_rate=0.0004) # 커미션 0.16% + 슬리피지 0.04% = 총 0.2%
        self.data_store = {'daily': {}, 'minute': {}} # {stock_code: DataFrame}
        self.portfolio_values = [] # (datetime, value) 튜플 저장
        self.initial_cash = initial_cash
        self.save_to_db = save_to_db  # DB 저장 여부 저장
        
        self.daily_strategy: DailyStrategy = None
        self.minute_strategy: MinuteStrategy = None

        # 외부에서 주입받은 인스턴스를 사용
        self.manager = manager
        self.report = report
        self.db_manager = db_manager
        
        #self.current_day_signals = {}
        # NEW: 현재 날짜의 분봉 매매를 위해 사용될 신호들을 저장
        #self.signals_for_minute_trading = {}
        # NEW: 포트폴리오 손절 체크 시간을 추적하기 위한 변수
        self.last_portfolio_check = None
        # 포트폴리오 손절 발생 시 당일 매매 중단 플래그
        self.portfolio_stop_flag = False

        # 일봉 업데이트 캐시 변수들 (성능 개선용)
        self._daily_update_cache = {}  # {stock_code: {date: last_update_time}}
        self._minute_data_cache = {}   # {stock_code: {date: filtered_data}}

        logging.info(f"백테스터 초기화 완료. 초기 현금: {self.initial_cash:,.0f}원, DB저장: {self.save_to_db}")

    def set_strategies(self, daily_strategy: DailyStrategy = None, minute_strategy: MinuteStrategy = None):
        if daily_strategy:
            self.daily_strategy = daily_strategy
            logging.info(f"일봉 전략 '{daily_strategy.__class__.__name__}' 설정 완료.")
        if minute_strategy:
            self.minute_strategy = minute_strategy
            logging.info(f"분봉 전략 '{minute_strategy.__class__.__name__}' 설정 완료.")

    def set_broker_stop_loss_params(self, params: dict = None):
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
        check_minutes = [9 * 60, 15 * 60 + 20]  # 9:00, 15:20
        
        
        if current_minutes in check_minutes and (self.last_portfolio_check.date() != current_dt.date() or 
                                               (self.last_portfolio_check.hour * 60 + self.last_portfolio_check.minute) not in check_minutes):
            self.last_portfolio_check = current_dt
            return True
            
        return False
    
    def _update_daily_data_from_minute_bars(self, current_dt: datetime.datetime):
        """
        매분 현재 시각까지의 1분봉 데이터를 집계하여 일봉 데이터를 생성하거나 업데이트합니다.
        캐시를 사용하여 성능을 개선합니다.
        :param current_dt: 현재 시각 (datetime 객체)
        """
        current_date = current_dt.date()
        
        for stock_code in self.data_store['daily'].keys():
            minute_data_for_today = self.data_store['minute'].get(stock_code)

            if minute_data_for_today is not None:
                # 캐시 키 생성
                cache_key = f"{stock_code}_{current_date}"
                
                # 캐시된 데이터가 있고, 마지막 업데이트 시간이 현재 시각과 같거나 최신이면 스킵
                if cache_key in self._daily_update_cache:
                    last_update = self._daily_update_cache[cache_key]
                    if last_update >= current_dt:
                        continue
                
                # minute_data_for_today는 {date: DataFrame} dict 구조
                # current_date에 해당하는 DataFrame을 가져옴
                today_minute_bars = minute_data_for_today.get(current_date)

                if today_minute_bars is not None and not today_minute_bars.empty:
                    # 현재 시각까지의 분봉 데이터만 필터링
                    # current_dt 이하의 분봉 데이터만 사용
                    filtered_minute_bars = today_minute_bars[today_minute_bars.index <= current_dt]
                    
                    if not filtered_minute_bars.empty:
                        # 캐시된 필터링된 데이터가 있는지 확인
                        if cache_key in self._minute_data_cache:
                            cached_filtered_data = self._minute_data_cache[cache_key]
                            # 캐시된 데이터가 현재 시각까지의 데이터를 포함하는지 확인
                            if cached_filtered_data.index.max() >= current_dt:
                                filtered_minute_bars = cached_filtered_data[cached_filtered_data.index <= current_dt]
                            else:
                                # 캐시 업데이트
                                self._minute_data_cache[cache_key] = filtered_minute_bars
                        else:
                            # 캐시에 저장
                            self._minute_data_cache[cache_key] = filtered_minute_bars
                        
                        # 현재 시각까지의 일봉 데이터 계산
                        daily_open = filtered_minute_bars.iloc[0]['open']  # 첫 분봉 시가
                        daily_high = filtered_minute_bars['high'].max()    # 현재까지 최고가
                        daily_low = filtered_minute_bars['low'].min()      # 현재까지 최저가
                        daily_close = filtered_minute_bars.iloc[-1]['close']  # 현재 시각 종가
                        daily_volume = filtered_minute_bars['volume'].sum()   # 현재까지 누적 거래량

                        # 새로운 일봉 데이터 생성
                        new_daily_bar = pd.Series({
                            'open': daily_open,
                            'high': daily_high,
                            'low': daily_low,
                            'close': daily_close,
                            'volume': daily_volume
                        }, name=pd.Timestamp(current_date))

                        # 일봉 데이터가 존재하면 업데이트, 없으면 추가
                        if pd.Timestamp(current_date) in self.data_store['daily'][stock_code].index:
                            # 기존 일봉 데이터 업데이트
                            self.data_store['daily'][stock_code].loc[pd.Timestamp(current_date)] = new_daily_bar
                        else:
                            # 새로운 일봉 데이터 추가
                            self.data_store['daily'][stock_code] = pd.concat([
                                self.data_store['daily'][stock_code], 
                                pd.DataFrame([new_daily_bar])
                            ])
                            # 인덱스 정렬
                            self.data_store['daily'][stock_code].sort_index(inplace=True)
                        
                        # 업데이트 시간 캐시
                        self._daily_update_cache[cache_key] = current_dt

    def _clear_daily_update_cache(self):
        """
        일봉 업데이트 캐시를 초기화합니다. 새로운 날짜로 넘어갈 때 호출됩니다.
        """
        self._daily_update_cache.clear()
        self._minute_data_cache.clear()

    def run(self, start_date: datetime.date, end_date: datetime.date):
        logging.info(f"백테스트 시작: {start_date} 부터 {end_date} 까지")
        
        # 시장 캘린더 데이터 로드 (전영업일 계산을 위해 충분히 가져옴)
        market_calendar_df = self.db_manager.fetch_market_calendar(start_date- timedelta(days=10), end_date)
        if market_calendar_df.empty:
            logger.error("시장 캘린더 데이터를 가져올 수 없습니다. 백테스트를 중단합니다.")
            return
        # 영업일만 필터링하고 날짜를 리스트로 변환하여 정렬
        trading_dates = market_calendar_df[market_calendar_df['is_holiday'] == 0]['date'].dt.date.tolist()
        trading_dates.sort() # 날짜가 오름차순으로 정렬되도록 보장
        # if not trading_dates:
        #     logger.warning("지정된 기간 내 영업일이 없습니다. daily_universe 채우기를 건너뜁니다.")
        #     return
        # # 전일 날짜 계산
        # trading_dates_for_run = [
        #     d for d in trading_dates 
        #     if start_date <= d <= end_date
        # ]
        # if not trading_dates_for_run:
        #     logger.error(f"백테스트 기간 [{start_date} ~ {end_date}] 내에 유효한 거래일이 없습니다.")
        #     return

        for idx, current_date in enumerate(trading_dates):
            prev_date = None
            if idx > 0 and trading_dates[idx] >= start_date:
                current_date = trading_dates[idx]
                prev_date = trading_dates[idx - 1]
                break
            else:
                continue
        # end 전일 날짜 계산 --------------------------

        # 시작일 이전의 초기 포트폴리오 가치 기록
        initial_portfolio_value = self.broker.get_portfolio_value({}) # 초기 현금만 반영
        self.portfolio_values.append((prev_date , initial_portfolio_value))
                              
        # 영업일을 순회하며 SetupManager 실행
        for current_date in trading_dates:  
            # Refactored: 매분 오늘의 일봉 데이터를 업데이트하는 함수 호출
            #self._update_daily_data_from_minute_bars(current_dt) 
            # 영업일이면 백테스트 시작
            logging.info(f"\n--- 현재 백테스트 날짜: {current_date.isoformat()} ---")

            # 새로운 날짜로 넘어갈 때 일봉 업데이트 캐시 초기화
            self._clear_daily_update_cache()

            # 전일 날짜 계산
            current_date_index = trading_dates.index(current_date)
            prev_date = trading_dates[current_date_index - 1]            # self.data_store 
            
            
            # 일봉 데이터를 가져오는 대상은 1. 보유종목 2. 유니버스 종목이다. 
            # OpenMinute는 오늘 일봉이 있어야, 분봉 가상화 처리를 하므로 오늘 일봉까지 전달한다.
            # 일봉전략은 지표계산을 할 때 iloc[-2] 처리 등으로 오늘 일봉을 지표에 반영하지 않도록 해야 한다.
            #보유종목 일봉 데이터 설정
            # fetch_date = current_date - timedelta(days=60)
            # self.data_store['daily'] = {} # 보유종목과 유니버스 종목을 다시 담기 위해 초기화
            # self.positions = {}  # {stock_code: {'size': int, 'avg_price': float, 'entry_date': datetime.date, 'highest_price': float}}
            # for stock_code, position_info in self.broker.positions.items(): # .items()를 사용하여 키와 값을 모두 가져옵니다.
            #     if position_info['size'] > 0: # position_info['size']로 직접 접근합니다.
            #         daily_df = self.manager.cache_daily_ohlcv(stock_code, fetch_date, prev_date)
            #         self.data_store['daily'][stock_code] = daily_df

            # # 하루전 선정된 유니버스 종목들을 가져온다.  
            # stocks = self.db_manager.fetch_daily_theme_stock(prev_date - timedelta(days=5), prev_date)
            # #print(stocks)
            # for i, (stock_code, stock_name) in enumerate(stocks):
            #     daily_df = self.manager.cache_daily_ohlcv(stock_code, fetch_date, current_date)
            #     self.data_store['daily'][stock_code] = daily_df



            # 2. 전일 일봉 데이터를 기반으로 '오늘 실행할' 신호를 생성합니다.
            if self.daily_strategy:
                ##############################
                # 일봉 전략 실행 - 오늘은 아직 장이 시작하지 않았으므로(전일 종가까지 데이터로)
                self.daily_strategy.run_daily_logic(prev_date)
                ##############################
                # 생성된 신호 중 'buy', 'sell', 'hold' 신호를 current_day_signals에 저장합니다.
                # 일봉--> 분봉 간 신호 전달###############################
                for stock_code, signal_info in self.daily_strategy.signals.items():
                    if signal_info['signal'] in ['buy', 'sell', 'hold']:
                        # 분봉에 전달해 줄 일봉전략 처리 결과를 current_day_signals 에 저장  
                        self.minute_strategy.signals[stock_code] = {
                            **signal_info,
                            'traded_today': False, # 초기화된 상태로 전달
                            'signal_date': prev_date 
                        }

            # 3. 오늘 분봉 매매 로직을 실행합니다.
            # 분봉전략명이 OpenMinute 인지 확인 해서 아래에서 다르게 처리한다. 
            if self.minute_strategy:
                
                # OpenMinute는 최적화 전용 분봉전략으로, 처리속도를 빠르게 하기위해 분봉데이터를 사용하지 않고, 분봉 가상화 처리
                is_open_minute_strategy = hasattr(self.minute_strategy, 'strategy_name') and self.minute_strategy.strategy_name == "OpenMinute"
                
                # OpenMinute 전략이면 매일 분봉 가상화 캐시 초기화
                if is_open_minute_strategy and hasattr(self.minute_strategy, 'reset_virtual_range_cache'):
                    self.minute_strategy.reset_virtual_range_cache()
                
                # 일봉전략 처리 결과에
                # 매수/매도 처리 유무, 손절 처리 유무 확인
                has_trading_signals = any(signal_info['signal'] in ['buy', 'sell'] for signal_info in self.daily_strategy.signals.values())
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
                        current_dt = datetime.combine(current_date, time(9, 1))
                        
                        # 9:01에 매매 실행
                        stocks_to_trade = set()
                        # 매수/매도 신호가 있는 종목들 추가
                        for stock_code, signal_info in self.minute_strategy.signals.items():
                            if signal_info.get('signal') in ['buy', 'sell']:
                                stocks_to_trade.add(stock_code)
                        
                        # OpenMinute 전략은 하루 한번 매매로 분봉매매를 시뮬레이션 한다.
                        # 그러므로 손절은 종가로만 처리한다. 이렇게 하지 않으면 과최적화 된다.
                        # 시가 손절도 있지만, 주가의 우상향 성질을 이용해서 종가가 시가 보다 나을 듯 하다.
                        # 포트폴리오 손절은 없음
                        for stock_code in stocks_to_trade:

                            ###################################
                            # 분봉 전략 매매 실행
                            self.minute_strategy.run_minute_logic(current_dt, stock_code)
                            ###################################
                    
                    # 일반 분봉 전략 실행
                    else:
                        # 3-1. 먼저 당일 실행할 신호(매수/매도/보유)가 있는 종목들의 분봉 데이터를 모두 로드
                        stocks_to_load = set()  # 분봉 데이터가 필요한 종목들

                        # 매수/매도 신호가 있는 종목들 추가
                        for stock_code, signal_info in self.minute_strategy.signals.items():
                            if signal_info['signal'] in ['buy', 'sell']:
                                stocks_to_load.add(stock_code)
                        
                        current_positions = set(self.broker.positions.keys()) #### 보유종목 구하기
                        stocks_to_load.update(current_positions)                        

                        # Refactored: 매분 오늘의 일봉 데이터를 업데이트하는 함수 호출
                        # self._update_daily_data_from_minute_bars(current_dt)                        

                        # 필요한 종목들의 분봉 데이터를 로드
                        for stock_code in stocks_to_load:
                            signal_info = self.minute_strategy.signals.get(stock_code)
                            prev_date = signal_info.get('signal_date', current_date) if signal_info else current_date

                            if not prev_date:
                                prev_date = current_date 

                            # BacktestManager를 사용하여 분봉 데이터 로드 (전일~당일)
                            minute_df = self.manager.cache_minute_ohlcv(
                                stock_code,
                                prev_date,  # 전일 영업일부터
                                current_date       # 당일까지
                            )
                            # 기존 백테스터와 동일하게 날짜별로 분봉 데이터 저장
                            if not minute_df.empty:
                                if stock_code not in self.data_store['minute']:
                                    self.data_store['minute'][stock_code] = {}
                                for date in [prev_date, current_date]:
                                    date_data = minute_df[minute_df.index.normalize() == pd.Timestamp(date).normalize()]
                                    if not date_data.empty:
                                        self.data_store['minute'][stock_code][date] = date_data
                                        logging.debug(f"{stock_code} 종목의 {date} 분봉 데이터 로드 완료. 데이터 수: {len(date_data)}행")
                        # for 분봉로드 끝
                        
                        # 3-3. 분봉 매매 대상은 신호(매수/매도) 손절 파라미터 설정시 보유 종목
                        stocks_to_trade = set() # 매매대상
                        
                        # 매수/매도 신호가 있는 종목들 추가
                        #for stock_code, signal_info in self.signals_for_minute_trading.items():
                        for stock_code, signal_info in self.minute_strategy.signals.items():
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
                        open_time = time(9, 0) # 9시 정각
                        market_open = datetime.combine(current_date, open_time)
                        for i in range(1, 391):
                        #while current_dt <= datetime.combine(current_date, end_time):
                            current_dt = market_open + timedelta(minutes=i)
                            current_date = current_dt.date() # 현재 날짜 (date 객체)
                            ##############################
                            # 분봉 전략 실행 및 매매 처리
                            ##############################                                            
                            for stock_code in stocks_to_trade:
                                # 종목에 해당 시간의 분봉 데이터가 없으면, 이 종목 스킵
                                if stock_code not in self.data_store['minute'] or \
                                   current_date not in self.data_store['minute'][stock_code] or \
                                   current_dt not in self.data_store['minute'][stock_code][current_date].index:
                                    logging.debug(f"[{current_dt.isoformat()}] {stock_code}: 해당 시간의 분봉 데이터 없음. 스킵.")
                                    continue
                                
                                # 분봉전략 실행
                                self.minute_strategy.run_minute_logic(current_dt, stock_code)
 
                            # ########################
                            # 포토폴리오 손절
                            # ------------------------
                            # 포트폴리오 손절을 위한 9:00, 15:20 시간 체크, 분봉마다하는 것이 정확하겠지만 속도상 
                            if self.broker.stop_loss_params is not None and self._should_check_portfolio(current_dt):
                                
                                current_prices = {}
                                for code in list(self.broker.positions.keys()):
                                    # 캐시된 가격 사용
                                    if code in self.minute_strategy.last_prices:
                                        current_prices[code] = self.minute_strategy.last_prices[code]
                                    else:
                                        price_data = self.minute_strategy._get_bar_at_time('minute', code, current_dt)
                                        if price_data is not None:
                                            current_prices[code] = price_data['close']
                                            self.minute_strategy.last_prices[code] = price_data['close']
                            
                                    # 현재 가격이 없는 종목은 제외
                                    current_prices = {k: v for k, v in current_prices.items() if not np.isnan(v)}                                
                               
                                # 포트폴리오 손절은 Broker에 위임처리
                                if self.broker.check_and_execute_portfolio_stop_loss(current_prices, current_dt):
                                    # 매도처리
                                    for code in list(self.minute_strategy.signals.keys()):
                                        # 매도된 것 신호 정리
                                        if code in self.broker.positions and self.broker.positions[code]['size'] == 0: # 매도 후 == 수량 0
                                            self.minute_strategy.reset_signal(stock_code)

                                    logging.info(f"[{current_dt.isoformat()}] 포트폴리오 손절매 실행 완료. 오늘의 매매 종료.")
                                    break # 분봉 루프를 종료, 일일 포트폴리오 처리 후 다음 "영업일"로 넘어감
                            # 포트폴리오 손절 ------------------------
                        # for -------------------------------------

                # end if 분봉매매

            # end if 분봉 전략 유무
                            
            # 일일 포트폴리오 가치 기록 (장 마감 시점)
            # 장 마감 종가를 가져오기 위해 일봉 데이터 사용
            current_day_close_prices = {}
            #for stock_code, df in self.data_store['daily'].items():
            for stock_code, position_info in self.broker.positions.items(): # .items()를 사용하여 키와 값을 모두 가져옵니다.
                # 오늘 일봉데이터 다시 가져오기
                df = self.manager.cache_daily_ohlcv(stock_code, current_date, current_date)
                self.data_store['daily'][stock_code] = df
                current_day_close_prices[stock_code] = df['close'].iloc[0]

            daily_portfolio_value = self.broker.get_portfolio_value(current_day_close_prices)
            self.portfolio_values.append((current_date, daily_portfolio_value))
            logging.info(f"[{current_date.isoformat()}] 장 마감 포트폴리오 가치: {daily_portfolio_value:,.0f}원, 현금: {self.broker.cash:,.0f}원")

            # 일일 거래 기록 초기화 (Broker 내부)
            self.broker.reset_daily_transactions() # 현재 pass 상태, 아래 로직으로 채워야 함
            
            # 수정: 다음날을 위해 모든 신호 초기화
            # 1. 일봉 전략의 신호 완전 초기화
            self.daily_strategy._reset_all_signals()  # 모든 신호를 완전히 삭제
            
            # 2. 백테스터의 신호 저장소 초기화 (다음날을 위해)
            self.last_portfolio_check = None # 다음 날을 위해 손절 체크 시간 초기화
            self.portfolio_stop_flag = False  # 새로운 날짜마다 플래그 초기화
            
            logging.debug(f"[{current_date.isoformat()}] 일일 신호 초기화 완료 - 다음날을 위해 모든 신호 저장소 비움")

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

        # 모든 저장 로직을 self.report에게 위임
        self.report.generate_and_save_report(
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
    
def load_stocks(trade, manager, db_manager, start_date, end_date):
    # 섹터별 대표 종목 리스트 (간소화) - 기존과 동일
    sector_stocks = {
        '반도체': [
            ('삼성전자', 'IT'), ('SK하이닉스', 'IT'), ('DB하이텍', 'IT'),
            ('네패스아크', 'IT'), ('와이아이케이', 'IT')
        ],
        '2차전지': [
            ('LG에너지솔루션', '2차전지'), ('삼성SDI', '2차전지'), ('SK이노베이션', '2차전지'),
            ('에코프로비엠', '2차전지'), ('포스코퓨처엠', '2차전지'), ('LG화학', '2차전지'),
            ('일진머티리얼즈', '2차전지'), ('엘앤에프', '2차전지')
        ],
        '바이오': [
            ('삼성바이오로직스', '바이오'), ('셀트리온', '바이오'), ('SK바이오사이언스', '바이오'),
            ('유한양행', '바이오'), ('한미약품', '바이오')
        ],
        '플랫폼/인터넷': [
            ('NAVER', 'IT'), ('카카오', 'IT'), ('크래프톤', 'IT'),
            ('엔씨소프트', 'IT'), ('넷마블', 'IT')
        ],
        '자동차': [
            ('현대차', '자동차'), ('기아', '자동차'), ('현대모비스', '자동차'),
            ('만도', '자동차'), ('한온시스템', '자동차')
        ],
        '철강/화학': [
            ('POSCO홀딩스', '철강'), ('고려아연', '철강'), ('롯데케미칼', '화학'),
            ('금호석유', '화학'), ('효성첨단소재', '화학')
        ],
        '금융': [
            ('KB금융', '금융'), ('신한지주', '금융'), ('하나금융지주', '금융'),
            ('우리금융지주', '금융'), ('메리츠금융지주', '금융')
        ],
        '통신': [
            ('SK텔레콤', '통신'), ('KT', '통신'), ('LG유플러스', '통신'),
            ('SK스퀘어', '통신')
        ],
        '유통/소비재': [
            ('CJ제일제당', '소비재'), ('오리온', '소비재'), ('롯데쇼핑', '유통'),
            ('이마트', '유통'), ('BGF리테일', '유통')
        ],
        '건설/기계': [
            ('현대건설', '건설'), ('대우건설', '건설'), ('GS건설', '건설'),
            ('두산에너빌리티', '기계'), ('두산밥캣', '기계')
        ],
        '조선/항공': [
            ('한국조선해양', '조선'), ('삼성중공업', '조선'), ('대한항공', '항공'),
            ('현대미포조선', '조선')
        ],
        '에너지': [
            ('한국전력', '에너지'), ('한국가스공사', '에너지'), ('두산퓨얼셀', '에너지'),
            ('에스디바이오센서', '에너지')
        ],
        '반도체장비': [
            ('원익IPS', 'IT'), ('피에스케이', 'IT'), ('주성엔지니어링', 'IT'),
            ('테스', 'IT'), ('에이피티씨', 'IT')
        ],
        '디스플레이': [
            ('LG디스플레이', 'IT'), ('덕산네오룩스', 'IT'), ('동운아나텍', 'IT'),
            ('매크로젠', 'IT')
        ],
        '방산': [
            ('한화에어로스페이스', '방산'), ('LIG넥스원', '방산'), ('한화시스템', '방산'),
            ('현대로템', '방산')
        ]
    }
    # 모든 종목 데이터 로딩
    # 모든 종목을 하나의 리스트로 변환
    fetch_start = start_date - timedelta(days=30)
    stock_names = []
    for sector, stocks in sector_stocks.items():
        for stock_name, _ in stocks:
            stock_names.append(stock_name)

    all_target_stock_names = stock_names
    for name in all_target_stock_names:
        code = api_client.get_stock_code(name)
        if code:
            logging.info(f"'{name}' (코드: {code}) 종목 일봉 데이터 로딩 중... (기간: {fetch_start.strftime('%Y%m%d')} ~ {end_date.strftime('%Y%m%d')})")
            daily_df = manager.cache_daily_ohlcv(code, fetch_start, end_date)
            
            if daily_df.empty:
                logging.warning(f"{name} ({code}) 종목의 일봉 데이터를 가져올 수 없습니다. 해당 종목을 건너뜁니다.")
                continue
            logging.debug(f"{name} ({code}) 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
            trade.add_daily_data(code, daily_df)
        else:
            logging.warning(f"'{name}' 종목의 코드를 찾을 수 없습니다. 해당 종목을 건너뜁니다.")

if __name__ == "__main__":
    """
    Backtest 클래스 테스트 실행 코드
    """
    from datetime import date, datetime
    # from strategies.triple_screen_daily import TripleScreenDaily
    # from strategies.dual_momentum_daily import DualMomentumDaily
    from strategies.contrarian_daily import ContrarianDaily
    from strategies.sma_daily import SMADaily
    from strategies.rsi_minute import RSIMinute
    from strategies.open_minute import OpenMinute
    
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/backtest_run.log", encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])    
    try:
        # 1. 필요한 인스턴스들 생성
        logger.info("=== Backtest 테스트 시작 ===")
        # API 클라이언트 (실제 거래용이 아닌 테스트용)
        api_client = CreonAPIClient()
        # DB 매니저
        db_manager = DBManager()
        # Backtest 매니저
        manager = BacktestManager()
        # 리포트 생성기
        report = BacktestReport(db_manager)
        
        # 2. Backtest 인스턴스 생성
        initial_cash = 10_000_000  # 1천만원
        
        backtest = Backtest(
            api_client=api_client,
            initial_cash=initial_cash,
            manager=manager,
            report=report,
            db_manager=db_manager,
            save_to_db=True
        )
        
        # 3. 전략 설정
        # 일봉 전략 파라미터 ------------------------------
        
        # 삼중창 전략 설정 (최적화 결과 반영)
        triple_screen_daily_params={
            'trend_ma_period': 20,          # 유지
            'momentum_rsi_period': 7,      # 유지
            'momentum_rsi_oversold': 35,    # 30 → 35 (매수 조건 더 보수적)
            'momentum_rsi_overbought': 65,  # 70 → 65 (매도 조건 더 보수적)
            'volume_ma_period': 7,         # 유지
            'num_top_stocks': 3,            # 5 → 3 (집중 투자로 승률 향상)
            'safe_asset_code': 'A439870',   # 안전자산 코드 (국고채 ETF)
            'min_trend_strength': 0.02,     # 기본값: 0.02 (2% 추세)
        }

        # 듀얼 모멘텀 전략 설정 (최적화 결과 반영)
        dual_daily_params={
            'momentum_period': 15,         #  15일
            'rebalance_weekday': 0,        #  월요일 (0)
            'num_top_stocks': 3,           #  5 → 3 (집중 투자)
            'safe_asset_code': 'A439870',  # 안전자산 코드 (국고채 ETF)
        }
        
        # 돌파매매 전략 설정 (최적화 결과 반영)
        breakout_daily_params = {
            'breakout_period': 10,          # 20일 → 10일 신고가 돌파
            'volume_ma_period': 20,         # 거래량 20일 이동평균
            'volume_multiplier': 1.5,       # 거래량 이동평균의 1.5배 이상일 때 돌파 인정
            'num_top_stocks': 3,            # 5 → 3 (집중 투자)
            'min_holding_days': 2           # 최소 보유 기간 2일
        }
        
        # SMA 일봉 전략 설정 (최적화 결과 반영)
        sma_daily_params={
            'short_sma_period': 5,          #  4 → 5일 (더 안정적인 단기 이동평균)
            'long_sma_period': 20,          #  10 → 20일 (더 안정적인 장기 이동평균)
            'volume_ma_period': 10,         #  6 → 10일 (거래량 이동평균 기간 확장)
            'num_top_stocks': 5,            #  5 → 3 (집중 투자)
        }

        # 역추세 일봉 전략 설정
        contrarian_daily_params={
            'rsi_period': 5,               # RSI 계산 기간 14
            'rsi_oversold': 30,             # RSI 과매도 기준
            'rsi_overbought': 70,           # RSI 과매수 기준
            'bb_period': 12,                # 볼린저 밴드 기간 20
            'bb_std': 2,                    # 볼린저 밴드 표준편차
            'stoch_period': 5,             # 스토캐스틱 기간 14
            'num_top_stocks': 5,            # 선택 종목 수
            'min_holding_days': 3,          # 최소 홀딩 기간 (역추세는 더 오래 홀딩)
        }

        # 분봉 전략 파라미터 ------------------------------
        
        # 돌파 분봉 전략 설정 (실제 RSI 계산 및 매매)
        breakout_minute_params={
            'minute_breakout_period': 10,       # 10분봉 최고가 돌파 확인 기간
            'minute_volume_multiplier': 1.8     # 분봉 거래량 이동평균의 1.8배 이상일 때 돌파 인정
        }

        # RSI 분봉 전략 설정 (최적화 결과 반영)
        rsi_minute_params={
            'minute_rsi_period': 35,        #  5 → 52분 (더 안정적인 RSI)
            'minute_rsi_oversold': 34,      # 30 → 34 (매수 조건 보수적)
            'minute_rsi_overbought': 70,    # 60 → 70 (매도 조건 보수적)
            'num_top_stocks': 5,            # 5 → 3 (일봉 전략과 동일)
        }
        
        # RSI 가상 분봉 전략 설정 (최적화 결과 반영)
        open_minute_params={
            'minute_rsi_period': 52,        #  52분
            'minute_rsi_oversold': 34,      # 34 (매수 조건 보수적)
            'minute_rsi_overbought': 70,    # 70 (매도 조건 보수적)
            'num_top_stocks': 3,            # 10 → 3 (집중 투자)
        }

        # 전략 인스턴스 생성
        daily_strategy = SMADaily(backtest.broker, backtest.data_store, strategy_params=sma_daily_params)
        #daily_strategy = ContrarianDaily(backtest.broker, backtest.data_store, strategy_params=contrarian_daily_params)
        #minute_strategy = OpenMinute(backtest.broker, backtest.data_store, strategy_params=open_minute_params)
        minute_strategy = RSIMinute(backtest.broker, backtest.data_store, strategy_params=rsi_minute_params)
        # 4. 손절매 파라미터 설정 (선택사항)
        stop_loss_params = {
            'take_profit_ratio': 20,       # 매수 후 익절
            'early_stop_loss': -5,        # 매수 후 초기 손실 제한: -3.5% (매수 후 3일 이내)
            'stop_loss_ratio': -10,        # 매수가 기준 손절율: -6.0%
            'trailing_stop_ratio': -7,    # 최고가 기준 트레일링 손절률: -4.0%
            'portfolio_stop_loss': -4,    # 전체 자본금 손실률 (전량매도 조건): -4.0%
            'max_losing_positions': 4       # 최대 손절 종목 수 (전량매도 조건): 3개
        }

        start_date = date(2025, 6, 1)
        end_date = date(2025, 7, 7)

        # 종목 일봉 설정 ===========================
        load_stocks(backtest, manager, db_manager, start_date, end_date)

        backtest.set_strategies(
            daily_strategy=daily_strategy,
            minute_strategy=minute_strategy
        )
        #손절 설정 None면 손절기능 사용 않음        
        #backtest.set_broker_stop_loss_params(stop_loss_params)
        backtest.set_broker_stop_loss_params() # 손절 않음
        
        # 5. 백테스트 실행

        # 끝 전략 설정 =========================
        
        # 백테스트 실행
        backtest.run(start_date, end_date)
        
        # logger.info("=== Backtest 테스트 완료 ===")
        
    except Exception as e:
        logger.error(f"Backtest 테스트 중 오류 발생: {e}", exc_info=True)
    finally:
        # 리소스 정리
        if 'db_manager' in locals():
            db_manager.close()

