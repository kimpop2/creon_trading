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
from selector.stock_selector import StockSelector # StockSelector 타입 힌트를 위해 남겨둠
from api.creon_api import CreonAPIClient

# --- 로거 설정 (스크립트 최상단에서 설정하여 항상 보이도록 함) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 테스트 시 DEBUG로 설정하여 모든 로그 출력 - 제거

class Backtest:
    # __init__ 메서드를 외부에서 필요한 인스턴스를 주입받도록 변경
    def __init__(self, api_client: CreonAPIClient, initial_cash: float, 
                 backtest_manager: BacktestManager, backtest_report: BacktestReport, stock_selector: StockSelector,
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
        self.stock_selector = stock_selector
        
        self.current_day_signals = {}
        # NEW: 현재 날짜의 분봉 매매를 위해 사용될 신호들을 저장
        self.signals_for_minute_trading = {}
        # NEW: 포트폴리오 손절 체크 시간을 추적하기 위한 변수
        self.last_portfolio_check_time = None
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

    def run(self, start_date: datetime.date, end_date: datetime.date):
        logging.info(f"백테스트 시작: {start_date} 부터 {end_date} 까지")
        current_date = start_date
        
        # 시작일 이전의 초기 포트폴리오 가치 기록
        initial_portfolio_value = self.broker.get_portfolio_value({}) # 초기 현금만 반영
        self.portfolio_values.append((current_date - datetime.timedelta(days=1), initial_portfolio_value))
        
        while current_date <= end_date:
            
            # 오늘이 영업일인지 확인 (데이터가 없으면 건너뛰기)
            daily_data_available = False
            for stock_code in self.data_store['daily']:
                # 의미 없음
                # if self.daily_strategy and stock_code == self.daily_strategy.strategy_params.get('safe_asset_code'):
                #     daily_data_available = True
                #     break
                
                # 오늘 일봉이 하나라도 있다면 영업일
                if not self.data_store['daily'][stock_code].empty and \
                   current_date in self.data_store['daily'][stock_code].index.date:
                    daily_data_available = True
                    break
            # end for

            # 영업일이 아니라면 일봉 신호, 분봉에 전달할 신호, 포토폴리오 체크시간 및 손절 플래그 초기화
            if not daily_data_available:
                logging.info(f"{current_date.isoformat()}는 휴장(공휴일)입니다. (일봉 데이터 없음)")
                # 신호 저장소도 초기화
                self.current_day_signals = {}
                self.signals_for_minute_trading = {}
                self.last_portfolio_check_time = None
                self.portfolio_stop_flag = False
                # 영럽일이 아니면 다음날로 이동하고 루프 다시 시작 == 여기서 실행 끝
                current_date += datetime.timedelta(days=1)
                continue 
            # 영업일 체크 끝

            # 영업일이면 백테스트 시작
            logging.info(f"\n--- 현재 백테스트 날짜: {current_date.isoformat()} ---")

            # 1. 이전 날짜에 일봉 전략이 생성한 신호를 '오늘' 사용할 신호로 복사하고, 다음 날을 위해 신호 저장소를 비웁니다.
            # 백테스트 첫날이 아닌 경우에만 이전 신호를 복사
            # if current_date > start_date:
            #     self.signals_for_minute_trading = self.current_day_signals.copy()

            # 2. 전일 일봉 데이터를 기반으로 '오늘 실행할' 신호를 생성합니다.
            if self.daily_strategy:
                # 전일 날짜 계산
                prev_trading_day = None
                for stock_code in self.data_store['daily']:
                    # 종목의 직전 영업일 찾는 로직
                    df = self.data_store['daily'][stock_code]
                    if not df.empty and current_date in df.index.date:
                        idx = list(df.index.date).index(current_date) # 오늘날짜 인덱스
                        if idx > 0:
                            prev_trading_day = df.index.date[idx-1]   # 오늘날짜 인덱스-1 -> 직전영업일
                            break
                
                # 직전영업일이 있다. (지표계산을 위해 백테스트 시작일 이전 1달치 일봉데이터도 저장되어 있음)
                if prev_trading_day:
                    ##############################
                    # 전일 데이터로 일봉 전략 실행 - 오늘은 아직 장이 시작하지 않았으므로(전일 종가까지 데이터로)
                    self.daily_strategy.run_daily_logic(prev_trading_day)
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
                                'signal_date': prev_trading_day 
                            }

                    logging.debug(f"[{current_date.isoformat()}] 전일({prev_trading_day.isoformat()}) 일봉 전략 실행 완료: {len(self.current_day_signals)}개의 당일 매매 신호 생성.")
                else:
                    # 직전영업일이(오류) 없다, 다음날짜 이동 - 일봉 신호, 분봉에 전달할 신호, 포토폴리오 체크시간 및 손절 플래그 초기화
                    logging.warning(f"[{current_date.isoformat()}] 전일 데이터를 찾을 수 없어(오류) 일봉 전략을 건너뜁니다.")
                    # 신호 저장소도 초기화
                    self.current_day_signals = {}
                    self.signals_for_minute_trading = {}
                    self.last_portfolio_check_time = None
                    self.portfolio_stop_flag = False
                    # 영럽일이 아니면 다음날로 이동하고 루프 다시 시작 == 여기서 실행 끝
                    current_date += datetime.timedelta(days=1)
                    continue 

            # 3. 오늘 분봉 매매 로직을 실행합니다.
            if self.minute_strategy:
                # 분봉전략명이 OpenMinute 인지 확인 해서 아래에서 다르게 처리한다. 
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
                
                ##########################
                # 분봉매매 실행
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
                        
                        # OpenMinute 전략은 특정 시간(9:01)에만 매매를 실행하므로,
                        # stocks_to_trade에 포함된 종목들에 대해 한 번씩만 호출
                        for stock_code in stocks_to_trade:
                            # 포트폴리오 손절은 run_minute_logic 에 직접 만들어야 한다. 현재 기능 없음 (종목손절, 트레일링 매도 있음)
                            # if self.portfolio_stop_flag:
                            #     logging.debug(f"[{current_date.isoformat()}] 포트폴리오 손절 발생으로 당일 매매 중단.")
                            #     break
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
                        
                        # 손절매 처리를 해야 한다면 보유 중인 종목들 추가
                        if has_stop_loss:
                            current_positions = set(self.broker.positions.keys()) #### 보유종목 구하기
                            stocks_to_load.update(current_positions)
                        
                        # 분봉 데이터가 필요한 종목들이 결정 되었으므로, 이 종목에 대한 분봉 데이터 로드
                        logging.info(f"[{current_date.isoformat()}] 분봉 데이터 로드 시작: {len(stocks_to_load)}개 종목")

                        # 필요한 종목들의 분봉 데이터를 로드
                        for stock_code in stocks_to_load:
                            # # 수정: 전일 영업일부터 당일까지의 분봉 데이터 로드
                            # # 전일 영업일 계산
                            # prev_trading_day = None
                            # for code in self.data_store['daily']:
                            #     df = self.data_store['daily'][code]
                            #     if not df.empty and current_date in df.index.date:
                            #         idx = list(df.index.date).index(current_date)
                            #         if idx > 0:
                            #             prev_trading_day = df.index.date[idx-1]
                            #             break
                            signal_info = self.current_day_signals.get(stock_code)
                            prev_trading_day = signal_info.get('signal_date', current_date) if signal_info else current_date
                            if prev_trading_day:
                                # BacktestManager를 사용하여 분봉 데이터 로드 (전일~당일)
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
                        
                        
                        # 3-3. 분봉 매매 로직 실행
                        # 실제로 매매가 필요한 종목들만 선별
                        stocks_to_trade = set()
                        # 매수/매도 신호가 있는 종목들 추가
                        for stock_code, signal_info in self.signals_for_minute_trading.items():
                            if signal_info['signal'] in ['buy', 'sell']:
                                stocks_to_trade.add(stock_code)
                        # 현재 보유 중인 종목들 추가 (손절매 체크용)
                        if has_stop_loss:
                            current_positions = set(self.broker.positions.keys())
                            stocks_to_trade.update(current_positions)

                        ###################################
                        # 분봉 전략 매매 실행
                        ###################################
                        # 장 시작 시간부터 장 마감 시간까지 1분 단위로 반복하며 분봉 전략 실행
                        start_time = time(9, 0) # 9시 정각
                        end_time = time(15, 30) # 3시 30분 (장 마감)
                        trade_time = datetime.datetime.combine(current_date, start_time)
                        # 오늘 장이 끝날때 까지 매분 반복 실행
                        while trade_time <= datetime.datetime.combine(current_date, end_time):
                            # # 시초
                            # if trade_time.time() == time(9, 0):
                            #     trade_time += datetime.timedelta(minutes=1)
                            #     continue
                            
                            # 포트폴리오 손절 시 당일 매매 중단 : 특정시간에 체크하여 설정 (처리 속도 때문)
                            if self.portfolio_stop_flag:
                                logging.debug(f"[{current_date.isoformat()} {trade_time.strftime('%H:%M')}] 포트폴리오 손절 발생으로 당일 매매 중단.")
                                break
                            for stock_code in stocks_to_trade:
                                # 해당 시간의 분봉 데이터가 없으면 건너뛰기
                                if stock_code not in self.data_store['minute'] or \
                                   current_date not in self.data_store['minute'][stock_code] or \
                                   trade_time not in self.data_store['minute'][stock_code][current_date].index:
                                    logging.debug(f"[{trade_time.isoformat()}] {stock_code}: 해당 시간의 분봉 데이터 없음. 스킵.")
                                    continue
                                ##############################
                                # 분봉 전략 실행 및 매매 처리
                                ##############################
                                self.minute_strategy.run_minute_logic(trade_time, stock_code)
                            
                            # 포트폴리오 손절매 체크 (하루 2번, 지정 시간에만)
                            if has_stop_loss:
                                current_minutes = trade_time.hour * 60 + trade_time.minute
                                check_times = [9 * 60, 15 * 60 + 20] # 지정 시간: 9:00, 15:20
            ########################### 포트폴리오 손절매 로직 재구성 필요 ##########################                     
                                # # 손절매 지정 시간이면 실행
                                # if current_minutes in check_times and (self.last_portfolio_check_time is None or self.last_portfolio_check_time.time() != trade_time.time()):
                                #     self.last_portfolio_check_time = trade_time
                                    
                                #     current_prices = {
                                #         s_code: self.data_store['minute'][s_code][current_date]['close'].get(trade_time, np.nan)
                                #         for s_code in self.broker.positions if s_code in self.data_store['minute'] and current_date in self.data_store['minute'][s_code]
                                #     }
                                #     # 현재 가격이 없는 종목은 제외
                                #     current_prices = {k: v for k, v in current_prices.items() if not np.isnan(v)}

                                #     if self.broker.check_and_execute_portfolio_stop_loss(current_prices, trade_time):
                                #         logging.info(f"[{trade_time.isoformat()}] 포트폴리오 손절매 실행 완료. 오늘의 매매 종료.")
                                #         self.portfolio_stop_flag = True  # 플래그 설정
                                #         break # 분봉 루프를 종료하고 일일 포트폴리오 처리 후 다음 "영업일"로 넘어감

                            trade_time += datetime.timedelta(minutes=1)
                # end if 분봉매매

            # end if 분봉 전략 유무
                            
            # 일일 포트폴리오 가치 기록 (장 마감 시점)
            # 장 마감 종가를 가져오기 위해 일봉 데이터 사용
            current_day_close_prices = {}
            for stock_code, df in self.data_store['daily'].items():
                # 오늘 날짜에 데이터가 있으면 사용
                day_data = df[df.index.date == current_date]
                if not day_data.empty:
                    # 오늘의 종가
                    current_day_close_prices[stock_code] = day_data['close'].iloc[0]
                else:
                    # 오늘 데이터가 없으면, 전 영업일의 종가 사용
                    # 종목의 직전 영업일 찾는 로직
                    # 전일 날짜 계산
                    prev_trading_day = None
                    df = self.data_store['daily'][stock_code]
                    if not df.empty and current_date in df.index.date:
                        idx = list(df.index.date).index(current_date) # 오늘날짜 인덱스
                        if idx > 0:
                            prev_trading_day = df.index.date[idx-1]   # 오늘날짜 인덱스-1 -> 직전영업일
                            break

                    prev_data = df[df.index.date == prev_trading_day]
                    if not prev_data.empty:
                        last_close = prev_data['close'].iloc[-1]
                        current_day_close_prices[stock_code] = last_close
                        logging.warning(f"경고: {stock_code}의 현재 가격 데이터가 없어 최근 영업일({prev_data.index[-1].date()}) 종가({last_close})를 사용합니다.")
                    else:
                        logging.warning(f"경고: {stock_code}의 현재 및 과거 가격 데이터가 모두 없습니다. 포트폴리오 가치 계산에서 제외됩니다.")
            
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