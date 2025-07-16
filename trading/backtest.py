# backtest/backtester.py
from datetime import datetime, time, timedelta
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import sys
import os
# 프로젝트 루트 경로를 sys.path에 추가 (manager 디렉토리에서 실행 가능하도록)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.broker import Broker
from api.creon_api import CreonAPIClient
from manager.backtest_manager import BacktestManager # BacktestManager 타입 힌트를 위해 남겨둠
from manager.db_manager import DBManager    
from util.strategies_util import *
from strategies.strategy import BaseStrategy
from trading.abstract_report import ReportGenerator, BacktestDB
logger = logging.getLogger(__name__)

class Backtest:
    # __init__ 메서드를 외부에서 필요한 인스턴스를 주입받도록 변경
    def __init__(self, 
                 api_client: CreonAPIClient, 
                 db_manager: DBManager,
                 initial_cash: float, 
                 save_to_db: bool = True
        ):  # DB 저장 여부 파라미터 추가
        
        self.api_client = api_client
        self.db_manager = db_manager
        # data_store의 minute 키는 이제 {stock_code: DataFrame}으로 변경됩니다.
        # 즉, 각 종목별로 전체 기간의 분봉 데이터를 하나의 DataFrame으로 가집니다.
        self.data_store = {'daily': {}, 'minute': {}} 
        self.portfolio_values = [] # (datetime, value) 튜플 저장
        self.initial_cash = initial_cash
        self.save_to_db = save_to_db  # DB 저장 여부 저장
        self.market_open_time = time(9, 0, 0)
        self.market_single_time = time(15, 20, 0)
        self.market_close_time = time(15, 30, 0)
        
        # NEW: DailyStrategy와 MinuteStrategy 인스턴스를 별도로 관리
        self.daily_strategy: BaseStrategy = None
        self.minute_strategy: BaseStrategy = None
        
        # 외부에서 주입받은 인스턴스를 사용
        self.manager = BacktestManager(self.api_client, self.db_manager) # manager 초기화
        self.broker = Broker(self.initial_cash)
        
        # NEW: 포트폴리오 손절 체크 시간을 추적하기 위한 변수
        self.last_portfolio_check = None
        # 포트폴리오 손절 발생 시 당일 매매 중단 플래그
        self.portfolio_stop_flag = False

        # 일봉 업데이트 캐시 변수들 (성능 개선용)
        # _daily_update_cache는 이제 필요 없습니다. _minute_data_cache가 그 역할을 대신합니다.
        # _minute_data_cache는 이제 각 종목별로 당일의 분봉 데이터를 저장합니다.
        self._minute_data_cache = {}   # {stock_code: DataFrame (오늘치 분봉 데이터)}
        # [추가] 가격 캐시를 Backtest 클래스가 직접 소유
        self.price_cache = {}
        logging.info(f"백테스터 초기화 완료. 초기 현금: {self.initial_cash:,.0f}원, DB저장: {self.save_to_db}")

    # NEW: set_strategies 메서드를 변경하여 DailyStrategy와 MinuteStrategy를 모두 받도록 함
    def set_strategies(self, daily_strategy: BaseStrategy, minute_strategy: BaseStrategy):
            self.daily_strategy = daily_strategy
            self.minute_strategy = minute_strategy
            logging.info(f"일봉 전략 '{daily_strategy.__class__.__name__}' 및 분봉 전략 '{minute_strategy.__class__.__name__}' 설정 완료.")
        
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

    # 분봉 데이터는 run 루프 내에서 cache_minute_ohlcv를 통해 직접 data_store['minute']에 저장됩니다.
 
    def _should_check_portfolio(self, current_dt):
        """포트폴리오 체크가 필요한 시점인지 확인합니다."""
        if self.last_portfolio_check is None:
            self.last_portfolio_check = current_dt # 첫 체크 시점 기록
            return True
        
        current_time = current_dt.time()
        # 시간 비교를 정수로 변환하여 효율적으로 비교
        current_minutes = current_time.hour * 60 + current_time.minute
        check_minutes = [9 * 60, 15 * 60 + 20]  # 9:00, 15:20
        
        # 오늘 날짜이고, 체크 시간에 해당하며, 마지막 체크 시점이 오늘이 아니거나 해당 체크 시간이 아닌 경우
        if current_dt.date() == self.last_portfolio_check.date():
            if current_minutes in check_minutes and (self.last_portfolio_check.hour * 60 + self.last_portfolio_check.minute) != current_minutes:
                self.last_portfolio_check = current_dt
                return True
        elif current_dt.date() > self.last_portfolio_check.date(): # 날짜가 바뀌면 무조건 체크
            self.last_portfolio_check = current_dt
            return True
            
        return False

    # 오늘의 일봉 데이터가 미래정보를 가지지 않도록 제한한
    def _update_daily_data_from_minute_bars(self, current_dt: datetime.datetime):
        """
        매분 현재 시각까지의 1분봉 데이터를 집계하여 일봉 데이터를 생성하거나 업데이트합니다.
        _minute_data_cache를 활용하여 성능을 개선합니다.
        :param current_dt: 현재 시각 (datetime 객체)
        """
        current_date = current_dt.date()
        
        for stock_code in self.data_store['daily'].keys():
            # _minute_data_cache에서 해당 종목의 오늘치 전체 분봉 데이터를 가져옵니다.
            # 이 데이터는 run 함수 진입 시 cache_minute_ohlcv를 통해 이미 로드되어 있어야 합니다.
            today_minute_bars = self._minute_data_cache.get(stock_code)

            if today_minute_bars is not None and not today_minute_bars.empty:
                # 현재 시각까지의 분봉 데이터만 필터링 (슬라이싱)
                # 이 부분에서 불필요한 복사를 줄이기 위해 .loc을 사용합니다.
                filtered_minute_bars = today_minute_bars.loc[today_minute_bars.index <= current_dt]
                
                if not filtered_minute_bars.empty:
                    # 현재 시각까지의 일봉 데이터 계산
                    daily_open = filtered_minute_bars.iloc[0]['open']  # 첫 분봉 시가
                    daily_high = filtered_minute_bars['high'].max()    # 현재까지 최고가
                    daily_low = filtered_minute_bars['low'].min()      # 현재까지 최저가
                    daily_close = filtered_minute_bars.iloc[-1]['close']  # 현재 시각 종가
                    daily_volume = filtered_minute_bars['volume'].sum()   # 현재까지 누적 거래량

                    # 새로운 일봉 데이터 생성 (Series로 생성하여 성능 개선)
                    new_daily_bar = pd.Series({
                        'open': daily_open,
                        'high': daily_high,
                        'low': daily_low,
                        'close': daily_close,
                        'volume': daily_volume
                    }, name=pd.Timestamp(current_date)) # 인덱스를 날짜로 설정

                    # 일봉 데이터가 존재하면 업데이트, 없으면 추가
                    # .loc을 사용하여 직접 업데이트 (기존 DataFrame의 인덱스를 활용)
                    self.data_store['daily'][stock_code].loc[pd.Timestamp(current_date)] = new_daily_bar
                    
                    # 일봉 데이터가 추가되거나 업데이트될 때 인덱스 정렬은 필요 없음 (loc으로 특정 위치 업데이트)
                    # 단, 새로운 날짜가 추가될 경우 기존 DataFrame에 없던 인덱스가 추가되므로 sort_index는 필요할 수 있습니다.
                    # 하지만 백테스트에서는 날짜 순서대로 진행되므로 대부분의 경우 문제가 되지 않습니다.
            else:
                logging.debug(f"[{current_dt.isoformat()}] {stock_code}의 오늘치 분봉 데이터가 없거나 비어있어 일봉 업데이트를 건너킵니다.")


    def _clear_daily_update_cache(self):
        """
        일봉 업데이트를 위한 분봉 데이터 캐시를 초기화합니다. 새로운 날짜로 넘어갈 때 호출됩니다.
        """
        self._minute_data_cache.clear()
        logging.debug("일봉 업데이트를 위한 분봉 데이터 캐시 초기화 완료.")

    def get_all_current_prices(self, current_dt):
        current_prices = {}
        for code in self.broker.positions.keys():
            price_data = self.minute_strategy._get_ohlcv_at_time(code, current_dt)
            # 가격 데이터가 있고, 종가가 NaN이 아닌 경우에만 추가
            if price_data is not None and not np.isnan(price_data['close']):
                current_prices[code] = price_data['close']  
        return current_prices
            
    def run(self, start_date: datetime.date, end_date: datetime.date):
        logging.info(f"백테스트 시작: {start_date} 부터 {end_date} 까지")
        
        # NEW: 전략 인스턴스 확인
        if not self.daily_strategy or not self.minute_strategy:
            logger.error("일봉 또는 분봉 전략이 설정되지 않았습니다. 백테스트를 중단합니다.")
            return
        
        has_stop_loss = self.broker.stop_loss_params is not None # 손절처리 유무

        # 시장 캘린더 데이터 로드 (전영업일 계산을 위해 충분히 가져옴)
        market_calendar_df = self.db_manager.fetch_market_calendar(start_date - timedelta(days=10), end_date)
        if market_calendar_df.empty:
            logger.error("시장 캘린더 데이터를 가져올 수 없습니다. 백테스트를 중단합니다.")
            return
        # 영업일만 필터링하고 날짜를 리스트로 변환하여 정렬
        trading_dates = market_calendar_df[market_calendar_df['is_holiday'] == 0]['date'].dt.date.tolist()
        trading_dates.sort() # 날짜가 오름차순으로 정렬되도록 보장

        # 실제 백테스트 시작 날짜 찾기 (trading_dates에 start_date가 포함될 때부터)
        actual_start_index = -1
        for i, d in enumerate(trading_dates):
            if d >= start_date:
                actual_start_index = i
                break
        
        if actual_start_index == -1:
            logger.error(f"백테스트 시작일 {start_date}이(가) 시장 캘린더에 없습니다. 백테스트를 중단합니다.")
            return

        # 백테스트 시작일 이전의 초기 포트폴리오 가치 기록을 위해 전 영업일 찾기
        prev_date_for_initial_portfolio = trading_dates[actual_start_index - 1] if actual_start_index > 0 else start_date - timedelta(days=1)
        initial_portfolio_value = self.broker.get_portfolio_value({}) # 초기 현금만 반영
        self.portfolio_values.append((prev_date_for_initial_portfolio, initial_portfolio_value))
        
        # #########################################                      
        # 영업일을 순회하며 전략 실행
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        for i in range(actual_start_index, len(trading_dates)):
            current_date = trading_dates[i]
            if current_date > end_date: # 종료일 초과하면 중단
                break

            logging.info(f"\n--- 현재 백테스트 날짜: {current_date.isoformat()} ---")

            # 새로운 날짜로 넘어갈 때 일봉 업데이트 캐시 초기화
            self._clear_daily_update_cache() # 오늘치 분봉 캐시 초기화

            # 전일 날짜 계산
            prev_date = trading_dates[i - 1] if i > 0 else current_date - timedelta(days=1) # 첫 날은 임시로 전날 사용

            logging.info(f"-------- {current_date.isoformat()} 매매 시작 --------")
            
            # 1. 일봉 로직 실행
            # DailyStrategy.run_daily_logic()을 호출하여 신호 생성
            self.daily_strategy.run_daily_logic(prev_date)
            
            # 2. 신호 동기화
            # DailyStrategy가 생성한 신호(self.daily_strategy.signals)를 MinuteStrategy로 전달
            # MinuteStrategy는 이 신호를 기반으로 분봉 매매를 실행
            self.minute_strategy.update_signals(self.daily_strategy.signals)
            
            # 3. 필요한 종목(보유+매매신호)들의 분봉 데이터를 로드 (전일 + 당일) vvvvvvvvvvvvvvvvvvvvvv
            stocks_to_load = set()  # 분봉 데이터가 필요한 종목들
            # 매수/매도 신호가 있는 종목들 추가
            for stock_code, signal_info in self.minute_strategy.signals.items():
                if signal_info['signal_type'] in ['buy', 'sell']:
                    stocks_to_load.add(stock_code)
            # 보유종목 추가
            current_positions = set(self.broker.positions.keys()) 
            stocks_to_load.update(current_positions)                        
            # 3-1. 먼저 당일 실행할 신호(매수/매도/보유)가 있는 종목들의 분봉 데이터를 모두 로드
            for stock_code in stocks_to_load:
                signal_info = self.minute_strategy.signals.get(stock_code)
                prev_date = signal_info.get('signal_date', current_date) if signal_info else current_date

                if not prev_date:
                    prev_date = current_date 

                # BacktestManager를 사용하여 분봉 데이터 로드 (전일~당일)
                minute_df = self.manager.cache_minute_ohlcv(
                    stock_code,
                    prev_date,      # 전일 영업일부터
                    current_date    # 당일까지
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
            
                else:
                     logging.warning(f"{stock_code} 종목의 {prev_date} ~ {current_date} 분봉 데이터 로드 실패.")
            
            # for 분봉로드 끝 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # 3-2. 분봉 매매 대상은 신호(매수/매도) 손절 파라미터 설정시 보유 종목
            stocks_to_trade = set() # 매매대상
            
            # 매수/매도 신호가 있는 종목들 추가
            for stock_code, signal_info in self.minute_strategy.signals.items():
                if signal_info['signal_type'] in ['buy', 'sell']:
                    stocks_to_trade.add(stock_code)
            # 손절매 기능이 있다면, 보유 중인 종목들 추가 (손절매 체크용)
            if has_stop_loss:
                current_positions = set(self.broker.positions.keys())
                stocks_to_trade.update(current_positions)

            ###################################
            # 오늘 장이 끝날때 까지 매분 반복 실행
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            open_time = time(9, 0) # 9시 정각
            market_open_dt = datetime.combine(current_date, open_time)

            for minute_offset in range(380): # 9:00부터 15:30까지 (6시간 30분 = 390분 - 10분 (9:00, 3:21~29분 포함))
                current_dt = market_open_dt + timedelta(minutes=minute_offset)
                
                if current_dt.time() <= time(9, 0) or current_dt.time() > time(15, 30):
                    continue

                logging.debug(f"[{current_dt.isoformat()}] 분봉 매매 로직 실행 중...")
                #self._update_daily_data_from_minute_bars(current_dt)
                
                # --- 1. 포지션 위험 관리 (가장 먼저 실행) ---
                # 현재 모든 종목의 가격을 가져옴
                current_prices = self.get_all_current_prices(current_dt) 
                
                # 보유 포지션에 대한 손절/익절 등을 먼저 체크하고 실행
                # 이 함수는 내부적으로 보유 종목 전체를 루프 돌며 확인합니다.
                self.broker.check_and_execute_stop_loss(current_prices, current_dt)

                
                
                # --- [핵심 수정] ---
                # 불필요한 시간 체크 조건을 제거하고, 모든 매매 대상 종목에 대해
                # run_minute_logic을 무조건 호출합니다.
                # 데이터 존재 여부는 run_minute_logic 내부에서 책임집니다.
                for stock_code in stocks_to_trade:
                    self.minute_strategy.run_minute_logic(current_dt, stock_code)
                # -----------------

                # 분봉 매매 끝 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                
                # 포토폴리오 손절 vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                # 포트폴리오 손절을 위한 9:00, 15:20 시간 체크, 분봉마다하는 것이 정확하겠지만 속도상 
                if self.broker.stop_loss_params is not None and self._should_check_portfolio(current_dt):
                    


                    # 손절/익절/트레일링 매도 처리 후 포트폴리오 손절 판단 True, False                          
                    if self.broker.check_and_execute_stop_loss(current_prices, current_dt):
                        # 매도처리
                        for code in list(self.minute_strategy.signals.keys()):
                            # 매도된 것 신호 정리
                            if code in self.broker.positions and self.broker.positions[code]['size'] == 0: # 매도 후 == 수량 0
                                self.minute_strategy.reset_signal(stock_code)
                        
                        logging.info(f"[{current_dt.isoformat()}] 포트폴리오 손절매 실행 완료. 오늘의 매매 종료.")
                        break # 분봉 루프를 종료, 일일 포트폴리오 처리 후 다음 "영업일"로 넘어감
                # 포트폴리오 손절 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # 끝 매분 반복
            ###################################

            logging.info(f"-------- {current_date.isoformat()} 매매 종료 --------")

            # 4. 하루 종료 후 보고서 생성 및 DB 저장 (TradingManager를 통해)
            # 일일 포트폴리오 가치 기록 (장 마감 시점)
            current_day_close_prices = {}
            for stock_code, position_info in self.broker.positions.items():
                # 오늘 일봉데이터 다시 가져오기 (매니저에서 캐시된 데이터 활용)
                # current_date에 해당하는 일봉 데이터의 종가를 가져와야 함
                daily_data_for_today = self.data_store['daily'].get(stock_code)
                if daily_data_for_today is not None and not daily_data_for_today.empty and pd.Timestamp(current_date).normalize() in daily_data_for_today.index.normalize():
                    current_day_close_prices[stock_code] = daily_data_for_today.loc[pd.Timestamp(current_date).normalize()]['close']
                else:
                    logging.warning(f"종목 {stock_code}의 {current_date} 일봉 마감 가격을 찾을 수 없어 포트폴리오 가치 계산에 어려움이 있을 수 있습니다.")
                    current_day_close_prices[stock_code] = position_info['avg_price'] # 대안으로 평균 단가 사용

            daily_portfolio_value = self.broker.get_portfolio_value(current_day_close_prices)
            self.portfolio_values.append((current_date, daily_portfolio_value))
            logging.info(f"[{current_date.isoformat()}] 장 마감 포트폴리오 가치: {daily_portfolio_value:,.0f}원, 현금: {self.broker.get_current_cash_balance():,.0f}원")
            
            # 수정: 다음날을 위해 모든 신호 초기화
            # DailyStrategy와 MinuteStrategy의 신호를 모두 초기화
            self.daily_strategy._reset_all_signals()
            
            # 2. 백테스터의 신호 저장소 초기화 (다음날을 위해)
            self.last_portfolio_check = None # 다음 날을 위해 손절 체크 시간 초기화
             
            logging.debug(f"[{current_date.isoformat()}] 일일 신호 초기화 완료 - 다음날을 위해 모든 신호 저장소 비움")

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # 끝 매 영업일 반복
        ###################################


        logging.info("백테스트 완료. 최종 보고서 생성 중...")
        # 백테스트 결과 보고서 생성 및 저장
        # 초기 포트폴리오 항목 (시작일 이전의 초기 자본)은 제외하고 실제 백테스트 기간 데이터만 사용
        if len(self.portfolio_values) > 1:
            portfolio_df = pd.DataFrame(self.portfolio_values[1:], columns=['Date', 'PortfolioValue'])
            portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
            portfolio_value_series = portfolio_df.set_index('Date')['PortfolioValue']
        else:
            # 백테스트 기간이 너무 짧아 포트폴리오 가치 데이터가 없는 경우
            portfolio_value_series = pd.Series(dtype=float)

        if self.save_to_db and not portfolio_value_series.empty:
            # 1. 백테스트용 저장 전략 선택
            storage = BacktestDB(self.db_manager)

            # 2. 리포트 생성기에 저장 전략 주입
            reporter = ReportGenerator(storage_strategy=storage)

            # 3. 리포트 생성 및 저장
            reporter.generate(
                start_date=start_date,
                end_date=end_date,
                initial_cash=self.initial_cash,
                portfolio_value_series=portfolio_value_series,
                transaction_log=self.broker.transaction_log,
                strategy_info={
                    'strategy_daily': self.daily_strategy.__class__.__name__,
                    'strategy_minute': self.minute_strategy.__class__.__name__,
                    'params_json_daily': self.daily_strategy.strategy_params,
                    'params_json_minute': self.minute_strategy.strategy_params
                }
            )
        elif portfolio_value_series.empty:
             logger.warning("포트폴리오 가치 데이터가 비어 있어 DB에 결과를 저장하지 않습니다.")
        else:
             logger.info("save_to_db=False로 설정되어 DB에 결과를 저장하지 않습니다.")


        final_metrics = calculate_performance_metrics(portfolio_value_series)
        return portfolio_value_series, final_metrics
    
    def cleanup(self) -> None:
        """
        시스템 종료 시 필요한 리소스 정리 작업을 수행합니다.
        """
        logger.info("Backtest 시스템 cleanup 시작.")
        if self.broker:
            self.broker.cleanup()
        if self.api_client:
            self.api_client.cleanup()
        if self.db_manager:
            self.db_manager.close()
        logger.info("Backtest 시스템 cleanup 완료.")    

    def load_stocks(self, start_date, end_date):
        from config.sector_stocks import sector_stocks

        fetch_start = start_date - timedelta(days=60) # 전략에 필요한 최대 기간 + 여유
        stock_codes_to_load = []
        for sector, stocks in sector_stocks.items():
            for stock_name, _ in stocks:
                code = self.api_client.get_stock_code(stock_name)
                if code:
                    stock_codes_to_load.append(code)
                else:
                    logging.warning(f"'{stock_name}' 종목의 코드를 찾을 수 없습니다. 해당 종목을 건너킵니다.")
        # 하드 코딩으로 추가 ###############
        stock_codes_to_load.append('U001')
        for code in stock_codes_to_load:
            stock_name = self.api_client.get_stock_name(code) # 종목명 다시 가져오기
            logging.info(f"'{stock_name}' (코드: {code}) 종목 일봉 데이터 로딩 중... (기간: {fetch_start.strftime('%Y%m%d')} ~ {end_date.strftime('%Y%m%d')})")
            daily_df = self.manager.cache_daily_ohlcv(code, fetch_start, end_date)
            
            if daily_df.empty:
                logging.warning(f"{stock_name} ({code}) 종목의 일봉 데이터를 가져올 수 없습니다. 해당 종목을 건너킵니다.")
                continue
            logging.debug(f"{stock_name} ({code}) 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
            self.add_daily_data(code, daily_df)
            
            # 분봉 데이터는 run 루프 내에서 필요한 시점에 로드되므로, 여기서는 미리 로드하지 않습니다.
            # 이전에 add_minute_data를 통해 전체 분봉을 로드하는 방식은 제거되었습니다.


if __name__ == "__main__":
    """
    Backtest 클래스 테스트 실행 코드
    """
    from datetime import date, datetime
    # 설정 파일 로드
    from config.settings import (
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
        MARKET_OPEN_TIME, MARKET_CLOSE_TIME,
        DAILY_STRATEGY_RUN_TIME, PORTFOLIO_UPDATE_TIME,
        SMA_DAILY_PARAMS, RSI_MINUTE_PARAMS, 
        STOP_LOSS_PARAMS, INITIAL_CASH,
        LOG_LEVEL, LOG_FILE
    )    

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
        
        # 2. Backtest 인스턴스 생성
        initial_cash = 100_000_000  # 1천만원
        
        backtest_instance = Backtest(
            api_client=api_client,
            db_manager=db_manager,
            initial_cash=initial_cash,
            save_to_db=True
        )
        
        # 전략 인스턴스 생성
        # SMA 전략 설정 (최적화 결과 반영)
        from strategies.sma_daily import SMADaily
        daily_strategy = SMADaily(broker=backtest_instance.broker, data_store=backtest_instance.data_store, strategy_params=SMA_DAILY_PARAMS)

        # RSI 분봉 전략 설정
        from strategies.rsi_minute import RSIMinute
        minute_strategy = RSIMinute(broker=backtest_instance.broker, data_store=backtest_instance.data_store, strategy_params=RSI_MINUTE_PARAMS)        
        
        # 분봉전략: 매매신호 바로 주문 (최적화용)
        # from strategies.open_minute import OpenMinute
        # minute_strategy = OpenMinute(broker=backtest_instance.broker, data_store=backtest_instance.data_store, strategy_params=RSI_MINUTE_PARAMS)        
        
        # 일봉/분봉 전략 설정
        backtest_instance.set_strategies(daily_strategy=daily_strategy, minute_strategy=minute_strategy)
        # 손절매 파라미터 설정 (선택사항)
        backtest_instance.set_broker_stop_loss_params(STOP_LOSS_PARAMS)
        
        end_date = date(2025, 7, 10)
        start_date = end_date - timedelta(days=365)
        # 일봉 데이터 로드
        backtest_instance.load_stocks(start_date, end_date)

        # 5. 백테스트 실행
        try:
            backtest_instance.run(start_date, end_date)
        except KeyboardInterrupt:
            logger.info("사용자에 의해 시스템 종료 요청됨.")
        finally:
            backtest_instance.cleanup()
            logger.info("시스템 종료 완료.")

    except Exception as e:
        logger.error(f"Backtest 테스트 중 오류 발생: {e}", exc_info=True)