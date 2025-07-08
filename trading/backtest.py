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
from trading.backtest_report import BacktestReport # Reporter 타입 힌트를 위해 남겨둠
from api.creon_api import CreonAPIClient
from manager.backtest_manager import BacktestManager # BacktestManager 타입 힌트를 위해 남겨둠
from manager.db_manager import DBManager    
from util.strategies_util import *
from strategies.strategy import BaseStrategy
from strategies.sma_strategy import SMAStrategy

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
        self.data_store = {'daily': {}, 'minute': {}} # {stock_code: DataFrame}
        self.portfolio_values = [] # (datetime, value) 튜플 저장
        self.initial_cash = initial_cash
        self.save_to_db = save_to_db  # DB 저장 여부 저장
        
        self.strategy: BaseStrategy = None
        
        # 외부에서 주입받은 인스턴스를 사용
        self.manager = BacktestManager(self.api_client, self.db_manager) # initial_cash ???
        self.broker = Broker(self.initial_cash)
        self.report = BacktestReport(self.db_manager)
        
        # NEW: 포트폴리오 손절 체크 시간을 추적하기 위한 변수
        self.last_portfolio_check = None
        # 포트폴리오 손절 발생 시 당일 매매 중단 플래그
        self.portfolio_stop_flag = False

        # 일봉 업데이트 캐시 변수들 (성능 개선용)
        self._daily_update_cache = {}  # 캐시 데이터 확인 용 키 : {stock_code: {date: last_update_time}} 
        self._minute_data_cache = {}   # 캐시 분봉 데이터 {stock_code: {date: filtered_data}}

        logging.info(f"백테스터 초기화 완료. 초기 현금: {self.initial_cash:,.0f}원, DB저장: {self.save_to_db}")

    def set_strategies(self, strategy: BaseStrategy):
            self.strategy = strategy
            logging.info(f"전략 '{strategy.__class__.__name__}' 설정 완료.")
        
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
    
    def check_portfolio_stop_loss(self, current_dt: datetime, current_prices: Dict[str, float]) -> bool:
        """
        포트폴리오 전체 손절매 조건을 확인하고 실행합니다.
        시뮬레이션에서는 이 함수가 매 분 호출될 수 있습니다.
        """
        if self.broker.stop_loss_params and self.broker.stop_loss_params.get('portfolio_stop_loss_enabled', False):
            # 특정 시간 이후에만 손절매 검사
            if current_dt.time() >= datetime.time(self.broker.stop_loss_params.get('portfolio_stop_loss_start_hour', 14), 0, 0):
                losing_positions_count = 0
                for stock_code, position in self.positions.items():
                    if position['size'] > 0 and stock_code in current_prices:
                        loss_ratio = self._calculate_loss_ratio(current_prices[stock_code], position['avg_price'])
                        if loss_ratio >= self.broker.stop_loss_params['stop_loss_ratio']: # 손실률이 기준 이상인 경우
                            losing_positions_count += 1

                if losing_positions_count >= self.broker.stop_loss_params['max_losing_positions']:
                    logger.info(f'[{current_dt.isoformat()}] [포트폴리오 손절] 손실 종목 수: {losing_positions_count}개 (기준: {self.broker.stop_loss_params["max_losing_positions"]}개 이상)')
                    self._execute_portfolio_sellout(current_prices, current_dt)
                    return True
        return False

    def get_current_market_prices(self, stock_codes: List[str]) -> Dict[str, float]:
        """
        현재 시점의 시장 가격을 가져옵니다. 백테스트 시뮬레이션에서는 data_store에서 가져옵니다.
        """
        prices = {}
        for code in stock_codes:
            # 가장 최신 분봉의 종가를 현재 가격으로 간주
            if code in self.data_store['minute'] and self.data_store['minute'][code]:
                # 마지막 날짜의 마지막 분봉 데이터를 가져옴
                latest_date = max(self.data_store['minute'][code].keys())
                latest_minute_df = self.data_store['minute'][code][latest_date]
                if not latest_minute_df.empty:
                    prices[code] = latest_minute_df.iloc[-1]['close']
                else:
                    logging.warning(f"종목 {code}의 분봉 데이터에 유효한 가격이 없습니다.")
            else:
                logging.warning(f"종목 {code}의 분봉 데이터가 data_store에 없습니다. 가격을 0으로 설정합니다.")
                prices[code] = 0.0 # 데이터가 없을 경우 0으로 처리하거나 에러 처리
        return prices


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
        # =========================================                      
        # 영업일을 순회하며 SetupManager 실행
        # =========================================
        for current_date in trading_dates:  
            # 영업일이면 백테스트 시작
            logging.info(f"\n--- 현재 백테스트 날짜: {current_date.isoformat()} ---")

            # 새로운 날짜로 넘어갈 때 일봉 업데이트 캐시 초기화
            self._clear_daily_update_cache()

            # 전일 날짜 계산
            current_date_index = trading_dates.index(current_date)
            prev_date = trading_dates[current_date_index - 1]           

            logging.info(f"-------- {current_date.isoformat()} 매매 시작 --------")
            # 전략 실행 ===================================
            self.strategy.run_strategy_logic(prev_date)
            # self.strategy.signals 을 통해 전략 run_strategy_logic 과 매매 run_trading_logic 간 신호 전달
            for stock_code, signal_info in self.strategy.signals.items():
                if signal_info['signal'] in ['buy', 'sell', 'hold']:
                    self.strategy.signals[stock_code] = {
                        **signal_info,
                        'traded_today': False, # 초기화된 상태로 전달
                        'signal_date': prev_date 
                    }
            # END for 신호, 분봉내로 들어 갈 때는 ???????????? 수정 필요 ?????????????????????
            # 매수/매도 처리 유무 확인
            has_trading_signals = any(signal_info['signal'] in ['buy', 'sell'] for signal_info in self.strategy.signals.values())
            
            # 신호(매수/매도/보유)가 있는 종목들의 분봉 데이터를 모두 로드 vvvvvvvvvvvvvvvvvvvvvv
            stocks_to_trade = set() # 분봉 데이터가 필요한 종목들
            # 매수/매도 신호가 있는 종목들 추가
            for stock_code, signal_info in self.strategy.signals.items():
                if signal_info['signal'] in ['buy', 'sell']:
                    stocks_to_trade.add(stock_code)
            
            # 보유중인 종목 추가
            current_positions = set(self.broker.positions.keys()) #### 보유종목 구하기
            stocks_to_trade.update(current_positions)

            # 필요한 종목들의 분봉 데이터를 로드 
            for stock_code in stocks_to_trade:
                signal_info = self.strategy.signals.get(stock_code)
                prev_date = signal_info.get('signal_date', current_date) if signal_info else current_date
                if not prev_date:
                    prev_date = current_date # Ensure prev_date is set

                # cache_minute_ohlcv (api + DB) 를 사용하여 전일(여유)애서 당일 의 분봉 데이터 로드 
                minute_df = self.manager.cache_minute_ohlcv(
                    stock_code,
                    prev_date,      # 전일 영업일부터
                    current_date    # 당일까지
                )
                # 기존 백테스터와 동일하게 날짜별로 분봉 데이터 저장
                if not minute_df.empty:
                    if stock_code not in self.data_store['minute']:
                        self.data_store['minute'][stock_code] = {}
                    for date_key in [prev_date, current_date]:
                        date_data = minute_df[minute_df.index.normalize() == pd.Timestamp(date_key).normalize()]
                        if not date_data.empty:
                            self.data_store['minute'][stock_code][date_key] = date_data
                            logging.debug(f"{stock_code} 종목의 {date_key} 분봉 데이터 로드 완료. 데이터 수: {len(date_data)}행")
            
            # END for 분봉로드 ^^^^^^^^^^^^^^^^^^^^^

            ###################################
            # 오늘 장이 끝날때 까지 매분 반복 실행
            ###################################
            # 장 시작 시간부터 장 마감 시간까지 1분 단위로 반복하며 분봉 전략 실행
            open_time = time(9, 0) # 9시 정각
            market_open = datetime.combine(current_date, open_time)

            for i in range(391): # 9:00부터 15:30까지 (6시간 30분 = 390분 + 1분 (9:00분 포함))
                current_dt = market_open + timedelta(minutes=i)
                
                # 9:00에 시작하여 9:00 분봉은 9:01에 완성되므로 9:01부터 매매 로직 시작
                if current_dt.time() <= time(9, 0): # 9시 정각은 건너뛰고 9:01부터 시작
                    continue
                # 15:30 이후 시간은 건너뛰기
                if current_dt.time() > time(15, 30):
                    break
                
                # 매분 오늘의 일봉 데이터를 업데이트하는 함수 호출 (최적의 위치)
                self._update_daily_data_from_minute_bars(current_dt)

                # 포트폴리오 손절 체크 (9시, 15시 20분)
                if self._should_check_portfolio(current_dt):
                    self.check_portfolio_stop_loss(current_dt=current_dt, current_prices=self.get_current_market_prices(list(self.broker.positions.keys())))
                    if self.portfolio_stop_flag:
                        logging.warning(f"[{current_dt.isoformat()}] 포트폴리오 손절매 발동. 당일 추가 매매 중단.")
                        self.portfolio_stop_flag = True # 당일 매매 중단 플래그 설정
                        break # 당일 분봉 루프 종료

                # 당일 매매 중단 플래그가 설정되었으면 더 이상 매매하지 않음
                if self.portfolio_stop_flag:
                    break
                
                # 여기 이전은 수정하지 않는다. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



                # 매매 실행 vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                logging.debug(f"[{current_dt.isoformat()}] 분봉 매매 로직 실행 중...")
                # 매분마다 모든 보유 종목과 신호 종목에 대해 분봉 전략 실행
                for stock_code in stocks_to_trade:
                    if stock_code not in self.data_store['minute'] or current_date not in self.data_store['minute'][stock_code]:
                        logging.warning(f"[{current_dt.isoformat()}] {stock_code}의 {current_date} 분봉 데이터가 없어 매매 로직을 건너깁니다.")
                        continue
                    else:
                        logging.debug(f"[{current_dt.isoformat()}] {stock_code}의 현재 시간 이전 분봉 데이터가 없어 매매 로직을 건너깁니다.")

                    # 분봉 처리 : 전략을 구현한 전략 파일 내의 run_strategy_logic() 에서 발생한 시그널에 따라
                    # 실제 매매를 처리한다.
                    self.strategy.run_trading_logic(current_dt, stock_code)

                # END 매매 끝 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            # END for 분봉 루프 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            logging.info(f"-------- {current_date.isoformat()} 매매 종료 --------")


            # 4. 하루 종료 후 보고서 생성 및 DB 저장 (TradingManager를 통해)
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
            logging.info(f"[{current_date.isoformat()}] 장 마감 포트폴리오 가치: {daily_portfolio_value:,.0f}원, 현금: {self.broker.initial_cash:,.0f}원")

            # 일일 거래 기록 초기화 (Brokerage 내부)
            self.broker.reset_daily_transactions() # 현재 pass 상태, 아래 로직으로 채워야 함
            
            # 수정: 다음날을 위해 모든 신호 초기화
            # 1. 일봉 전략의 신호 완전 초기화
            self.strategy._reset_all_signals()  # 모든 신호를 완전히 삭제
            
            # 2. 백테스터의 신호 저장소 초기화 (다음날을 위해)
            self.last_portfolio_check = None # 다음 날을 위해 손절 체크 시간 초기화
            self.portfolio_stop_flag = False  # 새로운 날짜마다 플래그 초기화
            
            logging.debug(f"[{current_date.isoformat()}] 일일 신호 초기화 완료 - 다음날을 위해 모든 신호 저장소 비움")


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

        # 전략 이름과 파라미터 추출
        strategy_name = self.strategy.__class__.__name__ if self.strategy else "N/A"
        
        # 모든 저장 로직을 self.report에게 위임
        self.report.generate_and_save_report(
            start_date=start_date,
            end_date=end_date,
            initial_cash=self.initial_cash,
            portfolio_value_series=portfolio_value_series,
            transaction_log=self.broker.transaction_log,
            strategy_name=self.strategy.strategy_name,
            strategy_params=self.strategy.strategy_params
        )
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
        # 모든 종목 데이터 로딩: 하나의 리스트로 변환
        fetch_start = start_date - timedelta(days=30)
        stock_names = []
        for sector, stocks in sector_stocks.items():
            for stock_name, _ in stocks:
                stock_names.append(stock_name)

        all_target_stock_names = stock_names
        for name in all_target_stock_names:
            code = self.api_client.get_stock_code(name)
            if code:
                logging.info(f"'{name}' (코드: {code}) 종목 일봉 데이터 로딩 중... (기간: {fetch_start.strftime('%Y%m%d')} ~ {end_date.strftime('%Y%m%d')})")
                daily_df = self.manager.cache_daily_ohlcv(code, fetch_start, end_date)
                
                if daily_df.empty:
                    logging.warning(f"{name} ({code}) 종목의 일봉 데이터를 가져올 수 없습니다. 해당 종목을 건너뜁니다.")
                    continue
                logging.debug(f"{name} ({code}) 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
                self.add_daily_data(code, daily_df)
            else:
                logging.warning(f"'{name}' 종목의 코드를 찾을 수 없습니다. 해당 종목을 건너뜁니다.")

if __name__ == "__main__":
    """
    Backtest 클래스 테스트 실행 코드
    """
    from datetime import date, datetime
    from strategies.sma_strategy import SMAStrategy
    # 설정 파일 로드
    from config.settings import (
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
        INITIAL_CASH,
        MARKET_OPEN_TIME, MARKET_CLOSE_TIME,
        DAILY_STRATEGY_RUN_TIME, PORTFOLIO_UPDATE_TIME,
        SMA_PARAMS, STOP_LOSS_PARAMS,
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
        initial_cash = 10_000_000  # 1천만원
        
        backtest_system = Backtest(
            api_client=api_client,
            db_manager=db_manager,
            initial_cash=initial_cash,
            save_to_db=True
        )
        
        # 전략 인스턴스 생성
        # SMA 전략 설정 (최적화 결과 반영)
        from strategies.sma_strategy import SMAStrategy
        strategy_instance = SMAStrategy(broker=backtest_system.broker, 
                                        data_store=backtest_system.data_store, 
                                        strategy_params=SMA_PARAMS)
        backtest_system.set_strategies(strategy=strategy_instance)
        # 손절매 파라미터 설정 (선택사항)
        backtest_system.set_broker_stop_loss_params(STOP_LOSS_PARAMS)
        
        end_date = date(2025, 7, 7)
        start_date = end_date - timedelta(days=30)
        # 일봉 데이터 로드
        backtest_system.load_stocks(start_date, end_date)

        # 5. 백테스트 실행
        
        try:
            backtest_system.run(start_date, end_date)
        except KeyboardInterrupt:
            logger.info("사용자에 의해 시스템 종료 요청됨.")
        finally:
            backtest_system.cleanup()
            logger.info("시스템 종료 완료.")

    except Exception as e:
        logger.error(f"Backtest 테스트 중 오류 발생: {e}", exc_info=True)

       

