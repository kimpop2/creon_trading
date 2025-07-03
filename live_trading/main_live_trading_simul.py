# main_live_trading_simul.py

import logging
import datetime
from datetime import time
import time as time_module # time.sleep을 위해
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import os, sys
# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 필요한 클래스들을 임포트 (경로에 맞게 조정 필요)
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager # BacktestManager는 data_manager 역할
from trade_broker import Broker
from strategies.sma_daily import SMADaily
from strategies.rsi_minute import RSIMinute
from util.strategies_util import get_next_weekday

# --- 로거 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LiveSimulMain')
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

def run_simulation():
    logger.info("라이브 트레이딩 시뮬레이션 시작...")

    # --- 1. 시뮬레이션 초기 설정 ---
    INITIAL_CASH = 10_000_000  # 초기 투자 자본 (1천만원)
    SIMULATION_START_DATE = datetime.date(2025, 6, 15) # 시뮬레이션 시작일 (DB에 데이터가 있는 날짜로 설정)
    SIMULATION_END_DATE = datetime.date(2025, 7, 1) # 시뮬레이션 종료일 (테스트를 위해 짧게 설정)
    SIMULATED_FILL_DELAY_SECONDS = 3 # 주문 접수 후 체결까지 걸리는 시뮬레이션 시간 (초)
    REAL_TIME_FACTOR = 0.0 # 시뮬레이션 속도: 1분 -> REAL_TIME_FACTOR 초 (0.0은 지연 없음, 빠르게 실행)

    db_manager = DBManager()
    manager = BacktestManager() # BacktestManager가 DB에서 데이터를 가져오는 역할 수행
    broker = Broker(initial_cash=INITIAL_CASH, simulated_fill_delay_seconds=SIMULATED_FILL_DELAY_SECONDS)
    portfolio_values = []
    signals_for_minute_trading = {}
    data_store = {}
    current_day_signals = {}
    # --- 2. 전략 초기화 및 브로커/데이터 매니저 주입 ---
    daily_strategy_params = {
        'strategy_name': 'SMADaily',
        'short_sma_period': 5,
        'long_sma_period': 20,
        'volume_ma_period': 20,
        'num_top_stocks': 10 # 매수/매도 후보 상위 N개
    }
    daily_strategy = SMADaily(data_store=data_store, strategy_params=daily_strategy_params, broker=broker)

    minute_strategy_params = {
        'strategy_name': 'RSIMinute',
        'minute_rsi_period': 14,
        'minute_rsi_oversold': 30,
        'minute_rsi_overbought': 70,
        'num_top_stocks': 5 # 분봉 전략이 고려할 상위 종목 수
    }
    minute_strategy = RSIMinute(data_store=data_store, strategy_params=minute_strategy_params, broker=broker)

    # --- 3. 시뮬레이션 메인 루프 (일별 진행) ---
    current_date = SIMULATION_START_DATE
    # 시장 캘린더 데이터 로드
    market_calendar_df = db_manager.fetch_market_calendar(SIMULATION_START_DATE, SIMULATION_END_DATE)
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

        logger.info(f"\n--- 시뮬레이션 날짜: {current_date.isoformat()} ---")

        # --- 장 시작 전 (데일리 전략 실행) ---
        # 1. 일봉 전략 실행: 당일 매매할 종목 선정 및 시그널 생성 (전일 종가 기준)
        # 이전 거래일 데이터를 기준으로 전략 실행
        # get_next_weekday 함수는 특정 날짜부터 시작하여 n번째 다음/이전 평일 날짜를 반환
        # 시작일 이전의 초기 포트폴리오 가치 기록
        initial_portfolio_value = broker.get_portfolio_value({}) # 초기 현금만 반영
        portfolio_values.append((current_date - datetime.timedelta(days=1), initial_portfolio_value))
        
        # 전일 날짜 계산
        current_date_index = trading_dates.index(current_date)
        prev_trading_day = None
        if current_date_index == 0 :
            continue
        else:
            prev_trading_day = trading_dates[current_date_index - 1] 
     
        logger.info(f"장 시작 전: {prev_trading_day.isoformat()} 기준 일봉 전략 실행 중...")
        
        fetch_date = current_date - datetime.timedelta(days=60)
        data_store['daily'] = {} # 보유종목과 유니버스 종목을 다시 담기 위해 초기화
        # 일봉 데이터를 가져오는 대상은 1. 보유종목 2. 유니버스 종목이다. 
        # OpenMinute는 오늘 일봉이 있어야, 분봉 가상화 처리를 하므로 오늘 일봉까지 전달한다.
        # 일봉전략은 지표계산을 할 때 iloc[-2] 처리 등으로 오늘 일봉을 지표에 반영하지 않도록 해야 한다.
        # 보유종목 일봉 데이터 설정
        #self.positions = {}  # {stock_code: {'size': int, 'avg_price': float, 'entry_date': datetime.date, 'highest_price': float}}
        for stock_code, position_info in broker.positions.items(): # .items()를 사용하여 키와 값을 모두 가져옵니다.
            if position_info['size'] > 0: # position_info['size']로 직접 접근합니다.
                daily_df = manager.cache_daily_ohlcv(stock_code, fetch_date, current_date)
                data_store['daily'][stock_code] = daily_df
                # data_store['daily']가 초기화되지 않았을 가능성을 대비하여 확인 후 할당합니다.
                # if 'daily' not in data_store:
                #     data_store['daily'] = {}
                #     data_store['daily'][stock_code] = daily_df

        # 하루전 선정된 유니버스 종목들을 가져온다.  
        stocks = db_manager.fetch_daily_theme_stock(current_date - datetime.timedelta(days=7), prev_trading_day)
        #print(stocks)
        for i, (stock_code, stock_name) in enumerate(stocks):
            daily_df = manager.cache_daily_ohlcv(stock_code, fetch_date, current_date)
            data_store['daily'][stock_code] = daily_df

        # 2. 전일 일봉 데이터를 기반으로 '오늘 실행할' 신호를 생성합니다.
        if daily_strategy:
            
            # 직전영업일이 있다. (지표계산을 위해 백테스트 시작일 이전 1달치 일봉데이터도 저장되어 있음)
            if prev_trading_day:
                ##############################
                # 전일 데이터로 일봉 전략 실행 - 오늘은 아직 장이 시작하지 않았으므로(전일 종가까지 데이터로)
                daily_strategy.run_daily_logic(prev_trading_day)
                # --- [핵심 변경 사항 시작] ---
                # 1. 일봉 전략에서 생성된 신호(self.signals)를 Trader 내부에 저장
                signals_for_minute_trading = daily_strategy.signals
                logging.debug(f"[{current_date.isoformat()}] Trader에 일봉 전략 신호 ({len(signals_for_minute_trading)}개) 저장.")

                # 2. 저장된 신호를 분봉 전략으로 전달
                if minute_strategy:
                    minute_strategy.update_signals(signals_for_minute_trading)
                    logging.info(f"[{current_date.isoformat()}] 일봉 전략 신호 분봉 전략으로 전달 완료.")
                # --- [핵심 변경 사항 끝] ---
                # self.daily_strategy.signals 에 전략 실행결과가 보관 됨
                ##############################

                # 생성된 신호 중 'buy', 'sell', 'hold' 신호를 current_day_signals에 저장합니다.
                # 종목별로 일봉전략 결과로 신호를 생성
                for stock_code, signal_info in daily_strategy.signals.items():
                    if signal_info['signal'] in ['buy', 'sell', 'hold']:
                        # 분봉에 전달해 줄 일봉전략 처리 결과를 current_day_signals 에 저장  
                        current_day_signals[stock_code] = {
                            **signal_info,
                            'traded_today': False, # 초기화된 상태로 전달
                            # 신호발생일은 전영업일: 신호는 전영업일 종가로 생성 (전영업일을 구하는 데 사용해도 됨)
                            'signal_date': prev_trading_day 
                        }

                logging.debug(f"[{current_date.isoformat()}] 전일({prev_trading_day.isoformat()}) 일봉 전략 실행 완료: {len(current_day_signals)}개의 당일 매매 신호 생성.")

    
                if daily_strategy.signals:
                    logger.info(f"일봉 전략에서 {len(daily_strategy.signals)}개의 매매 시그널 생성됨: {list(daily_strategy.signals.keys())}")
                else:
                    logger.info("일봉 전략에서 생성된 매매 시그널이 없습니다. (주말이거나, 데이터 부족 또는 전략 조건 미충족)")
                    continue 

        # --- 장 중 시뮬레이션 (분봉 단위 진행) ---
        # 장 시작 시각 (9:00 AM)부터 장 마감 시각 (3:30 PM)까지
        simulated_time = datetime.datetime.combine(current_date, datetime.time(9, 0, 0))
        market_close_time = datetime.datetime.combine(current_date, datetime.time(15, 30, 0))

        logger.info(f"장 중 시뮬레이션 시작: {simulated_time.isoformat()} 부터 {market_close_time.isoformat()} 까지")
        # 3. 오늘 분봉 매매 로직을 실행합니다.
        # 분봉전략명이 OpenMinute 인지 확인 해서 아래에서 다르게 처리한다. 
        if minute_strategy:
            data_store['minute'] = {}
            # OpenMinute는 최적화 전용 분봉전략으로, 처리속도를 빠르게 하기위해 분봉데이터를 사용하지 않고, 분봉 가상화 처리
            is_open_minute_strategy = hasattr(minute_strategy, 'strategy_name') and minute_strategy.strategy_name == "OpenMinute"
            
            # OpenMinute 전략이면 매일 분봉 가상화 캐시 초기화
            if is_open_minute_strategy and hasattr(minute_strategy, 'reset_virtual_range_cache'):
                minute_strategy.reset_virtual_range_cache()
            # 오늘 분봉은 일봉전략 처리결과를 signals_for_minute_trading 에 담아 사용
            signals_for_minute_trading = current_day_signals.copy()
            
            # 일봉전략 처리 결과에
            # 매수/매도 처리 유무, 손절 처리 유무 확인
            has_trading_signals = any(signal_info['signal'] in ['buy', 'sell'] for signal_info in signals_for_minute_trading.values())
            has_stop_loss = broker.stop_loss_params is not None # 손절처리 유무
            
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
                    trade_time = datetime.datetime.combine(current_date, datetime.time(9, 1))
                    
                    # OpenMinute 전략에 신호 업데이트 (target_quantity 정보 포함)
                    # 신호 전달 브릿지 로그 추가
                    num_signals = len(signals_for_minute_trading)
                    num_buy = sum(1 for s in signals_for_minute_trading.values() if s.get('signal') == 'buy')
                    num_sell = sum(1 for s in signals_for_minute_trading.values() if s.get('signal') == 'sell')
                    num_hold = sum(1 for s in signals_for_minute_trading.values() if s.get('signal') == 'hold')
                    logging.info(f"[신호전달] 분봉전략에 신호 전달: 총 {num_signals}개 (매수: {num_buy}, 매도: {num_sell}, 홀딩: {num_hold})")
                    minute_strategy.update_signals(signals_for_minute_trading)
                    
                    # 9:01에 매매 실행
                    stocks_to_trade = set()
                    # 매수/매도 신호가 있는 종목들 추가
                    for stock_code, signal_info in signals_for_minute_trading.items():
                        if signal_info.get('signal') in ['buy', 'sell']:
                            stocks_to_trade.add(stock_code)
                    
                    # OpenMinute 전략은 하루 한번 매매로 분봉매매를 시뮬레이션 한다.
                    # 그러므로 손절은 종가로만 처리한다. 이렇게 하지 않으면 과최적화 된다.
                    # 시가 손절도 있지만, 주가의 우상향 성질을 이용해서 종가가 시가 보다 나을 듯 하다.
                    # 포트폴리오 손절은 없음
                    for stock_code in stocks_to_trade:

                        ###################################
                        # 분봉 전략 매매 실행
                        minute_strategy.run_minute_logic(trade_time, stock_code)
                        ###################################
                
                # 일반 분봉 전략 실행
                else:
                    # 3-1. 먼저 당일 실행할 신호(매수/매도/보유)가 있는 종목들의 분봉 데이터를 모두 로드
                    stocks_to_load = set()  # 분봉 데이터가 필요한 종목들

                    # 매수/매도 신호가 있는 종목들 추가
                    for stock_code, signal_info in signals_for_minute_trading.items():
                        if signal_info['signal'] in ['buy', 'sell']:
                            stocks_to_load.add(stock_code)
                    
                    current_positions = set(broker.positions.keys()) #### 보유종목 구하기
                    stocks_to_load.update(current_positions)                        

                    # 필요한 종목들의 분봉 데이터를 로드
                    for stock_code in stocks_to_load:
                        signal_info = current_day_signals.get(stock_code)
                        prev_trading_day = signal_info.get('signal_date', current_date) if signal_info else current_date

                        if prev_trading_day:
                            # TraderManager를 사용하여 분봉 데이터 로드 (전일~당일)
                            minute_df = manager.cache_minute_ohlcv(
                                stock_code,
                                prev_trading_day,  # 전일 영업일부터
                                current_date       # 당일까지
                            )

                            # 기존 백테스터와 동일하게 날짜별로 분봉 데이터 저장
                            if not minute_df.empty:
                                for date in [prev_trading_day, current_date]:
                                    date_data = minute_df[minute_df.index.normalize() == pd.Timestamp(date).normalize()]
                                    if not date_data.empty:
                                        if 'stock_code' not in data_store['minute']:
                                            data_store['minute'][stock_code] = {}
                                        data_store['minute'][stock_code][date] = date_data
                                        logging.debug(f"{stock_code} 종목의 {date} 분봉 데이터 로드 완료. 데이터 수: {len(date_data)}행")
                    # for 분봉로드 끝

                    # 3-2. 모든 시그널을 분봉 전략에 한 번에 업데이트
                    # 로그출력을 위한 처리
                    num_signals = len(signals_for_minute_trading)
                    num_buy = sum(1 for s in signals_for_minute_trading.values() if s.get('signal') == 'buy')
                    num_sell = sum(1 for s in signals_for_minute_trading.values() if s.get('signal') == 'sell')
                    num_hold = sum(1 for s in signals_for_minute_trading.values() if s.get('signal') == 'hold')
                    logging.info(f"[신호 전달] 분봉전략에 총 {num_signals}개 (매수: {num_buy}, 매도: {num_sell}, 홀딩: {num_hold})")
                    # 분봉전략 내의 signals 변수에 전달
                    minute_strategy.update_signals(signals_for_minute_trading)
                    logging.debug(f"[{current_date.isoformat()}] 분봉 전략에 {len(signals_for_minute_trading)}개의 시그널 업데이트 완료.")
                    
                    # 3-3. 분봉 매매 대상은 신호(매수/매도) 손절 파라미터 설정시 보유 종목
                    stocks_to_trade = set() # 매매대상
                    
                    # 매수/매도 신호가 있는 종목들 추가
                    for stock_code, signal_info in signals_for_minute_trading.items():
                        if signal_info['signal'] in ['buy', 'sell']:
                            stocks_to_trade.add(stock_code)
                    # 손절매 기능이 있다면, 보유 중인 종목들 추가 (손절매 체크용)
                    if has_stop_loss:
                        current_positions = set(broker.positions.keys())
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
                            if stock_code not in data_store['minute'] or \
                                current_date not in data_store['minute'][stock_code] or \
                                trade_time not in data_store['minute'][stock_code][current_date].index:
                                logging.debug(f"[{trade_time.isoformat()}] {stock_code}: 해당 시간의 분봉 데이터 없음. 스킵.")
                                continue
                            
                            # 분봉전략 실행
                            minute_strategy.run_minute_logic(trade_time, stock_code)

                        # ########################
                        # 포토폴리오 손절
                        # ------------------------
                        # 포트폴리오 손절을 위한 9:00, 15:20 시간 체크, 분봉마다하는 것이 정확하겠지만 속도상 
                        if broker.stop_loss_params is not None and _should_check_portfolio(trade_time):
                            
                            current_prices = {}
                            for code in list(broker.positions.keys()):
                                # 캐시된 가격 사용
                                if code in minute_strategy.last_prices:
                                    current_prices[code] = minute_strategy.last_prices[code]
                                else:
                                    price_data = minute_strategy._get_bar_at_time('minute', code, trade_time)
                                    if price_data is not None:
                                        current_prices[code] = price_data['close']
                                        minute_strategy.last_prices[code] = price_data['close']
                        
                                # 현재 가격이 없는 종목은 제외
                                current_prices = {k: v for k, v in current_prices.items() if not np.isnan(v)}                                
                            
                            # 포트폴리오 손절은 Broker에 위임처리
                            if broker.check_and_execute_portfolio_stop_loss(current_prices, trade_time):
                                # 매도처리
                                for code in list(minute_strategy.signals.keys()):
                                    # 매도된 것 신호 정리
                                    if code in broker.positions and broker.positions[code]['size'] == 0: # 매도 후 == 수량 0
                                        minute_strategy.reset_signal(stock_code)

                                logging.info(f"[{trade_time.isoformat()}] 포트폴리오 손절매 실행 완료. 오늘의 매매 종료.")
                                #self.portfolio_stop_flag = True 불필요 break
                                break # 분봉 루프를 종료, 일일 포트폴리오 처리 후 다음 "영업일"로 넘어감
                        # 포트폴리오 손절 ------------------------
                        
                        # 다음 분으로 이동
                        trade_time += datetime.timedelta(minutes=1)
                        #time_module.sleep(REAL_TIME_FACTOR) 
                    # while -------------------------------------

            # end if 분봉매매


        # --- 장 마감 후 정리 ---
        daily_strategy._reset_all_signals() # 일별 전략 관련 캐시/상태 초기화
        broker.reset_daily_transactions() # 브로커의 일별 상태 초기화 (필요시)

        current_date += datetime.timedelta(days=1)

    # --- 4. 시뮬레이션 최종 결과 출력 ---
    logger.info("\n--- 시뮬레이션 최종 결과 ---")
    logger.info(f"시작일: {SIMULATION_START_DATE.isoformat()}")
    logger.info(f"종료일: {SIMULATION_END_DATE.isoformat()}")
    logger.info(f"초기 현금: {INITIAL_CASH:,.0f} KRW")
    logger.info(f"최종 현금: {broker.get_current_cash():,.0f} KRW")
    logger.info(f"최종 보유 종목: {broker.get_current_positions()}")
    logger.info(f"총 거래 내역 수: {len(broker.get_transaction_log())}")
    
    if broker.get_transaction_log():
        logger.info("--- 전체 거래 내역 ---")
        for log_entry in broker.get_transaction_log():
            logger.info(f"  - {log_entry}")
    else:
        logger.info("발생한 거래 내역이 없습니다.")

if __name__ == "__main__":
    run_simulation()