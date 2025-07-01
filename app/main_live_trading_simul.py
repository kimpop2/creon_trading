# main_live_trading_simul.py

import logging
import datetime
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

def run_simulation():
    logger.info("라이브 트레이딩 시뮬레이션 시작...")

    # --- 1. 시뮬레이션 초기 설정 ---
    INITIAL_CASH = 10_000_000  # 초기 투자 자본 (1천만원)
    SIMULATION_START_DATE = datetime.date(2024, 1, 2) # 시뮬레이션 시작일 (DB에 데이터가 있는 날짜로 설정)
    SIMULATION_END_DATE = datetime.date(2024, 1, 5) # 시뮬레이션 종료일 (테스트를 위해 짧게 설정)
    SIMULATED_FILL_DELAY_SECONDS = 3 # 주문 접수 후 체결까지 걸리는 시뮬레이션 시간 (초)
    REAL_TIME_FACTOR = 0.0 # 시뮬레이션 속도: 1분 -> REAL_TIME_FACTOR 초 (0.0은 지연 없음, 빠르게 실행)

    db_manager = DBManager()
    data_manager = BacktestManager() # BacktestManager가 DB에서 데이터를 가져오는 역할 수행
    broker = Broker(initial_cash=INITIAL_CASH, simulated_fill_delay_seconds=SIMULATED_FILL_DELAY_SECONDS)

    # --- 2. 전략 초기화 및 브로커/데이터 매니저 주입 ---
    daily_strategy_params = {
        'strategy_name': 'SMADaily',
        'short_sma_period': 5,
        'long_sma_period': 20,
        'volume_ma_period': 20,
        'num_top_stocks': 10 # 매수/매도 후보 상위 N개
    }
    daily_strategy = SMADaily(data_store=broker.data_store, strategy_params=daily_strategy_params, broker=broker)

    minute_strategy_params = {
        'strategy_name': 'RSIMinute',
        'minute_rsi_period': 14,
        'minute_rsi_oversold': 30,
        'minute_rsi_overbought': 70,
        'num_top_stocks': 5 # 분봉 전략이 고려할 상위 종목 수
    }
    minute_strategy = RSIMinute(data_store=broker.data_store, strategy_params=minute_strategy_params, broker=broker)

    # --- 3. 시뮬레이션 메인 루프 (일별 진행) ---
    current_date = SIMULATION_START_DATE
    while current_date <= SIMULATION_END_DATE:
        if current_date.weekday() >= 5: # 주말 건너뛰기 (토, 일)
            current_date += datetime.timedelta(days=1)
            continue

        logger.info(f"\n--- 시뮬레이션 날짜: {current_date.isoformat()} ---")

        # --- 장 시작 전 (데일리 전략 실행) ---
        # 1. 일봉 전략 실행: 당일 매매할 종목 선정 및 시그널 생성 (전일 종가 기준)
        # 이전 거래일 데이터를 기준으로 전략 실행
        # get_next_weekday 함수는 특정 날짜부터 시작하여 n번째 다음/이전 평일 날짜를 반환
        prev_trading_day = get_next_weekday(current_date - datetime.timedelta(days=1), -1) 
        logger.info(f"장 시작 전: {prev_trading_day.isoformat()} 기준 일봉 전략 실행 중...")
        
        # SMADaily.generate_signals는 current_date를 매개변수로 받음
        daily_signals = daily_strategy.run_daily_logic(prev_trading_day) 
        
        if daily_signals:
            logger.info(f"일봉 전략에서 {len(daily_signals)}개의 매매 시그널 생성됨: {list(daily_signals.keys())}")
            # 생성된 데일리 시그널을 분봉 전략에 전달
            minute_strategy.update_daily_signals(daily_signals)
        else:
            logger.info("일봉 전략에서 생성된 매매 시그널이 없습니다. (주말이거나, 데이터 부족 또는 전략 조건 미충족)")
            # 이 날은 매매할 종목이 없으므로 다음 날로 넘어감
            current_date += datetime.timedelta(days=1)
            continue 

        # --- 장 중 시뮬레이션 (분봉 단위 진행) ---
        # 장 시작 시각 (9:00 AM)부터 장 마감 시각 (3:30 PM)까지
        simulated_time = datetime.datetime.combine(current_date, datetime.time(9, 0, 0))
        market_close_time = datetime.datetime.combine(current_date, datetime.time(15, 30, 0))

        logger.info(f"장 중 시뮬레이션 시작: {simulated_time.isoformat()} 부터 {market_close_time.isoformat()} 까지")

        while simulated_time <= market_close_time:
            # 1. 현재 시뮬레이션 시간의 분봉 데이터 가져오기
            current_minute_prices: Dict[str, float] = {}
            for stock_code in daily_signals.keys(): # 일봉 전략에서 선택된 종목만 처리
                # RSIMinute 전략에 필요한 충분한 과거 분봉 데이터를 가져옴
                # minute_rsi_period 만큼의 데이터가 필요하므로, 그 이전 시점부터 데이터를 요청
                minute_df = data_manager.get_minute_data_for_strategy(
                    stock_code, 
                    simulated_time - datetime.timedelta(minutes=minute_strategy_params['minute_rsi_period'] + 5), 
                    simulated_time,
                    minute_strategy_params # RSI 지표 계산을 위해 파라미터 전달
                )
                
                if not minute_df.empty and simulated_time in minute_df.index:
                    current_minute_prices[stock_code] = minute_df.loc[simulated_time]['close']
                elif not minute_df.empty and not minute_df.index.empty:
                    # 데이터 누락 시 가장 가까운 유효한 데이터 사용 (백테스팅/시뮬레이션 편의상)
                    # 실제 라이브에서는 이 경우 현재가 정보가 없으므로 해당 종목은 거래 불가
                    closest_idx = minute_df.index.asof(simulated_time)
                    if closest_idx and closest_idx <= simulated_time:
                         current_minute_prices[stock_code] = minute_df.loc[closest_idx]['close']
                    else:
                        logger.debug(f"[{simulated_time.isoformat()}] 종목 {stock_code}의 현재 분봉 데이터 또는 유효한 인접 데이터 없음. 건너뜀.")
                else:
                    logger.debug(f"[{simulated_time.isoformat()}] 종목 {stock_code}의 분봉 데이터프레임이 비어있음. 건너뜀.")

            # 2. 분봉 전략 실행 (매매 주문 발생)
            # current_minute_prices에 현재 시각의 종목별 종가가 담겨있어야 함
            if current_minute_prices: # 현재가 정보가 있는 종목이 있을 경우에만 전략 실행
                minute_strategy.on_minute_data(simulated_time, current_minute_prices)

            # 3. 브로커의 미체결 주문 처리 (시뮬레이션 체결 감시 이벤트 루프)
            # simulate_fill_delay_seconds 설정에 따라 지연 체결된 주문들을 여기서 처리
            broker.process_simulated_time_events(simulated_time)
            
            # 4. 포트폴리오 손절매 검사 (선택 사항)
            broker.check_portfolio_stop_loss(simulated_time, current_minute_prices)

            # --- 시뮬레이션 시간 진행 ---
            simulated_time += datetime.timedelta(minutes=1)
            # 실제 시간 지연 (1분 시뮬레이션을 실제 시간 REAL_TIME_FACTOR 초로 단축)
            if REAL_TIME_FACTOR > 0:
                time_module.sleep(REAL_TIME_FACTOR) 

        # --- 장 마감 후 정리 ---
        daily_strategy.reset_daily_transactions() # 일별 전략 관련 캐시/상태 초기화
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