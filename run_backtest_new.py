"""
파일명: run_daily_minute.py
설명: 듀얼 모멘텀 + RSI 전략 백테스팅 (전략 패턴 적용)
작성일: 2024-03-19 (업데이트: 2025-06-09)
"""

import datetime
import logging
import pandas as pd
import numpy as np
import time
import sys
import os 

# 현재 스크립트의 경로를 sys.path에 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.creon_api import CreonAPIClient
from backtest.backtester import Backtester
from backtest.broker import Broker
from backtest.reporter import Reporter
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.rsi_minute import RSIMinute
from strategies.temp_daily import TempletDaily
from strategies.sma_daily import SMADaily
from manager.data_manager import DataManager
from manager.db_manager import DBManager
from selector.stock_selector import StockSelector
from strategies.open_minute import OpenMinute
from config.sector_config import sector_stocks  # 공통 설정 파일에서 import

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[ 
                        logging.FileHandler("backtest_run.log", encoding='utf-8'), 
                        logging.StreamHandler(sys.stdout) 
                    ])

if __name__ == '__main__':
    logging.info("백테스트를 실행합니다.")

    # 백테스트 기간 설정
    # 하락장 (최적화 기간과 동일)
    # backtest_start_date     = datetime.datetime(2024, 12, 1, 9, 0, 0).date()
    # backtest_end_date       = datetime.datetime(2025, 4, 1, 3, 30, 0).date()
    
    # 추세전환
    # backtest_start_date     = datetime.datetime(2025, 2, 1, 9, 0, 0).date()
    # backtest_end_date       = datetime.datetime(2025, 5, 1, 3, 30, 0).date()
    
    # 상승장
    backtest_start_date     = datetime.datetime(2025, 5, 1, 9, 0, 0).date()
    backtest_end_date       = datetime.datetime(2025, 6, 15, 3, 30, 0).date()

    # 일봉 데이터 가져오기 시작일을 백테스트 시작일 한 달 전으로 자동 설정
    daily_data_fetch_start = (backtest_start_date - datetime.timedelta(days=30)).replace(day=1)

    creon_api = CreonAPIClient()
    if not creon_api.connected:
        logging.error("Creon API에 연결할 수 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    # 핵심 컴포넌트 초기화
    data_manager = DataManager()
    db_manager = DBManager() # DBManager 인스턴스 생성
    reporter = Reporter(db_manager=db_manager) # Reporter 초기화 시 db_manager 전달
    stock_selector = StockSelector(data_manager=data_manager, api_client=creon_api, sector_stocks_config=sector_stocks)
    
    # 백테스터 초기화 - DataManager, Reporter, StockSelector 인스턴스 주입
    backtester_instance = Backtester(
        data_manager=data_manager, 
        api_client=creon_api, 
        reporter=reporter, 
        stock_selector=stock_selector,
        initial_cash=10_000_000
    )


    # 듀얼 모멘텀 전략 설정 (DualMomentumDaily 인스턴스 생성 및 Backtester에 주입)
    dual_daily_strategy = DualMomentumDaily(
        data_store=backtester_instance.data_store,
        strategy_params={
            'momentum_period': 15,         # 모멘텀 계산 기간 (거래일)
            'rebalance_weekday': 1,        # 리밸런싱 요일 (0: 월요일, 4: 금요일)
            'num_top_stocks': 7,           # 상위 N종목 선택
            'safe_asset_code': 'A439870',  # 안전자산 코드 (국고채 ETF)
        },
        broker=backtester_instance.broker
    )

    temp_daily_strategy = TempletDaily(
        data_store=backtester_instance.data_store,
        strategy_params={
            'momentum_period': 15,         # 듀얼 모멘텀처럼 기간 설정이 필요하다면 추가
            'num_top_stocks': 7,           # 듀얼 모멘텀처럼 상위 N종목 설정
            'safe_asset_code': 'A439870', # 안전자산 코드
        },
        broker=backtester_instance.broker 
    )
    
    # SMA 일봉 전략 설정 (최적화 결과 반영)
    sma_daily_strategy = SMADaily(
        data_store=backtester_instance.data_store,
        strategy_params={
            'short_sma_period': 6,          # 최적화 결과: 6일
            'long_sma_period': 15,          # 최적화 결과: 15일
            'volume_ma_period': 8,          # 최적화 결과: 8일
            'num_top_stocks': 7,            # 최적화 결과: 7개
            'safe_asset_code': 'A439870',   # 안전자산 코드
        },
        broker=backtester_instance.broker
    )

    # RSI 분봉 전략 설정 (실제 RSI 계산 및 매매)
    rsi_minute_strategy = RSIMinute(
        data_store=backtester_instance.data_store,
        strategy_params={
            'minute_rsi_period': 52,        # 최적화 결과: 52분
            'minute_rsi_oversold': 34,      # 최적화 결과: 34
            'minute_rsi_overbought': 74,    # 최적화 결과: 74
            'num_top_stocks': 7,            # 일봉 전략과 동일한 값으로 설정
        },
        broker=backtester_instance.broker
    )

    # 전략 설정 (SMA 일봉 + RSI 분봉 전략 사용)
    #backtester_instance.set_strategies(daily_strategy=dual_daily_strategy, minute_strategy=rsi_minute_strategy)
    #backtester_instance.set_strategies(daily_strategy=temp_daily_strategy, minute_strategy=rsi_minute_strategy)
    backtester_instance.set_strategies(daily_strategy=sma_daily_strategy, minute_strategy=rsi_minute_strategy)
    
    # Broker에 손절매 파라미터 설정 (최적화 결과 반영)
    stop_loss_params = {
        'stop_loss_ratio': -3.7,         # 최적화 결과: -3.7%
        'trailing_stop_ratio': -4.1,     # 최적화 결과: -4.1%
        'portfolio_stop_loss': -3.7,     # 최적화 결과: -3.7%
        'early_stop_loss': -3.7,         # 최적화 결과: -3.7%
        'max_losing_positions': 5,       # 최적화 결과: 5개
    }
    #stop_loss_params = None #손절하지 않기
    backtester_instance.set_broker_stop_loss_params(stop_loss_params)
    
    data_manager = DataManager()
    # 공통 설정 파일에서 모든 종목 이름 가져오기
    from config.sector_config import get_all_stock_names
    stock_names = get_all_stock_names()

    # 종목 코드 확인 및 일봉 데이터 로딩
    # 안전자산 코드도 미리 추가
    safe_asset_code = dual_daily_strategy.strategy_params['safe_asset_code'] # 듀얼 모멘텀 전략의 안전자산 코드 사용

    logging.info(f"'안전자산' (코드: {safe_asset_code}) 안전자산 일봉 데이터 로딩 중... (기간: {daily_data_fetch_start.strftime('%Y%m%d')} ~ {backtest_end_date.strftime('%Y%m%d')})")
    daily_df = data_manager.cache_daily_ohlcv(safe_asset_code, daily_data_fetch_start, backtest_end_date)
    backtester_instance.add_daily_data(safe_asset_code, daily_df)
    if daily_df.empty:
        logging.warning(f"'안전자산' (코드: {safe_asset_code}) 종목의 일봉 데이터를 가져올 수 없습니다. 종료합니다.")
        exit(1)
    logging.debug(f"'안전자산' (코드: {safe_asset_code}) 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")

    # 모든 종목 데이터 로딩
    all_target_stock_names = stock_names
    for name in all_target_stock_names:
        code = creon_api.get_stock_code(name)
        if code:
            logging.info(f"'{name}' (코드: {code}) 종목 일봉 데이터 로딩 중... (기간: {daily_data_fetch_start.strftime('%Y%m%d')} ~ {backtest_end_date.strftime('%Y%m%d')})")
            daily_df = data_manager.cache_daily_ohlcv(code, daily_data_fetch_start, backtest_end_date)
            
            if daily_df.empty:
                logging.warning(f"{name} ({code}) 종목의 일봉 데이터를 가져올 수 없습니다. 해당 종목을 건너뜁니다.")
                continue
            logging.debug(f"{name} ({code}) 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
            backtester_instance.add_daily_data(code, daily_df)
        else:
            logging.warning(f"'{name}' 종목의 코드를 찾을 수 없습니다. 해당 종목을 건너뜁니다.")

    if not backtester_instance.data_store['daily']:
        logging.error("백테스트를 위한 유효한 일봉 데이터가 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)
            
    # 백테스트 실행
    portfolio_values, metrics = backtester_instance.run(backtest_start_date, backtest_end_date)