"""
파일명: run_backtest_new.py
설명: 삼중창 전략 백테스팅 (최적화된 파라미터 적용)
작성일: 2024-03-19 (업데이트: 2025-06-18)
"""

import datetime
import logging
import pandas as pd
import numpy as np
import time
import sys
import os 
import codecs

# Windows 환경에서 한글 출력을 위한 콘솔 인코딩 설정
if sys.platform.startswith('win'):
    import locale
    # 콘솔 출력 인코딩을 UTF-8로 설정
    sys.stdout.reconfigure(encoding='utf-8')

# 현재 스크립트의 경로를 sys.path에 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.creon_api import CreonAPIClient
from trader.backtester import Backtester
from trader.broker import Broker
from trader.reporter import Reporter
from strategies.triple_screen_daily import TripleScreenDaily
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.rsi_minute import RSIMinute
from strategies.temp_daily import TempletDaily
from strategies.sma_daily import SMADaily
from strategies.open_minute import OpenMinute
from manager.data_manager import DataManager
from manager.db_manager import DBManager
from selector.stock_selector import StockSelector
from config.sector_config import sector_stocks  # 공통 설정 파일에서 import

# --- 로깅 설정 ---
# UTF-8 인코딩으로 콘솔 출력을 위한 StreamHandler 생성
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)

# 파일 핸들러 생성
file_handler = logging.FileHandler("backtest_run.log", encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

# 로거 설정
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)



if __name__ == '__main__':
    logging.info("삼중창 전략 백테스트를 실행합니다.")

    # 백테스트 기간 설정 (최적화 기간과 동일)
    backtest_start_date     = datetime.datetime(2025, 3, 1, 9, 0, 0).date()
    backtest_end_date       = datetime.datetime(2025, 6, 20, 3, 30, 0).date()

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

    # 삼중창 전략 설정 (기본 설정 + 거래비용 고려)
    triple_screen_daily_strategy = TripleScreenDaily(
        data_store=backtester_instance.data_store,
        strategy_params={
            'trend_ma_period': 20,          # 유지
            'momentum_rsi_period': 7,      # 유지
            'momentum_rsi_oversold': 35,    # 30 → 35 (매수 조건 더 보수적)
            'momentum_rsi_overbought': 65,  # 70 → 65 (매도 조건 더 보수적)
            'volume_ma_period': 7,         # 유지
            'num_top_stocks': 7,            # 5 → 3 (집중 투자로 승률 향상)
            'safe_asset_code': 'A439870',   # 안전자산 코드 (국고채 ETF)
            'min_trend_strength': 0.02,     # 기본값: 0.02 (2% 추세)
        },
        broker=backtester_instance.broker
    )

    # 듀얼 모멘텀 전략 설정 (DualMomentumDaily 인스턴스 생성 및 Backtester에 주입)
    dual_daily_strategy = DualMomentumDaily(
        data_store=backtester_instance.data_store,
        strategy_params={
            'momentum_period': 15,         #  15일
            'rebalance_weekday': 0,        #  월요일 (0)
            'num_top_stocks': 7,           #  5개
            'safe_asset_code': 'A439870',  # 안전자산 코드 (국고채 ETF)
        },
        broker=backtester_instance.broker
    )

    temp_daily_strategy = TempletDaily(
        data_store=backtester_instance.data_store,
        strategy_params={
            'momentum_period': 15,         # 듀얼 모멘텀처럼 기간 설정이 필요하다면 추가
            'num_top_stocks': 5,           #  5개
            'safe_asset_code': 'A439870', # 안전자산 코드
        },
        broker=backtester_instance.broker 
    )
    
    # SMA 일봉 전략 설정 (최적화 결과 반영)
    sma_daily_strategy = SMADaily(
        data_store=backtester_instance.data_store,
        strategy_params={
            'short_sma_period': 4,          #  4일
            'long_sma_period': 15,          #  10일
            'volume_ma_period': 6,          #  6일
            'num_top_stocks': 5,            #  5개
            'safe_asset_code': 'A439870',   # 안전자산 코드
        },
        broker=backtester_instance.broker
    )

    # OpenMinute 분봉 전략 설정
    open_minute_strategy = OpenMinute(
        data_store=backtester_instance.data_store,
        strategy_params={
            'minute_rsi_period': 52,        #  52분
            'minute_rsi_oversold': 34,      # 과매도 
            'minute_rsi_overbought': 70,    # 과매수
            'num_top_stocks': 7,            # 일봉 전략과 동일한 값으로 설정
        },
        broker=backtester_instance.broker
    )

    # RSI 분봉 전략 설정 (실제 RSI 계산 및 매매)
    rsi_minute_strategy = RSIMinute(
        data_store=backtester_instance.data_store,
        strategy_params={
            'minute_rsi_period': 52,        #  52분
            'minute_rsi_oversold': 34,      # 과매도 
            'minute_rsi_overbought': 70,    # 과매수
            'num_top_stocks': 7,            # 일봉 전략과 동일한 값으로 설정
        },
        broker=backtester_instance.broker
    )

    # 전략 설정 (삼중창 일봉 + RSI 분봉 전략 사용)
    # 전환 14.91
    #backtester_instance.set_strategies(daily_strategy=triple_screen_daily_strategy, minute_strategy=rsi_minute_strategy)
    # 전환 57.51
    #backtester_instance.set_strategies(daily_strategy=dual_daily_strategy, minute_strategy=rsi_minute_strategy)
    # 전환 84.41%
    #backtester_instance.set_strategies(daily_strategy=temp_daily_strategy, minute_strategy=rsi_minute_strategy)
    # 전환 -4.75 
    backtester_instance.set_strategies(daily_strategy=sma_daily_strategy, minute_strategy=rsi_minute_strategy)
    # 전환 83.11
    #backtester_instance.set_strategies(daily_strategy=dual_daily_strategy, minute_strategy=open_minute_strategy)
    
    # Broker에 손절매 파라미터 설정 (기본 설정)
    stop_loss_params = {
        'stop_loss_ratio': -2.5,         # 기본값: -5.0% (충분한 여유)
        'trailing_stop_ratio': -2.5,     # 기본값: -3.0% (표준 트레일링)
        'portfolio_stop_loss': -2.0,     # 기본값: -5.0% (포트폴리오 손절)
        'early_stop_loss': -2.5,         # 기본값: -5.0% (조기 손절)
        'max_losing_positions': 2,       # 기본값: 3개 (적당한 손실 허용)
    }
    backtester_instance.set_broker_stop_loss_params(stop_loss_params)
    
    data_manager = DataManager()
    # 공통 설정 파일에서 모든 종목 이름 가져오기
    from config.sector_config import get_all_stock_names
    stock_names = get_all_stock_names()

    # 종목 코드 확인 및 일봉 데이터 로딩
    # 안전자산 코드도 미리 추가
    safe_asset_code = triple_screen_daily_strategy.strategy_params['safe_asset_code'] # 삼중창 전략의 안전자산 코드 사용

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