"""
파일명: run_trade.py
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
# if sys.platform.startswith('win'):
#     import locale
#     # 콘솔 출력 인코딩을 UTF-8로 설정
#     sys.stdout.reconfigure(encoding='utf-8')

# 현재 스크립트의 경로를 sys.path에 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.creon_api import CreonAPIClient
from trade.trader import Trader
# from trade.brokerage import Brokerage trader 가 생성시킴
from trade.trader_report import TraderReport
from manager.trader_manager import TraderManager
from manager.backtest_manager import BacktestManager
from manager.db_manager import DBManager
# 전략파일 임포트
from strategies.triple_screen_daily import TripleScreenDaily
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.temp_daily import TempletDaily
from strategies.sma_daily import SMADaily
from strategies.breakout_daily import BreakoutDaily
from strategies.breakout_minute import BreakoutMinute

from strategies.rsi_minute import RSIMinute
from strategies.open_minute import OpenMinute

# # --- 로깅 설정 ---
# # UTF-8 인코딩으로 콘솔 출력을 위한 StreamHandler 생성
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setLevel(logging.INFO)
# console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# console_handler.setFormatter(console_formatter)

# # 파일 핸들러 생성
# file_handler = logging.FileHandler("trader_run.log", encoding='utf-8')
# file_handler.setLevel(logging.DEBUG)
# file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# file_handler.setFormatter(file_formatter)

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/trader_run.log", encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.info("적응형 전략 자동매매를 실행합니다.")

    # 자동매매 기간 설정 (최적화 기간과 동일)
    trader_start_date     = datetime.datetime(2025, 6, 1, 9, 0, 0).date()
    trader_end_date       = datetime.datetime(2025, 7, 1, 3, 30, 0).date()

    # 일봉 데이터 가져오기 시작일을 자동매매 시작일 한 달 전으로 자동 설정
    trader_fetch_start = (trader_start_date - datetime.timedelta(days=30)).replace(day=1)

    creon_api = CreonAPIClient()
    if not creon_api.connected:
        logging.error("Creon API에 연결할 수 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    # 핵심 컴포넌트 초기화
    trader_manager = TraderManager()
    backtest_manager = BacktestManager() # StockSelector 생성자
    
    db_manager = DBManager() # DBManager 인스턴스 생성
    trader_report = TraderReport(db_manager=db_manager) # Report 초기화 시 db_manager 전달
    
    # 백테스터 초기화 - TraderManager, Report, StockSelector 인스턴스 주입
    trader_instance = Trader(
        manager=trader_manager, 
        api_client=creon_api, 
        report=trader_report, 
        db_manager=db_manager,
        initial_cash=10_000_000
    )

    # 삼중창 전략 설정 (기본 설정 + 거래비용 고려)
    triple_screen_daily_strategy = TripleScreenDaily(
        data_store=trader_instance.data_store,
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
        broker=trader_instance.broker
    )

    # 듀얼 모멘텀 전략 설정 (DualMomentumDaily 인스턴스 생성 및 Trader에 주입)
    dual_daily_strategy = DualMomentumDaily(
        data_store=trader_instance.data_store,
        strategy_params={
            'momentum_period': 15,         #  15일
            'rebalance_weekday': 0,        #  월요일 (0)
            'num_top_stocks': 7,           #  5개
            'safe_asset_code': 'A439870',  # 안전자산 코드 (국고채 ETF)
        },
        broker=trader_instance.broker
    )

    temp_daily_strategy = TempletDaily(
        data_store=trader_instance.data_store,
        strategy_params={
            'momentum_period': 15,         # 듀얼 모멘텀처럼 기간 설정이 필요하다면 추가
            'num_top_stocks': 5,           #  5개
            'safe_asset_code': 'A439870', # 안전자산 코드
        },
        broker=trader_instance.broker 
    )
    
    # SMA 일봉 전략 설정 (최적화 결과 반영)
    sma_daily_strategy = SMADaily(
        data_store=trader_instance.data_store,
        strategy_params={
            'short_sma_period': 3,          #  4일
            'long_sma_period': 5,          #  10일
            'volume_ma_period': 5,          #  6일
            'num_top_stocks': 5,            #  5개
            'safe_asset_code': 'A439870',   # 안전자산 코드
        },
        broker=trader_instance.broker
    )
    
    # RSI 분봉 전략 설정 (실제 RSI 계산 및 매매)
    rsi_minute_strategy = RSIMinute(
        data_store=trader_instance.data_store,
        strategy_params={
            'minute_rsi_period': 52,       #  52분
            'minute_rsi_oversold': 30,      # 과매도 -> 매수실행
            'minute_rsi_overbought': 70,    # 과매수 -> 매도실행
            'num_top_stocks': 7,            # 일봉 전략과 동일한 값으로 설정
        },
        broker=trader_instance.broker
    )
    
    
    # 돌파매매 전략 설정 (최적화 결과 반영)
    breakout_daily_strategy = BreakoutDaily(
        data_store=trader_instance.data_store,
        strategy_params = {
            'breakout_period': 10,          # 20일 신고가 돌파
            'volume_ma_period': 20,         # 거래량 20일 이동평균
            'volume_multiplier': 1.5,       # 거래량 이동평균의 1.5배 이상일 때 돌파 인정
            'num_top_stocks': 5,            # 매수할 상위 5개 종목 선정
            'min_holding_days': 2           # 최소 보유 기간 2일 (5일 이내 매도 유도를 위함)
        },
        broker=trader_instance.broker
    )

    # RSI 분봉 전략 설정 (실제 RSI 계산 및 매매)
    breakout_minute_strategy = BreakoutMinute(
        data_store=trader_instance.data_store,
        strategy_params={
            'minute_breakout_period': 10,       # 10분봉 최고가 돌파 확인 기간 (예: 지난 10개 분봉 중 최고가)
            'minute_volume_multiplier': 1.8     # 분봉 거래량 이동평균의 1.8배 이상일 때 돌파 인정
        },
        broker=trader_instance.broker
    )


    # OpenMinute 분봉 전략 설정
    open_minute_strategy = OpenMinute(
        data_store=trader_instance.data_store,
        strategy_params={
            'minute_rsi_period': 52,        #  52분
            'minute_rsi_oversold': 34,      # 과매도 
            'minute_rsi_overbought': 70,    # 과매수
            'num_top_stocks': 7,            # 일봉 전략과 동일한 값으로 설정
        },
        broker=trader_instance.broker
    )

    # 전략 설정 (삼중창 일봉 + RSI 분봉 전략 사용)
    # 전환 14.91
    #trader_instance.set_strategies(daily_strategy=triple_screen_daily_strategy, minute_strategy=rsi_minute_strategy)
    # 상승 20.5 -> 손절 9.37 하락 손절 3.35 미손절 7.29
    #trader_instance.set_strategies(daily_strategy=dual_daily_strategy, minute_strategy=rsi_minute_strategy)
    # 상승 30.97            하락 손절 손절 13.81 미손절 8.74 
    #trader_instance.set_strategies(daily_strategy=dual_daily_strategy, minute_strategy=open_minute_strategy)
    #trader_instance.set_strategies(daily_strategy=breakout_daily_strategy, minute_strategy=breakout_minute_strategy)
    # 전환 84.41%
    #trader_instance.set_strategies(daily_strategy=temp_daily_strategy, minute_strategy=rsi_minute_strategy)
    # 전환 2.59 -> 상승: 미손절 20.54, 손절 -> 5.29 하락 : 손절 0.25 
    trader_instance.set_strategies(daily_strategy=sma_daily_strategy, minute_strategy=rsi_minute_strategy)
    # 전환 2.59 -> 상승: 미손절 20.54, 손절 -> 5.29 하락 : 손절 0.25 
    #trader_instance.set_strategies(daily_strategy=sma_daily_strategy, minute_strategy=open_minute_strategy)
    
    # Broker에 손절매 파라미터 설정 (기본 설정)
    stop_loss_params = {
        'early_stop_loss': -4,        # 매수 후 초기 손실 제한: -3.5% (매수 후 3일 이내)
        'stop_loss_ratio': -6,        # 매수가 기준 손절율: -6.0%
        'trailing_stop_ratio': -4,    # 최고가 기준 트레일링 손절률: -4.0%
        'portfolio_stop_loss': -4,    # 전체 자본금 손실률 (전량매도 조건): -4.0%
        'max_losing_positions': 5       # 최대 손절 종목 수 (전량매도 조건): 3개
    }
    #stop_loss_params = None # 주석을 풀면 미작동
    trader_instance.set_broker_stop_loss_params(stop_loss_params)
    
    trader_manager = TraderManager()
    # 공통 설정 파일에서 모든 종목 이름 가져오기
    from config.sector_config import get_all_stock_names
    stock_names = get_all_stock_names()

    # 자동매매 실행
    portfolio_values, metrics = trader_instance.run(trader_start_date, trader_end_date)