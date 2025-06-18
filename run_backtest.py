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
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.rsi_minute import RSIMinute
from strategies.temp_daily import TempletDaily
from manager.data_manager import DataManager
# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[ 
                        logging.FileHandler("backtest_run.log", encoding='utf-8'), 
                        logging.StreamHandler(sys.stdout) 
                    ])

if __name__ == '__main__':
    logging.info("듀얼 모멘텀 주간 백테스트 스크립트를 실행합니다.")

    # 섹터별 대표 종목 리스트 (코스피 25개 + 코스닥 25개, 대표종목)
    sector_stocks = {
        # 코스피 대형주 (25개)
        '코스피_반도체': [
            ('삼성전자', 'IT'), ('SK하이닉스', 'IT'), ('DB하이텍', 'IT')
        ],
        '코스피_2차전지': [
            ('LG에너지솔루션', '2차전지'), ('삼성SDI', '2차전지'), ('SK이노베이션', '2차전지')
        ],
        '코스피_바이오': [
            ('삼성바이오로직스', '바이오'), ('셀트리온', '바이오'), ('SK바이오사이언스', '바이오')
        ],
        '코스피_플랫폼': [
            ('NAVER', 'IT'), ('카카오', 'IT'), ('크래프톤', 'IT')
        ],
        '코스피_자동차': [
            ('현대차', '자동차'), ('기아', '자동차'), ('현대모비스', '자동차')
        ],
        '코스피_철강화학': [
            ('POSCO홀딩스', '철강'), ('고려아연', '철강'), ('롯데케미칼', '화학')
        ],
        '코스피_금융': [
            ('KB금융', '금융'), ('신한지주', '금융'), ('하나금융지주', '금융')
        ],
        '코스피_통신': [
            ('SK텔레콤', '통신'), ('KT', '통신'), ('LG유플러스', '통신')
        ],
        '코스피_유통': [
            ('CJ제일제당', '소비재'), ('오리온', '소비재'), ('롯데쇼핑', '유통')
        ],
        
        # 코스닥 중소형주 (25개)
        '코스닥_반도체': [
            ('한미반도체', 'IT'), ('아이에이', 'IT'), ('테스', 'IT')
        ],
        '코스닥_2차전지': [
            ('에코프로비엠', '2차전지'), ('일진머티리얼즈', '2차전지'), ('엘앤에프', '2차전지')
        ],
        '코스닥_바이오': [
            ('한미사이언스', '바이오'), ('녹십자', '바이오'), ('동아쏘시오홀딩스', '바이오')
        ],
        '코스닥_게임': [
            ('펄어비스', 'IT'), ('스마일게이트홀딩스', 'IT'), ('컴투스', 'IT')
        ],
        '코스닥_자동차부품': [
            ('현대위아', '자동차'), ('현대트랜시스', '자동차'), ('현대제철', '자동차')
        ],
        '코스닥_화학': [
            ('한화솔루션', '화학'), ('S-OIL', '화학'), ('GS칼텍스', '화학')
        ],
        '코스닥_증권': [
            ('NH투자증권', '금융'), ('미래에셋증권', '금융'), ('한국투자증권', '금융')
        ],
        '코스닥_IT': [
            ('SK바이오팜', '통신'), ('SK디스커버리', '통신'), ('SK브로드밴드', '통신')
        ],
        '코스닥_소비재': [
            ('BGF리테일', '유통'), ('롯데마트', '유통'), ('홈플러스', '유통')
        ]
    }
    
    # 백테스트 기간 설정
    # 하락장
    # daily_data_fetch_start  = datetime.datetime(2024, 11, 1, 9, 0, 0).date()
    # backtest_start_date     = datetime.datetime(2024, 12, 1, 9, 0, 0).date()
    # backtest_end_date       = datetime.datetime(2025, 2, 1, 3, 30, 0).date()
    
    # 추세전환
    backtest_start_date     = datetime.datetime(2025, 3, 1, 9, 0, 0).date()
    backtest_end_date       = datetime.datetime(2025, 4, 1, 3, 30, 0).date()
    # 일봉 데이터 가져오기 시작일을 백테스트 시작일 한 달 전으로 자동 설정
    daily_data_fetch_start = (backtest_start_date - datetime.timedelta(days=30)).replace(day=1)
    
    # 상승장
    # daily_data_fetch_start  = datetime.datetime(2025, 4, 1, 9, 0, 0).date()
    # backtest_start_date     = datetime.datetime(2025, 5, 1, 9, 0, 0).date()
    # backtest_end_date       = datetime.datetime(2025, 6, 15, 3, 30, 0).date()
    
    creon_api = CreonAPIClient()
    if not creon_api.connected:
        logging.error("Creon API에 연결할 수 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    # 백테스터 초기화
    backtester_instance = Backtester(creon_api, initial_cash=10_000_000)

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
    # RSI 분봉 전략 설정 (RSIMinute 인스턴스 생성 및 Backtester에 주입)
    rsi_minute_strategy = RSIMinute(
        data_store=backtester_instance.data_store,
        strategy_params={
            'minute_rsi_period': 45,
            'minute_rsi_oversold': 30,
            'minute_rsi_overbought': 70,
            # 손절매 파라미터는 이제 Broker에 직접 설정합니다.
        },
        broker=backtester_instance.broker # Broker 인스턴스 전달
    )

    # 전략 설정 (듀얼 모멘텀 전략 사용)
    #backtester_instance.set_strategies(daily_strategy=dual_daily_strategy, minute_strategy=rsi_minute_strategy)
    backtester_instance.set_strategies(daily_strategy=temp_daily_strategy, minute_strategy=rsi_minute_strategy)
    
    # Broker에 손절매 파라미터 설정
    stop_loss_params = {
        # 'stop_loss_ratio': -5.0,      # 기본 손절 비율
        # 'trailing_stop_ratio': -3.0,   # 트레일링 스탑 비율
        # 'portfolio_stop_loss': -5.0,   # 포트폴리오 전체 손절 비율
        # 'early_stop_loss': -5.0,       # 초기 손절 비율 (5일 이내)
        # 'max_losing_positions': 5,     # 동시 손실 허용 종목 수
        'stop_loss_ratio': -5.0,      # 기본 손절 비율
        'trailing_stop_ratio': -5.0,   # 트레일링 스탑 비율
        'portfolio_stop_loss': -5.0,   # 포트폴리오 전체 손절 비율
        'early_stop_loss': -5.0,       # 초기 손절 비율 (5일 이내)
        'max_losing_positions': 3,     # 동시 손실 허용 종목 수

    }
    #stop_loss_params = None #손절하지 않기
    backtester_instance.set_broker_stop_loss_params(stop_loss_params)
    
    data_manager = DataManager()
    # 모든 종목을 하나의 리스트로 변환
    stock_names = []
    for sector, stocks in sector_stocks.items():
        for stock_name, _ in stocks:
            stock_names.append(stock_name)

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