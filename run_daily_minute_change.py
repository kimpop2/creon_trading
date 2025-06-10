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

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[ 
                        logging.FileHandler("backtest_run.log"), 
                        logging.StreamHandler(sys.stdout) 
                    ])

if __name__ == '__main__':
    logging.info("듀얼 모멘텀 주간 백테스트 스크립트를 실행합니다.")

    creon_api_client = CreonAPIClient()
    if not creon_api_client.connected:
        logging.error("Creon API에 연결할 수 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    # 백테스트 기간 설정
    daily_data_fetch_start = '20250301' 
    backtest_start_date = datetime.datetime(2025, 4, 1, 9, 0, 0)
    backtest_end_date = datetime.datetime(2025, 6, 4, 15, 30, 0)

    # --- 전략 파라미터 정의 (여기에 직접 설정) ---
    strategy_params = {
        # 'momentum_period': 20, # <-- 이 값을 여기서 직접 변경 가능
        # 'rebalance_weekday': 4, # 0:월, 1:화, 2:수, 3:목, 4:금
        # 'num_top_stocks': 5,
        # 'safe_asset_code': 'A439870',  # KODEX 국고채30년 액티브
        # 'initial_cash': 10_000_000,
        # 'minute_rsi_period': 14,
        # 'minute_rsi_overbought': 70,
        # 'minute_rsi_oversold': 30,
        # 'stop_loss_ratio': -5, # -5% 손실 시 손절
        # 'trailing_stop_ratio': -2, # 최고가 대비 -2% 하락 시 트레일링 스탑
        # 'early_stop_loss': -1, # 매수 후 5거래일 이내 -1% 손실 시 초기 손절
        # 'max_losing_positions': 3 # 최대 허용 손실 포지션 수 (현재 사용되지 않음)
            'momentum_period': 20,         # 모멘텀 계산 기간 (거래일)
            'rebalance_weekday': 4,        # 리밸런싱 요일 (0: 월요일, 4: 금요일)
            'num_top_stocks': 7,           # 상위 N종목 선택
            'safe_asset_code': 'A439870',  # 안전자산 코드 (국고채 ETF)
            'minute_rsi_period': 45,       # 분봉 RSI 기간 (60분 → 45분)
            'minute_rsi_overbought': 65,   # RSI 과매수 기준 (70 → 65)
            'minute_rsi_oversold': 35,     # RSI 과매도 기준 (30 → 35)
            'stop_loss_ratio': -5.0,       # 기본 손절 비율
            'trailing_stop_ratio': -3.0,   # 트레일링 스탑 비율
            'portfolio_stop_loss': -10.0,  # 포트폴리오 전체 손절 비율
            'early_stop_loss': -3.0,       # 초기 손절 비율 (5일 이내)
            'max_losing_positions': 3,      # 동시 손실 허용 종목 수
            'initial_cash': 10_000_000 # 포트폴리오 손절 계산을 위해 추가
    }
    # --- 전략 파라미터 정의 끝 ---

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


    # 모든 종목을 하나의 리스트로 변환
    stock_names = []
    for sector, stocks in sector_stocks.items():
        for stock_name, _ in stocks:
            stock_names.append(stock_name)
    
    # Backtester 인스턴스 초기화 (strategy_params 전달)
    backtester_instance = Backtester(creon_api_client, initial_cash=10_000_000)

    # --- 전략 인스턴스 생성 및 Backtester에 주입 ---
    # 이제 strategy_params는 이 파일에서 정의한 것을 직접 전달
    dual_momentum_strategy = DualMomentumDaily(
        data_store=backtester_instance.data_store, 
        strategy_params=strategy_params, # <-- 여기를 직접 정의한 strategy_params로 변경
        broker=backtester_instance.broker
    )

    rsi_minute_strategy = RSIMinute(
        data_store=backtester_instance.data_store, 
        strategy_params=strategy_params, # <-- 여기를 직접 정의한 strategy_params로 변경
        broker=backtester_instance.broker,
        position_info=backtester_instance.position_info 
    )

    # Backtester에 전략 인스턴스 설정 (Dependency Injection)
    backtester_instance.set_strategies(
        daily_strategy=dual_momentum_strategy,
        minute_strategy=rsi_minute_strategy
    )
    # --- 전략 인스턴스 주입 끝 ---


    # 종목 코드 확인 및 일봉 데이터 로딩
    # 안전자산 코드도 미리 추가
    safe_asset_code = strategy_params['safe_asset_code'] # <-- 여기서 직접 정의한 strategy_params에서 가져옴
    logging.info(f"'안전자산' (코드: {safe_asset_code}) 안전자산 일봉 데이터 로딩 중... (기간: {daily_data_fetch_start} ~ {backtest_end_date.strftime('%Y%m%d')})")
    daily_df = creon_api_client.get_daily_ohlcv(safe_asset_code, daily_data_fetch_start, backtest_end_date.strftime('%Y%m%d'))
    backtester_instance.add_daily_data(safe_asset_code, daily_df)
    if daily_df.empty:
        logging.warning(f"'안전자산' (코드: {safe_asset_code}) 종목의 일봉 데이터를 가져올 수 없습니다. 종료합니다다.")
        exit(1)
    logging.info(f"'안전자산' (코드: {safe_asset_code}) 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
    
    all_target_stock_names = stock_names

    for name in all_target_stock_names:
        code = creon_api_client.get_stock_code(name)
        if code:
            logging.info(f"'{name}' (코드: {code}) 종목 일봉 데이터 로딩 중... (기간: {daily_data_fetch_start} ~ {backtest_end_date.strftime('%Y%m%d')})")
            daily_df = creon_api_client.get_daily_ohlcv(code, daily_data_fetch_start, backtest_end_date.strftime('%Y%m%d'))
            time.sleep(0.3) 
            
            if daily_df.empty:
                logging.warning(f"{name} ({code}) 종목의 일봉 데이터를 가져올 수 없습니다. 해당 종목을 건너뜁니다.")
                continue
            logging.info(f"{name} ({code}) 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
            backtester_instance.add_daily_data(code, daily_df)
        else:
            logging.warning(f"'{name}' 종목의 코드를 찾을 수 없습니다. 해당 종목을 건너뜁니다.")

    if not backtester_instance.data_store['daily']:
        logging.error("백테스트를 위한 유효한 일봉 데이터가 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)
            
    # 백테스트 실행
    portfolio_values, metrics = backtester_instance.run(backtest_start_date, backtest_end_date)