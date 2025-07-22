# config/settings.py
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
# 이 함수는 load_dotenv()를 호출한 시점부터 환경 변수를 사용할 수 있게 합니다.
load_dotenv()

# Database Settings
DB_HOST = 'localhost'
DB_PORT = 3306 # MariaDB 또는 MySQL 기본 포트
DB_NAME = 'backtest_db' # 데이터베이스 이름 (예: task 2에서 생성 예정)
# .env 파일에서 DB 사용자 이름과 비밀번호를 환경 변수로 불러옵니다.
# .env 파일에 DB_USER와 DB_PASSWORD가 정의되어 있어야 합니다.
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
# DBManager에서 사용할 수 있도록 DB_CONFIG 딕셔너리 추가
DB_CONFIG = {
    'host': DB_HOST,
    'port': DB_PORT,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'database': DB_NAME
}

# Creon API Settings (향후 필요시 추가)
API_CONNECT_TIMEOUT = 30 # Creon API 연결 시도 타임아웃 (초)
API_REQUEST_INTERVAL = 0.2 # API 요청 간 최소 대기 시간 (초)
# --- Creon 자동 로그인 설정 (보안 주의: 실제 계정 정보) ---
CREON_ID = os.getenv('CREON_ID')        # 크레온 HTS 로그인 ID
CREON_PWD = os.getenv('CREON_PWD')  # 크레온 HTS 로그인 비밀번호
CREON_CERT_PWD = os.getenv('CREON_CERT_PWD') # 크레온 공동인증서 비밀번호
# --- 텔레그램 설정 (보안 주의: 실제 계정 정보) ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')        # 크레온 HTS 로그인 ID
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')  # 크레온 HTS 로그인 비밀번호

# --- 자동매매 기본 설정 ---
# 초기 투자 자본 (백테스트 및 실전 거래의 초기 예수금으로 사용됩니다.)
INITIAL_CASH = 300_000 # 30만원 예시

# 시장 개장 시간 (KST 기준)
MARKET_OPEN_TIME = "09:00:00"
# 시장 마감 시간 (KST 기준)
MARKET_CLOSE_TIME = "15:30:00" # 동시 호가 시간 포함

# 일봉 전략 실행 시간 (장 시작 전)
DAILY_STRATEGY_RUN_TIME = "08:30:00"
# 장 마감 후 포트폴리오 업데이트 및 결산 시간
PORTFOLIO_UPDATE_TIME = "16:00:00"

# --- 전략 파라미터 ---
# 각 전략에 대한 파라미터를 딕셔너리 형태로 정의합니다.
# --- 공통 파라미터 정의 ---
COMMON_PARAMS = {
    #'num_top_stocks': 5,       # 매매 대상 상위 종목 수
    'max_deviation_ratio': 2.0, # 단위: %
    'min_holding_days': 5,
    'safe_asset_code': 'U001',
}
# 손절매 파라미터 설정 예시 (선택 사항)
STOP_LOSS_PARAMS = {
    **COMMON_PARAMS,
    'stop_loss_ratio': -5.0,        # 단순 손절매 비율 (예: -5% 손실 시 손절)
    'trailing_stop_ratio': -4.0,    # 트레일링 스탑 비율 (최고가 대비 -2% 하락 시 손절)
    'early_stop_loss': -3.0,        # 조기 손절매 (매수 후 3일 이내 -3% 손실 시 손절)
    'take_profit_ratio': 10.0,      # 익절 비율 (예: 10% 수익 시 익절)
    'portfolio_stop_loss': -10.0,   # 포트폴리오 전체 손실률 기준 (예: -10% 손실 시 전체 청산)
    'max_losing_positions': 3       # 손실 중인 종목 수 기준 (예: 3개 이상 손실 종목 발생 시 전체 청산)
}

# SMA 일봉 전략 파라미터
SMA_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'short_sma_period': 2,          # 단기 이동평균선 기간 (일봉)
    'long_sma_period': 5,          # 장기 이동평균선 기간 (일봉)
    'volume_ma_period': 7,          # 거래량 이동평균선 기간 (일봉)
    'num_top_stocks': 10,            # 매매 대상 상위 종목 수
    # market_sma_period
    'market_sma_period': 60,        # 시장 트랜드 이동 평균선 기간 (일봉)
    'volume_lookback_days': 5,
    'range_coefficient': 0.5,       # 변동성 계수
    'market_index_code': 'U001',    # 시장 지수 코드(코스피 200)
}

# RSI 분봉 전략 파라미터
RSI_MINUTE_PARAMS = {
    **COMMON_PARAMS,
    'minute_rsi_period': 10,                # 분봉 RSI 계산 기간
    'minute_rsi_oversold': 30,              # RSI 과매도 기준
    'minute_rsi_overbought': 70,            # RSI 과매수 기준
    'num_top_stocks': 5,                    # 매매 대상 상위 종목 수
}

# OpenMinute 전략 파라미터 (RSI 기반)
OPEN_MINUTE_PARAMS = {
    **COMMON_PARAMS,
    'minute_rsi_period': 14,                # 분봉 RSI 계산 기간
    'minute_rsi_oversold': 30,              # RSI 과매도 기준
    'minute_rsi_overbought': 70,            # RSI 과매수 기준
    'num_top_stocks': 5,                    # 매매 대상 상위 종목 수
}

# Breakout 일봉 전략 파라미터
BREAKOUT_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'breakout_period': 20,                  # 돌파 기간 (일봉)
    'volume_ma_period': 20,                 # 거래량 이동평균 기간
    'volume_multiplier': 1.5,               # 거래량 배수
    'num_top_stocks': 5,                    # 매매 대상 상위 종목 수
    'min_holding_days': 3,                  # 최소 보유 기간
}

# Breakout 분봉 전략 파라미터
BREAKOUT_MINUTE_PARAMS = {
    **COMMON_PARAMS,
    'minute_breakout_period': 10,           # 분봉 돌파 기간
    'minute_volume_multiplier': 1.2,        # 분봉 거래량 배수
}

# Dual Momentum 일봉 전략 파라미터
DUAL_MOMENTUM_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'momentum_period': 20,                  # 모멘텀 계산 기간
    'rebalance_weekday': 0,                 # 리밸런싱 요일 (0=월요일)
    'num_top_stocks': 5,                    # 매매 대상 상위 종목 수
    'safe_asset_code': 'A001',              # 안전자산 코드 (KOSPI)
}

# Sector Rotation 일봉 전략 파라미터
SECTOR_ROTATION_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'momentum_period': 20,                  # 모멘텀 계산 기간
    'rebalance_weekday': 0,                 # 리밸런싱 요일 (0=월요일)
    'num_top_sectors': 3,                   # 상위 섹터 수
    'stocks_per_sector': 2,                 # 섹터당 종목 수
    'safe_asset_code': 'A001',              # 안전자산 코드 (KOSPI)
}

# Triple Screen 일봉 전략 파라미터
TRIPLE_SCREEN_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'trend_ma_period': 50,                  # 추세 이동평균 기간
    'momentum_rsi_period': 14,              # 모멘텀 RSI 기간
    'momentum_rsi_oversold': 30,            # RSI 과매도 기준
    'momentum_rsi_overbought': 70,          # RSI 과매수 기준
    'volume_ma_period': 20,                 # 거래량 이동평균 기간
    'num_top_stocks': 5,                    # 매매 대상 상위 종목 수
    'safe_asset_code': 'A001',              # 안전자산 코드 (KOSPI)
    'min_trend_strength': 0.02,             # 최소 추세 강도
}

# Bollinger RSI 일봉 전략 파라미터
BOLLINGER_RSI_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'bb_period': 20,                        # 볼린저 밴드 기간
    'bb_std': 2,                            # 볼린저 밴드 표준편차
    'rsi_period': 14,                       # RSI 기간
    'rsi_oversold': 30,                     # RSI 과매도 기준
    'rsi_overbought': 70,                   # RSI 과매수 기준
    'volume_ma_period': 20,                 # 거래량 이동평균 기간
    'num_top_stocks': 5,                    # 매매 대상 상위 종목 수
    'safe_asset_code': 'A001',              # 안전자산 코드 (KOSPI)
}

# Templet 일봉 전략 파라미터 (기본 모멘텀)
TEMPLET_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'momentum_period': 20,                  # 모멘텀 계산 기간
    'num_top_stocks': 5,                    # 매매 대상 상위 종목 수
    'safe_asset_code': 'A001',              # 안전자산 코드 (KOSPI)
}



# --- 로깅 설정 ---
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "trading_system.log"
LOG_FILE_STARTUP = "startup.log" # startup.py 전용 로그 파일 추가
LOG_FILE_CLOSING = "closing.log" # closing.py 전용 로그 파일 추가
