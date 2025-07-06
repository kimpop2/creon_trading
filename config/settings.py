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
CREON_ID = "KIMPOP2"         # 크레온 HTS 로그인 ID
CREON_PWD = "@uncle1"  # 크레온 HTS 로그인 비밀번호
CREON_CERT_PWD = "hana#uncle1" # 크레온 공동인증서 비밀번호

# --- 자동매매 기본 설정 ---
# 초기 투자 자본 (백테스트 및 실전 거래의 초기 예수금으로 사용됩니다.)
INITIAL_DEPOSIT = 50_000_000 # 5천만원 예시

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

# SMADaily 전략 파라미터
SMADAILY_PARAMS = {
    'short_sma_period': 5,          # 단기 이동평균 기간
    'long_sma_period': 20,          # 장기 이동평균 기간
    'volume_ma_period': 20,         # 거래량 이동평균 기간
    'num_top_stocks': 10,           # 매수 후보 상위 종목 수
    'min_holding_days': 3,          # 매도 후보 선정 시 최소 홀딩 일수
    'buy_capital_ratio': 0.1        # 매수 시 총 현금 자산의 비율 (10%)
                                    # 예: 남은 현금 5천만원의 10% (5백만원)을 매수에 사용
    # 'buy_quantity_per_stock': None # 종목당 고정 매수 수량 (None이면 buy_capital_ratio에 따라 계산)
}

# RSIMinute 전략 파라미터
RSIMINUTE_PARAMS = {
    'minute_rsi_period': 14,                # 분봉 RSI 계산 기간
    'minute_rsi_oversold': 30,              # RSI 과매도 기준
    'minute_rsi_overbought': 70,            # RSI 과매수 기준
    'minute_lookback': 120,                 # 분봉 데이터 조회 기간 (분)
    'time_cut_sell_after_minutes': (15 * 60) + 5, # 장 시작 후 15시 5분 (분 단위, 즉 9시부터 6시간 5분 = 365분)
                                                # 이 시간 이후 미체결 매수 주문에 대한 타임컷 검토
    'max_price_diff_ratio_for_timecut': 0.01 # 타임컷 매도 허용 괴리율 (1%)
}

# --- 로깅 설정 ---
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "trading_system.log"
LOG_FILE_STARTUP = "startup.log" # startup.py 전용 로그 파일 추가
LOG_FILE_CLOSING = "closing.log" # closing.py 전용 로그 파일 추가
