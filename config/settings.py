# backtesting/config/settings.py

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

#Data Manager Settings (향후 필요시 추가)
DEFAULT_OHLCV_DAYS_TO_FETCH = 365 # 기본적으로 가져올 일봉 데이터 기간 (일)
DEFAULT_MINUTE_DAYS_TO_FETCH = 5 # 기본적으로 가져올 분봉 데이터 기간 (일)

#Strategy Settings (향후 필요시 추가)
DUAL_MOMENTUM_LOOKBACK_PERIOD = 12 # 듀얼 모멘텀 전략의 모멘텀 측정 기간 (개월)
DUAL_MOMENTUM_REBALANCE_PERIOD = 'monthly' # 듀얼 모멘텀 전략의 리밸런싱 주기