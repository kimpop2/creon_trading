# config/settings.py
import os
from dotenv import load_dotenv
# 1. 이 settings.py 파일의 절대 경로를 기준으로 프로젝트 루트 폴더의 경로를 계산합니다.
#    (settings.py가 config 폴더 안에 있으므로, 상위 폴더로 두 번 올라갑니다)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. 프로젝트 루트 폴더를 기준으로 .env 파일의 전체 경로를 만듭니다.
dotenv_path = os.path.join(project_root, '.env')

# 3. .env 파일의 절대 경로를 명시적으로 지정하여 환경 변수를 로드합니다.
print(f"Loading .env file from: {dotenv_path}") # 경로가 올바른지 확인하기 위한 출력
load_dotenv(dotenv_path=dotenv_path)
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
# --- 로깅 설정 ---
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "trading_system.log"
LOG_FILE_STARTUP = "startup.log" # startup.py 전용 로그 파일 추가
LOG_FILE_CLOSING = "closing.log" # closing.py 전용 로그 파일 추가

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
MARKET_OPEN_TIME = "09:00:00" # 시장 개장 시간 (KST 기준)
MARKET_CLOSE_TIME = "15:30:00" # 시장 마감 시간 (KST 기준)
DAILY_STRATEGY_RUN_TIME = "08:30:00" # 일봉 전략 실행 시간 (장 시작 전)
PORTFOLIO_UPDATE_TIME = "16:00:00" # 장 마감 후 포트폴리오 업데이트 및 결산 시간

FETCH_DAILY_PERIOD = 60 # 여유 일봉 데어터 기간(일)
FETCH_MINUTE_PERIOD = 5 # 여유 분봉데이터 기간(일)
# --- 투자금 기본 설정 ---
INITIAL_CASH = 5000_000
PRINCIPAL_RATIO = 0.5
MIN_STOCK_CAPITAL = 500_000 # 예시: 종목당 최소 5만원으로 매수
# 각 전략에 대한 파라미터를 딕셔너리 형태로 정의합니다.
# --- 공통 파라미터 정의 ---
COMMON_PARAMS = {
    'min_trading_value': 1_000_000_000,
    'max_deviation_ratio': 2.0, # 단위: %
    'min_holding_days': 5,  
    # [수정] 공통으로 사용할 코드들을 이곳으로 이동
    'num_top_stocks': 5,           # 매매 대상 상위 종목 수
    'market_index_code': 'U001',    # 시장 지수 코드 (코스피)
    'safe_asset_code': 'A122630',   # 안전자산 코드 (예: KODEX 200)
    'inverse_etf_code': 'A114800',
}
# 손절매 파라미터 설정
STOP_LOSS_PARAMS = {
#    **COMMON_PARAMS,
    'stop_loss_ratio': -5.0,        # 단순 손절매 비율 (예: -5% 손실 시 손절)
    'trailing_stop_ratio': -4.0,    # 트레일링 스탑 비율 (최고가 대비 -2% 하락 시 손절)
    'early_stop_loss': -3.0,        # 조기 손절매 (매수 후 3일 이내 -3% 손실 시 손절)
    'take_profit_ratio': 10.0,      # 익절 비율 (예: 10% 수익 시 익절)
    'portfolio_stop_loss': -10.0,   # 포트폴리오 전체 손실률 기준 (예: -10% 손실 시 전체 청산)
    'max_losing_positions': 3       # 손실 중인 종목 수 기준 (예: 3개 이상 손실 종목 발생 시 전체 청산)
}
# --- 일봉전략 파라미터 정의 ---
# SMA 일봉 전략 파라미터
SMA_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'short_sma_period': 5,
    'long_sma_period': 10,
#    'num_top_stocks': 3,            # 매매 대상 상위 종목 수
}
# Triple Screen 일봉 전략 파라미터
TRIPLE_SCREEN_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'trend_ma_period': 50,          # 1단계: 장기 추세 판단을 위한 이동평균 기간
    'momentum_rsi_period': 14,      # 2단계: 중기 모멘텀 판단을 위한 RSI 기간
    'momentum_rsi_oversold': 30,    # 2단계: 매수 기회를 포착할 RSI 과매도 기준
    'momentum_rsi_overbought': 70,  # (참고용) RSI 과매수 기준
    'min_trend_strength': 0.02,     # 1단계: 최소 추세 강도 (현재가/이평선 괴리율, 2%)
#    'num_top_stocks': 3,            # 최종적으로 몇 개의 종목에 투자할 것인지 결정
}
# Dual Momentum 일봉 전략 파라미터
DUAL_MOMENTUM_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'momentum_period': 14,          # 모멘텀 계산 기간
    'rebalance_weekday': 0,         # 리밸런싱 요일 (0=월요일)
}
VOL_QUALITY_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'vol_quantile': 0.2,
    'roe_quantile': 0.8,
}
RSI_REVERSION_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'rsi_period': 2,
    'buy_threshold': 10,
    'sell_threshold': 55,
}
VOL_BREAKOUT_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'k_value': 0.5,
}
PAIRS_TRADING_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'pairs_list': [
        ['A005930', 'A000660'],
        ['A005380', 'A000270'],
        ['A035420', 'A035720']
    ],
    'lookback_period': 20,
    'entry_std_dev': 2.0,
    'exit_std_dev': 0.5,
}
INVERSE_DAILY_PARAMS = {
    **COMMON_PARAMS,
    'ma_period': 60,

}

# --- 분봉전략 파라미터 정의 ---
# RSI 분봉 전략 파라미터
RSI_MINUTE_PARAMS = {
    **COMMON_PARAMS,
    'minute_rsi_period': 10,                # 분봉 RSI 계산 기간
    'minute_rsi_oversold': 30,              # RSI 과매도 기준
    'minute_rsi_overbought': 70,            # RSI 과매수 기준
    'num_top_stocks': 5,                    # 매매 대상 상위 종목 수
}

INTELLIGENT_MINUTE_PARAMS = {
    'risk_aversion': 0.1,           # (γ) 위험 회피도: 높을수록 재고 보유를 기피하고 스프레드를 넓힘 (보수적)
    'order_flow_intensity': 1.0,    # (k) 주문 흐름 강도: 높을수록 시장이 활발하다고 판단하여 스프레드를 좁힘 (공격적)
    'volatility_period': 20,        # (σ) 변동성 계산 기간 (분봉 기준)
    'max_inventory': 100,           # 최대 보유 가능 재고 (단위: 주). 재고 페널티를 제한하는 역할
}

# --- [신규] 최적화용 고정/공통 파라미터 ---
COMMON_OPTIMIZATION_PARAMS = {
    'min_trading_value': 1_000_000_000,
    'market_index_code': 'U001',
    'safe_asset_code': 'A122630'
}

# --- [수정] SMA 전략 최적화 파라미터 범위 ('type' 제거) ---
SMA_OPTIMIZATION_PARAMS = {
    'short_sma_period': {'min': 2, 'max': 20, 'step': 1},
    'long_sma_period': {'min': 30, 'max': 80, 'step': 5},
    'volume_ma_period': {'min': 5, 'max': 20, 'step': 1},
    'num_top_stocks': {'min': 2, 'max': 8, 'step': 1}
}

# --- [수정] HMM 최적화 파라미터 범위 ('type' 제거) ---
HMM_OPTIMIZATION_PARAMS = {
    'hmm_n_states': {'min': 3, 'max': 5, 'step': 1},
    'policy_crisis_ratio': {'min': 0.1, 'max': 0.3, 'step': 0.1},
    'policy_bear_ratio': {'min': 0.3, 'max': 0.6, 'step': 0.1},
    'rebalance_performance_metric': {'values': ['sharpe_ratio', 'total_return']}
}


# [신규] 전략별 포트폴리오 설정 (자금 관리용)
STRATEGY_CONFIGS = [
    {'name': 'SMADaily', 'weight': 0.22, 'params': SMA_DAILY_PARAMS},
    {'name': 'DualMomentumDaily', 'weight': 0.13, 'params': DUAL_MOMENTUM_DAILY_PARAMS},
    {'name': 'VolBreakoutDaily', 'weight': 0.65, 'params': VOL_BREAKOUT_DAILY_PARAMS},
    {'name': 'TripleScreenDaily', 'weight': 0.01, 'params': TRIPLE_SCREEN_DAILY_PARAMS},
    {'name': 'VolQualityDaily', 'weight': 0.01, 'params': VOL_QUALITY_DAILY_PARAMS},
    {'name': 'RsiReversionDaily', 'weight': 0.01, 'params': RSI_REVERSION_DAILY_PARAMS},
    {'name': 'PairsTradingDaily', 'weight': 0.01, 'params': PAIRS_TRADING_DAILY_PARAMS},
    {'name': 'InverseDaily', 'weight': 0.01, 'params': INVERSE_DAILY_PARAMS},
]

HMM_OPTIMIZATION_PARAMS = {
    # 1. HMM 모델 자체의 파라미터
    'hmm_n_states': {
        'type': 'int',
        'min': 3,
        'max': 5,
        'step': 1
    },
    # (필요시 다른 HMM 파라미터 추가 가능)

    # 2. 정책 테이블(거시적 자산배분)의 규칙 파라미터
    'policy_crisis_ratio': {
        'type': 'float',
        'min': 0.1,
        'max': 0.3,
        'step': 0.1
    },
    'policy_bear_ratio': {
        'type': 'float',
        'min': 0.3,
        'max': 0.6,
        'step': 0.1
    },
    # (다른 장세에 대한 투자 비중도 추가 가능)

    # 3. 미시적 자산배분(제2두뇌) 관련 파라미터
    'rebalance_performance_metric': {
        'type': 'categorical',
        'values': ['sharpe_ratio', 'total_return', 'win_rate'] # 어떤 지표를 기준으로 기대성과를 계산할지
    }
}
# --- [신규 추가] 실거래용 HMM 모델 설정 ---
# 옵티마이저로 찾은 최적 모델의 이름으로 변경하여 사용합니다.
LIVE_HMM_MODEL_NAME = "Test_HMM_v1" 