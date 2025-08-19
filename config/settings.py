# config/settings.py (리팩토링 완료)
import os
from dotenv import load_dotenv

# --- 프로젝트 루트 경로 설정 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# =================================================================
# 1. 시스템 기본 설정 (DB, API, 로깅 등)
# =================================================================

# --- Database Settings ---
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'backtest_db'
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# --- Logging Settings ---
LOG_LEVEL = "INFO"
LOG_FILE = "trading_system.log"
LOG_FILE_STARTUP = "startup.log"
LOG_FILE_CLOSING = "closing.log"

# --- Creon API Settings ---
CREON_ID = os.getenv('CREON_ID')
CREON_PWD = os.getenv('CREON_PWD')
CREON_CERT_PWD = os.getenv('CREON_CERT_PWD')

# --- Telegram Settings ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# =================================================================
# 2. 자동매매 공통 설정
# =================================================================

# --- Market Hours & Execution Timing ---
MARKET_OPEN_TIME = "09:00:00"
MARKET_CLOSE_TIME = "15:30:00"

# --- Data Fetching Settings ---
FETCH_DAILY_PERIOD = 60
FETCH_MINUTE_PERIOD = 5

# --- Capital & Risk Management Settings ---
INITIAL_CASH = 5_000_000
PRINCIPAL_RATIO = 0.5  # HMM 모드가 아닐 때 사용할 기본 투자 비중
MIN_STOCK_CAPITAL = 300_000

STOP_LOSS_PARAMS = {
    'stop_loss_ratio': -5.0,
    'trailing_stop_ratio': -4.0,
    'early_stop_loss': -3.0,
    'take_profit_ratio': 10.0,
    'portfolio_stop_loss': -10.0,
    'max_losing_positions': 3
}

# =================================================================
# 3. 통합 전략 설정 (STRATEGY_CONFIGS)
# - 각 전략의 실행 파라미터와 최적화 파라미터를 통합 관리합니다.
# - 이 구조는 향후 DB의 'strategies' 테이블과 1:1로 매핑될 수 있습니다.
# =================================================================
COMMON_PARAMS = {
    'market_index_code': 'U001',    # 시장 지수 코드 (코스피)
    'safe_asset_code': 'A122630',   # 안전자산 코드 (예: KODEX 200)
    'inverse_etf_code': 'A114800',  # 인버스 ETF
}

OPTIMIZATION_LOOKBACK_MONTHS = 6 # 전략 프로파일링 시 사용할 과거 데이터 기간 (월)

STRATEGY_CONFIGS = {
    "SMADaily": {
        "description": "단순 이동평균 골든/데드크로스 전략",
        "strategy_status": True,
        "strategy_weight": 0.22,
        "default_params": {
            "short_sma_period": 5,
            "long_sma_period": 17,
            "min_trading_value": 1_000_000_000,
            "max_deviation_ratio": 2.0,
            "max_position_count": 5,
            "min_position_period": 1,
        },
        "optimization_params": {
            "short_sma_period": {"min": 2, "max": 7, "step": 1},
            "long_sma_period": {"min": 10, "max": 30, "step": 5},
            "max_position_count": {"min": 2, "max": 10, "step": 1},
            "min_position_period": {"min": 0, "max": 20, "step": 2}
        },
    },

    "DualMomentumDaily": {
        "description": "듀얼 모멘텀 전략",
        "strategy_status": True,
        "strategy_weight": 0.22,
        "default_params": {
            "momentum_period": 3,
            "rebalance_weekday": 0,
            "market_index_code": "U001",
            "safe_asset_code": "A122630",
            "inverse_etf_code": "A114800",
            "min_trading_value": 1_000_000_000,
            "max_position_count": 5,
            "min_position_period": 1,
        },
        "optimization_params": {
            "momentum_period": {"min": 2, "max": 20, "step": 2},
            "max_position_count": {"min": 2, "max": 10, "step": 1},
            "min_position_period": {"min": 0, "max": 20, "step": 2}
        },
    },

    # =================================================================
    # [수정] BreakoutDaily: 개발된 코드에 맞춰 파라미터 구체화
    # =================================================================
    "BreakoutDaily": {
        "description": "60일 신고가 돌파 및 거래량 급증 추세추종 전략",
        "strategy_status": False,
        "strategy_weight": 0.22,
        "default_params": {
            "box_period": 60,              # 신고가 판단 기간 (일)
            "volume_period": 20,           # 평균 거래량 계산 기간 (일)
            "volume_multiplier": 2.5,      # 평균 거래량 대비 폭증 배수
            "sell_sma_period": 10,         # 추세 이탈 판단을 위한 이동평균 기간
            "min_trading_value": 5_000_000_000, # 최소 평균 거래대금
            "max_position_count": 5,
            "min_position_period": 1,
        },
        "optimization_params": {
            "box_period": {"min": 20, "max": 120, "step": 20},
            "volume_multiplier": {"min": 2.0, "max": 5.0, "step": 0.5},
            "sell_sma_period": {"min": 5, "max": 20, "step": 5},
            "max_position_count": {"min": 2, "max": 10, "step": 1},
            "min_position_period": {"min": 0, "max": 20, "step": 2}
        },
    },

    # =================================================================
    # [신규] PullbackDaily: 눌림목 매매 전략 추가
    # =================================================================
    "PullbackDaily": {
        "description": "상승 추세 중 단기 조정(눌림목) 시 반등을 노리는 전략",
        "strategy_status": False,
        "strategy_weight": 0.22,
        "default_params": {
            "trend_sma_period": 20,       # 대추세 판단을 위한 이동평균 기간
            "pullback_sma_period": 5,     # 눌림목 판단을 위한 단기 이동평균 기간
            "min_trading_value": 1_000_000_000,
            "max_position_count": 5,
            "min_position_period": 1,
        },
        "optimization_params": {
            "trend_sma_period": {"min": 20, "max": 60, "step": 10},
            "pullback_sma_period": {"min": 3, "max": 10, "step": 1},
            "max_position_count": {"min": 2, "max": 10, "step": 1},
            "min_position_period": {"min": 0, "max": 20, "step": 2}
        },
    },

    # =================================================================
    # [신규] ClosingBetDaily: 종가 베팅 전략 추가
    # =================================================================
    "ClosingBetDaily": {
        "description": "강한 마감 종목을 종가 매수 후 익일 갭 상승을 노리는 초단기 전략",
        "strategy_status": False,
        "strategy_weight": 0.22,
        "default_params": {
            "trend_sma_period": 5,        # 단기 추세 판단을 위한 이동평균 기간
            "close_high_ratio": 0.8,      # 종가/고가 비율로 마감 강도 판단
            "min_trading_value": 2_000_000_000,
            "max_position_count": 5,
            "min_position_period": 1,
        },
        "optimization_params": {
            "trend_sma_period": {"min": 3, "max": 10, "step": 1},
            "close_high_ratio": {"min": 0.7, "max": 0.95, "step": 0.05},
            "max_position_count": {"min": 2, "max": 10, "step": 1},
            "min_position_period": {"min": 0, "max": 20, "step": 2}
        },
    },
    
    # --- 분봉 전략도 일관성을 위해 이 구조에 포함 ---
    "TargetPriceMinute": {
        "description": "일봉 전략의 목표가(Target Price) 기반 분할 매매 실행",
        "strategy_status": False,
        "strategy_weight": 0.22,
        "default_params": {
             # 분봉 전략은 보통 자체 파라미터보다 일봉 전략의 신호를 받아 동작
        },
        "optimization_params": {},  # 분봉 전략은 보통 최적화 대상이 아님
        "portfolio_params": {},     # 분봉 전략은 포토폴리오 대상이 아님
    },
    "PassMinute": {
        "description": "분봉 없이 분봉매매 시뮬레이션 매매 실행",
        "default_params": {
             # 분봉 전략은 보통 자체 파라미터보다 일봉 전략의 신호를 받아 동작
        },
        "optimization_params": {},  # 분봉 전략은 보통 최적화 대상이 아님
    },

    "IntelligentMinute": {
        "description": "지능형 주문 집행 전략",
        "strategy_status": False,
        "strategy_weight": 0.22,
        "default_params": {
            "risk_aversion": 0.5,
            "order_flow_intensity": 1000,
            "volatility_period": 20,
            "max_chase_count": 5,
            "chase_interval_seconds": 20
        },
        "optimization_params": {}
    }
}

# =================================================================
# 4. HMM 시스템 및 포트폴리오 설정
# =================================================================

# --- HMM 모델 자체와 포트폴리오 운영 정책에 대한 최적화 파라미터 ---
HMM_OPTIMIZATION_PARAMS = {
    "hmm_n_states": {"min": 3, "max": 5, "step": 1},
    "policy_crisis_ratio": {"min": 0.1, "max": 0.3, "step": 0.1},
    "policy_bear_ratio": {"min": 0.3, "max": 0.6, "step": 0.1},
    "rebalance_performance_metric": {"values": ["sharpe_ratio", "total_return"]}
}

# --- 실거래 환경에서 사용할 HMM 모델의 이름 ---
LIVE_HMM_MODEL_NAME = "EKLMNO_4s_2301-2412"