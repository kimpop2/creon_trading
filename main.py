# main.py

import logging
import sys
import os
from datetime import time as dt_time, timedelta

# 프로젝트 루트 경로를 sys.path에 추가 (다른 모듈 임포트를 위함)
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 설정 파일 로드
from config.settings import (
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    INITIAL_DEPOSIT,
    MARKET_OPEN_TIME, MARKET_CLOSE_TIME,
    DAILY_STRATEGY_RUN_TIME, PORTFOLIO_UPDATE_TIME,
    SMADAILY_PARAMS, RSIMINUTE_PARAMS,
    LOG_LEVEL, LOG_FILE
)

# 모듈 임포트
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from util.notifier import Notifier
from trading.trading import Trading
from strategy.sma_daily import SMADaily
from strategy.rsi_minute import RSIMinute

# --- 로거 설정 ---
def setup_logging():
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    
    # 기본 로거 설정
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'), # 파일 로깅
            logging.StreamHandler(sys.stdout) # 콘솔 로깅
        ]
    )
    # 특정 라이브러리 로깅 레벨 조정 (선택 사항)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

# 메인 함수
def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("========================================")
    logger.info("   자동매매 시스템 시작 준비 중...     ")
    logger.info("========================================")

    # 1. Creon API 클라이언트 초기화 및 연결
    creon_api_client = CreonAPIClient()
    
    # Creon HTS 및 API 연결 확인 (반복 시도)
    max_retries = 5
    for i in range(max_retries):
        if creon_api_client.connect():
            logger.info("Creon API 연결 성공.")
            break
        else:
            logger.error(f"Creon API 연결 시도 {i+1}/{max_retries} 실패. 5초 후 재시도...")
            if i == max_retries - 1:
                logger.critical("Creon API 연결에 최종 실패했습니다. 시스템을 종료합니다.")
                sys.exit(1)
            creon_api_client.cleanup() # 이전 COM 객체 정리
            import time
            time.sleep(5)
            creon_api_client = CreonAPIClient() # 새로운 CreonAPIClient 인스턴스 생성 (COM 객체 초기화를 위해)


    # 2. DBManager 초기화
    db_manager = DBManager()
    if not db_manager.get_db_connection():
        logger.critical("데이터베이스 연결 실패. 시스템을 종료합니다.")
        creon_api_client.cleanup()
        sys.exit(1)
    logger.info("DBManager 초기화 완료.")

    # 3. Notifier 초기화
    notifier = Notifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    notifier.send_message("💡 자동매매 시스템이 초기화되고 있습니다.")
    logger.info("Notifier 초기화 완료.")

    # 4. Trading 시스템 초기화
    trading_system = Trading(
        creon_api_client=creon_api_client,
        db_manager=db_manager,
        notifier=notifier,
        initial_deposit=INITIAL_DEPOSIT
    )
    logger.info("Trading 시스템 초기화 완료.")

    # 5. 시간 설정 적용
    # settings.py의 문자열 시간을 datetime.time 객체로 변환하여 적용
    trading_system.market_open_time = dt_time(*map(int, MARKET_OPEN_TIME.split(':')))
    trading_system.market_close_time = dt_time(*map(int, MARKET_CLOSE_TIME.split(':')))
    trading_system.daily_strategy_run_time = dt_time(*map(int, DAILY_STRATEGY_RUN_TIME.split(':')))
    trading_system.portfolio_update_time = dt_time(*map(int, PORTFOLIO_UPDATE_TIME.split(':')))
    logger.info(f"시장 개장 시간: {trading_system.market_open_time}")
    logger.info(f"시장 마감 시간: {trading_system.market_close_time}")
    logger.info(f"일봉 전략 실행 시간: {trading_system.daily_strategy_run_time}")
    logger.info(f"포트폴리오 업데이트 시간: {trading_system.portfolio_update_time}")


    # 6. 매매 전략 설정
    # SMADaily 전략 인스턴스 생성
    daily_strategy_instance = SMADaily(
        brokerage=trading_system.brokerage,
        trading_manager=trading_system.trading_manager,
        strategy_params=SMADAILY_PARAMS
    )
    
    # RSIMinute 전략 인스턴스 생성
    minute_strategy_instance = RSIMinute(
        brokerage=trading_system.brokerage,
        trading_manager=trading_system.trading_manager,
        strategy_params=RSIMINUTE_PARAMS
    )

    trading_system.set_strategies(
        daily_strategy=daily_strategy_instance,
        minute_strategy=minute_strategy_instance
    )
    logger.info("매매 전략 설정 완료.")

    logger.info("========================================")
    logger.info("   자동매매 시스템 시작합니다.        ")
    logger.info("========================================")
    notifier.send_message("✅ 자동매매 시스템이 성공적으로 시작됩니다.")

    try:
        # 메인 자동매매 루프 시작
        trading_system.start_trading_loop()
    except KeyboardInterrupt:
        logger.info("사용자에 의해 시스템 종료 요청됨.")
        notifier.send_message("⚠️ 사용자 요청으로 자동매매 시스템이 종료됩니다.")
    except Exception as e:
        logger.critical(f"자동매매 시스템 실행 중 치명적인 오류 발생: {e}", exc_info=True)
        notifier.send_message(f"🚨 자동매매 시스템 치명적 오류 발생: {e}")
    finally:
        trading_system.cleanup()
        logger.info("시스템 종료 및 자원 정리 완료.")
        notifier.send_message("🛑 자동매매 시스템이 완전히 종료되었습니다.")
        sys.exit(0)

if __name__ == "__main__":
    main()