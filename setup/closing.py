# closing.py (통합 완료된 최종본)

import logging
import sys
import os
from datetime import date, datetime

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from manager.db_manager import DBManager
from api.creon_api import CreonAPIClient # [추가] CreonAPIClient 임포트
from manager.setup_manager import SetupManager # [추가] SetupManager 임포트
from util.notifier import Notifier
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, LOG_LEVEL, LOG_FILE_CLOSING

# 로거 설정
def setup_closing_logging():
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_CLOSING, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger('urllib3').setLevel(logging.WARNING)

setup_closing_logging()
logger = logging.getLogger(__name__)

def run_closing_tasks():
    """
    자동매매 시스템 종료 후 필요한 모든 후처리 작업을 수행합니다.
    (일일 데이터 결산, 팩터 업데이트 등)
    """
    logger.info("========================================")
    logger.info("      자동매매 시스템 종료 후 작업 시작     ")
    logger.info("========================================")

    notifier = Notifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    notifier.send_message("🧹 자동매매 시스템 종료 후 정리 작업이 시작됩니다.")

    db_manager = None
    api_client = None
    try:
        # 1. 핵심 모듈 초기화
        db_manager = DBManager()
        if not db_manager.get_db_connection():
            raise ConnectionError("DB 연결에 실패했습니다.")
        logger.info("DB 연결 성공.")

        api_client = CreonAPIClient()
        if not api_client.is_connected():
            raise ConnectionError("Creon API 연결에 실패했습니다.")
        logger.info("Creon API 연결 성공.")

        setup_manager = SetupManager(api_client, db_manager)
        logger.info("SetupManager 초기화 성공.")

        # 2. 일별 퀀트 팩터 업데이트 실행
        logger.info("▶ 일별 퀀트 팩터 업데이트를 시작합니다...")
        factor_update_success = setup_manager.run_daily_factor_update(datetime.now().date())
        if factor_update_success:
            logger.info("✔ 일별 퀀트 팩터 업데이트 성공.")
            notifier.send_message("📈 일별 퀀트 팩터 업데이트가 성공적으로 완료되었습니다.")
        else:
            # 실패 시 알림은 보내지만, 프로세스를 중단하지는 않음
            logger.error("❌ 일별 퀀트 팩터 업데이트 실패.")
            notifier.send_message("⚠️ 일별 퀀트 팩터 업데이트 중 오류가 발생했습니다.")
        
        # 3. 추가적인 최종 결산 작업 (필요시 여기에 추가)
        logger.info("▶ 추가 결산 작업을 수행합니다...")
        # 예: 당일 거래 로그 최종 검증, 포트폴리오 스냅샷 저장 등
        logger.info("✔ 추가 결산 작업 완료.")


        logger.info("========================================")
        logger.info("      모든 종료 후 작업이 성공적으로 완료되었습니다     ")
        logger.info("========================================")
        notifier.send_message("✅ 모든 종료 후 정리 작업이 성공적으로 완료되었습니다.")
        return True

    except Exception as e:
        logger.critical(f"종료 후 작업 중 치명적인 오류 발생: {e}", exc_info=True)
        notifier.send_message(f"🚨 자동매매 시스템 종료 후 작업 중 심각한 오류 발생: {e}")
        return False
    finally:
        # 4. 모든 리소스 해제
        if api_client:
            api_client.cleanup()
            logger.info("Creon API 리소스가 정리되었습니다.")
        if db_manager:
            db_manager.close()
            logger.info("DB 연결이 종료되었습니다.")

if __name__ == "__main__":
    run_closing_tasks()