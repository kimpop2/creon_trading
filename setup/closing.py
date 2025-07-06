# closing.py (수정된 부분)

import logging
import sys
import os
from datetime import date, datetime

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from manager.db_manager import DBManager
from util.notifier import Notifier
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, LOG_LEVEL, LOG_FILE_CLOSING # LOG_FILE_CLOSING 추가

# 로거 설정
def setup_closing_logging():
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_CLOSING, encoding='utf-8'), # 종료 스크립트 전용 로그 파일
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger('urllib3').setLevel(logging.WARNING)

setup_closing_logging() # 로깅 설정 호출
logger = logging.getLogger(__name__)

def run_closing_tasks():
    logger.info("========================================")
    logger.info("      자동매매 시스템 종료 후 작업 시작     ")
    logger.info("========================================")

    notifier = Notifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    notifier.send_message("🧹 자동매매 시스템 종료 후 정리 작업이 시작됩니다.")

    db_manager = None
    try:
        db_manager = DBManager()
        if not db_manager.get_db_connection():
            logger.error("DB 연결 실패. 종료 후 작업을 진행할 수 없습니다.")
            notifier.send_message("❌ DB 연결 실패. 종료 후 작업 중단.")
            return False
        logger.info("DB 연결 성공.")

        # 1. 최종 데이터 동기화 및 결산 (예: 당일 거래 로그 최종 저장 등)
        logger.info("최종 거래 데이터 동기화 및 결산 시작...")
        # 여기에 당일 미체결 주문 처리, 최종 포트폴리오 스냅샷 저장 등 로직 추가
        # 예: trading_manager.finalize_daily_operations(datetime.now().date())
        logger.info("최종 거래 데이터 동기화 및 결산 완료.")

        # 2. 불필요한 리소스 해제 (예: 특정 API 연결 정리)
        logger.info("불필요한 리소스 해제...")
        # 특정 서비스가 계속 실행 중인 경우 여기서 종료 명령을 보낼 수 있습니다.
        logger.info("리소스 해제 완료.")

        logger.info("========================================")
        logger.info("      자동매매 시스템 종료 후 작업 완료     ")
        logger.info("========================================")
        notifier.send_message("✅ 자동매매 시스템 종료 후 정리 작업이 성공적으로 완료되었습니다.")
        return True

    except Exception as e:
        logger.critical(f"종료 후 작업 중 치명적인 오류 발생: {e}", exc_info=True)
        notifier.send_message(f"🚨 자동매매 시스템 종료 후 작업 중 오류 발생: {e}")
        return False
    finally:
        if db_manager:
            db_manager.close() # db_manager의 close 메서드 호출

if __name__ == "__main__":
    run_closing_tasks()