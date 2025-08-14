# setup/startup_process.py

import logging
from datetime import date
import sys
import os
import pandas as pd

# 프로젝트 루트 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Manager 및 API 클래스 임포트
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.data_manager import DataManager
# from util.notifier import TelegramNotifier # (구현 필요)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('StartupProcess')

def run_startup_process():
    """
    장 시작 전, 거래 시스템을 준비하는 전체 프로세스를 실행합니다.
    """
    logger.info("========== 🚀 거래 시스템 준비 프로세스 시작 ==========")
    api_client = None
    db_manager = None
    try:
        # --- 1. 필수 모듈 초기화 ---
        logger.info("[1/3] API 및 DB Manager 초기화...")
        api_client = CreonAPIClient()
        if not api_client.is_connected():
            raise ConnectionError("Creon API 연결에 실패했습니다.")
        
        db_manager = DBManager()
        data_manager = DataManager(api_client, db_manager)
        logger.info("초기화 완료.")

        # --- 2. 계좌 상태 확인 ---
        logger.info("[2/3] 계좌 상태 확인 (예수금, 잔고, 미체결)...")
        # 아래 API 메서드들은 CreonAPIClient에 구현이 필요합니다. (가상 메서드)
        cash_balance = api_client.get_cash_balance()
        position_df = api_client.get_account_positions()
        unfilled_orders_df = api_client.get_unfilled_orders()

        logger.info(f"  - 예수금: {cash_balance:,.0f}원")
        logger.info(f"  - 보유 종목 수: {len(position_df)}개")
        logger.info(f"  - 미체결 주문 수: {len(unfilled_orders_df)}건")
        
        # --- 4. 준비 완료 알림 ---
        logger.info("[3/3] 사용자에게 준비 완료 알림 전송...")
        # notifier = TelegramNotifier() # 알림 클래스 초기화
        today = date.today()
        message = (
            f"✅ **자동매매 시스템 시작**\n"
            f"--------------------\n"
            f"**- 날짜:** {today}\n"
            f"**- 예수금:** {cash_balance:,.0f}원\n"
            f"**- 보유종목:** {len(position_df)}개\n"
            f"**- 미체결:** {len(unfilled_orders_df)}건\n"
            f"--------------------\n"
            f"모든 준비를 마치고 거래 시작을 대기합니다."
        )
        # notifier.send_message(message)
        logger.info("알림 전송 완료 (가상).")

    except Exception as e:
        logger.critical(f"거래 시스템 준비 중 치명적인 오류 발생: {e}", exc_info=True)
        # notifier.send_message(f"🚨 **시스템 시작 실패**\n오류: {e}")
    finally:
        if db_manager:
            db_manager.close()
        logger.info("========== ✅ 거래 시스템 준비 프로세스 종료 ==========")

if __name__ == '__main__':
    run_startup_process()