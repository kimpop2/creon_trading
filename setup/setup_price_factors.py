# setup/setup_price.py
import logging
import sys
import os
from datetime import date, timedelta

# 프로젝트 루트 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.setup_manager import SetupManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_price_for_recent_themes(days=30):
    """
    최근 N일간 daily_theme에 등록된 종목들의 일봉 데이터를 수집합니다.
    """
    logger.info(f"=== 최근 {days}일 테마 종목 일봉 데이터 수집 시작 ===")
    api_client = None
    db_manager = None
    try:
        api_client = CreonAPIClient()
        if not api_client.is_connected():
            raise ConnectionError("Creon API 연결 실패")
        
        db_manager = DBManager()
        setup_manager = SetupManager(api_client, db_manager)

        # 1. 최근 N일간의 대상 종목 코드 조회
        target_codes = db_manager.fetch_recent_theme_stocks(days=30)
        if not target_codes:
            logger.info("업데이트할 최근 테마 종목이 없습니다.")
            return

        logger.info(f"총 {len(target_codes)}개 종목의 일봉 데이터를 수집합니다.")
        
        # 2. 각 종목에 대해 데이터 업데이트
        today = date.today()
        start_date = today - timedelta(days=days + 5) # 여유 기간을 두고 조회
        
        for i, stock_code in enumerate(target_codes):
            logger.info(f"  - 처리 중 ({i+1}/{len(target_codes)}): {stock_code}")
            setup_manager.cache_daily_data(stock_code, start_date, today)  # 일봉정보 팩터
        logger.info("일봉 데이터 수집 완료.")

    except Exception as e:
        logger.critical(f"일봉 데이터 수집 중 오류 발생: {e}", exc_info=True)
    finally:
        if db_manager:
            db_manager.close()
        logger.info(f"=== 최근 {days}일 테마 종목 일봉 데이터 수집 종료 ===")

if __name__ == '__main__':
    update_price_for_recent_themes(days=30)
