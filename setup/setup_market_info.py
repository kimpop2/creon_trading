# setup/setup_market_info.py
import logging
import sys
import os
from datetime import date

# 프로젝트 루트 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.setup_manager import SetupManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_all_info_setup():
    """
    종목 기본 정보, 시장 캘린더, MarketEye 상세 정보를 순차적으로 업데이트하는 메인 함수.
    """
    logger.info("========== 전체 종목 정보 셋업 시작 ==========")
    api_client = None
    db_manager = None
    try:
        api_client = CreonAPIClient()
        if not api_client.is_connected():
            raise ConnectionError("Creon API 연결 실패")
        
        db_manager = DBManager()
        setup_manager = SetupManager(api_client, db_manager)
        
        # --- 단계 1: 모든 종목 기본 정보 업데이트 (stock_info 테이블) ---
        logger.info("--- [1/3] 종목 기본 정보 업데이트 시작 ---")
        if setup_manager.update_all_stock_info():
            logger.info("✓ 종목 기본 정보 업데이트 완료")
        else:
            logger.error("✗ 종목 기본 정보 업데이트 실패")
            
        # --- ▼ [핵심 수정] 10년치 기간 설정 및 루프 실행 ---
        current_year = date.today().year
        start_year = current_year - 9        
        logger.info(f"--- [2/3] {start_year}년 ~ {current_year}년 (10년치) 주식시장 캘린더 업데이트 시작 ---")

        # 전체 성공 여부를 추적하기 위한 플래그
        all_success = True

        # 10년 전부터 올해까지 반복
        for year_to_update in range(start_year, current_year + 1):
            logger.info(f" -> {year_to_update}년 캘린더 업데이트 중...")
            
            # 각 연도별로 캘린더 업데이트 함수 호출
            if not setup_manager.update_market_calendar(year=year_to_update):
                logger.warning(f"   ✗ {year_to_update}년 캘린더 업데이트 실패")
                all_success = False # 한 번이라도 실패하면 False로 설정

        # --- 최종 결과 로깅 ---
        if all_success:
            logger.info("✓ 주식시장 캘린더 전체(10년치) 업데이트 완료")
        else:
            logger.error("✗ 주식시장 캘린더 업데이트 중 일부 실패 항목이 있습니다.")

        # --- 단계 3: MarketEye를 이용한 상세 정보 업데이트 ---
        logger.info("--- [3/3] MarketEye를 이용한 상세 정보 업데이트 시작 ---")
        if setup_manager.update_stock_info_with_marketeye():
            logger.info("✓ MarketEye 상세 정보 업데이트 완료")
        else:
            logger.error("✗ MarketEye 상세 정보 업데이트 실패")

    except Exception as e:
        logger.critical(f"전체 종목 정보 셋업 중 오류 발생: {e}", exc_info=True)
    finally:
        if db_manager:
            db_manager.close()
        logger.info("========== 전체 종목 정보 셋업 종료 ==========")

if __name__ == '__main__':
    run_all_info_setup()
