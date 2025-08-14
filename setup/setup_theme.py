# setup/setup_theme.py
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

def find_strong_stocks_and_save_theme():
    """
    당일 등락률 상위 종목 중 특정 조건을 만족하는 종목을 daily_theme에 등록합니다.
    """
    logger.info("=== 등락률 상위 기반 daily_theme 업데이트 시작 ===")
    api_client = None
    db_manager = None
    try:
        api_client = CreonAPIClient()
        if not api_client.is_connected():
            raise ConnectionError("Creon API 연결 실패")
        
        db_manager = DBManager()
        setup_manager = SetupManager(api_client, db_manager)
        # 1. 코스피/코스닥 지수 등락률 조회
        #kospi_rate = api_client.get_index_change_rate("U001") # 코스피
        
        # 2. 당일 등락률 상위 종목 조회 (API에 get_top_movers 구현 필요)
        top_movers_df = api_client.get_top_movers(market='all', top_n=200)
        if top_movers_df.empty:
            logger.warning("등락률 상위 종목 조회 결과가 없습니다.")
            return

        # 3. 조건 필터링
        # - 주가 1000원 이상
        # - 코스피 등락률 + 7% 이상 상승
        strong_stocks = top_movers_df[
            (top_movers_df['current_price'] >= 1000) &
            (top_movers_df['change_rate'] > 7)
        ]

        if strong_stocks.empty:
            logger.info("조건을 만족하는 강세 종목이 없습니다.")
            return
            
        logger.info(f"총 {len(strong_stocks)}개의 강세 종목 발견. daily_theme에 저장합니다.")

        # 4. daily_theme에 저장할 데이터 준비
        data_to_save = []
        for index, row in strong_stocks.iterrows():
            # 가장 최근 reason 조회
            latest_reason = setup_manager.get_latest_theme_reason(row['stock_code'])
            
            data_to_save.append({
                'date': date.today(),
                'market': '정규장',
                'stock_code': row['stock_code'],
                'stock_name': row['stock_name'],
                'rate': row['change_rate'],
                'amount': row.get('trading_value', 0) , # 백만 단위
                'reason': latest_reason or "당일 등락률 상위 조건 충족",
            })
        
        # 5. DB에 저장
        if setup_manager.save_daily_theme(data_to_save):
            logger.info(f"{len(data_to_save)}건의 데이터를 daily_theme에 저장했습니다.")
        else:
            logger.error("daily_theme 저장 실패.")

    except Exception as e:
        logger.critical(f"등락률 상위 종목 처리 중 오류 발생: {e}", exc_info=True)
    finally:

        if db_manager:
            db_manager.close()
        logger.info("=== 등락률 상위 기반 daily_theme 업데이트 종료 ===")

if __name__ == '__main__':
    find_strong_stocks_and_save_theme()
