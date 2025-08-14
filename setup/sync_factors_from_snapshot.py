# setup/sync_factors_from_snapshot.py

import logging
from datetime import date
import sys
import os

# 프로젝트 루트 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from manager.db_manager import DBManager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SyncFactors')

def sync_daily_factors_from_snapshot():
    """
    [최종 수정 로직]
    1. daily_factors에 데이터가 있는 마지막 날짜를 조회합니다.
    2. 해당 날짜에 데이터가 있는 종목 목록을 가져옵니다.
    3. stock_info에서 regdate가 위 날짜와 일치하는 종목들의 스냅샷을 가져옵니다.
    4. 가져온 스냅샷 데이터로 daily_factors를 업데이트합니다.
    """
    logger.info("========== 팩터 스냅샷 동기화 시작 (날짜 기준 동적 로직) ==========")
    db_manager = None
    try:
        db_manager = DBManager()

        # 1. daily_factors에서 가장 최근 날짜를 가져옵니다.
        target_date = db_manager.fetch_latest_date_from_factors()
        if not target_date:
            logger.warning("daily_factors에 데이터가 없어 동기화할 대상 날짜를 찾을 수 없습니다.")
            return

        logger.info(f"동기화 대상 날짜: {target_date}")

        # 2. 해당 날짜의 대상 종목 코드를 조회합니다.
        target_stock_codes = db_manager.fetch_factor_stocks_by_date(target_date)
        if not target_stock_codes:
            logger.warning(f"{target_date}에 해당하는 종목이 daily_factors에 없습니다.")
            return

        logger.info(f"총 {len(target_stock_codes)}개 종목이 동기화 대상으로 확인되었습니다.")

        # 3. stock_info에서 해당 날짜(regdate 기준)의 스냅샷 데이터를 가져옵니다.
        logger.info(f"stock_info 테이블에서 {target_date} 스냅샷 데이터를 로드합니다.")
        stock_info_df = db_manager.fetch_stock_info(
            stock_codes=target_stock_codes,
            target_date=target_date  # regdate의 날짜 부분을 필터링
        )
        
        if stock_info_df.empty:
            logger.warning(f"{target_date} 날짜의 스냅샷 정보를 stock_info에서 찾을 수 없습니다.")
            return

        # 4. daily_factors에 업데이트할 데이터 준비
        columns_to_sync = [
            'stock_code', 'per', 'pbr', 'dividend_yield', 'foreigner_net_buy', 
            'institution_net_buy', 'trading_value', 'short_volume', 'credit_ratio', 
            'beta_coefficient'
        ]
        valid_columns = [col for col in columns_to_sync if col in stock_info_df.columns]
        
        update_df = stock_info_df[valid_columns].copy()
        update_df.fillna(0, inplace=True)
        update_df['date'] = target_date

        data_to_update = update_df.to_dict('records')

        # 5. DB 업데이트 실행
        logger.info(f"{len(data_to_update)}개 종목의 daily_factors를 업데이트합니다.")
        success = db_manager.update_daily_factors_from_snapshot(data_to_update)

        if success:
            logger.info("✓ daily_factors 스냅샷 동기화가 성공적으로 완료되었습니다.")
        else:
            logger.error("✗ daily_factors 스냅샷 동기화에 실패했습니다.")

    except Exception as e:
        logger.critical(f"스냅샷 동기화 중 오류 발생: {e}", exc_info=True)
    finally:
        if db_manager:
            db_manager.close()
        logger.info("========== 팩터 스냅샷 동기화 종료 ==========")

if __name__ == '__main__':
    sync_daily_factors_from_snapshot()