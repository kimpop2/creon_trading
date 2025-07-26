import os
import sys
from typing import Dict, List, Any
import pandas as pd
import numpy as np # 수치 계산을 위해 numpy 임포트
from datetime import date, timedelta
import logging


# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from manager.setup_manager import SetupManager
from manager.db_manager import DBManager
from api.creon_api import CreonAPIClient

# --- 로거 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('backfill_factors')

def run_backfill():
    """
    유니버스에 등록된 모든 종목에 대해 지정된 기간의 daily_factors 데이터를 채웁니다.
    아래 값들은 장이 끝난 후 
    program_net_buy: get_market_trading_volume_by_date 함수가 이 값을 제공하지 않습니다.
    trading_intensity: 장중에만 의미 있는 실시간 지표입니다.
    credit_ratio: pykrx에 일자별 신용잔고율을 조회하는 기능이 없습니다.
    psr: 계산에 필요한 주당매출액(SPS)을 pykrx가 제공하지 않습니다.
    q_revenue_growth_rate, q_op_income_growth_rate: 특정 시점의 분기 성장률은 조회하기 어렵습니다.
    """
    # --- 데이터 채우기 기간 설정 ---
    # 예: 5년 전부터 오늘까지
    END_DATE = date.today()- timedelta(days=1) # 어제까지만
    START_DATE = END_DATE - timedelta(days=1*365+20)
    
    logger.info(f"=== Daily Factors 데이터 백필링 시작 ===")
    logger.info(f"대상 기간: {START_DATE} ~ {END_DATE}")

    try:
        api_client = CreonAPIClient()
        db_manager = DBManager()
        manager = SetupManager(api_client, db_manager)
        # 1. DB에서 전체 유니버스 종목 코드 가져오기
        universe_codes = manager.get_universe_codes()
        if not universe_codes:
            logger.warning("유니버스에 등록된 종목이 없습니다. 백필링을 종료합니다.")
            return
            
        total_codes = len(universe_codes)
        logger.info(f"총 {total_codes}개 종목에 대한 백필링을 시작합니다.")

        # 2. 각 종목에 대해 팩터 데이터 캐싱 실행
        for i, stock_code in enumerate(universe_codes):
            logger.info(f"--- 종목 처리 중 ({i+1}/{total_codes}): [{stock_code}] ---")
            try:
                # cache_factors는 DB에 없는 날짜만 알아서 조회하고 저장합니다.
                manager.cache_factors(START_DATE, END_DATE, stock_code)
                logger.info(f"--- [{stock_code}] 처리 완료 ---")
            except Exception as e:
                logger.error(f"[{stock_code}] 처리 중 오류 발생: {e}", exc_info=False)
        
        logger.info("모든 유니버스 종목에 대한 백필링 작업이 완료되었습니다.")

    except Exception as e:
        logger.critical(f"백필링 프로세스 중 심각한 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    run_backfill()