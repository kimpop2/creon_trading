# tests/test_live_factors.py

import logging
import pandas as pd
from datetime import date, timedelta
import os
import sys
import time

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 모듈 import
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.data_manager import DataManager

def run_live_factor_update_test():
    """
    실시간 팩터 업데이트 기능을 검증하는 스크립트.
    """
    logging.info("===== 실시간 팩터 업데이트 통합 테스트 시작 =====")
    
    # --- 1. 테스트 환경 설정 ---
    test_stock_code = 'A005930'  # 테스트용 종목: 삼성전자
    today = date.today()
    start_date = today - timedelta(days=15)

    manager = None
    try:
        api = CreonAPIClient()
        db = DBManager()
        manager = DataManager(api_client=api, db_manager=db)
        
        # [수정] 테스트 스크립트 내부에 data_store 딕셔너리를 직접 생성하여 관리
        data_store = {'daily': {}}
        
        # --- 2. 기준 데이터 로드 ---
        # cache_daily_data가 반환한 DataFrame을 data_store에 저장
        data_store['daily'][test_stock_code] = manager.cache_daily_data(
            stock_code=test_stock_code,
            from_date=start_date,
            to_date=today
        )
        
        logging.info(f"'{test_stock_code}'의 기준 데이터 로드 완료.")
        logging.info("업데이트 전 마지막 데이터:")
        print(data_store['daily'][test_stock_code].tail(1))
        print("-" * 50)

        # --- 3. 실시간 팩터 조회 ---
        logging.info("Creon API를 통해 실시간 팩터 데이터를 조회합니다...")
        realtime_factors = manager.api_client.get_market_eye_datas([test_stock_code])
        
        if not realtime_factors or test_stock_code not in realtime_factors:
            logging.error("실시간 팩터 데이터 조회에 실패했습니다.")
            return

        live_data = realtime_factors[test_stock_code]
        logging.info(f"실시간 조회 결과 (일부): 체결강도={live_data.get('trading_intensity')}, 예상체결가={live_data.get('expected_price')}")
        print("-" * 50)
        
        # --- 4. 실시간 데이터로 업데이트 (실패 예상) ---
        logging.info("신규 메서드 'update_today_data'를 호출하여 데이터스토어를 업데이트합니다.")
        
        # [수정] 기존 DataFrame과 실시간 데이터를 인자로 넘겨주고, 업데이트된 DataFrame을 반환받음
        updated_df = manager.update_today_data(
            existing_df=data_store['daily'][test_stock_code], 
            live_data=live_data
        )
        # [수정] 반환받은 DataFrame으로 data_store를 갱신
        data_store['daily'][test_stock_code] = updated_df

        # --- 5. 결과 확인 ---
        logging.info("업데이트 후 마지막 데이터:")
        print(data_store['daily'][test_stock_code].tail(1))
        
        # 간단한 검증
        last_row = data_store['daily'][test_stock_code].iloc[-1]
        if last_row.name.date() == today and last_row['trading_intensity'] == live_data.get('trading_intensity'):
            logging.info("✅ 검증 성공: 오늘 날짜의 '체결강도' 값이 실시간으로 업데이트되었습니다.")
        else:
            logging.error("❌ 검증 실패: 데이터가 실시간으로 업데이트되지 않았습니다.")

    except Exception as e:
        logging.error(f"테스트 중 오류 발생: {e}", exc_info=True)
    finally:
        if manager:
            manager.close()
        logging.info("===== 테스트 종료 =====")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
    run_live_factor_update_test()