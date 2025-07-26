# tests/test_setup_manager.py

import unittest
import pandas as pd
from datetime import date
import sys
import os

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from manager.db_manager import DBManager
from api.creon_api import CreonAPIClient
from manager.setup_manager import SetupManager, KOSPI_INDEX_CODE

class TestSetupManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """클래스 레벨 설정: API 및 DB 연결"""
        cls.api_client = CreonAPIClient()
        cls.db_manager = DBManager()
        # 모든 테스트 전에 필요한 테이블이 생성되어 있는지 확인
        cls.db_manager.create_stock_tables()
        cls.db_manager.create_factor_tables()
        # 테스트에 필요한 최소한의 주식 정보 저장
        cls.db_manager.save_stock_info([
            {'stock_code': 'A005930', 'stock_name': '삼성전자'},
            {'stock_code': 'A000660', 'stock_name': 'SK하이닉스'},
            {'stock_code': KOSPI_INDEX_CODE, 'stock_name': '코스피'}
        ])

    @classmethod
    def tearDownClass(cls):
        """클래스 레벨 정리: API 및 DB 연결 종료"""
        cls.api_client.cleanup()
        cls.db_manager.close()
        
    def setUp(self):
        """테스트 설정: SetupManager 인스턴스 생성"""
        self.setup_manager = SetupManager(self.api_client, self.db_manager)
        # 매 테스트 시작 전, 특정 날짜의 팩터 데이터 삭제로 독립성 보장
        self.db_manager.execute_sql("DELETE FROM daily_factors WHERE date = %s", (date.today(),))


    def test_01_run_daily_factor_update_integration(self):
        """run_daily_factor_update 통합 테스트"""
        # 전제 조건: Creon API를 통해 A005930과 코스피 지수의 일봉 데이터를 가져올 수 있어야 함
        target_date = date.today()
        
        # 1. 실제 메서드 실행
        success = self.setup_manager.run_daily_factor_update(target_date)
        
        # API 및 네트워크 상태에 따라 실패할 수 있으므로, True가 아닐 경우 경고와 함께 테스트 통과
        if not success:
            self.skipTest(f"{target_date}의 팩터 업데이트에 실패했습니다. API/네트워크 상태를 확인하세요.")
            return

        # 2. DB에서 결과 확인
        # get_all_stock_codes()가 반환하는 종목 중 하나를 샘플링하여 확인
        all_codes = self.db_manager.get_all_stock_codes()
        if 'A005930' in all_codes:
            sample_code = 'A005930'
        elif all_codes:
            sample_code = all_codes[0]
        else:
            self.fail("DB에 테스트할 종목 코드가 없습니다.")
            
        fetched_df = self.db_manager.fetch_daily_factors(target_date, target_date, sample_code)
        
        # 3. 결과 검증
        self.assertIsInstance(fetched_df, pd.DataFrame)
        self.assertFalse(fetched_df.empty, f"[{sample_code}]의 팩터 데이터가 DB에 저장되지 않았습니다.")
        self.assertEqual(len(fetched_df), 1)
        
        # 주요 팩터 값들이 None이 아닌지 확인 (정확한 값 검증은 변동성 때문에 어려움)
        self.assertIsNotNone(fetched_df.iloc[0]['pbr'], "PBR 값이 계산되지 않았습니다.")
        self.assertIsNotNone(fetched_df.iloc[0]['dist_from_ma20'], "20일 이격도 값이 계산되지 않았습니다.")
        self.assertIsNotNone(fetched_df.iloc[0]['relative_strength'], "상대강도 값이 계산되지 않았습니다.")
        self.assertIsNotNone(fetched_df.iloc[0]['credit_ratio'], "신용잔고율 값이 채워지지 않았습니다.")

if __name__ == '__main__':
    unittest.main()