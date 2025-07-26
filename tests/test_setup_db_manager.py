# tests/test_setup_db_manager.py

import unittest
import pandas as pd
from datetime import date
import sys
import os

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from manager.db_manager import DBManager

class TestSetupDBManager(unittest.TestCase):
    def setUp(self):
        """테스트 설정: DBManager 인스턴스 생성 및 테스트 데이터 준비"""
        self.db_manager = DBManager()
        
        # 테스트 데이터 정의
        self.test_factor_data_1 = {
            'date': date(2025, 7, 25),
            'stock_code': 'A005930',
            'foreigner_net_buy': 100000,
            'institution_net_buy': 50000,
            'program_net_buy': 75000,
            'trading_intensity': 120.5,
            'credit_ratio': 5.5,
            'short_volume': 12345,
            'trading_value': 100000000000,
            'per': 15.1, 'pbr': 1.8, 'psr': 1.5,
            'dividend_yield': 2.1,
            'relative_strength': 1.2,
            'beta_coefficient': 1.1,
            'days_since_52w_high': 10,
            'dist_from_ma20': 2.5,
            'historical_volatility_20d': 1.5,
            'q_revenue_growth_rate': 10.5,
            'q_op_income_growth_rate': 12.5
        }
        self.test_factor_data_2 = {
            'date': date(2025, 7, 25),
            'stock_code': 'A000660',
            'foreigner_net_buy': -20000,
            'institution_net_buy': 80000,
            'program_net_buy': 60000,
            'trading_intensity': 98.2,
            'credit_ratio': 3.1,
            'short_volume': 5432,
            'trading_value': 80000000000,
            'per': 12.3, 'pbr': 1.4, 'psr': 1.2,
            'dividend_yield': 2.5,
            'relative_strength': -0.5,
            'beta_coefficient': 1.2,
            'days_since_52w_high': 25,
            'dist_from_ma20': -1.8,
            'historical_volatility_20d': 1.8,
            'q_revenue_growth_rate': 8.2,
            'q_op_income_growth_rate': 9.1
        }
        self.test_factor_data_1_updated = self.test_factor_data_1.copy()
        self.test_factor_data_1_updated['pbr'] = 1.9
        self.test_factor_data_1_updated['per'] = 15.5

        # 테스트 전 팩터 관련 테이블들을 삭제하고 다시 생성하여 클린 상태 보장
        # self.db_manager.drop_factor_tables() # drop_factor_tables 메서드 추가 후 활성화
        self.db_manager.create_factor_tables()

    def tearDown(self):
        """테스트 정리: DB 연결 종료"""
        self.db_manager.close()

    def test_01_create_and_drop_factor_tables(self):
        """팩터 관련 테이블 생성 및 삭제 테스트"""
        # setUp에서 이미 create를 수행하므로 존재 여부만 확인
        self.assertTrue(self.db_manager.check_table_exist('daily_factors'))

        # drop 시도 후 존재하지 않음을 확인 (drop_factor_tables 구현 필요)
        # result = self.db_manager.drop_factor_tables()
        # self.assertTrue(result)
        # self.assertFalse(self.db_manager.check_table_exist('daily_factors'))
        
        # 다시 생성하여 다음 테스트에 영향 없도록 복구
        self.db_manager.create_factor_tables()

    def test_02_save_and_fetch_daily_factors(self):
        """일별 팩터 저장 및 조회 테스트"""
        # 1. 데이터 저장
        save_result = self.db_manager.save_daily_factors([self.test_factor_data_1, self.test_factor_data_2])
        self.assertTrue(save_result)

        # 2. 데이터 조회
        fetched_df = self.db_manager.fetch_daily_factors(date(2025, 7, 25), date(2025, 7, 25), 'A005930')
        self.assertIsInstance(fetched_df, pd.DataFrame)
        self.assertEqual(len(fetched_df), 1)
        self.assertEqual(fetched_df.iloc[0]['pbr'], self.test_factor_data_1['pbr'])
        self.assertEqual(fetched_df.iloc[0]['stock_code'], 'A005930')

        # 3. 데이터 업데이트 (ON DUPLICATE KEY UPDATE)
        update_result = self.db_manager.save_daily_factors([self.test_factor_data_1_updated])
        self.assertTrue(update_result)
        
        # 4. 업데이트된 데이터 확인
        fetched_updated_df = self.db_manager.fetch_daily_factors(date(2025, 7, 25), date(2025, 7, 25), 'A005930')
        self.assertEqual(len(fetched_updated_df), 1)
        self.assertEqual(fetched_updated_df.iloc[0]['pbr'], self.test_factor_data_1_updated['pbr'])
        self.assertEqual(fetched_updated_df.iloc[0]['per'], self.test_factor_data_1_updated['per'])

        # 5. 없는 데이터 조회
        fetched_empty_df = self.db_manager.fetch_daily_factors(date(2025, 7, 25), date(2025, 7, 25), 'A999999')
        self.assertTrue(fetched_empty_df.empty)

if __name__ == '__main__':
    unittest.main()