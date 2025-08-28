# tests/test_data_manager_factors.py

import unittest
import logging
import pandas as pd
from datetime import date
import os
import sys

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 테스트 대상 모듈 import
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.data_manager import DataManager

class TestDataManagerFactors(unittest.TestCase):
    """DataManager의 실제 DB/API 연동 기능에 대한 통합 테스트"""
    
    manager: DataManager = None
    test_stock_code: str = 'A499790'
    start_date: date = date(2025, 8, 8)
    end_date: date = date(2025, 8, 20)
    
    # 클래스 레벨에서 상태를 저장할 변수
    merged_df: pd.DataFrame = None

    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행"""
        logging.info("===== DataManager 통합 테스트 시작 =====")
        try:
            cls.api = CreonAPIClient()
            cls.db = DBManager()
            cls.manager = DataManager(api_client=cls.api, db_manager=cls.db)
            logging.info("CreonAPI, DBManager, DataManager 초기화 완료")
        except Exception as e:
            logging.critical(f"테스트 환경 설정 실패: {e}", exc_info=True)
            raise ConnectionError("테스트 환경 설정에 실패했습니다. Creon HTS 또는 DB 연결을 확인하세요.")

    def test_01_prerequisite_data_check(self):
        """1단계: DB에서 OHLCV와 팩터 데이터를 개별적으로 조회할 수 있는지 확인"""
        logging.info("=== 1. 선행조건 확인: DB 데이터 개별 조회 테스트 ===")
        
        price_df = self.manager.db_manager.fetch_daily_price(self.test_stock_code, self.start_date, self.end_date)
        logging.info(f"[{self.test_stock_code}] daily_price 조회 결과: {price_df.shape[0]} 행")
        self.assertFalse(price_df.empty, "테스트를 위한 daily_price 데이터가 DB에 존재하지 않습니다.")
        self.assertIn('close', price_df.columns)

        factors_df = self.manager.db_manager.fetch_daily_factors(self.test_stock_code, self.start_date, self.end_date)
        logging.info(f"[{self.test_stock_code}] daily_factors 조회 결과: {factors_df.shape[0]} 행")
        self.assertFalse(factors_df.empty, "테스트를 위한 daily_factors 데이터가 DB에 존재하지 않습니다.")
        self.assertIn('per', factors_df.columns)
        
    def test_02_run_cache_daily_data(self):
        """2단계: cache_daily_data 메서드 실행 및 결과 확인"""
        logging.info("=== 2. 핵심 기능 테스트: cache_daily_data 메서드 실행 ===")
        
        # [수정] 메서드가 반환하는 DataFrame을 직접 받습니다.
        result_df = self.manager.cache_daily_data(self.test_stock_code, self.start_date, self.end_date)
        
        # [수정] 반환된 DataFrame이 유효한지 직접 검증합니다.
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertFalse(result_df.empty, "cache_daily_data 실행 후 결과 데이터프레임이 비어있습니다.")
        
        logging.info(f"cache_daily_data 실행 결과: {result_df.shape[0]} 행, {result_df.shape[1]} 열")
        logging.info(f"결과 컬럼: {result_df.columns.tolist()}")

        # 다음 테스트에서 사용하기 위해 결과를 클래스 변수에 저장
        self.__class__.merged_df = result_df

    def test_03_verify_merged_columns(self):
        """3단계: 병합된 데이터프레임에 OHLCV와 팩터 컬럼이 모두 포함되었는지 검증"""
        logging.info("=== 3. 결과 검증: 병합된 컬럼 확인 ===")
        
        self.assertIsNotNone(self.merged_df, "이전 테스트에서 병합된 데이터가 생성되지 않았습니다.")
        
        # OHLCV 대표 컬럼 확인
        self.assertIn('close', self.merged_df.columns)
        self.assertIn('volume', self.merged_df.columns)
        
        # 팩터 대표 컬럼 확인
        self.assertIn('per', self.merged_df.columns)
        self.assertIn('pbr', self.merged_df.columns)
        self.assertIn('foreigner_net_buy', self.merged_df.columns)
        
        logging.info("OHLCV 및 팩터의 대표 컬럼들이 최종 데이터프레임에 모두 포함된 것을 확인했습니다.")
        logging.info(f"데이터 샘플 (상위 3행):\n{self.merged_df.head(3)}")

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시 한 번만 실행"""
        if cls.manager:
            cls.manager.close()
        logging.info("===== DataManager 통합 테스트 종료 =====")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
    unittest.main(verbosity=2)