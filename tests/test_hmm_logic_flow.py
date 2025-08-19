# tests/test_hmm_logic_flow.py

import unittest
import pandas as pd
import numpy as np
import sys
import os
from datetime import date, timedelta
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from analyzer.hmm_model import RegimeAnalysisModel
from analyzer import train_and_save_hmm

class TestHmmLogicFlow(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.db_manager = DBManager()
        cls.backtest_manager = BacktestManager(None, cls.db_manager)
        
    @classmethod
    def tearDownClass(cls):
        cls.db_manager.close()

    def test_1_verify_full_model_instability(self):
        """[가설 검증] 'full' 타입 모델은 우리 데이터로 학습 시 불안정하여 에러가 발생함을 확인합니다."""
        print("\n--- [Test 1] 'full' 타입 모델 학습 불안정성 테스트 ---")
        training_data = self.backtest_manager.get_market_data_for_hmm(date.today() - timedelta(days=100), date.today())
        
        # 'full' 타입 모델 학습 시 ValueError가 발생하는 것을 기대함
        with self.assertRaises(ValueError) as context:
            old_model = RegimeAnalysisModel(n_states=4, covariance_type="full")
            old_model.fit(training_data)
        
        self.assertIn("positive-definite", str(context.exception))
        print(f"--- [Test 1] 성공: 예상대로 'full' 타입 학습 시 'positive-definite' 오류가 발생했습니다. ---")

    def test_2_verify_diag_model_save_and_load(self):
        """[해결 검증] 'diag' 타입 모델 생성 및 로드 과정을 상세히 디버깅합니다."""
        print("\n--- [Test 2] 'diag' 타입 모델 저장 및 로드 디버깅 테스트 ---")
        MODEL_NAME = "New_Diag_Model_For_Debug"

        self.db_manager.execute_sql("DELETE FROM hmm_models WHERE model_name = %s", (MODEL_NAME,))
        train_and_save_hmm.train_and_save_production_model(model_name_override=MODEL_NAME, days=100)
        
        model_info = self.db_manager.fetch_hmm_model_by_name(MODEL_NAME)
        self.assertIsNotNone(model_info, "'diag' 모델이 DB에 저장되지 않았습니다.")
        
        model_params_from_db = model_info['model_params']
        
        # --- ▼ [핵심 디버깅 코드] ---
        print("\n--- DB에서 불러온 파라미터 직접 확인 ---")
        cov_type = model_params_from_db.get('covariance_type')
        covars_data = np.array(model_params_from_db.get('covars_'))
        
        print(f"DB에서 읽은 covariance_type: {cov_type}")
        print(f"DB에서 읽은 covars_ 데이터의 shape: {covars_data.shape}")
        print("-------------------------------------\n")
        # --- ▲ [핵심 디버깅 코드] ---

        # 이 테스트는 이제 원인 파악이 목적이므로, 에러 발생 시 실패하도록 함
        try:
            hmm_model = RegimeAnalysisModel.load_from_params(model_params_from_db)
            self.assertIsInstance(hmm_model, RegimeAnalysisModel)
            print(f"--- [Test 2] 성공: 'diag' 타입 모델을 성공적으로 불러왔습니다. ---")
        except ValueError as e:
            self.fail(f"'diag' 타입 모델을 불러오는 중 예기치 않은 ValueError가 발생했습니다: {e}")

if __name__ == '__main__':
    unittest.main()