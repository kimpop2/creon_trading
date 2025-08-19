# tests/test_regimes_workflow.py

import unittest
import pandas as pd
from datetime import date
import sys
import os

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 테스트 대상 모듈 임포트 ---
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from analyzer import train_and_save_hmm
from optimizer.regimes_optimizer import RegimesOptimizer



class TestRegimesWorkflow(unittest.TestCase):
    """
    HMM 국면 정의부터 국면별 파라미터 최적화까지의 전체 워크플로우를 테스트합니다.
    """
    
    @classmethod
    def setUpClass(cls):
        """테스트 전체에 걸쳐 한 번만 실행되며, 실제 API와 DB 연결을 설정합니다."""
        cls.api_client = CreonAPIClient()
        cls.db_manager = DBManager()
        cls.backtest_manager = BacktestManager(api_client=cls.api_client, db_manager=cls.db_manager)
        
        # [수정] 특징 이니셜을 포함하는 모델명으로 변경
        cls.test_model_name = 'EKLMNO_4s_2401-2408' 
        cls.start_date = date(2024, 7, 1)
        cls.end_date = date(2024, 8, 31)
        cls.model_id_from_db = None # 1단계에서 DB에 저장된 모델 ID를 저장할 변수

        # 테스트 시작 전, 이전 테스트 데이터를 삭제하여 독립성 보장
        model_info = cls.db_manager.fetch_hmm_model_by_name(cls.test_model_name)
        if model_info:
            model_id = model_info['model_id']
            cls.db_manager.execute_sql("DELETE FROM daily_regimes WHERE model_id = %s", (model_id,))
            cls.db_manager.execute_sql("DELETE FROM hmm_models WHERE model_id = %s", (model_id,))
            print(f"이전 테스트 모델({cls.test_model_name}, ID:{model_id}) 데이터를 삭제했습니다.")


    @classmethod
    def tearDownClass(cls):
        """테스트가 모두 끝난 후 한 번 실행됩니다."""
        cls.db_manager.close()

    def test_01_run_hmm_training_and_save_to_db(self):
        """
        [Phase 1] HMM 모델을 학습하고, 모델 및 국면 분석 결과를 DB에 저장하는지 테스트합니다.
        """
        print("\n--- Phase 1: HMM Training & DB Save Test ---")
        
        # train_and_save_hmm.run_hmm_training 함수를 호출
        success, _, _ = train_and_save_hmm.run_hmm_training(
            model_name=self.test_model_name,
            start_date=self.start_date,
            end_date=self.end_date,
            backtest_manager=self.backtest_manager
        )

        self.assertTrue(success, "HMM 모델 학습 및 저장 프로세스에 실패했습니다.")

        # --- [수정] 실제 DB에 데이터가 저장되었는지 검증 ---
        model_info_from_db = self.db_manager.fetch_hmm_model_by_name(self.test_model_name)
        self.assertIsNotNone(model_info_from_db, f"'{self.test_model_name}' 모델이 DB에서 조회되지 않습니다.")
        
        model_id = model_info_from_db['model_id']
        TestRegimesWorkflow.model_id_from_db = model_id # 클래스 변수에 저장
        
        regime_data_from_db = self.db_manager.fetch_daily_regimes(model_id)
        self.assertFalse(regime_data_from_db.empty, f"모델 ID {model_id}에 대한 국면 데이터가 DB에 저장되지 않았습니다.")
        self.assertIn('regime_id', regime_data_from_db.columns)

        print(f"--- Phase 1: Test Passed --- (Model ID: {model_id}, {len(regime_data_from_db)} days of regime data saved to DB)")


    def test_02_regime_specific_optimization(self):
        """
        [Phase 2] RegimesOptimizer가 국면별로 최적의 파라미터를 찾아내는지 테스트합니다.
        (이 테스트는 초기에 실패해야 하며, RegimesOptimizer 구현을 통해 통과시켜야 합니다.)
        """
        print("\n--- Phase 2: Regime-Specific Parameter Optimization Test ---")
        
        # 1단계에서 생성된 모델 ID가 있는지 확인
        self.assertIsNotNone(self.model_id_from_db, "선행 HMM 학습 테스트가 통과되지 않아 최적화를 진행할 수 없습니다.")
        
        # DB에서 국면 데이터 로드
        regime_map_df = self.db_manager.fetch_daily_regimes(self.model_id_from_db)

        # 1. 최적화 대상 전략과 기간 정의
        strategy_name = 'SMADaily' # 예시 전략
        optimization_start_date = self.start_date
        optimization_end_date = self.end_date

        # 2. 앞으로 만들 RegimesOptimizer 인스턴스 생성
        optimizer = RegimesOptimizer(
            backtest_manager=self.backtest_manager,
            initial_cash=10_000_000
        )

        # 3. 최적화 실행
        champion_params_by_regime = optimizer.run_optimization_for_strategy(
            strategy_name=strategy_name,
            start_date=optimization_start_date,
            end_date=optimization_end_date,
            regime_map=regime_map_df,
            model_name=self.test_model_name 
        )
        
        # 4. 결과 검증
        self.assertIsInstance(champion_params_by_regime, dict, "최적화 결과는 딕셔너리여야 합니다.")
        # HMM 상태 개수(4개)만큼 결과가 있는지 확인
        self.assertEqual(len(champion_params_by_regime), 4, "모든 국면에 대한 최적화 결과가 반환되지 않았습니다.")
        self.assertIn(0, champion_params_by_regime)
        self.assertIn(1, champion_params_by_regime)
        self.assertIn(2, champion_params_by_regime)
        self.assertIn(3, champion_params_by_regime)

        self.assertIn('strategy_params', champion_params_by_regime[0]['params'])
        self.assertIn('sharpe_ratio', champion_params_by_regime[0]['metrics'])
        
        print("--- Phase 2: Test Passed ---")
        
        # TDD 초기 단계이므로, 구현 전까지는 pass 처리
        


if __name__ == '__main__':
    unittest.main()