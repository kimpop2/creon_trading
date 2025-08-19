# tests/test_walk_forward_optimizer.py

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import date
import sys
import os

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 테스트 대상 및 필요 모듈 임포트 ---
from api.creon_api import CreonAPIClient
from optimizer.walk_forward_optimizer import WalkForwardOptimizer
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
# --- 동적 전략 로딩을 위해 모든 전략 클래스 임포트 ---
from strategies.sma_daily import SMADaily
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.breakout_daily import BreakoutDaily
from strategies.closing_bet_daily import ClosingBetDaily
from strategies.pass_minute import PassMinute
class TestWalkForwardOptimizer(unittest.TestCase):
    """
    WalkForwardOptimizer의 개발 과정을 TDD 방식으로 검증합니다.
    """
    
    @classmethod
    def setUpClass(cls):
        """테스트 전체에 걸쳐 한 번만 실행되며, DB 연결을 설정합니다."""
        print("\n=== TestWalkForwardOptimizer: 테스트 환경 초기화 ===")
        # 실제 DBManager를 사용하되, 테스트 중에는 실제 API 호출을 막을 수 있습니다.
        cls.api_client = CreonAPIClient()
        cls.db_manager = DBManager()
        # BacktestManager는 DBManager에 의존합니다.
        cls.backtest_manager = BacktestManager(api_client=cls.api_client, db_manager=cls.db_manager)
        
        # 테스트에 사용할 고유 모델 이름 접두사
        cls.model_prefix = "EKLMNO_4s"
        cls.test_model_id = None

    @classmethod
    def tearDownClass(cls):
        """테스트가 모두 끝난 후 생성된 테스트 데이터를 정리합니다."""
        print("\n=== TestWalkForwardOptimizer: 테스트 환경 정리 ===")
        if cls.test_model_id:
            print(f"테스트 데이터 삭제 (model_id: {cls.test_model_id})")
            cls.db_manager.execute_sql("DELETE FROM strategy_profiles WHERE model_id = %s", (cls.test_model_id,))
            cls.db_manager.execute_sql("DELETE FROM daily_regimes WHERE model_id = %s", (cls.test_model_id,))
            cls.db_manager.execute_sql("DELETE FROM hmm_models WHERE model_id = %s", (cls.test_model_id,))
        cls.db_manager.close()

    # =================================================================
    # TDD 1단계: HMM 모델 학습 기능 테스트
    # =================================================================
    @unittest.skip("이유: test_03 개발 중이므로 임시 비활성화")
    def test_01_run_hmm_training_step(self):
        """
        [RED] 1단계: 단일 윈도우의 학습 기간에 대해 HMM 모델을 학습하고 DB에 저장하는지 테스트합니다.
        """
        print("\n--- TDD 1단계 테스트: HMM 모델 학습 ---")
        
        # 1. 준비 (Arrange)
        optimizer = WalkForwardOptimizer(self.backtest_manager)
        train_start = date(2022, 1, 1)
        train_end = date(2023, 12, 31)
        
        # 2. 실행 (Act)
        # 아직 존재하지 않는 내부 메서드를 호출합니다. 이로 인해 테스트는 실패합니다.
        model_name, model_id = optimizer._run_hmm_training_step(
            train_start, train_end, self.model_prefix
        )
        self.__class__.test_model_id = model_id # 정리(cleanup)를 위해 ID 저장

        # 3. 단언 (Assert)
        self.assertIsNotNone(model_id, "model_id가 반환되지 않았습니다.")
        
        # DB에 실제 데이터가 저장되었는지 직접 확인
        model_from_db = self.backtest_manager.fetch_hmm_model_by_name(model_name)
        regimes_df_from_db = self.backtest_manager.fetch_daily_regimes(model_id)

        self.assertIsNotNone(model_from_db, "hmm_models 테이블에서 모델을 찾을 수 없습니다.")
        self.assertEqual(model_from_db['model_name'], model_name)
        self.assertEqual(model_from_db['model_id'], model_id)

        self.assertFalse(regimes_df_from_db.empty, "daily_regimes 테이블에 국면 데이터가 저장되지 않았습니다.")
        self.assertGreater(len(regimes_df_from_db), 0, "국면 데이터의 개수가 0보다 커야 합니다.")
        
        print(" > 통과: HMM 모델 및 국면 데이터가 DB에 성공적으로 저장되었습니다.")


    # =================================================================
    # TDD 2단계: 전략 프로파일링 기능 테스트
    # =================================================================
    @unittest.skip("이유: test_03 개발 중이므로 임시 비활성화")
    def test_02_run_in_sample_profiling_step(self):
        """
        [RED] 2단계: 학습 기간에 파라미터 최적화, 인메모리 백테스트, 프로파일링을 수행하고
        그 결과를 strategy_profiles 테이블에 저장하는지 테스트합니다.
        """
        print("\n--- TDD 2단계 테스트: 전략 프로파일링 ---")

        # 1. 준비 (Arrange)
        optimizer = WalkForwardOptimizer(self.backtest_manager)
        #train_start = date(2022, 1, 1)
        train_start = date(2023, 12, 1)
        train_end = date(2023, 12, 31)

        # 사전 조건: HMM 모델이 먼저 생성되어 있어야 함
        model_name, model_id = optimizer._run_hmm_training_step(
            train_start, train_end, self.model_prefix
        )
        self.__class__.test_model_id = model_id # tearDown에서 정리하도록 설정
        self.assertIsNotNone(model_id, "테스트 사전 조건인 HMM 모델 생성에 실패했습니다.")
        
        # 2. 실행 (Act)
        # 아직 존재하지 않는 내부 메서드를 호출합니다. 이로 인해 테스트는 실패합니다.
        profiling_success = optimizer._run_in_sample_profiling_step(
            train_start, train_end, model_id
        )

        # 3. 단언 (Assert)
        self.assertTrue(profiling_success, "프로파일링 단계가 성공적으로 완료되지 않았습니다.")

        # DB에 실제 데이터가 저장되었는지 직접 확인
        profiles_df = self.backtest_manager.fetch_strategy_profiles_by_model(model_id)
        
        self.assertFalse(profiles_df.empty, "strategy_profiles 테이블에 프로파일이 저장되지 않았습니다.")
        
        # 설정(settings.py)에 활성화된 전략의 수만큼 프로파일이 생성되었는지 확인 (국면 수 * 전략 수)
        from config.settings import STRATEGY_CONFIGS
        active_strategies_count = sum(1 for config in STRATEGY_CONFIGS.values() if config.get("strategy_status"))
        
        # HMM 상태 개수는 4개로 가정
        expected_profiles_count = active_strategies_count * 4 
        self.assertGreaterEqual(len(profiles_df), active_strategies_count, "활성화된 전략의 수보다 적은 프로파일이 생성되었습니다.")
        
        print(f" > 통과: {len(profiles_df)}개의 전략 프로파일이 DB에 성공적으로 저장되었습니다.")

    # =================================================================
    # TDD 3단계: 최종 검증 백테스트 기능 테스트
    # =================================================================
    def test_03_run_out_of_sample_validation_step(self):
        """
        [RED] 3단계: 학습된 모델과 프로파일을 사용하여 Out-of-Sample 기간에 대한
        최종 백테스트를 실행하고, 그 결과를 DB에 저장하는지 테스트합니다.
        """
        print("\n--- TDD 3단계 테스트: 최종 검증 백테스트 ---")

        # 1. 준비 (Arrange)
        optimizer = WalkForwardOptimizer(self.backtest_manager)
        train_start = date(2022, 1, 1)
        train_end = date(2023, 12, 31)
        test_start = date(2024, 1, 1)
        test_end = date(2024, 12, 31)

        # 사전 조건: HMM 모델과 전략 프로파일이 먼저 생성되어 있어야 함
        model_name, model_id = optimizer._run_hmm_training_step(train_start, train_end, self.model_prefix)
        self.__class__.test_model_id = model_id
        profiling_success = optimizer._run_in_sample_profiling_step(train_start, train_end, model_id)
        self.assertTrue(profiling_success, "테스트 사전 조건인 프로파일링에 실패했습니다.")

        # 2. 실행 (Act)
        oos_series, run_id = optimizer._run_out_of_sample_validation_step(
            test_start, test_end, model_name, model_id
        )

        # 3. 단언 (Assert)
        self.assertIsInstance(oos_series, pd.Series, "백테스트 결과가 pandas Series가 아닙니다.")
        self.assertFalse(oos_series.empty, "Out-of-Sample 백테스트 결과가 비어있습니다.")
        self.assertIsNotNone(run_id, "DB에 저장된 run_id가 반환되지 않았습니다.")

        # DB에 실제 백테스트 결과가 저장되었는지 직접 확인
        run_df = self.backtest_manager.fetch_backtest_run(run_id=run_id)
        perf_df = self.backtest_manager.fetch_backtest_performance(run_id=run_id)

        self.assertFalse(run_df.empty, "backtest_run 테이블에 결과가 저장되지 않았습니다.")
        self.assertFalse(perf_df.empty, "backtest_performance 테이블에 결과가 저장되지 않았습니다.")
        
        print(f" > 통과: Out-of-Sample 백테스트 결과가 DB에 성공적으로 저장되었습니다 (run_id: {run_id}).")

if __name__ == '__main__':
    unittest.main()