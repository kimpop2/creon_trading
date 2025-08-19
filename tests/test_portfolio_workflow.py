import unittest
import pandas as pd
import numpy as np
from datetime import date
import sys
import os
import json
from unittest.mock import patch

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 테스트 대상 및 필요 모듈 임포트 ---
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from analyzer import train_and_save_hmm
from optimizer.monte_carlo_optimizer import MonteCarloOptimizer
from trading.hmm_backtest import HMMBacktest
from strategies.sma_daily import SMADaily
from strategies.breakout_daily import BreakoutDaily
from strategies.pass_minute import PassMinute

class TestPortfolioWorkflow(unittest.TestCase):
    """
    몬테카를로 시뮬레이션 기반 포트폴리오 최적화 워크플로우를 테스트합니다.
    """
    
    @classmethod
    def setUpClass(cls):
        """테스트 전체에 걸쳐 한 번만 실행되며, 실제 API와 DB 연결을 설정합니다."""
        print("\n=== TestPortfolioWorkflow: Initializing Test Environment ===")
        # --- ▼ [수정] CreonAPIClient 실제 객체를 생성하고 연결합니다. ---
        cls.api_client = CreonAPIClient()
        cls.db_manager = DBManager()
        cls.backtest_manager = BacktestManager(api_client=cls.api_client, db_manager=cls.db_manager)
        # --- ▲ 수정 종료 ---
        
        cls.test_model_name = 'EKLMNO_4s_2407-2408' 
        cls.start_date = date(2024, 7, 1)
        cls.end_date = date(2024, 8, 31)
        cls.test_strategies = ['SMADaily', 'BreakoutDaily']
        cls.model_id = None

        # --- 사전 조건 1: HMM 모델 및 국면 데이터 준비 ---
        print(f"Preparing HMM model '{cls.test_model_name}'...")
        success, model_info, _ = train_and_save_hmm.run_hmm_training(
            model_name=cls.test_model_name,
            start_date=cls.start_date,
            end_date=cls.end_date,
            backtest_manager=cls.backtest_manager
        )
        if not success:
            raise Exception("Prerequisite failed: HMM model training was unsuccessful.")
        
        cls.model_id = model_info['model_id']
        cls.transition_matrix = np.array(model_info['model_params']['transmat_'])
        print(f"HMM model prepared successfully. Model ID: {cls.model_id}")

        # --- 사전 조건 2: 테스트용 전략 프로파일 데이터 준비 ---
        print("Preparing dummy strategy profiles for testing...")
        cls._prepare_dummy_strategy_profiles(cls.model_id, cls.test_strategies)
        
        # --- 테스트에 사용할 데이터 로드 ---
        cls.strategy_profiles_from_db = cls._load_strategy_profiles_as_dict(cls.model_id)
        print("Test environment initialized successfully.")

    @classmethod
    def tearDownClass(cls):
        """테스트가 모두 끝난 후 정리 작업을 수행합니다."""
        print("\n=== TestPortfolioWorkflow: Tearing Down Test Environment ===")
        if hasattr(cls, 'model_id') and cls.model_id:
            cls.db_manager.execute_sql("DELETE FROM strategy_profiles WHERE model_id = %s", (cls.model_id,))
            cls.db_manager.execute_sql("DELETE FROM daily_regimes WHERE model_id = %s", (cls.model_id,))
            cls.db_manager.execute_sql("DELETE FROM hmm_models WHERE model_id = %s", (cls.model_id,))
            print(f"Cleaned up test data for model_id: {cls.model_id}")
        cls.db_manager.close()

    @classmethod
    def _prepare_dummy_strategy_profiles(cls, model_id, strategy_names):
        """MonteCarloOptimizer 테스트를 위한 가상의 전략 프로파일을 DB에 저장합니다."""
        dummy_profiles = []
        for name in strategy_names:
            for regime in range(4):
                dummy_profiles.append({
                    'strategy_name': name, 'model_id': model_id, 'regime_id': regime,
                    'sharpe_ratio': 1.5 - (regime * 0.4), 'mdd': 0.10 + (regime * 0.05),
                    'total_return': 0.20 - (regime * 0.05), 'win_rate': 0.60 - (regime * 0.05),
                    'num_trades': 50, 'profiling_start_date': cls.start_date,
                    'profiling_end_date': cls.end_date, 'params_json': json.dumps({'period': 20 + regime})
                })
        if not cls.db_manager.save_strategy_profiles(dummy_profiles):
            raise Exception("Prerequisite failed: Could not save dummy strategy profiles to DB.")
        print(f"Saved {len(dummy_profiles)} dummy profiles to DB.")
        
    @classmethod
    def _load_strategy_profiles_as_dict(cls, model_id) -> dict:
        """DB에서 프로파일을 로드하여 Optimizer가 사용할 딕셔너리 형태로 변환합니다."""
        profiles_df = cls.backtest_manager.fetch_strategy_profiles_by_model(model_id)
        profiles_dict = {}
        for _, row in profiles_df.iterrows():
            name = row['strategy_name']
            regime = row['regime_id']
            if name not in profiles_dict:
                profiles_dict[name] = {}
            profiles_dict[name][regime] = row.to_dict()
        return profiles_dict

    def test_01_monte_carlo_optimizer_initialization(self):
        """[Phase 1] MonteCarloOptimizer가 정상적으로 초기화되는지 테스트합니다."""
        print("\n--- Phase 1: Optimizer Initialization Test ---")
        optimizer = MonteCarloOptimizer(
            strategy_profiles=self.strategy_profiles_from_db,
            transition_matrix=self.transition_matrix
        )
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.num_strategies, len(self.test_strategies))
        self.assertEqual(optimizer.mean_returns.shape, (len(self.test_strategies), 4))
        self.assertFalse(np.isnan(optimizer.mean_returns).any())
        print("--- Phase 1: Test Passed ---")

    def test_02_run_optimization_and_validate_weights(self):
        """[Phase 2] run_optimization 메서드가 유효한 포트폴리오 가중치를 반환하는지 테스트합니다."""
        print("\n--- Phase 2: Optimization Execution and Weight Validation Test ---")
        optimizer = MonteCarloOptimizer(
            strategy_profiles=self.strategy_profiles_from_db,
            transition_matrix=self.transition_matrix,
            num_simulations=100
        )
        current_probs = np.array([0.7, 0.1, 0.1, 0.1])
        optimal_weights = optimizer.run_optimization(current_regime_probabilities=current_probs)
        
        self.assertIsInstance(optimal_weights, dict)
        self.assertEqual(len(optimal_weights), len(self.test_strategies))
        self.assertAlmostEqual(sum(optimal_weights.values()), 1.0, places=5)
        for strategy, weight in optimal_weights.items():
            self.assertTrue(0 <= weight <= 1)
        print(f"Optimization Result: {optimal_weights}")
        print("--- Phase 2: Test Passed ---")

    @patch('manager.portfolio_manager.PortfolioManager.get_strategy_capitals')
    def test_03_integration_with_hmmbacktest(self, mock_get_strategy_capitals):
        """[Phase 3] HMMBacktest가 업그레이드된 PortfolioManager를 올바르게 호출하는지 통합 테스트합니다."""
        print("\n--- Phase 3: HMMBacktest Integration Test (with Mocking) ---")
        mock_get_strategy_capitals.return_value = {'SMADaily': 5_000_000, 'BreakoutDaily': 5_000_000}

        backtest_system = HMMBacktest(
            manager=self.backtest_manager, initial_cash=10_000_000,
            start_date=self.start_date, end_date=date(2024, 7, 5),
            save_to_db=False
        )
        daily_strategies_list = [
            SMADaily(broker=backtest_system.broker, data_store=backtest_system.data_store),
            BreakoutDaily(broker=backtest_system.broker, data_store=backtest_system.data_store)
        ]
        minute_strategy = PassMinute(broker=backtest_system.broker, data_store=backtest_system.data_store)
        
        backtest_system.reset_and_rerun(
            daily_strategies=daily_strategies_list,
            minute_strategy=minute_strategy,
            mode='hmm',
            model_name=self.test_model_name
        )

        self.assertTrue(mock_get_strategy_capitals.called, "HMMBacktest가 PortfolioManager.get_strategy_capitals를 호출하지 않았습니다.")
        print(f"PortfolioManager.get_strategy_capitals가 {mock_get_strategy_capitals.call_count}회 호출되었습니다.")
        print("--- Phase 3: Test Passed ---")

if __name__ == '__main__':
    unittest.main()