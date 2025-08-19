import unittest
import pandas as pd
import numpy as np
from datetime import date
import sys
import os
import json

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 필요 모듈 임포트 ---
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from analyzer import train_and_save_hmm
from trading.hmm_backtest import HMMBacktest
from strategies.sma_daily import SMADaily
from strategies.breakout_daily import BreakoutDaily
from strategies.pass_minute import PassMinute

class TestEndToEndBacktest(unittest.TestCase):
    """
    업그레이드된 포트폴리오 매니저를 탑재한 HMMBacktest의
    End-To-End 실행을 검증합니다.
    """

    @classmethod
    def setUpClass(cls):
        """테스트 전체에 걸쳐 한 번만 실행되며, 사전 준비 작업을 수행합니다."""
        print("\n=== TestEndToEndBacktest: 테스트 환경 초기화 시작 ===")
        cls.api_client = CreonAPIClient()
        cls.db_manager = DBManager()
        cls.backtest_manager = BacktestManager(api_client=cls.api_client, db_manager=cls.db_manager)
        
        cls.test_model_name = 'EKLMNO_4s_2407-2408' 
        cls.start_date = date(2024, 7, 1)
        cls.end_date = date(2024, 8, 30)
        cls.test_strategies = ['SMADaily', 'BreakoutDaily']
        cls.model_id = None

        # --- 사전 조건 1: HMM 모델 및 국면 데이터 준비 ---
        print(f"HMM 모델 준비 중: '{cls.test_model_name}'...")
        success, model_info, _ = train_and_save_hmm.run_hmm_training(
            model_name=cls.test_model_name,
            start_date=cls.start_date,
            end_date=cls.end_date,
            backtest_manager=cls.backtest_manager
        )
        if not success:
            raise Exception("사전 조건 실패: HMM 모델 학습에 실패했습니다.")
        
        cls.model_id = model_info['model_id']
        print(f"HMM 모델 준비 완료. 모델 ID: {cls.model_id}")

        # --- 사전 조건 2: 테스트용 전략 프로파일 데이터 준비 ---
        print("테스트용 가상 전략 프로파일 준비 중...")
        cls._prepare_dummy_strategy_profiles(cls.model_id, cls.test_strategies)
        print("테스트 환경 초기화 완료.")

    @classmethod
    def tearDownClass(cls):
        """테스트가 모두 끝난 후 정리 작업을 수행합니다."""
        print("\n=== TestEndToEndBacktest: 테스트 환경 정리 시작 ===")
        if cls.model_id:
            cls.db_manager.execute_sql("DELETE FROM strategy_profiles WHERE model_id = %s", (cls.model_id,))
            cls.db_manager.execute_sql("DELETE FROM daily_regimes WHERE model_id = %s", (cls.model_id,))
            cls.db_manager.execute_sql("DELETE FROM hmm_models WHERE model_id = %s", (cls.model_id,))
            print(f"테스트 데이터 정리 완료 (모델 ID: {cls.model_id})")
        cls.db_manager.close()

    @classmethod
    def _prepare_dummy_strategy_profiles(cls, model_id, strategy_names):
        """테스트를 위한 가상의 전략 프로파일을 DB에 저장합니다."""
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
            raise Exception("사전 조건 실패: 가상 전략 프로파일을 DB에 저장할 수 없습니다.")
        print(f"가상 프로파일 {len(dummy_profiles)}개 DB 저장 완료.")

    def test_full_portfolio_backtest_run(self):
        """
        [E2E Test] 전체 포트폴리오 백테스트가 오류 없이 실행되고 유효한 결과를 반환하는지 테스트합니다.
        """
        print("\n--- 포트폴리오 End-To-End 백테스트 실행 ---")
        
        # 1. 백테스트 시스템 초기화
        backtest_system = HMMBacktest(
            manager=self.backtest_manager,
            initial_cash=100_000_000,
            start_date=self.start_date,
            end_date=self.end_date,
            save_to_db=False
        )

        # 2. 테스트에 필요한 전략 객체 생성
        daily_strategies_list = [
            SMADaily(broker=backtest_system.broker, data_store=backtest_system.data_store),
            BreakoutDaily(broker=backtest_system.broker, data_store=backtest_system.data_store)
        ]
        minute_strategy = PassMinute(broker=backtest_system.broker, data_store=backtest_system.data_store)

        # 3. 백테스트 시스템에 전략 설정
        backtest_system.set_strategies(
            daily_strategies=daily_strategies_list,
            minute_strategy=minute_strategy
        )

        # 4. 데이터 사전 로딩
        backtest_system.prepare_for_system()

        # 5. 백테스트 실행
        portfolio_series, metrics, _, _ = backtest_system.reset_and_rerun(
            daily_strategies=daily_strategies_list,
            minute_strategy=minute_strategy,
            mode='hmm',
            model_name=self.test_model_name
        )

        # 6. 결과 검증
        self.assertIsInstance(portfolio_series, pd.Series, "결과 포트폴리오가 pandas Series가 아닙니다.")
        self.assertFalse(portfolio_series.empty, "결과 포트폴리오가 비어있습니다.")
        
        self.assertIsInstance(metrics, dict, "성과 지표가 딕셔너리가 아닙니다.")
        self.assertIn('sharpe_ratio', metrics, "성과 지표에 'sharpe_ratio'가 포함되어야 합니다.")
        self.assertIn('mdd', metrics, "성과 지표에 'mdd'가 포함되어야 합니다.")
        self.assertIn('total_return', metrics, "성과 지표에 'total_return'가 포함되어야 합니다.")
        
        final_value = portfolio_series.iloc[-1]
        self.assertGreater(final_value, 0, "최종 포트폴리오 가치가 0보다 커야 합니다.")

        print(f"\n--- End-To-End 테스트 통과 ---")
        print("최종 포트폴리오 가치:", f"{final_value:,.0f}원")
        print("성과 지표:", metrics)

if __name__ == '__main__':
    unittest.main()