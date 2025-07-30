# tests/test_db_manager_analysis.py

import unittest
import pandas as pd
from datetime import date
import sys
import os
import numpy as np 
import json
from unittest.mock import MagicMock # MagicMock 임포트 추가
# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from manager.db_manager import DBManager
from analyzer.hmm_model import RegimeAnalysisModel # 이제 이 import가 정상적으로 작동합니다.
from analyzer.inference_service import RegimeInferenceService # 신규 클래스 임포트
from analyzer.policy_map import PolicyMap
from analyzer.profiler import StrategyProfiler

class TestDBManagerAnalysis(unittest.TestCase):
    """
    DBManager에 추가된 HMM 및 전략 프로파일 관련 메서드를 테스트합니다.
    """
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행: DBManager 인스턴스 공유"""
        cls.db_manager = DBManager()
        # 전제 조건: 테스트를 실행하려면 hmm_models, strategy_profiles 테이블이 DB에 생성되어 있어야 합니다.
        # (이전 단계에서 생성 스크립트를 실행했으므로 존재해야 함)
        if not cls.db_manager.check_table_exist('hmm_models') or not cls.db_manager.check_table_exist('strategy_profiles'):
            raise Exception("테스트를 위한 hmm_models 또는 strategy_profiles 테이블이 존재하지 않습니다. 생성 스크립트를 먼저 실행해주세요.")

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시 한 번만 실행: DB 연결 종료"""
        cls.db_manager.close()

    def setUp(self):
        """각 테스트 메서드 실행 전 호출: 테스트 데이터 준비 및 테이블 초기화"""
        # 테스트 데이터 정의
        self.hmm_params = {'transmat_': [[0.9, 0.1], [0.1, 0.9]], 'means_': [[0.1], [-0.1]]}
        self.model_1_data = {
            'model_name': 'Test_HMM_v1',
            'n_states': 2,
            'observation_vars': ['kospi_return'],
            'model_params': self.hmm_params,
            'training_start_date': '2024-01-01',
            'training_end_date': '2024-12-31'
        }

        # 테스트 전 관련 테이블의 모든 데이터를 삭제하여 테스트 간 독립성 보장
        conn = self.db_manager.get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
            cursor.execute("TRUNCATE TABLE strategy_profiles;")
            cursor.execute("TRUNCATE TABLE hmm_models;")
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
        conn.commit()

    def test_01_save_and_fetch_hmm_model(self):
        """HMM 모델 저장 및 이름으로 조회 테스트"""
        # 1. HMM 모델 저장
        saved_id = self.db_manager.save_hmm_model(**self.model_1_data)
        self.assertIsInstance(saved_id, int, "모델 저장 후 정수형 ID가 반환되어야 합니다.")

        # 2. 이름으로 모델 조회
        fetched_model = self.db_manager.fetch_hmm_model_by_name(self.model_1_data['model_name'])
        self.assertIsInstance(fetched_model, dict, "조회된 모델은 딕셔너리 형태여야 합니다.")
        self.assertEqual(fetched_model['model_id'], saved_id)
        self.assertEqual(fetched_model['n_states'], self.model_1_data['n_states'])
        self.assertEqual(fetched_model['model_params']['transmat_'], self.hmm_params['transmat_'])

        # 3. 없는 모델 조회
        non_existent_model = self.db_manager.fetch_hmm_model_by_name('Non_Existent_Model')
        self.assertIsNone(non_existent_model, "존재하지 않는 모델 조회 시 None이 반환되어야 합니다.")

    def test_02_save_and_fetch_strategy_profiles(self):
        """전략 프로파일 저장(생성 및 업데이트) 및 모델 ID로 조회 테스트"""
        # 1. 사전 조건: 프로파일이 참조할 HMM 모델 먼저 저장
        model_id = self.db_manager.save_hmm_model(**self.model_1_data)
        self.assertIsNotNone(model_id)

        # 2. 프로파일 데이터 정의
        profile_list = [
            {
                'strategy_name': 'SMADaily', 'model_id': model_id, 'regime_id': 0,
                'sharpe_ratio': 1.5, 'mdd': -0.10, 'total_return': 25.5, 'win_rate': 0.6, 'num_trades': 50,
                'profiling_start_date': '2025-01-01', 'profiling_end_date': '2025-06-30'
            },
            {
                'strategy_name': 'SMADaily', 'model_id': model_id, 'regime_id': 1,
                'sharpe_ratio': -0.5, 'mdd': -0.25, 'total_return': -10.2, 'win_rate': 0.3, 'num_trades': 30,
                'profiling_start_date': '2025-01-01', 'profiling_end_date': '2025-06-30'
            }
        ]
        
        # 3. 새로운 프로파일 저장
        save_result = self.db_manager.save_strategy_profiles(profile_list)
        self.assertTrue(save_result)
        
        # 4. 저장된 프로파일 조회 및 검증
        fetched_df = self.db_manager.fetch_strategy_profiles_by_model(model_id)
        self.assertIsInstance(fetched_df, pd.DataFrame)
        self.assertEqual(len(fetched_df), 2)
        # regime_id=0인 데이터의 sharpe_ratio 검증
        sharpe_regime0 = fetched_df[fetched_df['regime_id'] == 0]['sharpe_ratio'].iloc[0]
        self.assertEqual(sharpe_regime0, 1.5)

        # 5. 프로파일 업데이트 (ON DUPLICATE KEY UPDATE 테스트)
        updated_profile = profile_list[0].copy()
        updated_profile['sharpe_ratio'] = 1.8 # sharpe_ratio 값 변경
        
        update_result = self.db_manager.save_strategy_profiles([updated_profile])
        self.assertTrue(update_result)
        
        # 6. 업데이트된 프로파일 조회 및 검증
        fetched_updated_df = self.db_manager.fetch_strategy_profiles_by_model(model_id)
        self.assertEqual(len(fetched_updated_df), 2, "업데이트 후 전체 레코드 수는 동일해야 합니다.")
        # regime_id=0인 데이터의 sharpe_ratio가 업데이트되었는지 검증
        updated_sharpe_regime0 = fetched_updated_df[fetched_updated_df['regime_id'] == 0]['sharpe_ratio'].iloc[0]
        self.assertEqual(updated_sharpe_regime0, 1.8)

    # Task HMM-01: RegimeAnalysisModel Tests
    def test_03_hmm_model_fit(self):
        """HMM 모델이 주어진 데이터로 정상적으로 학습되는지 테스트"""
        # given: 모의 데이터 준비
        mock_data = pd.DataFrame(np.random.randn(100, 2), columns=['returns', 'volatility'])
        model = RegimeAnalysisModel(n_states=4)
        
        # when: 모델 학습
        model.fit(mock_data)
        
        # then: 모델이 학습되었는지 검증
        self.assertTrue(model.is_fitted)
        self.assertEqual(model.model.n_components, 4)

    def test_04_hmm_model_get_and_load_params(self):
        """모델 파라미터 추출 및 복원 기능 테스트"""
        # given: 학습된 모델 준비
        mock_data = pd.DataFrame(np.random.randn(100, 2), columns=['returns', 'volatility'])
        original_model = RegimeAnalysisModel(n_states=3)
        original_model.fit(mock_data)
        
        # when: 파라미터 추출 및 새 모델로 복원
        params = original_model.get_params()
        loaded_model = RegimeAnalysisModel.load_from_params(params)
        
        # then: 파라미터 구조 및 복원된 모델 검증
        self.assertIn('transmat_', params)
        self.assertEqual(len(params['transmat_']), 3)
        self.assertTrue(loaded_model.is_fitted)
        self.assertEqual(loaded_model.model.n_components, 3)
        np.testing.assert_array_equal(original_model.model.transmat_, loaded_model.model.transmat_)

    # Task HMM-02: RegimeInferenceService Tests
    def test_05_inference_service_get_probabilities(self):
        """장세 추론 서비스가 정확한 형태의 확률 벡터를 반환하는지 테스트"""
        # given: 학습이 완료된 가짜(Mock) HMM 모델 준비
        mock_model = MagicMock(spec=RegimeAnalysisModel)
        mock_model.is_fitted = True
        # predict_proba가 (n_samples, n_states) 형태의 배열을 반환하도록 설정
        mock_model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]])
        
        # when: 가짜 모델로 서비스 객체 생성 및 확률 추론
        service = RegimeInferenceService(hmm_model=mock_model)
        # 2개의 샘플 데이터를 넣었다고 가정
        probs = service.get_regime_probabilities(pd.DataFrame([[1,2],[3,4]]))
        
        # then: 마지막 시점의 확률 [0.2, 0.3, 0.5]를 정확히 반환하는지 검증
        self.assertIsNotNone(probs)
        self.assertEqual(probs.shape, (3,))
        np.testing.assert_almost_equal(np.sum(probs), 1.0)
        np.testing.assert_array_equal(probs, [0.2, 0.3, 0.5])
        mock_model.predict_proba.assert_called_once()

    # Task ALLOC-01: PolicyMap Tests
    def test_06_policy_map_loads_rules_and_gets_ratio(self):
        """정책 맵이 규칙 파일을 로드하고 정확한 투자 비중을 반환하는지 테스트"""
        # given: 임시 규칙 파일과 PolicyMap 객체 생성
        mock_rules = {
            "regime_to_principal_ratio": {
                "0": 0.2, "1": 0.5, "2": 1.0
            },
            "default_principal_ratio": 0.4
        }
        rule_file_path = "temp_policy_rules.json"
        with open(rule_file_path, 'w') as f:
            json.dump(mock_rules, f)
        
        policy_map = PolicyMap()
        policy_map.load_rules(rule_file_path)

        # when/then: 지배적 장세에 따른 비중 반환 테스트
        self.assertEqual(policy_map.get_target_principal_ratio(np.array([0.8, 0.1, 0.1])), 0.2) # 장세 0
        self.assertEqual(policy_map.get_target_principal_ratio(np.array([0.1, 0.1, 0.8])), 1.0) # 장세 2
        
        # when/then: 규칙에 없는 장세일 경우 default 값 반환 테스트 (장세 3)
        self.assertEqual(policy_map.get_target_principal_ratio(np.array([0.1, 0.1, 0.1, 0.7])), 0.4)

        # cleanup: 임시 파일 삭제
        os.remove(rule_file_path)

    def test_07_profiler_generates_correct_profiles(self):
        """전략 프로파일러가 통계를 정확하게 계산하는지 테스트"""
        # given: 모의 거래 로그와 장세 데이터 준비
        trade_logs_df = pd.DataFrame({
            'trade_datetime': pd.to_datetime(['2025-01-05', '2025-01-06', '2025-01-15', '2025-01-16']),
            'strategy_name': ['SMADaily', 'SMADaily', 'SMADaily', 'SMADaily'],
            'realized_pnl': [100, -50, 200, 80] # 컬럼명을 realized_pnl로 가정
        })
        
        # 1월 1일~9일: 장세 0, 1월 10일~: 장세 1
        date_range = pd.to_datetime(pd.date_range('2025-01-01', '2025-01-20'))
        regime_ids = [0] * 9 + [1] * 11
        regime_data_df = pd.DataFrame({'date': date_range, 'regime_id': regime_ids})

        # when: 프로파일 생성
        profiler = StrategyProfiler()
        profiles = profiler.generate_profiles(trade_logs_df, regime_data_df, model_id=1)

        # then: 생성된 프로파일 검증
        self.assertEqual(len(profiles), 2)  # 장세 0과 1에 대한 2개의 프로파일 생성
        
        profile_regime0 = next((p for p in profiles if p['regime_id'] == 0), None)
        profile_regime1 = next((p for p in profiles if p['regime_id'] == 1), None)

        self.assertIsNotNone(profile_regime0)
        self.assertIsNotNone(profile_regime1)

        # 장세 0 (1/5, 1/6 거래) 통계 검증
        self.assertEqual(profile_regime0['strategy_name'], 'SMADaily')
        self.assertEqual(profile_regime0['num_trades'], 2)
        self.assertEqual(profile_regime0['total_return'], 50) # 100 - 50
        self.assertEqual(profile_regime0['win_rate'], 0.5)   # 2번 중 1번 승리
        
        # 장세 1 (1/15, 1/16 거래) 통계 검증
        self.assertEqual(profile_regime1['num_trades'], 2)
        self.assertEqual(profile_regime1['total_return'], 280) # 200 + 80
        self.assertEqual(profile_regime1['win_rate'], 1.0)    # 2번 중 2번 승리

if __name__ == '__main__':
    unittest.main()