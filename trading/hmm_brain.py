# trading/hmm_brain.py

import logging
import pandas as pd
import numpy as np
import json
import sys
import os
from typing import Optional
from datetime import date, timedelta
# --- 프로젝트 경로 설정 및 모듈 임포트 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from analyzer.hmm_model import RegimeAnalysisModel
from analyzer.inference_service import RegimeInferenceService
from analyzer.policy_map import PolicyMap
from config.settings import STRATEGY_CONFIGS
# --- ▼ [1. 신규 임포트] 몬테카를로 옵티마이저를 임포트합니다. ---
from optimizer.monte_carlo_optimizer import MonteCarloOptimizer

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HMMBrain")

class HMMBrain:
    """ HMM 모델을 기반으로 '오늘의 전략 포트폴리오'를 결정하는 '두뇌' 클래스 """
    def __init__(self, db_manager: DBManager, backtest_manager: BacktestManager):
        self.db_manager = db_manager
        self.backtest_manager = backtest_manager
        self.model_id: int = -1
        self.model_name: str = ""
        self.inference_service: Optional[RegimeInferenceService] = None
        self.policy_map: Optional[PolicyMap] = None
        self._initialize_hmm_brain()

    def _initialize_hmm_brain(self):
        """ [수정] 가장 최신 워크포워드 모델을 찾아 HMM 관련 모듈을 초기화합니다. """
        model_info = self.db_manager.fetch_latest_wf_model()
        if not model_info:
            raise ValueError("HMM 두뇌를 초기화할 워크포워드 모델이 DB에 없습니다. run_walkforward_analysis.py를 먼저 실행하세요.")
        
        self.model_id = model_info['model_id']
        self.model_name = model_info['model_name']
        hmm_model = RegimeAnalysisModel.load_from_params(model_info['model_params'])
        self.inference_service = RegimeInferenceService(hmm_model)
        
        self.policy_map = PolicyMap()
        self.policy_map.load_rules(os.path.join(project_root, 'config', 'policy.json'))
        logger.info(f"HMM 두뇌 모듈(모델: {self.model_name}, ID: {self.model_id}) 초기화 완료.")

    def create_daily_directive(self) -> Optional[dict]:
        """ 오늘의 최적 전략 포트폴리오(작전 계획)를 생성합니다. """
        logger.info(f"====== 오늘의 전략 포트폴리오 최적화 시작 (기준 모델: {self.model_name}) ======")
        
        # 1. 현재 시장 국면 확률 추론
        self.backtest_manager.prepare_pykrx_data_for_period(date.today() - timedelta(days=900), date.today())
        latest_market_data = self.backtest_manager.get_market_data_for_hmm(date.today() - timedelta(days=100), date.today())
        
        regime_probs = self.inference_service.get_regime_probabilities(latest_market_data)
        if regime_probs is None:
            logger.error("현재 시장 국면 추론 실패.")
            return None
        dominant_regime = np.argmax(regime_probs)
        logger.info(f"현재 시장 국면 추론: 국면 {dominant_regime} (확률: {regime_probs.max():.2%})")

        # 2. 총 투자원금 비율 결정
        total_principal_ratio = self.policy_map.get_target_principal_ratio(regime_probs)
        
        # --- ▼ [2. 핵심 수정] 포트폴리오 최적화 로직 전면 교체 ---
        
        # 3. 포트폴리오 최적화를 위한 데이터 준비
        # 3-1. 모든 국면의 전략 프로파일 조회
        all_profiles_df = self.db_manager.fetch_strategy_profiles_by_model(self.model_id)
        if all_profiles_df.empty:
            logger.warning(f"모델 ID {self.model_id}에 대한 전략 프로파일이 없습니다. 오늘은 투자하지 않습니다.")
            return {'total_principal_ratio': 0.0, 'portfolio': []}

        # 3-2. DataFrame을 Optimizer가 사용할 딕셔너리 형태로 변환
        profiles_dict = {}
        for _, row in all_profiles_df.iterrows():
            name = row['strategy_name']
            regime = row['regime_id']
            if name not in profiles_dict:
                profiles_dict[name] = {}
            profiles_dict[name][regime] = row.to_dict()

        # 3-3. HMM 모델의 전이 행렬 추출
        transition_matrix = self.inference_service.hmm_model.model.transmat_

        # 4. 몬테카를로 옵티마이저 실행
        optimizer = MonteCarloOptimizer(
            strategy_profiles=profiles_dict,
            transition_matrix=transition_matrix
        )
        optimal_weights = optimizer.run_optimization(current_regime_probabilities=regime_probs)

        # 5. 최적화된 가중치를 기반으로 최종 포트폴리오 구성
        portfolio = []
        if optimal_weights:
            # 개별 전략에 적용할 파라미터는 '가장 확률이 높은 현재 국면'의 챔피언 파라미터를 사용
            dominant_regime_profiles = all_profiles_df[all_profiles_df['regime_id'] == dominant_regime]
            
            for name, weight in optimal_weights.items():
                if weight <= 0: # 가중치가 0인 전략은 포트폴리오에 포함하지 않음
                    continue

                profile_row = dominant_regime_profiles[dominant_regime_profiles['strategy_name'] == name]
                
                if profile_row.empty:
                    logger.warning(f"'{name}' 전략은 우세 국면({dominant_regime})에 대한 프로파일이 없어 포트폴리오에서 제외됩니다.")
                    continue
                
                params_json_str = profile_row.iloc[0].get('params_json')
                strategy_params = json.loads(params_json_str) if params_json_str and pd.notna(params_json_str) else {}
                
                portfolio.append({
                    "name": name,
                    "weight": weight,
                    "params": strategy_params
                })

        # --- ▲ 수정 종료 ---
            
        directive = {
            "total_principal_ratio": total_principal_ratio,
            "portfolio": portfolio
        }
        logger.info("====== 오늘의 전략 포트폴리오 최적화 완료 ======")
        logger.info(json.dumps(directive, indent=2, ensure_ascii=False))
        return directive

if __name__ == "__main__":
    db_manager = None
    try:
        db_manager = DBManager()
        backtest_manager = BacktestManager(None, db_manager)
        
        brain = HMMBrain(db_manager, backtest_manager)
        daily_directive = brain.create_daily_directive()

        if daily_directive:
            directive_path = os.path.join(os.path.dirname(__file__), 'daily_directive.json')
            with open(directive_path, 'w', encoding='utf-8') as f:
                json.dump(daily_directive, f, ensure_ascii=False, indent=4)
            logger.info(f"오늘의 작전 계획을 '{directive_path}'에 저장했습니다.")
        else:
            logger.error("오늘의 작전 계획 생성에 실패했습니다.")
            
    except Exception as e:
        logger.critical(f"HMM Brain 실행 중 심각한 오류 발생: {e}", exc_info=True)
    finally:
        if db_manager:
            db_manager.close()