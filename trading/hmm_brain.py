# trading/hmm_brain.py (신규 파일)

import logging
import pandas as pd
import numpy as np
import json
import sys
import os
from typing import Optional
# --- 프로젝트 경로 설정 및 모듈 임포트 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from analyzer.hmm_model import RegimeAnalysisModel
from analyzer.inference_service import RegimeInferenceService
from analyzer.policy_map import PolicyMap
from config.settings import LIVE_HMM_MODEL_NAME, STRATEGY_CONFIGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HMMBrain")

class HMMBrain:
    """ HMM 모델을 기반으로 '오늘의 전략 포트폴리오'를 결정하는 '두뇌' 클래스 """
    def __init__(self, db_manager: DBManager, backtest_manager: BacktestManager):
        self.db_manager = db_manager
        self.backtest_manager = backtest_manager
        self.model_id: int = -1
        self.inference_service: RegimeInferenceService
        self.policy_map: PolicyMap
        self._initialize_hmm_brain()

    def _initialize_hmm_brain(self):
        """ HMM 관련 모듈을 초기화합니다. """
        model_info = self.db_manager.fetch_hmm_model_by_name(LIVE_HMM_MODEL_NAME)
        if not model_info:
            raise ValueError(f"HMM 모델 '{LIVE_HMM_MODEL_NAME}'을 DB에서 찾을 수 없습니다.")
        self.model_id = model_info['model_id']
        hmm_model = RegimeAnalysisModel.load_from_params(model_info['model_params'])
        self.inference_service = RegimeInferenceService(hmm_model)
        self.policy_map = PolicyMap()
        self.policy_map.load_rules(os.path.join(project_root, 'config', 'policy.json'))
        logger.info(f"HMM 두뇌 모듈(모델 ID: {self.model_id}) 초기화 완료.")

    def create_daily_directive(self) -> Optional[dict]:
        """ 오늘의 최적 전략 포트폴리오(작전 계획)를 생성합니다. """
        logger.info("====== 오늘의 전략 포트폴리오 최적화 시작 ======")
        
        # 1. 현재 시장 국면 확률 추론
        latest_market_data = self.backtest_manager.get_market_data_for_hmm(days=10)
        regime_probs = self.inference_service.get_regime_probabilities(latest_market_data)
        if regime_probs is None:
            logger.error("현재 시장 국면 추론 실패.")
            return None
        dominant_regime = np.argmax(regime_probs)
        logger.info(f"현재 시장 국면 추론: 국면 {dominant_regime} (확률: {regime_probs.max():.2%})")

        # 2. 총 투자원금 비율 결정
        total_principal_ratio = self.policy_map.get_target_principal_ratio(regime_probs)
        
        # 3. 국면에 맞는 전략 프로파일 조회
        profiles_df = self.db_manager.fetch_strategy_profiles_by_model_and_regime(self.model_id, dominant_regime)
        if profiles_df.empty:
            logger.warning(f"국면 {dominant_regime}에 대한 전략 프로파일이 없습니다.")
            # 투자할 전략이 없으므로 투자 비율 0, 빈 포트폴리오 반환
            return {'total_principal_ratio': 0.0, 'portfolio': []}

        # 4. 포트폴리오 최적화 및 자본 배분
        positive_sharpe_profiles = profiles_df[profiles_df['sharpe_ratio'] > 0].copy()
        if positive_sharpe_profiles.empty:
            logger.warning(f"국면 {dominant_regime}에서 유효한(샤프>0) 전략이 없습니다.")
            return {'total_principal_ratio': 0.0, 'portfolio': []}

        total_sharpe = positive_sharpe_profiles['sharpe_ratio'].sum()
        # --- ▼ [핵심 수정] ZeroDivisionError 방지 ---
        if total_sharpe <= 0:
            logger.warning(f"유효 전략들의 샤프 지수 합이 0 이하({total_sharpe})이므로 가중치를 계산할 수 없습니다.")
            return {'total_principal_ratio': 0.0, 'portfolio': []}
        # --- ▲ [핵심 수정] ---

        positive_sharpe_profiles['weight'] = positive_sharpe_profiles['sharpe_ratio'] / total_sharpe
        
        portfolio = []
        for _, row in positive_sharpe_profiles.iterrows():
            default_params = next((item['params'] for item in STRATEGY_CONFIGS if item['name'] == row['strategy_name']), {})
            
            # --- ▼ [핵심 수정] params_json이 None일 경우에 대한 처리 ---
            params_json_str = row.get('params_json')
            strategy_params = json.loads(params_json_str) if params_json_str and pd.notna(params_json_str) else default_params
            # --- ▲ [핵심 수정] ---
            
            portfolio.append({
                "strategy_name": row['strategy_name'],
                "weight": row['weight'],
                "params": strategy_params
            })
            
        directive = {
            "total_principal_ratio": total_principal_ratio,
            "portfolio": portfolio
        }
        logger.info("====== 오늘의 전략 포트폴리오 최적화 완료 ======")
        logger.info(json.dumps(directive, indent=2, ensure_ascii=False))
        return directive

if __name__ == "__main__":
    try:
        db_manager = DBManager()
        backtest_manager = BacktestManager(None, db_manager)
        
        brain = HMMBrain(db_manager, backtest_manager)
        daily_directive = brain.create_daily_directive()

        if daily_directive:
            # 작전 계획을 JSON 파일로 저장
            directive_path = os.path.join(project_root, 'config', 'daily_directive.json')
            with open(directive_path, 'w', encoding='utf-8') as f:
                json.dump(daily_directive, f, ensure_ascii=False, indent=4)
            logger.info(f"오늘의 작전 계획을 '{directive_path}'에 저장했습니다.")
        else:
            logger.error("오늘의 작전 계획 생성에 실패했습니다.")
            
    except Exception as e:
        logger.critical(f"HMM Brain 실행 중 심각한 오류 발생: {e}", exc_info=True)
    finally:
        if 'db_manager' in locals():
            db_manager.close()