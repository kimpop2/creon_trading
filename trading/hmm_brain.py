# trading/hmm_brain.py (최종 수정본)

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
# [수정] STRATEGY_CONFIGS를 임포트
from config.settings import STRATEGY_CONFIGS

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
        # DB에서 'wf_model_'로 시작하는 가장 최신 모델을 가져옵니다.
        model_info = self.db_manager.fetch_latest_wf_model()
        if not model_info:
            raise ValueError("HMM 두뇌를 초기화할 워크포워드 모델이 DB에 없습니다. run_walkforward_analysis.py를 먼저 실행하세요.")
        
        self.model_id = model_info['model_id']
        self.model_name = model_info['model_name']
        hmm_model = RegimeAnalysisModel.load_from_params(model_info['model_params'])
        self.inference_service = RegimeInferenceService(hmm_model)
        
        # policy.json은 최적화된 결과이므로, 해당 모델의 학습 기간으로 최적화된 policy를 사용해야 합니다.
        # 이 부분은 TradingManager에서 최종 생성된 optimal_policy.json을 어떻게 관리할지에 따라 달라질 수 있습니다.
        # 여기서는 우선 기본 policy.json을 로드합니다.
        self.policy_map = PolicyMap()
        self.policy_map.load_rules(os.path.join(project_root, 'config', 'policy.json'))
        logger.info(f"HMM 두뇌 모듈(모델: {self.model_name}, ID: {self.model_id}) 초기화 완료.")

    def create_daily_directive(self) -> Optional[dict]:
        """ 오늘의 최적 전략 포트폴리오(작전 계획)를 생성합니다. """
        logger.info(f"====== 오늘의 전략 포트폴리오 최적화 시작 (기준 모델: {self.model_name}) ======")
        
        # 1. 현재 시장 국면 확률 추론
        # HMM 모델 학습에 사용된 피처와 동일한 데이터를 가져와야 함
        self.backtest_manager.prepare_pykrx_data_for_period(date.today() - timedelta(days=900), date.today())
        latest_market_data = self.backtest_manager.get_market_data_for_hmm(days=100) # 추론을 위해 충분한 데이터 제공
        
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
            logger.warning(f"국면 {dominant_regime}에 대한 전략 프로파일이 없습니다. 오늘은 투자하지 않습니다.")
            return {'total_principal_ratio': 0.0, 'portfolio': []}

        # 4. 포트폴리오 최적화 및 자본 배분
        positive_sharpe_profiles = profiles_df[profiles_df['sharpe_ratio'] > 0].copy()
        if positive_sharpe_profiles.empty:
            logger.warning(f"국면 {dominant_regime}에서 유효한(샤프>0) 전략이 없습니다. 오늘은 투자하지 않습니다.")
            return {'total_principal_ratio': 0.0, 'portfolio': []}

        total_sharpe = positive_sharpe_profiles['sharpe_ratio'].sum()
        if total_sharpe <= 0:
            logger.warning(f"유효 전략들의 샤프 지수 합이 0 이하({total_sharpe})이므로 가중치를 계산할 수 없습니다. 오늘은 투자하지 않습니다.")
            return {'total_principal_ratio': 0.0, 'portfolio': []}

        positive_sharpe_profiles['weight'] = positive_sharpe_profiles['sharpe_ratio'] / total_sharpe
        
        portfolio = []
        for _, row in positive_sharpe_profiles.iterrows():
            strategy_name = row['strategy_name']
            
            # [수정] STRATEGY_CONFIGS에서 기본 파라미터를 가져옴
            default_params = STRATEGY_CONFIGS.get(strategy_name, {}).get('default_params', {})
            
            params_json_str = row.get('params_json')
            strategy_params = json.loads(params_json_str) if params_json_str and pd.notna(params_json_str) else default_params
            
            portfolio.append({
                "name": strategy_name,
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
    db_manager = None  # finally 블록에서 사용하기 위해 미리 선언
    try:
        db_manager = DBManager()
        # HMMBrain은 과거 시장 데이터(pykrx)만 필요하므로 api_client는 None으로 전달
        backtest_manager = BacktestManager(None, db_manager)
        
        brain = HMMBrain(db_manager, backtest_manager)
        daily_directive = brain.create_daily_directive()

        if daily_directive:
            # daily_directive.json 파일도 trading 폴더 안에 저장됩니다.
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