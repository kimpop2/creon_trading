# analyzer/inference_service.py

import pandas as pd
import numpy as np
from typing import Optional
import logging
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
sys.path.insert(0, project_root)

from analyzer.hmm_model import RegimeAnalysisModel

logger = logging.getLogger(__name__)

class RegimeInferenceService:
    """학습된 HMM 모델을 사용하여 실시간 장세 추론을 수행하는 서비스."""

    def __init__(self, hmm_model: RegimeAnalysisModel):
        if not hmm_model.is_fitted:
            raise ValueError("주입된 HMM 모델이 학습되지 않았습니다.")
        self.hmm_model = hmm_model

    def get_regime_probabilities(self, latest_data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        최신 관찰 데이터에 대한 각 장세의 확률을 반환합니다.
        
        Args:
            latest_data (pd.DataFrame): 추론에 사용할 최신 관찰 변수 데이터
            
        Returns:
            Optional[np.ndarray]: 각 장세에 속할 확률 벡터 (예: [0.1, 0.2, 0.6, 0.1])
        """
        try:
            # predict_proba는 모든 시점에 대한 확률을 반환하므로 마지막 시점의 값만 사용합니다.
            all_probs = self.hmm_model.predict_proba(latest_data)
            return all_probs[-1]
        except Exception as e:
            logger.error(f"장세 확률 추론 실패: {e}", exc_info=True)
            return None