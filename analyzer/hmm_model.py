# analyzer/hmm_model.py

import numpy as np
import pandas as pd
from hmmlearn import hmm
import logging

logger = logging.getLogger(__name__)

class RegimeAnalysisModel:
    """
    GaussianHMM을 래핑하여 금융 시계열의 장세(Regime)를 분석하는 모델 클래스.
    모델 학습, 파라미터 추출, 파라미터를 통한 모델 복원 기능을 제공합니다.
    """
    def __init__(self, n_states: int = 4, covariance_type: str = "full", n_iter: int = 100):
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=1e-4,
            random_state=42
        )
        self.is_fitted = False

    def fit(self, data: pd.DataFrame):
        """주어진 데이터로 HMM 모델을 학습시킵니다."""
        if data.isnull().values.any():
            raise ValueError("학습 데이터에 NaN 값이 포함되어 있습니다.")
        self.model.fit(data)
        self.is_fitted = True
        logger.info(f"{self.n_states}개 상태 HMM 모델 학습 완료.")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """학습된 모델로 숨겨진 상태(장세)를 예측합니다."""
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출해주세요.")
        return self.model.predict(data)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """각 시점 별로 각 상태에 속할 확률을 예측합니다."""
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출해주세요.")
        return self.model.predict_proba(data)

    def get_params(self) -> dict:
        """학습된 모델의 파라미터를 JSON으로 저장 가능한 딕셔너리 형태로 반환합니다."""
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        return {
            'n_states': self.model.n_components,
            'covariance_type': self.model.covariance_type,
            'startprob_': self.model.startprob_.tolist(),
            'transmat_': self.model.transmat_.tolist(),
            'means_': self.model.means_.tolist(),
            'covars_': self.model.covars_.tolist()
        }

    @classmethod
    def load_from_params(cls, params: dict) -> 'RegimeAnalysisModel':
        """파라미터 딕셔너리로부터 모델 객체를 생성하고 복원합니다."""
        n_states = params['n_states']
        covariance_type = params.get('covariance_type', 'full')
        
        instance = cls(n_states=n_states, covariance_type=covariance_type)
        instance.model.startprob_ = np.array(params['startprob_'])
        instance.model.transmat_ = np.array(params['transmat_'])
        instance.model.means_ = np.array(params['means_'])
        instance.model.covars_ = np.array(params['covars_'])
        instance.is_fitted = True
        
        logger.info(f"{n_states}개 상태 HMM 모델 파라미터 로드 완료.")
        return instance