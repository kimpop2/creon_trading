# analyzer/hmm_model.py

import numpy as np
import pandas as pd
from hmmlearn import hmm
import logging

logger = logging.getLogger(__name__)

class RegimeAnalysisModel:
    """
    GaussianHMM을 래핑하여 금융 시계열의 장세(Regime)를 분석하는 모델 클래스.
    """
    def __init__(self, n_states: int = 4, covariance_type: str = "diag", n_iter: int = 100):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=self.covariance_type,
            n_iter=n_iter,
            tol=1e-4,
            random_state=42
        )
        self.is_fitted = False

    def fit(self, data: pd.DataFrame):
        if data.isnull().values.any():
            raise ValueError("학습 데이터에 NaN 값이 포함되어 있습니다.")
        self.model.fit(data)
        self.is_fitted = True
        logger.info(f"{self.n_states}개 상태 HMM 모델 학습 완료.")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출해주세요.")
        return self.model.predict(data)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출해주세요.")
        return self.model.predict_proba(data)

    def get_params(self) -> dict:
        """ [최종 수정] 학습된 모델의 파라미터를 JSON으로 저장 가능한 딕셔너리 형태로 반환합니다. """
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        
        # --- ▼ [핵심 수정] ---
        # covariance_type에 따라 covars_의 형태를 올바르게 처리합니다.
        covars_to_save = self.model.covars_
        if self.model.covariance_type == 'diag' and covars_to_save.ndim == 2:
            # hmmlearn 최신 버전 등에서 이미 (N, D) 형태로 반환될 경우 그대로 사용
            pass
        elif self.model.covariance_type == 'diag' and covars_to_save.ndim == 3:
            # (N, D, D) 형태의 대각행렬로 반환될 경우, 대각 성분만 추출하여 (N, D) 형태로 변환
            covars_to_save = np.array([np.diag(cov) for cov in covars_to_save])
        # --- ▲ [핵심 수정] ---

        return {
            'n_states': self.model.n_components,
            'covariance_type': self.model.covariance_type,
            'startprob_': self.model.startprob_.tolist(),
            'transmat_': self.model.transmat_.tolist(),
            'means_': self.model.means_.tolist(),
            'covars_': covars_to_save.tolist() # 수정된 covars_를 저장
        }

    @classmethod
    def load_from_params(cls, params: dict) -> 'RegimeAnalysisModel':
        """ 파라미터 딕셔너리로부터 모델 객체를 생성하고 복원합니다. """
        n_states = params['n_states']
        # DB에 저장된 covariance_type을 명시적으로 읽어옵니다.
        # 만약 키가 없다면, 이는 'full' 타입으로 저장된 낡은 모델이므로 'full'로 간주합니다.
        covariance_type = params.get('covariance_type', 'full') 
        
        instance = cls(n_states=n_states, covariance_type=covariance_type)
        instance.model.startprob_ = np.array(params['startprob_'])
        instance.model.transmat_ = np.array(params['transmat_'])
        instance.model.means_ = np.array(params['means_'])
        instance.model.covars_ = np.array(params['covars_'])
        instance.is_fitted = True
        
        logger.info(f"{n_states}개 상태 HMM 모델 (type: {covariance_type}) 파라미터 로드 완료.")
        return instance