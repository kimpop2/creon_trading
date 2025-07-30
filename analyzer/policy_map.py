# analyzer/policy_map.py
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PolicyMap:
    """
    장세 확률에 따라 거시적 자산 배분(총 투자원금) 비율을 결정하는 정책 테이블.
    """
    def __init__(self):
        self.rules = {}

    def load_rules(self, file_path: str):
        """JSON 파일로부터 정책 규칙을 로드합니다."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            logger.info(f"정책 테이블 로드 완료: {file_path}")
        except FileNotFoundError:
            logger.error(f"정책 테이블 파일을 찾을 수 없습니다: {file_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"정책 테이블 파일의 JSON 형식이 올바르지 않습니다: {file_path}")
            raise

    def get_target_principal_ratio(self, regime_probabilities: np.ndarray) -> float:
        """
        현재 장세 확률에 따라 목표 투자원금 비율을 반환합니다.
        가장 확률이 높은 장세의 규칙을 따릅니다.
        """
        if not self.rules:
            logger.warning("정책 규칙이 로드되지 않았습니다. 기본값 1.0을 반환합니다.")
            return 1.0

        # 가장 확률이 높은 장세의 ID를 문자열로 변환 (JSON 키는 문자열)
        dominant_regime_id = str(np.argmax(regime_probabilities))
        
        # 정의된 규칙에서 투자 비중을 찾음
        ratio = self.rules.get('regime_to_principal_ratio', {}).get(dominant_regime_id)
        
        if ratio is not None:
            logger.info(f"지배적 장세 {dominant_regime_id} (확률: {regime_probabilities.max():.2%})에 따라 투자 비중 {ratio:.2f} 결정.")
            return ratio
        else:
            # 해당 장세 ID에 대한 규칙이 없으면 기본값 사용
            default_ratio = self.rules.get('default_principal_ratio', 1.0)
            logger.warning(f"ID {dominant_regime_id}에 대한 규칙이 정책 테이블에 없습니다. 기본값 {default_ratio}을(를) 반환합니다.")
            return default_ratio