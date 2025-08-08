# manager/portfolio_manager.py
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trading.abstract_broker import AbstractBroker
    # analyzer 폴더의 클래스들을 타입 힌팅에 사용
    from analyzer.inference_service import RegimeInferenceService
    from analyzer.policy_map import PolicyMap

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    [최종본] HMM 기반 동적 자산배분 로직이 통합된 포트폴리오 관리자.
    """
    # [수정] __init__에서 strategy_profiles를 주입받도록 변경
    def __init__(self, broker: 'AbstractBroker', portfolio_configs: List[Dict[str, Any]],
                 inference_service: 'RegimeInferenceService', policy_map: 'PolicyMap',
                 strategy_profiles: Dict[str, Any] = None):
        self.broker = broker
        self.strategy_configs = {config['name']: config for config in portfolio_configs}
        self.inference_service = inference_service
        self.policy_map = policy_map
        self.strategy_profiles = strategy_profiles if strategy_profiles else {} # 주입받은 프로파일 저장
        
        logger.info("PortfolioManager가 HMM 두뇌와 함께 초기화되었습니다.")
        if self.strategy_profiles:
            logger.info(f"전략 프로파일 {len(self.strategy_profiles)}개 탑재 완료.")

    def get_account_equity(self, current_prices: Dict[str, float]) -> float:
        # 이 메서드는 제공된 최종 소스와 동일
        cash_balance = self.broker.get_current_cash_balance()
        portfolio_value = self.broker.get_portfolio_value(current_prices)
        return cash_balance + portfolio_value

    def get_total_principal(self, account_equity: float, market_data: pd.DataFrame) -> tuple[float, np.ndarray]:
        # 이 메서드는 제공된 최종 소스와 동일
        regime_probabilities = self.inference_service.get_regime_probabilities(market_data)
        if regime_probabilities is None:
            logger.warning("장세 확률 추론 실패. 투자 비중 0으로 설정.")
            return 0.0, np.array([])
        
        principal_ratio = self.policy_map.get_target_principal_ratio(regime_probabilities)
        total_principal = account_equity * principal_ratio
        logger.info(f"HMM 기반 동적 투자 비중({principal_ratio:.0%}) 적용. 총 투자원금: {total_principal:,.0f}원")
        return total_principal, regime_probabilities

    # [수정] get_strategy_capitals 메서드가 인자로 받던 strategy_profiles를 self에서 사용하도록 변경
    def get_strategy_capitals(self, total_principal: float, regime_probabilities: np.ndarray) -> Dict[str, float]:
        """
        '제2두뇌' 로직: 장세 확률과 전략 프로파일을 기반으로 전략별 투자금을 동적으로 배분합니다.
        """
        # 프로파일이 없거나(self.strategy_profiles), 확률 추론에 실패하면(regime_probabilities.size == 0)
        if not self.strategy_profiles or regime_probabilities.size == 0:
            logger.warning("장세 확률 또는 전략 프로파일이 없어 정적 가중치 배분으로 대체합니다.")
            # 비상시: 정적 가중치로 배분
            capitals = {}
            total_weight = sum(config.get('weight', 1) for config in self.strategy_configs.values())
            if total_weight > 0:
                for name, config in self.strategy_configs.items():
                    capitals[name] = total_principal * config.get('weight', 1) / total_weight
            return capitals

        # 1. 포트폴리오 내 각 전략의 기대 성과 점수 계산 (제공된 최종 소스의 우월한 로직 그대로 사용)
        expected_scores = {}
        # self.strategy_profiles를 사용하도록 변경
        for strategy_name in self.strategy_configs.keys():
            profiles_by_regime = self.strategy_profiles.get(strategy_name, {})
            score = 0.0
            for regime_id, probability in enumerate(regime_probabilities):
                # 해당 국면에 프로파일이 없으면 성과를 0으로 간주
                performance = profiles_by_regime.get(regime_id, {}).get('sharpe_ratio', 0.0)
                score += probability * performance
            
            expected_scores[strategy_name] = max(0, score) # 음수 점수는 0으로 처리
            logger.info(f"  - 전략 '{strategy_name}' 기대 점수: {expected_scores[strategy_name]:.4f}")

        # 2. 기대 점수에 비례하여 자본 할당 (제공된 최종 소스의 로직 그대로 사용)
        strategy_capitals = {}
        total_score = sum(expected_scores.values())

        if total_score > 0:
            for strategy_name, score in expected_scores.items():
                weight = score / total_score
                strategy_capitals[strategy_name] = total_principal * weight
        else:
            logger.warning("모든 전략의 기대 점수가 0 이하이므로 자본을 할당하지 않습니다.")
            for strategy_name in self.strategy_configs.keys():
                strategy_capitals[strategy_name] = 0

        return strategy_capitals