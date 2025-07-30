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
    HMM 기반 동적 자산배분 로직이 통합된 포트폴리오 관리자.
    """
    def __init__(self, broker: 'AbstractBroker', portfolio_configs: List[Dict[str, Any]],
                 inference_service: 'RegimeInferenceService', policy_map: 'PolicyMap'):
        self.broker = broker
        # [수정] 정적 가중치는 이제 참고용으로만 사용될 수 있습니다.
        self.strategy_configs = {config['name']: config for config in portfolio_configs}
        self.inference_service = inference_service # HMM 추론기 주입
        self.policy_map = policy_map             # 정책 맵 주입
        logger.info("PortfolioManager가 HMM 두뇌와 함께 초기화되었습니다.")

    def get_account_equity(self, current_prices: Dict[str, float]) -> float:
        # 이 메서드는 기존과 동일
        cash_balance = self.broker.get_current_cash_balance()
        portfolio_value = self.broker.get_portfolio_value(current_prices)
        account_equity = cash_balance + portfolio_value
        logger.debug(f"총자산(Account Equity) 계산: {account_equity:,.0f}원 (현금: {cash_balance:,.0f} + 주식: {portfolio_value:,.0f})")
        return account_equity

    def get_total_principal(self, account_equity: float, market_data: pd.DataFrame) -> tuple[float, np.ndarray]:
        """
        [수정됨] '제1두뇌' 로직: HMM으로 장세 확률을 추론하고 동적 투자원금을 계산합니다.
        """
        # 1. HMM으로 현재 장세 확률 추론
        regime_probabilities = self.inference_service.get_regime_probabilities(market_data)
        if regime_probabilities is None:
            logger.warning("장세 확률 추론에 실패하여 기본 투자 비중(100%)을 사용합니다.")
            return account_equity, np.array([])

        # 2. 정책 맵으로 동적 투자 비중 결정
        principal_ratio = self.policy_map.get_target_principal_ratio(regime_probabilities)

        total_principal = account_equity * principal_ratio
        logger.info(f"HMM 기반 동적 투자 비중({principal_ratio:.0%}) 적용. 총 투자원금: {total_principal:,.0f}원")
        # 다음 단계를 위해 추론된 확률도 함께 반환
        return total_principal, regime_probabilities

    def get_strategy_capitals(self, total_principal: float, regime_probabilities: np.ndarray, strategy_profiles: dict) -> Dict[str, float]:
        """
        [신규] '제2두뇌' 로직: 장세 확률과 전략 프로파일을 기반으로 전략별 투자금을 동적으로 배분합니다.
        """
        if regime_probabilities.size == 0 or not strategy_profiles:
            logger.warning("장세 확률 또는 전략 프로파일이 없어 정적 가중치 배분으로 대체합니다.")
            # 비상시: 정적 가중치로 배분
            capitals = {}
            for name, config in self.strategy_configs.items():
                capitals[name] = total_principal * config.get('weight', 0)
            return capitals

        # 1. 포트폴리오 내 각 전략의 기대 성과 점수 계산
        expected_scores = {}
        for strategy_name, profiles_by_regime in strategy_profiles.items():
            score = 0.0
            for regime_id, probability in enumerate(regime_probabilities):
                performance = profiles_by_regime.get(regime_id, {}).get('sharpe_ratio', 0.0)
                score += probability * performance
            expected_scores[strategy_name] = max(0, score) # 음수 점수는 0으로 처리
            logger.info(f"  - 전략 '{strategy_name}' 기대 점수: {expected_scores[strategy_name]:.4f}")

        # 2. 기대 점수에 비례하여 자본 할당
        strategy_capitals = {}
        total_score = sum(expected_scores.values())

        if total_score > 0:
            for strategy_name, score in expected_scores.items():
                weight = score / total_score
                strategy_capitals[strategy_name] = total_principal * weight
        else:
            logger.warning("모든 전략의 기대 점수가 0 이하이므로 자본을 할당하지 않습니다.")
            for strategy_name in strategy_profiles.keys():
                strategy_capitals[strategy_name] = 0

        return strategy_capitals