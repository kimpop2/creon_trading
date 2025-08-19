import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

# --- ▼ [1. 신규 임포트] 몬테카를로 옵티마이저를 임포트합니다. ---
from optimizer.monte_carlo_optimizer import MonteCarloOptimizer
from config.settings import STRATEGY_CONFIGS
if TYPE_CHECKING:
    from trading.abstract_broker import AbstractBroker
    from analyzer.inference_service import RegimeInferenceService
    from analyzer.policy_map import PolicyMap

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    [업그레이드] HMM 기반 동적 자산배분 로직에 '제 2 두뇌'인 MonteCarloOptimizer를 탑재한 포트폴리오 관리자.
    """
    def __init__(self, broker: 'AbstractBroker', portfolio_configs: List[Dict[str, Any]],
                 inference_service: 'RegimeInferenceService', policy_map: 'PolicyMap',
                 strategy_profiles: Dict[str, Any],
                 transition_matrix: np.ndarray): # <-- [2. 수정] HMM 전이 행렬을 주입받습니다.
        self.broker = broker
        self.strategy_configs = {config['name']: config for config in portfolio_configs}
        self.inference_service = inference_service
        self.policy_map = policy_map
        self.strategy_profiles = strategy_profiles if strategy_profiles else {}
        
        # --- ▼ [3. 신규] MonteCarloOptimizer를 인스턴스화하여 탑재합니다. ---
        if self.strategy_profiles and transition_matrix is not None:
            self.optimizer = MonteCarloOptimizer(
                strategy_profiles=self.strategy_profiles,
                transition_matrix=transition_matrix,
                # 필요 시 settings.py에서 시뮬레이션 설정값을 가져올 수 있습니다.
                num_simulations=5000,
                sim_horizon_days=21
            )
            logger.info("✅ '제 2 두뇌' MonteCarloOptimizer 탑재 완료.")
        else:
            self.optimizer = None
            logger.warning("전략 프로파일 또는 전이 행렬이 없어 MonteCarloOptimizer를 초기화할 수 없습니다.")
        # --- ▲ 종료 ---

    def get_account_equity(self, current_prices: Dict[str, float]) -> float:
        cash_balance = self.broker.get_current_cash_balance()
        portfolio_value = self.broker.get_portfolio_value(current_prices)
        return cash_balance + portfolio_value

    def get_total_principal(self, account_equity: float, market_data: pd.DataFrame) -> tuple[float, np.ndarray]:
        regime_probabilities = self.inference_service.get_regime_probabilities(market_data)
        if regime_probabilities is None:
            logger.warning("장세 확률 추론 실패. 투자 비중 0으로 설정.")
            return 0.0, np.array([])
        
        principal_ratio = self.policy_map.get_target_principal_ratio(regime_probabilities)
        total_principal = account_equity * principal_ratio
        logger.info(f"HMM 기반 동적 투자 비중({principal_ratio:.0%}) 적용. 총 투자원금: {total_principal:,.0f}원")
        return total_principal, regime_probabilities

    def get_strategy_capitals(self, total_principal: float, regime_probabilities: np.ndarray) -> Dict[str, float]:
        """
        [전면 교체] '제2두뇌' MonteCarloOptimizer를 사용하여 전략별 투자금을 동적으로 배분합니다.
        """
        # --- ▼ [4. 핵심 수정] 기존 로직을 삭제하고 새로운 최적화 로직으로 교체 ---
        # 옵티마이저가 없거나, 확률 추론에 실패하면 정적 가중치 배분으로 폴백(Fallback)
        if not self.optimizer or regime_probabilities.size == 0:
            logger.warning("몬테카를로 옵티마이저를 사용할 수 없어 정적 가중치 배분으로 대체합니다.")
            capitals = {}
            active_strategies = [name for name, config in STRATEGY_CONFIGS.items() if config.get('strategy_status')]
            total_weight = sum(STRATEGY_CONFIGS[name].get('strategy_weight', 0) for name in active_strategies)
            if total_weight > 0:
                for name in active_strategies:
                    weight = STRATEGY_CONFIGS[name].get('strategy_weight', 0)
                    capitals[name] = total_principal * weight / total_weight
            return capitals

        # 1. 몬테카를로 옵티마이저 실행
        optimal_weights = self.optimizer.run_optimization(
            current_regime_probabilities=regime_probabilities
        )
        
        # 2. 최적 가중치에 따라 자본 할당
        strategy_capitals = {}
        if not optimal_weights:
            logger.warning("최적화 결과가 비어있어 자본을 할당하지 않습니다.")
            for strategy_name in self.strategy_configs.keys():
                strategy_capitals[strategy_name] = 0
            return strategy_capitals

        logger.info("동적 포트폴리오 최적 가중치:")
        for strategy_name, weight in optimal_weights.items():
            capital = total_principal * weight
            strategy_capitals[strategy_name] = capital
            logger.info(f"  - [{strategy_name}]: {weight:.2%} -> {capital:,.0f}원")
        
        return strategy_capitals
        # --- ▲ 교체 종료 ---