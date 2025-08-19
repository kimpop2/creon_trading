import logging
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any

# --- 로거 설정 ---
logger = logging.getLogger(__name__)

class MonteCarloOptimizer:
    """
    '제 2 두뇌': 몬테카를로 시뮬레이션 기반 포트폴리오 최적화 클래스.
    HMM으로 추론된 시장 국면 전환 가능성을 바탕으로 미래 시나리오를 생성하고,
    포트폴리오 전체의 샤프 지수를 극대화하는 최적의 전략별 가중치를 계산합니다.
    """

    def __init__(self,
                 strategy_profiles: Dict[str, Any],
                 transition_matrix: np.ndarray,
                 num_simulations: int = 5000,
                 sim_horizon_days: int = 21,
                 risk_free_rate: float = 0.035):
        """
        MonteCarloOptimizer를 초기화합니다.

        Args:
            strategy_profiles (Dict): 모든 국면에 대한 전략별 성과 프로파일.
                (예: {'SMADaily': {0: {'total_return': 0.15, 'mdd': 0.1, ...}, 1: {...}}, ...})
            transition_matrix (np.ndarray): HMM의 상태 전이 확률 행렬.
            num_simulations (int): 시뮬레이션 횟수.
            sim_horizon_days (int): 시뮬레이션 기간 (영업일 기준, 예: 1개월 = 21일).
            risk_free_rate (float): 연간 무위험 수익률 (샤프 지수 계산용).
        """
        self.strategy_profiles = strategy_profiles
        self.transition_matrix = transition_matrix
        self.num_simulations = num_simulations
        self.sim_horizon_days = sim_horizon_days
        # 일간 무위험 수익률로 변환
        self.daily_risk_free_rate = (1 + risk_free_rate)**(1 / 252) - 1
        
        # 최적화 대상 전략 리스트 (프로파일에 있는 전략들)
        self.strategy_names = sorted(list(self.strategy_profiles.keys()))
        self.num_strategies = len(self.strategy_names)
        
        # 빠른 조회를 위해 프로파일 데이터를 NumPy 배열로 사전 처리
        self._prepare_profile_arrays()

        logger.info(
            f"MonteCarloOptimizer 초기화 완료. "
            f"전략: {self.num_strategies}개, "
            f"시뮬레이션: {self.num_simulations}회 x {self.sim_horizon_days}일"
        )

    def _prepare_profile_arrays(self):
        """
        시뮬레이션 성능 향상을 위해 프로파일 데이터를 NumPy 배열로 변환하여 저장합니다.
        배열 차원: (전략 수, 국면 수)
        """
        num_regimes = self.transition_matrix.shape[0]
        self.mean_returns = np.zeros((self.num_strategies, num_regimes))
        self.std_devs = np.zeros((self.num_strategies, num_regimes))

        for i, name in enumerate(self.strategy_names):
            for j in range(num_regimes):
                profile = self.strategy_profiles.get(name, {}).get(j, {})
                # 일간 평균 수익률로 변환 (total_return은 기간 수익률이므로)
                # 우선 근사치로 사용. 실제로는 프로파일에 'daily_mean_return'이 있는 것이 더 정확함.
                self.mean_returns[i, j] = profile.get('total_return', 0.0) / self.sim_horizon_days
                # MDD 절댓값를 표준편차의 근사치로 사용 (더 정확한 'daily_std_dev'가 있다면 대체)
                self.std_devs[i, j] = abs(profile.get('mdd', 0.0)) / 1.96 # 95% 신뢰수준 근사

    def _generate_future_regime_paths(self, current_regime_probabilities: np.ndarray) -> np.ndarray:
        """
        현재 국면 확률에서 시작하여 미래의 국면 경로들을 생성합니다. (벡터화 방식)

        Args:
            current_regime_probabilities (np.ndarray): 현재 각 국면에 속할 확률 벡터.

        Returns:
            np.ndarray: (시뮬레이션 횟수, 시뮬레이션 기간) 형태의 국면 경로 행렬.
        """
        paths = np.zeros((self.num_simulations, self.sim_horizon_days), dtype=int)
        
        # 시작 국면 설정 (Day 0)
        initial_states = np.random.choice(
            len(current_regime_probabilities),
            size=self.num_simulations,
            p=current_regime_probabilities
        )
        paths[:, 0] = initial_states

        # 다음 국면들 생성 (Day 1 to N)
        for t in range(1, self.sim_horizon_days):
            prev_states = paths[:, t - 1]
            # 각 시뮬레이션의 이전 상태에 해당하는 전이 확률을 가져와 다음 상태를 결정
            probabilities = self.transition_matrix[prev_states]
            # cumsum 트릭을 이용한 효율적인 다항 샘플링
            # 참고: https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a
            random_uniform = np.random.rand(self.num_simulations, 1)
            cumulative_probs = probabilities.cumsum(axis=1)
            paths[:, t] = (random_uniform < cumulative_probs).argmax(axis=1)
            
        return paths

    def _simulate_portfolio_performance(self, weights: np.ndarray, regime_paths: np.ndarray) -> np.ndarray:
        """
        주어진 가중치와 국면 경로에 따라 포트폴리오의 최종 수익률 분포를 시뮬레이션합니다. (벡터화 방식)

        Args:
            weights (np.ndarray): 전략별 투자 가중치 벡터.
            regime_paths (np.ndarray): 생성된 미래 국면 경로 행렬.

        Returns:
            np.ndarray: 각 시뮬레이션별 최종 누적 수익률 벡터.
        """
        # 1. 각 경로 스텝별 평균 수익률과 표준편차 배열 생성
        # mean_returns[전략, 국면] -> mean_returns_for_paths[시뮬레이션, 기간, 전략]
        mean_returns_for_paths = self.mean_returns.T[regime_paths]
        std_devs_for_paths = self.std_devs.T[regime_paths]

        # 2. 각 전략의 일별 수익률을 무작위로 생성
        # shape: (시뮬레이션 횟수, 시뮬레이션 기간, 전략 수)
        strategy_daily_returns = np.random.normal(
            loc=mean_returns_for_paths,
            scale=std_devs_for_paths
        )

        # 3. 포트폴리오의 일별 수익률 계산 (행렬 곱)
        # (시뮬레이션, 기간, 전략수) @ (전략수,) -> (시뮬레이션, 기간)
        portfolio_daily_returns = strategy_daily_returns @ weights

        # 4. 시뮬레이션별 최종 누적 수익률 계산
        # (1 + 일별 수익률)을 모두 곱한 후 1을 빼줌
        final_returns = np.prod(1 + portfolio_daily_returns, axis=1) - 1
        
        return final_returns

    def _calculate_objective(self, weights: np.ndarray, regime_paths: np.ndarray) -> float:
        """
        최적화의 목표 함수. 샤프 지수를 계산하여 음수 값을 반환 (최소화 문제로 변환).

        Args:
            weights (np.ndarray): 전략별 투자 가중치.
            regime_paths (np.ndarray): 생성된 미래 국면 경로 행렬.

        Returns:
            float: -(샤프 지수) 값.
        """
        # 시뮬레이션을 통해 포트폴리오의 최종 수익률 분포를 얻음
        simulated_returns = self._simulate_portfolio_performance(weights, regime_paths)
        
        # 수익률 분포에서 평균과 표준편차 계산
        mean_return = np.mean(simulated_returns)
        std_dev = np.std(simulated_returns)

        if std_dev == 0:
            return 0  # 수익률이 일정하면 샤프 지수는 0

        # 샤프 지수 계산 (기간 조정)
        # 지금 수익률은 sim_horizon_days 기간 동안의 수익률이므로 연율화는 불필요.
        # 대신 무위험 수익률도 기간에 맞게 조정
        period_risk_free_rate = (1 + self.daily_risk_free_rate)**self.sim_horizon_days - 1
        sharpe_ratio = (mean_return - period_risk_free_rate) / std_dev
        
        # scipy.optimize.minimize는 최소값을 찾으므로, 샤프 지수 최대화를 위해 음수 반환
        return -sharpe_ratio

    def run_optimization(self, current_regime_probabilities: np.ndarray) -> Dict[str, float]:
        """
        몬테카를로 기반 포트폴리오 최적화를 실행합니다.

        Args:
            current_regime_probabilities (np.ndarray): 현재 시장 국면 확률 벡터.

        Returns:
            Dict[str, float]: 최적화된 전략별 가중치 딕셔너리.
        """
        if self.num_strategies == 0:
            logger.warning("최적화를 진행할 전략이 없습니다.")
            return {}

        logger.info("몬테카를로 최적화를 시작합니다...")
        
        # 1. 미래 국면 경로 일괄 생성
        regime_paths = self._generate_future_regime_paths(current_regime_probabilities)
        
        # 2. 최적화 설정
        # 초기 가중치: 모든 전략에 동일하게 배분
        initial_weights = np.array([1 / self.num_strategies] * self.num_strategies)
        
        # 제약 조건: 모든 가중치의 합은 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        # 경계 조건: 각 가중치는 0과 1 사이
        bounds = tuple((0, 1) for _ in range(self.num_strategies))

        # 3. 최적화 실행
        result = minimize(
            fun=self._calculate_objective,  # 목표 함수
            x0=initial_weights,             # 초기 추정치
            args=(regime_paths,),           # 목표 함수에 전달될 추가 인자
            method='SLSQP',                 # 제약 조건이 있는 문제에 적합한 알고리즘
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.warning(f"최적화 실패: {result.message}. 초기 가중치를 반환합니다.")
            # 실패 시 균등 가중치 또는 다른 폴백(Fallback) 로직 적용 가능
            best_weights = initial_weights
        else:
            best_weights = result.x
            final_sharpe = -result.fun
            logger.info(f"최적화 성공. 예상 샤프 지수: {final_sharpe:.4f}")

        # 가중치가 매우 작은 값(예: 1e-10)들을 0으로 정리
        best_weights[best_weights < 1e-5] = 0
        best_weights /= np.sum(best_weights) # 정규화

        # 결과를 딕셔너리 형태로 변환하여 반환
        optimized_weights = {name: weight for name, weight in zip(self.strategy_names, best_weights)}
        
        return optimized_weights