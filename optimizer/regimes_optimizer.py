# optimizer/regimes_optimizer.py

import logging
import pandas as pd
from datetime import date
from typing import Dict, List, Any, Tuple
import sys
import os
import itertools
import numpy as np

# --- 베이지안 최적화 라이브러리 ---
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 필요한 모듈 임포트 ---
from manager.backtest_manager import BacktestManager
from trading.hmm_backtest import HMMBacktest, calculate_performance_metrics
from config.settings import STRATEGY_CONFIGS

# --- 동적 전략 클래스 로딩을 위해 모든 전략 임포트 ---
from strategies.sma_daily import SMADaily
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.breakout_daily import BreakoutDaily
from strategies.closing_bet_daily import ClosingBetDaily
from strategies.pass_minute import PassMinute

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegimesOptimizer:
    def __init__(self, backtest_manager: "BacktestManager", initial_cash: float):
        self.backtest_manager = backtest_manager
        self.initial_cash = initial_cash
        self.backtest = HMMBacktest(
            manager=self.backtest_manager, initial_cash=self.initial_cash,
            start_date=None, end_date=None, save_to_db=False
        )
        self.is_data_prepared = False
        self.param_map = [] # 베이지안 최적화를 위해 파라미터 순서를 기억
        
        minute_strategy = PassMinute(
            broker=self.backtest.broker,
            data_store=self.backtest.data_store,
        )
        self.backtest.set_strategies(daily_strategies=[], minute_strategy=minute_strategy)

    def _prepare_data_once(self, start_date, end_date):
        if self.is_data_prepared: return
        self.backtest.start_date = start_date
        self.backtest.end_date = end_date
        self.backtest.prepare_for_system()
        self.is_data_prepared = True

    def run_optimization_for_strategy(self, strategy_name: str, start_date: date, end_date: date, 
                                      regime_map: pd.DataFrame, model_name: str) -> Dict[int, Any]:
        logger.info(f"'{strategy_name}' 전략에 대한 국면별 최적화를 시작합니다 (Model: {model_name}).")
        self._prepare_data_once(start_date, end_date)
        
        n_regimes = 4
        champion_params_by_regime = {}

        for regime_id in range(n_regimes):
            logger.info(f"\n{'='*20} 국면(Regime) #{regime_id} 최적화 시작 {'='*20}")
            
            best_result_for_regime = self._run_hybrid_for_regime(
                strategy_name=strategy_name, regime_id=regime_id,
                start_date=start_date, end_date=end_date, regime_map=regime_map
            )
            
            if best_result_for_regime:
                champion_params_by_regime[regime_id] = best_result_for_regime
            else:
                logger.warning(f"국면 #{regime_id}에 대한 유효한 최적 파라미터를 찾지 못했습니다.")
        
        # --- ▼ [핵심 수정] 모든 최적화 완료 후 최종 요약 로그 호출 ---
        self._log_final_summary(champion_params_by_regime, strategy_name)
        return champion_params_by_regime

    def _run_hybrid_for_regime(self, strategy_name: str, regime_id: int, start_date: date, 
                               end_date: date, regime_map: pd.DataFrame) -> Dict[str, Any]:
        """특정 국면에 대한 하이브리드 최적화를 수행합니다."""
        
        grid_best_result = self._run_broad_grid_search_for_regime(
            strategy_name, regime_id, start_date, end_date, regime_map
        )
        if not grid_best_result:
            return None

        bayesian_best_result = self._run_refined_bayesian_search_for_regime(
            grid_best_result, # [수정] 파라미터뿐만 아니라 전체 결과 딕셔너리를 전달
            strategy_name, regime_id, start_date, end_date, regime_map
        )
        if not bayesian_best_result:
            return grid_best_result

        grid_score = grid_best_result['metrics']['sharpe_ratio']
        bayesian_score = bayesian_best_result['metrics']['sharpe_ratio']
        
        # --- ▼ [핵심 수정] 개별 비교 로그는 제거하고, 어떤 방식이 선택되었는지 정보만 추가 ---
        if bayesian_score > grid_score:
            final_best = bayesian_best_result
            final_best['chosen_method'] = '베이지안'
        else:
            final_best = grid_best_result
            final_best['chosen_method'] = '그리드'
        
        logger.info(f"국면 #{regime_id} 최적화 완료. 최종 선택: {final_best['chosen_method']} (샤프: {final_best['metrics']['sharpe_ratio']:.3f})")
        # --- ▲ [핵심 수정] 종료 ▲ ---
        return final_best

    def _run_broad_grid_search_for_regime(self, strategy_name, regime_id, start_date, end_date, regime_map):
        logger.info(f"--- [국면 #{regime_id}] 1단계: 그리드 서치를 이용한 넓은 탐색 시작 ---")
        grid_search_results = []
        combinations = self._generate_grid_combinations(strategy_name, num_intervals=3)
        logger.info(f"넓은 탐색: 총 {len(combinations)}개 조합 테스트")

        for i, params_dict in enumerate(combinations):
            logger.info(f"--- 진행률: {i+1}/{len(combinations)} ({((i+1)/len(combinations)*100):.1f}%) ---")
            portfolio_series, _, trade_log, _  = self._run_single_backtest(params_dict, start_date, end_date, strategy_name)
            if portfolio_series.empty or len(portfolio_series) < 2: continue

            regime_metrics = self._calculate_metrics_for_regime(portfolio_series, regime_map, regime_id, trade_log)
            if regime_metrics:
                grid_search_results.append({'params': params_dict, 'metrics': regime_metrics})

        if not grid_search_results:
            logger.error(f"국면 #{regime_id}의 그리드 탐색에서 유효한 결과를 얻지 못했습니다.")
            return None

        best_result = max(grid_search_results, key=lambda x: x['metrics'].get('sharpe_ratio', -999))
        logger.info(f"국면 #{regime_id} 그리드 서치 최고 성과: 샤프지수 {best_result['metrics']['sharpe_ratio']:.2f}, 파라미터: {best_result['params']}")
        return best_result

    def _run_refined_bayesian_search_for_regime(self, grid_best_result: dict, strategy_name: str, regime_id: int, 
                                                 start_date, end_date, regime_map, n_initial=4, n_iter=15): # n_initial 수정
        logger.info(f"--- [국면 #{regime_id}] 2단계: 베이지안을 이용한 정밀 탐색 시작 ---")
        param_config = STRATEGY_CONFIGS[strategy_name]['optimization_params']
        best_strat_params = grid_best_result['params'].get('strategy_params', {})
        
        bounds, self.param_map = [], []
        for name, p_conf in param_config.items():
            best_val = best_strat_params.get(name)
            if best_val is None: continue
            
            step = p_conf['step']
            bound_range = step * 2
            lower_bound = max(p_conf['min'], best_val - bound_range)
            upper_bound = min(p_conf['max'], best_val + bound_range)
            bounds.append((lower_bound, upper_bound))
            self.param_map.append({'name': name, 'type': int if isinstance(step, int) else float})

        if not self.param_map:
            return None

        X_observed, y_observed = [], []
        
        # --- ▼ [핵심 수정] 그리드 서치 최고 결과를 베이지안 탐색의 첫 데이터로 추가 ---
        grid_best_vector = self._params_to_vector(best_strat_params)
        grid_best_sharpe = grid_best_result['metrics']['sharpe_ratio']
        
        X_observed.append(grid_best_vector)
        y_observed.append(grid_best_sharpe)
        logger.info(f"베이지안 탐색 시드 데이터 추가: Sharpe={grid_best_sharpe:.3f}, Params={best_strat_params}")
        # --- ▲ [핵심 수정] 종료 ▲ ---

        initial_points = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(n_initial, len(bounds)))

        for x_init in initial_points:
            params_dict = {'strategy_params': self._vector_to_params(x_init)}
            portfolio_series, _, trade_log, _ = self._run_single_backtest(params_dict, start_date, end_date, strategy_name)
            if not portfolio_series.empty:
                metrics = self._calculate_metrics_for_regime(portfolio_series, regime_map, regime_id, trade_log)
                if metrics and 'sharpe_ratio' in metrics:
                    X_observed.append(x_init)
                    y_observed.append(metrics['sharpe_ratio'])
        
        for i in range(n_iter):
            if not X_observed: break
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, n_restarts_optimizer=5, random_state=42)
            gp.fit(np.array(X_observed), np.array(y_observed))
            
            next_point = self._propose_next_point(gp, np.array(bounds))
            params_dict = {'strategy_params': self._vector_to_params(next_point)}
            portfolio_series, _, trade_log, _ = self._run_single_backtest(params_dict, start_date, end_date, strategy_name)
            
            if not portfolio_series.empty:
                metrics = self._calculate_metrics_for_regime(portfolio_series, regime_map, regime_id, trade_log)
                if metrics and 'sharpe_ratio' in metrics:
                    X_observed.append(next_point)
                    y_observed.append(metrics['sharpe_ratio'])
        
        if not y_observed:
            logger.error(f"국면 #{regime_id}의 베이지안 탐색에서 유효한 결과를 얻지 못했습니다.")
            return None

        best_idx = np.argmax(y_observed)
        best_params_vec = X_observed[best_idx]
        final_params_dict = {'strategy_params': self._vector_to_params(best_params_vec)}
        
        # 최종 확인을 위해 최고 파라미터로 다시 백테스트 및 성과 계산
        final_series, _, trade_log, _ = self._run_single_backtest(final_params_dict, start_date, end_date, strategy_name)
        final_metrics = self._calculate_metrics_for_regime(final_series, regime_map, regime_id, trade_log)

        logger.info(f"국면 #{regime_id} 베이지안 탐색 최고 성과: 샤프지수 {final_metrics.get('sharpe_ratio', -999):.2f}, 파라미터: {final_params_dict}")
        return {'params': final_params_dict, 'metrics': final_metrics}

    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """파라미터 딕셔너리를 param_map 순서에 맞는 벡터로 변환합니다."""
        vec = [params[p_info['name']] for p_info in self.param_map]
        return np.array(vec)

    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, Any]:
        params = {}
        for i, p_info in enumerate(self.param_map):
            val = vector[i]
            params[p_info['name']] = p_info['type'](round(val)) if p_info['type'] is int else float(val)
        return params

    # --- Helper Methods (system_optimizer.py에서 가져온 유틸리티 함수들) ---

    def _generate_grid_combinations(self, strategy_name: str, num_intervals=3) -> List[Dict[str, Any]]:
        params_config = STRATEGY_CONFIGS[strategy_name].get('optimization_params', {})
        param_names, value_lists = list(params_config.keys()), []
        for name in param_names:
            p_conf = params_config[name]
            step = p_conf.get('step', 1)
            values = np.unique(np.linspace(p_conf['min'], p_conf['max'], num_intervals))
            if isinstance(step, int):
                values = np.round(values / step) * step
                values = [int(v) for v in np.unique(values)]
            value_lists.append(values)
        combinations = [dict(zip(param_names, v)) for v in itertools.product(*value_lists)]
        return [{'strategy_params': combo} for combo in combinations]

    def _run_single_backtest(self, params: Dict[str, Any], start_date: date, end_date: date, strategy_name: str) -> Tuple[pd.Series, Dict]:
        try:
            strategy_params = params.get('strategy_params', {})
            strategy_class = globals().get(strategy_name)
            if not strategy_class: raise ValueError(f"전략 클래스 '{strategy_name}'를 찾을 수 없습니다.")
            daily_strategy = strategy_class(broker=self.backtest.broker, data_store=self.backtest.data_store)
            daily_strategy.strategy_params.update(strategy_params)
            return self.backtest.reset_and_rerun(
                daily_strategies=[daily_strategy], minute_strategy=self.backtest.minute_strategy, mode='strategy'
            )
        except Exception as e:
            logger.error(f"백테스트 실패 (파라미터: {params}): {e}", exc_info=False)
            return pd.Series(dtype=float), {}, []

    def _calculate_metrics_for_regime(self, portfolio_series: pd.Series, regime_map: pd.DataFrame,
                                      regime_id: int, trade_log: List[Dict[str, Any]]) -> Dict[str, float]:
        if portfolio_series.empty: return {}
        
        regime_map_ts = regime_map.set_index(pd.to_datetime(regime_map['date']))
        merged_df = pd.merge(portfolio_series.rename('PortfolioValue'), regime_map_ts[['regime_id']], left_index=True, right_index=True, how='inner')
        regime_portfolio = merged_df[merged_df['regime_id'] == regime_id]['PortfolioValue']

        if len(regime_portfolio) < 2:
            return {}

        # --- ▼ [핵심 추가] strategy_profiler.py 로직 참고하여 metrics 계산 보강 ---
        metrics = calculate_performance_metrics(regime_portfolio)
        
        # 일별 수익률 계산
        daily_returns = regime_portfolio.pct_change().dropna()
        
        # --- ▼ [4. 수정] 거래 로그를 기반으로 실제 거래 횟수를 계산합니다. ---
        if not daily_returns.empty:
            metrics['win_rate'] = (daily_returns > 0).mean()

            # 해당 국면에 속하는 날짜들만 추출
            regime_dates = set(regime_map[regime_map['regime_id'] == regime_id]['date'])

            # 거래 로그에서 해당 국면 내에 발생한 'SELL' 거래만 필터링
            regime_sell_trades = [
                trade for trade in trade_log
                if trade['trade_type'] == 'SELL' and trade['trade_datetime'].date() in regime_dates
            ]
            metrics['num_trades'] = len(regime_sell_trades) # 실제 매도 횟수를 거래 횟수로 정의
        else:
            metrics['win_rate'] = 0.0
            metrics['num_trades'] = 0

        return metrics    
    
    def _propose_next_point(self, gp, bounds, y_max=None):
        if y_max is None: y_max = np.max(gp.y_train_)
        def expected_improvement(x, gp, y_max):
            mean, std = gp.predict(x.reshape(1, -1), return_std=True)
            std = std[0]
            if std == 0: return 0
            z = (mean - y_max) / std
            return (mean - y_max) * norm.cdf(z) + std * norm.pdf(z)
        
        res = minimize(lambda x: -expected_improvement(x, gp, y_max), 
                       x0=bounds.mean(axis=1), bounds=bounds, method='L-BFGS-B')
        return res.x

    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, Any]:
        params = {}
        for i, p_info in enumerate(self.param_map):
            val = vector[i]
            params[p_info['name']] = p_info['type'](round(val)) if p_info['type'] is int else float(val)
        return params
    
    # --- ▼ [신규 추가] 최종 요약 로그 출력 메서드 ---
    def _log_final_summary(self, results: Dict[int, Any], strategy_name: str):
        """모든 국면의 최종 최적화 결과를 요약하여 출력합니다."""
        logger.info("\n\n" + "="*70)
        logger.info(f"### [{strategy_name}] 전체 국면 최종 최적화 결과 요약 ###")
        logger.info("="*70)
        
        if not results:
            logger.warning("요약할 최적화 결과가 없습니다.")
            return

        for regime_id, result in sorted(results.items()):
            params = result.get('params', {}).get('strategy_params', {})
            metrics = result.get('metrics', {})
            sharpe = metrics.get('sharpe_ratio', 0.0)
            method = result.get('chosen_method', 'N/A')
            
            logger.info(f"\n--- 국면(Regime) #{regime_id} ---")
            logger.info(f"  - 선택된 방식: {method}")
            logger.info(f"  - 최고 샤프 지수: {sharpe:.3f}")
            logger.info(f"  - 최종 파라미터: {params}")
            
        logger.info("\n" + "="*70 + "\n")    