# optimizer/portfolio_optimizer.py

import itertools
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Any
import sys
import os
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize
# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 필요한 모듈 임포트 ---
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from trading.hmm_backtest import HMMBacktest
# [변경] 모든 전략 클래스를 임포트해야 동적으로 생성 가능
from strategies.sma_daily import SMADaily
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.breakout_daily import BreakoutDaily
from strategies.closing_bet_daily import ClosingBetDaily
from strategies.pass_minute import PassMinute

from config.settings import (
    HMM_OPTIMIZATION_PARAMS, 
    STRATEGY_CONFIGS  # [추가] STRATEGY_CONFIGS 임포트
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, backtest_manager: BacktestManager, initial_cash: float):
        self.backtest_manager = backtest_manager
        self.initial_cash = initial_cash
        self.backtest = HMMBacktest(
            manager=self.backtest_manager, initial_cash=self.initial_cash,
            start_date=None, end_date=None, save_to_db=False
        )
        self.is_data_prepared = False
        self.param_map = []
        
        # 포트폴리오 옵티마이저는 항상 PassMinute를 사용
        minute_strategy = PassMinute(
            broker=self.backtest.broker,
            data_store=self.backtest.data_store
        )
        self.backtest.set_strategies(daily_strategies=[], minute_strategy=minute_strategy)


    def _prepare_data_once(self, start_date, end_date):
        if self.is_data_prepared: return
        logger.info("최적화를 위한 데이터 사전 로딩을 시작합니다 (최초 1회 실행)...")
        self.backtest.start_date = start_date
        self.backtest.end_date = end_date
        self.backtest.prepare_for_system()
        self.is_data_prepared = True
        logger.info("데이터 사전 로딩 완료.")


    def _generate_grid_combinations(self, num_intervals=4) -> List[Dict[str, Any]]:
        """
        [핵심 변경] HMM 정책 파라미터에 대해서만 조합을 생성합니다.
        """
        def get_values(p_config):
            if 'values' in p_config: return p_config['values']
            step = p_config.get('step', 1)
            dtype = int if isinstance(step, int) else float
            values = np.linspace(p_config['min'], p_config['max'], num_intervals, dtype=dtype)
            return np.unique(values)

        logger.info("HMM 정책 파라미터 최적화를 위한 조합을 생성합니다.")
        hmm_params_config = HMM_OPTIMIZATION_PARAMS
        hmm_param_names = list(hmm_params_config.keys())
        hmm_value_lists = [get_values(hmm_params_config[name]) for name in hmm_param_names]
        hmm_combinations = [dict(zip(hmm_param_names, values)) for values in itertools.product(*hmm_value_lists)]
        
        # HMM 파라미터만 있으므로, 바로 HMM 모드 구조로 반환
        final_combinations = [{'hmm_params': combo} for combo in hmm_combinations]
        
        logger.info(f"총 {len(final_combinations)}개의 HMM 정책 파라미터 조합 생성 완료.")
        return final_combinations


    def _run_single_backtest(self, params: Dict[str, Any], start_date: date, end_date: date, model_name: str) -> Dict[str, Any]:
        try:
            hmm_params_for_run = params.get('hmm_params', {})
            
            daily_strategies_list = []
            for strategy_name, config in STRATEGY_CONFIGS.items():
                if config.get('strategy_status') is True: # 'strategy_status'가 True인 전략만
                    # [수정] globals()를 사용해 문자열 이름으로 클래스 객체를 동적으로 찾음
                    strategy_class = globals().get(strategy_name)
                
                if strategy_class:
                    instance = strategy_class(
                        broker=self.backtest.broker, 
                        data_store=self.backtest.data_store
                    )
                    # settings.py에서 해당 전략의 default_params를 가져와 할당합니다.
                    # 이렇게 하면 전략이 자신의 기본 파라미터로 동작하게 됩니다.
                    default_params = STRATEGY_CONFIGS.get(strategy_name, {}).get('default_params', {})
                    if not default_params:
                        logger.warning(f"'{strategy_name}'의 default_params를 settings.py에서 찾을 수 없습니다.")
                    instance.strategy_params = default_params

                    daily_strategies_list.append(instance)
                else:
                    logger.warning(f"전략 클래스 '{strategy_name}'를 찾을 수 없습니다. 임포트되었는지 확인하세요.")

            _, metrics, _, _ = self.backtest.reset_and_rerun(
                daily_strategies=daily_strategies_list,
                minute_strategy=self.backtest.minute_strategy,
                mode='hmm',
                hmm_params=hmm_params_for_run,
                model_name=model_name
            )
            
            if not metrics: raise ValueError("백테스트 결과가 유효하지 않습니다.")
            return {'params': params, 'metrics': metrics, 'success': True}
        except Exception as e:
            logger.error(f"백테스트 실패 (파라미터: {params}): {e}", exc_info=False)
            return {'params': params, 'metrics': {}, 'success': False}


    def run_hybrid_optimization(self, start_date: date, end_date: date, model_name: str):
        logger.info(f"포트폴리오 HMM 정책 최적화를 시작합니다. (모델: {model_name})")
        self._prepare_data_once(start_date, end_date)
        
        grid_best_result = self._run_broad_grid_search(start_date, end_date, model_name)
        if grid_best_result is None: return None

        bayesian_best_result = self._run_refined_bayesian_search(grid_best_result['params'], start_date, end_date, model_name)
        
        logger.info("\n" + "="*50 + "\n--- 최종 최적화 결과 비교 ---\n" + "="*50)
        grid_score = grid_best_result['metrics']['sharpe_ratio']
        bayesian_score = bayesian_best_result['metrics']['sharpe_ratio']
        
        logger.info(f"그리드 서치 최고 샤프 지수: {grid_score:.3f}")
        logger.info(f"베이지안 정밀 탐색 최고 샤프 지수: {bayesian_score:.3f}")
        
        final_best = bayesian_best_result if bayesian_best_result['metrics']['sharpe_ratio'] > grid_best_result['metrics']['sharpe_ratio'] else grid_best_result
        logger.info(f"최종 선택: {'베이지안' if bayesian_score > grid_score else '그리드 서치'} 결과")
        logger.info(f"최종 최적 파라미터: {final_best['params']}")
        logger.info(f"최종 최고 성과: {final_best['metrics']}")
        return final_best


    def _run_broad_grid_search(self, start_date, end_date, model_name):
        logger.info("\n" + "="*50 + "\n--- 1단계: 그리드 서치를 이용한 넓은 탐색 시작 ---\n" + "="*50)
        combinations = self._generate_grid_combinations(num_intervals=3)
        logger.info(f"넓은 탐색: 총 {len(combinations)}개 조합 테스트")

        results = []
        for i, p_dict in enumerate(combinations):
            logger.info(f"--- 진행률: {i+1}/{len(combinations)} ({((i+1)/len(combinations)*100):.1f}%) ---")
            results.append(self._run_single_backtest(p_dict, start_date, end_date, model_name))
        
        successful_results = [r for r in results if r['success']]
        if not successful_results:
            logger.error("넓은 탐색 단계에서 성공한 결과가 없습니다.")
            return None
            
        best_result = max(successful_results, key=lambda x: x['metrics'].get('sharpe_ratio', -999))
        logger.info(f"1단계 최고 성과: 샤프지수 {best_result['metrics']['sharpe_ratio']:.2f}, 파라미터: {best_result['params']}")
        return best_result


    def _run_refined_bayesian_search(self, best_params_from_grid, start_date, end_date, model_name, n_initial=10, n_iter=20):
        logger.info("\n" + "="*50 + "\n--- 2단계: 베이지안을 이용한 정밀 탐색 시작 ---\n" + "="*50)
        
        # [변경] param_config를 HMM_OPTIMIZATION_PARAMS로 고정
        param_config = HMM_OPTIMIZATION_PARAMS
        
        # HMM 파라미터만 있으므로 구조가 단순함
        hmm_params_for_bounds = best_params_from_grid.get('hmm_params', {})
        
        # [핵심 수정 1] 숫자형 파라미터와 범주형 파라미터를 분리합니다.
        bounds, self.param_map = [], []
        fixed_categorical_params = {}

        for name, p_conf in param_config.items():
            best_val = hmm_params_for_bounds.get(name)
            if best_val is None:
                raise KeyError(f"그리드 서치 결과에서 '{name}' 키를 찾을 수 없습니다.")

            # 숫자형 파라미터인 경우 (min/max/step 존재)
            if 'min' in p_conf:
                step = p_conf['step']
                lower_bound = max(p_conf['min'], best_val - step * 5)
                upper_bound = min(p_conf['max'], best_val + step * 5)
                bounds.append((lower_bound, upper_bound))
                self.param_map.append({'name': name, 'type': int if isinstance(step, int) else float})
            # 범주형 파라미터인 경우 ('values' 존재)
            elif 'values' in p_conf:
                # 그리드 서치에서 찾은 최적의 값을 상수로 고정
                fixed_categorical_params[name] = best_val
        
        X_observed, y_observed = [], []
        
        # ... (이하 베이지안 로직은 기존과 거의 동일하게 재사용)
        initial_points = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(n_initial, len(bounds)))
        for i, x_init in enumerate(initial_points):
            # [핵심 수정 2] 숫자형 파라미터와 고정된 범주형 파라미터를 결합
            hmm_params = self._vector_to_params(x_init)
            hmm_params.update(fixed_categorical_params)
            params_for_backtest = {'hmm_params': hmm_params}
            result = self._run_single_backtest(params_for_backtest, start_date, end_date, model_name)
            if result['success']:
                X_observed.append(x_init)
                y_observed.append(result['metrics']['sharpe_ratio'])
        
        for i in range(n_iter):
            if not X_observed or not y_observed: continue
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=10)
            gp.fit(np.array(X_observed), np.array(y_observed))
            next_point = self._propose_next_point(gp, bounds)
            # [핵심 수정 2] 숫자형 파라미터와 고정된 범주형 파라미터를 결합
            hmm_params = self._vector_to_params(next_point)
            hmm_params.update(fixed_categorical_params)
            params_for_backtest = {'hmm_params': hmm_params}
             
            # [수정] model_name 인자 추가
            result = self._run_single_backtest(params_for_backtest, start_date, end_date, model_name)
            if result['success']:
                X_observed.append(next_point)
                y_observed.append(result['metrics']['sharpe_ratio'])

        if not y_observed:
            return self._run_single_backtest(best_params_from_grid, start_date, end_date, model_name)

        best_idx = np.argmax(y_observed)
        best_params = {'hmm_params': self._vector_to_params(X_observed[best_idx])}
        # [핵심 수정 2] 숫자형 파라미터와 고정된 범주형 파라미터를 결합
        best_params.update(fixed_categorical_params)

        final_params_structure = {'hmm_params': best_params}
        best_metrics = self._run_single_backtest(final_params_structure, start_date, end_date, model_name)['metrics']
        
        logger.info(f"2단계 최고 성과: 샤프지수 {best_metrics['sharpe_ratio']:.2f}, 파라미터: {final_params_structure}")
        return {'params': final_params_structure, 'metrics': best_metrics}
    

    def _propose_next_point(self, gp, bounds):
        def ei(x, gp, y_max):
            mean, std = gp.predict(x.reshape(1, -1), return_std=True)
            std = std[0]
            if std == 0: return 0
            z = (mean - y_max) / std
            return (mean - y_max) * norm.cdf(z) + std * norm.pdf(z)

        y_max = np.max(gp.y_train_)
        res = minimize(lambda x: -ei(x, gp, y_max), 
                       x0=np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds]),
                       bounds=bounds, method='L-BFGS-B')
        return res.x

    def _vector_to_params(self, vector):
        params = {}
        for i, p_info in enumerate(self.param_map):
            val = vector[i]
            params[p_info['name']] = p_info['type'](round(val) if p_info['type'] == int else val)
        return params

