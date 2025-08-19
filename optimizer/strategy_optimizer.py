# optimizer/system_optimizer.py

import itertools
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Any
import sys
import os
import json
# 베이지안 최적화를 위한 라이브러리 (설치 필요: pip install scikit-learn)
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
from manager.backtest_manager import BacktestManager
from manager.db_manager import DBManager
#from trading.backtest import Backtest
from trading.hmm_backtest import HMMBacktest
# [변경] 모든 전략 클래스를 임포트해야 동적으로 생성 가능
from strategies.sma_daily import SMADaily
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.breakout_daily import BreakoutDaily
from strategies.closing_bet_daily import ClosingBetDaily
from strategies.pass_minute import PassMinute
from config.settings import STRATEGY_CONFIGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StrategyOptimizer:
    def __init__(self, backtest_manager: "BacktestManager", initial_cash: float):
        self.backtest_manager = backtest_manager
        self.initial_cash = initial_cash
        self.backtest = HMMBacktest(
            manager=self.backtest_manager, initial_cash=self.initial_cash,
            start_date=None, end_date=None, save_to_db=False
        )
        self.is_data_prepared = False
        self.param_map = []
        
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


    def _generate_grid_combinations(self, strategy_name: str, num_intervals=4) -> List[Dict[str, Any]]:
        if strategy_name not in STRATEGY_CONFIGS:
            raise ValueError(f"'{strategy_name}'에 대한 설정이 STRATEGY_CONFIGS에 없습니다.")
        
        params_config = STRATEGY_CONFIGS[strategy_name].get('optimization_params', {})
        param_names = list(params_config.keys())
        value_lists = []
        for name in param_names:
            p_conf = params_config[name]
            step = p_conf.get('step', 1)
            dtype = int if isinstance(step, int) else float
            values = np.unique(np.linspace(p_conf['min'], p_conf['max'], num_intervals, dtype=dtype))
            value_lists.append(values)
        
        combinations = [dict(zip(param_names, values)) for values in itertools.product(*value_lists)]
        return [{'strategy_params': combo} for combo in combinations]


    def _run_single_backtest(self, params: Dict[str, Any], start_date: date, end_date: date, strategy_name: str) -> Dict[str, Any]:
        try:
            strategy_params = params.get('strategy_params', {})
            
            # [수정] globals()를 사용하여 동적으로 클래스 찾기
            strategy_class = globals().get(strategy_name)
            if not strategy_class:
                raise ValueError(f"전략 클래스 '{strategy_name}'를 찾을 수 없습니다. 임포트되었는지 확인하세요.")

            daily_strategy = strategy_class(
                broker=self.backtest.broker,
                data_store=self.backtest.data_store
            )
            # settings.py의 기본 파라미터 위에 최적화 파라미터를 덮어쓰기
            daily_strategy.strategy_params.update(strategy_params) 

            _, metrics, trade_log, _ = self.backtest.reset_and_rerun(
                daily_strategies=[daily_strategy],
                minute_strategy=self.backtest.minute_strategy,
                mode='strategy'
            )
            
            if not metrics: raise ValueError("백테스트 결과가 유효하지 않습니다.")
            return {'params': params, 'metrics': metrics, 'success': True}
        except Exception as e:
            logger.error(f"백테스트 실패 (파라미터: {params}): {e}", exc_info=False)
            return {'params': params, 'metrics': {}, 'success': False}


    def _run_broad_grid_search(self, strategy_name: str, start_date, end_date):
        """[수정] 불필요한 mode 인자 제거"""
        logger.info("\n" + "="*50 + "\n--- 1단계: 그리드 서치를 이용한 넓은 탐색 시작 ---\n" + "="*50)
        
        combinations = self._generate_grid_combinations(strategy_name, num_intervals=3)
        logger.info(f"넓은 탐색: 총 {len(combinations)}개 조합 테스트")

        results = []
        for i, p_dict in enumerate(combinations):
            logger.info(f"--- 진행률: {i+1}/{len(combinations)} ({((i+1)/len(combinations)*100):.1f}%) ---")
            # [수정] _run_single_backtest 호출 시 mode='strategy' 명시
            results.append(self._run_single_backtest(p_dict, start_date, end_date, strategy_name))
        
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            logger.error("넓은 탐색 단계에서 성공한 결과가 없습니다.")
            return None
            
        best_result = max(successful_results, key=lambda x: x['metrics'].get('sharpe_ratio', -999))
        logger.info(f"1단계 최고 성과: 샤프지수 {best_result['metrics']['sharpe_ratio']:.2f}, 파라미터: {best_result['params']}")
        return best_result
    
       
    def run_hybrid_optimization(self, strategy_name: str, start_date: date, end_date: date):
        """
        [수정] 불필요한 mode 인자 제거
        하이브리드 최적화 (그리드 탐색 + 베이지안 최적화)를 실행합니다.
        """
        logger.info(f"'{strategy_name}'에 대한 하이브리드 최적화를 시작합니다.")
        self._prepare_data_once(start_date, end_date)
        
        # [수정] mode 인자 없이 호출
        grid_best_result = self._run_broad_grid_search(strategy_name, start_date, end_date)
        if grid_best_result is None:
            logger.error(f"[{strategy_name}] 그리드 탐색 단계에서 유효한 결과를 찾지 못해 최적화를 중단합니다.")
            return None

        # [수정] mode 인자 없이 호출
        bayesian_best_result = self._run_refined_bayesian_search(grid_best_result['params'], strategy_name, start_date, end_date)
        if bayesian_best_result is None:
            logger.warning(f"[{strategy_name}] 베이지안 탐색 단계에서 유효한 결과를 찾지 못했습니다. 그리드 탐색 결과를 최종 결과로 사용합니다.")
            return grid_best_result

        # 최종 결과 비교
        logger.info("\n" + "="*50 + f"\n--- [{strategy_name}] 최종 최적화 결과 비교 ---\n" + "="*50)
        grid_score = grid_best_result['metrics']['sharpe_ratio']
        bayesian_score = bayesian_best_result['metrics']['sharpe_ratio']
        
        logger.info(f"그리드 서치 최고 샤프 지수: {grid_score:.3f}")
        logger.info(f"베이지안 정밀 탐색 최고 샤프 지수: {bayesian_score:.3f}")
        
        if bayesian_score > grid_score:
            final_best = bayesian_best_result
            logger.info("최종 선택: 베이지안 정밀 탐색 결과")
        else:
            final_best = grid_best_result
            logger.info("최종 선택: 그리드 서치 결과")
            
        logger.info(f"최종 최적 파라미터: {final_best['params']}")
        logger.info(f"최종 최고 성과: {final_best['metrics']}")
        return final_best


    def _run_refined_bayesian_search(self, best_params_from_grid: dict, strategy_name: str, start_date, end_date, n_initial=10, n_iter=20):
        logger.info("\n" + "="*50 + "\n--- 2단계: 베이지안을 이용한 정밀 탐색 시작 ---\n" + "="*50)
        if strategy_name not in STRATEGY_CONFIGS or 'optimization_params' not in STRATEGY_CONFIGS[strategy_name]:
            raise ValueError(f"'{strategy_name}'에 대한 최적화 파라미터가 config.settings.STRATEGY_CONFIGS에 정의되지 않았습니다.")
        
        param_config = STRATEGY_CONFIGS[strategy_name]['optimization_params']
        
        # 1단계 그리드 서치에서 항상 이 구조로 결과를 반환하기 때문입니다.
        strategy_params_for_bounds = best_params_from_grid.get('strategy_params', {})
        
        # HMM 파라미터는 그리드 서치에서 찾은 최적값으로 고정합니다. (없으면 빈 딕셔너리)
        fixed_hmm_params = best_params_from_grid.get('hmm_params', {})

        # [핵심 수정 1] 변수 파라미터와 상수 파라미터를 분리합니다.
        variable_params_config = {}
        constant_params = {}
        for name, config in param_config.items():
            if isinstance(config, dict) and 'min' in config:
                variable_params_config[name] = config
            else:
                constant_params[name] = config

        # 1. 최고 파라미터 주변으로 새로운 탐색 공간(bounds) 정의
        bounds = []
        self.param_map = []
        # [핵심 수정 2] '변수 그룹'에 대해서만 bounds와 param_map을 생성합니다.
        for name, p_conf in variable_params_config.items():
            best_val = strategy_params_for_bounds.get(name)
            if best_val is None:
                raise KeyError(f"베이지안 탐색 시작 파라미터('strategy_params')에 '{name}' 키가 없습니다.")
            
            step = p_conf['step']
            lower_bound = max(p_conf['min'], best_val - step * 2)
            upper_bound = min(p_conf['max'], best_val + step * 2)
            bounds.append((lower_bound, upper_bound))
            self.param_map.append({'name': name, 'type': int if isinstance(step, int) else float})

        # 초기 랜덤 샘플링
        logger.info(f"베이지안 최적화 초기 샘플링 {n_initial}회 시작...")
        if not self.param_map: # 최적화할 변수가 없는 경우
            logger.warning("베이지안 최적화를 진행할 변수 파라미터가 없습니다. 그리드 서치 결과를 최종 결과로 사용합니다.")
            return self._run_single_backtest(best_params_from_grid, start_date, end_date, strategy_name,  mode='strategy')

        # 2. 베이지안 최적화 실행
        X_observed, y_observed = [], []
        initial_points = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(n_initial, len(bounds)))

        for x_init in initial_points:
            strat_params = {**self._vector_to_params(x_init), **constant_params}
            result = self._run_single_backtest({'strategy_params': strat_params}, start_date, end_date, strategy_name)
            if result['success']:
                X_observed.append(x_init)
                y_observed.append(result['metrics']['sharpe_ratio'])
        
        for i in range(n_iter):
            if not X_observed: break
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=10)
            gp.fit(np.array(X_observed), np.array(y_observed))
            next_point = self._propose_next_point(gp, bounds)
            strat_params = {**self._vector_to_params(next_point), **constant_params}
            result = self._run_single_backtest({'strategy_params': strat_params}, start_date, end_date, strategy_name)
            if result['success']:
                X_observed.append(next_point)
                y_observed.append(result['metrics']['sharpe_ratio'])
        
        if not y_observed:
            logger.error("베이지안 탐색에서 유효한 결과를 얻지 못했습니다.")
            return self._run_single_backtest(best_params_from_grid, start_date, end_date, strategy_name)

        best_params = {**self._vector_to_params(X_observed[np.argmax(y_observed)]), **constant_params}
        final_params_structure = {'strategy_params': best_params}
        best_metrics_result = self._run_single_backtest(final_params_structure, start_date, end_date, strategy_name)
        
        logger.info(f"2단계 최고 성과: 샤프지수 {best_metrics_result['metrics']['sharpe_ratio']:.2f}, 파라미터: {final_params_structure}")
        return {'params': final_params_structure, 'metrics': best_metrics_result['metrics']}


    def _propose_next_point(self, gp, bounds):
        """Expected Improvement(EI)를 최대화하는 다음 지점 제안"""
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
    
    # def _vector_to_params(self, vector):
    #     """[수정] Numpy 타입을 Python 기본 타입으로 변환"""
    #     params = {}
    #     for i, p_info in enumerate(self.param_map):
    #         val = vector[i]
    #         # [핵심 수정] 파이썬 기본 타입(int, float)으로 명시적 형변환
    #         if p_info['type'] == int:
    #             params[p_info['name']] = int(round(val))
    #         else:
    #             params[p_info['name']] = float(val)
    #     return params