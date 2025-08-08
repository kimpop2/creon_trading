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
from strategies.sma_daily import SMADaily
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.breakout_daily import BreakoutDaily
from strategies.pass_minute import PassMinute
from config.settings import (
        INITIAL_CASH, PORTFOLIO_FOR_HMM_OPTIMIZATION,
        COMMON_OPTIMIZATION_PARAMS, HMM_OPTIMIZATION_PARAMS,
        SMA_OPTIMIZATION_PARAMS, DUAL_OPTIMIZATION_PARAMS, BREAKOUT_OPTIMIZATION_PARAMS,
        SMA_DAILY_PARAMS, DUAL_MOMENTUM_DAILY_PARAMS, BREAKOUT_DAILY_PARAMS,

    )
# 베이지안 최적화를 위한 라이브러리 (설치 필요: pip install scikit-learn)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemOptimizer:
    def __init__(self, backtest_manager: BacktestManager, initial_cash: float):
        self.backtest_manager = backtest_manager
        self.initial_cash = initial_cash
        self.backtest = HMMBacktest(
            manager=self.backtest_manager, initial_cash=self.initial_cash,
            start_date=None, end_date=None, save_to_db=False
        )
        self.is_data_prepared = False
        self.param_map = [] # 벡터와 파라미터명 매핑
        logging.info("옵티마이저 설정: 분봉 전략으로 PassMinute를 기본 탑재합니다.")
        minute_strategy_for_optimizer = PassMinute(
            broker=self.backtest.broker,
            data_store=self.backtest.data_store,
            strategy_params={} # 이 시점에서는 특정 파라미터 불필요
        )
        # 일봉 전략은 아직 미정이므로 빈 리스트로 설정
        self.backtest.set_strategies(
            daily_strategies=[],
            minute_strategy=minute_strategy_for_optimizer
        )

    def _prepare_data_once(self, start_date, end_date):
        if self.is_data_prepared: return
        logger.info("최적화를 위한 데이터 사전 로딩을 시작합니다 (최초 1회 실행)...")
        self.backtest.start_date = start_date
        self.backtest.end_date = end_date
        self.backtest.prepare_for_system()
        self.is_data_prepared = True
        logger.info("데이터 사전 로딩 완료.")

    def _generate_grid_combinations(self, strategy_name: str, mode: str = 'strategy', num_intervals=4) -> List[Dict[str, Any]]:
        """
        [수정됨] settings.py와 mode에 따라 파라미터 조합을 동적으로 생성합니다.
        """
        # 1. 최적화할 전략 파라미터 설정 로드
        strategy_params_config = {}
        if strategy_name == 'SMADaily':
            strategy_params_config = SMA_OPTIMIZATION_PARAMS
        elif strategy_name == 'DualMomentumDaily':
            strategy_params_config = DUAL_OPTIMIZATION_PARAMS
        elif strategy_name == 'BreakoutDaily':
            strategy_params_config = BREAKOUT_OPTIMIZATION_PARAMS
        # ... (다른 전략 추가)
        else:
            raise ValueError(f"지원하지 않는 전략 이름 또는 최적화 파라미터 설정 없음: {strategy_name}")
        
        # 헬퍼 함수 정의
        def get_values(p_config):
            if 'values' in p_config: return p_config['values']
            step = p_config.get('step', 1)
            dtype = int if isinstance(step, int) else float
            values = np.linspace(p_config['min'], p_config['max'], num_intervals, dtype=dtype)
            return np.unique(values)

        # 2. 전략 파라미터 조합 생성
        # strategy_param_names = list(strategy_params_config.keys())
        # strategy_value_lists = [get_values(strategy_params_config[name]) for name in strategy_param_names]
        # strategy_combinations = [dict(zip(strategy_param_names, values)) for values in itertools.product(*strategy_value_lists)]
        # [핵심 수정 시작]
        # 2. 변수 파라미터와 상수 파라미터를 분리
        variable_params_config = {}
        constant_params = {}
        for name, config in strategy_params_config.items():
            # config가 min/max 키를 가진 딕셔셔리이면 변수로 취급
            if isinstance(config, dict) and 'min' in config:
                variable_params_config[name] = config
            else: # 그 외 (정수, 문자열 등)는 상수로 취급
                constant_params[name] = config
        
        # 3. '변수 그룹'에 대해서만 모든 조합을 생성
        variable_param_names = list(variable_params_config.keys())
        variable_value_lists = [get_values(variable_params_config[name]) for name in variable_param_names]
        
        # 변수가 하나도 없으면 빈 딕셔너리 하나로 시작
        if not variable_value_lists:
            strategy_combinations = [{}]
        else:
            strategy_combinations = [dict(zip(variable_param_names, values)) for values in itertools.product(*variable_value_lists)]

        # 4. 생성된 모든 조합에 '상수 그룹'의 값들을 다시 합쳐줌
        if constant_params:
            for combo in strategy_combinations:
                combo.update(constant_params)
        # [핵심 수정 끝]

        if mode != 'hmm':
            logger.info(f"'{strategy_name}' 전략에 대해 {len(strategy_combinations)}개의 파라미터 조합 생성 완료.")
            return [{'strategy_params': combo} for combo in strategy_combinations]

        # 3. 'hmm' 모드인 경우, HMM 파라미터 조합 추가 생성 및 결합
        logger.info("HMM 시스템 최적화 모드로 파라미터 조합을 생성합니다.")
        hmm_params_config = HMM_OPTIMIZATION_PARAMS
        hmm_param_names = list(hmm_params_config.keys())
        hmm_value_lists = [get_values(hmm_params_config[name]) for name in hmm_param_names]
        hmm_combinations = [dict(zip(hmm_param_names, values)) for values in itertools.product(*hmm_value_lists)]

        # 4. 모든 조합 결합
        final_combinations = []
        for strat_comb in strategy_combinations:
            for hmm_comb in hmm_combinations:
                final_combinations.append({'strategy_params': strat_comb, 'hmm_params': hmm_comb})
        
        logger.info(f"총 {len(final_combinations)}개의 HMM 시스템 파라미터 조합 생성 완료.")
        return final_combinations

    def _run_single_backtest(self, params: Dict[str, Any], start_date: datetime.date, end_date: datetime.date,
                            strategy_name: str, mode: str) -> Dict[str, Any]:
        """
        [최종 수정] 통합된 Backtest 클래스를 사용하여 단일 백테스트를 실행합니다.
        mode에 따라 reset_and_rerun 메서드의 동작이 달라집니다.
        """
        try:
            # 1. 파라미터에서 실제 전략/HMM 파라미터를 추출합니다.
            strategy_params_for_run = params.get('strategy_params', {})
            hmm_params_for_run = params.get('hmm_params', {})

            if strategy_name == 'SMADaily':
                base_params = SMA_DAILY_PARAMS.copy()
            elif strategy_name == 'DualMomentumDaily':
                base_params = DUAL_MOMENTUM_DAILY_PARAMS.copy()
            elif strategy_name == 'BreakoutDaily':
                base_params = DUAL_MOMENTUM_DAILY_PARAMS.copy()
            else:
                # 다른 전략에 대한 기본 파라미터가 필요하면 여기에 추가
                base_params = {}

            final_strategy_params = {**COMMON_OPTIMIZATION_PARAMS, **base_params, **strategy_params_for_run}
            
            # 2. 전략 인스턴스 생성
            if strategy_name == 'SMADaily':
                daily_strategy = SMADaily(broker=self.backtest.broker, data_store=self.backtest.data_store, strategy_params=final_strategy_params)
            elif strategy_name == 'DualMomentumDaily':
                daily_strategy = DualMomentumDaily(broker=self.backtest.broker, data_store=self.backtest.data_store, strategy_params=final_strategy_params)
            elif strategy_name == 'BreakoutDaily':
                daily_strategy = BreakoutDaily(broker=self.backtest.broker, data_store=self.backtest.data_store, strategy_params=final_strategy_params)
            
            minute_strategy = PassMinute(broker=self.backtest.broker, data_store=self.backtest.data_store, strategy_params=final_strategy_params)

            # 3. 통합된 reset_and_rerun을 mode와 함께 호출
            _, metrics = self.backtest.reset_and_rerun(
                daily_strategies=[daily_strategy],
                minute_strategy=minute_strategy,
                mode=mode,
                hmm_params=hmm_params_for_run # hmm 모드일 때만 사용됨
            )
            
            if not metrics:
                raise ValueError("백테스트 실행 후 유효한 성과 지표(metrics)가 반환되지 않았습니다.")

            logger.info(f"백테스트 성공: 샤프지수 {metrics.get('sharpe_ratio', 0):.2f}")
            return {'params': params, 'metrics': metrics, 'success': True}
            
        except Exception as e:
            logger.error(f"백테스트 실패 (파라미터: {params}): {e}", exc_info=False)
            return {'params': params, 'metrics': {}, 'success': False}

    def _run_broad_grid_search(self, strategy_name: str, start_date, end_date, mode: str):
        logger.info("\n" + "="*50 + "\n--- 1단계: 그리드 서치를 이용한 넓은 탐색 시작 ---\n" + "="*50)
        
        # [수정] 올바른 메서드 호출
        combinations = self._generate_grid_combinations(strategy_name, mode=mode, num_intervals=3)
        logger.info(f"넓은 탐색: 총 {len(combinations)}개 조합 테스트")

        results = []
        for i, p_dict in enumerate(combinations): # 변수명을 p에서 p_dict로 변경
            logger.info(f"--- 진행률: {i+1}/{len(combinations)} ({((i+1)/len(combinations)*100):.1f}%) ---")
            results.append(self._run_single_backtest(p_dict, start_date, end_date, strategy_name, mode))
        
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            logger.error("넓은 탐색 단계에서 성공한 결과가 없습니다.")
            return None
            
        best_result = max(successful_results, key=lambda x: x['metrics'].get('sharpe_ratio', -999))
        logger.info(f"1단계 최고 성과: 샤프지수 {best_result['metrics']['sharpe_ratio']:.2f}, 파라미터: {best_result['params']}")
        return best_result

    def _run_refined_bayesian_search(self, best_params_from_grid: dict, strategy_name: str, start_date, end_date, mode: str, n_initial=10, n_iter=20):
        logger.info("\n" + "="*50 + "\n--- 2단계: 베이지안을 이용한 정밀 탐색 시작 ---\n" + "="*50)
        if strategy_name == 'SMADaily':
            param_config = SMA_OPTIMIZATION_PARAMS
        elif strategy_name == 'DualMomentumDaily':
            param_config = DUAL_OPTIMIZATION_PARAMS
        elif strategy_name == 'BreakoutDaily':
            param_config = BREAKOUT_OPTIMIZATION_PARAMS
        
        # [수정] 모드와 관계없이 'strategy_params' 키에서 전략 파라미터를 먼저 추출합니다.
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

        # 2. 베이지안 최적화 실행
        X_observed, y_observed = [], []

        # 초기 랜덤 샘플링
        logger.info(f"베이지안 최적화 초기 샘플링 {n_initial}회 시작...")
        if not self.param_map: # 최적화할 변수가 없는 경우
            logger.warning("베이지안 최적화를 진행할 변수 파라미터가 없습니다. 그리드 서치 결과를 최종 결과로 사용합니다.")
            return self._run_single_backtest(best_params_from_grid, start_date, end_date, strategy_name, mode)

        initial_points = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(n_initial, len(bounds)))
        for i, x_init in enumerate(initial_points):
            logger.info(f"--- 초기 샘플링 진행률: {i+1}/{n_initial} ---")
            
            strat_params = self._vector_to_params(x_init)
            # [핵심 수정 3] 상수 파라미터를 다시 합쳐줍니다.
            strat_params.update(constant_params)
            params_for_backtest = {'strategy_params': strat_params, 'hmm_params': fixed_hmm_params}
            
            result = self._run_single_backtest(params_for_backtest, start_date, end_date, strategy_name, mode)
            if result['success']:
                X_observed.append(x_init)
                y_observed.append(result['metrics']['sharpe_ratio'])

        # GP 모델 학습 및 반복
        for i in range(n_iter):
            logger.info(f"베이지안 탐색 진행률: {i+1}/{n_iter}")
            if not X_observed or not y_observed: continue

            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=10)
            gp.fit(np.array(X_observed), np.array(y_observed))
            
            next_point = self._propose_next_point(gp, bounds)
            
            strat_params = self._vector_to_params(next_point)
            params_for_backtest = {'strategy_params': strat_params, 'hmm_params': fixed_hmm_params}

            result = self._run_single_backtest(params_for_backtest, start_date, end_date, strategy_name, mode)
            if result['success']:
                X_observed.append(next_point)
                y_observed.append(result['metrics']['sharpe_ratio'])
        
        if not y_observed:
            logger.error("베이지안 탐색에서 유효한 결과를 얻지 못했습니다.")
            return self._run_single_backtest(best_params_from_grid, start_date, end_date, strategy_name, mode)

        best_idx = np.argmax(y_observed)
        best_params = self._vector_to_params(X_observed[best_idx])
        # [핵심 수정 3] 상수 파라미터를 다시 합쳐줍니다.
        best_params.update(constant_params)
        
        final_params_structure = {'strategy_params': best_params, 'hmm_params': fixed_hmm_params}
        best_metrics = self._run_single_backtest(final_params_structure, start_date, end_date, strategy_name, mode)['metrics']
        
        logger.info(f"2단계 최고 성과: 샤프지수 {best_metrics['sharpe_ratio']:.2f}, 파라미터: {final_params_structure}")
        return {'params': final_params_structure, 'metrics': best_metrics}

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

        
    def run_hybrid_optimization(self, strategy_name: str, start_date: datetime.date, end_date: datetime.date, mode: str = 'strategy'):
        
        logger.info(f"'{strategy_name}'에 대한 하이브리드 최적화를 '{mode}' 모드로 시작합니다.")
        self._prepare_data_once(start_date, end_date)
        
        # [수정] mode 인자 전달
        grid_best_result = self._run_broad_grid_search(strategy_name, start_date, end_date, mode)
        if grid_best_result is None: return

        # [수정] mode 인자 전달
        bayesian_best_result = self._run_refined_bayesian_search(grid_best_result['params'], strategy_name, start_date, end_date, mode)

        # 3. 최종 결과 비교
        logger.info("\n" + "="*50 + "\n--- 최종 최적화 결과 비교 ---\n" + "="*50)
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
    
# --- ▼ [신규] 전체 전략 최적화를 지휘하는 래퍼 함수 ▼ ---
def run_system_optimization(start_date: date, end_date: date, backtest_manager: BacktestManager) -> Dict[str, Dict[str, Any]]:
    """
    settings.py에 정의된 포트폴리오 내 모든 전략에 대해 최적화를 수행하고,
    전략별 '챔피언 파라미터' 딕셔너리를 반환합니다.
    """
    logger.info(f"\n{'='*60}\nSYSTEM-WIDE STRATEGY OPTIMIZATION START: {start_date} ~ {end_date}\n{'='*60}")
    
    # 1. 핵심 컴포넌트 초기화
    optimizer = SystemOptimizer(
        backtest_manager=backtest_manager,
        initial_cash=INITIAL_CASH
    )
    
    champion_params_all_strategies = {}

    # 2. 설정 파일에 정의된 모든 전략에 대해 최적화 반복 수행
    strategies_to_optimize = [config['name'] for config in PORTFOLIO_FOR_HMM_OPTIMIZATION]

    for strategy_name in strategies_to_optimize:
        logger.info(f"\n--- Optimizing Strategy: {strategy_name} ---")
        # 개별 전략 최적화는 항상 'strategy' 모드로 실행
        final_results = optimizer.run_hybrid_optimization(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            mode='strategy' 
        )
        
        if final_results:
            best_params = final_results.get('params', {}).get('strategy_params', {})
            champion_params_all_strategies[strategy_name] = best_params
            logger.info(f"✅ Champion parameters for {strategy_name}: {best_params}")
        else:
            logger.warning(f"Could not find optimal parameters for {strategy_name}.")
            # 최적화 실패 시, settings.py의 기본값을 찾아오거나 빈 딕셔너리 저장
            default_config = next((c for c in PORTFOLIO_FOR_HMM_OPTIMIZATION if c['name'] == strategy_name), {})
            champion_params_all_strategies[strategy_name] = default_config.get('params', {})
            
    logger.info(f"\n{'='*60}\nSYSTEM-WIDE STRATEGY OPTIMIZATION FINISHED\n{'='*60}")
    return champion_params_all_strategies


# [수정] __main__ 블록은 새로운 래퍼 함수를 테스트하도록 변경
if __name__ == '__main__':
    # --- [필수] 로깅 설정 ---
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("logs/system_optimizer_run.log", encoding='utf-8'),
                            logging.StreamHandler(sys.stdout)
                        ])
    
    # --- 1. 컴포넌트 초기화 ---
    api_client = CreonAPIClient() # 최적화 시 실제 API 연결 불필요
    db_manager = DBManager()
    backtest_manager = BacktestManager(api_client, db_manager)

    # --- 2. 최적화 기간 설정 ---
    start_date = date(2024, 1, 1)
    end_date = date(2024, 6, 30)

    # --- 3. 새로 추가된 래퍼 함수 호출 ---
    all_champion_params = run_system_optimization(start_date, end_date)
    
    logger.info("\n" + "="*50 + "\n--- 최종 챔피언 파라미터 요약 ---\n" + "="*50)
    logger.info(json.dumps(all_champion_params, indent=4))    

# if __name__ == '__main__':
#     # --- [필수] 로깅 설정 ---
#     logging.basicConfig(level=logging.INFO,
#                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                         handlers=[
#                             logging.FileHandler("logs/system_optimizer_run.log", encoding='utf-8'),
#                             logging.StreamHandler(sys.stdout)
#                         ])
    
#     # --- 1. 핵심 컴포넌트 초기화 ---
#     # 실제 API 연결이 필요 없는 백테스트 전용 Manager를 사용하거나, 실제 API를 사용
#     api_client = CreonAPIClient() 
#     db_manager = DBManager()
#     backtest_manager = BacktestManager(api_client, db_manager)

#     # --- 2. 최적화 기간 및 기타 설정 ---
#     start_date = datetime(2025, 1, 1).date()
#     end_date = datetime(2025, 6, 1).date()
#     backtest_manager.prepare_pykrx_data_for_period(start_date, end_date)
#     # --- 3. 옵티마이저 생성 및 실행 ---
#     optimizer = SystemOptimizer(
#         backtest_manager=backtest_manager,
#         initial_cash=INITIAL_CASH
#     )
#     # --- 4. HMM 시스템 최적화 실행 ---
#     # 하이브리드 최적화 실행 (예: SMADaily 전략 대상)
#     final_results = optimizer.run_hybrid_optimization(
#         strategy_name='SMADaily', # HMM 시스템에 포함될 기본 전략
#         start_date=start_date,
#         end_date=end_date,
#         mode='hmm' # <--- strategy or hmm 모드로 실행
#     )
#     # --- 5. 최종 결과 분석 및 출력 ---
#     if final_results:
#         logger.info("\n" + "="*50 + "\n--- 최종 최적화 결과 ---\n" + "="*50)
#         logger.info(f"최고 성과 파라미터: {final_results.get('params')}")
#         logger.info(f"최고 성과 지표: {final_results.get('metrics')}")