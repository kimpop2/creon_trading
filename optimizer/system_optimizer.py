# optimizer/system_optimizer.py

import itertools
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 필요한 모듈 임포트 ---
from api.creon_api import CreonAPIClient
from manager.backtest_manager import BacktestManager
from manager.db_manager import DBManager
from trading.backtest import Backtest
from trading.hmm_backtest import HMMBacktest
from strategies.sma_daily import SMADaily
from strategies.pass_minute import PassMinute
from config.settings import (
    INITIAL_CASH, 
    COMMON_OPTIMIZATION_PARAMS, HMM_OPTIMIZATION_PARAMS,
    SMA_OPTIMIZATION_PARAMS,
    SMA_DAILY_PARAMS,
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
        self.backtest = Backtest(
            manager=self.backtest_manager, initial_cash=self.initial_cash,
            start_date=None, end_date=None, save_to_db=False
        )
        self.is_data_prepared = False
        self.param_map = [] # 벡터와 파라미터명 매핑

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
        strategy_param_names = list(strategy_params_config.keys())
        strategy_value_lists = [get_values(strategy_params_config[name]) for name in strategy_param_names]
        strategy_combinations = [dict(zip(strategy_param_names, values)) for values in itertools.product(*strategy_value_lists)]

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

            # 2. 전략 인스턴스 생성
            #    (실제 구현에서는 strategy_name에 따라 다른 기본 파라미터를 로드해야 함)
            base_params = SMA_DAILY_PARAMS.copy() 
            final_strategy_params = {**COMMON_OPTIMIZATION_PARAMS, **base_params, **strategy_params_for_run}
            
            #    (실제 구현에서는 strategy_name에 따라 다른 전략 클래스를 동적으로 생성해야 함)
            daily_strategy = SMADaily(broker=self.backtest.broker, data_store=self.backtest.data_store, strategy_params=final_strategy_params)
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
        combinations = self._generate_grid_combinations(strategy_name, mode=mode, num_intervals=4)
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

    def _run_refined_bayesian_search(self, best_params_from_grid: dict, strategy_name: str, start_date, end_date, mode: str, n_initial=10, n_iter=20): # mode 인자 추가
        logger.info("\n" + "="*50 + "\n--- 2단계: 베이지안을 이용한 정밀 탐색 시작 ---\n" + "="*50)
        param_config = SMA_OPTIMIZATION_PARAMS
        
        # 1. 최고 파라미터 주변으로 새로운 탐색 공간(bounds) 정의
        bounds = []
        self.param_map = []
        for name, p_conf in param_config.items():
            best_val = best_params_from_grid[name]
            step = p_conf['step']
            lower_bound = max(p_conf['min'], best_val - step * 2)
            upper_bound = min(p_conf['max'], best_val + step * 2)
            bounds.append((lower_bound, upper_bound))
            self.param_map.append({'name': name, 'type': int if isinstance(step, int) else float})

        # 2. 베이지안 최적화 실행
        X_observed, y_observed = [], []

        # 초기 랜덤 샘플링
        logger.info(f"베이지안 최적화 초기 샘플링 {n_initial}회 시작...")
        initial_points = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(n_initial, len(bounds)))
        for i, x_init in enumerate(initial_points):
            logger.info(f"--- 초기 샘플링 진행률: {i+1}/{n_initial} ---")
            params = self._vector_to_params(x_init)
            result = self._run_single_backtest(params, start_date, end_date, strategy_name, mode)
            if result['success']:
                X_observed.append(x_init)
                y_observed.append(result['metrics']['sharpe_ratio'])

        # GP 모델 학습 및 반복
        for i in range(n_iter):
            logger.info(f"베이지안 탐색 진행률: {i+1}/{n_iter}")
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=10)
            gp.fit(np.array(X_observed), np.array(y_observed))
            
            next_point = self._propose_next_point(gp, bounds)
            params = self._vector_to_params(next_point)
            result = self._run_single_backtest(params, start_date, end_date, strategy_name, mode)

            if result['success']:
                X_observed.append(next_point)
                y_observed.append(result['metrics']['sharpe_ratio'])
        
        best_idx = np.argmax(y_observed)
        best_params = self._vector_to_params(X_observed[best_idx])
        best_metrics = self._run_single_backtest(best_params, start_date, end_date, strategy_name, mode)['metrics']
        
        logger.info(f"2단계 최고 성과: 샤프지수 {best_metrics['sharpe_ratio']:.2f}, 파라미터: {best_params}")
        return {'params': best_params, 'metrics': best_metrics}

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
    

if __name__ == '__main__':
    # --- [필수] 로깅 설정 ---
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("logs/system_optimizer_run.log", encoding='utf-8'),
                            logging.StreamHandler(sys.stdout)
                        ])
    
    # --- 1. 핵심 컴포넌트 초기화 ---
    # 실제 API 연결이 필요 없는 백테스트 전용 Manager를 사용하거나, 실제 API를 사용
    api_client = CreonAPIClient() 
    db_manager = DBManager()
    backtest_manager = BacktestManager(api_client, db_manager)

    # --- 2. 최적화 기간 및 기타 설정 ---
    start_date = datetime(2025, 6, 1).date()
    end_date = datetime(2025, 7, 1).date()
    
    # --- 3. 옵티마이저 생성 및 실행 ---
    optimizer = SystemOptimizer(
        backtest_manager=backtest_manager,
        initial_cash=INITIAL_CASH
    )
    # --- 4. HMM 시스템 최적화 실행 ---
    # 하이브리드 최적화 실행 (예: SMADaily 전략 대상)
    final_results = optimizer.run_hybrid_optimization(
        strategy_name='SMADaily', # HMM 시스템에 포함될 기본 전략
        start_date=start_date,
        end_date=end_date,
        mode='hmm' # <--- HMM 모드로 실행
    )
    # --- 5. 최종 결과 분석 및 출력 ---
    if final_results:
        logger.info("\n" + "="*50 + "\n--- 최종 최적화 결과 ---\n" + "="*50)
        logger.info(f"최고 성과 파라미터: {final_results.get('params')}")
        logger.info(f"최고 성과 지표: {final_results.get('metrics')}")