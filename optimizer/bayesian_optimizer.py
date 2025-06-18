"""
Bayesian Optimization Strategy
베이지안 최적화를 통한 파라미터 최적화 엔진
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import json
import os
import sys

# 베이지안 최적화 라이브러리
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
    from scipy.optimize import minimize
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logging.warning("베이지안 최적화를 위한 라이브러리가 설치되지 않았습니다. pip install scikit-learn scipy")

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from optimizer.progressive_refinement_optimizer import OptimizationStrategy

logger = logging.getLogger(__name__)

class BayesianOptimizationStrategy(OptimizationStrategy):
    """
    베이지안 최적화 기반 파라미터 최적화 전략
    """
    
    def __init__(self, n_initial_points: int = 10, n_iterations: int = 50):
        """
        Args:
            n_initial_points: 초기 랜덤 탐색 포인트 수
            n_iterations: 베이지안 최적화 반복 횟수
        """
        if not BAYESIAN_AVAILABLE:
            raise ImportError("베이지안 최적화를 위한 라이브러리가 필요합니다.")
        
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.gp = None
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -np.inf
        
    def get_strategy_name(self) -> str:
        return "Bayesian Optimization"
    
    def get_default_refinement_config(self) -> Dict[str, Any]:
        """기본 파라미터 범위 설정"""
        return {
            'strategy_params': {
                'sma_daily': {
                    'parameter_ranges': {
                        'short_sma_period': [2, 8],        # 실제 신호 발생 가능한 범위
                        'long_sma_period': [10, 25],       # 실제 신호 발생 가능한 범위
                        'volume_ma_period': [3, 15],       # 실제 신호 발생 가능한 범위
                        'num_top_stocks': [2, 8],          # 실제 신호 발생 가능한 범위
                    }
                },
                'rsi_daily': {
                    'parameter_ranges': {
                        'rsi_period': [10, 60],           # 연속 범위
                        'oversold_level': [20, 40],       # 연속 범위
                        'overbought_level': [60, 80],     # 연속 범위
                        'volume_ma_period': [2, 20],      # 연속 범위
                        'num_top_stocks': [2, 15],        # 연속 범위
                    }
                }
            },
            'common_params': {
                'stop_loss': [-8.0, -2.0],               # 실제 사용 가능한 범위
                'trailing_stop': [-6.0, -1.0],           # 실제 사용 가능한 범위
                'max_losing_positions': [2, 6]           # 실제 사용 가능한 범위
            },
            'minute_params': {
                'rsi_minute': {
                    'parameter_ranges': {
                        'minute_rsi_period': [30, 60],    # 실제 신호 발생 가능한 범위
                        'minute_rsi_oversold': [20, 35],  # 실제 신호 발생 가능한 범위
                        'minute_rsi_overbought': [65, 80] # 실제 신호 발생 가능한 범위
                    }
                },
                'open_minute': {
                    'parameter_ranges': {
                        'minute_rsi_period': [30, 60],    # 실제 신호 발생 가능한 범위
                        'minute_rsi_oversold': [20, 35],  # 실제 신호 발생 가능한 범위
                        'minute_rsi_overbought': [65, 80] # 실제 신호 발생 가능한 범위
                    }
                }
            }
        }
    
    def generate_parameter_combinations(self, 
                                      current_best_params: Dict[str, Any], 
                                      refinement_level: int,
                                      total_levels: int,
                                      daily_strategy_name: str = 'sma_daily',
                                      minute_strategy_name: str = 'open_minute') -> List[Dict[str, Any]]:
        """
        베이지안 최적화를 통해 다음 테스트할 파라미터 조합 생성
        """
        config = self.get_default_refinement_config()
        
        if refinement_level == 0:
            # 초기 단계: 랜덤 샘플링으로 시작
            combinations = self._generate_random_combinations(
                config, self.n_initial_points, daily_strategy_name, minute_strategy_name
            )
        else:
            # 베이지안 최적화 단계
            combinations = self._generate_bayesian_combinations(
                config, self.n_iterations, daily_strategy_name, minute_strategy_name
            )
        
        logger.info(f"베이지안 최적화 단계 {refinement_level + 1}: {len(combinations)}개 조합 생성")
        return combinations
    
    def _generate_random_combinations(self, config: Dict[str, Any], n_samples: int, 
                                    daily_strategy: str, minute_strategy: str) -> List[Dict[str, Any]]:
        """초기 랜덤 파라미터 조합 생성"""
        combinations = []
        
        for _ in range(n_samples):
            params = {}
            
            # 일봉 전략 파라미터
            if daily_strategy == 'sma_daily':
                sma_ranges = config['strategy_params']['sma_daily']['parameter_ranges']
                params['sma_params'] = {
                    'short_sma_period': int(np.random.uniform(sma_ranges['short_sma_period'][0], 
                                                             sma_ranges['short_sma_period'][1])),
                    'long_sma_period': int(np.random.uniform(sma_ranges['long_sma_period'][0], 
                                                            sma_ranges['long_sma_period'][1])),
                    'volume_ma_period': int(np.random.uniform(sma_ranges['volume_ma_period'][0], 
                                                             sma_ranges['volume_ma_period'][1])),
                    'num_top_stocks': int(np.random.uniform(sma_ranges['num_top_stocks'][0], 
                                                           sma_ranges['num_top_stocks'][1])),
                    'safe_asset_code': 'A439870'
                }
            
            # 분봉 전략 파라미터
            if minute_strategy == 'open_minute':
                rsi_ranges = config['minute_params']['open_minute']['parameter_ranges']
                params['rsi_params'] = {
                    'minute_rsi_period': int(np.random.uniform(rsi_ranges['minute_rsi_period'][0], 
                                                              rsi_ranges['minute_rsi_period'][1])),
                    'minute_rsi_oversold': int(np.random.uniform(rsi_ranges['minute_rsi_oversold'][0], 
                                                                rsi_ranges['minute_rsi_oversold'][1])),
                    'minute_rsi_overbought': int(np.random.uniform(rsi_ranges['minute_rsi_overbought'][0], 
                                                                  rsi_ranges['minute_rsi_overbought'][1]))
                }
            
            # 공통 파라미터
            common_ranges = config['common_params']
            params['stop_loss_params'] = {
                'stop_loss_ratio': np.random.uniform(common_ranges['stop_loss'][0], 
                                                   common_ranges['stop_loss'][1]),
                'trailing_stop_ratio': np.random.uniform(common_ranges['trailing_stop'][0], 
                                                       common_ranges['trailing_stop'][1]),
                'max_losing_positions': int(np.random.uniform(common_ranges['max_losing_positions'][0], 
                                                             common_ranges['max_losing_positions'][1]))
            }
            
            combinations.append(params)
        
        return combinations
    
    def _generate_bayesian_combinations(self, config: Dict[str, Any], n_iterations: int,
                                      daily_strategy: str, minute_strategy: str) -> List[Dict[str, Any]]:
        """베이지안 최적화를 통한 파라미터 조합 생성"""
        combinations = []
        
        # 가우시안 프로세스 모델 초기화
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        for i in range(n_iterations):
            # 다음 최적 지점 찾기
            next_params = self._acquire_next_point(config, daily_strategy, minute_strategy)
            combinations.append(next_params)
        
        return combinations
    
    def _acquire_next_point(self, config: Dict[str, Any], daily_strategy: str, 
                           minute_strategy: str) -> Dict[str, Any]:
        """Acquisition Function을 사용하여 다음 최적 지점 찾기"""
        
        # 파라미터 공간 정의
        param_bounds = self._get_parameter_bounds(config, daily_strategy, minute_strategy)
        
        # Expected Improvement (EI) Acquisition Function
        def acquisition_function(x):
            x = x.reshape(1, -1)
            mean, std = self.gp.predict(x, return_std=True)
            
            # EI 계산
            if len(self.y_observed) > 0:
                best_f = max(self.y_observed)
                z = (mean - best_f) / (std + 1e-8)
                ei = (mean - best_f) * norm_cdf(z) + std * norm_pdf(z)
                return -ei  # 최소화를 위해 음수
            else:
                return -mean
        
        # 최적화 실행
        result = minimize(
            acquisition_function,
            x0=np.random.uniform([b[0] for b in param_bounds], [b[1] for b in param_bounds]),
            bounds=param_bounds,
            method='L-BFGS-B'
        )
        
        # 결과를 파라미터 딕셔너리로 변환
        return self._vector_to_params(result.x, config, daily_strategy, minute_strategy)
    
    def _get_parameter_bounds(self, config: Dict[str, Any], daily_strategy: str, 
                             minute_strategy: str) -> List[Tuple[float, float]]:
        """파라미터 경계 정의"""
        bounds = []
        
        if daily_strategy == 'sma_daily':
            sma_ranges = config['strategy_params']['sma_daily']['parameter_ranges']
            bounds.extend([
                (sma_ranges['short_sma_period'][0], sma_ranges['short_sma_period'][1]),
                (sma_ranges['long_sma_period'][0], sma_ranges['long_sma_period'][1]),
                (sma_ranges['volume_ma_period'][0], sma_ranges['volume_ma_period'][1]),
                (sma_ranges['num_top_stocks'][0], sma_ranges['num_top_stocks'][1])
            ])
        
        if minute_strategy == 'open_minute':
            rsi_ranges = config['minute_params']['open_minute']['parameter_ranges']
            bounds.extend([
                (rsi_ranges['minute_rsi_period'][0], rsi_ranges['minute_rsi_period'][1]),
                (rsi_ranges['minute_rsi_oversold'][0], rsi_ranges['minute_rsi_oversold'][1]),
                (rsi_ranges['minute_rsi_overbought'][0], rsi_ranges['minute_rsi_overbought'][1])
            ])
        
        # 공통 파라미터
        common_ranges = config['common_params']
        bounds.extend([
            (common_ranges['stop_loss'][0], common_ranges['stop_loss'][1]),
            (common_ranges['trailing_stop'][0], common_ranges['trailing_stop'][1]),
            (common_ranges['max_losing_positions'][0], common_ranges['max_losing_positions'][1])
        ])
        
        return bounds
    
    def _vector_to_params(self, x: np.ndarray, config: Dict[str, Any], 
                         daily_strategy: str, minute_strategy: str) -> Dict[str, Any]:
        """벡터를 파라미터 딕셔너리로 변환"""
        params = {}
        idx = 0
        
        if daily_strategy == 'sma_daily':
            params['sma_params'] = {
                'short_sma_period': int(x[idx]),
                'long_sma_period': int(x[idx + 1]),
                'volume_ma_period': int(x[idx + 2]),
                'num_top_stocks': int(x[idx + 3]),
                'safe_asset_code': 'A439870'
            }
            idx += 4
        
        if minute_strategy == 'open_minute':
            params['rsi_params'] = {
                'minute_rsi_period': int(x[idx]),
                'minute_rsi_oversold': int(x[idx + 1]),
                'minute_rsi_overbought': int(x[idx + 2])
            }
            idx += 3
        
        params['stop_loss_params'] = {
            'stop_loss_ratio': x[idx],
            'trailing_stop_ratio': x[idx + 1],
            'max_losing_positions': int(x[idx + 2])
        }
        
        return params
    
    def update_model(self, params: Dict[str, Any], score: float):
        """관찰된 결과로 모델 업데이트"""
        # 파라미터를 벡터로 변환
        x = self._params_to_vector(params)
        
        self.X_observed.append(x)
        self.y_observed.append(score)
        
        # 최고 성과 업데이트
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
        
        # 가우시안 프로세스 모델 재학습
        if len(self.X_observed) > 1:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp.fit(X, y)
    
    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """파라미터 딕셔너리를 벡터로 변환"""
        vector = []
        
        if 'sma_params' in params:
            sma = params['sma_params']
            vector.extend([
                sma['short_sma_period'],
                sma['long_sma_period'],
                sma['volume_ma_period'],
                sma['num_top_stocks']
            ])
        
        if 'rsi_params' in params:
            rsi = params['rsi_params']
            vector.extend([
                rsi['minute_rsi_period'],
                rsi['minute_rsi_oversold'],
                rsi['minute_rsi_overbought']
            ])
        
        if 'stop_loss_params' in params:
            stop_loss = params['stop_loss_params']
            vector.extend([
                stop_loss['stop_loss_ratio'],
                stop_loss['trailing_stop_ratio'],
                stop_loss['max_losing_positions']
            ])
        
        return np.array(vector)

# 정규분포 함수 (EI 계산용)
def norm_cdf(x):
    return 0.5 * (1 + np.tanh(x / np.sqrt(2)))

def norm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi) 