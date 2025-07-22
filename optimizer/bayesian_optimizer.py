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
from trading.backtest import Backtest
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager

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
                'dual_momentum_daily': {
                    'parameter_ranges': {
                        'momentum_period': [5, 30],        # 모멘텀 계산 기간
                        'rebalance_weekday': [0, 4],       # 월~금 (0=월요일, 4=금요일)
                        'num_top_stocks': [2, 10],         # 선택 종목 수
                    }
                },
                'triple_screen_daily': {
                    'parameter_ranges': {
                        'trend_ma_period': [15, 60],       # 장기 추세 이동평균
                        'momentum_rsi_period': [10, 25],   # RSI 기간
                        'momentum_rsi_oversold': [20, 35], # RSI 과매도 기준
                        'momentum_rsi_overbought': [65, 85], # RSI 과매수 기준
                        'volume_ma_period': [5, 20],       # 거래량 이동평균
                        'num_top_stocks': [3, 10],         # 상위 종목 수
                        'min_trend_strength': [0.01, 0.08] # 최소 추세 강도
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
            elif daily_strategy == 'dual_momentum_daily':
                dual_ranges = config['strategy_params']['dual_momentum_daily']['parameter_ranges']
                params['dual_momentum_params'] = {
                    'momentum_period': int(np.random.uniform(dual_ranges['momentum_period'][0], 
                                                            dual_ranges['momentum_period'][1])),
                    'rebalance_weekday': int(np.random.uniform(dual_ranges['rebalance_weekday'][0], 
                                                              dual_ranges['rebalance_weekday'][1])),
                    'num_top_stocks': int(np.random.uniform(dual_ranges['num_top_stocks'][0], 
                                                           dual_ranges['num_top_stocks'][1])),
                    'safe_asset_code': 'A439870'
                }
            elif daily_strategy == 'triple_screen_daily':
                triple_ranges = config['strategy_params']['triple_screen_daily']['parameter_ranges']
                params['triple_screen_params'] = {
                    'trend_ma_period': int(np.random.uniform(triple_ranges['trend_ma_period'][0], 
                                                            triple_ranges['trend_ma_period'][1])),
                    'momentum_rsi_period': int(np.random.uniform(triple_ranges['momentum_rsi_period'][0], 
                                                                triple_ranges['momentum_rsi_period'][1])),
                    'momentum_rsi_oversold': int(np.random.uniform(triple_ranges['momentum_rsi_oversold'][0], 
                                                                  triple_ranges['momentum_rsi_oversold'][1])),
                    'momentum_rsi_overbought': int(np.random.uniform(triple_ranges['momentum_rsi_overbought'][0], 
                                                                    triple_ranges['momentum_rsi_overbought'][1])),
                    'volume_ma_period': int(np.random.uniform(triple_ranges['volume_ma_period'][0], 
                                                             triple_ranges['volume_ma_period'][1])),
                    'num_top_stocks': int(np.random.uniform(triple_ranges['num_top_stocks'][0], 
                                                           triple_ranges['num_top_stocks'][1])),
                    'min_trend_strength': np.random.uniform(triple_ranges['min_trend_strength'][0], 
                                                           triple_ranges['min_trend_strength'][1]),
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
        
        if daily_strategy == 'dual_momentum_daily':
            dual_ranges = config['strategy_params']['dual_momentum_daily']['parameter_ranges']
            bounds.extend([
                (dual_ranges['momentum_period'][0], dual_ranges['momentum_period'][1]),
                (dual_ranges['rebalance_weekday'][0], dual_ranges['rebalance_weekday'][1]),
                (dual_ranges['num_top_stocks'][0], dual_ranges['num_top_stocks'][1])
            ])
        
        if daily_strategy == 'triple_screen_daily':
            triple_ranges = config['strategy_params']['triple_screen_daily']['parameter_ranges']
            bounds.extend([
                (triple_ranges['trend_ma_period'][0], triple_ranges['trend_ma_period'][1]),
                (triple_ranges['momentum_rsi_period'][0], triple_ranges['momentum_rsi_period'][1]),
                (triple_ranges['momentum_rsi_oversold'][0], triple_ranges['momentum_rsi_oversold'][1]),
                (triple_ranges['momentum_rsi_overbought'][0], triple_ranges['momentum_rsi_overbought'][1]),
                (triple_ranges['volume_ma_period'][0], triple_ranges['volume_ma_period'][1]),
                (triple_ranges['num_top_stocks'][0], triple_ranges['num_top_stocks'][1]),
                (triple_ranges['min_trend_strength'][0], triple_ranges['min_trend_strength'][1])
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
        elif daily_strategy == 'dual_momentum_daily':
            params['dual_momentum_params'] = {
                'momentum_period': int(x[idx]),
                'rebalance_weekday': int(x[idx + 1]),
                'num_top_stocks': int(x[idx + 2]),
                'safe_asset_code': 'A439870'
            }
            idx += 3
        elif daily_strategy == 'triple_screen_daily':
            params['triple_screen_params'] = {
                'trend_ma_period': int(x[idx]),
                'momentum_rsi_period': int(x[idx + 1]),
                'momentum_rsi_oversold': int(x[idx + 2]),
                'momentum_rsi_overbought': int(x[idx + 3]),
                'volume_ma_period': int(x[idx + 4]),
                'num_top_stocks': int(x[idx + 5]),
                'min_trend_strength': x[idx + 6],
                'safe_asset_code': 'A439870'
            }
            idx += 7
        
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
        elif 'dual_momentum_params' in params:
            dual = params['dual_momentum_params']
            vector.extend([
                dual['momentum_period'],
                dual['rebalance_weekday'],
                dual['num_top_stocks']
            ])
        elif 'triple_screen_params' in params:
            triple = params['triple_screen_params']
            vector.extend([
                triple['trend_ma_period'],
                triple['momentum_rsi_period'],
                triple['momentum_rsi_oversold'],
                triple['momentum_rsi_overbought'],
                triple['volume_ma_period'],
                triple['num_top_stocks'],
                triple['min_trend_strength']
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

def run_bayesian_backtest(params: Dict[str, Any], 
                         api_client: CreonAPIClient,
                         db_manager: DBManager,
                         start_date: datetime.date, 
                         end_date: datetime.date,
                         sector_stocks: Dict[str, List[Tuple[str, str]]],
                         initial_cash: float = 10_000_000) -> Dict[str, Any]:
    """
    베이지안 최적화를 위한 백테스트 실행 함수
    
    Args:
        params: 최적화된 파라미터 조합
        api_client: Creon API 클라이언트
        db_manager: DB 매니저
        start_date: 백테스트 시작일
        end_date: 백테스트 종료일
        sector_stocks: 섹터별 종목 정보
        initial_cash: 초기 자본금
        
    Returns:
        백테스트 결과
    """
    try:
        # 백테스터 초기화 (새로운 생성자 사용)
        backtest = Backtest(
            api_client=api_client,
            db_manager=db_manager,
            initial_cash=initial_cash,
            save_to_db=True
        )
        
        # 일봉 전략 생성
        daily_strategy = None
        num_top_stocks = 5  # 기본값
        
        if 'sma_params' in params:
            from strategies.sma_daily import SMADaily
            daily_strategy = SMADaily(
                broker=backtest.broker,
                data_store=backtest.data_store,
                strategy_params=params['sma_params']
            )
            num_top_stocks = params['sma_params']['num_top_stocks']
        elif 'dual_momentum_params' in params:
            from strategies.dual_momentum_daily import DualMomentumDaily
            daily_strategy = DualMomentumDaily(
                broker=backtest.broker,
                data_store=backtest.data_store,
                strategy_params=params['dual_momentum_params']
            )
            num_top_stocks = params['dual_momentum_params']['num_top_stocks']
        elif 'triple_screen_params' in params:
            from strategies.triple_screen_daily import TripleScreenDaily
            daily_strategy = TripleScreenDaily(
                broker=backtest.broker,
                data_store=backtest.data_store,
                strategy_params=params['triple_screen_params']
            )
            num_top_stocks = params['triple_screen_params']['num_top_stocks']
        
        if daily_strategy is None:
            raise ValueError("지원하는 일봉 전략 파라미터가 없습니다.")
        
        # 분봉 전략 생성
        minute_strategy = None
        if 'rsi_params' in params:
            from strategies.rsi_minute import RSIMinute
            minute_params = {
                'num_top_stocks': num_top_stocks,
                **params['rsi_params']
            }
            minute_strategy = RSIMinute(
                broker=backtest.broker,
                data_store=backtest.data_store,
                strategy_params=minute_params
            )
        else:
            # OpenMinute 전략 사용
            from strategies.pass_minute import OpenMinute
            minute_params = {
                'num_top_stocks': num_top_stocks
            }
            minute_strategy = OpenMinute(
                broker=backtest.broker,
                data_store=backtest.data_store,
                strategy_params=minute_params
            )
        
        # 전략 설정 (새로운 방식)
        backtest.set_strategies(
            daily_strategy=daily_strategy,
            minute_strategy=minute_strategy
        )
        
        # 손절매 파라미터 설정
        if 'stop_loss_params' in params:
            backtest.set_broker_stop_loss_params(params['stop_loss_params'])
        
        # 데이터 로딩
        backtest.load_stocks(start_date, end_date)
        
        # 백테스트 실행
        portfolio_values, metrics = backtest.run(start_date, end_date)
        
        # 결과 정리
        result = {
            'params': params,
            'metrics': metrics,
            'portfolio_values': portfolio_values,
            'success': True
        }
        
        logger.info(f"베이지안 백테스트 성공: 수익률 {metrics.get('total_return', 0)*100:.2f}%, "
                   f"샤프지수 {metrics.get('sharpe_ratio', 0):.2f}")
        
        return result
        
    except Exception as e:
        logger.error(f"베이지안 백테스트 실패: {str(e)}")
        return {
            'params': params,
            'metrics': {},
            'portfolio_values': pd.DataFrame(),
            'success': False,
            'error': str(e)
        }

# def load_backtest_data_for_bayesian(backtest: Backtest, 
#                                    api_client: CreonAPIClient,
#                                    db_manager: DBManager,
#                                    start_date: datetime.date, 
#                                    end_date: datetime.date,
#                                    sector_stocks: Dict[str, List[Tuple[str, str]]]):
#     """
#     베이지안 최적화를 위한 백테스트 데이터 로딩
#     새로운 backtest.load_stocks() 방식을 사용합니다.
#     """
#     # backtest.load_stocks() 메서드를 사용하여 데이터 로딩
#     # 이 메서드는 전략 파라미터를 기반으로 필요한 기간을 자동으로 계산합니다.
#     backtest.load_stocks(start_date, end_date) 