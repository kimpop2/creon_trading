"""
Progressive Refinement Optimizer
점진적 세밀화를 통한 파라미터 최적화 엔진
다양한 최적화 기법을 전략 패턴으로 주입 가능
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import json
import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.backtest import Backtest
from api.creon_api import CreonAPIClient
from manager.backtest_manager import BacktestManager
from manager.db_manager import DBManager
from trading.backtest_report import BacktestReport

# 로깅 설정 (콘솔 + 파일)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/progressive_optimizer_run.log", encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

# 로거 설정
logger = logging.getLogger(__name__)

class OptimizationStrategy(ABC):
    """
    최적화 전략 인터페이스
    다양한 최적화 기법들이 이 인터페이스를 구현
    """
    
    @abstractmethod
    def generate_parameter_combinations(self, 
                                      current_best_params: Dict[str, Any], 
                                      refinement_level: int,
                                      total_levels: int) -> List[Dict[str, Any]]:
        """
        현재 최적 파라미터를 기반으로 다음 단계의 파라미터 조합을 생성
        
        Args:
            current_best_params: 현재까지의 최적 파라미터
            refinement_level: 현재 세밀화 단계 (0부터 시작)
            total_levels: 전체 세밀화 단계 수
            
        Returns:
            파라미터 조합 리스트
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """최적화 전략 이름 반환"""
        pass
    
    @abstractmethod
    def get_default_refinement_config(self) -> Dict[str, Any]:
        """기본 세밀화 설정 반환"""
        pass

class GridSearchStrategy(OptimizationStrategy):
    """
    그리드서치 기반 최적화 전략
    """
    
    def get_strategy_name(self) -> str:
        return "Grid Search"
    
    def get_default_refinement_config(self) -> Dict[str, Any]:
        return {
            'initial_combinations': 100,
            'refinement_ratio': 0.5,  # 각 단계마다 조합 수를 50%로 줄임
            'daily_params': {
                'sma_daily': {
                    'parameter_ranges': {
                        'short_sma_period': [1, 2, 3, 4, 5, 7, 10, 15],
                        'long_sma_period': [8, 10, 12, 15, 20, 25, 30, 45],
                        'volume_ma_period': [2, 3, 5, 10, 15],
                        'num_top_stocks': [2, 3, 5, 7, 10],
                        'safe_asset_code': ['A439870']
                    }
                },
                'dual_momentum_daily': {
                    'parameter_ranges': {
                        'momentum_period': [5, 10, 15, 20, 25, 30],
                        'rebalance_weekday': [0, 1, 2, 3, 4],  # 월~금
                        'num_top_stocks': [2, 3, 5, 7, 10],
                        'safe_asset_code': ['A439870']
                    }
                },
                'triple_screen_daily': {
                    'parameter_ranges': {
                        'trend_ma_period': [20, 30, 40, 50, 60],
                        'momentum_rsi_period': [10, 14, 20, 25],
                        'momentum_rsi_oversold': [20, 25, 30, 35],
                        'momentum_rsi_overbought': [65, 70, 75, 80, 85],
                        'volume_ma_period': [5, 10, 15, 20],
                        'num_top_stocks': [3, 5, 7, 10],
                        'min_trend_strength': [0.02, 0.03, 0.05, 0.08],
                        'safe_asset_code': ['A439870']
                    }
                },
                'rsi_daily': {
                    'parameter_ranges': {
                        'rsi_period': [14, 20, 30, 45, 60],
                        'oversold_level': [20, 25, 30, 35],
                        'overbought_level': [65, 70, 75, 80],
                        'volume_ma_period': [2, 3, 5, 10, 15],
                        'num_top_stocks': [2, 3, 5, 7, 10],
                        'safe_asset_code': ['A439870']
                    }
                },
                'macd_daily': {
                    'parameter_ranges': {
                        'fast_period': [8, 10, 12, 15, 20],
                        'slow_period': [20, 25, 30, 35, 40],
                        'signal_period': [5, 8, 10, 12, 15],
                        'volume_ma_period': [2, 3, 5, 10, 15],
                        'num_top_stocks': [2, 3, 5, 7, 10],
                        'safe_asset_code': ['A439870']
                    }
                },
                'bollinger_daily': {
                    'parameter_ranges': {
                        'bb_period': [10, 15, 20, 25, 30],
                        'bb_std_dev': [1.5, 2.0, 2.5, 3.0],
                        'volume_ma_period': [2, 3, 5, 10, 15],
                        'num_top_stocks': [2, 3, 5, 7, 10],
                        'safe_asset_code': ['A439870']
                    }
                }
            },
            'common_params': {
                'stop_loss': [-2.5, -3.0, -3.5, -4.0, -5.0, -7.0],
                'trailing_stop': [-2.5, -3.0, -3.5, -5.0],
                'max_losing_positions': [2, 3, 4, 5]
            },
            'minute_params': {
                'rsi_minute': {
                    'parameter_ranges': {
                        'minute_rsi_period': [30, 40, 45, 50, 60],
                        'minute_rsi_oversold': [20, 25, 30, 35],
                        'minute_rsi_overbought': [60, 65, 70, 80]
                    }
                },
                'open_minute': {
                    'parameter_ranges': {
                        'minute_rsi_period': [30, 40, 45, 50, 60],
                        'minute_rsi_oversold': [20, 25, 30, 35],
                        'minute_rsi_overbought': [60, 65, 70, 80]
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
        그리드서치 방식으로 파라미터 조합 생성
        세밀화 단계에 따라 범위를 점점 좁힘
        """
        config = self.get_default_refinement_config()
        
        # 전략별 파라미터 범위 가져오기
        if daily_strategy_name not in config['daily_params']:
            raise ValueError(f"지원하지 않는 일봉 전략: {daily_strategy_name}")
        
        daily_strategy_config = config['daily_params'][daily_strategy_name]
        minute_strategy_config = config['minute_params'].get(minute_strategy_name, {})
        
        # 세밀화 단계에 따른 범위 조정
        if refinement_level == 0:
            # 초기 단계: 넓은 범위
            daily_ranges = daily_strategy_config['parameter_ranges']
            minute_ranges = minute_strategy_config.get('parameter_ranges', {})
        else:
            # 세밀화 단계: 최적점 주변으로 범위 축소
            daily_ranges = self._narrow_daily_parameter_ranges(
                current_best_params, refinement_level, total_levels, daily_strategy_name
            )
            minute_ranges = self._narrow_minute_parameter_ranges(
                current_best_params, refinement_level, total_levels, minute_strategy_name
            )
        
        # 조합 수 계산 (단계별로 감소)
        base_combinations = config['initial_combinations']
        combinations_count = int(base_combinations * (config['refinement_ratio'] ** refinement_level))
        
        # 파라미터 조합 생성
        combinations = self._generate_grid_combinations(
            daily_ranges, minute_ranges, config['common_params'], 
            combinations_count, daily_strategy_name, minute_strategy_name
        )
        
        logger.info(f"Grid Search 단계 {refinement_level + 1}: {len(combinations)}개 조합 생성 "
                   f"(일봉: {daily_strategy_name}, 분봉: {minute_strategy_name})")
        return combinations
    
    def _narrow_daily_parameter_ranges(self, 
                                     best_params: Dict[str, Any], 
                                     refinement_level: int,
                                     total_levels: int,
                                     strategy_name: str) -> Dict[str, List]:
        """
        일봉 전략별 최적 파라미터 주변으로 범위를 점점 좁힘
        """
        config = self.get_default_refinement_config()
        
        if strategy_name not in config['daily_params']:
            return {}
        
        strategy_config = config['daily_params'][strategy_name]
        full_ranges = strategy_config['parameter_ranges']
        
        # 축소 비율 계산
        reduction_ratio = 0.5 ** refinement_level
        
        ranges = {}
        
        if strategy_name == 'sma_daily':
            # SMA 전략 파라미터 축소
            best_short = best_params['sma_params']['short_sma_period']
            ranges['short_sma_period'] = self._get_narrowed_range(
                best_short, full_ranges['short_sma_period'], reduction_ratio
            )
            
            best_long = best_params['sma_params']['long_sma_period']
            ranges['long_sma_period'] = self._get_narrowed_range(
                best_long, full_ranges['long_sma_period'], reduction_ratio
            )
            
            best_volume = best_params['sma_params']['volume_ma_period']
            ranges['volume_ma_period'] = self._get_narrowed_range(
                best_volume, full_ranges['volume_ma_period'], reduction_ratio
            )
            
            best_stocks = best_params['sma_params']['num_top_stocks']
            ranges['num_top_stocks'] = self._get_narrowed_range(
                best_stocks, full_ranges['num_top_stocks'], reduction_ratio
            )
            
            ranges['safe_asset_code'] = full_ranges['safe_asset_code']
            
        elif strategy_name == 'rsi_daily':
            # RSI 일봉 전략 파라미터 축소
            best_rsi_period = best_params['rsi_daily_params']['rsi_period']
            ranges['rsi_period'] = self._get_narrowed_range(
                best_rsi_period, full_ranges['rsi_period'], reduction_ratio
            )
            
            best_oversold = best_params['rsi_daily_params']['oversold_level']
            ranges['oversold_level'] = self._get_narrowed_range(
                best_oversold, full_ranges['oversold_level'], reduction_ratio
            )
            
            best_overbought = best_params['rsi_daily_params']['overbought_level']
            ranges['overbought_level'] = self._get_narrowed_range(
                best_overbought, full_ranges['overbought_level'], reduction_ratio
            )
            
            best_volume = best_params['rsi_daily_params']['volume_ma_period']
            ranges['volume_ma_period'] = self._get_narrowed_range(
                best_volume, full_ranges['volume_ma_period'], reduction_ratio
            )
            
            best_stocks = best_params['rsi_daily_params']['num_top_stocks']
            ranges['num_top_stocks'] = self._get_narrowed_range(
                best_stocks, full_ranges['num_top_stocks'], reduction_ratio
            )
            
            ranges['safe_asset_code'] = full_ranges['safe_asset_code']
            
        elif strategy_name == 'macd_daily':
            # MACD 전략 파라미터 축소
            best_fast = best_params['macd_params']['fast_period']
            ranges['fast_period'] = self._get_narrowed_range(
                best_fast, full_ranges['fast_period'], reduction_ratio
            )
            
            best_slow = best_params['macd_params']['slow_period']
            ranges['slow_period'] = self._get_narrowed_range(
                best_slow, full_ranges['slow_period'], reduction_ratio
            )
            
            best_signal = best_params['macd_params']['signal_period']
            ranges['signal_period'] = self._get_narrowed_range(
                best_signal, full_ranges['signal_period'], reduction_ratio
            )
            
            best_volume = best_params['macd_params']['volume_ma_period']
            ranges['volume_ma_period'] = self._get_narrowed_range(
                best_volume, full_ranges['volume_ma_period'], reduction_ratio
            )
            
            best_stocks = best_params['macd_params']['num_top_stocks']
            ranges['num_top_stocks'] = self._get_narrowed_range(
                best_stocks, full_ranges['num_top_stocks'], reduction_ratio
            )
            
            ranges['safe_asset_code'] = full_ranges['safe_asset_code']
            
        elif strategy_name == 'bollinger_daily':
            # 볼린저밴드 전략 파라미터 축소
            best_period = best_params['bollinger_params']['bb_period']
            ranges['bb_period'] = self._get_narrowed_range(
                best_period, full_ranges['bb_period'], reduction_ratio
            )
            
            best_std_dev = best_params['bollinger_params']['bb_std_dev']
            ranges['bb_std_dev'] = self._get_narrowed_range(
                best_std_dev, full_ranges['bb_std_dev'], reduction_ratio
            )
            
            best_volume = best_params['bollinger_params']['volume_ma_period']
            ranges['volume_ma_period'] = self._get_narrowed_range(
                best_volume, full_ranges['volume_ma_period'], reduction_ratio
            )
            
            best_stocks = best_params['bollinger_params']['num_top_stocks']
            ranges['num_top_stocks'] = self._get_narrowed_range(
                best_stocks, full_ranges['num_top_stocks'], reduction_ratio
            )
            
            ranges['safe_asset_code'] = full_ranges['safe_asset_code']
        
        elif strategy_name == 'dual_momentum_daily':
            # 듀얼모멘텀 전략 파라미터 축소
            best_momentum = best_params['dual_momentum_params']['momentum_period']
            ranges['momentum_period'] = self._get_narrowed_range(
                best_momentum, full_ranges['momentum_period'], reduction_ratio
            )
            
            best_weekday = best_params['dual_momentum_params']['rebalance_weekday']
            ranges['rebalance_weekday'] = self._get_narrowed_range(
                best_weekday, full_ranges['rebalance_weekday'], reduction_ratio
            )
            
            best_stocks = best_params['dual_momentum_params']['num_top_stocks']
            ranges['num_top_stocks'] = self._get_narrowed_range(
                best_stocks, full_ranges['num_top_stocks'], reduction_ratio
            )
            
            ranges['safe_asset_code'] = full_ranges['safe_asset_code']
        
        elif strategy_name == 'triple_screen_daily':
            # 삼중창 전략 파라미터 축소
            best_trend_ma = best_params['triple_screen_params']['trend_ma_period']
            ranges['trend_ma_period'] = self._get_narrowed_range(
                best_trend_ma, full_ranges['trend_ma_period'], reduction_ratio
            )
            
            best_rsi_period = best_params['triple_screen_params']['momentum_rsi_period']
            ranges['momentum_rsi_period'] = self._get_narrowed_range(
                best_rsi_period, full_ranges['momentum_rsi_period'], reduction_ratio
            )
            
            best_oversold = best_params['triple_screen_params']['momentum_rsi_oversold']
            ranges['momentum_rsi_oversold'] = self._get_narrowed_range(
                best_oversold, full_ranges['momentum_rsi_oversold'], reduction_ratio
            )
            
            best_overbought = best_params['triple_screen_params']['momentum_rsi_overbought']
            ranges['momentum_rsi_overbought'] = self._get_narrowed_range(
                best_overbought, full_ranges['momentum_rsi_overbought'], reduction_ratio
            )
            
            best_volume = best_params['triple_screen_params']['volume_ma_period']
            ranges['volume_ma_period'] = self._get_narrowed_range(
                best_volume, full_ranges['volume_ma_period'], reduction_ratio
            )
            
            best_stocks = best_params['triple_screen_params']['num_top_stocks']
            ranges['num_top_stocks'] = self._get_narrowed_range(
                best_stocks, full_ranges['num_top_stocks'], reduction_ratio
            )
            
            best_trend_strength = best_params['triple_screen_params']['min_trend_strength']
            ranges['min_trend_strength'] = self._get_narrowed_range(
                best_trend_strength, full_ranges['min_trend_strength'], reduction_ratio
            )
            
            ranges['safe_asset_code'] = full_ranges['safe_asset_code']
        
        return ranges
    
    def _narrow_minute_parameter_ranges(self, 
                                      best_params: Dict[str, Any], 
                                      refinement_level: int,
                                      total_levels: int,
                                      strategy_name: str) -> Dict[str, List]:
        """
        분봉 전략별 최적 파라미터 주변으로 범위를 점점 좁힘
        """
        config = self.get_default_refinement_config()
        
        if strategy_name not in config['minute_params']:
            return {}
        
        strategy_config = config['minute_params'][strategy_name]
        full_ranges = strategy_config['parameter_ranges']
        
        # 축소 비율 계산
        reduction_ratio = 0.5 ** refinement_level
        
        ranges = {}
        
        if strategy_name == 'rsi_minute':
            best_rsi_period = best_params['rsi_params']['minute_rsi_period']
            ranges['minute_rsi_period'] = self._get_narrowed_range(
                best_rsi_period, full_ranges['minute_rsi_period'], reduction_ratio
            )
            
            best_oversold = best_params['rsi_params']['minute_rsi_oversold']
            ranges['minute_rsi_oversold'] = self._get_narrowed_range(
                best_oversold, full_ranges['minute_rsi_oversold'], reduction_ratio
            )
            
            best_overbought = best_params['rsi_params']['minute_rsi_overbought']
            ranges['minute_rsi_overbought'] = self._get_narrowed_range(
                best_overbought, full_ranges['minute_rsi_overbought'], reduction_ratio
            )
        elif strategy_name == 'open_minute':
            best_rsi_period = best_params['rsi_params']['minute_rsi_period']
            ranges['minute_rsi_period'] = self._get_narrowed_range(
                best_rsi_period, full_ranges['minute_rsi_period'], reduction_ratio
            )
            
            best_oversold = best_params['rsi_params']['minute_rsi_oversold']
            ranges['minute_rsi_oversold'] = self._get_narrowed_range(
                best_oversold, full_ranges['minute_rsi_oversold'], reduction_ratio
            )
            
            best_overbought = best_params['rsi_params']['minute_rsi_overbought']
            ranges['minute_rsi_overbought'] = self._get_narrowed_range(
                best_overbought, full_ranges['minute_rsi_overbought'], reduction_ratio
            )
        
        return ranges
    
    def _get_narrowed_range(self, best_value: Any, full_range: List, reduction_ratio: float) -> List:
        """
        최적값 주변으로 범위를 축소
        """
        if best_value in full_range:
            idx = full_range.index(best_value)
            # 범위 축소 (최소 2개, 최대 4개)
            range_size = max(2, min(4, int(len(full_range) * reduction_ratio)))
            start_idx = max(0, idx - range_size // 2)
            end_idx = min(len(full_range), start_idx + range_size)
            return full_range[start_idx:end_idx]
        else:
            # 최적값이 범위에 없으면 전체 범위 반환
            return full_range
    
    def _generate_grid_combinations(self, daily_ranges: Dict[str, List], minute_ranges: Dict[str, List], common_params: Dict[str, Any], max_combinations: int, daily_strategy_name: str, minute_strategy_name: str) -> List[Dict[str, Any]]:
        """
        그리드 조합 생성 - 전략별로 동적 생성
        """
        import itertools
        
        combinations = []
        
        # 일봉 전략별 파라미터 생성
        if daily_strategy_name == 'sma_daily':
            daily_combinations = self._generate_sma_combinations(daily_ranges)
        elif daily_strategy_name == 'dual_momentum_daily':
            daily_combinations = self._generate_dual_momentum_combinations(daily_ranges)
        elif daily_strategy_name == 'triple_screen_daily':
            daily_combinations = self._generate_triple_screen_combinations(daily_ranges)
        elif daily_strategy_name == 'rsi_daily':
            daily_combinations = self._generate_rsi_daily_combinations(daily_ranges)
        elif daily_strategy_name == 'macd_daily':
            daily_combinations = self._generate_macd_combinations(daily_ranges)
        elif daily_strategy_name == 'bollinger_daily':
            daily_combinations = self._generate_bollinger_combinations(daily_ranges)
        else:
            raise ValueError(f"지원하지 않는 일봉 전략: {daily_strategy_name}")
        
        # 분봉 전략별 파라미터 생성
        if minute_strategy_name == 'open_minute':
            minute_combinations = self._generate_rsi_minute_combinations(minute_ranges)
        else:
            minute_combinations = [{}]  # 분봉 전략이 없는 경우
        
        # 공통 파라미터 조합 생성
        common_combinations = self._generate_common_combinations(common_params)
        
        # 모든 조합 생성
        for daily_params in daily_combinations:
            for minute_params in minute_combinations:
                for common_param in common_combinations:
                    combination = {
                        'daily_strategy_name': daily_strategy_name,
                        'minute_strategy_name': minute_strategy_name,
                        **daily_params,
                        **minute_params,
                        **common_param
                    }
                    combinations.append(combination)
                    
                    # 최대 조합 수 제한
                    if len(combinations) >= max_combinations:
                        return combinations
        
        return combinations
    
    def _generate_sma_combinations(self, ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """SMA 전략 조합 생성"""
        import itertools
        
        combinations = []
        for short_period, long_period, volume_period, num_stocks in itertools.product(
            ranges['short_sma_period'], ranges['long_sma_period'], 
            ranges['volume_ma_period'], ranges['num_top_stocks']):
            
            # 유효한 조합만 필터링 (단기 < 장기)
            if short_period >= long_period:
                continue
            
            combinations.append({
                'sma_params': {
                    'short_sma_period': short_period,
                    'long_sma_period': long_period,
                    'volume_ma_period': volume_period,
                    'num_top_stocks': num_stocks,
                    'safe_asset_code': ranges['safe_asset_code'][0]
                }
            })
        
        return combinations
    
    def _generate_dual_momentum_combinations(self, ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """듀얼모멘텀 전략 조합 생성"""
        import itertools
        
        combinations = []
        for momentum_period, rebalance_weekday, num_stocks in itertools.product(
            ranges['momentum_period'], ranges['rebalance_weekday'], ranges['num_top_stocks']):
            
            combinations.append({
                'dual_momentum_params': {
                    'momentum_period': momentum_period,
                    'rebalance_weekday': rebalance_weekday,
                    'num_top_stocks': num_stocks,
                    'safe_asset_code': ranges['safe_asset_code'][0]
                }
            })
        
        return combinations
    
    def _generate_triple_screen_combinations(self, ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """삼중창 전략 조합 생성"""
        import itertools
        
        combinations = []
        for trend_ma, rsi_period, oversold, overbought, volume_period, num_stocks, min_strength in itertools.product(
            ranges['trend_ma_period'], ranges['momentum_rsi_period'], 
            ranges['momentum_rsi_oversold'], ranges['momentum_rsi_overbought'],
            ranges['volume_ma_period'], ranges['num_top_stocks'], ranges['min_trend_strength']):
            
            # 유효한 조합만 필터링 (oversold < overbought)
            if oversold >= overbought:
                continue
            
            combinations.append({
                'triple_screen_params': {
                    'trend_ma_period': trend_ma,
                    'momentum_rsi_period': rsi_period,
                    'momentum_rsi_oversold': oversold,
                    'momentum_rsi_overbought': overbought,
                    'volume_ma_period': volume_period,
                    'num_top_stocks': num_stocks,
                    'min_trend_strength': min_strength,
                    'safe_asset_code': ranges['safe_asset_code'][0]
                }
            })
        
        return combinations
    
    def _generate_rsi_daily_combinations(self, ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """RSI 일봉 전략 조합 생성"""
        import itertools
        
        combinations = []
        for rsi_period, oversold, overbought, volume_period, num_stocks in itertools.product(
            ranges['rsi_period'], ranges['oversold_level'], ranges['overbought_level'],
            ranges['volume_ma_period'], ranges['num_top_stocks']):
            
            # 유효한 조합만 필터링 (oversold < overbought)
            if oversold >= overbought:
                continue
            
            combinations.append({
                'rsi_daily_params': {
                    'rsi_period': rsi_period,
                    'oversold_level': oversold,
                    'overbought_level': overbought,
                    'volume_ma_period': volume_period,
                    'num_top_stocks': num_stocks,
                    'safe_asset_code': ranges['safe_asset_code'][0]
                }
            })
        
        return combinations
    
    def _generate_macd_combinations(self, ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """MACD 전략 조합 생성"""
        import itertools
        
        combinations = []
        for fast_period, slow_period, signal_period, volume_period, num_stocks in itertools.product(
            ranges['fast_period'], ranges['slow_period'], ranges['signal_period'],
            ranges['volume_ma_period'], ranges['num_top_stocks']):
            
            # 유효한 조합만 필터링 (fast < slow)
            if fast_period >= slow_period:
                continue
            
            combinations.append({
                'macd_params': {
                    'fast_period': fast_period,
                    'slow_period': slow_period,
                    'signal_period': signal_period,
                    'volume_ma_period': volume_period,
                    'num_top_stocks': num_stocks,
                    'safe_asset_code': ranges['safe_asset_code'][0]
                }
            })
        
        return combinations
    
    def _generate_bollinger_combinations(self, ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """볼린저밴드 전략 조합 생성"""
        import itertools
        
        combinations = []
        for period, std_dev, volume_period, num_stocks in itertools.product(
            ranges['bb_period'], ranges['bb_std_dev'], 
            ranges['volume_ma_period'], ranges['num_top_stocks']):
            
            combinations.append({
                'bollinger_params': {
                    'period': period,
                    'std_dev': std_dev,
                    'volume_ma_period': volume_period,
                    'num_top_stocks': num_stocks,
                    'safe_asset_code': ranges['safe_asset_code'][0]
                }
            })
        
        return combinations
    
    def _generate_rsi_minute_combinations(self, ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """RSI 분봉 전략 조합 생성"""
        import itertools
        
        combinations = []
        for rsi_period, oversold, overbought in itertools.product(
            ranges['minute_rsi_period'], ranges['minute_rsi_oversold'], ranges['minute_rsi_overbought']):
            
            # 유효한 조합만 필터링 (oversold < overbought)
            if oversold >= overbought:
                continue
            
            combinations.append({
                'rsi_params': {
                    'minute_rsi_period': rsi_period,
                    'minute_rsi_oversold': oversold,
                    'minute_rsi_overbought': overbought
                }
            })
        
        return combinations
    
    def _generate_common_combinations(self, common_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """공통 파라미터 조합 생성"""
        import itertools
        
        combinations = []
        for stop_loss, trailing_stop, max_losing in itertools.product(
            common_params['stop_loss'], common_params['trailing_stop'], common_params['max_losing_positions']):
            
            combinations.append({
                'stop_loss_params': {
                    'stop_loss_ratio': stop_loss,
                    'trailing_stop_ratio': trailing_stop,
                    'portfolio_stop_loss': stop_loss,
                    'early_stop_loss': stop_loss,
                    'max_losing_positions': max_losing
                }
            })
        
        return combinations

class ProgressiveRefinementOptimizer:
    """
    점진적 세밀화 최적화 엔진
    다양한 최적화 전략을 주입받아 점진적으로 세밀화
    """
    
    def __init__(self, 
                 strategy: OptimizationStrategy,
                 api_client: CreonAPIClient,
                 backtest_manager: BacktestManager,
                 report: BacktestReport,
                 initial_cash: float = 10_000_000):
        """
        ProgressiveRefinementOptimizer 초기화
        
        Args:
            strategy: 최적화 전략 (GridSearch, Bayesian, Genetic 등)
            api_client: Creon API 클라이언트
            backtest_manager: 데이터 매니저
            report: 리포터
            stock_selector: 종목 선택기
            initial_cash: 초기 자본금
        """
        self.strategy = strategy
        self.api_client = api_client
        self.backtest_manager = backtest_manager
        self.report = report
        self.initial_cash = initial_cash
        
        # 최적화 결과 저장
        self.optimization_history = []
        self.final_best_result = None

        logger.info(f"ProgressiveRefinementOptimizer 초기화 완료 (전략: {strategy.get_strategy_name()})")
    
    def run_progressive_optimization(self, 
                                   start_date: datetime.date, 
                                   end_date: datetime.date,
                                   refinement_levels: int = 4,
                                   initial_combinations: int = None,
                                   daily_strategy_name: str = 'sma_daily',
                                   minute_strategy_name: str = 'open_minute') -> Dict[str, Any]:
        """
        점진적 세밀화 최적화 실행
        
        Args:
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            refinement_levels: 세밀화 단계 수
            initial_combinations: 초기 조합 수 (None이면 전략 기본값 사용)
            daily_strategy_name: 일봉 전략 이름
            minute_strategy_name: 분봉 전략 이름
            
        Returns:
            최적화 결과
        """
        logger.info(f"점진적 세밀화 최적화 시작 (전략: {self.strategy.get_strategy_name()}, "
                   f"일봉: {daily_strategy_name}, 분봉: {minute_strategy_name}, 단계: {refinement_levels})")
        
        current_best_params = None
        current_best_metrics = None
        
        for level in range(refinement_levels):
            logger.info(f"=== 세밀화 단계 {level + 1}/{refinement_levels} ===")
            
            # 현재 단계의 파라미터 조합 생성
            combinations = self.strategy.generate_parameter_combinations(
                current_best_params, level, refinement_levels, 
                daily_strategy_name, minute_strategy_name
            )
            
            if initial_combinations and level == 0:
                combinations = combinations[:initial_combinations]
            
            logger.info(f"단계 {level + 1}: {len(combinations)}개 조합 테스트")
            
            # 현재 단계의 최적화 실행
            level_results = self._run_level_optimization(
                combinations, start_date, end_date, daily_strategy_name, minute_strategy_name
            )
            
            # 현재 단계의 최고 결과 찾기
            if level_results['successful_results']:
                best_result = max(level_results['successful_results'], 
                                key=lambda x: x['metrics'].get('sharpe_ratio', -999))
                
                current_best_params = best_result['params']
                current_best_metrics = best_result['metrics']
                
                logger.info(f"단계 {level + 1} 최고 결과: "
                           f"샤프지수 {current_best_metrics.get('sharpe_ratio', 0):.2f}, "
                           f"수익률 {current_best_metrics.get('total_return', 0)*100:.2f}%")
                
                # 단계별 결과 저장
                self.optimization_history.append({
                    'level': level + 1,
                    'combinations_tested': len(combinations),
                    'best_params': current_best_params,
                    'best_metrics': current_best_metrics,
                    'all_results': level_results
                })
            else:
                logger.warning(f"단계 {level + 1}: 성공한 결과가 없습니다.")
                break
        
        # 최종 결과 정리
        self.final_best_result = {
            'strategy_name': self.strategy.get_strategy_name(),
            'daily_strategy_name': daily_strategy_name,
            'minute_strategy_name': minute_strategy_name,
            'refinement_levels': refinement_levels,
            'best_params': current_best_params,
            'best_metrics': current_best_metrics,
            'optimization_history': self.optimization_history
        }
        
        logger.info(f"점진적 세밀화 최적화 완료")
        return self.final_best_result
    
    def _run_level_optimization(self, 
                               combinations: List[Dict[str, Any]], 
                               start_date: datetime.date, 
                               end_date: datetime.date,
                               daily_strategy_name: str,
                               minute_strategy_name: str) -> Dict[str, Any]:
        """
        단일 단계의 최적화 실행
        """
        successful_results = []
        failed_results = []
        
        for i, params in enumerate(combinations):
            logger.info(f"진행률: {i+1}/{len(combinations)} ({((i+1)/len(combinations)*100):.1f}%)")
            
            result = self._run_single_backtest(params, start_date, end_date, daily_strategy_name, minute_strategy_name)
            
            if result['success']:
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        return {
            'successful_results': successful_results,
            'failed_results': failed_results,
            'total_combinations': len(combinations)
        }
    
    def _run_single_backtest(self, 
                           params: Dict[str, Any], 
                           start_date: datetime.date, 
                           end_date: datetime.date,
                           daily_strategy_name: str,
                           minute_strategy_name: str) -> Dict[str, Any]:
        """
        단일 파라미터 조합으로 백테스트 실행
        """
        try:
            # 백테스터 초기화 (새로운 생성자 사용)
            backtest = Backtest(
                api_client=self.api_client,
                db_manager=self.backtest_manager.db_manager,
                initial_cash=self.initial_cash,
                save_to_db=True
            )
            
            # 일봉 전략 생성
            daily_strategy = self._create_daily_strategy(backtest, params, daily_strategy_name)
            
            # 분봉 전략 생성
            minute_strategy = self._create_minute_strategy(backtest, params, minute_strategy_name)
            
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
            
            return result
            
        except Exception as e:
            logger.error(f"백테스트 실패: {str(e)}")
            return {
                'params': params,
                'metrics': {},
                'portfolio_values': pd.DataFrame(),
                'success': False,
                'error': str(e)
            }
    
    def _create_daily_strategy(self, backtest: Backtest, params: Dict[str, Any], strategy_name: str):
        """일봉 전략 생성"""
        
        if strategy_name == 'sma_daily':
            from strategies.sma_daily import SMADaily
            return SMADaily(
                broker=backtest.broker,
                data_store=backtest.data_store,
                strategy_params=params['sma_params']
            )
        elif strategy_name == 'dual_momentum_daily':
            from strategies.dual_momentum_daily import DualMomentumDaily
            return DualMomentumDaily(
                broker=backtest.broker,
                data_store=backtest.data_store,
                strategy_params=params['dual_momentum_params']
            )
        elif strategy_name == 'triple_screen_daily':
            from strategies.triple_screen_daily import TripleScreenDaily
            return TripleScreenDaily(
                broker=backtest.broker,
                data_store=backtest.data_store,
                strategy_params=params['triple_screen_params']
            )
        elif strategy_name == 'rsi_daily':
            # RSI 일봉 전략 클래스가 있다면 여기에 추가
            # return RSIDaily(...)
            return None
        elif strategy_name == 'macd_daily':
            # MACD 일봉 전략 클래스가 있다면 여기에 추가
            # return MACDDaily(...)
            return None
        elif strategy_name == 'bollinger_daily':
            # 볼린저밴드 일봉 전략 클래스가 있다면 여기에 추가
            # return BollingerDaily(...)
            return None
        else:
            raise ValueError(f"지원하지 않는 일봉 전략: {strategy_name}")
    
    def _create_minute_strategy(self, backtest: Backtest, params: Dict[str, Any], strategy_name: str):
        """분봉 전략 생성"""
        if strategy_name == 'open_minute':
            from strategies.open_minute import OpenMinute
            # 전략별로 올바른 파라미터 참조
            if 'sma_params' in params:
                num_top_stocks = params['sma_params']['num_top_stocks']
            elif 'dual_momentum_params' in params:
                num_top_stocks = params['dual_momentum_params']['num_top_stocks']
            elif 'triple_screen_params' in params:
                num_top_stocks = params['triple_screen_params']['num_top_stocks']
            else:
                num_top_stocks = 5  # 기본값
            
            minute_params = {
                'num_top_stocks': num_top_stocks
            }
            # rsi_params가 있으면 병합
            if 'rsi_params' in params:
                minute_params.update(params['rsi_params'])
            return OpenMinute(
                broker=backtest.broker,
                data_store=backtest.data_store,
                strategy_params=minute_params
            )
        elif strategy_name == 'rsi_minute':
            from strategies.rsi_minute import RSIMinute
            # 전략별로 올바른 파라미터 참조
            if 'sma_params' in params:
                num_top_stocks = params['sma_params']['num_top_stocks']
            elif 'dual_momentum_params' in params:
                num_top_stocks = params['dual_momentum_params']['num_top_stocks']
            elif 'triple_screen_params' in params:
                num_top_stocks = params['triple_screen_params']['num_top_stocks']
            else:
                num_top_stocks = 5  # 기본값
            
            minute_params = {
                'num_top_stocks': num_top_stocks
            }
            # rsi_params가 있으면 병합
            if 'rsi_params' in params:
                minute_params.update(params['rsi_params'])
            return RSIMinute(
                broker=backtest.broker,
                data_store=backtest.data_store,
                strategy_params=minute_params
            )
        else:
            return None
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """
        최적화 결과를 파일로 저장합니다.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = self.strategy.get_strategy_name().replace(" ", "_").lower()
            filename = f"progressive_{strategy_name}_results_{timestamp}.json"
        
        # optimizer/results 폴더에 저장
        import os
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)
        
        # JSON으로 저장할 수 있는 형태로 변환
        save_data = {
            'strategy_name': results['strategy_name'],
            'daily_strategy_name': results['daily_strategy_name'],
            'minute_strategy_name': results['minute_strategy_name'],
            'refinement_levels': results['refinement_levels'],
            'best_params': results['best_params'],
            'best_metrics': results['best_metrics'],
            'optimization_history': []
        }
        
        # 최적화 히스토리 저장
        for history in results['optimization_history']:
            save_data['optimization_history'].append({
                'level': history['level'],
                'combinations_tested': history['combinations_tested'],
                'best_params': history['best_params'],
                'best_metrics': history['best_metrics']
            })
        
        # JSON 파일 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"결과 저장 완료: {filepath}")
    
    def print_summary(self, results: Dict[str, Any]):
        """
        최적화 결과 요약을 출력합니다.
        """
        print("\n" + "="*60)
        print(f"점진적 세밀화 최적화 결과 요약 ({results['strategy_name']})")
        print("="*60)
        
        print(f"전략: {results['strategy_name']}")
        print(f"일봉 전략: {results['daily_strategy_name']}")
        print(f"분봉 전략: {results['minute_strategy_name']}")
        print(f"세밀화 단계: {results['refinement_levels']}")
        
        if results['best_metrics']:
            metrics = results['best_metrics']
            print(f"\n최종 최적 파라미터:")
            self._print_best_params(results['best_params'])
            print(f"\n최종 성과:")
            print(f"  수익률: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"  샤프지수: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  승률: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"  MDD: {metrics.get('mdd', 0)*100:.2f}%")
        
        print(f"\n" + "-"*60)
        print("세밀화 단계별 진행 상황")
        print("-"*60)
        
        for history in results['optimization_history']:
            metrics = history['best_metrics']
            print(f"단계 {history['level']}: "
                  f"조합 {history['combinations_tested']}개 → "
                  f"샤프지수 {metrics.get('sharpe_ratio', 0):.2f}, "
                  f"수익률 {metrics.get('total_return', 0)*100:.2f}%")
    
    def _print_best_params(self, params: Dict[str, Any]):
        """
        최적 파라미터를 출력합니다.
        """
        if not params:
            print("  최적 파라미터 없음")
            return
        
        # 일봉 전략 파라미터 출력
        if 'sma_params' in params:
            print(f"  SMA: {params['sma_params']['short_sma_period']}일/{params['sma_params']['long_sma_period']}일")
            print(f"  거래량MA: {params['sma_params']['volume_ma_period']}일")
            print(f"  종목수: {params['sma_params']['num_top_stocks']}개")
        elif 'dual_momentum_params' in params:
            weekday_names = ['월', '화', '수', '목', '금']
            weekday_name = weekday_names[params['dual_momentum_params']['rebalance_weekday']]
            print(f"  모멘텀 기간: {params['dual_momentum_params']['momentum_period']}일")
            print(f"  리밸런싱 요일: {weekday_name}요일")
            print(f"  종목수: {params['dual_momentum_params']['num_top_stocks']}개")
        elif 'rsi_daily_params' in params:
            print(f"  RSI 기간: {params['rsi_daily_params']['rsi_period']}일")
            print(f"  과매도/과매수: {params['rsi_daily_params']['oversold_level']}/{params['rsi_daily_params']['overbought_level']}")
            print(f"  거래량MA: {params['rsi_daily_params']['volume_ma_period']}일")
            print(f"  종목수: {params['rsi_daily_params']['num_top_stocks']}개")
        elif 'macd_params' in params:
            print(f"  MACD: {params['macd_params']['fast_period']}/{params['macd_params']['slow_period']}/{params['macd_params']['signal_period']}")
            print(f"  거래량MA: {params['macd_params']['volume_ma_period']}일")
            print(f"  종목수: {params['macd_params']['num_top_stocks']}개")
        elif 'bollinger_params' in params:
            print(f"  볼린저밴드: {params['bollinger_params']['period']}일, 표준편차 {params['bollinger_params']['std_dev']}")
            print(f"  거래량MA: {params['bollinger_params']['volume_ma_period']}일")
            print(f"  종목수: {params['bollinger_params']['num_top_stocks']}개")
        elif 'triple_screen_params' in params:
            print(f"  추세MA: {params['triple_screen_params']['trend_ma_period']}일")
            print(f"  RSI: {params['triple_screen_params']['momentum_rsi_period']}일 "
                  f"({params['triple_screen_params']['momentum_rsi_oversold']}-{params['triple_screen_params']['momentum_rsi_overbought']})")
            print(f"  거래량MA: {params['triple_screen_params']['volume_ma_period']}일")
            print(f"  종목수: {params['triple_screen_params']['num_top_stocks']}개")
            print(f"  최소추세강도: {params['triple_screen_params']['min_trend_strength']}")
        
        # 손절매 파라미터 출력
        if 'stop_loss_params' in params:
            print(f"  손절매: {params['stop_loss_params']['stop_loss_ratio']}%")
            print(f"  트레일링스탑: {params['stop_loss_params']['trailing_stop_ratio']}%")
        
        # 분봉 전략 파라미터 출력
        if 'rsi_params' in params:
            print(f"  RSI 분봉: {params['rsi_params']['minute_rsi_period']}기간 "
                  f"({params['rsi_params']['minute_rsi_oversold']}/{params['rsi_params']['minute_rsi_overbought']})")

if __name__ == "__main__":
    from datetime import datetime

    # 컴포넌트 초기화
    api_client = CreonAPIClient()
    db_manager = DBManager()
    backtest_manager = BacktestManager(api_client, db_manager)
    report = BacktestReport(db_manager=db_manager)
    
    # 백테스트 기간 설정
    start_date = datetime(2025, 3, 1).date()
    end_date = datetime(2025, 4, 1).date()

    # 그리드서치 전략으로 점진적 세밀화 실행
    grid_strategy = GridSearchStrategy()
    optimizer = ProgressiveRefinementOptimizer(
        strategy=grid_strategy,
        api_client=api_client,
        backtest_manager=backtest_manager,
        report=report,
        initial_cash=10_000_000
    )

    # 점진적 세밀화 최적화 실행
    results = optimizer.run_progressive_optimization(
        start_date=start_date,
        end_date=end_date,
        refinement_levels=3,  # 3단계 세밀화
        initial_combinations=50,  # 초기 50개 조합
        daily_strategy_name='sma_daily',  # 일봉 전략: SMA
        minute_strategy_name='open_minute'  # 분봉 전략: Open Minute
    )

    optimizer.save_results(results)
    optimizer.print_summary(results) 