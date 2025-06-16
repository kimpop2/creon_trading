"""
GridSearchOptimizer 클래스 단위 테스트
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import date, datetime

import sys
import os
# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from optimizer.grid_search import GridSearchOptimizer
from optimizer.base_optimizer import BaseOptimizer

class TestGridSearchOptimizer(unittest.TestCase):
    def setUp(self):
        """테스트 설정"""
        self.mock_backtester = Mock()
        self.mock_backtester.data_store = Mock()
        self.mock_backtester.broker = Mock()
        self.initial_cash = 10_000_000
        self.optimizer = GridSearchOptimizer(self.mock_backtester, self.initial_cash)
        
        # 테스트용 전략 클래스 정의
        class MockDailyStrategy:
            def __init__(self, momentum_period=15, num_top_stocks=5):
                self.momentum_period = momentum_period
                self.num_top_stocks = num_top_stocks
            
            def set_parameters(self, params):
                """파라미터 설정 메서드"""
                for key, value in params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            
            def run_daily_logic(self, data, position):
                """일봉 전략 로직 실행"""
                return position
        
        class MockMinuteStrategy:
            def __init__(self, rsi_period=45, rsi_oversold=30):
                self.rsi_period = rsi_period
                self.rsi_oversold = rsi_oversold
            
            def set_parameters(self, params):
                """파라미터 설정 메서드"""
                for key, value in params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            
            def run_minute_logic(self, data, position):
                """분봉 전략 로직 실행"""
                return position
        
        self.daily_strategy_class = MockDailyStrategy
        self.minute_strategy_class = MockMinuteStrategy
        
        # 테스트용 파라미터 그리드 설정
        self.param_grid = {
            'daily_params': {
                'momentum_period': [10, 15, 20],
                'num_top_stocks': [3, 5, 7]
            },
            'minute_params': {
                'rsi_period': [30, 45, 60],
                'rsi_oversold': [25, 30, 35]
            }
        }
        
        # 테스트용 백테스트 결과 설정
        self.mock_backtest_result = {
            'portfolio_value': pd.Series([10_000_000, 10_500_000, 11_000_000]),
            'returns': pd.Series([0.0, 0.05, 0.0476]),
            'trades': pd.DataFrame({
                'entry_date': [datetime(2025, 3, 1), datetime(2025, 3, 15)],
                'exit_date': [datetime(2025, 3, 10), datetime(2025, 3, 20)],
                'profit_loss': [50000, 45000],
                'entry_price': [1000, 1100],
                'exit_price': [1050, 1150]
            })
        }
        
        # 백테스터의 run 메서드 모킹
        self.mock_backtester.run.return_value = (
            self.mock_backtest_result['portfolio_value'],
            {
                'sharpe_ratio': 1.5,
                'total_return': 0.1,
                'max_drawdown': -0.05,
                'win_rate': 0.6
            }
        )
    
    def test_generate_param_combinations(self):
        """파라미터 조합 생성 테스트"""
        # 파라미터 조합 생성
        combinations = self.optimizer._generate_param_combinations(self.param_grid)
        
        # 조합 개수 검증 (3 * 3 * 3 * 3 = 81)
        self.assertEqual(len(combinations), 81)
        
        # 첫 번째 조합 검증
        first_combination = combinations[0]
        self.assertIn('daily_params', first_combination)
        self.assertIn('minute_params', first_combination)
        self.assertEqual(first_combination['daily_params']['momentum_period'], 10)
        self.assertEqual(first_combination['daily_params']['num_top_stocks'], 3)
        self.assertEqual(first_combination['minute_params']['rsi_period'], 30)
        self.assertEqual(first_combination['minute_params']['rsi_oversold'], 25)
    
    def test_optimize(self):
        """최적화 실행 테스트"""
        # 최적화 실행
        results = self.optimizer.optimize(
            daily_strategy=self.daily_strategy_class,
            minute_strategy=self.minute_strategy_class,
            param_grid=self.param_grid,
            start_date=date(2025, 3, 1),
            end_date=date(2025, 3, 31)
        )
        
        # 결과 검증
        self.assertIsNotNone(results)
        self.assertIn('best_params', results)
        self.assertIn('best_metrics', results)
        self.assertIn('total_tests', results)
        self.assertIn('successful_tests', results)
        
        # 백테스터 호출 횟수 검증
        self.assertEqual(self.mock_backtester.run.call_count, 81)
    
    def test_optimize_with_invalid_params(self):
        """잘못된 파라미터로 최적화 실행 테스트"""
        # 잘못된 파라미터 그리드 설정
        invalid_param_grid = {
            'daily_params': {
                'invalid_param': [1, 2, 3]  # 존재하지 않는 파라미터
            },
            'minute_params': {
                'rsi_period': [30, 45, 60]
            }
        }
        
        # 최적화 실행
        results = self.optimizer.optimize(
            daily_strategy=self.daily_strategy_class,
            minute_strategy=self.minute_strategy_class,
            param_grid=invalid_param_grid,
            start_date=date(2025, 3, 1),
            end_date=date(2025, 3, 31)
        )
        
        # 결과 검증
        self.assertIsNotNone(results)
        self.assertEqual(results['successful_tests'], 0)
    
    def test_optimize_with_backtest_error(self):
        """백테스트 오류 발생 시 테스트"""
        # 백테스터 오류 발생 설정
        self.mock_backtester.run.side_effect = Exception("테스트 오류")
        
        # 최적화 실행
        results = self.optimizer.optimize(
            daily_strategy=self.daily_strategy_class,
            minute_strategy=self.minute_strategy_class,
            param_grid=self.param_grid,
            start_date=date(2025, 3, 1),
            end_date=date(2025, 3, 31)
        )
        
        # 결과 검증
        self.assertIsNotNone(results)
        self.assertEqual(results['successful_tests'], 0)
        self.assertEqual(results['total_tests'], 81)
    
    def test_optimize_with_empty_param_grid(self):
        """빈 파라미터 그리드로 최적화 실행 테스트"""
        # 빈 파라미터 그리드 설정
        empty_param_grid = {
            'daily_params': {},
            'minute_params': {}
        }
        
        # 최적화 실행
        results = self.optimizer.optimize(
            daily_strategy=self.daily_strategy_class,
            minute_strategy=self.minute_strategy_class,
            param_grid=empty_param_grid,
            start_date=date(2025, 3, 1),
            end_date=date(2025, 3, 31)
        )
        
        # 결과 검증
        self.assertIsNotNone(results)
        self.assertEqual(results['total_tests'], 1)  # 기본 파라미터만 테스트

if __name__ == '__main__':
    unittest.main() 