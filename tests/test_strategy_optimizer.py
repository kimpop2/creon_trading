"""
StrategyOptimizer 클래스 단위 테스트
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

from optimizer.strategy_optimizer import StrategyOptimizer

class TestStrategyOptimizer(unittest.TestCase):
    def setUp(self):
        """테스트 설정"""
        self.mock_backtester = Mock()
        self.mock_backtester.data_store = Mock()
        self.mock_backtester.broker = Mock()
        self.initial_cash = 10_000_000
        self.optimizer = StrategyOptimizer(self.mock_backtester, self.initial_cash)
        
        # 테스트용 전략 클래스 정의
        class MockDailyStrategy1:
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
        
        class MockDailyStrategy2:
            def __init__(self, sma_period=20, sma_std=2.0):
                self.sma_period = sma_period
                self.sma_std = sma_std
            
            def set_parameters(self, params):
                """파라미터 설정 메서드"""
                for key, value in params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            
            def run_daily_logic(self, data, position):
                """일봉 전략 로직 실행"""
                return position
        
        class MockMinuteStrategy1:
            def __init__(self, rsi_period=14, rsi_oversold=30):
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
        
        class MockMinuteStrategy2:
            def __init__(self, macd_fast=12, macd_slow=26):
                self.macd_fast = macd_fast
                self.macd_slow = macd_slow
            
            def set_parameters(self, params):
                """파라미터 설정 메서드"""
                for key, value in params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            
            def run_minute_logic(self, data, position):
                """분봉 전략 로직 실행"""
                return position
        
        # 전략 클래스 저장
        self.daily_strategies = {
            'MockDailyStrategy1': MockDailyStrategy1,
            'MockDailyStrategy2': MockDailyStrategy2
        }
        self.minute_strategies = {
            'MockMinuteStrategy1': MockMinuteStrategy1,
            'MockMinuteStrategy2': MockMinuteStrategy2
        }
        
        # 테스트용 파라미터 그리드 설정
        self.param_grids = {
            'MockDailyStrategy1': {
                'momentum_period': [10, 15, 20],
                'num_top_stocks': [3, 5, 7]
            },
            'MockDailyStrategy2': {
                'sma_period': [20, 30, 40],
                'sma_std': [1.5, 2.0, 2.5]
            },
            'MockMinuteStrategy1': {
                'rsi_period': [14, 21, 28],
                'rsi_oversold': [25, 30, 35]
            },
            'MockMinuteStrategy2': {
                'macd_fast': [12, 15, 18],
                'macd_slow': [26, 30, 34]
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
    
    def test_generate_strategy_combinations(self):
        """전략 조합 생성 테스트"""
        # 전략 조합 생성
        combinations = self.optimizer._generate_strategy_combinations(
            self.daily_strategies,
            self.minute_strategies
        )
        
        # 조합 개수 검증 (2 * 2 = 4)
        self.assertEqual(len(combinations), 4)
        
        # 조합 내용 검증
        expected_combinations = [
            ('MockDailyStrategy1', 'MockMinuteStrategy1'),
            ('MockDailyStrategy1', 'MockMinuteStrategy2'),
            ('MockDailyStrategy2', 'MockMinuteStrategy1'),
            ('MockDailyStrategy2', 'MockMinuteStrategy2')
        ]
        for combo in combinations:
            self.assertIn(combo, expected_combinations)
    
    def test_create_strategy_instance(self):
        """전략 인스턴스 생성 테스트"""
        # 전략 인스턴스 생성
        daily_strategy = self.optimizer._create_strategy_instance(
            self.daily_strategies['MockDailyStrategy1'],
            {'momentum_period': 15, 'num_top_stocks': 5}
        )
        minute_strategy = self.optimizer._create_strategy_instance(
            self.minute_strategies['MockMinuteStrategy1'],
            {'rsi_period': 45, 'rsi_oversold': 30}
        )
        
        # 결과 검증
        self.assertIsInstance(daily_strategy, self.daily_strategies['MockDailyStrategy1'])
        self.assertEqual(daily_strategy.momentum_period, 15)
        self.assertEqual(daily_strategy.num_top_stocks, 5)
        
        self.assertIsInstance(minute_strategy, self.minute_strategies['MockMinuteStrategy1'])
        self.assertEqual(minute_strategy.rsi_period, 45)
        self.assertEqual(minute_strategy.rsi_oversold, 30)
    
    def test_optimize(self):
        """최적화 실행 테스트"""
        # 최적화 실행
        results = self.optimizer.optimize(
            daily_strategies=self.daily_strategies,
            minute_strategies=self.minute_strategies,
            param_grids=self.param_grids,
            start_date=date(2025, 3, 1),
            end_date=date(2025, 3, 31)
        )
        
        # 결과 검증
        self.assertIsNotNone(results)
        self.assertIn('best_strategy_combination', results)
        self.assertIn('best_params', results)
        self.assertIn('best_metrics', results)
        self.assertIn('total_tests', results)
        self.assertIn('successful_tests', results)
        
        # 전략 조합 검증
        best_combo = results['best_strategy_combination']
        self.assertIn(best_combo['daily'], self.daily_strategies)
        self.assertIn(best_combo['minute'], self.minute_strategies)
    
    def test_optimize_with_invalid_strategy(self):
        """잘못된 전략으로 최적화 실행 테스트"""
        # 잘못된 전략 클래스 설정
        invalid_strategies = {
            'daily': {'InvalidStrategy': None},
            'minute': {'MockMinuteStrategy1': self.minute_strategies['MockMinuteStrategy1']}
        }
        
        # 최적화 실행
        results = self.optimizer.optimize(
            daily_strategies=invalid_strategies['daily'],
            minute_strategies=invalid_strategies['minute'],
            param_grids=self.param_grids,
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
            daily_strategies=self.daily_strategies,
            minute_strategies=self.minute_strategies,
            param_grids=self.param_grids,
            start_date=date(2025, 3, 1),
            end_date=date(2025, 3, 31)
        )
        
        # 결과 검증
        self.assertIsNotNone(results)
        self.assertEqual(results['successful_tests'], 0)
        self.assertEqual(results['total_tests'], 4)  # 2 * 2 전략 조합

if __name__ == '__main__':
    unittest.main() 