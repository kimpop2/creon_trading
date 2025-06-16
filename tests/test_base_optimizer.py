"""
BaseOptimizer 클래스 단위 테스트
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

import sys
import os
# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from datetime import date, datetime, timedelta
from optimizer.base_optimizer import BaseOptimizer
from optimizer.grid_search import GridSearchOptimizer
from strategies.strategy import DailyStrategy, MinuteStrategy

class TestDailyStrategy(DailyStrategy):
    """테스트용 일봉 전략"""
    def run_daily_logic(self, current_date):
        """일봉 로직 실행"""
        # 테스트용 신호 생성
        self.signals = {
            'A005930': {  # 삼성전자
                'signal': 'buy',
                'target_quantity': 10,
                'traded_today': False
            }
        }
    
    def set_parameters(self, params):
        """파라미터 설정 메서드"""
        self.strategy_params.update(params)
    
    def _initialize_signals_for_all_stocks(self):
        """신호 초기화 메서드"""
        self.signals = {}

class TestMinuteStrategy(MinuteStrategy):
    """테스트용 분봉 전략"""
    def run_minute_logic(self, stock_code, current_minute_dt):
        """분봉 로직 실행"""
        pass
    
    def set_parameters(self, params):
        """파라미터 설정 메서드"""
        self.strategy_params.update(params)
    
    def update_signals(self, signals):
        """신호 업데이트 메서드"""
        self.signals = signals

class TestBaseOptimizer(unittest.TestCase):
    def setUp(self):
        """테스트 설정"""
        self.mock_backtester = Mock()
        
        # 테스트용 일봉 데이터 생성
        dates = pd.date_range(start='2025-03-01', end='2025-03-31', freq='B')
        self.test_daily_data = pd.DataFrame({
            'open': np.random.uniform(1000, 2000, len(dates)),
            'high': np.random.uniform(1000, 2000, len(dates)),
            'low': np.random.uniform(1000, 2000, len(dates)),
            'close': np.random.uniform(1000, 2000, len(dates)),
            'volume': np.random.uniform(1000000, 2000000, len(dates))
        }, index=dates)
        
        self.mock_backtester.data_store = {
            'daily': {'A005930': self.test_daily_data},  # 삼성전자
            'minute': {}
        }
        self.mock_backtester.broker = Mock()
        self.mock_backtester.broker.get_position_size.return_value = 0
        self.mock_backtester.broker.get_portfolio_value.return_value = 10_000_000
        
        self.initial_cash = 10_000_000
        self.optimizer = GridSearchOptimizer(self.mock_backtester, self.initial_cash)
        
        # 테스트용 전략 인스턴스 생성
        self.mock_daily_strategy = TestDailyStrategy(
            data_store=self.mock_backtester.data_store,
            strategy_params={'momentum_period': 15, 'num_top_stocks': 5},
            broker=self.mock_backtester.broker
        )
        self.mock_minute_strategy = TestMinuteStrategy(
            data_store=self.mock_backtester.data_store,
            strategy_params={'rsi_period': 45, 'rsi_oversold': 30},
            broker=self.mock_backtester.broker
        )
        
        # 테스트용 파라미터 설정
        self.test_params = {
            'daily_params': {
                'momentum_period': 15,
                'num_top_stocks': 7
            },
            'minute_params': {
                'rsi_period': 45,
                'rsi_oversold': 30
            }
        }
        
        # 테스트용 백테스트 결과 설정
        test_dates = pd.date_range(start='2025-03-01', periods=3, freq='B')
        self.test_portfolio_values = pd.Series(
            [10_000_000, 10_500_000, 11_000_000],
            index=test_dates
        )
        self.test_metrics = {
            'sharpe_ratio': 1.5,
            'total_return': 0.1,
            'max_drawdown': -0.05,
            'win_rate': 0.6
        }
        
        # 백테스터의 run 메서드 모킹
        self.mock_backtester.run.return_value = (
            self.test_portfolio_values,
            self.test_metrics
        )
        
        # 기본 target_metric 설정
        self.optimizer.target_metric = 'sharpe_ratio'
    
    def test_evaluate_params(self):
        """파라미터 평가 테스트"""
        # 파라미터 평가 실행
        result = self.optimizer.evaluate_params(
            self.mock_daily_strategy,
            self.mock_minute_strategy,
            self.test_params,
            date(2025, 3, 1),
            date(2025, 3, 31),
            target_metric='sharpe_ratio'
        )
        
        # 결과 검증
        self.assertIsNotNone(result)
        self.assertIn('params', result)
        self.assertIn('metrics', result)
        self.assertIn('portfolio_values', result)
        self.assertIn('timestamp', result)
        
        # 파라미터 검증
        self.assertEqual(result['params'], self.test_params)
        
        # 메트릭 검증
        metrics = result['metrics']
        self.assertEqual(metrics, self.test_metrics)
        
        # 포트폴리오 가치 검증
        pd.testing.assert_series_equal(result['portfolio_values'], self.test_portfolio_values)
        
        # 백테스터 호출 검증
        self.mock_backtester.set_strategies.assert_called_once_with(
            self.mock_daily_strategy,
            self.mock_minute_strategy
        )
        self.mock_backtester.run.assert_called_once_with(
            date(2025, 3, 1),
            date(2025, 3, 31)
        )
        
        # 전략 파라미터 설정 검증
        self.assertEqual(self.mock_daily_strategy.strategy_params['momentum_period'], 15)
        self.assertEqual(self.mock_daily_strategy.strategy_params['num_top_stocks'], 7)
        self.assertEqual(self.mock_minute_strategy.strategy_params['rsi_period'], 45)
        self.assertEqual(self.mock_minute_strategy.strategy_params['rsi_oversold'], 30)
    
    def test_save_results(self):
        """결과 저장 테스트"""
        # 테스트용 결과 생성
        test_results = [
            {
                'params': self.test_params,
                'metrics': self.test_metrics,
                'portfolio_values': self.test_portfolio_values,
                'timestamp': datetime.now()
            }
        ]
        
        # 결과 저장 실행
        self.optimizer.results = test_results
        self.optimizer.save_results()
        
        # 파일 존재 여부 검증
        import glob
        import os
        files = glob.glob('optimizer_results_*.csv')
        self.assertTrue(len(files) > 0)
        
        # CSV 파일 내용 검증
        df = pd.read_csv(files[0])
        self.assertEqual(len(df), 1)
        self.assertIn('sharpe_ratio', df.columns)
        self.assertIn('total_return', df.columns)
        self.assertIn('max_drawdown', df.columns)
        self.assertIn('win_rate', df.columns)
        
        # 메트릭 값 검증
        self.assertEqual(df['sharpe_ratio'].iloc[0], self.test_metrics['sharpe_ratio'])
        self.assertEqual(df['total_return'].iloc[0], self.test_metrics['total_return'])
        self.assertEqual(df['max_drawdown'].iloc[0], self.test_metrics['max_drawdown'])
        self.assertEqual(df['win_rate'].iloc[0], self.test_metrics['win_rate'])
        
        # 테스트 파일 삭제
        os.remove(files[0])
    
    def test_get_best_params(self):
        """최적 파라미터 조회 테스트"""
        # 테스트용 결과 설정
        self.optimizer.results = [
            {
                'params': {'daily_params': {'momentum_period': 10}},
                'metrics': {'sharpe_ratio': 1.0, 'total_return': 0.05},
                'portfolio_values': self.test_portfolio_values,
                'timestamp': datetime.now()
            },
            {
                'params': {'daily_params': {'momentum_period': 15}},
                'metrics': {'sharpe_ratio': 1.5, 'total_return': 0.1},
                'portfolio_values': self.test_portfolio_values,
                'timestamp': datetime.now()
            }
        ]
        
        # target_metric 설정
        self.optimizer.target_metric = 'sharpe_ratio'
        
        # 최적 파라미터 조회
        best_params = self.optimizer.get_best_params()
        
        # 결과 검증
        self.assertEqual(best_params['daily_params']['momentum_period'], 15)
    
    def test_get_best_metrics(self):
        """최적 메트릭 조회 테스트"""
        # 테스트용 결과 설정
        self.optimizer.results = [
            {
                'params': {'daily_params': {'momentum_period': 10}},
                'metrics': {'sharpe_ratio': 1.0, 'total_return': 0.05},
                'portfolio_values': self.test_portfolio_values,
                'timestamp': datetime.now()
            },
            {
                'params': {'daily_params': {'momentum_period': 15}},
                'metrics': {'sharpe_ratio': 1.5, 'total_return': 0.1},
                'portfolio_values': self.test_portfolio_values,
                'timestamp': datetime.now()
            }
        ]
        
        # target_metric 설정
        self.optimizer.target_metric = 'sharpe_ratio'
        
        # 최적 메트릭 조회
        best_metrics = self.optimizer.get_best_metrics()
        
        # 결과 검증
        self.assertEqual(best_metrics['sharpe_ratio'], 1.5)
        self.assertEqual(best_metrics['total_return'], 0.1)
    
    def test_evaluate_params_error_handling(self):
        """파라미터 평가 오류 처리 테스트"""
        # 백테스터 오류 발생 설정
        self.mock_backtester.run.side_effect = Exception("테스트 오류")
        
        # 파라미터 평가 실행
        result = self.optimizer.evaluate_params(
            self.mock_daily_strategy,
            self.mock_minute_strategy,
            self.test_params,
            date(2025, 3, 1),
            date(2025, 3, 31),
            target_metric='sharpe_ratio'
        )
        
        # 결과 검증
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main() 