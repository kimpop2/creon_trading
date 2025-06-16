"""
MarketOptimizer 클래스 단위 테스트
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

import sys
import os
# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from optimizer.market_optimizer import MarketOptimizer

class TestMarketOptimizer(unittest.TestCase):
    def setUp(self):
        """테스트 설정"""
        self.mock_backtester = Mock()
        self.mock_backtester.data_store = Mock()
        self.mock_backtester.broker = Mock()
        self.initial_cash = 10_000_000
        self.optimizer = MarketOptimizer(self.mock_backtester, self.initial_cash)
        
        # 테스트용 전략 클래스 정의
        class MockDailyStrategy:
            def __init__(self, momentum_period=15, num_top_stocks=5):
                self.momentum_period = momentum_period
                self.num_top_stocks = num_top_stocks
                self.name = "MockDailyStrategy"
            
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
                self.name = "MockMinuteStrategy"
            
            def set_parameters(self, params):
                """파라미터 설정 메서드"""
                for key, value in params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            
            def run_minute_logic(self, data, position):
                """분봉 전략 로직 실행"""
                return position
        
        # 전략 클래스 등록
        self.daily_strategies = {'MockDailyStrategy': MockDailyStrategy}
        self.minute_strategies = {'MockMinuteStrategy': MockMinuteStrategy}
        
        # 테스트용 파라미터 그리드 설정
        self.param_grids = {
            'MockDailyStrategy': {
                'momentum_period': [10, 15, 20],
                'num_top_stocks': [3, 5, 7]
            },
            'MockMinuteStrategy': {
                'rsi_period': [30, 45, 60],
                'rsi_oversold': [25, 30, 35]
            }
        }
        
        # 테스트용 시장 데이터 생성
        dates = pd.date_range(start='2025-03-01', end='2025-03-31', freq='D')
        self.market_data = pd.DataFrame({
            'close': np.random.normal(1000, 50, len(dates)),
            'volume': np.random.normal(1000000, 200000, len(dates)),
            'high': np.random.normal(1020, 50, len(dates)),
            'low': np.random.normal(980, 50, len(dates))
        }, index=dates)
        
        # 백테스터의 get_daily_data 메서드 모킹
        self.mock_backtester.data_store.get_daily_data.return_value = self.market_data
        
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
            {'sharpe_ratio': 1.5, 'max_drawdown': -0.1}
        )
    
    def test_analyze_market_state(self):
        """시장 상태 분석 테스트"""
        # 시장 상태 분석 실행
        market_state = self.optimizer._analyze_market_state(
            date(2025, 3, 1),
            date(2025, 3, 31)
        )
        
        # 결과 검증
        self.assertIsNotNone(market_state)
        self.assertIn(market_state, ['bull', 'bear', 'sideways', 'volatile'])
    
    def test_get_market_data(self):
        """시장 데이터 조회 테스트"""
        # 시장 데이터 조회
        data = self.optimizer._get_market_data(
            'U001',
            date(2025, 3, 1),
            date(2025, 3, 31)
        )
        
        # 결과 검증
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        
        # 백테스터 호출 검증
        self.mock_backtester.data_store.get_daily_data.assert_called_once_with(
            'U001',
            date(2025, 3, 1),
            date(2025, 3, 31)
        )
    
    def test_calculate_trend(self):
        """추세 계산 테스트"""
        # 추세 계산
        trend = self.optimizer._calculate_trend(self.market_data)
        
        # 결과 검증
        self.assertIsInstance(trend, str)
        self.assertIn(trend, ['up', 'down', 'neutral', 'mixed'])
    
    def test_calculate_volatility(self):
        """변동성 계산 테스트"""
        # 변동성 계산
        volatility = self.optimizer._calculate_volatility(self.market_data)
        
        # 결과 검증
        self.assertIsInstance(volatility, str)
        self.assertIn(volatility, ['low', 'medium', 'high', 'very_high'])
    
    def test_calculate_momentum(self):
        """모멘텀 계산 테스트"""
        # 모멘텀 계산
        momentum = self.optimizer._calculate_momentum(self.market_data)
        
        # 결과 검증
        self.assertIsInstance(momentum, str)
        self.assertIn(momentum, ['low', 'medium', 'high', 'mixed'])
    
    def test_determine_market_state(self):
        """시장 상태 판단 테스트"""
        # 각 시장 상태별 테스트
        test_cases = [
            ('up', 'low', 'high', 'bull'),      # 강한 상승장
            ('down', 'high', 'low', 'bear'),    # 강한 하락장
            ('neutral', 'medium', 'medium', 'sideways'),  # 횡보장
            ('mixed', 'very_high', 'mixed', 'volatile')   # 변동성 장
        ]
        
        for trend, volatility, momentum, expected_state in test_cases:
            state = self.optimizer._determine_market_state(trend, volatility, momentum)
            self.assertEqual(state, expected_state)
    
    def test_adjust_param_grid(self):
        """파라미터 그리드 조정 테스트"""
        # 각 시장 상태별 파라미터 그리드 조정 테스트
        market_states = ['bull', 'bear', 'sideways', 'volatile']
        
        for state in market_states:
            adjusted_grid = self.optimizer._adjust_param_grids_for_market_state(
                self.param_grids,
                state
            )
            
            # 결과 검증
            self.assertIsNotNone(adjusted_grid)
            self.assertIn('MockDailyStrategy', adjusted_grid)
            self.assertIn('MockMinuteStrategy', adjusted_grid)
    
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
        self.assertIn('market_state', results)
        self.assertIn('best_strategy_combination', results)
        self.assertIn('best_params', results)
        self.assertIn('best_metrics', results)
        
        # 시장 상태 검증
        self.assertIn(results['market_state'], ['bull', 'bear', 'sideways', 'volatile'])
    
    def test_optimize_with_specific_market_state(self):
        """특정 시장 상태로 최적화 실행 테스트"""
        # 특정 시장 상태로 최적화 실행
        market_state = 'bull'
        results = self.optimizer.optimize(
            daily_strategies=self.daily_strategies,
            minute_strategies=self.minute_strategies,
            param_grids=self.param_grids,
            start_date=date(2025, 3, 1),
            end_date=date(2025, 3, 31),
            market_state=market_state
        )
        
        # 결과 검증
        self.assertIsNotNone(results)
        self.assertEqual(results['market_state'], market_state)
    
    def test_optimize_with_market_data_error(self):
        """시장 데이터 오류 발생 시 테스트"""
        # 시장 데이터 조회 오류 발생 설정
        self.mock_backtester.data_store.get_daily_data.side_effect = Exception("데이터 조회 오류")
        
        # 최적화 실행
        results = self.optimizer.optimize(
            daily_strategies=self.daily_strategies,
            minute_strategies=self.minute_strategies,
            param_grids=self.param_grids,
            start_date=date(2025, 3, 1),
            end_date=date(2025, 3, 31)
        )
        
        # 결과 검증
        self.assertIsNone(results)

if __name__ == '__main__':
    unittest.main() 