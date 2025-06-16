import unittest
from datetime import date, timedelta
from typing import Dict, List, Type, Any
import pandas as pd
import numpy as np
from unittest.mock import Mock

from optimizer.grid_search import GridSearchOptimizer
from optimizer.strategy_optimizer import StrategyOptimizer
from optimizer.market_optimizer import MarketOptimizer
from optimizer.parameter_analyzer import ParameterAnalyzer
from optimizer.base_optimizer import BaseOptimizer

class TestOptimizerIntegration(unittest.TestCase):
    """옵티마이저 통합 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        # 테스트 기간 설정
        self.end_date = date.today()
        self.start_date = self.end_date - timedelta(days=365)
        
        # 테스트용 전략 클래스 정의
        class MockDailyStrategy:
            def __init__(self, params: Dict[str, Any] = None):
                self.params = params or {}
                self.name = "MockDailyStrategy"
                self.position = 0
                self.trades = []
                self.window = self.params.get('window', 5)
                self.threshold = self.params.get('threshold', 0.01)
                self.momentum_period = self.params.get('momentum_period', 10)
                self.volume_ma = self.params.get('volume_ma', 5)
            
            def set_parameters(self, params: Dict[str, Any]) -> None:
                """파라미터 설정"""
                self.params = params
                self.window = params.get('window', self.window)
                self.threshold = params.get('threshold', self.threshold)
                self.momentum_period = params.get('momentum_period', self.momentum_period)
                self.volume_ma = params.get('volume_ma', self.volume_ma)
            
            def run_daily_logic(self, data: pd.DataFrame) -> pd.DataFrame:
                """일봉 데이터 처리"""
                if data is None or data.empty:
                    return pd.DataFrame()
                
                # 단순 이동평균 계산
                data['ma'] = data['close'].rolling(window=self.window).mean()
                data['signal'] = np.where(data['close'] > data['ma'] * (1 + self.threshold), 1,
                                        np.where(data['close'] < data['ma'] * (1 - self.threshold), -1, 0))
                return data
            
            def run_minute_logic(self, data: pd.DataFrame) -> pd.DataFrame:
                """분봉 데이터 처리 (일봉 전략이므로 빈 DataFrame 반환)"""
                return pd.DataFrame()
        
        class MockMinuteStrategy:
            def __init__(self, params: Dict[str, Any] = None):
                self.params = params or {}
                self.name = "MockMinuteStrategy"
                self.position = 0
                self.trades = []
                self.window = self.params.get('window', 10)
                self.threshold = self.params.get('threshold', 0.005)
                self.rsi_period = self.params.get('rsi_period', 14)
                self.volume_ma = self.params.get('volume_ma', 5)
            
            def set_parameters(self, params: Dict[str, Any]) -> None:
                """파라미터 설정"""
                self.params = params
                self.window = params.get('window', self.window)
                self.threshold = params.get('threshold', self.threshold)
                self.rsi_period = params.get('rsi_period', self.rsi_period)
                self.volume_ma = params.get('volume_ma', self.volume_ma)
            
            def run_daily_logic(self, data: pd.DataFrame) -> pd.DataFrame:
                """일봉 데이터 처리 (분봉 전략이므로 빈 DataFrame 반환)"""
                return pd.DataFrame()
            
            def run_minute_logic(self, data: pd.DataFrame) -> pd.DataFrame:
                """분봉 데이터 처리"""
                if data is None or data.empty:
                    return pd.DataFrame()
                
                # 단순 이동평균 계산
                data['ma'] = data['close'].rolling(window=self.window).mean()
                data['signal'] = np.where(data['close'] > data['ma'] * (1 + self.threshold), 1,
                                        np.where(data['close'] < data['ma'] * (1 - self.threshold), -1, 0))
                return data
        
        self.mock_daily_strategy = MockDailyStrategy
        self.mock_minute_strategy = MockMinuteStrategy
        
        # 테스트용 파라미터 그리드
        self.param_grids = {
            'MockDailyStrategy': {
                'window': [5, 10, 20],
                'threshold': [0.01, 0.02, 0.03]
            },
            'MockMinuteStrategy': {
                'window': [10, 20, 30],
                'threshold': [0.005, 0.01, 0.015]
            }
        }
        
        # 옵티마이저 인스턴스 생성
        self.mock_backtester = Mock()
        # 백테스터의 run 메서드 설정
        self.mock_backtester.run.return_value = (
            10000000,  # portfolio_value
            {
                'sharpe_ratio': 1.5,
                'total_return': 0.15,
                'max_drawdown': -0.1,
                'win_rate': 0.6,
                'profit_factor': 1.8
            }
        )
        self.grid_search = GridSearchOptimizer(self.mock_backtester)
        self.strategy_optimizer = StrategyOptimizer(self.mock_backtester)
        self.market_optimizer = MarketOptimizer(self.mock_backtester)
        self.parameter_analyzer = ParameterAnalyzer(self.mock_backtester)
    
    def test_basic_workflow(self):
        """기본 워크플로우 테스트: GridSearch → StrategyOptimizer → MarketOptimizer → ParameterAnalyzer"""
        # 1. GridSearch로 최적 파라미터 찾기
        grid_results = self.grid_search.optimize(
            daily_strategy=self.mock_daily_strategy,
            minute_strategy=self.mock_minute_strategy,
            param_grid={
                'daily_params': self.param_grids['MockDailyStrategy'],
                'minute_params': self.param_grids['MockMinuteStrategy']
            },
            start_date=self.start_date,
            end_date=self.end_date
        )
        self.assertIsNotNone(grid_results)
        self.assertIn('best_params', grid_results)
        
        # 2. StrategyOptimizer로 전략 조합 최적화
        strategy_results = self.strategy_optimizer.optimize(
            daily_strategies={'MockDailyStrategy': self.mock_daily_strategy},
            minute_strategies={'MockMinuteStrategy': self.mock_minute_strategy},
            param_grids=self.param_grids,
            start_date=self.start_date,
            end_date=self.end_date
        )
        self.assertIsNotNone(strategy_results)
        self.assertIn('best_combination', strategy_results)
        
        # 3. MarketOptimizer로 시장 상태 기반 최적화
        market_results = self.market_optimizer.optimize(
            daily_strategies={'MockDailyStrategy': self.mock_daily_strategy},
            minute_strategies={'MockMinuteStrategy': self.mock_minute_strategy},
            param_grids=self.param_grids,
            start_date=self.start_date,
            end_date=self.end_date
        )
        self.assertIsNotNone(market_results)
        self.assertIn('market_state', market_results)
        
        # 4. ParameterAnalyzer로 파라미터 영향도 분석
        analysis_results = self.parameter_analyzer.analyze_parameter_impact(
            results=[grid_results, strategy_results, market_results],
            target_metric='sharpe_ratio'
        )
        self.assertIsNotNone(analysis_results)
        self.assertIsInstance(analysis_results, dict)
    
    def test_market_state_optimization(self):
        """시장 상태별 최적화 테스트: MarketOptimizer → StrategyOptimizer → ParameterAnalyzer"""
        # 1. MarketOptimizer로 시장 상태 분석 및 최적화
        market_results = self.market_optimizer.optimize(
            daily_strategies={'MockDailyStrategy': self.mock_daily_strategy},
            minute_strategies={'MockMinuteStrategy': self.mock_minute_strategy},
            param_grids=self.param_grids,
            start_date=self.start_date,
            end_date=self.end_date,
            market_state='bull'  # 특정 시장 상태 지정
        )
        self.assertIsNotNone(market_results)
        self.assertEqual(market_results['market_state'], 'bull')
        
        # 2. 최적화된 파라미터로 StrategyOptimizer 실행
        adjusted_params = market_results.get('best_params', self.param_grids)
        strategy_results = self.strategy_optimizer.optimize(
            daily_strategies={'MockDailyStrategy': self.mock_daily_strategy},
            minute_strategies={'MockMinuteStrategy': self.mock_minute_strategy},
            param_grids=adjusted_params,
            start_date=self.start_date,
            end_date=self.end_date
        )
        self.assertIsNotNone(strategy_results)
        
        # 3. ParameterAnalyzer로 결과 분석
        analysis_results = self.parameter_analyzer.analyze_parameter_impact(
            results=[market_results, strategy_results],
            target_metric='sharpe_ratio'
        )
        self.assertIsNotNone(analysis_results)
    
    def test_parameter_analysis_based_optimization(self):
        """파라미터 분석 기반 최적화 테스트: ParameterAnalyzer → GridSearch → StrategyOptimizer"""
        # 1. ParameterAnalyzer로 초기 파라미터 영향도 분석
        initial_results = self.grid_search.optimize(
            daily_strategy=self.mock_daily_strategy,
            minute_strategy=self.mock_minute_strategy,
            param_grid={
                'daily_params': self.param_grids['MockDailyStrategy'],
                'minute_params': self.param_grids['MockMinuteStrategy']
            },
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        analysis_results = self.parameter_analyzer.analyze_parameter_impact(
            results=[initial_results],
            target_metric='sharpe_ratio'
        )
        self.assertIsNotNone(analysis_results)
        
        # 2. 분석 결과를 바탕으로 파라미터 그리드 조정
        adjusted_grids = self._adjust_param_grids_based_on_analysis(
            self.param_grids,
            analysis_results
        )
        
        # 3. 조정된 파라미터로 GridSearch 실행
        grid_results = self.grid_search.optimize(
            daily_strategy=self.mock_daily_strategy,
            minute_strategy=self.mock_minute_strategy,
            param_grid={
                'daily_params': adjusted_grids['MockDailyStrategy'],
                'minute_params': adjusted_grids['MockMinuteStrategy']
            },
            start_date=self.start_date,
            end_date=self.end_date
        )
        self.assertIsNotNone(grid_results)
        
        # 4. 최종 StrategyOptimizer 실행
        strategy_results = self.strategy_optimizer.optimize(
            daily_strategies={'MockDailyStrategy': self.mock_daily_strategy},
            minute_strategies={'MockMinuteStrategy': self.mock_minute_strategy},
            param_grids=grid_results['best_params'],
            start_date=self.start_date,
            end_date=self.end_date
        )
        self.assertIsNotNone(strategy_results)
    
    def _adjust_param_grids_based_on_analysis(
        self,
        original_grids: Dict[str, Dict[str, List[Any]]],
        analysis_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, List[Any]]]:
        """파라미터 분석 결과를 바탕으로 파라미터 그리드 조정"""
        adjusted_grids = original_grids.copy()
        
        for strategy_name, param_impacts in analysis_results.items():
            if strategy_name in adjusted_grids:
                for param_name, impact in param_impacts.items():
                    if param_name in adjusted_grids[strategy_name]:
                        # 영향도가 높은 파라미터는 더 세밀한 범위로 조정
                        if abs(impact) > 0.5:
                            current_values = adjusted_grids[strategy_name][param_name]
                            if isinstance(current_values[0], (int, float)):
                                min_val = min(current_values)
                                max_val = max(current_values)
                                step = (max_val - min_val) / 5
                                adjusted_grids[strategy_name][param_name] = [
                                    min_val + i * step for i in range(6)
                                ]
        
        return adjusted_grids

if __name__ == '__main__':
    unittest.main() 