"""
ParameterAnalyzer 클래스 단위 테스트
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import date, datetime
import shutil

import sys
import os
# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from optimizer.parameter_analyzer import ParameterAnalyzer

class TestParameterAnalyzer(unittest.TestCase):
    def setUp(self):
        """테스트 설정"""
        self.mock_backtester = Mock()
        self.mock_backtester.data_store = Mock()
        self.mock_backtester.broker = Mock()
        self.initial_cash = 10_000_000
        self.analyzer = ParameterAnalyzer(self.mock_backtester, self.initial_cash)
        
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
        
        self.daily_strategy_class = MockDailyStrategy
        self.minute_strategy_class = MockMinuteStrategy
        
        # 테스트용 파라미터 그리드 설정
        self.param_grid = {
            'daily_params': {
                'momentum_period': [10, 15, 20],
                'num_top_stocks': [3, 5, 7]
            },
            'minute_params': {
                'rsi_period': [14, 21, 28],
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
        
        # 테스트용 최적화 결과 생성
        self.test_results = []
        for momentum_period in [10, 15, 20]:
            for num_top_stocks in [3, 5, 7]:
                for rsi_period in [30, 45, 60]:
                    for rsi_oversold in [25, 30, 35]:
                        # 샤프 비율은 파라미터 값들의 조합에 따라 달라지도록 설정
                        sharpe_ratio = (momentum_period / 10) * (num_top_stocks / 5) * (rsi_period / 45) * (rsi_oversold / 30)
                        
                        self.test_results.append({
                            'params': {
                                'daily_params': {
                                    'momentum_period': momentum_period,
                                    'num_top_stocks': num_top_stocks
                                },
                                'minute_params': {
                                    'rsi_period': rsi_period,
                                    'rsi_oversold': rsi_oversold
                                }
                            },
                            'metrics': {
                                'sharpe_ratio': sharpe_ratio,
                                'total_return': sharpe_ratio * 0.1,
                                'max_drawdown': -sharpe_ratio * 0.05,
                                'win_rate': 0.5 + (sharpe_ratio * 0.1)
                            }
                        })
        
        # 테스트용 디렉토리 설정
        self.test_dir = 'test_analysis_results'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
    
    def tearDown(self):
        """테스트 정리"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_prepare_analysis_dataframe(self):
        """분석용 데이터프레임 생성 테스트"""
        # 데이터프레임 생성
        df = self.analyzer._prepare_analysis_dataframe(self.test_results)
        
        # 기본 검증
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.test_results))
        
        # 컬럼 검증
        expected_columns = [
            'daily_params.momentum_period',
            'daily_params.num_top_stocks',
            'minute_params.rsi_period',
            'minute_params.rsi_oversold',
            'sharpe_ratio',
            'total_return',
            'max_drawdown',
            'win_rate'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)
    
    def test_analyze_parameter_impact(self):
        """파라미터 영향도 분석 테스트"""
        # 영향도 분석 실행
        impact_analysis = self.analyzer.analyze_parameter_impact(
            self.test_results,
            target_metric='sharpe_ratio',
            save_dir=self.test_dir
        )
        
        # 결과 검증
        self.assertIsNotNone(impact_analysis)
        self.assertIn('daily_params', impact_analysis)
        self.assertIn('minute_params', impact_analysis)
        
        # 일봉 전략 파라미터 검증
        daily_impact = impact_analysis['daily_params']
        self.assertIn('momentum_period', daily_impact)
        self.assertIn('num_top_stocks', daily_impact)
        
        # 분봉 전략 파라미터 검증
        minute_impact = impact_analysis['minute_params']
        self.assertIn('rsi_period', minute_impact)
        self.assertIn('rsi_oversold', minute_impact)
        
        # 그래프 파일 존재 여부 검증
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'parameter_impact.png')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'correlation_heatmap.png')))
    
    def test_analyze_parameter_impact_with_empty_results(self):
        """빈 결과로 파라미터 영향도 분석 테스트"""
        # 빈 결과로 분석 실행
        impact_analysis = self.analyzer.analyze_parameter_impact(
            [],
            target_metric='sharpe_ratio',
            save_dir=self.test_dir
        )
        
        # 결과 검증
        self.assertIsNone(impact_analysis)
    
    def test_analyze_parameter_impact_with_invalid_metric(self):
        """잘못된 지표로 파라미터 영향도 분석 테스트"""
        # 잘못된 지표로 분석 실행
        impact_analysis = self.analyzer.analyze_parameter_impact(
            self.test_results,
            target_metric='invalid_metric',
            save_dir=self.test_dir
        )
        
        # 결과 검증
        self.assertIsNotNone(impact_analysis)
        self.assertIn('daily_params', impact_analysis)
        self.assertIn('minute_params', impact_analysis)
    
    def test_generate_analysis_report(self):
        """분석 보고서 생성 테스트"""
        # 보고서 생성
        report_path = self.analyzer.generate_analysis_report(
            self.test_results,
            save_dir=self.test_dir
        )
        
        # 결과 검증
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'parameter_impact.png')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'correlation_heatmap.png')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'parameter_distribution.png')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'metric_distribution.png')))
    
    def test_generate_analysis_report_with_empty_results(self):
        """빈 결과로 분석 보고서 생성 테스트"""
        # 빈 결과로 보고서 생성
        report_path = self.analyzer.generate_analysis_report(
            [],
            save_dir=self.test_dir
        )
        
        # 결과 검증
        self.assertIsNone(report_path)

if __name__ == '__main__':
    unittest.main() 