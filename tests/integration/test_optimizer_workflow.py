import unittest
from datetime import date, timedelta
from typing import Dict, List, Type, Any, Optional
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from optimizer.grid_search import GridSearchOptimizer
from optimizer.strategy_optimizer import StrategyOptimizer
from optimizer.market_optimizer import MarketOptimizer
from optimizer.parameter_analyzer import ParameterAnalyzer
from optimizer.base_optimizer import BaseOptimizer

class TestOptimizerWorkflow(unittest.TestCase):
    """옵티마이저 워크플로우 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        # 테스트 기간 설정
        self.end_date = date.today()
        self.start_date = self.end_date - timedelta(days=365)
        
        # 실제 시장 데이터 모의 생성
        self.market_data = self._generate_mock_market_data()
        
        # 테스트용 전략 클래스 정의
        class MockDailyStrategy:
            def __init__(self, params: Dict[str, Any]):
                self.params = params
                self.name = "MockDailyStrategy"
                self.position = 0
                self.trades = []
                self.window = params.get('window', 5)
                self.threshold = params.get('threshold', 0.01)
                self.momentum_period = params.get('momentum_period', 10)
                self.volume_ma = params.get('volume_ma', 5)
            
            def set_parameters(self, params: Dict[str, Any]) -> None:
                self.params = params
                self.window = params.get('window', self.window)
                self.threshold = params.get('threshold', self.threshold)
                self.momentum_period = params.get('momentum_period', self.momentum_period)
                self.volume_ma = params.get('volume_ma', self.volume_ma)
            
            def run_daily_logic(self, data: pd.DataFrame) -> pd.DataFrame:
                if data is None or data.empty:
                    return pd.DataFrame()
                
                # 이동평균 계산
                data['ma'] = data['close'].rolling(window=self.window).mean()
                # 모멘텀 계산
                data['momentum'] = data['close'].pct_change(self.momentum_period)
                # 거래량 이동평균
                data['volume_ma'] = data['volume'].rolling(window=self.volume_ma).mean()
                # 시그널 생성
                data['signal'] = np.where(
                    (data['close'] > data['ma']) & (data['momentum'] > self.threshold),
                    1,
                    np.where(
                        (data['close'] < data['ma']) & (data['momentum'] < -self.threshold),
                        -1,
                        0
                    )
                )
                return data
            
            def run_minute_logic(self, data: pd.DataFrame) -> pd.DataFrame:
                """분봉 데이터 처리 (일봉 전략이므로 빈 DataFrame 반환)"""
                return pd.DataFrame()
            
            def execute_trade(self, signal: int, price: float, timestamp: pd.Timestamp) -> None:
                if signal != 0 and signal != self.position:
                    self.trades.append({
                        'timestamp': timestamp,
                        'price': price,
                        'signal': signal,
                        'position': self.position
                    })
                    self.position = signal
        
        class MockMinuteStrategy:
            def __init__(self, params: Dict[str, Any]):
                self.params = params
                self.name = "MockMinuteStrategy"
                self.position = 0
                self.trades = []
                self.window = params.get('window', 10)
                self.threshold = params.get('threshold', 0.005)
                self.rsi_period = params.get('rsi_period', 14)
                self.volume_ma = params.get('volume_ma', 5)
            
            def set_parameters(self, params: Dict[str, Any]) -> None:
                self.params = params
                self.window = params.get('window', self.window)
                self.threshold = params.get('threshold', self.threshold)
                self.rsi_period = params.get('rsi_period', self.rsi_period)
                self.volume_ma = params.get('volume_ma', self.volume_ma)
            
            def run_daily_logic(self, data: pd.DataFrame) -> pd.DataFrame:
                """일봉 데이터 처리 (분봉 전략이므로 빈 DataFrame 반환)"""
                return pd.DataFrame()
            
            def run_minute_logic(self, data: pd.DataFrame) -> pd.DataFrame:
                if data is None or data.empty:
                    return pd.DataFrame()
                
                # 이동평균 계산
                data['ma'] = data['close'].rolling(window=self.window).mean()
                # RSI 계산
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
                rs = gain / loss
                data['rsi'] = 100 - (100 / (1 + rs))
                # 거래량 이동평균
                data['volume_ma'] = data['volume'].rolling(window=self.volume_ma).mean()
                # 시그널 생성
                data['signal'] = np.where(
                    (data['rsi'] < 30) & (data['close'] < data['ma'] * (1 - self.threshold)),
                    1,
                    np.where(
                        (data['rsi'] > 70) & (data['close'] > data['ma'] * (1 + self.threshold)),
                        -1,
                        0
                    )
                )
                return data
            
            def execute_trade(self, signal: int, price: float, timestamp: pd.Timestamp) -> None:
                if signal != 0 and signal != self.position:
                    self.trades.append({
                        'timestamp': timestamp,
                        'price': price,
                        'signal': signal,
                        'position': self.position
                    })
                    self.position = signal
        
        self.mock_daily_strategy = MockDailyStrategy
        self.mock_minute_strategy = MockMinuteStrategy
        
        # 테스트용 파라미터 그리드
        self.param_grids = {
            'MockDailyStrategy': {
                'window': [5, 10, 20, 30],
                'threshold': [0.01, 0.02, 0.03, 0.04],
                'momentum_period': [10, 20, 30, 40],
                'volume_ma': [5, 10, 15, 20]
            },
            'MockMinuteStrategy': {
                'window': [10, 20, 30, 40],
                'threshold': [0.005, 0.01, 0.015, 0.02],
                'rsi_period': [14, 28, 42, 56],
                'volume_ma': [5, 10, 15, 20]
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
    
    def _generate_mock_market_data(self) -> Dict[str, pd.DataFrame]:
        """테스트용 시장 데이터 생성"""
        # 일봉 데이터 생성
        daily_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='D'
        )
        
        daily_data = pd.DataFrame({
            'open': np.random.normal(100, 2, len(daily_dates)),
            'high': np.random.normal(102, 2, len(daily_dates)),
            'low': np.random.normal(98, 2, len(daily_dates)),
            'close': np.random.normal(100, 2, len(daily_dates)),
            'volume': np.random.normal(1000000, 200000, len(daily_dates))
        }, index=daily_dates)
        
        # 분봉 데이터 생성 (장중 데이터만)
        minute_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='1min'
        )
        minute_dates = minute_dates[
            (minute_dates.time >= pd.Timestamp('09:00').time()) &
            (minute_dates.time <= pd.Timestamp('15:30').time())
        ]
        
        minute_data = pd.DataFrame({
            'open': np.random.normal(100, 1, len(minute_dates)),
            'high': np.random.normal(101, 1, len(minute_dates)),
            'low': np.random.normal(99, 1, len(minute_dates)),
            'close': np.random.normal(100, 1, len(minute_dates)),
            'volume': np.random.normal(10000, 2000, len(minute_dates))
        }, index=minute_dates)
        
        return {
            'daily': daily_data,
            'minute': minute_data
        }
    
    @patch('optimizer.market_optimizer.MarketOptimizer._get_market_data')
    def test_complete_trading_workflow(self, mock_get_market_data):
        """전체 트레이딩 워크플로우 테스트"""
        # 시장 데이터 모의 설정
        mock_get_market_data.return_value = self.market_data
        
        # 1. 시장 상태 분석 및 최적화
        market_results = self.market_optimizer.optimize(
            daily_strategies={'MockDailyStrategy': self.mock_daily_strategy},
            minute_strategies={'MockMinuteStrategy': self.mock_minute_strategy},
            param_grids=self.param_grids,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        if market_results is None:
            self.fail("시장 상태 분석 실패")
        
        self.assertIn('market_state', market_results)
        
        # 2. 시장 상태에 따른 파라미터 조정
        adjusted_params = market_results.get('best_params', self.param_grids)
        self.assertIsNotNone(adjusted_params)
        
        # 3. 전략 조합 최적화
        strategy_results = self.strategy_optimizer.optimize(
            daily_strategies={'MockDailyStrategy': self.mock_daily_strategy},
            minute_strategies={'MockMinuteStrategy': self.mock_minute_strategy},
            param_grids=adjusted_params,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        if strategy_results is None:
            self.fail("전략 최적화 실패")
        
        self.assertIn('best_combination', strategy_results)
        
        # 4. 파라미터 영향도 분석
        analysis_results = self.parameter_analyzer.analyze_parameter_impact(
            results=[market_results, strategy_results],
            target_metric='sharpe_ratio'
        )
        self.assertIsNotNone(analysis_results)
        
        # 5. 최종 전략 실행 및 성과 평가
        best_params = strategy_results['best_params']
        daily_strategy = self.mock_daily_strategy(params=best_params.get('MockDailyStrategy', {}))
        minute_strategy = self.mock_minute_strategy(params=best_params.get('MockMinuteStrategy', {}))
        
        # 일봉 전략 실행
        daily_signals = daily_strategy.run_daily_logic(self.market_data['daily'])
        self.assertIn('signal', daily_signals.columns)
        
        # 분봉 전략 실행
        minute_signals = minute_strategy.run_minute_logic(self.market_data['minute'])
        self.assertIn('signal', minute_signals.columns)
        
        # 거래 실행 및 성과 평가
        daily_trades = self._execute_trades(daily_strategy, daily_signals)
        minute_trades = self._execute_trades(minute_strategy, minute_signals)
        
        # 거래 결과 검증
        self.assertGreater(len(daily_trades), 0)
        self.assertGreater(len(minute_trades), 0)
        
        # 수익률 계산 및 검증
        daily_returns = self._calculate_returns(daily_trades)
        minute_returns = self._calculate_returns(minute_trades)
        
        self.assertIsNotNone(daily_returns)
        self.assertIsNotNone(minute_returns)
    
    def _execute_trades(
        self,
        strategy: Any,
        signals: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """거래 실행 시뮬레이션"""
        for timestamp, row in signals.iterrows():
            if 'signal' in row and not pd.isna(row['signal']):
                strategy.execute_trade(
                    signal=int(row['signal']),
                    price=float(row['close']),
                    timestamp=timestamp
                )
        return strategy.trades
    
    def _calculate_returns(self, trades: List[Dict[str, Any]]) -> float:
        """거래 수익률 계산"""
        if not trades:
            return 0.0
        
        total_pnl = 0.0
        for i in range(1, len(trades)):
            prev_trade = trades[i-1]
            curr_trade = trades[i]
            
            if prev_trade['position'] != 0:
                price_diff = curr_trade['price'] - prev_trade['price']
                pnl = price_diff * prev_trade['position']
                total_pnl += pnl
        
        initial_capital = trades[0]['price'] * abs(trades[0]['position'])
        return total_pnl / initial_capital if initial_capital > 0 else 0.0
    
    def test_error_handling_and_recovery(self):
        """오류 처리 및 복구 테스트"""
        # 1. 잘못된 파라미터로 테스트
        invalid_params = {
            'daily_params': {
                'window': [-1, 0],  # 잘못된 값
                'threshold': ['invalid']  # 잘못된 타입
            },
            'minute_params': {
                'window': [-1, 0],
                'threshold': ['invalid']
            }
        }
        
        with self.assertRaises(ValueError):
            self.grid_search.optimize(
                daily_strategy=self.mock_daily_strategy,
                minute_strategy=self.mock_minute_strategy,
                param_grid=invalid_params,
                start_date=self.start_date,
                end_date=self.end_date
            )
        
        # 2. 데이터 누락 상황 테스트
        incomplete_data = self.market_data.copy()
        incomplete_data['daily'] = incomplete_data['daily'].iloc[:10]  # 일부 데이터만 사용
        
        with patch('optimizer.market_optimizer.MarketOptimizer._get_market_data') as mock_get_data:
            mock_get_data.return_value = incomplete_data
            
            results = self.market_optimizer.optimize(
                daily_strategies={'MockDailyStrategy': self.mock_daily_strategy},
                minute_strategies={'MockMinuteStrategy': self.mock_minute_strategy},
                param_grids=self.param_grids,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            self.assertIsNone(results)  # 데이터가 부족하면 None 반환
        
        # 3. 복구 테스트 - 정상 파라미터로 재시도
        results = self.grid_search.optimize(
            daily_strategy=self.mock_daily_strategy,
            minute_strategy=self.mock_minute_strategy,
            param_grid={
                'daily_params': self.param_grids['MockDailyStrategy'],
                'minute_params': self.param_grids['MockMinuteStrategy']
            },
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        self.assertIsNotNone(results)  # 정상 파라미터로는 성공

if __name__ == '__main__':
    unittest.main() 