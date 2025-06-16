import unittest
from datetime import date, timedelta
from typing import Dict, List, Type, Any, Optional
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import os
import json

from optimizer.grid_search import GridSearchOptimizer
from optimizer.strategy_optimizer import StrategyOptimizer
from optimizer.market_optimizer import MarketOptimizer
from optimizer.parameter_analyzer import ParameterAnalyzer
from optimizer.base_optimizer import BaseOptimizer

class TestOptimizerData(unittest.TestCase):
    """실제 데이터 기반 옵티마이저 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        # 테스트 기간 설정
        self.end_date = date.today()
        self.start_date = self.end_date - timedelta(days=365)
        
        # 테스트 데이터 디렉토리 설정
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # 실제 시장 데이터 로드 또는 생성
        self.market_data = self._load_or_generate_market_data()
        
        # 테스트용 전략 클래스 정의
        class RealDailyStrategy:
            def __init__(self, **params):
                self.window = params.get('window', 20)
                self.threshold = params.get('threshold', 0.02)
                self.momentum_period = params.get('momentum_period', 10)
                self.volume_ma = params.get('volume_ma', 5)
                self.position = 0
                self.trades = []
            
            def set_parameters(self, params: Dict[str, Any]) -> None:
                """파라미터 설정"""
                self.window = params.get('window', self.window)
                self.threshold = params.get('threshold', self.threshold)
                self.momentum_period = params.get('momentum_period', self.momentum_period)
                self.volume_ma = params.get('volume_ma', self.volume_ma)
            
            def run_daily_logic(self, data: pd.DataFrame) -> pd.DataFrame:
                """일봉 데이터 처리"""
                if data is None or data.empty:
                    return pd.DataFrame()
                
                # 이동평균 계산
                data['ma'] = data['close'].rolling(window=self.window).mean()
                
                # 모멘텀 계산
                data['momentum'] = data['close'].pct_change(periods=self.momentum_period)
                
                # 변동성 계산
                data['volatility'] = data['close'].rolling(window=self.window).std()
                
                # 거래량 이동평균
                data['volume_ma'] = data['volume'].rolling(window=self.volume_ma).mean()
                
                # 매매 신호 생성
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
                        'position': self.position,
                        'volume': self._calculate_position_size(price, signal)
                    })
                    self.position = signal
            
            def _calculate_position_size(self, price: float, signal: int) -> int:
                """포지션 크기 계산"""
                base_size = 100  # 기본 주문 수량
                return base_size * abs(signal)
        
        class RealMinuteStrategy:
            def __init__(self, **params):
                self.window = params.get('window', 30)
                self.threshold = params.get('threshold', 0.01)
                self.rsi_period = params.get('rsi_period', 14)
                self.volume_ma = params.get('volume_ma', 5)
                self.position = 0
                self.trades = []
            
            def set_parameters(self, params: Dict[str, Any]) -> None:
                """파라미터 설정"""
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
                
                # 매매 신호 생성
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
                        'position': self.position,
                        'volume': self._calculate_position_size(price, signal)
                    })
                    self.position = signal
            
            def _calculate_position_size(self, price: float, signal: int) -> int:
                """포지션 크기 계산"""
                base_size = 50  # 기본 주문 수량
                return base_size * abs(signal)
        
        self.real_daily_strategy = RealDailyStrategy
        self.real_minute_strategy = RealMinuteStrategy
        
        # 실제 전략용 파라미터 그리드
        self.param_grids = {
            'RealDailyStrategy': {
                'window': [10, 20, 30, 40],
                'threshold': [0.01, 0.02, 0.03, 0.04],
                'volume_ma': [3, 5, 7, 10]
            },
            'RealMinuteStrategy': {
                'window': [20, 30, 40, 50],
                'threshold': [0.005, 0.01, 0.015, 0.02],
                'rsi_period': [9, 14, 21, 28]
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
    
    def _load_or_generate_market_data(self) -> Dict[str, pd.DataFrame]:
        """실제 시장 데이터 로드 또는 생성"""
        data_file = os.path.join(self.test_data_dir, 'market_data.json')
        
        if os.path.exists(data_file):
            try:
                # 저장된 데이터 로드
                with open(data_file, 'r') as f:
                    data_dict = json.load(f)
                    
                    # 데이터프레임 생성 및 인덱스 설정
                    daily_df = pd.DataFrame(data_dict['daily'])
                    minute_df = pd.DataFrame(data_dict['minute'])
                    
                    # 날짜 컬럼이 있는지 확인
                    if 'date' not in daily_df.columns or 'date' not in minute_df.columns:
                        raise KeyError("'date' 컬럼이 없습니다")
                    
                    # 날짜 컬럼을 datetime으로 변환하여 인덱스로 설정
                    daily_df['date'] = pd.to_datetime(daily_df['date'])
                    minute_df['date'] = pd.to_datetime(minute_df['date'])
                    
                    # 숫자형 컬럼 변환
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    for df in [daily_df, minute_df]:
                        for col in numeric_columns:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    return {
                        'daily': daily_df.set_index('date'),
                        'minute': minute_df.set_index('date')
                    }
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"데이터 로드 중 오류 발생: {e}")
                # 오류 발생 시 파일 삭제하고 새로 생성
                if os.path.exists(data_file):
                    os.remove(data_file)
        
        # 새로운 데이터 생성 및 저장
        market_data = self._generate_realistic_market_data()
        
        # JSON 직렬화를 위해 DataFrame을 딕셔너리로 변환
        data_dict = {
            'daily': market_data['daily'].reset_index().rename(columns={'index': 'date'}).to_dict(orient='records'),
            'minute': market_data['minute'].reset_index().rename(columns={'index': 'date'}).to_dict(orient='records')
        }
        
        # 데이터 저장 (숫자형 값은 그대로 저장)
        with open(data_file, 'w') as f:
            json.dump(data_dict, f, default=str)
        
        return market_data
    
    def _generate_realistic_market_data(self) -> Dict[str, pd.DataFrame]:
        """현실적인 시장 데이터 생성"""
        # 일봉 데이터 생성
        daily_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='D'
        )
        
        # 실제 시장과 유사한 가격 움직임 생성
        np.random.seed(42)  # 재현성을 위한 시드 설정
        daily_returns = np.random.normal(0.0001, 0.015, len(daily_dates))
        daily_prices = 100 * (1 + daily_returns).cumprod()
        
        daily_data = pd.DataFrame({
            'open': daily_prices * (1 + np.random.normal(0, 0.002, len(daily_dates))),
            'high': daily_prices * (1 + np.abs(np.random.normal(0, 0.003, len(daily_dates)))),
            'low': daily_prices * (1 - np.abs(np.random.normal(0, 0.003, len(daily_dates)))),
            'close': daily_prices,
            'volume': np.random.lognormal(10, 1, len(daily_dates))
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
        
        # 일봉 데이터를 기반으로 분봉 데이터 생성
        minute_prices = []
        for date in daily_dates:
            daily_price = daily_data.loc[date, 'close']
            daily_volatility = (daily_data.loc[date, 'high'] - daily_data.loc[date, 'low']) / daily_data.loc[date, 'close']
            
            # 해당 일자의 분봉 데이터 생성
            day_minutes = minute_dates[minute_dates.date == date.date()]
            if len(day_minutes) > 0:
                minute_returns = np.random.normal(0, daily_volatility/len(day_minutes), len(day_minutes))
                day_prices = daily_price * (1 + minute_returns).cumprod()
                minute_prices.extend(day_prices)
        
        minute_data = pd.DataFrame({
            'open': minute_prices * (1 + np.random.normal(0, 0.001, len(minute_dates))),
            'high': minute_prices * (1 + np.abs(np.random.normal(0, 0.002, len(minute_dates)))),
            'low': minute_prices * (1 - np.abs(np.random.normal(0, 0.002, len(minute_dates)))),
            'close': minute_prices,
            'volume': np.random.lognormal(8, 0.5, len(minute_dates))
        }, index=minute_dates)
        
        return {
            'daily': daily_data,
            'minute': minute_data
        }
    
    @patch('optimizer.market_optimizer.MarketOptimizer._get_market_data')
    def test_real_market_optimization(self, mock_get_market_data):
        """실제 시장 데이터 기반 최적화 테스트"""
        # 시장 데이터 모의 설정
        mock_get_market_data.return_value = self.market_data
        
        # 1. 시장 상태 분석
        market_results = self.market_optimizer.optimize(
            daily_strategies={'RealDailyStrategy': self.real_daily_strategy},
            minute_strategies={'RealMinuteStrategy': self.real_minute_strategy},
            param_grids=self.param_grids,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        if market_results is None:
            self.fail("시장 상태 분석 실패")
        
        self.assertIn('market_state', market_results)
        
        # 2. 전략 최적화
        strategy_results = self.strategy_optimizer.optimize(
            daily_strategies={'RealDailyStrategy': self.real_daily_strategy},
            minute_strategies={'RealMinuteStrategy': self.real_minute_strategy},
            param_grids=self.param_grids,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        if strategy_results is None:
            self.fail("전략 최적화 실패")
        
        self.assertIn('best_params', strategy_results)
    
    def test_market_state_impact(self):
        """시장 상태별 성과 영향 테스트"""
        market_states = ['bull', 'bear', 'sideways', 'volatile']
        state_results = {}
        
        for state in market_states:
            # 각 시장 상태별 최적화 실행
            results = self.market_optimizer.optimize(
                daily_strategies={'RealDailyStrategy': self.real_daily_strategy},
                minute_strategies={'RealMinuteStrategy': self.real_minute_strategy},
                param_grids=self.param_grids,
                start_date=self.start_date,
                end_date=self.end_date,
                market_state=state
            )
            
            if results is None:
                print(f"시장 상태 '{state}'에서 최적화 실패")
                continue
            
            # 최적 파라미터로 백테스트 실행
            best_params = results.get('best_params')
            if best_params is None:
                print(f"시장 상태 '{state}'에서 최적 파라미터를 찾을 수 없음")
                continue
            
            # 전략 인스턴스 생성
            daily_strategy = self.real_daily_strategy(params=best_params.get('RealDailyStrategy', {}))
            minute_strategy = self.real_minute_strategy(params=best_params.get('RealMinuteStrategy', {}))
            
            # 전략 실행
            daily_signals = daily_strategy.run_daily_logic(self.market_data['daily'])
            minute_signals = minute_strategy.run_minute_logic(self.market_data['minute'])
            
            # 거래 실행
            daily_trades = self._execute_trades(daily_strategy, daily_signals)
            minute_trades = self._execute_trades(minute_strategy, minute_signals)
            
            # 수익률 계산
            daily_returns = self._calculate_returns(daily_trades)
            minute_returns = self._calculate_returns(minute_trades)
            
            state_results[state] = {
                'daily_returns': daily_returns,
                'minute_returns': minute_returns,
                'trade_count': len(daily_trades) + len(minute_trades)
            }
        
        # 시장 상태별 성과 검증
        self.assertGreater(len(state_results), 0)
        
        # 각 시장 상태별 최소 성과 기준 검증
        for state, results in state_results.items():
            if state == 'bull':
                self.assertGreaterEqual(results['daily_returns'], -0.1)  # 최대 10% 손실 허용
                self.assertGreaterEqual(results['minute_returns'], -0.05)  # 최대 5% 손실 허용
            elif state == 'bear':
                self.assertLessEqual(results['daily_returns'], 0.1)  # 최대 10% 수익 허용
                self.assertLessEqual(results['minute_returns'], 0.05)  # 최대 5% 수익 허용
            elif state == 'sideways':
                self.assertGreaterEqual(results['daily_returns'], -0.05)  # 최대 5% 손실 허용
                self.assertGreaterEqual(results['minute_returns'], -0.03)  # 최대 3% 손실 허용
            elif state == 'volatile':
                self.assertGreater(results['trade_count'], 0)  # 거래 발생 확인
    
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
                pnl = price_diff * prev_trade['position'] * prev_trade['volume']
                total_pnl += pnl
        
        initial_capital = trades[0]['price'] * abs(trades[0]['position']) * trades[0]['volume']
        return total_pnl / initial_capital if initial_capital > 0 else 0.0

if __name__ == '__main__':
    unittest.main() 