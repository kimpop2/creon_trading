import unittest
import pandas as pd
import datetime
import sys
import os

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import BaseStrategy, DailyStrategy, MinuteStrategy

class MockDailyStrategy(DailyStrategy):
    def calculate_indicators(self, data):
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        return data
        
    def generate_daily_signals(self, data):
        return {'mock_stock': 'buy'}
        
    def generate_signals(self, data):
        return self.generate_daily_signals(data)

class MockMinuteStrategy(MinuteStrategy):
    def should_execute(self, data, signal, current_time):
        return True
        
    def calculate_execution_price(self, data):
        return data['close'].iloc[-1]
        
    def generate_signals(self, data):
        return {'mock_stock': 'execute'}

class TestBaseStrategies(unittest.TestCase):
    def setUp(self):
        # 테스트용 데이터 생성
        self.test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [103, 104, 105],
            'low': [98, 99, 100],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2025-01-01', periods=3))
        
        self.daily_strategy = MockDailyStrategy('mock_daily')
        self.minute_strategy = MockMinuteStrategy('mock_minute')

    def test_data_validation(self):
        """데이터 유효성 검증 테스트"""
        # 정상 데이터
        self.assertTrue(self.daily_strategy.validate_data(self.test_data))
        
        # 비정상 데이터
        invalid_data = pd.DataFrame({'close': [100, 101]})
        self.assertFalse(self.daily_strategy.validate_data(invalid_data))

    def test_daily_strategy(self):
        """일봉 전략 테스트"""
        # 지표 계산 테스트
        data_with_indicators = self.daily_strategy.calculate_indicators(self.test_data)
        self.assertIn('SMA_20', data_with_indicators.columns)
        
        # 신호 생성 테스트
        signals = self.daily_strategy.generate_signals(self.test_data)
        self.assertEqual(signals['mock_stock'], 'buy')

    def test_minute_strategy(self):
        """분봉 전략 테스트"""
        # 실행 조건 테스트
        current_time = datetime.datetime.now()
        should_execute = self.minute_strategy.should_execute(
            self.test_data, 'buy', current_time
        )
        self.assertTrue(should_execute)
        
        # 실행 가격 계산 테스트
        price = self.minute_strategy.calculate_execution_price(self.test_data)
        self.assertEqual(price, 103)  # 마지막 종가

if __name__ == '__main__':
    unittest.main() 