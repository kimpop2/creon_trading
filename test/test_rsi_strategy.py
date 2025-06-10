import unittest
import pandas as pd
import numpy as np
import datetime
import sys
import os

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.rsi_minute_strategy import RSIMinuteStrategy

class TestRSIStrategy(unittest.TestCase):
    def setUp(self):
        """테스트 설정"""
        # RSI 전략 인스턴스 생성 (run_dual_momentum_week_onefile.py의 파라미터와 동일하게 설정)
        self.strategy = RSIMinuteStrategy(params={
            'rsi_period': 45,          # 분봉 RSI 기간 (60분 → 45분)
            'rsi_upper': 65,           # RSI 과매수 기준 (70 → 65)
            'rsi_lower': 35,           # RSI 과매도 기준 (30 → 35)
            'morning_entry_hour': 10,   # 매수 시작 시간 (시)
            'morning_entry_minute': 0,  # 매수 시작 시간 (분)
            'morning_exit_hour': 9,     # 매도 시작 시간 (시)
            'morning_exit_minute': 5,   # 매도 시작 시간 (분)
            'force_entry_hour': 15,     # 강제 매수 시간 (시)
            'force_entry_minute': 20,   # 강제 매수 시간 (분)
            'force_exit_hour': 15,      # 강제 매도 시간 (시)
            'force_exit_minute': 25     # 강제 매도 시간 (분)
        })
        
        # 테스트용 시계열 인덱스 생성 (9:00 ~ 15:30)
        self.dates = pd.date_range(
            start='2025-01-01 09:00:00',
            end='2025-01-01 15:30:00',
            freq='1min'
        )
        
        # 기본 OHLCV 데이터 생성
        self.base_data = pd.DataFrame({
            'open': np.ones(len(self.dates)) * 100,
            'high': np.ones(len(self.dates)) * 100,
            'low': np.ones(len(self.dates)) * 100,
            'close': np.ones(len(self.dates)) * 100,
            'volume': np.ones(len(self.dates)) * 1000
        }, index=self.dates)
        
        # 과매수 상태의 데이터 생성 (RSI > 65)
        self.overbought_data = self.base_data.copy()
        for i in range(45, len(self.dates)):  # RSI 기간(45)부터 시작
            if i % 2 == 0:
                self.overbought_data.loc[self.dates[i], 'close'] = 101
            else:
                self.overbought_data.loc[self.dates[i], 'close'] = 100.5
        
        # 과매도 상태의 데이터 생성 (RSI < 35)
        self.oversold_data = self.base_data.copy()
        for i in range(45, len(self.dates)):  # RSI 기간(45)부터 시작
            if i % 2 == 0:
                self.oversold_data.loc[self.dates[i], 'close'] = 99
            else:
                self.oversold_data.loc[self.dates[i], 'close'] = 99.5

    def test_calculate_rsi(self):
        """RSI 계산 테스트"""
        # 과매수 데이터 RSI 테스트
        rsi_overbought = self.strategy.calculate_rsi(self.overbought_data)
        self.assertGreater(rsi_overbought.iloc[-1], 65)
        
        # 과매도 데이터 RSI 테스트
        rsi_oversold = self.strategy.calculate_rsi(self.oversold_data)
        self.assertLess(rsi_oversold.iloc[-1], 35)
        
        # RSI 값 범위 확인 (0~100)
        self.assertTrue(all(0 <= x <= 100 for x in rsi_overbought.dropna()))
        self.assertTrue(all(0 <= x <= 100 for x in rsi_oversold.dropna()))

    def test_should_execute_buy_morning_with_position(self):
        """보유 포지션이 있을 때 매수 실행 테스트"""
        current_time = datetime.datetime(2025, 1, 1, 10, 0)
        should_execute = self.strategy.should_execute(
            self.oversold_data, 'buy', current_time, current_position_size=100
        )
        self.assertFalse(should_execute)  # 이미 포지션이 있으면 매수하지 않음

    def test_should_execute_buy_morning_without_position(self):
        """보유 포지션이 없을 때 매수 실행 테스트"""
        # 오전 10시, 과매도 상태, 포지션 없음
        current_time = datetime.datetime(2025, 1, 1, 10, 0)
        should_execute = self.strategy.should_execute(
            self.oversold_data, 'buy', current_time, current_position_size=0
        )
        self.assertTrue(should_execute)
        
        # 오전 10시 이전, 과매도 상태, 포지션 없음
        current_time = datetime.datetime(2025, 1, 1, 9, 30)
        should_execute = self.strategy.should_execute(
            self.oversold_data, 'buy', current_time, current_position_size=0
        )
        self.assertFalse(should_execute)

    def test_should_execute_buy_force(self):
        """강제 매수 조건 테스트"""
        # 오후 3시 20분, 포지션 없음
        current_time = datetime.datetime(2025, 1, 1, 15, 20)
        should_execute = self.strategy.should_execute(
            self.base_data, 'buy', current_time, current_position_size=0
        )
        self.assertTrue(should_execute)
        
        # 오후 3시 20분, 포지션 있음
        should_execute = self.strategy.should_execute(
            self.base_data, 'buy', current_time, current_position_size=100
        )
        self.assertFalse(should_execute)

    def test_should_execute_sell_morning(self):
        """오전 강제 매도 조건 테스트"""
        # 오전 9시 5분, 포지션 있음
        current_time = datetime.datetime(2025, 1, 1, 9, 5)
        should_execute = self.strategy.should_execute(
            self.base_data, 'sell', current_time, current_position_size=100
        )
        self.assertTrue(should_execute)
        
        # 오전 9시 5분, 포지션 없음
        should_execute = self.strategy.should_execute(
            self.base_data, 'sell', current_time, current_position_size=0
        )
        self.assertFalse(should_execute)

    def test_should_execute_sell_rsi(self):
        """RSI 기반 매도 조건 테스트"""
        # 과매수 상태, 포지션 있음
        current_time = datetime.datetime(2025, 1, 1, 10, 30)
        should_execute = self.strategy.should_execute(
            self.overbought_data, 'sell', current_time, current_position_size=100
        )
        self.assertTrue(should_execute)
        
        # 과매수 상태, 포지션 없음
        should_execute = self.strategy.should_execute(
            self.overbought_data, 'sell', current_time, current_position_size=0
        )
        self.assertFalse(should_execute)

    def test_should_execute_sell_force(self):
        """장 마감 강제 매도 조건 테스트"""
        # 오후 3시 25분, 포지션 있음
        current_time = datetime.datetime(2025, 1, 1, 15, 25)
        should_execute = self.strategy.should_execute(
            self.base_data, 'sell', current_time, current_position_size=100
        )
        self.assertTrue(should_execute)
        
        # 오후 3시 25분, 포지션 없음
        should_execute = self.strategy.should_execute(
            self.base_data, 'sell', current_time, current_position_size=0
        )
        self.assertFalse(should_execute)

    def test_should_not_execute_hold(self):
        """홀딩 신호 테스트"""
        current_time = datetime.datetime(2025, 1, 1, 11, 0)
        should_execute = self.strategy.should_execute(
            self.base_data, 'hold', current_time, current_position_size=100
        )
        self.assertFalse(should_execute)

    def test_calculate_execution_price(self):
        """실행 가격 계산 테스트"""
        # 마지막 종가 확인
        last_price = 105.0
        test_data = self.base_data.copy()
        test_data.iloc[-1]['close'] = last_price
        
        price = self.strategy.calculate_execution_price(test_data)
        self.assertEqual(price, last_price)

if __name__ == '__main__':
    unittest.main() 