"""
파일명: test_sma_strategy.py
설명: SMA 전략 단위 테스트 코드
작성일: 2024-03-19
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import time

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.sma_daily_strategy import SMADailyStrategy
from api.creon_api import CreonAPIClient

class TestSMAStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화 - Creon API 연결 및 삼성전자 데이터 로드"""
        cls.creon_api = CreonAPIClient()
        if not cls.creon_api.connected:
            raise unittest.SkipTest("Creon API에 연결할 수 없습니다.")
        
        # 삼성전자 종목코드 조회
        cls.stock_code = cls.creon_api.get_stock_code('삼성전자')
        if not cls.stock_code:
            raise unittest.SkipTest("삼성전자 종목 코드를 찾을 수 없습니다.")
        
        # 테스트 기간 설정 (현재 날짜 기준 1달)
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        # 일봉 데이터 로드
        cls.test_data = cls.creon_api.get_daily_ohlcv(cls.stock_code, start_date, end_date)
        if cls.test_data.empty:
            raise unittest.SkipTest("삼성전자 데이터를 가져올 수 없습니다.")
        
        print("\n=== 테스트 데이터 정보 ===")
        print(f"종목: 삼성전자 ({cls.stock_code})")
        print(f"기간: {start_date} ~ {end_date}")
        print(f"데이터 수: {len(cls.test_data)}행")
        print("======================\n")

    def setUp(self):
        """각 테스트 케이스 실행 전 초기화"""
        self.strategy = SMADailyStrategy(params={
            'short_window': 5,
            'long_window': 20,
            'volume_window': 20,
            'volume_ratio': 1.0,
            'lookback_window': 10,
            'trend_threshold': 0.001,
            'stop_loss': -0.05,
            'trailing_stop': -0.03,
            'take_profit': 0.1,
            'position_size': 1.0
        })

    def test_calculate_indicators(self):
        """지표 계산 메서드 테스트"""
        data = self.strategy.calculate_indicators(self.test_data.copy())
        
        # 필수 지표들이 계산되었는지 확인
        required_columns = [
            'SMA_short', 'SMA_long', 'MA_diff', 'MA_diff_prev',
            'SMA_short_pct', 'SMA_long_pct', 'volume_MA', 'volume_ratio'
        ]
        for col in required_columns:
            self.assertIn(col, data.columns)
        
        # 이동평균선이 계산되었는지 확인
        self.assertEqual(data['SMA_short'].isna().sum(), self.strategy.params['short_window'] - 1)
        self.assertEqual(data['SMA_long'].isna().sum(), self.strategy.params['long_window'] - 1)
        
        # 결과 출력
        print("\n=== 지표 계산 결과 ===")
        print(data[['close', 'SMA_short', 'SMA_long', 'volume_ratio']].tail())
        print("==================\n")

    def test_check_trend(self):
        """추세 확인 메서드 테스트"""
        data = self.strategy.calculate_indicators(self.test_data.copy())
        last_idx = len(data) - 1
        
        trend_result = self.strategy.check_trend(data, last_idx)
        print("\n=== 추세 확인 결과 ===")
        print(f"최근 5일 종가: {data['close'].tail().values}")
        print(f"추세 판정: {'상승/하락' if trend_result else '횡보'}")
        print("==================\n")

    def test_check_cross(self):
        """이동평균선 교차 확인 메서드 테스트"""
        data = self.strategy.calculate_indicators(self.test_data.copy())
        cross_type = self.strategy.check_cross(data)
        
        print("\n=== 이동평균선 교차 확인 ===")
        print(f"최근 이동평균선 교차: {cross_type}")
        print("최근 5일 데이터:")
        print(data[['close', 'SMA_short', 'SMA_long', 'MA_diff']].tail())
        print("======================\n")

    def test_check_risk_management(self):
        """리스크 관리 메서드 테스트"""
        # 마지막 종가 기준으로 테스트
        current_price = self.test_data['close'].iloc[-1]
        entry_price = current_price * 0.95  # 5% 낮은 가격으로 매수했다고 가정
        
        # 포지션 설정
        self.strategy.position_info[self.stock_code] = {
            'entry_price': entry_price,
            'highest_price': current_price
        }
        
        # 리스크 관리 신호 확인
        signal = self.strategy.check_risk_management(self.stock_code, current_price)
        
        print("\n=== 리스크 관리 테스트 ===")
        print(f"매수가: {entry_price:,.0f}원")
        print(f"현재가: {current_price:,.0f}원")
        print(f"수익률: {((current_price/entry_price)-1)*100:.2f}%")
        print(f"리스크 관리 신호: {signal}")
        print("=====================\n")

    def test_generate_signals(self):
        """매매 신호 생성 메서드 테스트"""
        data = self.strategy.calculate_indicators(self.test_data.copy())
        signals = self.strategy.generate_signals(data)
        
        print("\n=== 매매 신호 생성 결과 ===")
        print(f"생성된 신호: {signals}")
        print("최근 5일 데이터:")
        print(data[['close', 'SMA_short', 'SMA_long', 'volume_ratio']].tail())
        print("======================\n")

if __name__ == '__main__':
    unittest.main() 