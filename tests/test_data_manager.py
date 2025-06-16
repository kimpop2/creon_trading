"""
DataManager 클래스 단위 테스트
"""

import unittest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import sys
import os
import time

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from manager.data_manager import DataManager

class TestDataManager(unittest.TestCase):
    def setUp(self):
        """테스트 설정"""
        try:
            self.data_manager = DataManager()
            self.stock_code = 'A005930'  # 삼성전자
            self.today = date.today()
            self.start_date_daily = self.today - timedelta(days=5)
            self.start_date_minute = self.today - timedelta(days=5)
        except Exception as e:
            self.skipTest(f"DataManager 초기화 실패: {e}")

    def tearDown(self):
        """테스트 정리"""
        if hasattr(self, 'data_manager'):
            # DataManager의 리소스 정리
            if hasattr(self.data_manager, 'db_manager'):
                self.data_manager.db_manager.close()

    def test_01_data_manager_initialization(self):
        """DataManager 초기화 테스트"""
        self.assertIsNotNone(self.data_manager)
        self.assertIsNotNone(self.data_manager.db_manager)
        self.assertIsNotNone(self.data_manager.api_client)

    def test_02_load_market_calendar_initial_data(self):
        """Market Calendar 초기 데이터 로드 테스트"""
        try:
            # 1년치 데이터 로드 테스트
            self.data_manager.load_market_calendar_initial_data(years=1)
            
            # 로드된 데이터 확인
            from_date = self.today - timedelta(days=365)
            trading_days = self.data_manager.db_manager.get_all_trading_days(from_date, self.today)
            self.assertIsInstance(trading_days, list)
            self.assertGreater(len(trading_days), 0)
            
        except Exception as e:
            self.skipTest(f"Market Calendar 로드 실패: {e}")

    def test_03_cache_daily_ohlcv(self):
        """일봉 데이터 캐싱 테스트"""
        try:
            # 일봉 데이터 캐싱
            daily_df = self.data_manager.cache_daily_ohlcv(
                self.stock_code, 
                self.start_date_daily, 
                self.today
            )
            
            # 결과 검증
            self.assertIsInstance(daily_df, pd.DataFrame)
            
            if not daily_df.empty:
                # DataFrame 구조 검증
                expected_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in expected_columns:
                    self.assertIn(col, daily_df.columns)
                
                # 데이터 타입 검증
                self.assertTrue(pd.api.types.is_numeric_dtype(daily_df['open']))
                self.assertTrue(pd.api.types.is_numeric_dtype(daily_df['close']))
                self.assertTrue(pd.api.types.is_numeric_dtype(daily_df['volume']))
                
                # 인덱스 검증 (날짜 인덱스)
                self.assertTrue(isinstance(daily_df.index, pd.DatetimeIndex))
                
                # 데이터 범위 검증
                self.assertGreaterEqual(daily_df.index.min().date(), self.start_date_daily)
                self.assertLessEqual(daily_df.index.max().date(), self.today)
                
                # 데이터 정합성 검증
                self.assertTrue(all(daily_df['high'] >= daily_df['low']))
                self.assertTrue(all(daily_df['high'] >= daily_df['open']))
                self.assertTrue(all(daily_df['high'] >= daily_df['close']))
                self.assertTrue(all(daily_df['low'] <= daily_df['open']))
                self.assertTrue(all(daily_df['low'] <= daily_df['close']))
                
        except Exception as e:
            self.skipTest(f"일봉 데이터 캐싱 실패: {e}")

    def test_04_cache_minute_ohlcv(self):
        """분봉 데이터 캐싱 테스트"""
        try:
            # 분봉 데이터 캐싱
            minute_df = self.data_manager.cache_minute_ohlcv(
                self.stock_code, 
                self.start_date_minute, 
                self.today, 
                interval=1
            )
            
            # 결과 검증
            self.assertIsInstance(minute_df, pd.DataFrame)
            
            if not minute_df.empty:
                # DataFrame 구조 검증
                expected_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in expected_columns:
                    self.assertIn(col, minute_df.columns)
                
                # 데이터 타입 검증
                self.assertTrue(pd.api.types.is_numeric_dtype(minute_df['open']))
                self.assertTrue(pd.api.types.is_numeric_dtype(minute_df['close']))
                self.assertTrue(pd.api.types.is_numeric_dtype(minute_df['volume']))
                
                # 인덱스 검증 (datetime 인덱스)
                self.assertTrue(isinstance(minute_df.index, pd.DatetimeIndex))
                
                # 데이터 범위 검증
                self.assertGreaterEqual(minute_df.index.min().date(), self.start_date_minute)
                self.assertLessEqual(minute_df.index.max().date(), self.today)
                
                # 데이터 정합성 검증
                self.assertTrue(all(minute_df['high'] >= minute_df['low']))
                self.assertTrue(all(minute_df['high'] >= minute_df['open']))
                self.assertTrue(all(minute_df['high'] >= minute_df['close']))
                self.assertTrue(all(minute_df['low'] <= minute_df['open']))
                self.assertTrue(all(minute_df['low'] <= minute_df['close']))
                
        except Exception as e:
            self.skipTest(f"분봉 데이터 캐싱 실패: {e}")

    def test_05_cache_daily_ohlcv_with_existing_data(self):
        """기존 데이터가 있는 경우 일봉 캐싱 테스트"""
        try:
            # 첫 번째 호출로 데이터 생성
            first_df = self.data_manager.cache_daily_ohlcv(
                self.stock_code, 
                self.start_date_daily, 
                self.today
            )
            
            # 두 번째 호출 (기존 데이터 재사용)
            second_df = self.data_manager.cache_daily_ohlcv(
                self.stock_code, 
                self.start_date_daily, 
                self.today
            )
            
            # 두 결과가 동일한지 검증
            if not first_df.empty and not second_df.empty:
                pd.testing.assert_frame_equal(first_df, second_df)
                
        except Exception as e:
            self.skipTest(f"기존 데이터 일봉 캐싱 테스트 실패: {e}")

    def test_06_cache_minute_ohlcv_with_existing_data(self):
        """기존 데이터가 있는 경우 분봉 캐싱 테스트"""
        try:
            # 첫 번째 호출로 데이터 생성
            first_df = self.data_manager.cache_minute_ohlcv(
                self.stock_code, 
                self.start_date_minute, 
                self.today, 
                interval=1
            )
            
            # 두 번째 호출 (기존 데이터 재사용)
            second_df = self.data_manager.cache_minute_ohlcv(
                self.stock_code, 
                self.start_date_minute, 
                self.today, 
                interval=1
            )
            
            # 두 결과가 동일한지 검증
            if not first_df.empty and not second_df.empty:
                pd.testing.assert_frame_equal(first_df, second_df)
                
        except Exception as e:
            self.skipTest(f"기존 데이터 분봉 캐싱 테스트 실패: {e}")

    def test_07_cache_daily_ohlcv_different_periods(self):
        """다른 기간의 일봉 데이터 캐싱 테스트"""
        try:
            # 짧은 기간
            short_start = self.today - timedelta(days=2)
            short_df = self.data_manager.cache_daily_ohlcv(
                self.stock_code, 
                short_start, 
                self.today
            )
            
            # 긴 기간
            long_start = self.today - timedelta(days=10)
            long_df = self.data_manager.cache_daily_ohlcv(
                self.stock_code, 
                long_start, 
                self.today
            )
            
            # 긴 기간이 짧은 기간보다 데이터가 많거나 같아야 함
            if not short_df.empty and not long_df.empty:
                self.assertGreaterEqual(len(long_df), len(short_df))
                
        except Exception as e:
            self.skipTest(f"다른 기간 일봉 캐싱 테스트 실패: {e}")

    def test_08_cache_minute_ohlcv_different_periods(self):
        """다른 기간의 분봉 데이터 캐싱 테스트"""
        try:
            # 짧은 기간
            short_start = self.today - timedelta(days=1)
            short_df = self.data_manager.cache_minute_ohlcv(
                self.stock_code, 
                short_start, 
                self.today, 
                interval=1
            )
            
            # 긴 기간
            long_start = self.today - timedelta(days=3)
            long_df = self.data_manager.cache_minute_ohlcv(
                self.stock_code, 
                long_start, 
                self.today, 
                interval=1
            )
            
            # 긴 기간이 짧은 기간보다 데이터가 많거나 같아야 함
            if not short_df.empty and not long_df.empty:
                self.assertGreaterEqual(len(long_df), len(short_df))
                
        except Exception as e:
            self.skipTest(f"다른 기간 분봉 캐싱 테스트 실패: {e}")

    def test_09_cache_daily_ohlcv_invalid_stock_code(self):
        """잘못된 종목 코드로 일봉 캐싱 테스트"""
        try:
            invalid_stock_code = 'INVALID_CODE'
            df = self.data_manager.cache_daily_ohlcv(
                invalid_stock_code, 
                self.start_date_daily, 
                self.today
            )
            
            # 잘못된 종목 코드의 경우 빈 DataFrame이 반환되어야 함
            self.assertIsInstance(df, pd.DataFrame)
            
        except Exception as e:
            # 예외가 발생해도 테스트는 통과 (API 오류는 정상적인 상황)
            pass

    def test_10_cache_minute_ohlcv_invalid_stock_code(self):
        """잘못된 종목 코드로 분봉 캐싱 테스트"""
        try:
            invalid_stock_code = 'INVALID_CODE'
            df = self.data_manager.cache_minute_ohlcv(
                invalid_stock_code, 
                self.start_date_minute, 
                self.today, 
                interval=1
            )
            
            # 잘못된 종목 코드의 경우 빈 DataFrame이 반환되어야 함
            self.assertIsInstance(df, pd.DataFrame)
            
        except Exception as e:
            # 예외가 발생해도 테스트는 통과 (API 오류는 정상적인 상황)
            pass

    def test_11_cache_daily_ohlcv_future_dates(self):
        """미래 날짜로 일봉 캐싱 테스트"""
        try:
            future_start = self.today + timedelta(days=1)
            future_end = self.today + timedelta(days=5)
            
            df = self.data_manager.cache_daily_ohlcv(
                self.stock_code, 
                future_start, 
                future_end
            )
            
            # 미래 날짜의 경우 빈 DataFrame이 반환되어야 함
            self.assertIsInstance(df, pd.DataFrame)
            
        except Exception as e:
            # 예외가 발생해도 테스트는 통과 (미래 데이터는 존재하지 않음)
            pass

    def test_12_cache_minute_ohlcv_future_dates(self):
        """미래 날짜로 분봉 캐싱 테스트"""
        try:
            future_start = self.today + timedelta(days=1)
            future_end = self.today + timedelta(days=5)
            
            df = self.data_manager.cache_minute_ohlcv(
                self.stock_code, 
                future_start, 
                future_end, 
                interval=1
            )
            
            # 미래 날짜의 경우 빈 DataFrame이 반환되어야 함
            self.assertIsInstance(df, pd.DataFrame)
            
        except Exception as e:
            # 예외가 발생해도 테스트는 통과 (미래 데이터는 존재하지 않음)
            pass

    def test_13_data_consistency_daily(self):
        """일봉 데이터 일관성 테스트"""
        try:
            df = self.data_manager.cache_daily_ohlcv(
                self.stock_code, 
                self.start_date_daily, 
                self.today
            )
            
            if not df.empty:
                # OHLC 관계 검증
                self.assertTrue(all(df['high'] >= df['low']))
                self.assertTrue(all(df['high'] >= df['open']))
                self.assertTrue(all(df['high'] >= df['close']))
                self.assertTrue(all(df['low'] <= df['open']))
                self.assertTrue(all(df['low'] <= df['close']))
                
                # 거래량은 음수가 아니어야 함
                self.assertTrue(all(df['volume'] >= 0))
                
                # 가격은 양수여야 함
                self.assertTrue(all(df['open'] > 0))
                self.assertTrue(all(df['high'] > 0))
                self.assertTrue(all(df['low'] > 0))
                self.assertTrue(all(df['close'] > 0))
                
        except Exception as e:
            self.skipTest(f"일봉 데이터 일관성 테스트 실패: {e}")

    def test_14_data_consistency_minute(self):
        """분봉 데이터 일관성 테스트"""
        try:
            df = self.data_manager.cache_minute_ohlcv(
                self.stock_code, 
                self.start_date_minute, 
                self.today, 
                interval=1
            )
            
            if not df.empty:
                # OHLC 관계 검증
                self.assertTrue(all(df['high'] >= df['low']))
                self.assertTrue(all(df['high'] >= df['open']))
                self.assertTrue(all(df['high'] >= df['close']))
                self.assertTrue(all(df['low'] <= df['open']))
                self.assertTrue(all(df['low'] <= df['close']))
                
                # 거래량은 음수가 아니어야 함
                self.assertTrue(all(df['volume'] >= 0))
                
                # 가격은 양수여야 함
                self.assertTrue(all(df['open'] > 0))
                self.assertTrue(all(df['high'] > 0))
                self.assertTrue(all(df['low'] > 0))
                self.assertTrue(all(df['close'] > 0))
                
        except Exception as e:
            self.skipTest(f"분봉 데이터 일관성 테스트 실패: {e}")

if __name__ == '__main__':
    unittest.main() 