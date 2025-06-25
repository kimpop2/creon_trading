"""
TraderManager 클래스 단위 테스트
"""

import unittest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import sys
import os
import time
import logging

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.trader_manager import TraderManager

class TestTraderManager(unittest.TestCase):
    """데이터 관리 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행"""
        cls.api = CreonAPIClient()
        cls.trader_manager = TraderManager()
        cls.test_stock_code = 'A005930'  # 삼성전자
        logging.info("테스트 환경 설정 완료")

    def setUp(self):
        """각 테스트 메서드 실행 전에 실행"""
        self.assertTrue(self.api.connected, "크레온 API가 연결되어 있지 않습니다.")

    def test_1_market_calendar(self):
        """거래일 캘린더 테스트"""
        today = date.today()
        from_date = today - timedelta(days=30)
        to_date = today + timedelta(days=30)
        
        # 1. 거래일 캘린더 업데이트
        success = self.trader_manager.update_market_calendar(from_date, to_date)
        self.assertTrue(success, "거래일 캘린더 업데이트 실패")
        
        # 2. 거래일 조회
        trading_days = self.trader_manager.db_manager.get_all_trading_days(from_date, to_date)
        self.assertGreater(len(trading_days), 0, "거래일 데이터가 없습니다")
        logging.info(f"거래일 데이터 {len(trading_days)}개 조회됨")

    def test_2_stock_info(self):
        """종목 정보 테스트"""
        # 1. 종목 정보 업데이트
        success = self.trader_manager.update_all_stock_info()
        self.assertTrue(success, "종목 정보 업데이트 실패")
        
        # 2. 종목 정보 조회
        stock_info_map = self.trader_manager.get_stock_info_map()
        self.assertGreater(len(stock_info_map), 0, "종목 정보가 없습니다")
        self.assertIn(self.test_stock_code, stock_info_map)
        logging.info(f"종목 정보 {len(stock_info_map)}개 조회됨")
        
        # 3. 재무 정보 업데이트
        self.trader_manager.update_financial_data_for_stock_info(self.test_stock_code)

    def test_3_daily_data(self):
        """일봉 데이터 테스트"""
        today = date.today()
        from_date = today - timedelta(days=30)
        to_date = today
        
        # 1. 일봉 데이터 캐싱
        daily_df = self.trader_manager.cache_daily_ohlcv(self.test_stock_code, from_date, to_date)
        self.assertFalse(daily_df.empty, "일봉 데이터가 없습니다")
        self.assertTrue(all(col in daily_df.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        logging.info(f"일봉 데이터 {len(daily_df)}개 조회됨")
        
        # 2. 지표 계산 테스트
        daily_strategy_params = {
            'strategy_name': 'SMADaily',
            'short_sma_period': 5,
            'long_sma_period': 20,
            'volume_ma_period': 20
        }
        
        daily_df_with_indicators = self.trader_manager.get_daily_ohlcv_with_indicators(
            self.test_stock_code,
            from_date,
            to_date,
            daily_strategy_params
        )
        
        self.assertFalse(daily_df_with_indicators.empty, "지표가 계산된 일봉 데이터가 없습니다")
        self.assertTrue(all(col in daily_df_with_indicators.columns for col in [
            f'SMA_{daily_strategy_params["short_sma_period"]}',
            f'SMA_{daily_strategy_params["long_sma_period"]}',
            f'Volume_MA_{daily_strategy_params["volume_ma_period"]}'
        ]))
        logging.info("일봉 데이터 지표 계산 완료")

    def test_4_minute_data(self):
        """분봉 데이터 테스트"""
        today = date.today()
        from_date = today - timedelta(days=1)
        to_date = today
        
        # 1. 분봉 데이터 캐싱
        minute_df = self.trader_manager.cache_minute_ohlcv(self.test_stock_code, from_date, to_date)
        self.assertFalse(minute_df.empty, "분봉 데이터가 없습니다")
        self.assertTrue(all(col in minute_df.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        logging.info(f"분봉 데이터 {len(minute_df)}개 조회됨")
        
        # 2. 지표 계산 테스트
        minute_strategy_params = {
            'strategy_name': 'RSIMinute',
            'minute_rsi_period': 14
        }
        
        minute_df_with_indicators = self.trader_manager.get_minute_ohlcv_with_indicators(
            self.test_stock_code,
            today,
            minute_strategy_params
        )
        
        self.assertFalse(minute_df_with_indicators.empty, "지표가 계산된 분봉 데이터가 없습니다")
        self.assertTrue('RSI' in minute_df_with_indicators.columns)
        logging.info("분봉 데이터 지표 계산 완료")

    def test_5_realtime_data(self):
        """실시간 데이터 테스트"""
        # 1. 현재가 조회
        current_price = self.trader_manager.get_realtime_price(self.test_stock_code)
        self.assertIsNotNone(current_price, "현재가 조회 실패")
        self.assertGreater(current_price, 0)
        logging.info(f"현재가: {current_price:,.0f}원")
        
        # 2. 실시간 분봉 데이터 조회
        minute_data = self.trader_manager.get_realtime_minute_data(self.test_stock_code)
        self.assertIsNotNone(minute_data, "실시간 분봉 데이터 조회 실패")
        self.assertFalse(minute_data.empty)
        self.assertTrue(all(col in minute_data.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        logging.info("실시간 분봉 데이터 조회 완료")

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시 한 번만 실행"""
        if hasattr(cls, 'trader_manager'):
            cls.trader_manager.close()
        logging.info("테스트 환경 정리 완료")

if __name__ == '__main__':
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    unittest.main()