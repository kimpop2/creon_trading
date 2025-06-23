import unittest
import sys
import os
import logging
from datetime import datetime, date, timedelta

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.data_manager import DataManager

class TestCreonAPIConnection(unittest.TestCase):
    """크레온 API 연결 및 기본 기능 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행"""
        cls.api = CreonAPIClient()
        cls.data_manager = DataManager()
        logging.info("테스트 환경 설정 완료")

    def setUp(self):
        """각 테스트 메서드 실행 전에 실행"""
        self.assertTrue(self.api.connected, "크레온 API가 연결되어 있지 않습니다.")
        self.test_stock_code = 'A005930'  # 삼성전자

    def test_account_info(self):
        """계좌 정보 조회 테스트"""
        # 1. 예수금 조회
        balance = self.api.get_account_balance()
        self.assertIsInstance(balance, dict)
        self.assertIn('cash', balance)
        self.assertIn('deposit', balance)
        logging.info(f"계좌 잔고 조회 결과: {balance}")

        # 2. 포트폴리오 포지션 조회
        positions = self.api.get_portfolio_positions()
        self.assertIsInstance(positions, list)
        logging.info(f"보유 종목 수: {len(positions)}")
        for pos in positions:
            self.assertIn('stock_code', pos)
            self.assertIn('size', pos)
            self.assertIn('avg_price', pos)

    def test_stock_info(self):
        """종목 정보 조회 테스트"""
        # 1. 종목명 조회
        stock_name = self.api.get_stock_name(self.test_stock_code)
        self.assertEqual(stock_name, "삼성전자")
        logging.info(f"종목명 조회 결과: {stock_name}")

        # 2. 현재가 조회
        current_price = self.api.get_current_price(self.test_stock_code)
        self.assertIsInstance(current_price, float)
        self.assertGreater(current_price, 0)
        logging.info(f"현재가 조회 결과: {current_price:,.0f}원")

        # 3. 재무 정보 조회
        #financial_data = self.api.get_latest_financial_data(self.test_stock_code)
        financial_data = self.api.get_latest_financial_data(['A005930', 'A005380'])
        self.assertFalse(financial_data.empty)
        self.assertIn('per', financial_data.columns)
        self.assertIn('pbr', financial_data.columns)
        logging.info(f"재무 정보 조회 결과: PER={financial_data['per'].iloc[0]:.2f}, PBR={financial_data['pbr'].iloc[0]:.2f}")

    def test_market_data(self):
        """시장 데이터 조회 테스트"""
        today = date.today()
        from_date = today - timedelta(days=10)
        to_date = today

        # 1. 일봉 데이터 조회
        daily_data = self.api.get_daily_ohlcv(self.test_stock_code, from_date.strftime('%Y%m%d'), to_date.strftime('%Y%m%d'))
        self.assertFalse(daily_data.empty)
        self.assertTrue(all(col in daily_data.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        logging.info(f"일봉 데이터 조회 결과: {len(daily_data)}개")

        # 2. 분봉 데이터 조회
        minute_data = self.api.get_minute_ohlcv(self.test_stock_code, from_date.strftime('%Y%m%d'), to_date.strftime('%Y%m%d'))
        self.assertFalse(minute_data.empty)
        self.assertTrue(all(col in minute_data.columns for col in ['open', 'high', 'low', 'close', 'volume']))
        logging.info(f"분봉 데이터 조회 결과: {len(minute_data)}개")

        # 3. 거래일 조회
        trading_days = self.api.get_all_trading_days_from_api(from_date, to_date)
        self.assertIsInstance(trading_days, list)
        self.assertGreater(len(trading_days), 0)
        logging.info(f"거래일 조회 결과: {len(trading_days)}일")

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시 한 번만 실행"""
        if hasattr(cls, 'data_manager'):
            cls.data_manager.close()
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