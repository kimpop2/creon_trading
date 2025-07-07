import unittest
from unittest.mock import MagicMock, patch # 'patch'도 임포트되어 있는지 확인
import sys
import os
from datetime import datetime, date, timedelta
import pandas as pd
import logging

# 로거 설정 (테스트 시에도 로그를 볼 수 있도록)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# sys.path에 프로젝트 루트 추가 (manager 및 api 임포트를 위함)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 실제 TradingManager, DBManager, CreonAPIClient 임포트
# !!! 중요: 이 부분은 실제 Creon API 및 DB 연결 설정에 따라 달라집니다.
# 'config.settings'에서 실제 DB_HOST, DB_PORT 등을 로드해야 합니다.
# CreonAPIClient는 실제 CreonPlus API에 연결되어 있어야 합니다.
try:
    from manager.db_manager import DBManager
    from api.creon_api import CreonAPIClient # 실제 CreonAPIClient
    from manager.trading_manager import TradingManager
    logger.info("모듈 임포트 성공: DBManager, CreonAPIClient, TradingManager")
except ImportError as e:
    logger.error(f"모듈 임포트 실패: {e}. 'python_modules' 디렉토리와 경로 설정을 확인하세요.")
    sys.exit(1) # 테스트 실행 중단

class TestTradingManager(unittest.TestCase):
    """
    TradingManager 클래스의 메서드를 테스트하는 유닛 테스트 클래스.
    실제 Creon API 및 DB 연결을 가정합니다.
    """
    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger(__name__)
        cls.api_client = CreonAPIClient()
        if not cls.api_client._check_creon_status(): # Creon API 연결 시도
            logger.error("Creon API 연결 실패. 테스트를 계속할 수 없습니다. CreonPlus 실행 상태 및 설정 확인 필요.")
            # 실제 테스트 환경에서는 sys.exit(1) 또는 테스트 스킵
            raise ConnectionError("Creon API connection failed in setUpClass.")
        else:
            logger.info("Creon API 연결 성공.")
            # Creon API 연결 성공 후 5초 대기 (안정화를 위함)
            # import time
            # time.sleep(5)

        # 실제 DBManager 인스턴스 생성 및 연결 확인
        cls.db_manager = DBManager()
        if not cls.db_manager.get_db_connection():
            logger.error("DB 연결 실패. 테스트를 계속할 수 없습니다. DB 서버 상태 및 설정 확인 필요.")
            raise ConnectionError("Database connection failed in setUpClass.")
        else:
            logger.info("DB 연결 성공.")
        try:
            # TradingManager 인스턴스를 생성할 때, 위에서 생성한 Mock 객체들을 주입합니다.
            cls.trading_manager = TradingManager(cls.api_client, cls.db_manager)
            cls.test_stock_code = "A005930"
            cls.test_stock_name = "삼성전자"
            cls.logger.info("TestTradingManager: setUpClass - Creon API 및 DB 연결 시도 (Mocking)")
        except ImportError:
            cls.logger.error("TradingManager, DBManager, CreonAPIClient 클래스 중 하나를 임포트할 수 없습니다. 경로를 확인해주세요.")
            raise

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 전체를 위한 정리 (마지막 1회 실행)"""
        logger.info("TestTradingManager: tearDownClass - DB 및 Creon API 연결 해제 시도")
        cls.db_manager.close()
        logger.info("DB 연결 해제 완료.")

    def setUp(self):
        """각 테스트 메서드 실행 전 호출"""
        # 개별 테스트를 위한 초기화가 필요한 경우 여기에 추가
        pass

    def tearDown(self):
        """각 테스트 메서드 실행 후 호출"""
        # 개별 테스트 후 정리 작업이 필요한 경우 여기에 추가
        pass

    def test_01_get_stock_name(self):
        """종목 코드로 종목명 조회 테스트"""
        logger.info(f"test_get_stock_name: {self.test_stock_code}")
        stock_name = self.trading_manager.get_stock_name(self.test_stock_code)
        self.assertIsNotNone(stock_name)
        self.assertIsInstance(stock_name, str)
        self.assertEqual(stock_name, self.test_stock_name) # 캐시 또는 DB에서 올바른 종목명 반환 확인
        logger.info(f"종목명 조회 결과: {stock_name}")

    # def test_fetch_daily_ohlcv(self):
    #     """일봉 OHLCV 데이터 조회 테스트 (DB 및 API 연동)"""
    #     logger.info(f"test_fetch_daily_ohlcv: {self.test_stock_code}")
    #     to_date = date.today()
    #     from_date = to_date - timedelta(days=30) # 최근 30일 데이터
    #     df = self.trading_manager.fetch_daily_ohlcv(self.test_stock_code, from_date, to_date)
    #     self.assertIsInstance(df, pd.DataFrame)
    #     self.assertFalse(df.empty) # 데이터가 비어있지 않아야 함
    #     self.assertIn('close', df.columns) # OHLCV 컬럼 확인
    #     logger.info(f"일봉 데이터 {len(df)}건 조회 완료.")
    #     # print(df.head()) # 데이터 확인용

    # def test_fetch_minute_ohlcv(self):
    #     """분봉 OHLCV 데이터 조회 테스트 (DB 및 API 연동)"""
    #     logger.info(f"test_fetch_minute_ohlcv: {self.test_stock_code}")
    #     # 테스트 실행 시점에 따라 분봉 데이터가 없을 수 있으므로, 조회 결과에 대한 유연한 검증 필요
    #     to_datetime = datetime.now()
    #     from_datetime = to_datetime - timedelta(minutes=60) # 최근 60분 데이터
    #     df = self.trading_manager.fetch_minute_ohlcv(self.test_stock_code, from_datetime, to_datetime)
    #     self.assertIsInstance(df, pd.DataFrame)
    #     # 분봉 데이터는 장 운영 시간에만 조회되므로, 테스트 실행 시간에 따라 empty일 수 있음
    #     if df.empty:
    #         logger.warning(f"장 운영 시간이 아니거나 최근 분봉 데이터가 없어 {self.test_stock_code}의 분봉 데이터가 비어있습니다.")
    #     else:
    #         self.assertIn('close', df.columns)
    #         logger.info(f"분봉 데이터 {len(df)}건 조회 완료.")
    #         # print(df.head()) # 데이터 확인용

    def test_04_get_account_balance(self):
        """계좌 잔고 조회 테스트"""
        logger.info("test_get_account_balance: 계좌 잔고 조회")
        balance = self.trading_manager.get_account_balance()
        self.assertIsInstance(balance, dict)
        self.assertIn('cash_balance', balance) # 'cash_balance' 키가 있는지 확인
        self.assertGreaterEqual(balance.get('cash_balance', 0), 0) # 현금 잔고가 0 이상인지 확인
        logger.info(f"계좌 잔고: {balance}")

    def test_05_get_open_positions(self):
        """현재 보유 종목 조회 테스트 (DB 동기화 포함)"""
        logger.info("test_get_open_positions: 현재 보유 종목 조회")
        positions = self.trading_manager.get_open_positions()
        self.assertIsInstance(positions, list)
        logger.info(f"현재 보유 종목 {len(positions)}건 조회 완료.")
        for position in positions:
            logger.info(f"종목코드 {position['stock_code']}, 종목명 {position['stock_name']}, 잔고수량 {position['quantity']}, 매도가능수량 {position['sell_avail_qty']}.")

    def test_06_get_unfilled_orders(self):
        """미체결 주문 내역 조회 테스트"""
        logger.info("test_get_unfilled_orders: 미체결 주문 조회")
        unfilled_orders = self.trading_manager.get_unfilled_orders()
        self.assertIsInstance(unfilled_orders, list)
        logger.info(f"미체결 주문 {len(unfilled_orders)}건 조회 완료.")
        # if unfilled_orders:
        #     print(f"첫 번째 미체결 주문: {unfilled_orders[0]}")

    def test_07_save_and_load_daily_signals(self):
        """일일 매매 신호 저장 및 로드 테스트"""
        logger.info("test_save_and_load_daily_signals: 매매 신호 테스트")
        signal_date = date.today()
        test_signal_data = {
            'stock_code': self.test_stock_code,
            'stock_name': self.test_stock_name,
            'signal_date': signal_date,
            'signal_type': 'BUY',
            'target_price': 100000.0,
            'target_quantity': 1,
            'strategy_name': 'TestStrategy',
            'is_executed': False
        }
        
        # 기존 신호 클리어 (테스트 환경 초기화)
        self.db_manager.clear_daily_signals(signal_date)
        
        # 신호 저장
        save_success = self.trading_manager.save_daily_signals(test_signal_data)
        self.assertTrue(save_success)
        logger.info("매매 신호 저장 성공.")

        # 신호 로드
        loaded_signals = self.trading_manager.load_daily_signals(signal_date)
        self.assertIn(self.test_stock_code, loaded_signals)
        loaded_signal = loaded_signals[self.test_stock_code]
        self.assertEqual(loaded_signal['signal_type'], 'BUY')
        self.assertEqual(loaded_signal['target_price'], 100000.0)
        self.assertFalse(loaded_signal['is_executed'])
        logger.info(f"저장된 신호 로드 성공: {loaded_signal}")

        # 신호 상태 업데이트
        signal_id = loaded_signal['signal_id']
        update_success = self.db_manager.update_daily_signal_status(signal_id, True, "ORDER_ID_TEST_123")
        self.assertTrue(update_success)
        logger.info(f"신호 ID {signal_id} 상태 업데이트 성공.")

        # 업데이트된 신호 다시 로드하여 확인
        reloaded_signals = self.trading_manager.load_daily_signals(signal_date)
        reloaded_signal = reloaded_signals[self.test_stock_code]
        self.assertTrue(reloaded_signal['is_executed'])
        self.assertEqual(reloaded_signal['executed_order_id'], "ORDER_ID_TEST_123")
        logger.info(f"업데이트된 신호 재로드 확인: {reloaded_signal}")

        # 테스트 후 정리
        self.db_manager.clear_daily_signals(signal_date)
        logger.info("테스트 신호 정리 완료.")

    def test_08_load_daily_data_for_strategy(self):
        """전략 분석을 위한 일봉 데이터 로드 테스트"""
        logger.info(f"test_load_daily_data_for_strategy: {self.test_stock_code}")
        target_date = date.today()
        lookback_days = 60
        stock_codes = [self.test_stock_code] # 여러 종목도 테스트 가능
        daily_data = self.trading_manager.load_daily_data_for_strategy(stock_codes, target_date, lookback_days)
        self.assertIsInstance(daily_data, dict)
        self.assertIn(self.test_stock_code, daily_data)
        self.assertIsInstance(daily_data[self.test_stock_code], pd.DataFrame)
        self.assertFalse(daily_data[self.test_stock_code].empty)
        logger.info(f"전략용 일봉 데이터 로드 성공. {self.test_stock_code} 데이터 수: {len(daily_data[self.test_stock_code])}")

    def test_09_get_universe_stocks(self):
        """유니버스 종목 조회 테스트"""
        logger.info("test_get_universe_stocks: 유니버스 종목 조회")
        end_date = date.today()
        start_date =  end_date - timedelta(days=10)
        # 이 테스트는 db_manager.fetch_daily_theme_stock에 실제 데이터가 있어야 유의미함
        # 더미 데이터나 테스트용 데이터를 DB에 미리 넣어두고 테스트하는 것이 좋습니다.
        universe_stocks = self.trading_manager.get_universe_stocks(start_date, end_date)
        self.assertIsInstance(universe_stocks, list)
        logger.info(f"유니버스 종목 {len(universe_stocks)}개 조회 완료.")
        # if universe_stocks:
        #     print(f"첫 번째 유니버스 종목: {universe_stocks[0]}")

    def test_20_get_current_prices(self):
        """현재 시장 가격 조회 테스트"""
        logger.info(f"test_get_current_prices: {self.test_stock_code}")
        prices = self.trading_manager.get_current_prices([self.test_stock_code])
        self.assertIsInstance(prices, dict)
        self.assertIn(self.test_stock_code, prices)
        self.assertIsInstance(prices[self.test_stock_code], int)
        self.assertGreater(prices[self.test_stock_code], 0) # 가격이 0보다 커야 함
        for stock_code in prices:
            stock_name = self.trading_manager.get_stock_name(stock_code)
            logger.info(f"({stock_code}) {stock_name} 현재 시장 가격: {prices[stock_code]}")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)