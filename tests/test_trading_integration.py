# tests/test_trading_integration.py

import unittest
import sys
import os
import logging
import time as pytime
from datetime import datetime
import pythoncom

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 필요한 모듈 임포트 ---
from api.creon_api import CreonAPIClient, OrderType
from manager.db_manager import DBManager
from scripts.execution.run_hmm_trading import Trading
from util.notifier import Notifier
from strategies.sma_daily import SMADaily
from strategies.target_price_minute import TargetPriceMinute
from config.settings import INITIAL_CASH, SMA_DAILY_PARAMS, COMMON_PARAMS, STOP_LOSS_PARAMS

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- 테스트 상수 ---
TEST_STOCK_CODE = 'A032820'  # 우리기술
TEST_ORDER_QUANTITY = 1     # 테스트용 최소 수량

class TestTradingIntegration(unittest.TestCase):
    """
    Trading 시스템의 전체 흐름을 검증하는 통합 테스트 클래스.
    - 실제 시스템처럼 Trading, Brokerage, Strategy를 모두 초기화합니다.
    - 전략에 신호를 주입하고, 비동기 콜백을 통해 Brokerage의 상태가
      올바르게 업데이트되는지 확인합니다.
    - `_wait_for...` 함수를 사용하지 않고, 실제 콜백 처리를 검증합니다.
    """
    cls_api: CreonAPIClient = None

    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행"""
        logger.info("--- 통합 테스트 클래스 설정 시작 ---")
        pythoncom.CoInitialize()
        try:
            cls.cls_api = CreonAPIClient()
            if not cls.cls_api.is_connected():
                raise ConnectionError("크레온 PLUS에 연결되어 있지 않습니다. HTS를 실행하고 로그인해주세요.")
        except Exception as e:
            logger.error(f"CreonAPIClient 초기화 실패: {e}", exc_info=True)
            pythoncom.CoUninitialize()
            raise
        logger.info("--- 통합 테스트 클래스 설정 완료 ---")

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시 한 번만 실행"""
        logger.info("--- 통합 테스트 클래스 정리 시작 ---")
        if cls.cls_api:
            cls.cls_api.cleanup()
        pythoncom.CoUninitialize()
        logger.info("--- 통합 테스트 클래스 정리 완료 ---")

    def setUp(self):
        """각 테스트 메서드 실행 전에 실행"""
        logger.info(f"\n--- {self._testMethodName} 테스트 준비 시작 ---")
        self.assertIsNotNone(self.cls_api, "CreonAPIClient가 초기화되지 않았습니다.")
        
        # 1. 테스트에 필요한 전체 시스템 구성
        db_manager = DBManager()
        notifier = Notifier() # 실제 알림을 보내지 않도록 mock notifier 사용 가능
        
        self.trading_system = Trading(
            api_client=self.cls_api,
            db_manager=db_manager,
            notifier=notifier,
            initial_cash=INITIAL_CASH
        )
        
        # 2. 전략 설정
        daily_strategy = SMADaily(broker=self.trading_system.broker, data_store=self.trading_system.data_store, strategy_params=SMA_DAILY_PARAMS)
        minute_strategy = TargetPriceMinute(broker=self.trading_system.broker, data_store=self.trading_system.data_store, strategy_params=COMMON_PARAMS)
        self.trading_system.set_strategies(daily_strategy, minute_strategy)
        self.trading_system.set_broker_stop_loss_params(STOP_LOSS_PARAMS)
        
        # 3. 테스트 전, 기존 포지션이 있다면 정리 (테스트 독립성 보장)
        self._cleanup_position(TEST_STOCK_CODE)

        # 4. 실제 거래처럼 데이터 준비 단계 실행
        self.assertTrue(self.trading_system.prepare_for_system(), "거래 시스템 준비에 실패했습니다.")
        logger.info(f"--- {self._testMethodName} 테스트 준비 완료 ---")

    def tearDown(self):
        """각 테스트 메서드 실행 후에 실행"""
        logger.info(f"--- {self._testMethodName} 테스트 정리 시작 ---")
        # 테스트 중 생성된 포지션 정리
        self._cleanup_position(TEST_STOCK_CODE)
        self.trading_system.cleanup()
        logger.info(f"--- {self._testMethodName} 테스트 정리 완료 ---\n")

    def _cleanup_position(self, stock_code):
        """테스트 시작 전후에 특정 종목의 포지션을 정리하는 헬퍼 함수"""
        broker = self.trading_system.broker
        initial_quantity = broker.get_position_size(stock_code)
        
        if initial_quantity > 0:
            logger.warning(f"테스트 시작/종료 전 [{stock_code}]의 기존 보유 수량 {initial_quantity}주를 정리합니다.")
            # 시장가로 즉시 매도
            broker.execute_order(stock_code, 'sell', 0, initial_quantity, order_time=datetime.now())
            # 정리가 완료될 때까지 잠시 대기 (테스트 환경에서는 안정성을 위해 짧은 sleep 사용)
            pytime.sleep(5) 
            # COM 메시지 처리
            for _ in range(5):
                pythoncom.PumpWaitingMessages()
                pytime.sleep(1)

    def test_buy_scenario_integration(self):
        """
        매수 신호 주입 -> 시스템 실행 -> 콜백 처리 -> 최종 포지션 상태 검증의 전체 흐름을 테스트합니다.
        """
        # 1. Arrange (테스트 조건 설정)
        logger.info("1. 테스트 조건 설정: 분봉 전략에 '매수' 신호를 직접 주입합니다.")
        
        # 현재가를 조회하여 목표가 설정
        price_info = self.cls_api.get_current_price_and_quotes(TEST_STOCK_CODE)
        self.assertIsNotNone(price_info, f"[{TEST_STOCK_CODE}]의 가격 정보를 가져올 수 없습니다.")
        current_price = price_info['current_price']
        target_price = self.cls_api.round_to_tick(current_price * 0.99) # 현재가보다 1% 낮은 가격으로 목표가 설정

        # 분봉 전략의 signals 딕셔너리에 매수 신호 주입
        self.trading_system.minute_strategy.signals[TEST_STOCK_CODE] = {
            'signal_type': 'buy',
            'target_price': target_price,
            'target_quantity': TEST_ORDER_QUANTITY,
        }
        
        # 2. Act (시스템 실행)
        logger.info(f"2. 시스템 실행: 20초간 테스트 루프를 실행하여 비동기 콜백 처리를 유도합니다. 목표가: {target_price:,.0f}원")
        
        start_time = pytime.time()
        end_time = start_time + 20  # 20초 동안 실행
        
        while pytime.time() < end_time:
            now = datetime.now()
            # 실제 trading.py의 루프처럼 분봉 전략 로직 실행
            self.trading_system.minute_strategy.run_minute_logic(now, TEST_STOCK_CODE)
            
            # COM 메시지 펌핑 (비동기 콜백이 처리되도록 함)
            pythoncom.PumpWaitingMessages()
            pytime.sleep(1) # 1초 간격으로 반복

        logger.info("20초간의 테스트 루프 실행 완료.")

        # 3. Assert (결과 검증)
        logger.info("3. 결과 검증: Brokerage의 최종 포지션 상태를 확인합니다.")
        
        # 콜백 처리가 완료된 후의 최종 보유 수량 확인
        # Brokerage의 상태는 handle_order_conclusion 콜백에 의해 비동기적으로 업데이트됨
        final_quantity = self.trading_system.broker.get_position_size(TEST_STOCK_CODE)
        
        self.assertEqual(final_quantity, TEST_ORDER_QUANTITY,
                         f"매수 주문 후 최종 수량이 예상({TEST_ORDER_QUANTITY})과 다릅니다: 실제 {final_quantity}")
        
        logger.info(f"✅ 검증 성공: [{TEST_STOCK_CODE}]의 최종 보유 수량이 {final_quantity}주로 정상 처리되었습니다.")


if __name__ == '__main__':
    unittest.main()