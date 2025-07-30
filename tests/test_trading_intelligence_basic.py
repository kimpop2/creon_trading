# tests/test_trading_intelligence_basic.py

import unittest
import sys
import os
import logging
import time
from datetime import datetime
import pythoncom

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 추가 (assuming tests/test_file.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from trading.brokerage import Brokerage
from manager.trading_manager import TradingManager
from manager.db_manager import DBManager
from strategies.intelligent_minute import IntelligentMinute # 개발할 전략
from util.notifier import Notifier
from config import settings

# --- 로깅 설정 ---
# ... (로깅 설정 코드는 기존과 동일) ...

TEST_STOCK_CODE = 'A090710' # 테스트 대상 종목 (휴림로봇)
TEST_ORDER_QUANTITY = 2 # 테스트용 수량 (매수 2주 -> 매도 2주)

class TestIntelligentBasic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # --- Creon API, Brokerage, Strategy 초기화 ---
        cls.api = CreonAPIClient()
        cls.db_manager = DBManager()
        cls.notifier = Notifier()
        # TradingManager 모의 객체 또는 실제 객체 전달
        cls.manager = TradingManager(cls.api, cls.db_manager) # TradingManager 인스턴스가 필요하면 여기에 추가
        cls.broker = Brokerage(cls.api, cls.manager, cls.notifier, settings.INITIAL_CASH)
        cls.strategy = IntelligentMinute(cls.broker, None, settings.INTELLIGENT_MINUTE_PARAMS)
        
        # API 콜백과 전략의 이벤트 핸들러 연결
        cls.api.set_conclusion_callback(cls.strategy.handle_conclusion)

    def test_single_stock_buy_and_sell_scenario(self):
        """단일 종목에 대한 매수 후 매도 임무를 순차적으로 테스트합니다."""
        
        # --- 1. 매수 임무 테스트 ---
        logger.info(f"--- [기본] {TEST_STOCK_CODE} 매수 임무 테스트 시작 ---")
        buy_signal = {TEST_STOCK_CODE: {'signal_type': 'buy', 'target_quantity': TEST_ORDER_QUANTITY}}
        self.strategy.update_signals(buy_signal)

        # 가상 메인 루프 (매수 임무 완료까지)
        mission_status = self._run_virtual_loop_until_completed(TEST_STOCK_CODE)
        self.assertEqual(mission_status, 'COMPLETED', "매수 임무가 정상적으로 완료되지 않았습니다.")
        
        # 검증: 실제 보유 수량 확인
        final_buy_position = self.broker.get_position_size(TEST_STOCK_CODE)
        self.assertEqual(final_buy_position, TEST_ORDER_QUANTITY, "매수 임무 후 보유 수량이 일치하지 않습니다.")
        logger.info(f"--- [기본] {TEST_STOCK_CODE} 매수 임무 성공 확인 ---")

        time.sleep(5) # 다음 시나리오를 위한 대기

        # --- 2. 매도 임무 테스트 ---
        logger.info(f"--- [기본] {TEST_STOCK_CODE} 매도 임무 테스트 시작 ---")
        sell_signal = {TEST_STOCK_CODE: {'signal_type': 'sell', 'target_quantity': TEST_ORDER_QUANTITY}}
        self.strategy.update_signals(sell_signal)

        # 가상 메인 루프 (매도 임무 완료까지)
        mission_status = self._run_virtual_loop_until_completed(TEST_STOCK_CODE)
        self.assertEqual(mission_status, 'COMPLETED', "매도 임무가 정상적으로 완료되지 않았습니다.")

        # 검증: 최종 보유 수량이 0인지 확인
        final_sell_position = self.broker.get_position_size(TEST_STOCK_CODE)
        self.assertEqual(final_sell_position, 0, "매도 임무 후 잔고가 0이 아닙니다.")
        logger.info(f"--- [기본] {TEST_STOCK_CODE} 매도 임무 성공 확인 ---")


    def _run_virtual_loop_until_completed(self, stock_code: str, timeout_seconds: int = 300) -> str:
        """특정 종목의 임무가 완료될 때까지 가상 메인 루프를 실행합니다."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            pythoncom.PumpWaitingMessages()

            # 전략의 심장 박동 역할
            self.strategy.run_minute_logic(datetime.now(), stock_code)

            mission = self.strategy.trade_missions.get(stock_code)
            if mission and mission['status'] == 'COMPLETED':
                return 'COMPLETED'
            
            time.sleep(3) # 실제 루프처럼 짧은 대기
        
        return self.strategy.trade_missions.get(stock_code, {}).get('status', 'TIMEOUT')
    
if __name__ == '__main__':
    unittest.main()