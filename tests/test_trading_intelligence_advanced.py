# tests/test_trading_intelligence_advanced.py
import unittest
import sys
import os
import logging
import time
import queue
from datetime import datetime, date, timedelta
from threading import Event, Lock
from typing import Optional, List, Dict, Any, Callable, Tuple
import pythoncom
# 프로젝트 루트 경로 추가 (assuming tests/test_file.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient, OrderType, OrderStatus
from trading.brokerage import Brokerage
from manager.trading_manager import TradingManager
from manager.db_manager import DBManager
from strategies.intelligent_minute import IntelligentMinute # 개발할 전략
from util.notifier import Notifier
from config import settings
# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 테스트 대상 종목 5개
TEST_STOCKS = {
    'BUY_A': 'A033340',  # 좋은사람들 (약 1,800원대)
    'BUY_B': 'A032820',  # 우리기술 (약 3,600원대)
    'SELL_C': 'A041020', # 폴라리스오피스 (약 5,800원대)
    'BUY_D': 'A117580',  # 대성에너지 (약 8,200원대)
    'SELL_E': 'A090710', # 휴림로봇 (약 2555원대)
}
TEST_ORDER_QUANTITY = 1

class TestIntelligentAdvanced(unittest.TestCase):
    
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

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시 한 번만 실행: 리소스 정리"""
        logger.info("--- 테스트 클래스 정리 시작 ---")
        if cls.api:
            cls.api.cleanup()
        pythoncom.CoUninitialize()
        logger.info("--- 테스트 클래스 정리 완료 ---")


    def test_multi_stock_concurrent_scenario(self):
        """다중/시차 신호 발생 시 동시 거래 처리 능력을 테스트합니다."""
        
        # 1. 사전 준비: 매도할 종목(C, E) 미리 1주씩 시장가 매수
        logger.info("--- [고급] 사전 준비: 매도 대상 종목 매수 ---")
        self._setup_initial_positions([TEST_STOCKS['SELL_C'], TEST_STOCKS['SELL_E']])
        
        # 2. T=0초: 초기 신호 3개 동시 주입
        logger.info("--- [고급] T=0s: 1차 신호(매수 2, 매도 1) 주입 ---")
        initial_signals = {
            TEST_STOCKS['BUY_A']: {'signal_type': 'buy', 'target_quantity': TEST_ORDER_QUANTITY},
            TEST_STOCKS['BUY_B']: {'signal_type': 'buy', 'target_quantity': TEST_ORDER_QUANTITY},
            TEST_STOCKS['SELL_C']: {'signal_type': 'sell', 'target_quantity': TEST_ORDER_QUANTITY},
        }
        self.strategy.update_signals(initial_signals)

        # 3. 가상 메인 루프
        start_time = time.time()
        added_secondary_signals = False
        all_codes = list(TEST_STOCKS.values())
        try:
            while True:
                pythoncom.PumpWaitingMessages()

                # T=30초: 2차 신호 2개 주입 (시차 발생)
                if not added_secondary_signals and (time.time() - start_time) > 30:
                    logger.info("--- [고급] T=30s: 2차 신호(매수 1, 매도 1) 주입 ---")
                    secondary_signals = {
                        TEST_STOCKS['BUY_D']: {'signal_type': 'buy', 'target_quantity': TEST_ORDER_QUANTITY},
                        TEST_STOCKS['SELL_E']: {'signal_type': 'sell', 'target_quantity': TEST_ORDER_QUANTITY},
                    }
                    self.strategy.update_signals(secondary_signals)
                    added_secondary_signals = True

                # 현재 진행 중인 모든 임무에 대해 로직 실행 (동시 처리)
                active_codes = list(self.strategy.trade_missions.keys())
                for code in active_codes:
                    self.strategy.run_minute_logic(datetime.now(), code)

                # 모든 임무 완료 시 종료
                if self._all_missions_completed(all_codes):
                    logger.info("--- [고급] 모든 거래 임무 완료 ---")
                    break
                
                if (time.time() - start_time) > 600: # 10분 타임아웃
                    self.fail("고급 시나리오 테스트 시간 초과")
                
                time.sleep(3)
        except Exception as e:
            # ✅ 어떤 에러가 왜 발생하는지 정확히 출력해줍니다.
            logger.error("테스트 루프 중 에러 발생!", exc_info=True)
            self.fail(f"테스트 실패: {e}")

        # 4. 최종 결과 검증
        logger.info("--- [고급] 최종 결과 검증 ---")
        self._assert_final_positions({
            TEST_STOCKS['BUY_A']: TEST_ORDER_QUANTITY,
            TEST_STOCKS['BUY_B']: TEST_ORDER_QUANTITY,
            TEST_STOCKS['BUY_D']: TEST_ORDER_QUANTITY,
            TEST_STOCKS['SELL_C']: 0,
            TEST_STOCKS['SELL_E']: 0,
        })

    def _setup_initial_positions(self, codes_to_buy: List[str]):
        """테스트를 위해 특정 종목을 미리 매수합니다."""
        for code in codes_to_buy:
            res = self.api.send_order(code, OrderType.BUY, TEST_ORDER_QUANTITY, 0, order_unit="03")
            # 체결 확인 로직 필요 (간략화)
        time.sleep(5) # 체결 대기

    def _all_missions_completed(self, all_codes: List[str]) -> bool:
        """모든 종목의 임무가 완료되었는지 확인합니다."""
        completed_count = 0
        for code in all_codes:
            mission = self.strategy.trade_missions.get(code)
            if mission and mission['status'] == 'COMPLETED':
                completed_count += 1
        return completed_count == len(all_codes)
    
    def _assert_final_positions(self, expected_positions: Dict[str, int]):
        """최종 보유 수량을 검증합니다."""
        for code, expected_qty in expected_positions.items():
            final_qty = self.broker.get_position_size(code)
            self.assertEqual(final_qty, expected_qty, f"[{code}] 최종 수량 불일치: 기대={expected_qty}, 실제={final_qty}")

if __name__ == '__main__':
    unittest.main()