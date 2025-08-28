# tests/test_state_management.py

import unittest
import sys
import os
import logging
import time as pytime
from datetime import datetime, date, timedelta, time

# --- 프로젝트 루트 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 필요한 모듈 임포트 ---
from api.creon_api import CreonAPIClient, OrderType # <--- 여기에 OrderType 추가
from manager.db_manager import DBManager
from manager.trading_manager import TradingManager
from trading.brokerage import Brokerage
from strategies.strategy import DailyStrategy
from strategies.sma_daily import SMADaily  # 테스트에 사용할 실제 전략 클래스
from config.settings import INITIAL_CASH, STRATEGY_CONFIGS, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from util.notifier import Notifier

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestStateManagement(unittest.TestCase):
    """
    강화된 상태 관리 로직을 검증하는 통합 테스트 클래스.
    Mock 객체 없이 실제 Creon API와 DB를 사용하여 테스트합니다.
    """
    
    # --- 테스트 상수 정의 ---
    TEST_STOCK_CODE = 'A032820'  # 동양철관 (변동성이 있고 저렴한 주식으로 테스트)
    TEST_STRATEGY_NAME = 'SMADaily'
    TEST_MODEL_ID = 3 # 테스트용 가상 모델 ID

    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행: 모든 객체 초기화"""
        logger.info("--- 테스트 클래스 설정 시작 ---")
        cls.api = CreonAPIClient()
        cls.db = DBManager()
        cls.notifier = Notifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        cls.manager = TradingManager(cls.api, cls.db)
        cls.broker = Brokerage(cls.api, cls.manager, cls.notifier, initial_cash=INITIAL_CASH)
        
        cls.data_store = {'daily': {}, 'minute': {}} # 전략 객체 생성에 필요한 빈 데이터 저장소
        cls.strategy = SMADaily(cls.broker, cls.data_store)
        
        logger.info("--- 테스트 클래스 설정 완료 ---")

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시 한 번만 실행: 리소스 정리"""
        logger.info("--- 테스트 클래스 정리 시작 ---")
        cls.db.close()
        logger.info("--- 테스트 클래스 정리 완료 ---")

    def setUp(self):
        """각 테스트 메서드 실행 전, 테스트 환경 초기화"""
        logger.info(f"\n--- {self._testMethodName} 테스트 시작 ---")
        self._cleanup_position()

    def tearDown(self):
        """각 테스트 메서드 실행 후, 테스트 환경 정리"""
        self._cleanup_position()
        logger.info(f"--- {self._testMethodName} 테스트 종료 ---\n")

    def _cleanup_position(self):
        """테스트용 종목 잔고 및 DB 기록을 깨끗하게 정리하는 헬퍼 함수"""
        logger.info(f"테스트 환경 정리 시작: {self.TEST_STOCK_CODE}")
        # DB 정리
        self.db.execute_sql("DELETE FROM current_positions WHERE stock_code = %s", (self.TEST_STOCK_CODE,))
        self.db.execute_sql("DELETE FROM trading_trade WHERE stock_code = %s AND model_id = %s", (self.TEST_STOCK_CODE, self.TEST_MODEL_ID))
        
        # 실제 계좌 잔고 정리 (보유 시 전량 시장가 매도)
        positions = self.api.get_portfolio_positions()
        for pos in positions:
            if pos['stock_code'] == self.TEST_STOCK_CODE and pos['quantity'] > 0:
                logger.info(f"계좌에 남은 테스트 종목 {pos['quantity']}주를 시장가 매도합니다.")
                self.api.send_order(self.TEST_STOCK_CODE, order_type=OrderType.SELL, quantity=pos['quantity'], price=0, order_unit="03")
                pytime.sleep(5) # 체결 대기
                break

    def test_sync_and_strategy_logic(self):
        """
        [핵심 테스트] sync_account_status가 상태를 정확히 복원하고,
        이를 바탕으로 전략 로직이 올바르게 동작하는지 검증합니다.
        """
        # 1. (준비) 테스트용 변수 설정
        entry_date = date.today() - timedelta(days=5)
        entry_price = 1000
        
        # 2. (모의 거래 기록 생성) DB의 trading_trade 테이블에 가짜 매수 기록 삽입
        logger.info(f"'{self.TEST_STRATEGY_NAME}' 전략이 {entry_date}에 매수했다는 모의 거래 기록을 DB에 생성합니다.")
        mock_trade_data = {
            'model_id': self.TEST_MODEL_ID, 'trade_date': entry_date, 'strategy_name': self.TEST_STRATEGY_NAME,
            'stock_code': self.TEST_STOCK_CODE, 'trade_type': 'BUY', 'trade_price': entry_price,
            'trade_quantity': 1, 'trade_datetime': datetime.combine(entry_date, time(10, 0))
        }
        self.assertTrue(self.db.save_trading_trade(mock_trade_data))

        # 3. (실제 포지션 생성) API를 통해 테스트 계좌에서 실제로 1주 매수
        logger.info(f"실제 계좌에서 테스트 종목({self.TEST_STOCK_CODE}) 1주를 시장가 매수합니다.")
        buy_result = self.api.send_order(self.TEST_STOCK_CODE, order_type=OrderType.BUY, quantity=1, price=0, order_unit="03")
        self.assertEqual(buy_result['status'], 'success', "테스트용 종목의 실제 매수 주문에 실패했습니다.")
        
        # 체결될 때까지 최대 20초 대기
        is_filled = False
        for _ in range(10):
            pytime.sleep(2)
            positions = self.api.get_portfolio_positions()
            if any(p['stock_code'] == self.TEST_STOCK_CODE and p['quantity'] >= 1 for p in positions):
                is_filled = True
                logger.info("테스트 종목 매수 체결 확인 완료.")
                break
        self.assertTrue(is_filled, "20초 내에 테스트 종목 매수 체결을 확인할 수 없습니다.")

        # 4. (실행) sync_account_status 메서드 호출
        logger.info(" Brokerage.sync_account_status() 메서드를 실행하여 상태 동기화를 수행합니다.")
        self.broker.sync_account_status()

        # 5. (1차 검증: 동기화) brokerage.positions에 상태가 정확히 복원되었는지 확인
        logger.info("동기화된 포지션 상태를 검증합니다...")
        synced_pos = self.broker.get_current_positions().get(self.TEST_STOCK_CODE)
        
        self.assertIsNotNone(synced_pos, "동기화 후 포지션 정보가 존재하지 않습니다.")
        logger.info(f"동기화된 정보: {synced_pos}")
        
        # 검증 5-1: strategy_name이 DB 이력에서 정확히 복원되었는가?
        self.assertEqual(synced_pos.get('strategy_name'), self.TEST_STRATEGY_NAME, "strategy_name이 정확하게 복원되지 않았습니다.")
        # 검증 5-2: entry_date가 DB 이력에서 정확히 복원되었는가?
        self.assertEqual(synced_pos.get('entry_date'), entry_date, "entry_date가 정확하게 복원되지 않았습니다.")
        # 검증 5-3: 평가수익률 필드가 유효한 값으로 채워졌는가? (API에서 직접 가져옴)
        self.assertIsNotNone(synced_pos.get('eval_return_rate'), "평가수익률(eval_return_rate)이 설정되지 않았습니다.")
        # 검증 5-4: highest_price가 초기화되었는가?
        self.assertGreater(synced_pos.get('highest_price', 0), 0, "highest_price가 초기화되지 않았습니다.")

        logger.info("✅ 1차 검증(동기화) 성공!")

        # 6. (2차 검증: 전략 로직) max_position_count가 전략별로 계산되는지 확인
        logger.info("전략별 max_position_count 계산 로직을 검증합니다...")
        self.strategy.strategy_params['max_position_count'] = 5 # 테스트를 위해 최대 보유 종목 수 5로 설정
        
        # _generate_signals 내부 로직을 그대로 가져와 검증
        all_positions = self.broker.get_current_positions()
        my_positions = {code: pos for code, pos in all_positions.items() if pos.get('strategy_name') == self.strategy.strategy_name}
        num_current_positions_for_this_strategy = len(my_positions)
        slots_available = self.strategy.strategy_params['max_position_count'] - num_current_positions_for_this_strategy
        
        logger.info(f"전략({self.strategy.strategy_name}) 보유 수: {num_current_positions_for_this_strategy}, 최대 보유 수: 5")
        self.assertEqual(num_current_positions_for_this_strategy, 1, "전략별 보유 종목 수가 1이 아닙니다.")
        self.assertEqual(slots_available, 4, "매수 가능 슬롯이 4가 아닙니다.")
        
        logger.info("✅ 2차 검증(전략 로직) 성공!")


if __name__ == '__main__':
    unittest.main()