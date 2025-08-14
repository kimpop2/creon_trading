import unittest
import sys
import os
import logging
from datetime import datetime, date, timedelta
import time
from typing import Optional, List, Dict, Any, Callable, Tuple

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from api.creon_api2 import CreonAPIClient
from manager.trader_manager import TraderManager
from util.constants import *

class TestCreonAPIOrder(unittest.TestCase):
    """크레온 API 주문 실행 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행"""
        cls.api = CreonAPIClient()
        cls.trader_manager = TraderManager()
        cls.test_stock_code = 'A032820'  # 우리기술 A042670
        logging.info("테스트 환경 설정 완료")

    def setUp(self):
        """각 테스트 메서드 실행 전에 실행"""
        self.assertTrue(self.api.connected, "크레온 API가 연결되어 있지 않습니다.")
        
        # 현재가 조회 (get_current_minute_data를 사용하여 가장 최신 종가를 가져옴)
        # get_current_minute_data는 DataFrame을 반환하므로, 'close' 컬럼에서 값을 추출합니다.
        current_minute_df = self.api.get_current_minute_data(self.test_stock_code)
        
        self.assertFalse(current_minute_df.empty, "현재가 데이터 조회 실패: DataFrame이 비어있습니다.")
        
        # 가장 최신 (마지막) 행의 'close' 가격을 가져옵니다.
        # DataFrame이 비어있지 않음을 assertFalse로 확인했으므로 iloc[-1]은 안전합니다.
        self.current_price = current_minute_df['close'].iloc[-1]
        
        logging.info(f"현재가: {self.current_price:,.0f}원")
        self.assertIsNotNone(self.current_price, "현재가 추출 실패: self.current_price가 None입니다.")
        # current_price가 0일 경우 문제 발생할 수 있으므로 0보다 큰지 확인
        self.assertGreater(self.current_price, 0, "현재가가 0 이하입니다. 유효하지 않은 가격입니다.")
        
        # 계좌 잔고 조회
        balance = self.api.get_current_account_balance()
        self.cash_balance = balance['cash_balance']
        logging.info(f"주문 가능 현금: {self.cash_balance:,.0f}원")

    def test_1_market_buy_order(self):
        """시장가 매수 주문 테스트"""
        # 1주 시장가 매수
        quantity = 1
        order_type = 'buy'
        order_kind = '03' # 시장가
        
        order_id = self.api.send_order(
            stock_code=self.test_stock_code,
            order_type=order_type,
            price=0,  # 시장가는 가격 0
            quantity=quantity,
            order_kind=order_kind
        )
        
        self.assertIsNotNone(order_id, "주문번호가 반환되지 않았습니다")
        logging.info(f"매수 주문번호: {order_id}")
        
        # 주문 상태 확인
        time.sleep(1)  # 체결 대기
        order_status = self.api.get_order_status(order_id)
        logging.info(f"주문 상태: {order_status}")
        
        # 포지션 확인
        positions = self.api.get_portfolio_positions()
        found = False
        for pos in positions:
            if pos['stock_code'] == self.test_stock_code:
                found = True
                self.assertGreaterEqual(pos['size'], quantity)
                logging.info(f"매수 후 보유수량: {pos['size']}주")
                break
        
        self.assertTrue(found, "매수 후 포지션이 생성되지 않았습니다")

    def test_2_limit_sell_order(self):
        """지정가 매도 주문 테스트"""
        # 현재 보유수량 확인
        positions = self.api.get_portfolio_positions()
        quantity_to_sell = 0
        for pos in positions:
            if pos['stock_code'] == self.test_stock_code:
                quantity_to_sell = pos['size']
                break
        
        if quantity_to_sell == 0:
            logging.warning("매도할 수량이 없습니다. test_1_market_buy_order가 먼저 실행되어야 합니다.")
            # self.skipTest("매도할 수량이 없습니다.") # 테스트를 건너뛸 수도 있습니다.
            return
        
        # 지정가 매도 주문 (현재가 + 5%)
        # self.current_price는 setUp에서 이미 최신 종가로 설정되어 있습니다.
        sell_price = int(self.current_price * 1.05)
        logging.info(f"지정가 매도 가격: {sell_price:,.0f}원")
        
        order_type = 'sell'
        order_kind = get_order_code('지정가')  # 지정가 (보통가) 주문
        
        logging.info(f"주문 유형: {order_type}, 호가구분: {order_kind} ({get_order_name(order_kind)})")

        order_id = self.api.send_order(
            stock_code=self.test_stock_code,
            order_type=order_type,
            price=sell_price,
            quantity=quantity_to_sell,
            order_kind=order_kind
        )
        
        self.assertIsNotNone(order_id, "주문번호가 반환되지 않았습니다")
        logging.info(f"매도 주문번호: {order_id}")
        
        # 주문 상태 확인
        time.sleep(1)  # 체결 대기
        order_status = self.api.get_order_status(order_id)
        logging.info(f"주문 상태: {order_status}")

    def test_3_order_status_check(self):
        """미체결 주문 조회 테스트"""
        # 지정가 매수 주문 (현재가 - 0.2% - 미체결 유도를 위해 현재가보다 약간 낮게)
        # self.current_price는 setUp에서 이미 최신 종가로 설정되어 있습니다.
        buy_price = int(self.current_price * 0.998)
        quantity = 1
        order_type = 'buy'
        order_kind = get_order_code('지정가')  # 지정가(보통가) 주문
        
        # 테스트를 위해 임시로 고정 가격 사용
        # 실제 시장 상황에 따라 미체결이 되거나 바로 체결될 수 있습니다.
        # buy_price = 12990 # 테스트를 위한 고정 가격 (환경에 따라 조절 필요)

        order_id = self.api.send_order(
            stock_code=self.test_stock_code,
            order_type=order_type,
            price=buy_price, # 변경된 부분
            quantity=quantity,
            order_kind=order_kind
        )
        
        self.assertIsNotNone(order_id, "주문번호가 반환되지 않았습니다")
        logging.info(f"미체결 유도 매수 주문번호: {order_id}")
        
        # 주문 상태 확인 (최대 180초 대기)
        for i in range(180):
            time.sleep(1)
            order_status = self.api.get_order_status(order_id)
            logging.info(f"주문 상태 확인 {i+1}회차: {order_status}")
            
            # 체결 수량이 있으면 중단
            if order_status.get('executed_quantity', 0) > 0:
                logging.info(f"주문이 체결되었습니다. 체결 수량: {order_status['executed_quantity']}")
                break
            
            # 주문이 취소되거나 거부된 경우 중단
            if order_status.get('order_status', '') in ['취소', '거부']: # '취소' 또는 '거부'는 예시, 실제 상태명 확인 필요
                 logging.warning(f"주문이 취소 또는 거부되었습니다: {order_status.get('order_status')}")
                 break
        else: # 루프가 break 없이 완료된 경우 (180초 동안 미체결)
            logging.warning(f"주문({order_id})이 180초 동안 미체결 상태입니다. 현재 주문 상태: {order_status}")

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