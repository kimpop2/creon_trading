import unittest
import sys
import os
import logging
from datetime import datetime, date, timedelta
import time


# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.data_manager import DataManager
from util.const import *

class TestCreonAPIOrder(unittest.TestCase):
    """크레온 API 주문 실행 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행"""
        cls.api = CreonAPIClient()
        cls.data_manager = DataManager()
        cls.test_stock_code = 'A005930'  # 삼성전자
        logging.info("테스트 환경 설정 완료")

    def setUp(self):
        """각 테스트 메서드 실행 전에 실행"""
        self.assertTrue(self.api.connected, "크레온 API가 연결되어 있지 않습니다.")
        
        # 현재가 조회
        self.current_price = self.api.get_current_price(self.test_stock_code)
        self.assertIsNotNone(self.current_price, "현재가 조회 실패")
        
        # 계좌 잔고 조회
        balance = self.api.get_account_balance()
        self.cash = balance['cash']
        logging.info(f"주문 가능 현금: {self.cash:,.0f}원")

    def test_1_market_buy_order(self):
        """시장가 매수 주문 테스트"""
        # 1주 시장가 매수
        quantity = 1
        order_type = 'buy' #get_order_code('매수')   # '매도': '1', '매수': '2' # 주문 유형
        order_kind = '03' #get_order_code('시장가')  # '지정가': '01', '임의': '02', '시장가': '03' 주문 호가 구분
        
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
            logging.warning("매도할 수량이 없습니다")
            return
        
        # 지정가 매도 주문 (현재가 + 5%)
        sell_price = int(self.current_price * 1.05)
        order_type = 'sell'
        order_kind = get_order_code('지정가')  # 지정가 (보통가) 주문
        
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
        # 지정가 매수 주문 (현재가 - 5%)
        buy_price = int(self.current_price * 0.95)
        quantity = 1
        order_type = 'buy'
        order_kind = get_order_code('지정가')  # 지정가(보통가) 주문
        
        order_id = self.api.send_order(
            stock_code=self.test_stock_code,
            order_type=order_type,
            price=buy_price,
            quantity=quantity,
            order_kind=order_kind
        )
        
        self.assertIsNotNone(order_id, "주문번호가 반환되지 않았습니다")
        logging.info(f"매수 주문번호: {order_id}")
        
        # 주문 상태 확인 (3회)
        for i in range(3):
            time.sleep(1)
            order_status = self.api.get_order_status(order_id)
            logging.info(f"주문 상태 확인 {i+1}회차: {order_status}")
            
            # 체결 수량이 있으면 중단
            if order_status.get('executed_quantity', 0) > 0:
                break

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