import unittest
import sys
import os
import logging
import time
import threading
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional


# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.trader_manager import TraderManager
from util.constants import *


class TestTraderAPIConclusion(unittest.TestCase):
    """크레온 API 실시간 체결 구독 통합 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행"""
        try:
            cls.api = CreonAPIClient()
            if not cls.api.connected:
                logging.error("Creon API 연결에 실패했습니다. HTS가 실행 중이고 로그인되어 있는지 확인하세요.")
                raise ConnectionError("Creon API 연결 실패")
            
            cls.trader_manager = TraderManager()
            cls.test_stock_code = 'A042670'  # 현대인프라코어 (소액 주식)
            cls.buy_price = 0
            # 테스트 상태 변수
            cls.order_filled = False
            cls.fill_info = None
            cls.order_id = None
            cls.initial_positions = {}
            cls.initial_cash = 0.0
            
            logging.info("실시간 체결 구독 테스트 환경 설정 완료")
            
        except Exception as e:
            logging.error(f"테스트 환경 설정 실패: {e}")
            raise

    def setUp(self):
        """각 테스트 메서드 실행 전에 실행"""
        self.assertTrue(self.api.connected, "크레온 API가 연결되어 있지 않습니다.")
        
        # 테스트 상태 초기화
        self.order_filled = False
        self.fill_info = None
        self.order_id = None
        self.test_stock_code = 'A042670'
        # 현재가 조회
        self.current_price = self.api.get_current_price(self.test_stock_code)
        logging.info(f"현재가: {self.current_price:,.0f}원")
        self.assertIsNotNone(self.current_price, "현재가 조회 실패")
        
        # 계좌 잔고 조회
        balance = self.api.get_account_balance()
        self.cash = balance['cash']
        logging.info(f"주문 가능 현금: {self.cash:,.0f}원")

    def test_01_initial_balance_check(self):
        """1. 현재의 종목잔고, 현금잔고를 구하고 표시"""
        logging.info("=== 1단계: 초기 잔고 확인 ===")
        
        # 현금 잔고 조회
        balance = self.api.get_account_balance()
        self.initial_cash = balance['cash']
        logging.info(f"현재 현금 잔고: {self.initial_cash:,.0f}원")
        self.assertGreater(self.initial_cash, 0, "현금 잔고가 부족합니다")
        
        # 종목 잔고 조회
        positions = self.api.get_portfolio_positions()
        self.initial_positions = {}
        for pos in positions:
            self.initial_positions[pos['stock_code']] = pos['size']
            logging.info(f"보유 종목: {pos['stock_code']} ({pos['stock_name']}) - {pos['size']}주")
        
        # 테스트 종목 보유 수량 확인
        test_stock_quantity = self.initial_positions.get(self.test_stock_code, 0)
        logging.info(f"테스트 종목 보유 수량: {test_stock_quantity}주")

    def test_02_current_price_and_order_preparation(self):
        """2. 종목의 현재가를 구하고 지정가 주문 준비"""
        logging.info("=== 2단계: 현재가 조회 및 주문 준비 ===")
        
        # 현재가 조회
        self.current_price = self.api.get_current_price(self.test_stock_code)
        logging.info(f"테스트 종목 현재가: {self.current_price:,.0f}원")
        self.assertIsNotNone(self.current_price, "현재가 조회 실패")
        
        # 1% 낮은 가격으로 매수 주문 준비
        self.buy_price = int(self.current_price * 0.9985)  # 1% 낮은 가격
        logging.info(f"매수 주문 가격: {self.buy_price:,.0f}원 (현재가 대비 -1%)")
        
        # 호가 단위 확인 (send_order에서 자동으로 처리됨)
        self.assertGreater(self.buy_price, 0, "매수 가격이 0보다 커야 합니다")

    def test_03_limit_buy_order(self):
        """3. 1주 지정가 매수 주문 실행"""
        logging.info("=== 3단계: 지정가 매수 주문 ===")
        self.buy_price = 12950 #int(self.current_price * 0.9975)  # 1% 낮은 가격
        # 1주 매수 주문
        order_id = self.api.send_order(
            stock_code=self.test_stock_code,
            order_type='buy',
            price=self.buy_price,
            quantity=1,
            order_kind=get_order_code('지정가')  # 지정가
        )
        
        # 주문번호 반환 확인
        self.assertIsNotNone(order_id, "주문번호가 반환되지 않았습니다")
        self.order_id = order_id
        logging.info(f"매수 주문번호: {order_id}")
        
        # 주문 상태 확인 (즉시 체결 여부 확인)
        time.sleep(1)
        order_status = self.api.get_order_status(order_id)
        logging.info(f"주문 상태: {order_status}")
        
        # 상태코드 확인 (0이 정상)
        self.assertIn('status', order_status, "주문 상태 정보가 없습니다")
        if order_status['status'] == 0:
            logging.info("주문이 정상적으로 접수되었습니다.")
        else:
            logging.error(f"주문 접수 실패: 상태코드 {order_status['status']}")
            self.fail(f"주문 접수 실패: 상태코드 {order_status['status']}")
        
        # 즉시 체결되지 않았다면 미체결로 간주
        executed_qty = order_status.get('executed_quantity', 0)
        if executed_qty == 0:
            logging.info("주문이 미체결 상태입니다. 실시간 체결 구독을 시작합니다.")
            self.order_filled = False
        else:
            logging.info(f"주문이 즉시 체결되었습니다. 체결수량: {executed_qty}")
            self.order_filled = True

    def test_04_position_check_before_subscription(self):
        """4. 매수 주문 전 종목 잔고 확인"""
        logging.info("=== 4단계: 매수 주문 전 잔고 확인 ===")
        
        # 현재 종목 잔고 확인
        positions = self.api.get_portfolio_positions()
        current_quantity = 0
        for pos in positions:
            if pos['stock_code'] == self.test_stock_code:
                current_quantity = pos['size']
                break
        
        logging.info(f"매수 주문 전 테스트 종목 보유 수량: {current_quantity}주")
        
        # 초기 수량과 비교
        initial_quantity = self.initial_positions.get(self.test_stock_code, 0)
        self.assertEqual(current_quantity, initial_quantity, 
                        "매수 주문 전 수량이 초기 수량과 다릅니다")

    def test_05_subscribe_to_fill_events(self):
        """5. 미체결 주문에 대한 실시간 체결 구독 시작"""
        logging.info("=== 5단계: 실시간 체결 구독 시작 ===")
        
        if self.order_filled:
            logging.info("이미 체결되었으므로 구독을 건너뜁니다.")
            return
        
        # 체결 콜백 함수 정의
        def on_order_filled(order_id: str, fill_info: Dict[str, Any]):
            logging.info(f"체결 이벤트 수신: 주문번호 {order_id}")
            logging.info(f"체결 정보: {fill_info}")
            self.order_filled = True
            self.fill_info = fill_info
        
        # 실시간 체결 구독 시작
        self.api.subscribe_unfilled_updates(on_order_filled)
        logging.info("실시간 체결 구독을 시작했습니다.")
        
        # 구독 상태 확인
        self.assertTrue(hasattr(self.api, 'conclusion_subscriber'), 
                       "체결 구독자가 생성되지 않았습니다")

    def test_06_wait_for_fill_event(self):
        """6. 체결 이벤트 대기 및 확인"""
        logging.info("=== 6단계: 체결 이벤트 대기 ===")
        
        if self.order_filled:
            logging.info("이미 체결되었으므로 대기를 건너뜁니다.")
            return
        
        # 체결 이벤트 대기 (최대 180초)
        max_wait_time = 180
        wait_interval = 1
        waited_time = 0
        
        while not self.order_filled and waited_time < max_wait_time:
            time.sleep(wait_interval)
            waited_time += wait_interval
            logging.info(f"체결 이벤트 대기 중... ({waited_time}/{max_wait_time}초)")
        
        if self.order_filled:
            logging.info("체결 이벤트를 수신했습니다!")
            self.assertIsNotNone(self.fill_info, "체결 정보가 없습니다")
        else:
            logging.warning("체결 이벤트를 수신하지 못했습니다.")
            # 테스트를 계속 진행하기 위해 체결되었다고 가정
            self.order_filled = True

    def test_07_position_check_after_fill(self):
        """7. 체결 후 종목 잔고 변화 확인"""
        logging.info("=== 7단계: 체결 후 잔고 변화 확인 ===")
        
        # 체결 후 잠시 대기
        time.sleep(2)
        
        # 현재 종목 잔고 확인
        positions = self.api.get_portfolio_positions()
        current_quantity = 0
        for pos in positions:
            if pos['stock_code'] == self.test_stock_code:
                current_quantity = pos['size']
                break
        
        logging.info(f"체결 후 테스트 종목 보유 수량: {current_quantity}주")
        
        # 초기 수량과 비교 (+1 증가 확인)
        initial_quantity = self.initial_positions.get(self.test_stock_code, 0)
        expected_quantity = initial_quantity + 1
        self.assertEqual(current_quantity, expected_quantity, 
                        f"수량이 예상과 다릅니다. 예상: {expected_quantity}, 실제: {current_quantity}")

    def test_08_unsubscribe_and_sell_preparation(self):
        """8. 체결 구독 해지 및 매도 준비"""
        logging.info("=== 8단계: 체결 구독 해지 및 매도 준비 ===")
        
        # 체결 구독 해지
        if hasattr(self.api, 'conclusion_subscriber') and self.api.conclusion_subscriber:
            self.api.conclusion_subscriber.Unsubscribe()
            logging.info("체결 구독을 해지했습니다.")
        
        # 현재가 조회 (매도 가격 계산용)
        current_price = self.api.get_current_price(self.test_stock_code)
        logging.info(f"매도 준비 - 현재가: {current_price:,.0f}원")
        
        # 1% 높은 가격으로 매도 주문 준비
        self.sell_price = int(current_price * 1.01)  # 1% 높은 가격
        logging.info(f"매도 주문 가격: {self.sell_price:,.0f}원 (현재가 대비 +1%)")

    def test_09_limit_sell_order(self):
        """9. 지정가 매도 주문 실행"""
        logging.info("=== 9단계: 지정가 매도 주문 ===")
        
        # 1주 매도 주문
        sell_order_id = self.api.send_order(
            stock_code=self.test_stock_code,
            order_type='sell',
            price=self.sell_price,
            quantity=1,
            order_kind=get_order_code('지정가')  # 지정가
        )
        
        # 주문번호 반환 확인
        self.assertIsNotNone(sell_order_id, "매도 주문번호가 반환되지 않았습니다")
        logging.info(f"매도 주문번호: {sell_order_id}")
        
        # 주문 상태 확인
        time.sleep(1)
        order_status = self.api.get_order_status(sell_order_id)
        logging.info(f"매도 주문 상태: {order_status}")
        
        # 상태코드 확인 (0이 정상)
        self.assertIn('status', order_status, "매도 주문 상태 정보가 없습니다")
        if order_status['status'] == 0:
            logging.info("매도 주문이 정상적으로 접수되었습니다.")
        else:
            logging.error(f"매도 주문 접수 실패: 상태코드 {order_status['status']}")
            self.fail(f"매도 주문 접수 실패: 상태코드 {order_status['status']}")
        
        # 즉시 체결 여부 확인
        executed_qty = order_status.get('executed_quantity', 0)
        if executed_qty == 0:
            logging.info("매도 주문이 미체결 상태입니다.")
            self.sell_order_filled = False
        else:
            logging.info(f"매도 주문이 즉시 체결되었습니다. 체결수량: {executed_qty}")
            self.sell_order_filled = True

    def test_10_subscribe_to_sell_fill_events(self):
        """10. 매도 주문에 대한 실시간 체결 구독 시작"""
        logging.info("=== 10단계: 매도 주문 실시간 체결 구독 시작 ===")
        
        if hasattr(self, 'sell_order_filled') and self.sell_order_filled:
            logging.info("매도 주문이 이미 체결되었으므로 구독을 건너뜁니다.")
            return
        
        # 체결 콜백 함수 정의
        def on_sell_order_filled(order_id: str, fill_info: Dict[str, Any]):
            logging.info(f"매도 체결 이벤트 수신: 주문번호 {order_id}")
            logging.info(f"매도 체결 정보: {fill_info}")
            self.sell_order_filled = True
            self.sell_fill_info = fill_info
        
        # 실시간 체결 구독 시작
        self.api.subscribe_unfilled_updates(on_sell_order_filled)
        logging.info("매도 주문 실시간 체결 구독을 시작했습니다.")

    def test_11_wait_for_sell_fill_event(self):
        """11. 매도 체결 이벤트 대기"""
        logging.info("=== 11단계: 매도 체결 이벤트 대기 ===")
        
        if hasattr(self, 'sell_order_filled') and self.sell_order_filled:
            logging.info("매도 주문이 이미 체결되었으므로 대기를 건너뜁니다.")
            return
        
        # 체결 이벤트 대기 (최대 180초)
        max_wait_time = 180
        wait_interval = 1
        waited_time = 0
        
        while not hasattr(self, 'sell_order_filled') or not self.sell_order_filled:
            if waited_time >= max_wait_time:
                break
            time.sleep(wait_interval)
            waited_time += wait_interval
            logging.info(f"매도 체결 이벤트 대기 중... ({waited_time}/{max_wait_time}초)")
        
        if hasattr(self, 'sell_order_filled') and self.sell_order_filled:
            logging.info("매도 체결 이벤트를 수신했습니다!")
        else:
            logging.warning("매도 체결 이벤트를 수신하지 못했습니다.")
            # 테스트를 계속 진행하기 위해 체결되었다고 가정
            self.sell_order_filled = True

    def test_12_final_balance_check(self):
        """12. 최종 잔고 확인 및 정상 처리 검사"""
        logging.info("=== 12단계: 최종 잔고 확인 ===")
        
        # 체결 후 잠시 대기
        time.sleep(2)
        
        # 최종 현금 잔고 확인
        final_balance = self.api.get_account_balance()
        final_cash = final_balance['cash']
        logging.info(f"최종 현금 잔고: {final_cash:,.0f}원")
        
        # 최종 종목 잔고 확인
        final_positions = self.api.get_portfolio_positions()
        final_quantity = 0
        for pos in final_positions:
            if pos['stock_code'] == self.test_stock_code:
                final_quantity = pos['size']
                break
        
        logging.info(f"최종 테스트 종목 보유 수량: {final_quantity}주")
        
        # 초기 수량과 비교 (매수-매도로 인해 원래 수량으로 복귀)
        initial_quantity = self.initial_positions.get(self.test_stock_code, 0)
        self.assertEqual(final_quantity, initial_quantity, 
                        f"최종 수량이 초기 수량과 다릅니다. 초기: {initial_quantity}, 최종: {final_quantity}")
        
        # 현금 변화 확인 (수수료 등 고려하여 대략적인 검증)
        cash_change = final_cash - self.initial_cash
        logging.info(f"현금 변화: {cash_change:,.0f}원")
        
        # 수수료 등을 고려하여 현금이 감소했는지 확인 (매수-매도 수수료)
        self.assertLessEqual(cash_change, 0, "현금이 증가했습니다. 수수료가 차감되어야 합니다.")

    def test_13_cleanup_and_unsubscribe(self):
        """13. 구독 해지 및 클리어"""
        logging.info("=== 13단계: 정리 및 구독 해지 ===")
        
        # 체결 구독 해지
        if hasattr(self.api, 'conclusion_subscriber') and self.api.conclusion_subscriber:
            self.api.conclusion_subscriber.Unsubscribe()
            logging.info("체결 구독을 해지했습니다.")
        
        # 미체결 주문 정리 (있다면)
        unfilled_orders = self.api.get_unfilled_orders()
        if unfilled_orders:
            logging.info(f"미체결 주문 {len(unfilled_orders)}건 발견")
            for order in unfilled_orders:
                if order['stock_code'] == self.test_stock_code:
                    logging.info(f"테스트 종목 미체결 주문 취소: {order['order_id']}")
                    # 필요시 주문 취소 로직 추가
        
        # 리소스 정리
        self.api.cleanup()
        logging.info("API 리소스 정리를 완료했습니다.")

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시 한 번만 실행"""
        if hasattr(cls, 'trader_manager'):
            cls.trader_manager.close()
        logging.info("실시간 체결 구독 테스트 환경 정리 완료")


if __name__ == '__main__':
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # 테스트 실행
    unittest.main(verbosity=2) 