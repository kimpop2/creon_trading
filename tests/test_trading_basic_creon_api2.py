# tests/test_trading_basic_creon_api2.py
import unittest
import sys
import os
import logging
import time
import queue # queue 모듈 추가
from datetime import datetime, date, timedelta
from threading import Event, Lock
from typing import Optional, List, Dict, Any, Callable, Tuple

# 프로젝트 루트 경로 추가 (assuming tests/test_file.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 리팩토링된 CreonAPIClient 및 관련 Enum 임포트
from api.creon_api2 import CreonAPIClient, OrderType, OrderStatus

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestCreonAPIScenario(unittest.TestCase):
    """
    리팩토링된 CreonAPIClient를 이용한 고급 매매 시나리오 테스트
    - 현재가/호가 이벤트 구독/해지 시점 제어
    - 단계별 주문 정정 및 취소/시장가 주문 로직 포함
    """
    cls_api: CreonAPIClient = None
    
    # 실시간 이벤트 핸들링 변수 (클래스 변수로 선언하여 콜백에서 접근)
    _conclusion_events_queue: List[Dict[str, Any]] = [] # 체결/응답 이벤트를 큐로 관리 (리스트로 변경)
    _conclusion_event_signal = Event() # 체결/응답 이벤트 수신 신호
    _price_event_data = None
    _price_event_received = Event() # 현재가 이벤트 수신 신호
    
    # 콜백 데이터에 대한 동시성 제어를 위한 락
    _callback_lock = Lock()

    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행: CreonAPIClient 초기화"""
        logger.info("--- 테스트 클래스 설정 시작 ---")
        try:
            cls.cls_api = CreonAPIClient()
            # 콜백 함수 등록
            cls.cls_api.set_conclusion_callback(cls._conclusion_callback)
            cls.cls_api.set_price_update_callback(cls._price_update_callback)
            cls.cls_api.set_bid_update_callback(cls._bid_update_callback) # 호가 콜백도 등록
            logger.info("CreonAPIClient 초기화 및 콜백 등록 완료.")
        except Exception as e:
            logger.error(f"CreonAPIClient 초기화 실패: {e}", exc_info=True)
            cls.cls_api = None
            raise
        logger.info("--- 테스트 클래스 설정 완료 ---")

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시 한 번만 실행: 리소스 정리"""
        logger.info("--- 테스트 클래스 정리 시작 ---")
        if cls.cls_api:
            cls.cls_api.cleanup() # CreonAPIClient의 cleanup 메서드 호출 (모든 구독 해지 포함)
        logger.info("--- 테스트 클래스 정리 완료 ---")

    def setUp(self):
        """각 테스트 메서드 실행 전에 실행"""
        self.assertIsNotNone(self.cls_api, "CreonAPIClient가 초기화되지 않았습니다. setUpClass를 확인하세요.")
        self.assertTrue(self.cls_api.is_connected(), "크레온 PLUS가 연결되어 있지 않습니다. HTS 로그인 상태를 확인하세요.")
        
        # 각 테스트 시작 시 이벤트 플래그 및 데이터 초기화
        with TestCreonAPIScenario._callback_lock:
            TestCreonAPIScenario._conclusion_events_queue.clear() # 큐 초기화
            TestCreonAPIScenario._conclusion_event_signal.clear() # 신호 초기화
            TestCreonAPIScenario._price_event_received.clear()
            TestCreonAPIScenario._price_event_data = None

        logger.info(f"\n--- {self._testMethodName} 테스트 시작 ---")

    def tearDown(self):
        """각 테스트 메서드 실행 후에 실행"""
        logger.info(f"--- {self._testMethodName} 테스트 종료 ---\n")
        # 테스트 종료 시 혹시 모를 잔여 구독 해지 (안전 장치)
        self.cls_api.unsubscribe_all_realtime_data()

    @classmethod
    def _conclusion_callback(cls, data: Dict[str, Any]):
        """실시간 주문 체결/응답 콜백 핸들러"""
        with cls._callback_lock:
            logger.info(f"실시간 체결/응답 수신: {data}")
            cls._conclusion_events_queue.append(data) # 큐에 이벤트 추가
            cls._conclusion_event_signal.set() # 이벤트 발생 신호

    @classmethod
    def _price_update_callback(cls, stock_code: str, current_price: int, timestamp: float):
        """실시간 현재가 업데이트 콜백 핸들러 (5초 룰은 여기서 처리)"""
        with cls._callback_lock:
            if stock_code not in cls.cls_api._last_price_print_time_per_stock:
                cls.cls_api._last_price_print_time_per_stock[stock_code] = 0.0

            if time.time() - cls.cls_api._last_price_print_time_per_stock[stock_code] >= 5:
                logger.info(f"실시간 현재가 수신: 종목={stock_code}, 가격={current_price:,.0f}원")
                cls.cls_api._last_price_print_time_per_stock[stock_code] = time.time()
                cls._price_event_data = {'stock_code': stock_code, 'current_price': current_price, 'timestamp': timestamp}
                cls._price_event_received.set() # 이벤트 발생 신호 (필요시)

    @classmethod
    def _bid_update_callback(cls, stock_code: str, offer_prices: List[int], bid_prices: List[int], offer_amounts: List[int], bid_amounts: List[int]):
        """실시간 10차 호가 업데이트 콜백 핸들러 (현재는 로깅만)"""
        logger.debug(f"실시간 호가 수신: 종목={stock_code}, 1차 매도={offer_prices[0]}, 1차 매수={bid_prices[0]}")
        # 이 콜백에서 직접 데이터를 사용하는 대신, 필요 시점에 get_current_price_and_quotes를 다시 호출하여 최신 스냅샷을 가져오는 방식 사용

    def _wait_for_conclusion_event(self, target_order_id: Optional[int], expected_flags: List[str], timeout: int = 60) -> Optional[Dict[str, Any]]:
        """
        특정 주문 ID에 대한 체결/응답 이벤트를 기다립니다.
        target_order_id가 None이면 어떤 체결 이벤트든 기다립니다.
        expected_flags 리스트에 있는 flag 중 하나라도 만족하면 반환합니다.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            TestCreonAPIScenario._conclusion_event_signal.clear() # 이전 신호 초기화
            logger.debug(f"체결/응답 이벤트 대기 중 (남은 시간: {int(timeout - (time.time() - start_time))}초), 기대 플래그: {expected_flags}, 대상 주문: {target_order_id}...")
            
            # 이벤트를 기다리거나, 0.5초마다 큐를 확인 (너무 짧으면 CPU 소모가 클 수 있음)
            if TestCreonAPIScenario._conclusion_event_signal.wait(0.5): # 짧은 대기 후 큐 확인
                pass # 신호가 왔으니 큐를 바로 확인
            
            with TestCreonAPIScenario._callback_lock:
                # 큐를 역순으로 탐색하여 가장 최신 이벤트를 먼저 확인
                for i in range(len(TestCreonAPIScenario._conclusion_events_queue) - 1, -1, -1):
                    data = TestCreonAPIScenario._conclusion_events_queue[i]
                    
                    if target_order_id is None or data.get('order_num') == target_order_id:
                        if data.get('flag') in expected_flags:
                            logger.info(f"대상 체결/응답 이벤트 수신: {data}")
                            # 해당 이벤트를 큐에서 제거 (선택 사항이지만, 중복 처리 방지)
                            del TestCreonAPIScenario._conclusion_events_queue[i]
                            return data
                        else:
                            logger.debug(f"대상 주문 ({target_order_id})의 다른 플래그 ({data.get('flag')}) 이벤트 수신. 기대 플래그 {expected_flags} 대기 계속...")
                            # 해당 이벤트를 큐에서 제거 (이미 확인했으므로)
                            del TestCreonAPIScenario._conclusion_events_queue[i]
                    
                # 큐에 남은 다른 주문 이벤트가 있을 경우 처리
                if TestCreonAPIScenario._conclusion_events_queue:
                    logger.debug(f"큐에 다른 미처리 체결/응답 이벤트가 남아있습니다: {TestCreonAPIScenario._conclusion_events_queue}")
            
        logger.warning(f"체결/응답 이벤트 타임아웃 ({timeout}초) 또는 대상 주문 ({target_order_id}) 이벤트 미수신.")
        return None

    def _wait_for_price_event(self, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """가격 업데이트 이벤트를 기다립니다."""
        TestCreonAPIScenario._price_event_received.clear()
        TestCreonAPIScenario._price_event_data = None
        logger.debug(f"가격 이벤트 대기 중 (최대 {timeout}초)...")
        if TestCreonAPIScenario._price_event_received.wait(timeout):
            with TestCreonAPIScenario._callback_lock:
                return TestCreonAPIScenario._price_event_data
        logger.warning(f"가격 이벤트 타임아웃 ({timeout}초).")
        return None

    def test_full_trading_scenario_advanced(self):
        """
        매수 -> 정정 -> 취소 -> 시장가 매수 -> 정정 -> 시장가 매도 시나리오 테스트
        """
        api = self.cls_api
        test_stock_code = 'A008970' # 동양철관 (테스트용)
        order_quantity = 1 # 테스트용 최소 수량

        try: # 전체 테스트 메서드에 try-except 블록 추가
            logger.info(f"테스트 종목: {test_stock_code}, 주문 수량: {order_quantity}")

            # --- 매수 시나리오 시작 ---
            logger.info("\n--- 매수 시나리오 시작 ---")

            # 1. 예수금 확인
            logger.info("1. 예수금 확인...")
            account_balance = api.get_account_balance()
            self.assertIsNotNone(account_balance, "계좌 잔고 조회 실패")
            current_cash = account_balance.get('cash_balance', 0)
            logger.info(f"현재 주문 가능 현금: {current_cash:,.0f}원")
            self.assertGreater(current_cash, 10000, "주문 가능 현금이 부족합니다 (최소 10,000원 필요).")

            # 2. 매수 종목 결정 (test_stock_code 사용)

            # 3. 현재가 실시간 이벤트 등록 (매매 시나리오 내내 유지)
            logger.info(f"3. 종목 [{test_stock_code}] 현재가 실시간 이벤트 등록...")
            api.subscribe_realtime_price(test_stock_code)
            # 실시간 구독 후 데이터가 들어올 때까지 충분히 대기
            time.sleep(5) # 5초 대기 추가
            self._wait_for_price_event(timeout=10) # 실시간 데이터가 들어올 때까지 잠시 대기

            # 4. 현재가 및 호가 조회 및 초기 매수 주문 (매수 3호가)
            logger.info("4. 현재가 및 호가 조회 후 매수 3호가 주문 시도...")
            api.subscribe_realtime_bid(test_stock_code) # 호가 수신 이벤트 등록 (주문 직전)
            time.sleep(5) # 5초 대기 추가
            price_info = api.get_current_price_and_quotes(test_stock_code)
            # api.unsubscribe_realtime_bid(test_stock_code) # <-- 이 위치에서 해지하지 않고, 주문 처리 루프 후에 해지

            self.assertIsNotNone(price_info, "현재가 및 호가 조회 실패")
            current_price = price_info['current_price']
            bid_prices = price_info['bid_prices']
            
            initial_buy_price = bid_prices[2] if len(bid_prices) >= 3 and bid_prices[2] > 0 else current_price 
            if initial_buy_price == 0: initial_buy_price = current_price
            initial_buy_price = api.round_to_tick(initial_buy_price)
            
            logger.info(f"현재가: {current_price:,.0f}원, 초기 매수 희망가격 (3차 매수호가): {initial_buy_price:,.0f}원")
            self.assertGreater(initial_buy_price, 0, "유효한 초기 매수 가격을 찾을 수 없습니다.")

            buy_order_result = api.send_order(
                stock_code=test_stock_code,
                order_type=OrderType.BUY,
                amount=order_quantity,
                price=initial_buy_price,
                order_unit="01" # 보통가
            )
            self.assertEqual(buy_order_result['status'], 'success', f"초기 매수 주문 실패: {buy_order_result['message']}")
            current_order_id = buy_order_result['order_num']
            current_buy_order_price = initial_buy_price # 현재 주문 가격 추적
            self.assertIsNotNone(current_order_id, "초기 매수 주문번호가 반환되지 않았습니다.")
            logger.info(f"초기 매수 주문번호: {current_order_id}")

            # 5. 매수 주문 정정/취소/시장가 매수 로직
            executed_buy_qty = 0
            order_filled = False
            
            # 정정 시나리오 (3차 -> 2차 -> 1차)
            for i in range(3): # 0: 3차, 1: 2차, 2: 1차
                logger.info(f"매수 주문 ({current_order_id}) 체결 대기 (단계 {i+1}/3 - {10}초)...")
                # '체결' 또는 '확인' 플래그를 기다립니다.
                conclusion_data = self._wait_for_conclusion_event(target_order_id=current_order_id, expected_flags=['체결', '확인'], timeout=10)
                
                if conclusion_data and conclusion_data['flag'] == '체결':
                    executed_buy_qty += conclusion_data['amount']
                    logger.info(f"매수 주문 ({current_order_id}) 부분 체결 발생. 현재 체결 수량: {executed_buy_qty}")
                    if executed_buy_qty >= order_quantity:
                        logger.info(f"매수 주문 ({current_order_id}) 전량 체결 완료.")
                        order_filled = True
                        break
                elif conclusion_data and conclusion_data['flag'] == '거부':
                    self.fail(f"매수 주문 ({current_order_id})이 거부되었습니다: {conclusion_data['message']}")
                elif conclusion_data and conclusion_data['flag'] == '확인':
                    logger.info(f"매수 주문 ({current_order_id}) 확인 이벤트 수신: {conclusion_data['flag']}")
                else: # 10초 내 미체결 (또는 원하는 플래그 미수신)
                    logger.info(f"매수 주문 ({current_order_id}) {10}초 내 미체결 또는 원하는 이벤트 미수신. 정정 시도...")
                    if i < 2: # 2차, 1차 호가로 정정
                        # 호가 정보는 이미 구독 중이므로 다시 구독할 필요 없음. 최신 정보 조회
                        # api.subscribe_realtime_bid(test_stock_code) # 제거
                        # time.sleep(5) # 제거
                        price_info_mod = api.get_current_price_and_quotes(test_stock_code)
                        # api.unsubscribe_realtime_bid(test_stock_code) # 제거
                        
                        if not price_info_mod:
                            logger.error("정정 시 호가 정보 조회 실패. 정정 중단.")
                            break

                        # bid_prices의 인덱스를 0, 1, 2 순으로 사용 (3차 -> 2차 -> 1차 매수호가)
                        # i=0: 3차 호가 (bid_prices[2]), i=1: 2차 호가 (bid_prices[1]), i=2: 1차 호가 (bid_prices[0])
                        next_bid_price = price_info_mod['bid_prices'][2-i] if len(price_info_mod['bid_prices']) > (2-i) and price_info_mod['bid_prices'][2-i] > 0 else price_info_mod['current_price']
                        if next_bid_price == 0: next_bid_price = price_info_mod['current_price']
                        next_bid_price = api.round_to_tick(next_bid_price)

                        # 정정 단가가 현재 주문 단가와 동일한지 확인
                        if next_bid_price == current_buy_order_price:
                            logger.info(f"매수 주문 ({current_order_id}) 정정 단가 ({next_bid_price})가 현재 주문 단가 ({current_buy_order_price})와 동일하여 정정 스킵.")
                            continue # 다음 단계로 넘어감
                        
                        logger.info(f"매수 주문 ({current_order_id}) 정정 시도: {3-i}차 매수호가 {next_bid_price:,.0f}원")
                        modify_result = api.send_order(
                            stock_code=test_stock_code,
                            order_type=OrderType.MODIFY,
                            amount=0, # 잔량 정정
                            price=next_bid_price,
                            org_order_num=current_order_id
                        )
                        self.assertEqual(modify_result['status'], 'success', f"매수 정정 주문 실패: {modify_result['message']}")
                        current_order_id = modify_result['order_num'] # 정정 시 새 주문번호 받을 수 있음
                        current_buy_order_price = next_bid_price # 정정 성공 시 가격 업데이트
                        logger.info(f"매수 정정 주문 성공. 새 주문번호: {current_order_id}")
                    else: # 1차 호가까지 정정 후에도 미체결 -> 취소
                        logger.info(f"매수 주문 ({current_order_id}) 1차 호가 정정 후에도 미체결. 취소 시도...")
                        cancel_result = api.send_order(
                            stock_code=test_stock_code,
                            order_type=OrderType.CANCEL,
                            amount=0, # 잔량 취소
                            org_order_num=current_order_id
                        )
                        self.assertEqual(cancel_result['status'], 'success', f"매수 취소 주문 실패: {cancel_result['message']}")
                        logger.info(f"매수 주문 ({current_order_id}) 취소 요청 성공. 체결 이벤트 대기...")
                        # 취소 확인 이벤트 대기
                        conclusion_data = self._wait_for_conclusion_event(target_order_id=current_order_id, expected_flags=['확인'], timeout=10)
                        if conclusion_data and conclusion_data['flag'] == '확인':
                            logger.info(f"매수 주문 ({current_order_id}) 취소 확인 완료.")
                            # --- NEW: Verify cancellation by checking unfilled orders ---
                            logger.info(f"미체결 주문 목록에서 주문번호 {current_order_id} 제거 확인 중...")
                            cancellation_confirmed_in_unfilled = False
                            start_time_unfilled_check = time.time()
                            while time.time() - start_time_unfilled_check < 10: # Wait up to 10 seconds for it to disappear
                                unfilled_orders_after_cancel = api.get_unfilled_orders()
                                found_cancelled_order = False
                                for order in unfilled_orders_after_cancel:
                                    if order['order_id'] == current_order_id:
                                        found_cancelled_order = True
                                        logger.debug(f"주문번호 {current_order_id}가 아직 미체결 목록에 있습니다. 대기 중...")
                                        time.sleep(1)
                                        break
                                if not found_cancelled_order:
                                    cancellation_confirmed_in_unfilled = True
                                    logger.info(f"주문번호 {current_order_id}가 미체결 목록에서 제거됨을 확인했습니다.")
                                    break
                            if not cancellation_confirmed_in_unfilled:
                                logger.warning(f"매수 주문 ({current_order_id})이 미체결 목록에서 제거되지 않았거나 타임아웃 발생.")
                        else:
                            logger.warning(f"매수 주문 ({current_order_id}) 취소 확인 실패 또는 타임아웃.")
                        order_filled = False # 취소되었으므로 체결되지 않음
                        break # 정정/취소 루프 종료
            
            # 매수 주문 처리 루프 종료 후 호가 구독 해지
            api.unsubscribe_realtime_bid(test_stock_code) # <-- 호가 구독 해지 위치 변경

            if not order_filled:
                logger.info("매수 주문이 최종적으로 체결되지 않았습니다. 시장가 매수 시도...")
                time.sleep(5) # 시장가 매수 전 잠시 대기 (API 부하 감소)
                market_buy_result = api.send_order(
                    stock_code=test_stock_code,
                    order_type=OrderType.BUY,
                    amount=order_quantity,
                    price=0, # 시장가
                    order_unit="03" # 시장가 주문
                )
                self.assertEqual(market_buy_result['status'], 'success', f"시장가 매수 주문 실패: {market_buy_result['message']}")
                current_order_id = market_buy_result['order_num']
                self.assertIsNotNone(current_order_id, "시장가 매수 주문번호가 반환되지 않았습니다.")
                logger.info(f"시장가 매수 주문번호: {current_order_id}. 시장가 주문은 성공 시 즉시 체결된 것으로 간주합니다.")
                order_filled = True # 시장가 매수 주문은 성공 시 즉시 체결된 것으로 간주

            # 9. 종목 잔고 확인 (매수 후)
            logger.info("9. 매수 후 종목 잔고 확인...")
            positions_after_buy = api.get_portfolio_positions()
            found_position = False
            for pos in positions_after_buy:
                if pos['stock_code'] == test_stock_code:
                    self.assertGreaterEqual(pos['quantity'], order_quantity, "매수 후 보유 수량이 예상보다 적습니다.")
                    logger.info(f"매수 후 {test_stock_code} 보유 수량: {pos['quantity']}주")
                    found_position = True
                    break
            self.assertTrue(found_position, f"매수 후 {test_stock_code} 종목이 포지션에 없습니다.")

            # 10. 매수 체결 완료 후 호가 실시간 이벤트 해제 (현재가는 유지)
            logger.info("10. 매수 체결 완료 후 호가 실시간 이벤트 해지.")
            # 호가는 이미 위에서 해지되었으므로, 추가 해지 불필요 (안전하게 한번 더 호출 가능)
            api.unsubscribe_realtime_bid(test_stock_code) 


            # --- 매도 시나리오 시작 ---
            logger.info("\n--- 매도 시나리오 시작 ---")

            # 11. 종목 잔고 확인 (매도 가능 종목)
            logger.info("11. 매도 전 종목 잔고 확인...")
            positions_before_sell = api.get_portfolio_positions()
            sellable_quantity = 0
            for pos in positions_before_sell:
                if pos['stock_code'] == test_stock_code:
                    sellable_quantity = pos['sell_avail_qty']
                    break
            self.assertGreaterEqual(sellable_quantity, order_quantity, "매도할 수량이 부족합니다.")
            logger.info(f"매도 가능 수량: {sellable_quantity}주")

            # 12. 매도 종목 결정 (test_stock_code 사용)

            # 13. 현재가 실시간 이벤트는 이미 등록되어 있음

            # 14. 현재가 및 호가 조회 및 초기 매도 주문 (매도 3호가)
            logger.info("14. 현재가 및 호가 조회 후 매도 3호가 주문 시도...")
            api.subscribe_realtime_bid(test_stock_code) # 호가 수신 이벤트 등록 (주문 직전)
            time.sleep(5) # 5초 대기 추가
            price_info_sell = api.get_current_price_and_quotes(test_stock_code)
            # api.unsubscribe_realtime_bid(test_stock_code) # <-- 이 위치에서 해지하지 않고, 주문 처리 루프 후에 해지

            self.assertIsNotNone(price_info_sell, "매도 시나리오를 위한 현재가 및 호가 조회 실패")
            current_price_sell = price_info_sell['current_price']
            offer_prices = price_info_sell['offer_prices']
            
            initial_sell_price = offer_prices[2] if len(offer_prices) >= 3 and offer_prices[2] > 0 else current_price_sell
            if initial_sell_price == 0: initial_sell_price = current_price_sell
            initial_sell_price = api.round_to_tick(initial_sell_price)

            logger.info(f"현재가: {current_price_sell:,.0f}원, 초기 매도 희망가격 (3차 매도호가): {initial_sell_price:,.0f}원")
            self.assertGreater(initial_sell_price, 0, "유효한 초기 매도 가격을 찾을 수 없습니다.")

            sell_order_result = api.send_order(
                stock_code=test_stock_code,
                order_type=OrderType.SELL,
                amount=order_quantity,
                price=initial_sell_price,
                order_unit="01" # 보통가
            )
            self.assertEqual(sell_order_result['status'], 'success', f"초기 매도 주문 실패: {sell_order_result['message']}")
            current_sell_order_id = sell_order_result['order_num']
            current_sell_order_price = initial_sell_price # 현재 주문 가격 추적
            self.assertIsNotNone(current_sell_order_id, "초기 매도 주문번호가 반환되지 않았습니다.")
            logger.info(f"초기 매도 주문번호: {current_sell_order_id}")

            # 15. 매도 주문 정정/시장가 매도 로직
            executed_sell_qty = 0
            sell_order_filled = False

            # 정정 시나리오 (3차 -> 2차 -> 1차)
            for i in range(3): # 0: 3차, 1: 2차, 2: 1차
                logger.info(f"매도 주문 ({current_sell_order_id}) 체결 대기 (단계 {i+1}/3 - {10}초)...")
                conclusion_data = self._wait_for_conclusion_event(target_order_id=current_sell_order_id, expected_flags=['체결', '확인'], timeout=10)
                
                if conclusion_data and conclusion_data['flag'] == '체결':
                    executed_sell_qty += conclusion_data['amount']
                    logger.info(f"매도 주문 ({current_sell_order_id}) 부분 체결 발생. 현재 체결 수량: {executed_sell_qty}")
                    if executed_sell_qty >= order_quantity:
                        logger.info(f"매도 주문 ({current_sell_order_id}) 전량 체결 완료.")
                        sell_order_filled = True
                        break
                elif conclusion_data and conclusion_data['flag'] == '거부':
                    self.fail(f"매도 주문 ({current_sell_order_id})이 거부되었습니다: {conclusion_data['message']}")
                elif conclusion_data and conclusion_data['flag'] == '확인':
                    logger.info(f"매도 주문 ({current_sell_order_id}) 확인 이벤트 수신: {conclusion_data['flag']}")
                else: # 10초 내 미체결
                    logger.info(f"매도 주문 ({current_sell_order_id}) {10}초 내 미체결 또는 원하는 이벤트 미수신. 정정 시도...")
                    if i < 2: # 2차, 1차 호가로 정정
                        # 호가 정보는 이미 구독 중이므로 다시 구독할 필요 없음. 최신 정보 조회
                        # api.subscribe_realtime_bid(test_stock_code) # 제거
                        # time.sleep(5) # 제거
                        price_info_mod_sell = api.get_current_price_and_quotes(test_stock_code)
                        # api.unsubscribe_realtime_bid(test_stock_code) # 제거

                        if not price_info_mod_sell:
                            logger.error("정정 시 호가 정보 조회 실패. 정정 중단.")
                            break

                        # offer_prices의 인덱스를 0, 1, 2 순으로 사용 (3차 -> 2차 -> 1차 매도호가)
                        next_offer_price = price_info_mod_sell['offer_prices'][2-i] if len(price_info_mod_sell['offer_prices']) > (2-i) and price_info_mod_sell['offer_prices'][2-i] > 0 else price_info_mod_sell['current_price']
                        if next_offer_price == 0: next_offer_price = price_info_mod_sell['current_price']
                        next_offer_price = api.round_to_tick(next_offer_price)

                        # 정정 단가가 현재 주문 단가와 동일한지 확인
                        if next_offer_price == current_sell_order_price:
                            logger.info(f"매도 주문 ({current_sell_order_id}) 정정 단가 ({next_offer_price})가 현재 주문 단가 ({current_sell_order_price})와 동일하여 정정 스킵.")
                            continue # 다음 단계로 넘어감

                        logger.info(f"매도 주문 ({current_sell_order_id}) 정정 시도: {3-i}차 매도호가 {next_offer_price:,.0f}원")
                        modify_result_sell = api.send_order(
                            stock_code=test_stock_code,
                            order_type=OrderType.MODIFY,
                            amount=0, # 잔량 정정
                            price=next_offer_price,
                            org_order_num=current_sell_order_id
                        )
                        self.assertEqual(modify_result_sell['status'], 'success', f"매도 정정 주문 실패: {modify_result_sell['message']}")
                        current_sell_order_id = modify_result_sell['order_num'] # 정정 시 새 주문번호 받을 수 있음
                        current_sell_order_price = next_offer_price # 정정 성공 시 가격 업데이트
                        logger.info(f"매도 정정 주문 성공. 새 주문번호: {current_sell_order_id}")
                    else: # 1차 호가까지 정정 후에도 미체결 -> 취소
                        logger.info(f"매도 주문 ({current_sell_order_id}) 1차 호가 정정 후에도 미체결. 취소 시도...")
                        cancel_result = api.send_order(
                            stock_code=test_stock_code,
                            order_type=OrderType.CANCEL,
                            amount=0, # 잔량 취소
                            org_order_num=current_sell_order_id
                        )
                        self.assertEqual(cancel_result['status'], 'success', f"매도 취소 주문 실패: {cancel_result['message']}")
                        logger.info(f"매도 주문 ({current_sell_order_id}) 취소 요청 성공. 체결 이벤트 대기...")
                        # 취소 확인 이벤트 대기
                        conclusion_data = self._wait_for_conclusion_event(target_order_id=current_sell_order_id, expected_flags=['확인'], timeout=10)
                        if conclusion_data and conclusion_data['flag'] == '확인':
                            logger.info(f"매도 주문 ({current_sell_order_id}) 취소 확인 완료.")
                            # --- NEW: Verify cancellation by checking unfilled orders ---
                            logger.info(f"미체결 주문 목록에서 주문번호 {current_sell_order_id} 제거 확인 중...")
                            cancellation_confirmed_in_unfilled = False
                            start_time_unfilled_check = time.time()
                            while time.time() - start_time_unfilled_check < 10: # Wait up to 10 seconds for it to disappear
                                unfilled_orders_after_cancel = api.get_unfilled_orders()
                                found_cancelled_order = False
                                for order in unfilled_orders_after_cancel:
                                    if order['order_id'] == current_sell_order_id:
                                        found_cancelled_order = True
                                        logger.debug(f"주문번호 {current_sell_order_id}가 아직 미체결 목록에 있습니다. 대기 중...")
                                        time.sleep(1)
                                        break
                                if not found_cancelled_order:
                                    cancellation_confirmed_in_unfilled = True
                                    logger.info(f"주문번호 {current_sell_order_id}가 미체결 목록에서 제거됨을 확인했습니다.")
                                    break
                            if not cancellation_confirmed_in_unfilled:
                                logger.warning(f"매도 주문 ({current_sell_order_id})이 미체결 목록에서 제거되지 않았거나 타임아웃 발생.")
                        else:
                            logger.warning(f"매도 주문 ({current_sell_order_id}) 취소 확인 실패 또는 타임아웃.")

                        sell_order_filled = False # 취소되었으므로 체결되지 않음
                        break # 정정/취소 루프 종료
            
            # 매도 주문 처리 루프 종료 후 호가 구독 해지
            api.unsubscribe_realtime_bid(test_stock_code) # <-- 호가 구독 해지 위치 변경

            if not sell_order_filled:
                logger.info("매도 주문이 최종적으로 체결되지 않았습니다. 시장가 매도 시도...")
                time.sleep(5) # 시장가 매도 전 잠시 대기 (API 부하 감소)
                
                # --- 기존 로직: 시장가 매도 전 미체결 매도 주문 확인 및 취소 ---
                # 이 부분은 이미 위에서 취소 확인 로직에 포함되었으므로,
                # 여기서는 추가적인 미체결 주문이 없다고 가정하고 바로 시장가 매도를 시도합니다.
                # 필요하다면 여기에 다시 미체결 주문을 확인하고 취소하는 로직을 추가할 수 있습니다.
                # 현재는 취소 확인 로직이 강화되었으므로, 대부분의 경우 추가적인 미체결 주문은 없을 것입니다.
                
                # 최종 시장가 매도 시 모든 잔량 매도
                logger.info("시장가 매도를 위해 현재 보유 잔량을 다시 조회합니다.")
                current_positions_for_market_sell = api.get_portfolio_positions()
                market_sell_quantity = 0
                for pos in current_positions_for_market_sell:
                    if pos['stock_code'] == test_stock_code:
                        market_sell_quantity = pos['sell_avail_qty']
                        break
                
                if market_sell_quantity == 0:
                    logger.warning(f"종목 [{test_stock_code}]의 매도 가능 잔량이 0이므로 시장가 매도를 스킵합니다.")
                    sell_order_filled = True # 매도할 것이 없으므로 매도 완료로 간주
                else:
                    logger.info(f"시장가 매도 주문 요청 - 종목: {test_stock_code}, 수량: {market_sell_quantity} (모든 잔량)")
                    market_sell_result = api.send_order(
                        stock_code=test_stock_code,
                        order_type=OrderType.SELL,
                        amount=market_sell_quantity, # 모든 잔량 매도
                        price=0, # 시장가
                        order_unit="03" # 시장가 주문
                    )
                    self.assertEqual(market_sell_result['status'], 'success', f"시장가 매도 주문 실패: {market_sell_result['message']}")
                    current_sell_order_id = market_sell_result['order_num']
                    self.assertIsNotNone(current_sell_order_id, "시장가 매도 주문번호가 반환되지 않았습니다.")
                    logger.info(f"시장가 매도 주문번호: {current_sell_order_id}. 시장가 주문은 성공 시 즉시 체결된 것으로 간주합니다.")
                    sell_order_filled = True # 시장가 매도 주문은 성공 시 즉시 체결된 것으로 간주

            # 19. 종목 잔고 확인 (매도 후)
            logger.info("19. 매도 후 종목 잔고 확인...")
            positions_after_sell = api.get_portfolio_positions()
            final_quantity = 0
            for pos in positions_after_sell:
                if pos['stock_code'] == test_stock_code:
                    final_quantity = pos['quantity']
                    break
            # 만약 매도 가능 수량이 전체 보유 수량보다 적어서 일부만 매도되었다면, final_quantity는 0이 아닐 수 있습니다.
            # 테스트의 목표가 '잔고에 남아 있는 모든 수량'을 시장가 매도하는 것이므로, 0이 되는 것을 기대합니다.
            self.assertEqual(final_quantity, 0, f"매도 후 {test_stock_code} 잔고가 0이 아닙니다. 현재 잔고: {final_quantity}주")
            logger.info(f"매도 후 {test_stock_code} 최종 보유 수량: {final_quantity}주")

            # 20. 최종 정리: 현재가 실시간 이벤트 해지
            logger.info("20. 최종 정리: 현재가 실시간 이벤트 해지.")
            api.unsubscribe_realtime_price(test_stock_code)
            logger.info("매매 시나리오 테스트 성공적으로 완료.")

        except Exception as e:
            logger.error(f"테스트 중 예외 발생: {e}", exc_info=True)
            self.fail(f"테스트 실패: {e}") # 예외 발생 시 테스트를 명시적으로 실패 처리

if __name__ == '__main__':
    unittest.main()
