# tests/test_trading_basic_3stocks_advanced.py
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

# 리팩토링된 CreonAPIClient 및 관련 Enum 임포트
# get_next_tick_price는 제외합니다.
from api.creon_api3 import CreonAPIClient, OrderType, OrderStatus

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 테스트 대상 종목 리스트 (휴림로봇, 동양철관, 우리기술)
TEST_STOCK_CODES = ['A090710', 'A008970', 'A032820']
TEST_ORDER_QUANTITY = 1 # 테스트용 최소 수량

class TestCreonAPIScenario(unittest.TestCase):
    """
    세 종목을 동시에 테스트하는 고급 매매 시나리오 테스트
    """
    cls_api: CreonAPIClient = None
    
    # 멀티 종목 지원을 위해 딕셔너리로 변경
    _conclusion_events_queue: queue.Queue = None 
    
    # 종목별 실시간 이벤트 데이터 및 수신 신호 딕셔너리
    _price_event_data: Dict[str, Dict[str, Any]] = {}
    _price_event_received: Dict[str, Event] = {}
    _bid_event_data: Dict[str, Dict[str, Any]] = {}
    _bid_event_received: Dict[str, Event] = {}
    
    # 콜백 데이터에 대한 동시성 제어를 위한 락
    _callback_lock = Lock()

    # 로그 중복 방지를 위한 변수 추가
    _last_logged_conclusion_event: Dict[str, Any] = {}
    _last_logged_conclusion_time: float = 0.0

    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행: CreonAPIClient 초기화"""
        logger.info("--- 테스트 클래스 설정 시작 ---")
        try:
            pythoncom.CoInitialize()
            cls.cls_api = CreonAPIClient()
            cls._conclusion_events_queue = queue.Queue()
            
            # 종목별 이벤트 객체 초기화
            for code in TEST_STOCK_CODES:
                cls._price_event_received[code] = Event()
                cls._bid_event_received[code] = Event()
                cls._price_event_data[code] = None
                cls._bid_event_data[code] = None

            # 콜백 함수 등록
            cls.cls_api.set_conclusion_callback(cls._conclusion_callback)
            cls.cls_api.set_price_update_callback(cls._price_update_callback)
            cls.cls_api.set_bid_update_callback(cls._bid_update_callback)
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
            cls.cls_api.cleanup() 
        pythoncom.CoUninitialize()
        logger.info("--- 테스트 클래스 정리 완료 ---")

    def setUp(self):
        """각 테스트 메서드 실행 전에 실행"""
        self.assertIsNotNone(self.cls_api, "CreonAPIClient가 초기화되지 않았습니다. setUpClass를 확인하세요.")
        self.assertTrue(self.cls_api.is_connected(), "크레온 PLUS가 연결되어 있지 않습니다. HTS 로그인 상태를 확인하세요.")
        
        # 각 테스트 시작 시 이벤트 플래그 및 데이터 초기화
        with TestCreonAPIScenario._callback_lock:
            while not TestCreonAPIScenario._conclusion_events_queue.empty():
                try:
                    TestCreonAPIScenario._conclusion_events_queue.get_nowait()
                    TestCreonAPIScenario._conclusion_events_queue.task_done()
                except queue.Empty:
                    pass
            
            # 종목별 이벤트 플래그 및 데이터 초기화
            for code in TEST_STOCK_CODES:
                TestCreonAPIScenario._price_event_received[code].clear()
                TestCreonAPIScenario._bid_event_received[code].clear()
                TestCreonAPIScenario._price_event_data[code] = None
                TestCreonAPIScenario._bid_event_data[code] = None

            # 로그 중복 방지 변수 초기화
            TestCreonAPIScenario._last_logged_conclusion_event = {}
            TestCreonAPIScenario._last_logged_conclusion_time = 0.0

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
            current_time = time.time()
            # 간단한 로그 중복 방지
            is_duplicate_log = False
            if cls._last_logged_conclusion_event and \
               cls._last_logged_conclusion_event.get('order_num') == data.get('order_num') and \
               cls._last_logged_conclusion_event.get('flag') == data.get('flag') and \
               (current_time - cls._last_logged_conclusion_time < 0.1):
                is_duplicate_log = True

            if not is_duplicate_log:
                logger.info(f"[CpEvent] 주문 체결/응답 수신: {data.get('flag')} {data.get('buy_sell')} 종목:{data.get('code')} 가격:{data.get('price'):,.0f} 수량:{data.get('amount')} 주문번호:{data.get('order_num')} 잔고:{data.get('balance')}")
                cls._last_logged_conclusion_event = data.copy()
                cls._last_logged_conclusion_time = current_time
            
            cls._conclusion_events_queue.put(data)

    @classmethod
    def _price_update_callback(cls, stock_code: str, current_price: int, timestamp: float):
        """실시간 현재가 업데이트 콜백 핸들러"""
        with cls._callback_lock:
            if stock_code not in cls.cls_api._last_price_print_time_per_stock:
                cls.cls_api._last_price_print_time_per_stock[stock_code] = 0.0

            if time.time() - cls.cls_api._last_price_print_time_per_stock[stock_code] >= 5:
                logger.info(f"실시간 현재가 수신: 종목={stock_code}, 가격={current_price:,.0f}원")
                cls.cls_api._last_price_print_time_per_stock[stock_code] = time.time()
            
            # 종목별 데이터 저장 및 이벤트 신호 설정
            cls._price_event_data[stock_code] = {'stock_code': stock_code, 'current_price': current_price, 'timestamp': timestamp}
            if stock_code in cls._price_event_received:
                cls._price_event_received[stock_code].set() 

    @classmethod
    def _bid_update_callback(cls, stock_code: str, offer_prices: List[int], bid_prices: List[int], offer_amounts: List[int], bid_amounts: List[int]):
        """실시간 10차 호가 업데이트 콜백 핸들러"""
        with cls._callback_lock:
            logger.debug(f"실시간 호가 수신: 종목={stock_code}, 1차 매도={offer_prices[0]}, 1차 매수={bid_prices[0]}")
            
            # 종목별 데이터 저장 및 이벤트 신호 설정
            cls._bid_event_data[stock_code] = {
                'stock_code': stock_code,
                'offer_prices': offer_prices,
                'bid_prices': bid_prices,
                'offer_amounts': offer_amounts,
                'bid_amounts': bid_amounts,
                'timestamp': time.time()
            }
            if stock_code in cls._bid_event_received:
                cls._bid_event_received[stock_code].set()

    def _wait_for_conclusion_event(self, target_order_id: Optional[int], expected_flags: List[str], timeout: int = 60) -> Optional[Dict[str, Any]]:
        """
        특정 주문 ID에 대한 체결/응답 이벤트를 기다립니다.
        _conclusion_events_queue를 모니터링합니다.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            pythoncom.PumpWaitingMessages() # 대기 중인 COM 메시지 처리
            try:
                # 큐에서 이벤트를 가져오되, 타임아웃을 설정하여 무한 대기 방지
                data = TestCreonAPIScenario._conclusion_events_queue.get(timeout=0.1) 
                
                with TestCreonAPIScenario._callback_lock:
                    if data.get('order_num') == target_order_id:
                        if data.get('flag') in expected_flags:
                            logger.info(f"대상 체결/응답 이벤트 수신: {data}")
                            TestCreonAPIScenario._conclusion_events_queue.task_done()
                            return data
                        else:
                            logger.debug(f"대상 주문 ({target_order_id})의 다른 플래그 ({data.get('flag')}) 이벤트 수신. 기대 플래그 {expected_flags} 대기 계속...")
                            TestCreonAPIScenario._conclusion_events_queue.task_done()
                    else:
                        # 다른 주문의 이벤트이므로 다시 큐에 넣고 다음 이벤트를 기다림
                        TestCreonAPIScenario._conclusion_events_queue.put(data)
                        TestCreonAPIScenario._conclusion_events_queue.task_done()
                        
            except queue.Empty:
                pass
            
        logger.warning(f"체결/응답 이벤트 타임아웃 ({timeout}초) 또는 대상 주문 ({target_order_id}) 이벤트 미수신.")
        return None

    def _wait_for_price_event(self, stock_code: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """특정 종목의 현재가 업데이트 이벤트를 기다립니다."""
        event_obj = TestCreonAPIScenario._price_event_received.get(stock_code)
        if not event_obj:
            logger.error(f"종목 {stock_code}에 대한 가격 이벤트 객체를 찾을 수 없습니다.")
            return None

        event_obj.clear()
        logger.debug(f"[{stock_code}] 가격 이벤트 대기 중 (최대 {timeout}초)...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            pythoncom.PumpWaitingMessages()
            if event_obj.wait(0.1):
                with TestCreonAPIScenario._callback_lock:
                    return TestCreonAPIScenario._price_event_data.get(stock_code)
        logger.warning(f"[{stock_code}] 가격 이벤트 타임아웃 ({timeout}초).")
        return None

    def _wait_for_bid_event(self, stock_code: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """특정 종목의 호가 업데이트 이벤트를 기다립니다."""
        event_obj = TestCreonAPIScenario._bid_event_received.get(stock_code)
        if not event_obj:
            logger.error(f"종목 {stock_code}에 대한 호가 이벤트 객체를 찾을 수 없습니다.")
            return None

        event_obj.clear()
        logger.debug(f"[{stock_code}] 호가 이벤트 대기 중 (최대 {timeout}초)...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            pythoncom.PumpWaitingMessages()
            if event_obj.wait(0.1):
                with TestCreonAPIScenario._callback_lock:
                    return TestCreonAPIScenario._bid_event_data.get(stock_code)
        logger.warning(f"[{stock_code}] 호가 이벤트 타임아웃 ({timeout}초).")
        return None

    # 미체결 주문 취소 로직 수정
    def _cancel_unexecuted_orders(self, stock_code: str, buy_sell: Optional[str] = None) -> bool:
        """
        특정 종목의 미체결 주문을 조회하고 취소합니다.
        
        get_unexecuted_orders가 없으므로, get_unfilled_orders를 사용해 전체 미체결을 조회하고 필터링합니다.
        """
        api = self.cls_api
        logger.info(f"[{stock_code}] 미체결 주문 조회 및 취소 시도...")
        
        # 1. CreonAPIClient의 get_unfilled_orders()를 호출하여 전체 미체결 주문을 가져옵니다.
        #    이 메서드는 creon_api2.py에 정의되어 있습니다.
        all_unfilled_orders = api.get_unfilled_orders()
        
        # 2. stock_code가 일치하는 미체결 주문만 필터링합니다.
        unexecuted_orders_for_stock = [
            order for order in all_unfilled_orders 
            if order.get('stock_code') == stock_code
        ]

        if not unexecuted_orders_for_stock:
            logger.info(f"[{stock_code}] 미체결 주문이 없습니다.")
            return True

        logger.info(f"[{stock_code}] 총 {len(unexecuted_orders_for_stock)}개의 미체결 주문을 찾았습니다. 취소를 진행합니다.")
        
        success_count = 0
        
        for order in unexecuted_orders_for_stock:
            order_num = order['order_num']
            order_type = order['buy_sell'] # '매수' or '매도'
            
            # 지정된 매수/매도 유형이 있으면 해당 유형만 취소 (예: 매수 시나리오 후에는 매수 미체결만 취소)
            if buy_sell and order_type != buy_sell:
                logger.info(f"[{stock_code}] 주문 {order_num} ({order_type})는 대상이 아니므로 스킵합니다. (요청 유형: {buy_sell})")
                continue

            logger.info(f"[{stock_code}] 미체결 주문 {order_num} ({order_type}) 취소 요청...")
            
            cancel_result = api.send_order(
                stock_code=stock_code,
                order_type=OrderType.CANCEL,
                amount=0,
                org_order_num=order_num
            )
            
            if cancel_result['status'] == 'success':
                logger.info(f"[{stock_code}] 주문 {order_num} 취소 요청 성공.")
                # 취소 확인 이벤트 대기
                conclusion_data = self._wait_for_conclusion_event(target_order_id=order_num, expected_flags=['확인'], timeout=10)
                if conclusion_data and conclusion_data['flag'] == '확인':
                    success_count += 1
                else:
                    logger.error(f"[{stock_code}] 주문 {order_num} 취소 확인 실패 (이벤트 미수신).")
                    return False # 취소 확인 실패 시 전체 취소 실패로 간주
            else:
                logger.error(f"[{stock_code}] 주문 {order_num} 취소 실패: {cancel_result['message']}")
                # 취소 요청 실패 시 전체 취소 실패로 간주
                return False

        logger.info(f"[{stock_code}] 총 {success_count}개의 미체결 주문 취소 완료.")
        return True

    def _execute_trading_scenario(self, stock_code: str, order_quantity: int):
        """
        단일 종목에 대한 매수 -> 정정/취소 -> 시장가 매수 -> 매도 시나리오 실행
        이 메서드는 멀티 종목 테스트 시나리오에서 호출됩니다.
        """
        api = self.cls_api
        logger.info(f"\n--- 종목 [{stock_code}] 매수 시나리오 시작 ---")
        
        # 1. 예수금 확인 (전체 계좌 기준)
        account_balance = api.get_account_balance()
        self.assertIsNotNone(account_balance, "계좌 잔고 조회 실패")
        current_cash = account_balance.get('cash_balance', 0)
        logger.info(f"[{stock_code}] 현재 주문 가능 현금: {current_cash:,.0f}원")
        
        # 2. 호가 조회 및 초기 매수 주문 (매수 3호가)
        logger.info(f"[{stock_code}] 호가 수신 이벤트 등록 (주문 직전)")
        api.subscribe_realtime_bid(stock_code) 
        self._wait_for_bid_event(stock_code, timeout=10)
        
        price_info = api.get_current_price_and_quotes(stock_code)
        self.assertIsNotNone(price_info, f"[{stock_code}] 현재가 및 호가 조회 실패")
        
        current_price = price_info['current_price']
        bid_prices = price_info['bid_prices']
        
        initial_buy_price = bid_prices[2] if len(bid_prices) >= 3 and bid_prices[2] > 0 else current_price 
        initial_buy_price = api.round_to_tick(initial_buy_price)
        
        logger.info(f"[{stock_code}] 현재가: {current_price:,.0f}원, 초기 매수 희망가격 (3차 매수호가): {initial_buy_price:,.0f}원")

        buy_order_result = api.send_order(
            stock_code=stock_code,
            order_type=OrderType.BUY,
            amount=order_quantity,
            price=initial_buy_price,
            order_unit="01" # 보통가
        )
        current_order_id = buy_order_result['order_num']
        current_buy_order_price = initial_buy_price
        
        # 호가 구독 해제 (주문 직후)
        logger.info(f"[{stock_code}] 호가 구독 해제 (주문 직후)")
        api.unsubscribe_realtime_bid(stock_code)

        if buy_order_result['status'] != 'success':
            logger.error(f"[{stock_code}] 초기 매수 주문 실패: {buy_order_result['message']}")
            return False 

        logger.info(f"[{stock_code}] 초기 매수 주문번호: {current_order_id}")
        order_filled = False
        executed_buy_qty = 0

        # 정정 시나리오 (3차 -> 2차 -> 1차)
        for i in range(3):
            if order_filled:
                break

            logger.info(f"[{stock_code}] 매수 주문 ({current_order_id}) 체결 대기 (단계 {i+1}/3 - 10초)...")
            conclusion_data = self._wait_for_conclusion_event(target_order_id=current_order_id, expected_flags=['체결', '확인'], timeout=10)
            
            if conclusion_data and conclusion_data['flag'] == '체결':
                executed_buy_qty += conclusion_data['amount']
                logger.info(f"[{stock_code}] 매수 주문 ({current_order_id}) 부분 체결 발생. 현재 체결 수량: {executed_buy_qty}")
                if executed_buy_qty >= order_quantity:
                    logger.info(f"[{stock_code}] 매수 주문 ({current_order_id}) 전량 체결 완료.")
                    order_filled = True
                    break
            elif conclusion_data and conclusion_data['flag'] == '거부':
                logger.error(f"[{stock_code}] 매수 주문 거부: {conclusion_data['message']}")
                return False
            elif conclusion_data and conclusion_data['flag'] == '확인':
                logger.info(f"[{stock_code}] 매수 주문 ({current_order_id}) 확인 이벤트 수신.")
            else: # 10초 내 미체결
                logger.info(f"[{stock_code}] 매수 주문 ({current_order_id}) 10초 내 미체결. 정정 시도...")
                if i < 2:
                    # 정정 전 호가 데이터 최신화 대기
                    api.subscribe_realtime_bid(stock_code)
                    self._wait_for_bid_event(stock_code, timeout=10)
                    price_info_mod = api.get_current_price_and_quotes(stock_code)
                    
                    if not price_info_mod:
                        logger.error(f"[{stock_code}] 정정 시 호가 정보 조회 실패. 정정 중단.")
                        break

                    # 2차 호가 -> 1차 호가
                    next_bid_price = price_info_mod['bid_prices'][2-i] if len(price_info_mod['bid_prices']) > (2-i) and price_info_mod['bid_prices'][2-i] > 0 else price_info_mod['current_price']
                    next_bid_price = api.round_to_tick(next_bid_price)

                    if next_bid_price != current_buy_order_price:
                        logger.info(f"[{stock_code}] 매수 주문 ({current_order_id}) 정정 시도: {3-i}차 매수호가 {next_bid_price:,.0f}원")
                        modify_result = api.send_order(
                            stock_code=stock_code,
                            order_type=OrderType.MODIFY,
                            amount=0, 
                            price=next_bid_price,
                            org_order_num=current_order_id
                        )
                        # 정정 주문 후 호가 구독 해제
                        api.unsubscribe_realtime_bid(stock_code)
                        
                        if modify_result['status'] == 'success':
                            current_order_id = modify_result['order_num']
                            current_buy_order_price = next_bid_price
                            logger.info(f"[{stock_code}] 매수 정정 주문 성공. 새 주문번호: {current_order_id}")
                        else:
                            logger.error(f"[{stock_code}] 매수 정정 주문 실패: {modify_result['message']}. 시나리오 중단.")
                            return False
                    else:
                        logger.info(f"[{stock_code}] 매수 정정 단가 동일. 정정 스킵.")
                else: # 1차 호가까지 정정 후에도 미체결 -> 취소
                    logger.info(f"[{stock_code}] 매수 주문 ({current_order_id}) 취소 시도...")
                    cancel_result = api.send_order(
                        stock_code=stock_code,
                        order_type=OrderType.CANCEL,
                        amount=0,
                        org_order_num=current_order_id
                    )
                    if cancel_result['status'] == 'success':
                        logger.info(f"[{stock_code}] 매수 주문 ({current_order_id}) 취소 요청 성공.")
                        # 취소 확인 이벤트 대기
                        self._wait_for_conclusion_event(target_order_id=current_order_id, expected_flags=['확인'], timeout=10)
                        order_filled = False 
                    else:
                        logger.error(f"[{stock_code}] 매수 취소 주문 실패: {cancel_result['message']}")
                        return False
                    break # 정정/취소 루프 종료

        # 매수 주문 미체결 시 시장가 매수 시도 (취소 + 시장가)
        if not order_filled:
            logger.info(f"[{stock_code}] 매수 주문이 최종적으로 체결되지 않았습니다. 미체결 주문 취소 후 시장가 매수 시도...")
            
            # 미체결 주문 취소 (성공해야만 시장가 주문 진행)
            if not self._cancel_unexecuted_orders(stock_code, buy_sell='매수'):
                logger.error(f"[{stock_code}] 시장가 매수 전 미체결 주문 취소 실패.")
                return False

            time.sleep(5) 
            pythoncom.PumpWaitingMessages() 
            
            market_buy_result = api.send_order(
                stock_code=stock_code,
                order_type=OrderType.BUY,
                amount=order_quantity,
                price=0, 
                order_unit="03" # 시장가 주문
            )
            if market_buy_result['status'] != 'success':
                logger.error(f"[{stock_code}] 시장가 매수 주문 실패: {market_buy_result['message']}")
                return False
            else:
                order_filled = True
                logger.info(f"[{stock_code}] 시장가 매수 주문 성공. 주문번호: {market_buy_result['order_num']}")

        # 매수 시나리오 완료 후 잔고 확인 (간략화)
        positions_after_buy = api.get_portfolio_positions()
        found_position = any(pos['stock_code'] == stock_code for pos in positions_after_buy)
        if not found_position:
            logger.error(f"[{stock_code}] 매수 후 포지션 확인 실패.")
            return False
        
        # --- 매도 시나리오 시작 ---
        logger.info(f"\n--- 종목 [{stock_code}] 매도 시나리오 시작 ---")

        # 3. 매도 가능 수량 확인 (get_portfolio_positions()를 통해 최신 정보 획득)
        positions_before_sell = api.get_portfolio_positions()
        sellable_quantity = 0
        total_bought_in_test = 0 
        for pos in positions_before_sell:
            if pos['stock_code'] == stock_code:
                sellable_quantity = pos['sell_avail_qty']
                total_bought_in_test = pos['quantity'] 
                break

        if sellable_quantity == 0:
            logger.warning(f"[{stock_code}] 매도 가능 수량(0)이 없어 매도 시나리오를 스킵합니다.")
            
            # 매도 시나리오를 스킵했지만, 잔고가 0이 아니라면 테스트 실패로 간주
            if total_bought_in_test > 0:
                 logger.error(f"[{stock_code}] 매도 가능 수량은 0이나, 총 보유 수량은 {total_bought_in_test}주입니다. 잔고 0 종료 요구사항 미충족.")
                 return False

            return True # 매도 가능 수량이 없고 총 보유 수량도 0인 경우, 시나리오 성공으로 간주

        # 4. 호가 조회 및 초기 매도 주문 (매도 3호가)
        logger.info(f"[{stock_code}] 호가 수신 이벤트 등록 (주문 직전)")
        api.subscribe_realtime_bid(stock_code)
        self._wait_for_bid_event(stock_code, timeout=10)

        price_info_sell = api.get_current_price_and_quotes(stock_code)
        api.unsubscribe_realtime_bid(stock_code) # 호가 구독 해제

        self.assertIsNotNone(price_info_sell, f"[{stock_code}] 매도 시나리오를 위한 호가 조회 실패")
        
        current_price_sell = price_info_sell['current_price']
        offer_prices = price_info_sell['offer_prices']
        
        initial_sell_price = offer_prices[2] if len(offer_prices) >= 3 and offer_prices[2] > 0 else current_price_sell
        initial_sell_price = api.round_to_tick(initial_sell_price)
        
        logger.info(f"[{stock_code}] 초기 매도 희망가격 (3차 매도호가): {initial_sell_price:,.0f}원")

        sell_order_result = api.send_order(
            stock_code=stock_code,
            order_type=OrderType.SELL,
            amount=sellable_quantity, # 매도 가능 수량 전체 매도 시도
            price=initial_sell_price,
            order_unit="01" 
        )
        current_sell_order_id = sell_order_result['order_num']
        current_sell_order_price = initial_sell_price
        
        if sell_order_result['status'] != 'success':
            logger.error(f"[{stock_code}] 초기 매도 주문 실패: {sell_order_result['message']}")
            return False

        logger.info(f"[{stock_code}] 초기 매도 주문번호: {current_sell_order_id}")
        sell_order_filled = False
        executed_sell_qty_in_loop = 0

        # 정정 시나리오 (3차 -> 2차 -> 1차)
        for i in range(3):
            if sell_order_filled:
                break
            
            conclusion_data = self._wait_for_conclusion_event(target_order_id=current_sell_order_id, expected_flags=['체결', '확인'], timeout=10)
            
            if conclusion_data and conclusion_data['flag'] == '체결':
                executed_sell_qty_in_loop += conclusion_data['amount']
                if executed_sell_qty_in_loop >= sellable_quantity:
                    logger.info(f"[{stock_code}] 매도 주문 전량 체결 완료.")
                    sell_order_filled = True
                    break
            elif conclusion_data and conclusion_data['flag'] == '확인':
                logger.info(f"[{stock_code}] 매도 주문 확인 이벤트 수신.")
            else:
                logger.info(f"[{stock_code}] 매도 주문 미체결. 정정 시도...")
                if i < 2:
                    # 정정 전 호가 데이터 최신화 대기 및 구독 해제
                    api.subscribe_realtime_bid(stock_code)
                    self._wait_for_bid_event(stock_code, timeout=10)
                    price_info_mod_sell = api.get_current_price_and_quotes(stock_code)
                    api.unsubscribe_realtime_bid(stock_code)

                    if not price_info_mod_sell:
                        logger.error(f"[{stock_code}] 정정 시 호가 정보 조회 실패. 정정 중단.")
                        break

                    # 2차 호가 -> 1차 호가
                    next_offer_price = price_info_mod_sell['offer_prices'][2-i] if len(price_info_mod_sell['offer_prices']) > (2-i) and price_info_mod_sell['offer_prices'][2-i] > 0 else price_info_mod_sell['current_price']
                    next_offer_price = api.round_to_tick(next_offer_price)

                    if next_offer_price != current_sell_order_price:
                        logger.info(f"[{stock_code}] 매도 주문 ({current_sell_order_id}) 정정 시도: {3-i}차 매도호가 {next_offer_price:,.0f}원")
                        modify_result_sell = api.send_order(
                            stock_code=stock_code,
                            order_type=OrderType.MODIFY,
                            amount=0,
                            price=next_offer_price,
                            org_order_num=current_sell_order_id
                        )
                        if modify_result_sell['status'] == 'success':
                            current_sell_order_id = modify_result_sell['order_num']
                            current_sell_order_price = next_offer_price
                            logger.info(f"[{stock_code}] 매도 정정 주문 성공.")
                        else:
                            logger.error(f"[{stock_code}] 매도 정정 주문 실패: {modify_result_sell['message']}")
                            return False
                    else:
                        logger.info(f"[{stock_code}] 매도 정정 단가 동일. 정정 스킵.")
                else: # 1차 호가까지 정정 후에도 미체결 -> 취소
                    logger.info(f"[{stock_code}] 매도 주문 ({current_sell_order_id}) 취소 시도...")
                    cancel_result = api.send_order(
                        stock_code=stock_code,
                        order_type=OrderType.CANCEL,
                        amount=0,
                        org_order_num=current_sell_order_id
                    )
                    if cancel_result['status'] == 'success':
                        logger.info(f"[{stock_code}] 매도 주문 ({current_sell_order_id}) 취소 요청 성공.")
                        self._wait_for_conclusion_event(target_order_id=current_sell_order_id, expected_flags=['확인'], timeout=10)
                        sell_order_filled = False 
                    else:
                        logger.error(f"[{stock_code}] 매도 취소 주문 실패: {cancel_result['message']}")
                        return False
                    break # 정정/취소 루프 종료

        # 매도 주문 미체결 시 시장가 매도 시도 (취소 + 시장가)
        if not sell_order_filled:
            logger.info(f"[{stock_code}] 매도 주문이 최종적으로 체결되지 않았습니다. 미체결 주문 취소 후 시장가 매도 시도...")
            
            # 미체결 주문 취소 (성공해야만 시장가 주문 진행)
            if not self._cancel_unexecuted_orders(stock_code, buy_sell='매도'):
                logger.error(f"[{stock_code}] 시장가 매도 전 미체결 주문 취소 실패.")
                return False
            
            time.sleep(5)
            pythoncom.PumpWaitingMessages() 
            
            # 매도 가능 잔량 재확인 (미체결 취소로 인해 잔량이 증가했을 수 있음)
            current_positions_for_market_sell = api.get_portfolio_positions()
            market_sell_quantity = 0
            for pos in current_positions_for_market_sell:
                if pos['stock_code'] == stock_code:
                    market_sell_quantity = pos['sell_avail_qty']
                    break

            if market_sell_quantity > 0:
                logger.info(f"[{stock_code}] 시장가 매도 주문 요청 - 수량: {market_sell_quantity} (매도 가능 잔량)")
                market_sell_result = api.send_order(
                    stock_code=stock_code,
                    order_type=OrderType.SELL,
                    amount=market_sell_quantity,
                    price=0,
                    order_unit="03" # 시장가 주문
                )
                if market_sell_result['status'] == 'success':
                    logger.info(f"[{stock_code}] 시장가 매도 주문 성공.")
                    sell_order_filled = True
                else:
                    logger.error(f"[{stock_code}] 시장가 매도 주문 실패: {market_sell_result['message']}")
                    return False
            else:
                logger.warning(f"[{stock_code}] 시장가 매도 시도 시점에 매도 가능 수량이 0이었습니다. 시장가 매도 스킵.")
                
        # 최종 잔고 확인 및 요구사항 충족 여부 반환
        positions_after_sell = api.get_portfolio_positions()
        final_quantity = 0
        for pos in positions_after_sell:
            if pos['stock_code'] == stock_code:
                final_quantity = pos['quantity']
                break
        
        logger.info(f"[{stock_code}] 최종 보유 수량: {final_quantity}주")
        
        # 잔고가 0이면 해당 종목의 현재가 구독 취소 (요구사항 1.) 및 성공 반환
        if final_quantity == 0:
            logger.info(f"[{stock_code}] 잔고가 0이므로 현재가 실시간 이벤트 해지.")
            api.unsubscribe_realtime_price(stock_code)
            return True 
        else:
            # 잔고가 0이 아니면 실패로 간주 (요구사항: 잔고 0 상태에서 종료)
            logger.error(f"[{stock_code}] 최종 보유 수량이 0이 아닙니다. 매도 실패.")
            return False

    def test_multi_stock_trading_scenario(self):
        """
        세 종목에 대한 동시 매매 시나리오 테스트를 실행하고,
        실시간 현재가 구독/해제 및 개별 종목 매매 시나리오를 처리합니다.
        """
        api = self.cls_api
        
        # 1. 초기 현재가 순차 구독 (요구사항 1.)
        logger.info("1. 실시간 현재가 순차 구독 시작...")
        
        for stock_code in TEST_STOCK_CODES:
            logger.info(f"[{stock_code}] 현재가 구독...")
            api.subscribe_realtime_price(stock_code)
            
            # 구독 완료 후 해당 종목의 현재가 데이터 수신 확인
            price_data = self._wait_for_price_event(stock_code, timeout=10)
            self.assertIsNotNone(price_data, f"[{stock_code}] 현재가 데이터 수신 실패.")
            
            # 다른 종목의 데이터도 정상적으로 수신되는지 확인 (선택 사항)
            # 이 부분은 콜백에서 처리되므로, 데이터가 정상적으로 저장되는지만 확인.
            logger.info(f"[{stock_code}] 현재가 데이터 수신 확인 완료. 가격: {price_data['current_price']}")

        logger.info("모든 종목의 현재가 구독 완료.")

        # 2. 매매 시나리오 동시 진행
        logger.info("\n2. 매매 시나리오 동시 진행 시작...")

        # 각 종목의 테스트 결과를 저장할 딕셔너리
        test_results = {}
        
        # --- A, B, C 매수 3호가 주문 동시 전송 ---
        # 실제 동시 처리는 _execute_trading_scenario 내부의 메시지 펌핑과 이벤트 처리에 의존함
        
        # 각 종목에 대한 시나리오를 실행 (unittest 환경이므로 순차적으로 호출하지만,
        # 내부 로직은 주문 체결 이벤트를 기다리며 COM 메시지 펌핑을 통해 동시성 흉내)

        for stock_code in TEST_STOCK_CODES:
            # 여기서는 순차적으로 시나리오를 시작하지만, 내부적으로 이벤트 기반으로 작동
            logger.info(f"\n[시나리오 실행] 종목 {stock_code} 매매 시나리오 시작...")
            try:
                result = self._execute_trading_scenario(stock_code, TEST_ORDER_QUANTITY)
                test_results[stock_code] = result
            except Exception as e:
                logger.error(f"종목 {stock_code} 시나리오 실행 중 예외 발생: {e}")
                test_results[stock_code] = False
        
        # 3. 최종 결과 확인
        logger.info("\n3. 최종 결과 확인...")
        all_passed = all(test_results.values())
        
        if all_passed:
            logger.info("모든 종목의 매매 시나리오 테스트가 성공적으로 완료되었습니다.")
        else:
            logger.error("일부 종목의 매매 시나리오 테스트가 실패했습니다.")
            for stock, result in test_results.items():
                if not result:
                    logger.error(f"종목 {stock} 테스트 실패.")
            self.fail("멀티 종목 매매 시나리오 테스트 실패. 모든 종목의 잔고가 0이 되어야 합니다.")

        # 테스트 종료 시 tearDown에서 전체 구독 해지가 실행됨
        
if __name__ == '__main__':
    unittest.main()