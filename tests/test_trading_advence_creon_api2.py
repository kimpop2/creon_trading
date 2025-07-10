import unittest
import sys
import os
import logging
import time
import queue
import threading
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Callable, Tuple

# win32com 관련 CoInitialize/CoUninitialize 임포트
import pythoncom

# 프로젝트 루트 경로 추가
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
    - 여러 종목에 대한 큐/워커 기반 동시성 처리 적용
    """
    cls_api: CreonAPIClient = None
    
    # 테스트 대상 종목 리스트
    TEST_STOCK_CODES = ['A090710', 'A008970', 'A032820'] # 휴림로봇, 동양철관, 우리기술
    ORDER_QUANTITY = 1 # 테스트용 최소 수량

    # 실시간 이벤트 핸들링 변수 (클래스 변수로 선언하여 콜백에서 접근)
    _conclusion_events_queue: queue.Queue = None # 체결/응답 이벤트를 큐로 관리
    _price_event_data_per_stock: Dict[str, Dict[str, Any]] = {} # 종목별 최신 현재가 데이터
    _price_event_received_per_stock: Dict[str, threading.Event] = {} # 종목별 현재가 이벤트 수신 신호

    # 매매 신호 큐 및 워커 스레드
    _buy_signal_queue: queue.Queue = None # (stock_code, quantity) 튜플 저장
    _sell_signal_queue: queue.Queue = None # (stock_code, quantity) 튜플 저장
    _buy_worker_thread: threading.Thread = None
    _sell_worker_thread: threading.Thread = None

    # 종목별 현재 주문 상태 및 시나리오 진행 상태 추적
    _stock_scenario_states: Dict[str, str] = {} # 'INITIAL', 'BUY_PENDING', 'BOUGHT', 'SELL_PENDING', 'SOLD'
    _stock_order_info: Dict[str, Dict[str, Any]] = {} # {stock_code: {'order_id': int, 'order_price': int, 'order_type': OrderType}}
    _stock_scenario_events: Dict[str, threading.Event] = {} # 종목별 시나리오 진행 동기화 이벤트

    # 콜백 데이터에 대한 동시성 제어를 위한 락
    _callback_lock = threading.Lock()

    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행: CreonAPIClient 초기화 및 큐/워커 설정"""
        logger.info("--- 테스트 클래스 설정 시작 ---")
        try:
            cls.cls_api = CreonAPIClient()
            
            # 큐 초기화
            cls._conclusion_events_queue = queue.Queue()
            cls._buy_signal_queue = queue.Queue()
            cls._sell_signal_queue = queue.Queue()

            # 종목별 상태 및 이벤트 초기화
            for code in cls.TEST_STOCK_CODES:
                cls._stock_scenario_states[code] = 'INITIAL'
                cls._stock_order_info[code] = {'order_id': None, 'order_price': 0, 'order_type': None}
                cls._stock_scenario_events[code] = threading.Event()
                cls._price_event_received_per_stock[code] = threading.Event()
                cls._price_event_data_per_stock[code] = None

            # 콜백 함수 등록
            cls.cls_api.set_conclusion_callback(cls._conclusion_callback)
            cls.cls_api.set_price_update_callback(cls._price_update_callback)
            cls.cls_api.set_bid_update_callback(cls._bid_update_callback) # 호가 콜백도 등록
            logger.info("CreonAPIClient 초기화 및 콜백 등록 완료.")

            # 워커 스레드 시작
            # 각 워커 스레드는 내부에서 CoInitialize/CoUninitialize를 호출해야 함
            cls._buy_worker_thread = threading.Thread(target=cls._buy_worker_task, daemon=True)
            cls._sell_worker_thread = threading.Thread(target=cls._sell_worker_task, daemon=True)
            
            cls._buy_worker_thread.start()
            cls._sell_worker_thread.start()
            logger.info("매수/매도 워커 스레드 시작 완료.")

        except Exception as e:
            logger.error(f"CreonAPIClient 초기화 실패: {e}", exc_info=True)
            cls.cls_api = None
            raise
        logger.info("--- 테스트 클래스 설정 완료 ---")

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시 한 번만 실행: 리소스 정리 및 워커 스레드 종료"""
        logger.info("--- 테스트 클래스 정리 시작 ---")
        if cls.cls_api:
            # 워커 스레드 종료 신호 전송
            # 각 큐에 None을 넣어 워커 스레드가 종료되도록 함 (워커 스레드 수만큼)
            cls._buy_signal_queue.put((None, None))
            cls._sell_signal_queue.put((None, None))
            
            cls._buy_worker_thread.join(timeout=30) # 워커 스레드 종료 대기
            cls._sell_worker_thread.join(timeout=30)
            
            if cls._buy_worker_thread.is_alive():
                logger.warning("매수 워커 스레드가 시간 내에 종료되지 않았습니다.")
            if cls._sell_worker_thread.is_alive():
                logger.warning("매도 워커 스레드가 시간 내에 종료되지 않았습니다.")

            cls.cls_api.cleanup() # CreonAPIClient의 cleanup 메서드 호출 (모든 구독 해지 포함)
        logger.info("--- 테스트 클래스 정리 완료 ---")

    def setUp(self):
        """각 테스트 메서드 실행 전에 실행"""
        self.assertIsNotNone(self.cls_api, "CreonAPIClient가 초기화되지 않았습니다. setUpClass를 확인하세요.")
        self.assertTrue(self.cls_api.is_connected(), "크레온 PLUS가 연결되어 있지 않습니다. HTS 로그인 상태를 확인하세요.")
        
        # _conclusion_events_queue는 콜백에서 처리되므로 여기서 초기화 불필요
        # _price_event_received_per_stock는 각 시나리오에서 clear/set
        
        logger.info(f"\n--- {self._testMethodName} 테스트 시작 ---")

    def tearDown(self):
        """각 테스트 메서드 실행 후에 실행"""
        logger.info(f"--- {self._testMethodName} 테스트 종료 ---\n")
        # 테스트 종료 시 혹시 모를 잔여 구독 해지 (안전 장치)
        self.cls_api.unsubscribe_all_realtime_data()

    @classmethod
    def _conclusion_callback(cls, data: Dict[str, Any]):
        """실시간 주문 체결/응답 콜백 핸들러"""
        with cls._callback_lock: # 큐에 넣는 작업도 락으로 보호
            logger.info(f"실시간 체결/응답 수신: {data}")
            cls._conclusion_events_queue.put(data) # 큐에 이벤트 추가
            # 특정 종목의 시나리오 진행 이벤트를 여기서 바로 set하지 않고,
            # 워커 스레드 내의 _wait_for_conclusion_event_for_stock에서 큐를 직접 모니터링하도록 변경

    @classmethod
    def _price_update_callback(cls, stock_code: str, current_price: int, timestamp: float):
        """실시간 현재가 업데이트 콜백 핸들러 (5초 룰은 여기서 처리)"""
        with cls._callback_lock:
            if stock_code not in cls.cls_api._last_price_print_time_per_stock:
                cls.cls_api._last_price_print_time_per_stock[stock_code] = 0.0

            if time.time() - cls.cls_api._last_price_print_time_per_stock[stock_code] >= 5:
                logger.info(f"실시간 현재가 수신: 종목={stock_code}, 가격={current_price:,.0f}원")
                cls.cls_api._last_price_print_time_per_stock[stock_code] = time.time()
                cls._price_event_data_per_stock[stock_code] = {'stock_code': stock_code, 'current_price': current_price, 'timestamp': timestamp}
                if stock_code in cls._price_event_received_per_stock:
                    cls._price_event_received_per_stock[stock_code].set() # 이벤트 발생 신호

    @classmethod
    def _bid_update_callback(cls, stock_code: str, offer_prices: List[int], bid_prices: List[int], offer_amounts: List[int], bid_amounts: List[int]):
        """실시간 10차 호가 업데이트 콜백 핸들러 (현재는 로깅만)"""
        logger.debug(f"실시간 호가 수신: 종목={stock_code}, 1차 매도={offer_prices[0]}, 1차 매수={bid_prices[0]}")
        # 이 콜백에서 직접 데이터를 사용하는 대신, 필요 시점에 get_current_price_and_quotes를 다시 호출하여 최신 스냅샷을 가져오는 방식 사용

    @classmethod
    def _wait_for_conclusion_event_for_stock(cls, stock_code: str, target_order_id: Optional[int], expected_flags: List[str], timeout: int = 60) -> Optional[Dict[str, Any]]:
        """
        특정 종목 및 주문 ID에 대한 체결/응답 이벤트를 기다립니다.
        _conclusion_events_queue를 모니터링합니다.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 큐에서 이벤트를 가져오되, 타임아웃을 설정하여 무한 대기 방지
                # get_nowait() 대신 timeout을 사용하여 블로킹을 최소화하고 CPU 사용률을 낮춤
                data = cls._conclusion_events_queue.get(timeout=0.5) 
                
                # 락을 사용하여 공유 데이터 접근 보호 (여기서는 큐에서 꺼낸 데이터 처리 로직)
                with cls._callback_lock: 
                    # 해당 종목 또는 주문 ID에 대한 이벤트인지 확인
                    if data.get('code') == stock_code and \
                       (target_order_id is None or data.get('order_num') == target_order_id):
                        if data.get('flag') in expected_flags:
                            logger.info(f"대상 체결/응답 이벤트 수신 (종목: {stock_code}, 주문: {target_order_id}): {data}")
                            cls._conclusion_events_queue.task_done() # 처리 완료
                            return data
                        else:
                            logger.debug(f"대상 종목 ({stock_code}) 및 주문 ({target_order_id})의 다른 플래그 ({data.get('flag')}) 이벤트 수신. 기대 플래그 {expected_flags} 대기 계속...")
                            # 다른 플래그이므로 큐에 다시 넣지 않고 다음 이벤트를 기다림 (이미 get 했으므로 task_done 호출)
                            cls._conclusion_events_queue.task_done()
                    else:
                        # 다른 종목 또는 다른 주문의 이벤트이므로 다시 큐에 넣고 다음 이벤트를 기다림
                        cls._conclusion_events_queue.put(data)
                        cls._conclusion_events_queue.task_done() # get 후 put 했으므로 task_done 호출
                        
            except queue.Empty:
                # 큐가 비어있으면 계속 대기
                pass
            
        logger.warning(f"체결/응답 이벤트 타임아웃 ({timeout}초) 또는 대상 주문 ({target_order_id}) 이벤트 미수신 (종목: {stock_code}).")
        return None

    @classmethod
    def _wait_for_price_event_for_stock(cls, stock_code: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """특정 종목의 가격 업데이트 이벤트를 기다립니다."""
        if stock_code not in cls._price_event_received_per_stock:
            cls._price_event_received_per_stock[stock_code] = threading.Event()
        
        cls._price_event_received_per_stock[stock_code].clear()
        logger.debug(f"가격 이벤트 대기 중 (종목: {stock_code}, 최대 {timeout}초)...")
        if cls._price_event_received_per_stock[stock_code].wait(timeout):
            with cls._callback_lock: # 데이터 접근도 락으로 보호
                return cls._price_event_data_per_stock.get(stock_code)
        logger.warning(f"가격 이벤트 타임아웃 (종목: {stock_code}, {timeout}초).")
        return None

    @classmethod
    def _buy_worker_task(cls):
        """매수 주문을 처리하는 워커 스레드 함수"""
        # 각 스레드에서 COM 초기화
        pythoncom.CoInitialize() 
        try:
            api = cls.cls_api
            while True:
                stock_code, order_quantity = cls._buy_signal_queue.get()
                if stock_code is None: # 종료 신호
                    logger.info("매수 워커 스레드 종료 신호 수신.")
                    cls._buy_signal_queue.task_done()
                    break

                logger.info(f"매수 워커: 종목 [{stock_code}] 매수 시나리오 시작. 수량: {order_quantity}")
                cls._execute_buy_scenario_for_stock(stock_code, order_quantity)
                cls._buy_signal_queue.task_done()
                time.sleep(1) # 다음 주문 처리 전 잠시 대기 (API 부하 관리)
        finally:
            # 스레드 종료 시 COM 정리
            pythoncom.CoUninitialize()

    @classmethod
    def _sell_worker_task(cls):
        """매도 주문을 처리하는 워커 스레드 함수"""
        # 각 스레드에서 COM 초기화
        pythoncom.CoInitialize()
        try:
            api = cls.cls_api
            while True:
                stock_code, order_quantity = cls._sell_signal_queue.get()
                if stock_code is None: # 종료 신호
                    logger.info("매도 워커 스레드 종료 신호 수신.")
                    cls._sell_signal_queue.task_done()
                    break

                logger.info(f"매도 워커: 종목 [{stock_code}] 매도 시나리오 시작. 수량: {order_quantity}")
                cls._execute_sell_scenario_for_stock(stock_code, order_quantity)
                cls._sell_signal_queue.task_done()
                time.sleep(1) # 다음 주문 처리 전 잠시 대기 (API 부하 관리)
        finally:
            # 스레드 종료 시 COM 정리
            pythoncom.CoUninitialize()


    @classmethod
    def _execute_buy_scenario_for_stock(cls, stock_code: str, order_quantity: int):
        """단일 종목에 대한 매수 시나리오 실행 (워커 스레드에서 호출)"""
        api = cls.cls_api
        
        logger.info(f"--- 종목 [{stock_code}] 매수 시나리오 시작 ---")

        # 1. 현재가 실시간 이벤트 등록
        logger.info(f"종목 [{stock_code}] 현재가 실시간 이벤트 등록...")
        api.subscribe_realtime_price(stock_code)
        # 실시간 구독 후 데이터가 들어올 때까지 충분히 대기
        time.sleep(5) # 5초 대기 추가
        cls._wait_for_price_event_for_stock(stock_code, timeout=10) # 실시간 데이터가 들어올 때까지 잠시 대기

        # 2. 현재가 및 호가 조회 및 초기 매수 주문 (매수 3호가)
        logger.info(f"종목 [{stock_code}] 현재가 및 호가 조회 후 매수 3호가 주문 시도...")
        api.subscribe_realtime_bid(stock_code) # 호가 수신 이벤트 등록 (주문 직전)
        time.sleep(5) # 5초 대기 추가
        price_info = api.get_current_price_and_quotes(stock_code)
        
        if not price_info:
            logger.error(f"종목 [{stock_code}] 현재가 및 호가 조회 실패. 매수 시나리오 중단.")
            return # 시나리오 중단

        current_price = price_info['current_price']
        bid_prices = price_info['bid_prices']
        
        initial_buy_price = bid_prices[2] if len(bid_prices) >= 3 and bid_prices[2] > 0 else current_price 
        if initial_buy_price == 0: initial_buy_price = current_price
        initial_buy_price = api.round_to_tick(initial_buy_price)
        
        logger.info(f"종목 [{stock_code}] 현재가: {current_price:,.0f}원, 초기 매수 희망가격 (3차 매수호가): {initial_buy_price:,.0f}원")
        if not (initial_buy_price > 0):
            logger.error(f"종목 [{stock_code}] 유효한 초기 매수 가격을 찾을 수 없습니다. 매수 시나리오 중단.")
            return # 시나리오 중단

        buy_order_result = api.send_order(
            stock_code=stock_code,
            order_type=OrderType.BUY,
            amount=order_quantity,
            price=initial_buy_price,
            order_unit="01" # 보통가
        )
        if buy_order_result['status'] != 'success':
            logger.error(f"종목 [{stock_code}] 초기 매수 주문 실패: {buy_order_result['message']}. 매수 시나리오 중단.")
            return # 시나리오 중단

        current_order_id = buy_order_result['order_num']
        current_buy_order_price = initial_buy_price # 현재 주문 가격 추적
        logger.info(f"종목 [{stock_code}] 초기 매수 주문번호: {current_order_id}")

        # 3. 매수 주문 정정/취소/시장가 매수 로직
        executed_buy_qty = 0
        order_filled = False
        
        # 정정 시나리오 (3차 -> 2차 -> 1차)
        for i in range(3): # 0: 3차, 1: 2차, 2: 1차
            logger.info(f"종목 [{stock_code}] 매수 주문 ({current_order_id}) 체결 대기 (단계 {i+1}/3 - {10}초)...")
            # '체결' 또는 '확인' 플래그를 기다립니다.
            conclusion_data = cls._wait_for_conclusion_event_for_stock(stock_code=stock_code, target_order_id=current_order_id, expected_flags=['체결', '확인'], timeout=10)
            
            if conclusion_data and conclusion_data['flag'] == '체결':
                executed_buy_qty += conclusion_data['amount']
                logger.info(f"종목 [{stock_code}] 매수 주문 ({current_order_id}) 부분 체결 발생. 현재 체결 수량: {executed_buy_qty}")
                if executed_buy_qty >= order_quantity:
                    logger.info(f"종목 [{stock_code}] 매수 주문 ({current_order_id}) 전량 체결 완료.")
                    order_filled = True
                    break
            elif conclusion_data and conclusion_data['flag'] == '거부':
                logger.error(f"종목 [{stock_code}] 매수 주문 ({current_order_id})이 거부되었습니다: {conclusion_data['message']}. 매수 시나리오 중단.")
                return # 시나리오 중단
            elif conclusion_data and conclusion_data['flag'] == '확인':
                logger.info(f"종목 [{stock_code}] 매수 주문 ({current_order_id}) 확인 이벤트 수신: {conclusion_data['flag']}")
            else: # 10초 내 미체결 (또는 원하는 플래그 미수신)
                logger.info(f"종목 [{stock_code}] 매수 주문 ({current_order_id}) {10}초 내 미체결 또는 원하는 이벤트 미수신. 정정 시도...")
                if i < 2: # 2차, 1차 호가로 정정
                    # 호가 정보는 이미 구독 중이므로 다시 구독할 필요 없음. 최신 정보 조회
                    price_info_mod = api.get_current_price_and_quotes(stock_code)
                    
                    if not price_info_mod:
                        logger.error(f"종목 [{stock_code}] 정정 시 호가 정보 조회 실패. 정정 중단.")
                        break # 정정 루프 종료

                    # bid_prices의 인덱스를 0, 1, 2 순으로 사용 (3차 -> 2차 -> 1차 매수호가)
                    # i=0: 3차 호가 (bid_prices[2]), i=1: 2차 호가 (bid_prices[1]), i=2: 1차 호가 (bid_prices[0])
                    next_bid_price = price_info_mod['bid_prices'][2-i] if len(price_info_mod['bid_prices']) > (2-i) and price_info_mod['bid_prices'][2-i] > 0 else price_info_mod['current_price']
                    if next_bid_price == 0: next_bid_price = price_info_mod['current_price']
                    next_bid_price = api.round_to_tick(next_bid_price)

                    # 정정 단가가 현재 주문 단가와 동일한지 확인
                    if next_bid_price == current_buy_order_price:
                        logger.info(f"종목 [{stock_code}] 매수 주문 ({current_order_id}) 정정 단가 ({next_bid_price})가 현재 주문 단가 ({current_buy_order_price})와 동일하여 정정 스킵.")
                        continue # 다음 단계로 넘어감
                    
                    logger.info(f"종목 [{stock_code}] 매수 주문 ({current_order_id}) 정정 시도: {3-i}차 매수호가 {next_bid_price:,.0f}원")
                    modify_result = api.send_order(
                        stock_code=stock_code,
                        order_type=OrderType.MODIFY,
                        amount=0, # 잔량 정정
                        price=next_bid_price,
                        org_order_num=current_order_id
                    )
                    if modify_result['status'] != 'success':
                        logger.error(f"종목 [{stock_code}] 매수 정정 주문 실패: {modify_result['message']}. 매수 시나리오 중단.")
                        return # 시나리오 중단
                    current_order_id = modify_result['order_num'] # 정정 시 새 주문번호 받을 수 있음
                    current_buy_order_price = next_bid_price # 정정 성공 시 가격 업데이트
                    logger.info(f"종목 [{stock_code}] 매수 정정 주문 성공. 새 주문번호: {current_order_id}")
                else: # 1차 호가까지 정정 후에도 미체결 -> 취소
                    logger.info(f"종목 [{stock_code}] 매수 주문 ({current_order_id}) 1차 호가 정정 후에도 미체결. 취소 시도...")
                    cancel_result = api.send_order(
                        stock_code=stock_code,
                        order_type=OrderType.CANCEL,
                        amount=0, # 잔량 취소
                        org_order_num=current_order_id
                    )
                    if cancel_result['status'] != 'success':
                        logger.error(f"종목 [{stock_code}] 매수 취소 주문 실패: {cancel_result['message']}. 매수 시나리오 중단.")
                        return # 시나리오 중단
                    logger.info(f"종목 [{stock_code}] 매수 주문 ({current_order_id}) 취소 요청 성공. 체결 이벤트 대기...")
                    # 취소 확인 이벤트 대기
                    conclusion_data = cls._wait_for_conclusion_event_for_stock(stock_code=stock_code, target_order_id=current_order_id, expected_flags=['확인'], timeout=10)
                    if conclusion_data and conclusion_data['flag'] == '확인':
                        logger.info(f"종목 [{stock_code}] 매수 주문 ({current_order_id}) 취소 확인 완료.")
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
                            logger.warning(f"종목 [{stock_code}] 매수 주문 ({current_order_id})이 미체결 목록에서 제거되지 않았거나 타임아웃 발생.")
                    else:
                        logger.warning(f"종목 [{stock_code}] 매수 주문 ({current_order_id}) 취소 확인 실패 또는 타임아웃.")
                    order_filled = False # 취소되었으므로 체결되지 않음
                    break # 정정/취소 루프 종료

        # 매수 주문 처리 루프 종료 후 호가 구독 해지
        api.unsubscribe_realtime_bid(stock_code) 

        if not order_filled:
            logger.info(f"종목 [{stock_code}] 매수 주문이 최종적으로 체결되지 않았습니다. 시장가 매수 시도...")
            time.sleep(5) # 시장가 매수 전 잠시 대기 (API 부하 감소)
            market_buy_result = api.send_order(
                stock_code=stock_code,
                order_type=OrderType.BUY,
                amount=order_quantity,
                price=0, # 시장가
                order_unit="03" # 시장가 주문
            )
            if market_buy_result['status'] != 'success':
                logger.error(f"종목 [{stock_code}] 시장가 매수 주문 실패: {market_buy_result['message']}. 매수 시나리오 중단.")
                return # 시나리오 중단
            current_order_id = market_buy_result['order_num']
            logger.info(f"종목 [{stock_code}] 시장가 매수 주문번호: {current_order_id}. 시장가 주문은 성공 시 즉시 체결된 것으로 간주합니다.")
            order_filled = True # 시장가 매수 주문은 성공 시 즉시 체결된 것으로 간주

        # 4. 종목 잔고 확인 (매수 후)
        logger.info(f"종목 [{stock_code}] 매수 후 종목 잔고 확인...")
        positions_after_buy = api.get_portfolio_positions()
        found_position = False
        for pos in positions_after_buy:
            if pos['stock_code'] == stock_code:
                if pos['quantity'] < order_quantity:
                    logger.error(f"종목 [{stock_code}] 매수 후 보유 수량이 예상보다 적습니다. 현재: {pos['quantity']}주. 예상: {order_quantity}주.")
                    return # 시나리오 중단
                logger.info(f"종목 [{stock_code}] 매수 후 보유 수량: {pos['quantity']}주")
                found_position = True
                break
        if not found_position:
            logger.error(f"종목 [{stock_code}] 매수 후 포지션에 없습니다. 매수 시나리오 실패.")
            return # 시나리오 중단

        # 5. 매수 체결 완료 후 호가 실시간 이벤트 해제 (현재가는 유지)
        logger.info(f"종목 [{stock_code}] 매수 체결 완료 후 호가 실시간 이벤트 해지.")
        api.unsubscribe_realtime_bid(stock_code) 

        logger.info(f"--- 종목 [{stock_code}] 매수 시나리오 완료. 매도 신호 큐에 추가 ---")
        cls._stock_scenario_states[stock_code] = 'BOUGHT'
        cls._sell_signal_queue.put((stock_code, order_quantity)) # 매수 완료 후 매도 신호 큐에 추가
        cls._stock_scenario_events[stock_code].set() # 시나리오 완료 신호

    @classmethod
    def _execute_sell_scenario_for_stock(cls, stock_code: str, order_quantity: int):
        """단일 종목에 대한 매도 시나리오 실행 (워커 스레드에서 호출)"""
        api = cls.cls_api

        logger.info(f"--- 종목 [{stock_code}] 매도 시나리오 시작 ---")

        # 1. 종목 잔고 확인 (매도 가능 종목)
        logger.info(f"종목 [{stock_code}] 매도 전 종목 잔고 확인...")
        positions_before_sell = api.get_portfolio_positions()
        sellable_quantity = 0
        for pos in positions_before_sell:
            if pos['stock_code'] == stock_code:
                sellable_quantity = pos['sell_avail_qty']
                break
        if sellable_quantity < order_quantity:
            logger.error(f"종목 [{stock_code}] 매도할 수량이 부족합니다. 현재: {sellable_quantity}주. 예상: {order_quantity}주. 매도 시나리오 중단.")
            return # 시나리오 중단

        logger.info(f"종목 [{stock_code}] 매도 가능 수량: {sellable_quantity}주")

        # 2. 현재가 및 호가 조회 및 초기 매도 주문 (매도 3호가)
        logger.info(f"종목 [{stock_code}] 현재가 및 호가 조회 후 매도 3호가 주문 시도...")
        api.subscribe_realtime_bid(stock_code) # 호가 수신 이벤트 등록 (주문 직전)
        time.sleep(5) # 5초 대기 추가
        price_info_sell = api.get_current_price_and_quotes(stock_code)
        
        if not price_info_sell:
            logger.error(f"종목 [{stock_code}] 매도 시나리오를 위한 현재가 및 호가 조회 실패. 매도 시나리오 중단.")
            return # 시나리오 중단

        current_price_sell = price_info_sell['current_price']
        offer_prices = price_info_sell['offer_prices']
        
        initial_sell_price = offer_prices[2] if len(offer_prices) >= 3 and offer_prices[2] > 0 else current_price_sell
        if initial_sell_price == 0: initial_sell_price = current_price_sell
        initial_sell_price = api.round_to_tick(initial_sell_price)

        logger.info(f"종목 [{stock_code}] 현재가: {current_price_sell:,.0f}원, 초기 매도 희망가격 (3차 매도호가): {initial_sell_price:,.0f}원")
        if not (initial_sell_price > 0):
            logger.error(f"종목 [{stock_code}] 유효한 초기 매도 가격을 찾을 수 없습니다. 매도 시나리오 중단.")
            return # 시나리오 중단

        sell_order_result = api.send_order(
            stock_code=stock_code,
            order_type=OrderType.SELL,
            amount=order_quantity,
            price=initial_sell_price,
            order_unit="01" # 보통가
        )
        if sell_order_result['status'] != 'success':
            logger.error(f"종목 [{stock_code}] 초기 매도 주문 실패: {sell_order_result['message']}. 매도 시나리오 중단.")
            return # 시나리오 중단

        current_sell_order_id = sell_order_result['order_num']
        current_sell_order_price = initial_sell_price # 현재 주문 가격 추적
        logger.info(f"종목 [{stock_code}] 초기 매도 주문번호: {current_sell_order_id}")

        # 3. 매도 주문 정정/시장가 매도 로직
        executed_sell_qty = 0
        sell_order_filled = False

        # 정정 시나리오 (3차 -> 2차 -> 1차)
        for i in range(3): # 0: 3차, 1: 2차, 2: 1차
            logger.info(f"종목 [{stock_code}] 매도 주문 ({current_sell_order_id}) 체결 대기 (단계 {i+1}/3 - {10}초)...")
            conclusion_data = cls._wait_for_conclusion_event_for_stock(stock_code=stock_code, target_order_id=current_sell_order_id, expected_flags=['체결', '확인'], timeout=10)
            
            if conclusion_data and conclusion_data['flag'] == '체결':
                executed_sell_qty += conclusion_data['amount']
                logger.info(f"종목 [{stock_code}] 매도 주문 ({current_sell_order_id}) 부분 체결 발생. 현재 체결 수량: {executed_sell_qty}")
                if executed_sell_qty >= order_quantity:
                    logger.info(f"종목 [{stock_code}] 매도 주문 ({current_sell_order_id}) 전량 체결 완료.")
                    sell_order_filled = True
                    break
            elif conclusion_data and conclusion_data['flag'] == '거부':
                logger.error(f"종목 [{stock_code}] 매도 주문 ({current_sell_order_id})이 거부되었습니다: {conclusion_data['message']}. 매도 시나리오 중단.")
                return # 시나리오 중단
            elif conclusion_data and conclusion_data['flag'] == '확인':
                logger.info(f"종목 [{stock_code}] 매도 주문 ({current_sell_order_id}) 확인 이벤트 수신: {conclusion_data['flag']}")
            else: # 10초 내 미체결
                logger.info(f"종목 [{stock_code}] 매도 주문 ({current_sell_order_id}) {10}초 내 미체결 또는 원하는 이벤트 미수신. 정정 시도...")
                if i < 2: # 2차, 1차 호가로 정정
                    # 호가 정보는 이미 구독 중이므로 다시 구독할 필요 없음. 최신 정보 조회
                    price_info_mod_sell = api.get_current_price_and_quotes(stock_code)

                    if not price_info_mod_sell:
                        logger.error(f"종목 [{stock_code}] 정정 시 호가 정보 조회 실패. 정정 중단.")
                        break # 정정 루프 종료

                    # offer_prices의 인덱스를 0, 1, 2 순으로 사용 (3차 -> 2차 -> 1차 매도호가)
                    next_offer_price = price_info_mod_sell['offer_prices'][2-i] if len(price_info_mod_sell['offer_prices']) > (2-i) and price_info_mod_sell['offer_prices'][2-i] > 0 else price_info_mod_sell['current_price']
                    if next_offer_price == 0: next_offer_price = price_info_mod_sell['current_price']
                    next_offer_price = api.round_to_tick(next_offer_price)

                    # 정정 단가가 현재 주문 단가와 동일한지 확인
                    if next_offer_price == current_sell_order_price:
                        logger.info(f"종목 [{stock_code}] 매도 주문 ({current_sell_order_id}) 정정 단가 ({next_offer_price})가 현재 주문 단가 ({current_sell_order_price})와 동일하여 정정 스킵.")
                        continue # 다음 단계로 넘어감

                    logger.info(f"종목 [{stock_code}] 매도 주문 ({current_sell_order_id}) 정정 시도: {3-i}차 매도호가 {next_offer_price:,.0f}원")
                    modify_result_sell = api.send_order(
                        stock_code=stock_code,
                        order_type=OrderType.MODIFY,
                        amount=0, # 잔량 정정
                        price=next_offer_price,
                        org_order_num=current_sell_order_id
                    )
                    if modify_result_sell['status'] != 'success':
                        logger.error(f"종목 [{stock_code}] 매도 정정 주문 실패: {modify_result_sell['message']}. 매도 시나리오 중단.")
                        return # 시나리오 중단
                    current_sell_order_id = modify_result_sell['order_num'] # 정정 시 새 주문번호 받을 수 있음
                    current_sell_order_price = next_offer_price # 정정 성공 시 가격 업데이트
                    logger.info(f"종목 [{stock_code}] 매도 정정 주문 성공. 새 주문번호: {current_sell_order_id}")
                else: # 1차 호가까지 정정 후에도 미체결 -> 취소
                    logger.info(f"종목 [{stock_code}] 매도 주문 ({current_sell_order_id}) 1차 호가 정정 후에도 미체결. 취소 시도...")
                    cancel_result = api.send_order(
                        stock_code=stock_code,
                        order_type=OrderType.CANCEL,
                        amount=0, # 잔량 취소
                        org_order_num=current_sell_order_id
                    )
                    if cancel_result['status'] != 'success':
                        logger.error(f"종목 [{stock_code}] 매도 취소 주문 실패: {cancel_result['message']}. 매도 시나리오 중단.")
                        return # 시나리오 중단
                    logger.info(f"종목 [{stock_code}] 매도 주문 ({current_sell_order_id}) 취소 요청 성공. 체결 이벤트 대기...")
                    # 취소 확인 이벤트 대기
                    conclusion_data = cls._wait_for_conclusion_event_for_stock(stock_code=stock_code, target_order_id=current_sell_order_id, expected_flags=['확인'], timeout=10)
                    if conclusion_data and conclusion_data['flag'] == '확인':
                        logger.info(f"종목 [{stock_code}] 매도 주문 ({current_sell_order_id}) 취소 확인 완료.")
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
                            logger.warning(f"종목 [{stock_code}] 매도 주문 ({current_sell_order_id})이 미체결 목록에서 제거되지 않았거나 타임아웃 발생.")
                    else:
                        logger.warning(f"종목 [{stock_code}] 매도 주문 ({current_sell_order_id}) 취소 확인 실패 또는 타임아웃.")

                    sell_order_filled = False # 취소되었으므로 체결되지 않음
                    break # 정정/취소 루프 종료
        
        # 매도 주문 처리 루프 종료 후 호가 구독 해지
        api.unsubscribe_realtime_bid(stock_code) 

        if not sell_order_filled:
            logger.info(f"종목 [{stock_code}] 매도 주문이 최종적으로 체결되지 않았습니다. 시장가 매도 시도...")
            time.sleep(5) # 시장가 매도 전 잠시 대기 (API 부하 감소)
            
            # 최종 시장가 매도 시 모든 잔량 매도
            logger.info(f"종목 [{stock_code}] 시장가 매도를 위해 현재 보유 잔량을 다시 조회합니다.")
            current_positions_for_market_sell = api.get_portfolio_positions()
            market_sell_quantity = 0
            for pos in current_positions_for_market_sell:
                if pos['stock_code'] == stock_code:
                    market_sell_quantity = pos['sell_avail_qty']
                    break
            
            if market_sell_quantity == 0:
                logger.warning(f"종목 [{stock_code}]의 매도 가능 잔량이 0이므로 시장가 매도를 스킵합니다.")
                sell_order_filled = True # 매도할 것이 없으므로 매도 완료로 간주
            else:
                logger.info(f"종목 [{stock_code}] 시장가 매도 주문 요청 - 종목: {stock_code}, 수량: {market_sell_quantity} (모든 잔량)")
                market_sell_result = api.send_order(
                    stock_code=stock_code,
                    order_type=OrderType.SELL,
                    amount=market_sell_quantity, # 모든 잔량 매도
                    price=0, # 시장가
                    order_unit="03" # 시장가 주문
                )
                if market_sell_result['status'] != 'success':
                    logger.error(f"종목 [{stock_code}] 시장가 매도 주문 실패: {market_sell_result['message']}. 매도 시나리오 중단.")
                    return # 시나리오 중단
                current_sell_order_id = market_sell_result['order_num']
                logger.info(f"종목 [{stock_code}] 시장가 매도 주문번호: {current_sell_order_id}. 시장가 주문은 성공 시 즉시 체결된 것으로 간주합니다.")
                sell_order_filled = True # 시장가 매도 주문은 성공 시 즉시 체결된 것으로 간주

        # 4. 종목 잔고 확인 (매도 후)
        logger.info(f"종목 [{stock_code}] 매도 후 종목 잔고 확인...")
        positions_after_sell = api.get_portfolio_positions()
        final_quantity = 0
        for pos in positions_after_sell:
            if pos['stock_code'] == stock_code:
                final_quantity = pos['quantity']
                break
        # 만약 매도 가능 수량이 전체 보유 수량보다 적어서 일부만 매도되었다면, final_quantity는 0이 아닐 수 있습니다.
        # 테스트의 목표가 '잔고에 남아 있는 모든 수량'을 시장가 매도하는 것이므로, 0이 되는 것을 기대합니다.
        if final_quantity != 0:
            logger.error(f"종목 [{stock_code}] 매도 후 잔고가 0이 아닙니다. 현재 잔고: {final_quantity}주.")
            # return # 엄격한 테스트라면 여기서 실패 처리
        logger.info(f"종목 [{stock_code}] 매도 후 최종 보유 수량: {final_quantity}주")

        # 5. 최종 정리: 현재가 실시간 이벤트 해지
        logger.info(f"종목 [{stock_code}] 최종 정리: 현재가 실시간 이벤트 해지.")
        api.unsubscribe_realtime_price(stock_code)
        logger.info(f"--- 종목 [{stock_code}] 매도 시나리오 완료 ---")
        cls._stock_scenario_states[stock_code] = 'SOLD'
        cls._stock_scenario_events[stock_code].set() # 시나리오 완료 신호


    def test_multi_stock_trading_scenario_with_queues(self):
        """
        여러 종목에 대한 매수 -> 정정 -> 취소 -> 시장가 매수 -> 정정 -> 시장가 매도 시나리오를
        큐와 워커 스레드를 사용하여 동시성으로 테스트합니다.
        """
        logger.info("\n--- 다중 종목 매매 시나리오 시작 (큐/워커 기반) ---")

        # 1. 예수금 확인 (시작 시 한 번만)
        logger.info("1. 예수금 확인...")
        account_balance = self.cls_api.get_account_balance()
        self.assertIsNotNone(account_balance, "계좌 잔고 조회 실패")
        current_cash = account_balance.get('cash_balance', 0)
        logger.info(f"현재 주문 가능 현금: {current_cash:,.0f}원")
        self.assertGreater(current_cash, 10000 * len(self.TEST_STOCK_CODES), "주문 가능 현금이 부족합니다 (각 종목당 최소 10,000원 필요).")

        # 각 종목에 대해 초기 매수 신호를 큐에 추가
        for stock_code in self.TEST_STOCK_CODES:
            self._stock_scenario_states[stock_code] = 'BUY_PENDING'
            self._buy_signal_queue.put((stock_code, self.ORDER_QUANTITY))
            logger.info(f"초기 매수 신호 큐에 추가: 종목 [{stock_code}]")

        # 모든 종목의 시나리오가 완료될 때까지 대기
        all_scenarios_completed = False
        start_time = time.time()
        timeout = 300 # 전체 시나리오 타임아웃 (예: 5분)

        while not all_scenarios_completed and (time.time() - start_time < timeout):
            all_scenarios_completed = True
            for stock_code in self.TEST_STOCK_CODES:
                if self._stock_scenario_states[stock_code] != 'SOLD':
                    all_scenarios_completed = False
                    # 각 종목의 시나리오 완료 이벤트를 기다림 (짧게 대기하며 주기적으로 상태 확인)
                    if not self._stock_scenario_events[stock_code].wait(1): # 1초마다 확인
                        logger.info(f"종목 [{stock_code}] 시나리오 진행 중... 현재 상태: {self._stock_scenario_states[stock_code]}")
                    else:
                        logger.info(f"종목 [{stock_code}] 시나리오 완료 신호 수신. 최종 상태: {self._stock_scenario_states[stock_code]}")
                        self._stock_scenario_events[stock_code].clear() # 이벤트 초기화 (다음 테스트를 위해)
            time.sleep(1) # 전체 상태 확인 주기

        # 모든 시나리오가 완료되었는지 최종 확인
        for stock_code in self.TEST_STOCK_CODES:
            self.assertEqual(self._stock_scenario_states[stock_code], 'SOLD', 
                             f"종목 [{stock_code}] 시나리오가 완료되지 않았습니다. 최종 상태: {self._stock_scenario_states[stock_code]}")
            # 최종 잔고 확인 (0이 아닐 수 있으므로 경고만)
            positions = self.cls_api.get_portfolio_positions()
            final_quantity = 0
            for pos in positions:
                if pos['stock_code'] == stock_code:
                    final_quantity = pos['quantity']
                    break
            self.assertEqual(final_quantity, 0, f"매도 후 {stock_code} 잔고가 0이 아닙니다. 현재 잔고: {final_quantity}주")
            logger.info(f"최종 확인: 종목 [{stock_code}] 매도 후 최종 보유 수량: {final_quantity}주")

        logger.info("\n--- 다중 종목 매매 시나리오 성공적으로 완료 ---")

if __name__ == '__main__':
    unittest.main()
