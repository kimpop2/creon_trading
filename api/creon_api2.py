# api_client/creon_api.py

import win32com.client
import ctypes
import time
import logging
import pandas as pd
import re
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Callable, Tuple
import threading # threading 모듈 추가
from enum import Enum

# API 요청 간격 (크레온 API 제한 준수)
# 이 값을 조정하여 테스트해보세요 (예: 0.3 또는 0.5)
API_REQUEST_INTERVAL = 1.5 # 기존 1.0에서 1.5로 증가

# 로거 설정
logger = logging.getLogger(__name__)

# --- 주문 관련 Enum (실시간주문체결.py의 orderStatus와 유사) ---
class OrderStatus(Enum):
    NOTHING = 1         # 별 일 없는 상태
    NEW_ORDER = 2       # 신규 주문 낸 상태
    ORDER_CONFIRM = 3   # 신규 주문 처리 확인 (접수)
    MODIFY_ORDER = 4    # 정정 주문 낸 상태
    CANCEL_ORDER = 5    # 취소 주문 낸 상태

class OrderType(Enum):
    BUY = "2"    # Creon API code for Buy (매수)
    SELL = "1"   # Creon API code for Sell (매도)
    MODIFY = "modify" # Custom type for modify logic (정정)
    CANCEL = "cancel" # Custom type for cancel logic (취소)

# --- 실시간 이벤트 핸들러 (실시간주문체결.py의 CpEvent와 유사) ---
class CpEvent:
    """
    Creon API로부터 실시간 이벤트를 수신하고 CreonAPIClient의 콜백 메서드를 호출합니다.
    """
    def set_params(self, client_obj, event_name: str, parent_instance, stock_code: Optional[str] = None):
        self.client = client_obj
        self.name = event_name
        self.parent = parent_instance # CreonAPIClient 인스턴스
        self.stock_code = stock_code
        self.concdic = {"1": "체결", "2": "확인", "3": "거부", "4": "접수"}
        self.buyselldic = {"1" : "매도", "2" : "매수"}

    def OnReceived(self):
        """PLUS로부터 실시간 이벤트를 수신 받아 처리하는 함수"""
        # 주문 체결/응답 이벤트 처리
        if self.name == "conclusion":
            conflag = self.client.GetHeaderValue(14)    # 체결 플래그 (1:체결, 2:확인, 3:거부, 4:접수)
            ordernum = self.client.GetHeaderValue(5)    # 주문번호
            amount = self.client.GetHeaderValue(3)      # 체결 수량 또는 접수 수량
            price = self.client.GetHeaderValue(4)       # 가격
            code = self.client.GetHeaderValue(9)        # 종목코드
            bs = self.client.GetHeaderValue(12)         # 매수/매도 구분 (1:매도, 2:매수)
            balance = self.client.GetHeaderValue(23)     # 체결 후 잔고 수량 (체결 시에만 유효)
            # order_original_amount = self.client.GetHeaderValue(2) # 원 주문 수량 (실시간주문체결.py에서 사용 안 함, 제거 또는 확인 필요)

            conflags_str = self.concdic.get(conflag, "알수없음")
            bs_str = self.buyselldic.get(bs, "알수없음")

            logger.info(f"[CpEvent] 주문 체결/응답 수신: {conflags_str} {bs_str} 종목:{code} 가격:{price:,.0f} 수량:{amount} 주문번호:{ordernum} 잔고:{balance}")

            if self.parent.conclusion_callback:
                self.parent.conclusion_callback({
                    'flag': conflags_str,
                    'order_num': ordernum,
                    'code': code,
                    'price': price,
                    'amount': amount,
                    'balance': balance,
                    # 'order_original_amount': order_original_amount, # 사용하지 않으므로 제거 또는 주석 처리
                    'buy_sell': bs_str
                })

        # 실시간 현재가 이벤트 처리 (실시간주문체결.py의 StockCur 이벤트와 유사)
        elif self.name == "stockcur":
            exFlag = self.client.GetHeaderValue(19)  # 예상체결 플래그
            cprice = self.client.GetHeaderValue(13)  # 현재가
            
            # 장중이 아니면 처리 안함. (예상체결 플래그 2: 장중)
            if exFlag != ord('2'):
                return
            
            if self.parent.price_update_callback:
                # 5초에 한번씩만 출력하는 로직은 콜백을 받는 쪽(Brokerage/TradingManager)에서 처리
                self.parent.price_update_callback(self.stock_code, cprice, time.time())

        # 실시간 10차 호가 이벤트 처리 (실시간주문체결.py의 StockBid 이벤트와 유사)
        elif self.name == "stockbid":
            # Dscbo1.StockJpBid COM 객체의 헤더 값 인덱스
            # 매도호가: 0, 2, 4, ... 18 (10개)
            # 매수호가: 1, 3, 5, ... 19 (10개)
            # 매도호가잔량: 20, 22, ... 38 (10개)
            # 매수호가잔량: 21, 23, ... 39 (10개)
            offer_prices = [self.client.GetHeaderValue(i) for i in range(0, 19, 2)]
            bid_prices = [self.client.GetHeaderValue(i) for i in range(1, 20, 2)]
            offer_amounts = [self.client.GetHeaderValue(i) for i in range(20, 39, 2)]
            bid_amounts = [self.client.GetHeaderValue(39 - i) for i in range(0, 19, 2)] # 매수호가잔량 역순 (실시간주문체결.py와 동일하게)
            
            # 실시간주문체결.py에서는 10차 호가도 parent.monitorPriceChange()를 호출
            if self.parent.bid_update_callback:
                self.parent.bid_update_callback(self.stock_code, offer_prices, bid_prices, offer_amounts, bid_amounts)

# --- 실시간 구독 클래스들의 공통 부모 ---
class CpPublish:
    """
    Creon API 실시간 구독 객체들의 기본 클래스.
    COM 객체 생성, 구독, 해지 기능을 캡슐화합니다.
    """
    def __init__(self, com_obj_prog_id: str, event_name: str):
        self.obj = win32com.client.Dispatch(com_obj_prog_id)
        self.event_handler = None
        self.stock_code = None
        self.event_name = event_name

    def Subscribe(self, parent, stock_code: Optional[str] = None):
        """실시간 구독을 시작합니다."""
        # 이 부분은 개별 CpPublish 인스턴스가 재사용될 경우를 위한 것이므로,
        # 현재의 "전체 해지 후 재구독" 로직에서는 불필요하지만 안전을 위해 유지.
        if self.event_handler: 
            self.Unsubscribe()

        self.stock_code = stock_code
        if stock_code:
            self.obj.SetInputValue(0, stock_code)
        
        self.event_handler = win32com.client.WithEvents(self.obj, CpEvent)
        self.event_handler.set_params(self.obj, self.event_name, parent, stock_code)

        self.obj.Subscribe()
        logger.info(f"실시간 구독 시작: {self.event_name} for {stock_code if stock_code else '계좌 전체'}")

    def Unsubscribe(self):
        """실시간 구독을 해지합니다."""
        if self.obj and self.event_handler: # 객체와 핸들러가 존재하는지 확인
            self.obj.Unsubscribe()
            logger.info(f"실시간  {'호가' if self.event_name == 'stockbid' else '현재가'} 구독 해지: {self.stock_code if self.stock_code else '계좌 전체'}")
        self.event_handler = None
        self.stock_code = None

# --- 특정 실시간 구독 클래스들 (실시간주문체결.py의 CpPBStockCur, CpPBStockBid, CpPBConclusion와 유사) ---
class ConclusionSubscriber(CpPublish):
    """주문 체결 실시간 구독"""
    def __init__(self):
        super().__init__("DsCbo1.CpConclusion", "conclusion")

class StockCurSubscriber(CpPublish):
    """주식 현재가 실시간 구독"""
    def __init__(self):
        super().__init__("DsCbo1.StockCur", "stockcur")

class StockBidSubscriber(CpPublish):
    """주식 10차 호가 실시간 구독"""
    def __init__(self):
        super().__init__("Dscbo1.StockJpBid", "stockbid")

# --- CreonAPIClient 클래스 ---
class CreonAPIClient:
    """
    Creon Plus API와 통신하는 클라이언트 클래스.
    연결 관리, 주문 전송, 데이터 조회, 실시간 이벤트 수신 기능을 제공합니다.
    """
    # 모든 BlockRequest 호출을 직렬화하기 위한 전역 락
    _api_request_lock = threading.Lock()
    # 실시간 구독/해지 작업을 보호하기 위한 락 추가
    _realtime_sub_lock = threading.Lock()

    def __init__(self):
        self.obj_cp_cybos = None
        self.obj_trade = None
        self.account_number = None
        self.account_flag = None
        self.connected = False # 연결 상태 초기화

        self._connect_creon_and_init_trade()

        # 실시간 구독 객체 관리 딕셔너리 (실제 구독 객체 인스턴스)
        self.conclusion_subscriber: Optional[ConclusionSubscriber] = None
        self.stock_cur_subscribers: Dict[str, StockCurSubscriber] = {} # {종목코드: StockCurSubscriber 객체}
        self.stock_bid_subscribers: Dict[str, StockBidSubscriber] = {} # {종목코드: StockBidSubscriber 객체}

        # 현재 활성화된 구독 종목 코드를 추적하는 세트
        self._active_cur_subscriptions: set[str] = set()
        self._active_bid_subscriptions: set[str] = set()

        # 실시간 이벤트 콜백 함수 (외부 모듈에서 등록)
        self.conclusion_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.price_update_callback: Optional[Callable[[str, int, float], None]] = None
        self.bid_update_callback: Optional[Callable[[str, List[int], List[int], List[int], List[int]], None]] = None

        # 실시간 현재가 출력 주기 제어용 (5초 룰)
        self._last_price_print_time_per_stock: Dict[str, float] = {}

        # CreonAPIClient 초기화 시 주문 체결 실시간 구독 시작
        self._init_conclusion_subscriber()


    def _connect_creon_and_init_trade(self):
        """Creon Plus에 연결하고 COM 객체 및 거래 초기화를 수행합니다."""
        if not ctypes.windll.shell32.IsUserAnAdmin():
            logger.warning("관리자 권한으로 실행되지 않았습니다. 일부 Creon 기능이 제한될 수 있습니다.")

        self.cp_cybos = win32com.client.Dispatch("CpUtil.CpCybos")
        if self.cp_cybos.IsConnect:
            self.connected = True
            logger.info("Creon Plus가 이미 연결되어 있습니다.")
        else:
            logger.info("Creon Plus 연결 시도 중...")
            max_retries = 10
            for i in range(max_retries):
                if self.cp_cybos.IsConnect:
                    self.connected = True
                    logger.info("Creon Plus 연결 성공.")
                    break
                else:
                    logger.warning(f"Creon Plus 연결 대기 중... ({i+1}/{max_retries})")
                    time.sleep(2)
            if not self.connected:
                logger.error("Creon Plus 연결 실패. HTS가 실행 중이고 로그인되어 있는지 확인하세요.")
                raise ConnectionError("Creon Plus 연결 실패.")

        try:
            self.obj_trade = win32com.client.Dispatch('CpTrade.CpTdUtil')
            if self.obj_trade.TradeInit(0) != 0:
                logger.error("주문 초기화 실패 (TradeInit)!")
                raise RuntimeError("Creon TradeInit 실패.")

            self.account_number = self.obj_trade.AccountNumber[0]
            self.account_flag = self.obj_trade.GoodsList(self.account_number, 1)[0] # 주식 상품 구분
            logger.info(f"Creon API 계좌 정보 확인: 계좌번호={self.account_number}, 상품구분={self.account_flag}")

        except Exception as e:
            logger.error(f"Creon TradeUtil 초기화 또는 계좌 정보 가져오는 중 오류 발생: {e}", exc_info=True)
            raise

    def _wait_for_api_limit(self):
        """Creon API 요청 제한 간격을 유지합니다."""
        time.sleep(API_REQUEST_INTERVAL)

    def _execute_block_request(self, com_object: Any, method_name: str = "BlockRequest") -> Tuple[int, str]:
        """
        COM 객체에 대한 BlockRequest를 실행하고 공통 오류를 처리합니다.
        실시간주문체결.py의 통신 상태 확인 로직을 반영합니다.
        :param com_object: Creon API COM 객체
        :param method_name: 호출할 메서드 이름 (기본값: "BlockRequest")
        :return: (상태 코드, 메시지) 튜플. 상태 코드가 0이면 성공.
        """
        # 전역 락을 사용하여 모든 BlockRequest 호출을 직렬화
        with CreonAPIClient._api_request_lock:
            try:
                # 요청 전 API 제한 간격 유지
                self._wait_for_api_limit()

                ret = getattr(com_object, method_name)()
                
                # BlockRequest 자체의 반환 값 확인 (0이 아니면 실패)
                if ret != 0:
                    # _get_clsid_ 대신 COM 객체의 __class__.__name__을 사용하거나, 단순히 객체 타입을 로깅
                    obj_identifier = getattr(com_object, '__class__', None).__name__ or str(type(com_object))
                    error_msg = f"COM 객체 {obj_identifier} {method_name} 호출 실패. 반환 코드: {ret}"
                    logger.error(error_msg)
                    return ret, error_msg

                # DIB 통신 상태 및 메시지 확인
                status = com_object.GetDibStatus()
                msg = com_object.GetDibMsg1()

                if status != 0:
                    obj_identifier = getattr(com_object, '__class__', None).__name__ or str(type(com_object))
                    error_msg = f"COM 객체 {obj_identifier} {method_name} 통신 오류: 상태={status}, 메시지={msg}"
                    logger.error(error_msg)
                    return status, msg
                
                return 0, "Success"
            except Exception as e:
                # 예외 발생 시에도 _get_clsid_ 대신 안전한 식별자 사용
                obj_identifier = getattr(com_object, '__class__', None).__name__ or str(type(com_object))
                logger.error(f"COM 객체 {obj_identifier} {method_name} 실행 중 예외 발생: {e}", exc_info=True)
                return -1, f"내부 예외 발생: {str(e)}" # -1은 내부 예외를 나타내는 임의의 코드

    def _init_conclusion_subscriber(self):
        """초기화 시 주문 체결 실시간 구독을 시작합니다."""
        with CreonAPIClient._realtime_sub_lock: # 실시간 구독 락으로 보호
            if not self.conclusion_subscriber:
                self.conclusion_subscriber = ConclusionSubscriber()
                self.conclusion_subscriber.Subscribe(self) # parent로 자신(CreonAPIClient)을 전달
            logger.info("주문 체결 실시간 구독 초기화 완료.")

    def set_conclusion_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """실시간 체결/주문 응답 이벤트를 수신할 콜백 함수를 등록합니다."""
        self.conclusion_callback = callback
        logger.info("체결/주문 응답 콜백 함수 등록 완료.")

    def set_price_update_callback(self, callback: Callable[[str, int, float], None]):
        """실시간 현재가 업데이트 이벤트를 수신할 콜백 함수를 등록합니다."""
        self.price_update_callback = callback
        logger.info("현재가 업데이트 콜백 함수 등록 완료.")

    def set_bid_update_callback(self, callback: Callable[[str, List[int], List[int], List[int], List[int]], None]):
        """실시간 10차 호가 업데이트 이벤트를 수신할 콜백 함수를 등록합니다."""
        self.bid_update_callback = callback
        logger.info("10차 호가 업데이트 콜백 함수 등록 완료.")

    def subscribe_realtime_price(self, stock_code: str):
        """
        특정 종목의 실시간 현재가 (StockCur)를 구독합니다.
        Creon API의 특성을 고려하여, 기존 구독을 모두 해지하고 새로운 목록으로 재구독합니다.
        """
        with CreonAPIClient._realtime_sub_lock: # 실시간 구독 락으로 보호
            if stock_code in self._active_cur_subscriptions:
                logger.warning(f"종목 [{stock_code}]의 현재가 실시간 구독이 이미 활성화되어 있습니다. 재구독을 시도합니다.")
            
            # 새로운 종목을 활성 구독 목록에 추가
            self._active_cur_subscriptions.add(stock_code)
            
            # 기존 현재가 구독 객체들을 모두 해지
            for s_code in list(self.stock_cur_subscribers.keys()):
                self.stock_cur_subscribers[s_code].Unsubscribe()
                del self.stock_cur_subscribers[s_code]
                if s_code in self._last_price_print_time_per_stock:
                    del self._last_price_print_time_per_stock[s_code]
            
            # 활성 구독 목록에 있는 모든 종목을 다시 구독
            for s_code in self._active_cur_subscriptions:
                subscriber = StockCurSubscriber()
                subscriber.Subscribe(self, s_code)
                self.stock_cur_subscribers[s_code] = subscriber
                self._last_price_print_time_per_stock[s_code] = 0.0 # 출력 시간 초기화
            
            logger.info(f"종목 [{stock_code}] 포함, 총 {len(self._active_cur_subscriptions)}개 종목 현재가 실시간 재구독 완료.")


    def unsubscribe_realtime_price(self, stock_code: str):
        """
        특정 종목의 실시간 현재가 (StockCur) 구독을 해지합니다.
        Creon API의 특성을 고려하여, 해당 종목을 제외한 나머지 종목들을 재구독합니다.
        """
        with CreonAPIClient._realtime_sub_lock: # 실시간 구독 락으로 보호
            if stock_code not in self._active_cur_subscriptions:
                logger.warning(f"종목 [{stock_code}]의 현재가 실시간 구독이 활성화되어 있지 않습니다.")
                return

            # 활성 구독 목록에서 해당 종목 제거
            self._active_cur_subscriptions.discard(stock_code) # remove 대신 discard 사용 (없을 경우 에러 방지)

            # 기존 현재가 구독 객체들을 모두 해지
            for s_code in list(self.stock_cur_subscribers.keys()):
                self.stock_cur_subscribers[s_code].Unsubscribe()
                del self.stock_cur_subscribers[s_code]
                if s_code in self._last_price_print_time_per_stock:
                    del self._last_price_print_time_per_stock[s_code]

            # 남은 활성 구독 목록에 있는 모든 종목을 다시 구독
            if self._active_cur_subscriptions:
                for s_code in self._active_cur_subscriptions:
                    subscriber = StockCurSubscriber()
                    subscriber.Subscribe(self, s_code)
                    self.stock_cur_subscribers[s_code] = subscriber
                    self._last_price_print_time_per_stock[s_code] = 0.0 # 출력 시간 초기화
                logger.info(f"종목 [{stock_code}] 해지 후, 총 {len(self._active_cur_subscriptions)}개 종목 현재가 실시간 재구독 완료.")
            else:
                logger.info(f"종목 [{stock_code}] 해지 후, 현재가 실시간 구독 중인 종목이 없습니다.")


    def subscribe_realtime_bid(self, stock_code: str):
        """
        특정 종목의 실시간 10차 호가 (StockJpBid)를 구독합니다.
        Creon API의 특성을 고려하여, 기존 구독을 모두 해지하고 새로운 목록으로 재구독합니다.
        """
        with CreonAPIClient._realtime_sub_lock: # 실시간 구독 락으로 보호
            if stock_code in self._active_bid_subscriptions:
                logger.warning(f"종목 [{stock_code}]의 10차 호가 실시간 구독이 이미 활성화되어 있습니다. 재구독을 시도합니다.")
            
            # 새로운 종목을 활성 구독 목록에 추가
            self._active_bid_subscriptions.add(stock_code)
            
            # 기존 호가 구독 객체들을 모두 해지
            for s_code in list(self.stock_bid_subscribers.keys()):
                self.stock_bid_subscribers[s_code].Unsubscribe()
                del self.stock_bid_subscribers[s_code]
            
            # 활성 구독 목록에 있는 모든 종목을 다시 구독
            for s_code in self._active_bid_subscriptions:
                subscriber = StockBidSubscriber()
                subscriber.Subscribe(self, s_code)
                self.stock_bid_subscribers[s_code] = subscriber
            
            logger.info(f"종목 [{stock_code}] 포함, 총 {len(self._active_bid_subscriptions)}개 종목 10차 호가 실시간 재구독 완료.")


    def unsubscribe_realtime_bid(self, stock_code: str):
        """
        특정 종목의 실시간 10차 호가 (StockJpBid) 구독을 해지합니다.
        Creon API의 특성을 고려하여, 해당 종목을 제외한 나머지 종목들을 재구독합니다.
        """
        with CreonAPIClient._realtime_sub_lock: # 실시간 구독 락으로 보호
            if stock_code not in self._active_bid_subscriptions:
                logger.warning(f"종목 [{stock_code}]의 10차 호가 실시간 구독이 활성화되어 있지 않습니다.")
                return

            # 활성 구독 목록에서 해당 종목 제거
            self._active_bid_subscriptions.discard(stock_code)

            # 기존 호가 구독 객체들을 모두 해지
            for s_code in list(self.stock_bid_subscribers.keys()):
                self.stock_bid_subscribers[s_code].Unsubscribe()
                del self.stock_bid_subscribers[s_code]

            # 남은 활성 구독 목록에 있는 모든 종목을 다시 구독
            if self._active_bid_subscriptions:
                for s_code in self._active_bid_subscriptions:
                    subscriber = StockBidSubscriber()
                    subscriber.Subscribe(self, s_code)
                    self.stock_bid_subscribers[s_code] = subscriber
                logger.info(f"종목 [{stock_code}] 해지 후, 총 {len(self._active_bid_subscriptions)}개 종목 10차 호가 실시간 재구독 완료.")
            else:
                logger.info(f"종목 [{stock_code}] 해지 후, 10차 호가 실시간 구독 중인 종목이 없습니다.")


    def unsubscribe_all_realtime_data(self):
        """모든 실시간 현재가 및 호가 구독을 해지합니다."""
        with CreonAPIClient._realtime_sub_lock: # 실시간 구독 락으로 보호
            # 현재가 구독 해지
            for stock_code in list(self.stock_cur_subscribers.keys()):
                self.stock_cur_subscribers[stock_code].Unsubscribe()
            self.stock_cur_subscribers.clear()
            self._active_cur_subscriptions.clear()
            
            # 호가 구독 해지
            for stock_code in list(self.stock_bid_subscribers.keys()):
                self.stock_bid_subscribers[stock_code].Unsubscribe()
            self.stock_bid_subscribers.clear()
            self._active_bid_subscriptions.clear()

            logger.info("모든 종목의 실시간 현재가/호가 구독 해지 완료.")

    def send_order(self, 
                   stock_code: str, 
                   order_type: OrderType, 
                   amount: int, 
                   price: int = 0, 
                   org_order_num: Optional[int] = 0, # 정정/취소 시 원주문번호
                   order_condition: str = "0", # 0:기본, 1:IOC, 2:FOK
                   order_unit: str = "01" # 01:보통
                   ) -> Dict[str, Any]:
        """
        주식 주문 (매수, 매도, 정정, 취소)을 전송합니다.
        실시간주문체결.py의 CpRPOrder.buyOrder/modifyOrder/cancelOrder와 유사한 로직
        :param stock_code: 종목 코드
        :param order_type: OrderType Enum (BUY, SELL, MODIFY, CANCEL)
        :param amount: 주문 수량
        :param price: 주문 단가 (정정/취소 시 0이면 잔량 정정/취소)
        :param org_order_num: 정정/취소 시 원 주문 번호
        :param order_condition: 주문 조건 구분 코드 (0:기본, 1:IOC, 2:FOK)
        :param order_unit: 주문호가 구분코드 (01:보통)
        :return: { 'status': 'success'/'fail', 'message': str, 'order_num': Optional[int] }
        """
        logger.info(f"주문 요청 - 유형: {order_type.name}, 종목: {stock_code}, 수량: {amount}, 가격: {price}, 원주문번호: {org_order_num}")

        com_obj = None
        if order_type == OrderType.BUY or order_type == OrderType.SELL:
            com_obj = win32com.client.Dispatch("CpTrade.CpTd0311") # 신규 매수/매도
            com_obj.SetInputValue(0, order_type.value) # 1: 매도, 2: 매수
            com_obj.SetInputValue(1, self.account_number) # 계좌번호
            com_obj.SetInputValue(2, self.account_flag)   # 상품구분
            com_obj.SetInputValue(3, stock_code)          # 종목코드
            com_obj.SetInputValue(4, amount)  # 주문 수량
            com_obj.SetInputValue(5, price)   # 주문 단가 
            com_obj.SetInputValue(7, order_condition)  
            com_obj.SetInputValue(8, order_unit)
        elif order_type == OrderType.MODIFY:
            if not org_order_num:
                logger.error("정정 주문은 원 주문 번호가 필수입니다.")
                return {'status': 'fail', 'message': '정정 주문은 원 주문 번호 필수', 'order_num': None}
            com_obj = win32com.client.Dispatch("CpTrade.CpTd0313") # 정정
            com_obj.SetInputValue(1, org_order_num) # 원주문 번호
            com_obj.SetInputValue(2, self.account_number) # 계좌번호
            com_obj.SetInputValue(3, self.account_flag)   # 상품구분
            com_obj.SetInputValue(4, stock_code)          # 종목코드
            com_obj.SetInputValue(5, amount) # 정정 수량 (0이면 잔량 정정)
            com_obj.SetInputValue(6, price)  # 정정 단가
        elif order_type == OrderType.CANCEL:
            if not org_order_num:
                logger.error("취소 주문은 원 주문 번호가 필수입니다.")
                return {'status': 'fail', 'message': '취소 주문은 원 주문 번호 필수', 'order_num': None}
            com_obj = win32com.client.Dispatch("CpTrade.CpTd0314") # 취소
            com_obj.SetInputValue(1, org_order_num) # 원주문 번호
            com_obj.SetInputValue(2, self.account_number) # 계좌번호
            com_obj.SetInputValue(3, self.account_flag)   # 상품구분
            com_obj.SetInputValue(4, stock_code)          # 종목코드
            com_obj.SetInputValue(5, amount) # 취소 수량 (0이면 잔량 취소)
        else:
            logger.error(f"지원하지 않는 주문 유형: {order_type}")
            return {'status': 'fail', 'message': '지원하지 않는 주문 유형', 'order_num': None}

        status_code, message = self._execute_block_request(com_obj)

        if status_code != 0:
            logger.error(f"주문 실패: 유형={order_type.name}, 종목={stock_code}, 메시지={message}")
            return {'status': 'fail', 'message': message, 'order_num': None}

        result_order_num = org_order_num # 기본값은 원주문번호 (취소/정정 시)
        if order_type == OrderType.BUY or order_type == OrderType.SELL:
            result_order_num = com_obj.GetHeaderValue(8) # 신규 주문번호
        elif order_type == OrderType.MODIFY:
            result_order_num = com_obj.GetHeaderValue(7) # 정정 후 새로 부여된 주문번호

        logger.info(f"주문 성공: 유형={order_type.name}, 주문번호={result_order_num}, 메시지={message}")
        return {'status': 'success', 'message': message, 'order_num': result_order_num}

    def get_current_price_and_quotes(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        특정 종목의 현재가(종가), 10차 호가 및 각 호가의 잔량을 조회합니다.
        실시간주문체결.py의 CpRPCurrentPrice.Request와 유사한 로직
        :param stock_code: 종목 코드
        :return: {
            'current_price': int,
            'offer_prices': List[int], 'bid_prices': List[int],
            'offer_amounts': List[int], 'bid_amounts': List[int],
            'stock_name': str, 'time': str, 'open_price': int, 'high_price': int, 'low_price': int,
            'volume': int, 'diff': int, 'diff_rate': float
        } 형태의 딕셔너리 또는 None
        """
        logger.info(f"종목 [{stock_code}] 현재가 및 10차 호가 상세 조회 중...")
        
        # 1. 현재가 기본 정보 조회 (StockMst)
        obj_stock_mst = win32com.client.Dispatch("DsCbo1.StockMst")
        obj_stock_mst.SetInputValue(0, stock_code)
        status_code_mst, message_mst = self._execute_block_request(obj_stock_mst)

        if status_code_mst != 0:
            logger.error(f"StockMst 요청 실패: {message_mst}")
            return None
        
        # 기본 정보 추출
        current_price_data = {
            'stock_name': obj_stock_mst.GetHeaderValue(1),
            'time': obj_stock_mst.GetHeaderValue(4),
            'current_price': obj_stock_mst.GetHeaderValue(11), # 종가 (현재가)
            'open_price': obj_stock_mst.GetHeaderValue(13),
            'high_price': obj_stock_mst.GetHeaderValue(14),
            'low_price': obj_stock_mst.GetHeaderValue(15),
            'volume': obj_stock_mst.GetHeaderValue(18),
            'diff': obj_stock_mst.GetHeaderValue(2),
            'diff_rate': obj_stock_mst.GetHeaderValue(3),
        }

        # 2. 10차 호가 및 잔량 조회 (StockJpBid2)
        obj_stock_jpbid2 = win32com.client.Dispatch("DsCbo1.StockJpBid2")
        obj_stock_jpbid2.SetInputValue(0, stock_code)
        status_code_bid, message_bid = self._execute_block_request(obj_stock_jpbid2)

        if status_code_bid != 0:
            logger.error(f"StockJpBid2 요청 실패: {message_bid}")
            # 호가 정보만 실패한 경우, 현재가 정보라도 반환할지 결정
            # 여기서는 완전한 정보가 아니므로 None 반환
            return None 

        # 10차 호가 및 잔량 추출 (실시간주문체결.py의 StockBid 이벤트와 동일한 인덱스 사용)
        offer_prices = [obj_stock_jpbid2.GetDataValue(0, i) for i in range(10)]
        bid_prices = [obj_stock_jpbid2.GetDataValue(1, i) for i in range(10)]
        
        # 실시간주문체결.py의 CpEvent.OnReceived (stockbid)에서 사용되는 인덱스를 참고
        # 매도호가잔량: 20, 22, ... 38 (10개)
        # 매수호가잔량: 21, 23, ... 39 (10개)
        # StockJpBid2의 필드는 0:매도호가, 1:매수호가, 2:매도호가잔량, 3:매수호가잔량
        offer_amounts = [obj_stock_jpbid2.GetDataValue(2, i) for i in range(10)]
        bid_amounts = [obj_stock_jpbid2.GetDataValue(3, i) for i in range(10)]


        # 모든 정보 합치기
        result = {
            **current_price_data, # 현재가 기본 정보
            'offer_prices': offer_prices,
            'bid_prices': bid_prices,
            'offer_amounts': offer_amounts,
            'bid_amounts': bid_amounts
        }
        
        logger.info(f"종목 [{stock_code}] 현재가 및 10차 호가 상세 조회 완료. 현재가: {result['current_price']}, 1차 매도/매수 호가: {offer_prices[0]}/{bid_prices[0]}")
        return result

    def get_account_balance(self) -> Optional[Dict[str, Any]]:
        """
        계좌의 현금 잔고 및 예수금 정보를 조회합니다.
        :return: 잔고 정보를 담은 딕셔너리 또는 None
        """
        logger.info("계좌 잔고 조회 중...")
        obj_cash = win32com.client.Dispatch("CpTrade.CpTdNew5331A")
        obj_cash.SetInputValue(0, self.account_number)
        obj_cash.SetInputValue(1, self.account_flag)
        
        status_code, message = self._execute_block_request(obj_cash)
        if status_code != 0:
            return None

        balance_data = {
            'cash_balance': obj_cash.GetHeaderValue(9),  # 주문가능현금
            'deposit': obj_cash.GetHeaderValue(10),     # 예수금
            'withdrawal_possible': obj_cash.GetHeaderValue(11), # 인출가능금액
            'loan_amount': obj_cash.GetHeaderValue(12), # 대출금액
        }
        logger.info(f"계좌 잔고 조회 완료: 예수금 {balance_data['deposit']:,.0f}원")
        return balance_data

    def get_portfolio_positions(self) -> List[Dict[str, Any]]:
        """
        현재 보유 종목 정보를 조회합니다.
        :return: 보유 종목 리스트 (딕셔너리 형태)
        """
        logger.info("보유 종목 조회 중...")
        obj_pos = win32com.client.Dispatch("CpTrade.CpTd6033")
        obj_pos.SetInputValue(0, self.account_number)
        obj_pos.SetInputValue(1, self.account_flag)
        obj_pos.SetInputValue(2, 50) # 요청 개수 (최대 50개)
        
        positions = []
        while True:
            status_code, message = self._execute_block_request(obj_pos)
            if status_code != 0:
                logger.error(f"보유 종목 조회 실패: {message}")
                break

            cnt = obj_pos.GetHeaderValue(7) # 수신 개수
            if not isinstance(cnt, int) or cnt <= 0: # cnt가 None이거나 유효하지 않은 경우 처리
                logger.info(f"보유 종목 수신 개수가 유효하지 않거나 0입니다 (cnt: {cnt}). 더 이상 조회할 보유 종목이 없습니다.")
                break

            for i in range(cnt):
                try:
                    positions.append({
                        'stock_code': obj_pos.GetDataValue(12, i), # 종목코드
                        'stock_name': obj_pos.GetDataValue(0, i), # 종목명
                        'quantity': int(obj_pos.GetDataValue(7, i)), # 현재 잔고 수량
                        'average_buy_price': float(obj_pos.GetDataValue(9, i)), # 매입단가
                        'eval_amt': float(obj_pos.GetDataValue(9, i) * obj_pos.GetDataValue(7, i)), # 평가금액 (간단 계산)
                        'eval_profit_loss': float(obj_pos.GetDataValue(10, i)), # 평가손익
                        'eval_return_rate': float(obj_pos.GetDataValue(11, i)), # 수익률
                        'sell_avail_qty': int(obj_pos.GetDataValue(15, i)), # 매도 가능 수량
                        'entry_date': datetime.now().date() # API에서 진입일자 제공하지 않으면 현재 날짜로 임시 설정
                    })
                except Exception as data_e:
                    logger.error(f"보유 종목 데이터 처리 중 오류 발생 (인덱스 {i}): {data_e}", exc_info=True)
                    continue

            if not obj_pos.Continue: # 다음 데이터가 없으면 종료
                break
        
        logger.info(f"총 {len(positions)}개의 보유 종목 조회 완료.")
        return positions

    def get_unfilled_orders(self) -> List[Dict[str, Any]]:
        """
        미체결 주문 정보를 조회합니다.
        :return: 미체결 주문 리스트
        """
        logger.info("미체결 주문 조회 중...")
        obj_unfilled = win32com.client.Dispatch("CpTrade.CpTd5339")
        obj_unfilled.SetInputValue(0, self.account_number)
        obj_unfilled.SetInputValue(1, self.account_flag)
        obj_unfilled.SetInputValue(2, 50) # 요청 개수 (최대 50개)

        unfilled_orders = []
        while True:
            status_code, message = self._execute_block_request(obj_unfilled)
            if status_code != 0:
                logger.error(f"미체결 주문 조회 실패: {message}")
                break

            # GetHeaderValue(7) can sometimes return None if no data or invalid data is present
            cnt = obj_unfilled.GetHeaderValue(7) 
            
            # Explicitly check if cnt is None or not an integer. Treat as 0 if invalid.
            if not isinstance(cnt, int) or cnt <= 0:
                logger.info(f"미체결 주문 수신 개수가 유효하지 않거나 0입니다 (cnt: {cnt}). 더 이상 조회할 미체결 주문이 없습니다.")
                break

            for i in range(cnt):
                try: # Add try-except for individual data access as well for robustness
                    mod_avali = obj_unfilled.GetDataValue(9, i) # 정정/취소 가능 수량 (미체결 수량)
                    if mod_avali > 0: # 미체결 수량이 0보다 큰 주문만 포함
                        unfilled_orders.append({
                            'order_id': obj_unfilled.GetDataValue(5, i),      # 주문번호
                            'stock_code': obj_unfilled.GetDataValue(12, i),   # 종목코드
                            'stock_name': obj_unfilled.GetDataValue(0, i),    # 종목명
                            'side': 'sell' if obj_unfilled.GetDataValue(13, i) == '1' else 'buy',  # 매수/매도 구분
                            'quantity': obj_unfilled.GetDataValue(7, i),      # 원 주문 수량
                            'price': obj_unfilled.GetDataValue(6, i),         # 주문 단가
                            'filled_quantity': obj_unfilled.GetDataValue(8, i),  # 체결 수량
                            'unfilled_quantity': mod_avali,              # 미체결 수량
                            'order_time': obj_unfilled.GetDataValue(3, i),    # 주문 시간
                            'credit_type': obj_unfilled.GetDataValue(1, i),   # 신용 구분
                            'order_flag': obj_unfilled.GetDataValue(14, i),   # 주문호가 구분
                            'timestamp': datetime.now()
                        })
                except Exception as data_e:
                    logger.error(f"미체결 주문 데이터 처리 중 오류 발생 (인덱스 {i}): {data_e}", exc_info=True)
                    # Continue to next item or break, depending on desired robustness.
                    # For now, just log and continue to avoid crashing for one bad data point.
                    continue
            
            if not obj_unfilled.Continue: # 다음 데이터가 없으면 종료
                break
        
        logger.info(f"총 {len(unfilled_orders)}개의 미체결 주문 조회 완료.")
        return unfilled_orders

    def cleanup(self) -> None:
        """리소스 정리 및 실시간 구독 해지."""
        try:
            if self.conclusion_subscriber:
                self.conclusion_subscriber.Unsubscribe()
                self.conclusion_subscriber = None
            
            self.unsubscribe_all_realtime_data() # 모든 실시간 현재가/호가 구독 해지

            # Creon Cybos Plus 연결 해제는 사용자 요청에 따라 제거
            # if self.obj_cp_cybos and self.obj_cp_cybos.IsConnect:
            #     self.obj_cp_cybos.Disconnect()
            #     logger.info("Creon Plus 연결 해제 완료.")

            # 모든 COM 객체 명시적으로 해제하여 프로세스 종료를 돕습니다.
            if self.obj_trade:
                self.obj_trade = None
                logger.debug("CpTrade.CpTdUtil COM 객체 해제.")
            if self.obj_cp_cybos:
                self.obj_cp_cybos = None
                logger.debug("CpUtil.CpCybos COM 객체 해제.")

            # 콜백 함수 정리
            self.conclusion_callback = None
            self.price_update_callback = None
            self.bid_update_callback = None

            logger.info("CreonAPIClient 리소스 정리 완료.")
        except Exception as e:
            logger.error(f"CreonAPIClient 리소스 정리 중 오류 발생: {e}", exc_info=True)

    def __del__(self):
        """소멸자: 객체 소멸 시 cleanup 호출."""
        self.cleanup()

    # --- 유틸리티 메서드 (기존 코드에서 이동) ---
    def round_to_tick(self, price: float) -> int:
        """가격을 호가 단위에 맞춰 반올림합니다."""
        price = int(price)
        if price < 1000: return round(price)
        elif price < 2000: return round(price)
        elif price < 5000: return round(price / 5) * 5
        elif price < 10000: return round(price / 10) * 10
        elif price < 20000: return round(price / 10) * 10
        elif price < 50000: return round(price / 50) * 50
        elif price < 100000: return round(price / 100) * 100
        elif price < 200000: return round(price / 100) * 100
        elif price < 500000: return round(price / 500) * 500
        else: return round(price / 1000) * 1000

    def is_connected(self) -> bool:
        """Creon API 연결 상태를 반환합니다."""
        return self.connected
