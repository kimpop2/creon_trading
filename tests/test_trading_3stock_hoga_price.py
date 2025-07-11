import unittest
import sys
import os
import logging
import time
# import queue # Removed threading related imports
# import threading # Removed threading related imports
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Callable, Tuple

# win32com 관련 CoInitialize/CoUninitialize 임포트
import pythoncom # COM 메시지 펌핑을 위해 필요

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

class TestRealtimeDataSubscription(unittest.TestCase):
    """
    CreonAPIClient를 사용하여 여러 종목의 실시간 현재가 및 호가 데이터 구독을 테스트합니다.
    매매 로직 없이 데이터 수신 여부에만 초점을 맞춥니다. (스레드 미사용, COM 메시지 펌핑 포함)
    """
    cls_api: CreonAPIClient = None
    
    # 테스트 대상 종목 리스트
    TEST_STOCK_CODES = ['A090710', 'A008970', 'A032820'] # 휴림로봇, 동양철관, 우리기술

    # 실시간 이벤트 수신 횟수 추적
    _received_price_updates: Dict[str, int] = {} # 종목별 현재가 업데이트 횟수
    _received_bid_updates: Dict[str, int] = {}   # 종목별 호가 업데이트 횟수

    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행: CreonAPIClient 초기화 및 실시간 구독 설정"""
        logger.info("--- 실시간 데이터 구독 테스트 설정 시작 (스레드 미사용, COM 메시지 펌핑) ---")
        try:
            # COM 라이브러리 초기화 (현재 스레드에서 COM 객체를 사용하기 전에 호출 필요)
            pythoncom.CoInitialize()

            # CreonAPIClient 인스턴스 생성
            cls.cls_api = CreonAPIClient()
            
            # 각 종목별 카운터 초기화
            for code in cls.TEST_STOCK_CODES:
                cls._received_price_updates[code] = 0
                cls._received_bid_updates[code] = 0

            # CreonAPIClient에 콜백 함수 등록
            cls.cls_api.set_price_update_callback(cls._price_update_callback)
            cls.cls_api.set_bid_update_callback(cls._bid_update_callback)
            # 체결 콜백은 이 테스트에서 필요 없지만, API 클라이언트가 초기화 시 구독하므로 등록은 유지
            cls.cls_api.set_conclusion_callback(cls._conclusion_callback) 
            logger.info("CreonAPIClient 초기화 및 콜백 등록 완료.")

            # 모든 테스트 종목에 대해 실시간 현재가 및 호가 구독 시작
            for stock_code in cls.TEST_STOCK_CODES:
                cls.cls_api.subscribe_realtime_price(stock_code)
                cls.cls_api.subscribe_realtime_bid(stock_code)
                logger.info(f"종목 [{stock_code}] 현재가 및 호가 실시간 구독 요청 완료.")
            
            logger.info("초기 실시간 구독 요청 완료. 데이터 수신을 위해 잠시 대기합니다 (COM 메시지 펌핑)...")
            
            # 초기 데이터 흐름을 위해 충분한 시간 대기하면서 메시지 펌핑
            initial_wait_time = 10 # 초
            start_time = time.time()
            while time.time() - start_time < initial_wait_time:
                pythoncom.PumpWaitingMessages() # 대기 중인 COM 메시지 처리
                time.sleep(0.1) # 짧게 대기

        except Exception as e:
            logger.error(f"테스트 설정 실패: {e}", exc_info=True)
            cls.cls_api = None # 실패 시 API 객체 None으로 설정하여 tearDownClass에서 오류 방지
            raise # 예외를 다시 발생시켜 테스트 실패를 알림
        logger.info("--- 실시간 데이터 구독 테스트 설정 완료 (스레드 미사용, COM 메시지 펌핑) ---")

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시 한 번만 실행: 리소스 정리 및 모든 실시간 구독 해지"""
        logger.info("--- 실시간 데이터 구독 테스트 정리 시작 (스레드 미사용, COM 메시지 펌핑) ---")
        if cls.cls_api:
            cls.cls_api.unsubscribe_all_realtime_data() # 모든 실시간 구독 해지
            cls.cls_api.cleanup() # CreonAPIClient 리소스 정리
        
        # COM 라이브러리 정리
        pythoncom.CoUninitialize()
        logger.info("--- 실시간 데이터 구독 테스트 정리 완료 (스레드 미사용, COM 메시지 펌핑) ---")

    @classmethod
    def _price_update_callback(cls, stock_code: str, current_price: int, timestamp: float):
        """실시간 현재가 업데이트 콜백 핸들러"""
        logger.info(f"실시간 현재가 수신: 종목={stock_code}, 가격={current_price:,.0f}원")
        cls._received_price_updates[stock_code] += 1 # 수신 횟수 증가

    @classmethod
    def _bid_update_callback(cls, stock_code: str, offer_prices: List[int], bid_prices: List[int], offer_amounts: List[int], bid_amounts: List[int]):
        """실시간 10차 호가 업데이트 콜백 핸들러"""
        logger.info(f"실시간 호가 수신: 종목={stock_code}, 1차 매도={offer_prices[0]}, 1차 매수={bid_prices[0]}")
        cls._received_bid_updates[stock_code] += 1 # 수신 횟수 증가

    @classmethod
    def _conclusion_callback(cls, data: Dict[str, Any]):
        """실시간 체결/주문 응답 콜백 핸들러 (이 테스트에서는 로깅만)"""
        logger.info(f"실시간 체결/응답 수신: {data['flag']} {data['buy_sell']} 종목:{data['code']} 주문번호:{data['order_num']}")


    def test_realtime_data_reception(self):
        """
        실시간 현재가 및 호가 데이터가 각 종목에 대해 성공적으로 수신되는지 확인합니다.
        """
        self.assertIsNotNone(self.cls_api, "CreonAPIClient가 초기화되지 않았습니다. setUpClass를 확인하세요.")
        self.assertTrue(self.cls_api.is_connected(), "크레온 PLUS가 연결되어 있지 않습니다. HTS 로그인 상태를 확인하세요.")

        logger.info("\n--- 실시간 데이터 수신 테스트 시작 (스레드 미사용, COM 메시지 펌핑) ---")

        test_duration = 60 # 초: 데이터를 충분히 수신할 수 있도록 대기 시간 설정
        logger.info(f"{test_duration}초 동안 실시간 데이터 수신을 대기합니다 (COM 메시지 펌핑)...")
        
        start_time = time.time()
        while time.time() - start_time < test_duration:
            pythoncom.PumpWaitingMessages() # 대기 중인 COM 메시지 처리
            time.sleep(0.1) # 짧게 대기하여 CPU 과부하 방지

        # 각 종목별로 실시간 데이터 수신 여부 및 횟수 확인
        for stock_code in self.TEST_STOCK_CODES:
            # 현재가 업데이트가 최소 한 번 이상 수신되었는지 확인
            self.assertGreater(self._received_price_updates[stock_code], 0, 
                               f"종목 [{stock_code}]에 대한 현재가 업데이트가 수신되지 않았습니다.")
            logger.info(f"종목 [{stock_code}]: 현재가 업데이트 {self._received_price_updates[stock_code]}회 수신됨.")

            # 호가 업데이트가 최소 한 번 이상 수신되었는지 확인
            self.assertGreater(self._received_bid_updates[stock_code], 0, 
                               f"종목 [{stock_code}]에 대한 호가 업데이트가 수신되지 않았습니다.")
            logger.info(f"종목 [{stock_code}]: 호가 업데이트 {self._received_bid_updates[stock_code]}회 수신됨.")
        
        logger.info("\n--- 실시간 데이터 수신 테스트 성공적으로 완료 (스레드 미사용, COM 메시지 펌핑) ---")

if __name__ == '__main__':
    unittest.main()
