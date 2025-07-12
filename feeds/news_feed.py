# feeds/news_feed.py

import logging
import time
from datetime import datetime, date, timedelta
import sys
import os
import win32com.client  # Creon API 사용
import ctypes  # 관리자 권한 확인용
import re
from typing import Dict, Any, List, Optional

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from feeds.base_feed import BaseFeed
from feeds.db_feed import DBFeed

# 로거 설정
logger = logging.getLogger(__name__)

# --- Creon API 관련 전역 객체 및 유틸리티 함수 (News_뉴스수신.py에서 통합) ---
# g_objCodeMgr과 g_objCpStatus는 전역으로 한 번만 Dispatch
g_objCodeMgr = None
g_objCpStatus = None

def init_creon_plus_check() -> bool:
    """
    Creon Plus의 연결 상태 및 관리자 권한 실행 여부를 확인합니다.
    """
    global g_objCodeMgr, g_objCpStatus
    if g_objCodeMgr is None:
        g_objCodeMgr = win32com.client.Dispatch('CpUtil.CpCodeMgr')
    if g_objCpStatus is None:
        g_objCpStatus = win32com.client.Dispatch('CpUtil.CpCybos')

    # 프로세스가 관리자 권한으로 실행 여부
    if not ctypes.windll.shell32.IsUserAnAdmin():
        logger.error('오류: 일반권한으로 실행됨. 관리자 권한으로 실행해 주세요')
        return False

    # 연결 여부 체크
    if g_objCpStatus.IsConnect == 0:
        logger.error("PLUS가 정상적으로 연결되지 않음.")
        return False

    logger.info("Creon Plus 연결 및 관리자 권한 확인 완료.")
    return True

def preprocess_news_message(text: str) -> str:
    """
    뉴스 메시지 전처리 함수.
    특수문자 제거, 숫자 시작 단어 제거, 공백 정규화 등을 수행합니다.
    """
    text = text[:500]  # 너무 긴 내용은 잘라냄 (DB 컬럼 길이 고려)
    text = re.sub(r'[….,·\-\"\'()\[\]{}<>~!?@#$%^&*“”\\|/+=_—–▶️▶️]', ' ', text)  # 특수문자 제거 후 공백 추가
    text = re.sub(r'\b\d+\w*\b', '', text)  # 숫자로 시작하는 단어 제거 (예: 2024년, 1000원)
    text = re.sub(r'\s+-\s+\S+$', '', text)  # 마지막 언론사 제거 (예: " - 연합뉴스")
    text = re.sub(r'::\s+\S+\s+::', '', text)  # 특정 패턴의 언론사 제거 (예: ":: 서울경제 ::")
    text = re.sub(r'\s+', ' ', text).strip()  # 2개 이상 공백을 1개로 변경하고 양쪽 공백 제거
    return text

def regex_korean_parser(text: str) -> List[str]:
    """
    주식 뉴스 관련 조사를 제거하는 함수.
    """
    words = text.split()
    results = []
    for word in words:
        # 주식 뉴스 관련 조사를 포함한 정규 표현식
        word = re.sub(r'(의|은|는|이|가|을|를|과|와|에|에서|에게|께|부터|까지|것|만|도|조차|마저|요|지요|든|처럼|같이|보다|로|으로|까지|부터|때문|따라|대해|향해|관련|대한|통해)$', '', word)
        results.append(word)
    return results

def is_valid_word(word: str) -> bool:
    """
    유효한 단어(한국어 또는 영어) 여부를 확인합니다.
    """
    word_pattern = re.compile(r'^[가-힣a-zA-Z0-9]+$')
    return word_pattern.match(word) is not None

# --- Creon 실시간 이벤트 수신 클래스 (News_뉴스수신.py에서 통합) ---
class CpEvent:
    """
    PLUS로부터 실시간 이벤트(체결/주문 응답/시세 이벤트 등)를 수신 받아 처리하는 기본 클래스.
    이 클래스는 win32com.client.WithEvents에 의해 동적으로 사용됩니다.
    """
    def set_params(self, client: Any, name: str, caller: Any):
        self.client = client  # CP 실시간 통신 object
        self.name = name      # 서비스가 다른 이벤트를 구분하기 위한 이름 (예: 'marketnews')
        self.caller = caller  # 콜백을 위해 NewsFeed 인스턴스를 보관

    def OnReceived(self):
        """
        PLUS로부터 실제로 이벤트가 수신될 때 호출되는 메서드.
        이 메서드는 NewsFeed 클래스의 _telegram_message_callback과 유사하게 동작하여
        수신된 데이터를 NewsFeed의 버퍼에 추가합니다.
        """
        if self.name == 'marketnews':
            item = {}
            # GetHeaderValue(0): 업데이트 구분 (D: 삭제)
            # GetHeaderValue(1): 종목 코드
            # GetHeaderValue(2): 시간 (HHMM)
            # GetHeaderValue(3): 뉴스 카테고리
            # GetHeaderValue(4): 뉴스 내용 (제목)
            
            update_type = self.client.GetHeaderValue(0)
            code = self.client.GetHeaderValue(1)
            time_val = self.client.GetHeaderValue(2)
            # cate = self.client.GetHeaderValue(3) # 현재 사용 안 함
            content = self.client.GetHeaderValue(4) # 뉴스 내용이 여기에 옴

            # 시간 포맷팅
            h, m = divmod(time_val, 100)
            # 현재 날짜에 수신된 시간 정보를 결합
            event_datetime = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)

            news_title = preprocess_news_message(content) # 전처리된 내용을 제목으로 사용

            item['source'] = 'Creon News'
            item['datetime'] = event_datetime
            item['title'] = news_title
            item['content'] = content # 원본 내용
            item['url'] = None # Creon 뉴스 피드에는 URL 정보가 직접 제공되지 않을 수 있음
            item['related_stocks'] = [code] if code and code != '000000' else [] # 관련 종목 코드 (전체 뉴스일 경우 빈 리스트)

            if update_type == ord('D'): # 삭제 뉴스인 경우
                item['title'] = '[삭제] ' + item['title']
                logger.info(f"[CpEvent] Deleted news: {item['title'][:50]}...")
            else:
                logger.info(f"[CpEvent] Received news: {item['title'][:50]}...")
            
            # NewsFeed 인스턴스의 버퍼에 추가
            self.caller.news_data_buffer.append(item)

class CpPublish:
    """
    Creon Plus 실시간 데이터를 구독하고 해지하는 기본 클래스.
    """
    def __init__(self, name: str, serviceID: str):
        self.name = name
        self.obj = win32com.client.Dispatch(serviceID)
        self.bIsSB = False # 구독 상태 플래그

    def Subscribe(self, var: str, caller: Any):
        """
        실시간 데이터 구독을 시작합니다.
        :param var: 구독할 입력 값 (예: 종목 코드, 빈 문자열로 전체 뉴스)
        :param caller: OnReceived 콜백을 받을 객체 (NewsFeed 인스턴스)
        """
        if self.bIsSB:
            self.Unsubscribe() # 이미 구독 중이면 해지 후 재구독

        if len(var) > 0:
            self.obj.SetInputValue(0, var) # 입력 값 설정

        # win32com.client.WithEvents를 사용하여 이벤트 핸들러 연결
        handler = win32com.client.WithEvents(self.obj, CpEvent)
        handler.set_params(self.obj, self.name, caller) # CpEvent에 파라미터 설정
        self.obj.Subscribe() # 구독 시작
        self.bIsSB = True
        logger.debug(f"Subscribed to {self.name} with var: '{var}'")

    def Unsubscribe(self):
        """
        실시간 데이터 구독을 해지합니다.
        """
        if self.bIsSB:
            self.obj.Unsubscribe()
            self.bIsSB = False
            logger.debug(f"Unsubscribed from {self.name}")

class CpPB8092news(CpPublish):
    """
    CpSysDib.CpSvr8092S (뉴스) 서비스를 위한 Publish 클래스.
    """
    def __init__(self):
        super().__init__('marketnews', 'Dscbo1.CpSvr8092S')

# --- NewsFeed 클래스 (기존 news_feed.py) ---
class NewsFeed(BaseFeed):
    """
    Creon API를 통해 실시간 뉴스 데이터를 수집하고 DB에 저장하는 Feed 모듈.
    news_raw 테이블에 데이터를 저장합니다.
    """
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        super().__init__("NewsFeed", redis_host, redis_port)
        self.api_client = CreonAPIClient()  # Creon API 클라이언트 (연결 확인용)
        self.db_feed = DBFeed()  # DBFeed 초기화

        # Creon 뉴스 수신 객체 초기화
        self.objpbNews = CpPB8092news()
        self.news_data_buffer: List[Dict[str, Any]] = []  # 수신된 뉴스 임시 저장 버퍼

        # CpEvent 콜백을 위한 내부 클래스 인스턴스 생성
        # CpEvent의 caller로 NewsFeed 인스턴스(self)를 전달하여 버퍼에 접근
        self.news_event_handler = CpEvent()
        self.news_event_handler.set_params(self.objpbNews.obj, self.objpbNews.name, self)

        logger.info("NewsFeed initialized.")

    def _subscribe_creon_news(self) -> bool:
        """Creon 실시간 뉴스 구독을 시작합니다."""
        # Creon API 연결 상태를 확인
        if not self.api_client.connected:
            logger.error("Creon API Client is not connected. Cannot subscribe to news.")
            return False

        try:
            # CpPB8092news의 Subscribe 메서드 호출
            # var 인자는 빈 문자열로 전체 뉴스 수신, caller 인자로 self를 전달하여 콜백 받음
            self.objpbNews.Subscribe("", self) # CpPublish.Subscribe에서 handler.set_params(self.obj, self.name, caller)로 전달됨
            logger.info("Subscribed to Creon real-time news.")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to Creon news: {e}", exc_info=True)
            return False

    def _process_news_buffer(self):
        """버퍼에 쌓인 뉴스 데이터를 DB에 저장하고 이벤트를 발행합니다."""
        if self.news_data_buffer:
            if self.db_feed.save_news_raw(self.news_data_buffer):
                logger.info(f"Saved {len(self.news_data_buffer)} raw news records to DB.")
                # FeedManager에게 뉴스 수집 완료 이벤트 발행
                self.publish_event('news_feed_events', {
                    'type': 'NEWS_COLLECTED',
                    'count': len(self.news_data_buffer),
                    'timestamp': datetime.now().isoformat()
                })
                self.news_data_buffer.clear()  # 버퍼 비우기
            else:
                logger.error("Failed to save raw news to DB.")

    def run(self):
        """
        NewsFeed의 메인 실행 루프.
        Creon API를 통해 실시간 뉴스를 수신하고 주기적으로 DB에 저장합니다.
        """
        logger.info(f"NewsFeed process started.")
        # Creon Plus 연결 및 관리자 권한 확인
        if not init_creon_plus_check():
            logger.error("Creon Plus is not connected or not running with admin privileges. NewsFeed cannot run.")
            return

        # Creon API 클라이언트의 연결 상태를 다시 확인 (init_creon_plus_check와는 별개)
        if not self.api_client.connected:
            logger.error("CreonAPIClient is not connected. Attempting to connect...")
            # CreonAPIClient는 내부적으로 InitPlusCheck를 호출하여 연결 시도
            # 그러나 여기서는 이미 init_creon_plus_check를 호출했으므로,
            # api_client.connected 상태만 확인하고 구독을 시도합니다.
            pass # _subscribe_creon_news에서 다시 확인하므로 여기서는 생략 가능

        if not self._subscribe_creon_news():
            logger.error("NewsFeed cannot start without Creon news subscription.")
            return

        while not self.stop_event.is_set():
            try:
                # 일정 주기마다 버퍼에 쌓인 뉴스 데이터를 DB에 저장
                self._process_news_buffer()
                self._wait_for_stop_signal(interval=5)  # 5초마다 버퍼 처리 시도
            except Exception as e:
                logger.error(f"[{self.feed_name}] Error in run loop: {e}", exc_info=True)
                time.sleep(5)

        logger.info(f"NewsFeed process stopped.")
        # 종료 시 구독 해지
        self.objpbNews.Unsubscribe()
        self.db_feed.close()

# 실행 예시 (별도 프로세스로 실행될 때)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    news_feed = NewsFeed()
    try:
        news_feed.run()
    except KeyboardInterrupt:
        logger.info("NewsFeed interrupted by user.")
    finally:
        news_feed.stop()
