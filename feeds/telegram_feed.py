# feeds/telegram_feed.py

import logging
import asyncio
import time
import threading
from datetime import datetime, timedelta
import sys
import os
import json
import re # 정규 표현식 사용 (preprocess_message 함수 등에서 사용될 가능성)
import traceback # 에러 트레이스백 로깅용
from typing import Dict, Any, List, Optional, Callable

# telethon 라이브러리 임포트
from telethon import TelegramClient, events

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from feeds.base_feed import BaseFeed
from feeds.db_feed import DBFeed

# 로거 인스턴스 생성 (모듈 이름을 로거 이름으로 사용)
logger = logging.getLogger(__name__)

# --- 유틸리티 함수 (News_뉴스수신.py에서 통합된 preprocess_message와 동일) ---
def preprocess_message(text: str) -> str:
    """
    메시지 전처리 함수.
    특수문자 제거, 숫자 시작 단어 제거, 공백 정규화 등을 수행합니다.
    """
    text = text[:500]  # 너무 긴 내용은 잘라냄 (DB 컬럼 길이 고려)
    text = re.sub(r'[….,·\-\"\'()\[\]{}<>~!?@#$%^&*“”\\|/+=_—–▶️▶️]', ' ', text)  # 특수문자 제거 후 공백 추가
    text = re.sub(r'\b\d+\w*\b', '', text)  # 숫자로 시작하는 단어 제거 (예: 2024년, 1000원)
    text = re.sub(r'\s+-\s+\S+$', '', text)  # 마지막 언론사 제거 (예: " - 연합뉴스")
    text = re.sub(r'::\s+\S+\s+::', '', text)  # 특정 패턴의 언론사 제거 (예: ":: 서울경제 ::")
    text = re.sub(r'\s+', ' ', text).strip()  # 2개 이상 공백을 1개로 변경하고 양쪽 공백 제거
    return text

def extract_url(text: str) -> Optional[str]:
    """
    텍스트에서 URL을 추출합니다.
    """
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return urls[0] if urls else None

def get_website_news(url: str) -> str:
    """
    (Placeholder) 주어진 URL에서 뉴스 내용을 가져오는 함수.
    실제 구현에서는 웹 스크래핑 라이브러리(예: BeautifulSoup, requests)를 사용해야 합니다.
    """
    logger.warning(f"get_website_news: Web scraping for URL {url} is not implemented. Returning placeholder content.")
    return f"Placeholder content for {url}"


# --- TelegramReceiver 클래스 (TelegramReceiver.py에서 통합) ---
class TelegramReceiver:
    """
    텔레그램 API를 통해 메시지를 수신하고 콜백 함수로 전달하는 클래스.
    """
    def __init__(self):
        # 실제 API ID와 Hash로 변경해야 합니다.
        # 환경 변수에서 로드하거나 설정 파일에서 가져오는 것이 권장됩니다.
        self.api_id = os.environ.get('TELEGRAM_API_ID') 
        self.api_hash = os.environ.get('TELEGRAM_API_HASH') 
        
        # 타겟 채널 ID 목록 (int 타입으로 변환)
        # 예시: 급등일보 int(-1001208429502), 세모뉴 int(-1001704234762)
        self.target_room_ids = [int(-1001208429502), int(-1001704234762)]
        self.session_name = 'my_telegram_session' # 세션 파일 이름

        self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)
        self.callback: Optional[Callable[[Dict[str, Any]], None]] = None
        logger.info("TelegramReceiver initialized.")

    def set_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        메시지 수신 시 호출될 콜백 함수를 등록합니다.
        :param callback: 메시지 데이터를 인자로 받는 함수
        """
        self.callback = callback
        logger.info("TelegramReceiver callback set.")

    async def _process_message(self, event: events.NewMessage.Event):
        """
        새로운 텔레그램 메시지 이벤트를 처리하는 비동기 함수.
        """
        message = event.message
        chat_id = event.chat_id

        # 타겟 채널 ID에 해당하는 메시지만 처리
        if chat_id not in self.target_room_ids:
            logger.debug(f"Skipping message from untargeted chat_id: {chat_id}")
            return

        if not message or not message.text:
            logger.debug(f"Skipping empty message from chat_id: {chat_id}")
            return

        logger.debug(f"Received message from {chat_id}: {message.text[:100]}...")

        # URL 추출 및 웹사이트 뉴스 가져오기 (필요시)
        url = extract_url(message.text)
        # web_content = get_website_news(url) if url else None # 현재는 placeholder 함수

        # 메시지 전처리
        processed_text = preprocess_message(message.text)

        # 콜백 함수 호출
        if self.callback:
            data = {
                'date': message.date,
                'chat_id': chat_id,
                'message': message.text,
                'processed_message': processed_text,
                'url': url,
                # 'web_content': web_content # 현재는 사용하지 않음
            }
            try:
                # 콜백이 비동기 함수일 수 있으므로 await로 호출
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(data)
                else:
                    self.callback(data)
            except Exception as e:
                logger.error(f"Error in TelegramReceiver callback: {e}", exc_info=True)
        else:
            logger.warning("TelegramReceiver: Callback not set. Message not processed.")

    async def run(self):
        """
        텔레그램 메시지 수신을 시작하는 메인 비동기 함수.
        """
        logger.info("TelegramReceiver: Connecting to Telegram...")
        try:
            await self.client.start()
            logger.info("TelegramReceiver: Connected to Telegram.")

            # 새 메시지 이벤트 핸들러 등록
            self.client.add_event_handler(self._process_message, events.NewMessage())

            # 클라이언트가 연결을 유지하도록 무한 루프 실행
            # stop_event가 설정될 때까지 대기
            while True:
                await asyncio.sleep(1) # 짧은 간격으로 대기하며 stop_event 확인
        except asyncio.CancelledError:
            logger.info("TelegramReceiver: Message reception loop cancelled.")
        except Exception as e:
            logger.error(f"TelegramReceiver: Error in message reception loop: {e}", exc_info=True)
        finally:
            logger.info("TelegramReceiver: Disconnecting from Telegram...")
            if self.client.is_connected():
                await self.client.disconnect()
            logger.info("TelegramReceiver: Disconnected from Telegram.")


# --- TelegramFeed 클래스 (기존 telegram_feed.py) ---
class TelegramFeed(BaseFeed):
    """
    텔레그램 메시지를 수신하고 DB에 저장하는 Feed 모듈.
    news_raw 테이블에 데이터를 저장합니다.
    """
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        super().__init__("TelegramFeed", redis_host, redis_port)
        self.db_feed = DBFeed()
        self.telegram_receiver = TelegramReceiver() # 통합된 TelegramReceiver 클래스 사용
        
        # TelegramReceiver의 콜백 함수 설정
        self.telegram_receiver.set_callback(self._telegram_message_callback)
        
        self.message_buffer: List[Dict[str, Any]] = [] # 수신된 메시지 임시 저장 버퍼
        self.last_save_time = datetime.now()

        logger.info("TelegramFeed initialized.")

    async def _telegram_message_callback(self, data: Dict[str, Any]):
        """
        TelegramReceiver로부터 메시지를 수신할 때 호출되는 콜백 함수.
        수신된 메시지를 버퍼에 추가합니다.
        """
        # data는 TelegramReceiver의 _process_message에서 오는 딕셔너리 형태
        # 예: {'date': message.date, 'chat_id': chat_id, 'message': message.text, 'processed_message': ..., 'url': ...}
        
        # news_raw 테이블 스키마에 맞게 데이터 변환
        item = {
            'source': 'Telegram',
            'datetime': data.get('date', datetime.now()),
            'title': data.get('processed_message', ''), # 전처리된 메시지를 제목으로 사용
            'content': data.get('message', ''), # 원본 메시지
            'url': data.get('url'),
            'related_stocks': [] # 초기에는 비어있음 (NLP 분석 후 채워짐)
        }
        self.message_buffer.append(item)
        logger.debug(f"[TelegramFeed] Buffered message: {item['title'][:50]}...")

    def _process_message_buffer(self):
        """버퍼에 쌓인 텔레그램 메시지를 DB에 저장하고 이벤트를 발행합니다."""
        if self.message_buffer:
            if self.db_feed.save_news_raw(self.message_buffer):
                logger.info(f"Saved {len(self.message_buffer)} raw Telegram messages to DB.")
                # FeedManager에게 텔레그램 메시지 수집 완료 이벤트 발행
                self.publish_event('telegram_feed_events', {
                    'type': 'TELEGRAM_MESSAGE_COLLECTED',
                    'count': len(self.message_buffer),
                    'timestamp': datetime.now().isoformat()
                })
                self.message_buffer.clear()
            else:
                logger.error("Failed to save Telegram messages to DB.")

    def run(self):
        """
        TelegramFeed의 메인 실행 루프.
        TelegramReceiver를 통해 메시지를 수신하고 주기적으로 DB에 저장합니다.
        """
        logger.info(f"TelegramFeed process started.")
        
        # asyncio 이벤트 루프를 생성하고 TelegramReceiver를 실행
        # TelegramReceiver의 run() 메서드는 비동기 함수이므로 await 필요
        # 이를 별도의 스레드에서 실행하여 현재 run() 메서드를 블록하지 않도록 함
        def start_telegram_receiver_loop():
            # 각 스레드에는 고유한 asyncio 이벤트 루프가 필요
            asyncio.set_event_loop(asyncio.new_event_loop()) 
            loop_child = asyncio.get_event_loop()
            try:
                # TelegramReceiver의 run 메서드 호출
                loop_child.run_until_complete(self.telegram_receiver.run())
            except asyncio.CancelledError:
                logger.info("TelegramReceiver loop cancelled.")
            except Exception as e:
                logger.error(f"TelegramReceiver loop error: {e}", exc_info=True)
            finally:
                # 자식 루프 종료 시 루프를 닫음
                if not loop_child.is_closed():
                    loop_child.close()
                logger.info("TelegramReceiver child loop closed.")

        # TelegramReceiver를 별도의 스레드에서 실행
        telegram_thread = threading.Thread(target=start_telegram_receiver_loop, daemon=True)
        telegram_thread.start()
        logger.info("TelegramReceiver started in a separate thread.")

        # 이 메인 스레드에서는 주기적으로 버퍼를 확인하고 DB에 저장
        while not self.stop_event.is_set():
            try:
                self._process_message_buffer()
                self._wait_for_stop_signal(interval=10) # 10초마다 버퍼 처리 시도
            except Exception as e:
                logger.error(f"[{self.feed_name}] Error in run loop: {e}", exc_info=True)
                time.sleep(5)

        logger.info(f"TelegramFeed process stopped.")
        self.db_feed.close()

# 실행 예시 (별도 프로세스로 실행될 때)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    telegram_feed = TelegramFeed()
    try:
        telegram_feed.run()
    except KeyboardInterrupt:
        logger.info("TelegramFeed interrupted by user.")
    finally:
        telegram_feed.stop()
