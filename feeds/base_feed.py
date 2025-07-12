# feeds/base_feed.py

import logging
import time
import threading
import redis # 64 bit
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# 로거 설정
logger = logging.getLogger(__name__)

class BaseFeed(ABC):
    """
    모든 데이터 수집 Feed 클래스를 위한 추상 기본 클래스.
    별도의 프로세스/스레드로 실행될 수 있도록 설계됩니다.
    """
    def __init__(self, feed_name: str, redis_host: str = 'localhost', redis_port: int = 6379):
        self.feed_name = feed_name
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        self.stop_event = threading.Event() # Feed 프로세스 종료를 위한 이벤트
        logger.info(f"BaseFeed initialized: {self.feed_name}")

    def publish_event(self, channel: str, event_data: Dict[str, Any]):
        """
        Redis Pub/Sub을 통해 이벤트를 발행합니다.
        :param channel: 메시지를 발행할 채널 이름
        :param event_data: 발행할 이벤트 데이터 (딕셔너리)
        """
        try:
            event_data['feed_name'] = self.feed_name # 어떤 Feed에서 발생했는지 식별자 추가
            self.redis_client.publish(channel, json.dumps(event_data))
            logger.debug(f"[{self.feed_name}] Published event to '{channel}': {event_data['type']}")
        except Exception as e:
            logger.error(f"[{self.feed_name}] Failed to publish event to '{channel}': {e}", exc_info=True)

    @abstractmethod
    def run(self):
        """
        Feed의 메인 실행 로직.
        이 메서드는 별도의 프로세스/스레드에서 실행됩니다.
        """
        pass

    def start(self):
        """Feed를 별도의 스레드로 시작합니다."""
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        logger.info(f"[{self.feed_name}] Feed started in a separate thread.")

    def stop(self):
        """Feed 프로세스를 안전하게 종료합니다."""
        self.stop_event.set()
        logger.info(f"[{self.feed_name}] Stop event set. Waiting for thread to finish...")
        if self.thread.is_alive():
            self.thread.join(timeout=5) # 5초 대기
            if self.thread.is_alive():
                logger.warning(f"[{self.feed_name}] Thread did not terminate gracefully.")
        logger.info(f"[{self.feed_name}] Feed stopped.")

    def _wait_for_stop_signal(self, interval: float = 1.0):
        """종료 신호가 올 때까지 주기적으로 대기합니다."""
        while not self.stop_event.is_set():
            time.sleep(interval)
