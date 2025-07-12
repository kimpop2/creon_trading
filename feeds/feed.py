# feeds/feed.py

import logging
import time
import threading
import json
import redis
import sys
import os
from typing import Dict, Any, List

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from feed_manager import FeedManager # refactored feed_manager.py

# 로거 설정
logger = logging.getLogger(__name__)

class Feed:
    """
    Feed는 자동매매 시스템의 최상위 데이터 오케스트레이터입니다.
    Trading 프로세스와 통신하며, FeedManager를 통해 모든 하위 데이터 수집/분석을 관리합니다.
    """
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()

        # Trading 프로세스와의 통신 채널
        self.trading_signal_channel = 'trading_signals' # Feed -> Trading
        self.trading_request_channel = 'trading_requests' # Trading -> Feed

        # FeedManager로부터 신호를 받는 채널
        self.feed_manager_signal_channel = 'feed_manager_signals' # FeedManager -> Feed

        self.pubsub.subscribe([self.trading_request_channel, self.feed_manager_signal_channel])

        self.feed_manager = FeedManager(redis_host, redis_port) # FeedManager 인스턴스 생성
        self.stop_event = threading.Event() # Feed 종료를 위한 이벤트

        logger.info("Feed (Orchestrator) initialized. Subscribing to trading requests and FeedManager signals.")

    def _process_incoming_message(self, message: Dict[str, Any]):
        """
        Redis Pub/Sub으로부터 수신된 메시지를 처리합니다.
        Trading으로부터의 요청 또는 FeedManager로부터의 신호를 처리합니다.
        """
        try:
            event_data = json.loads(message['data'])
            channel = message['channel']
            event_type = event_data.get('type')

            logger.debug(f"[Feed Orchestrator] Received message from '{channel}': {event_type}")

            if channel == self.trading_request_channel:
                # Trading으로부터의 요청을 FeedManager로 전달
                logger.info(f"[Feed Orchestrator] Forwarding Trading request '{event_type}' to FeedManager.")
                self.feed_manager.handle_trading_request(event_data) # FeedManager의 메서드 호출
            elif channel == self.feed_manager_signal_channel:
                # FeedManager로부터의 최종 신호를 Trading으로 전달
                logger.info(f"[Feed Orchestrator] Publishing final signal '{event_type}' to Trading.")
                self.publish_trading_signal(event_data)
            else:
                logger.warning(f"[Feed Orchestrator] Unhandled message channel: {channel}")

        except json.JSONDecodeError as e:
            logger.error(f"[Feed Orchestrator] Failed to decode JSON from Redis message: {e}. Data: {message['data']}")
        except Exception as e:
            logger.error(f"[Feed Orchestrator] Error processing incoming message: {e}", exc_info=True)

    def publish_trading_signal(self, signal_data: Dict[str, Any]):
        """
        Trading 프로세스로 최종 매매 신호를 발행합니다.
        """
        try:
            self.redis_client.publish(self.trading_signal_channel, json.dumps(signal_data))
            logger.debug(f"[Feed Orchestrator] Published trading signal to '{self.trading_signal_channel}': {signal_data.get('type')}")
        except Exception as e:
            logger.error(f"[Feed Orchestrator] Failed to publish trading signal: {e}", exc_info=True)

    def run(self):
        """
        Feed 오케스트레이터의 메인 실행 루프.
        FeedManager를 시작하고, Redis 메시지를 수신하여 처리합니다.
        """
        logger.info("Feed (Orchestrator) process started.")
        self.feed_manager.start() # FeedManager 시작

        for message in self.pubsub.listen():
            if self.stop_event.is_set():
                logger.info("Feed Orchestrator stop event received. Exiting listen loop.")
                break
            if message['type'] == 'message':
                self._process_incoming_message(message)
            
        logger.info("Feed (Orchestrator) process stopped.")
        self.stop()

    def stop(self):
        """Feed 오케스트레이터 및 하위 FeedManager를 안전하게 종료합니다."""
        logger.info("Stopping Feed (Orchestrator) and FeedManager...")
        self.stop_event.set() # Feed Orchestrator 루프 종료 신호
        self.feed_manager.stop() # FeedManager 종료 요청

        self.pubsub.unsubscribe([self.trading_request_channel, self.feed_manager_signal_channel])
        self.redis_client.close()
        logger.info("Feed (Orchestrator) stopped.")

# 실행 예시
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    feed_orchestrator = Feed()
    try:
        feed_orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Feed Orchestrator interrupted by user.")
    finally:
        feed_orchestrator.stop()
