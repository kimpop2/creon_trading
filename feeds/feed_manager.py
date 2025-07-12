# feed_manager.py (Refactored)

import logging
import time
import threading
import json
import redis
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
import sys
import os

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from feeds.db_feed import DBFeed # FeedManager도 DBFeed를 사용하여 데이터 조회
from feeds.base_feed import BaseFeed # BaseFeed 임포트 (개별 Feed 인스턴스 관리를 위함)
from feeds.price_feed import PriceFeed
from feeds.news_feed import NewsFeed
from feeds.signal_feed import SignalFeed
from feeds.invester_feed import InvesterFeed
from feeds.telegram_feed import TelegramFeed
from feeds.nlp_analysis_feed import NLPAnalysisFeed # NLP 분석 모듈

# 로거 설정
logger = logging.getLogger(__name__)

class FeedManager:
    """
    FeedManager는 모든 데이터 수집 및 분석 Feed 프로세스를 오케스트레이션하고,
    수집된 데이터를 종합하여 최종 신호를 Feed (최상위 오케스트레이터)에 전달합니다.
    """
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        self.db_feed = DBFeed() # FeedManager도 DB에서 데이터를 조회할 수 있음

        self.feed_processes: Dict[str, BaseFeed] = {} # 실행 중인 Feed 프로세스 인스턴스 저장
        self.stop_event = threading.Event() # FeedManager 종료를 위한 이벤트

        # Feed (최상위 오케스트레이터)로 신호를 보내는 채널
        self.feed_orchestrator_signal_channel = 'feed_manager_signals' 

        # 각 개별 Feed에서 발행하는 이벤트 채널 구독
        self.subscribed_channels = [
            'price_feed_events',
            'news_feed_events',
            'signal_feed_events',
            'invester_feed_events',
            'telegram_feed_events',
            'nlp_analysis_feed_events',
            # Trading에서 Feed (오케스트레이터)를 통해 전달되는 요청을 FeedManager가 처리할 수 있도록 채널 추가
            'trading_requests_to_feed_manager' 
        ]
        self.pubsub.subscribe(self.subscribed_channels)

        logger.info("FeedManager initialized. Subscribing to individual feed events and trading requests.")

    def _init_feed_processes(self):
        """
        모든 개별 Feed 프로세스 인스턴스를 초기화하고 시작합니다.
        각 Feed는 별도의 프로세스/스레드로 실행될 수 있습니다.
        여기서는 간단히 스레드로 시작하는 예시를 보여줍니다.
        """
        # Creon API를 사용하는 Feed들은 32bit 환경에서 실행되어야 함.
        # 실제 배포 시에는 각 Feed를 별도의 Python 스크립트로 만들고,
        # `multiprocessing`을 사용하거나 OS의 프로세스 관리 도구(예: Supervisor)를 통해 실행해야 합니다.
        
        self.feed_processes['PriceFeed'] = PriceFeed(self.redis_host, self.redis_port)
        self.feed_processes['NewsFeed'] = NewsFeed(self.redis_host, self.redis_port)
        self.feed_processes['SignalFeed'] = SignalFeed(self.redis_host, self.redis_port)
        self.feed_processes['InvesterFeed'] = InvesterFeed(self.redis_host, self.redis_port)
        self.feed_processes['TelegramFeed'] = TelegramFeed(self.redis_host, self.redis_port)
        self.feed_processes['NLPAnalysisFeed'] = NLPAnalysisFeed(self.redis_host, self.redis_port)

        for name, feed_instance in self.feed_processes.items():
            feed_instance.start() # 각 Feed를 스레드로 시작
            logger.info(f"Started {name} as a background thread.")

    def _process_incoming_feed_event(self, message: Dict[str, Any]):
        """
        각 개별 Feed 프로세스 또는 Trading으로부터 수신된 이벤트를 처리합니다.
        :param message: Redis Pub/Sub 메시지 딕셔너리
        """
        try:
            event_data = json.loads(message['data'])
            event_type = event_data.get('type')
            feed_name = event_data.get('feed_name')
            timestamp = event_data.get('timestamp')

            logger.info(f"[FeedManager] Received event from {feed_name if feed_name else message['channel']}: {event_type} at {timestamp}")

            # 데이터 수집 완료 이벤트 처리 (DB에 저장되었음을 의미)
            if event_type in ['MINUTE_OHLCV_COLLECTED', 'MARKET_VOLUME_COLLECTED', 
                              'NEWS_COLLECTED', 'ANNOUNCEMENT_NEWS_COLLECTED',
                              'INVESTOR_TRENDS_COLLECTED', 'TELEGRAM_MESSAGE_COLLECTED']:
                # 이 단계에서는 데이터가 이미 DB에 저장되었으므로,
                # FeedManager는 이 데이터를 활용하여 분석을 시작할 수 있음을 인지합니다.
                logger.debug(f"[FeedManager] Data collection event received: {event_type}. Triggering analysis if needed.")
                # 특정 데이터 수집 완료 시, NLPAnalysisFeed를 트리거하거나 직접 분석 로직 실행
                # NLPAnalysisFeed가 주기적으로 DB를 스캔하여 처리하므로, 여기서는 별도 트리거 없이 진행
                pass

            # 분석 완료 이벤트 처리 (NLPAnalysisFeed에서 발생)
            elif event_type in ['NEWS_SUMMARIZED', 'THEMATIC_STOCKS_IDENTIFIED', 'DAILY_UNIVERSE_UPDATED']:
                # NLPAnalysisFeed에서 최종 분석 결과가 나왔으므로, 이를 Feed (오케스트레이터)로 전달
                logger.info(f"[FeedManager] Analysis completed event received: {event_type}. Publishing to Feed Orchestrator.")
                self.publish_signal_to_orchestrator(event_data)

            # Trading 프로세스에서 Feed (오케스트레이터)를 거쳐 온 요청 처리
            elif event_type == 'GET_LATEST_OHLCV_MINUTE':
                self._handle_trading_request(event_data)

            # 기타 이벤트 처리...

        except json.JSONDecodeError as e:
            logger.error(f"[FeedManager] Failed to decode JSON from Redis message: {e}. Data: {message['data']}")
        except Exception as e:
            logger.error(f"[FeedManager] Error processing feed event: {e}", exc_info=True)

    def handle_trading_request(self, request_data: Dict[str, Any]):
        """
        Feed (최상위 오케스트레이터)로부터 Trading의 요청을 받아 처리합니다.
        """
        request_type = request_data.get('type')
        logger.info(f"[FeedManager] Handling Trading request: {request_type}")

        if request_type == 'GET_LATEST_OHLCV_MINUTE':
            stock_code = request_data.get('stock_code')
            # DB에서 최신 분봉 데이터를 조회하여 Feed (오케스트레이터)에 응답
            latest_minute_df = self.db_feed.fetch_ohlcv_minute(stock_code, datetime.now() - timedelta(minutes=10), datetime.now())
            if not latest_minute_df.empty:
                response_data = {
                    'type': 'LATEST_OHLCV_MINUTE_RESPONSE',
                    'stock_code': stock_code,
                    'data': latest_minute_df.iloc[-1].to_dict(), # 최신 1분봉 데이터
                    'timestamp': datetime.now().isoformat()
                }
                self.publish_signal_to_orchestrator(response_data)
                logger.info(f"[FeedManager] Sent latest minute OHLCV for {stock_code} to Feed Orchestrator.")
            else:
                logger.warning(f"No latest minute OHLCV found for {stock_code}.")
                self.publish_signal_to_orchestrator({
                    'type': 'LATEST_OHLCV_MINUTE_RESPONSE',
                    'stock_code': stock_code,
                    'data': None,
                    'timestamp': datetime.now().isoformat()
                })
        # 다른 Trading 요청 유형에 대한 처리 추가
        # elif request_type == 'GET_DAILY_UNIVERSE':
        #     ...

    def publish_signal_to_orchestrator(self, signal_data: Dict[str, Any]):
        """
        Feed (최상위 오케스트레이터)로 최종 매매 신호를 발행합니다.
        """
        try:
            self.redis_client.publish(self.feed_orchestrator_signal_channel, json.dumps(signal_data))
            logger.debug(f"[FeedManager] Published signal to '{self.feed_orchestrator_signal_channel}': {signal_data.get('type')}")
        except Exception as e:
            logger.error(f"[FeedManager] Failed to publish signal to orchestrator: {e}", exc_info=True)

    def run(self):
        """
        FeedManager의 메인 실행 루프.
        모든 개별 Feed 프로세스를 시작하고, Redis 메시지를 수신하여 처리합니다.
        """
        logger.info("FeedManager process started.")
        self._init_feed_processes() # 개별 Feed 프로세스 시작

        for message in self.pubsub.listen():
            if self.stop_event.is_set():
                logger.info("FeedManager stop event received. Exiting listen loop.")
                break
            if message['type'] == 'message':
                self._process_incoming_feed_event(message)
            
        logger.info("FeedManager process stopped.")
        self.stop()

    def stop(self):
        """FeedManager 및 모든 하위 개별 Feed 프로세스를 안전하게 종료합니다."""
        logger.info("Stopping FeedManager and all child Feed processes...")
        self.stop_event.set() # FeedManager 루프 종료 신호

        for name, feed_instance in self.feed_processes.items():
            feed_instance.stop() # 각 Feed 프로세스에 종료 신호 전달

        self.pubsub.unsubscribe(self.subscribed_channels)
        self.redis_client.close()
        self.db_feed.close()
        logger.info("All Feed processes and FeedManager stopped.")


# 실행 예시 (Feed (오케스트레이터)에 의해 호출되므로, 단독 실행은 거의 없음)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 실제 환경에서는 Creon API가 32bit 프로세스에서만 동작하므로,
    # Creon 관련 Feed들은 별도의 32bit Python 프로세스로 실행되어야 합니다.
    # 이 예시 코드는 개념적인 구조를 보여주며, 실제 멀티프로세스 구현은 `multiprocessing` 모듈을 사용해야 합니다.

    feed_manager = FeedManager()
    try:
        feed_manager.run()
    except KeyboardInterrupt:
        logger.info("FeedManager interrupted by user.")
    finally:
        feed_manager.stop()
