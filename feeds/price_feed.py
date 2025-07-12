# feeds/price_feed.py

import logging
import time
from datetime import datetime, date, timedelta
import pandas as pd
import sys
import os
from typing import List, Dict, Any, Optional
# 프로젝트 루트 경로를 sys.path에 추가 (creon_api 및 db_feed 임포트를 위함)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from feeds.base_feed import BaseFeed
from feeds.db_feed import DBFeed

# 로거 설정
logger = logging.getLogger(__name__)

class PriceFeed(BaseFeed):
    """
    Creon API를 통해 주가 데이터를 수집하고 DB에 저장하는 Feed 모듈.
    ohlcv_minute, market_volume 테이블에 데이터를 저장합니다.
    """
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        super().__init__("PriceFeed", redis_host, redis_port)
        self.api_client = CreonAPIClient() # Creon API 클라이언트 초기화
        self.db_feed = DBFeed() # DBFeed 초기화
        self.stock_codes: List[str] = [] # 수집 대상 종목 코드 리스트

        # 초기 종목 코드 로드 (stock_info 테이블에서)
        self._load_stock_codes()
        logger.info(f"PriceFeed initialized. Monitoring {len(self.stock_codes)} stocks.")

    def _load_stock_codes(self):
        """DB에서 모든 종목 코드를 로드합니다."""
        # stock_info 테이블에서 종목 코드를 가져오는 메서드가 DBFeed에 필요
        # 현재 DBFeed에는 fetch_stock_info_map이 있으므로 활용
        stock_map = self.db_feed.fetch_stock_info_map()
        self.stock_codes = list(stock_map.keys())
        logger.info(f"Loaded {len(self.stock_codes)} stock codes from DB for PriceFeed.")

    def _collect_and_save_ohlcv_minute(self, stock_code: str, from_date: date, to_date: date):
        """
        특정 종목의 분봉 데이터를 Creon API에서 가져와 DB에 저장합니다.
        API 호출은 CreonAPIClient의 get_minute_ohlcv를 사용합니다.
        """
        logger.info(f"Collecting minute OHLCV for {stock_code} from {from_date} to {to_date}")
        from_date_str = from_date.strftime('%Y%m%d')
        to_date_str = to_date.strftime('%Y%m%d')

        # CreonAPIClient의 get_minute_ohlcv는 datetime 인덱스를 가진 DataFrame을 반환
        df = self.api_client.get_minute_ohlcv(stock_code, from_date_str, to_date_str)

        if not df.empty:
            data_to_save = []
            for index, row in df.iterrows():
                data_to_save.append({
                    'stock_code': stock_code,
                    'datetime': index, # datetime 인덱스 사용
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                })
            if self.db_feed.save_ohlcv_minute(data_to_save):
                logger.info(f"Saved {len(data_to_save)} minute OHLCV records for {stock_code}.")
                # 데이터 수집 완료 이벤트 발행 (FeedManager로 전달)
                self.publish_event('price_feed_events', {
                    'type': 'MINUTE_OHLCV_COLLECTED',
                    'stock_code': stock_code,
                    'count': len(data_to_save),
                    'from_datetime': data_to_save[0]['datetime'].isoformat() if data_to_save else None,
                    'to_datetime': data_to_save[-1]['datetime'].isoformat() if data_to_save else None
                })
            else:
                logger.error(f"Failed to save minute OHLCV for {stock_code} to DB.")
        else:
            logger.warning(f"No minute OHLCV data found for {stock_code} from API.")

    def _collect_and_save_market_volume(self):
        """
        코스피/코스닥 현재 거래대금 및 평균 거래대금을 수집하고 DB에 저장합니다.
        `코스피_코스닥_거래대금(현재or평균).py`의 함수를 활용합니다.
        """
        logger.info("Collecting market volume data.")
        current_time = datetime.now()
        current_date = current_time.date()
        current_time_only = current_time.time()

        # 현재 거래대금 (코스피/코스닥)
        kospi_current_amount = self.api_client.get_current_market_amount(market='kospi') # CreonAPIClient에 추가 필요
        kosdaq_current_amount = self.api_client.get_current_market_amount(market='kosdaq') # CreonAPIClient에 추가 필요

        data_to_save = []
        if kospi_current_amount is not None:
            data_to_save.append({
                'market_type': 'KOSPI',
                'date': current_date,
                'time': current_time_only,
                'total_amount': kospi_current_amount
            })
        if kosdaq_current_amount is not None:
            data_to_save.append({
                'market_type': 'KOSDAQ',
                'date': current_date,
                'time': current_time_only,
                'total_amount': kosdaq_current_amount
            })

        if data_to_save:
            if self.db_feed.save_market_volume(data_to_save):
                logger.info(f"Saved {len(data_to_save)} market volume records.")
                self.publish_event('price_feed_events', {
                    'type': 'MARKET_VOLUME_COLLECTED',
                    'date': current_date.isoformat(),
                    'time': current_time_only.isoformat()
                })
            else:
                logger.error("Failed to save market volume to DB.")
        else:
            logger.warning("No market volume data collected.")

    def run(self):
        """
        PriceFeed의 메인 실행 루프.
        주기적으로 주가 데이터와 시장 거래대금을 수집하여 DB에 저장합니다.
        """
        logger.info(f"PriceFeed process started.")
        # CreonAPIClient가 연결되어 있는지 확인
        if not self.api_client.connected:
            logger.error("Creon API Client is not connected. PriceFeed cannot run.")
            return

        # 초기 데이터 로드 및 저장 (예: 최근 30일치 분봉 데이터)
        today = date.today()
        # Creon API는 보통 당일 데이터만 실시간으로 제공하므로, 과거 데이터는 백테스트 매니저에서 처리
        # 여기서는 매일 장 시작 전/후에 필요한 데이터를 업데이트하는 방식으로 가정
        # 실시간 데이터는 별도의 실시간 구독으로 처리하거나, 주기적인 풀링으로 최신 데이터만 가져옴
        
        # 예시: 매분마다 현재 시장 거래대금 수집
        # 이 부분은 실시간 스트리밍이 아닌 주기적인 요청 방식이므로, API 제한을 고려해야 함
        # 실시간 분봉은 CreonAPIClient의 실시간 구독 기능을 활용해야 함 (현재 샘플 코드에는 없음)

        # 현재는 주기적으로 모든 종목의 최신 1분봉을 요청하는 것은 비효율적
        # 대신, `FeedManager`로부터 특정 종목의 최신 분봉 데이터 요청을 받거나,
        # 실시간으로 구독하는 방식으로 변경해야 함.
        # 이 예시에서는 주기적으로 시장 거래대금만 수집하는 로직을 포함
        
        # 주가 수집 모듈이 '주가수집' 신호를 통해 외부 시스템의 주가 수집모듈에 주가 수집을 위임하고,
        # 주가 수집 모듈이 주가를 수집해서, 일봉, 분봉 테이블에 저장하는 작업이 완료돼었다는 신호를 주도록 할 수 도 있지 않나?
        # -> 이 부분은 FeedManager가 요청을 받고 PriceFeed가 처리하는 방식으로 구현 가능

        while not self.stop_event.is_set():
            try:
                # 여기에 주기적인 데이터 수집 로직 추가
                # 예: 매 5분마다 시장 거래대금 업데이트
                self._collect_and_save_market_volume()

                # TODO: FeedManager로부터 특정 종목의 분봉/일봉 데이터 요청을 받아 처리하는 로직 추가
                # self.redis_client.subscribe('price_feed_requests')
                # for message in self.pubsub.listen():
                #     if message['type'] == 'message':
                #         request_data = json.loads(message['data'])
                #         if request_data.get('type') == 'COLLECT_OHLCV_MINUTE':
                #             stock_code = request_data['stock_code']
                #             from_dt = datetime.fromisoformat(request_data['from_datetime'])
                #             to_dt = datetime.fromisoformat(request_data['to_datetime'])
                #             self._collect_and_save_ohlcv_minute(stock_code, from_dt.date(), to_dt.date())

                self._wait_for_stop_signal(interval=60) # 1분마다 실행 (예시)
            except Exception as e:
                logger.error(f"[{self.feed_name}] Error in run loop: {e}", exc_info=True)
                time.sleep(5) # 오류 발생 시 잠시 대기 후 재시도

        logger.info(f"PriceFeed process stopped.")

# 실행 예시 (별도 프로세스로 실행될 때)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    price_feed = PriceFeed()
    try:
        price_feed.run() # 이 프로세스가 직접 실행되는 경우
    except KeyboardInterrupt:
        logger.info("PriceFeed interrupted by user.")
    finally:
        price_feed.stop()
        price_feed.db_feed.close()
