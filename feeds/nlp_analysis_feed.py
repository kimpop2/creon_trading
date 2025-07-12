# feeds/nlp_analysis_feed.py

import logging
import time
from datetime import datetime, date, timedelta
import pandas as pd
import sys
import os
import json
from typing import Dict, Any, List, Optional

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from feeds.base_feed import BaseFeed
from feeds.db_feed import DBFeed
# 기존 분석 모듈 import (경로 확인 필요)
# from daily_news_summarizer import summarize_news_with_ai # AI 요약 함수
# from news_theme_analyzer import process_breaking_news, load_stock_names # 테마/종목 연관 분석
# from thematic_stocks import run_daily_analysis_and_get_actionable_insights # 테마 종목 발굴

# 로거 설정
logger = logging.getLogger(__name__)

class NLPAnalysisFeed(BaseFeed):
    """
    뉴스 NLP 처리 후 분석, 테마, 종목 발굴을 수행하고 DB에 저장하는 Feed 모듈.
    news_summaries, thematic_stocks, daily_universe 테이블에 데이터를 저장합니다.
    """
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        super().__init__("NLPAnalysisFeed", redis_host, redis_port)
        self.db_feed = DBFeed()
        self.stock_names_map: Dict[str, str] = {} # 종목명 -> 종목코드 매핑
        self._load_stock_names()
        logger.info("NLPAnalysisFeed initialized.")

    def _load_stock_names(self):
        """DB에서 종목명 맵을 로드합니다."""
        self.stock_names_map = self.db_feed.fetch_stock_info_map()
        logger.info(f"Loaded {len(self.stock_names_map)} stock names for NLP analysis.")

    def _process_raw_news_for_nlp(self):
        """
        news_raw 테이블에서 아직 처리되지 않은 뉴스를 가져와 NLP 분석을 수행하고 저장합니다.
        """
        logger.info("Checking for new raw news to process for NLP.")
        # 마지막 처리 시점 이후의 뉴스만 가져오기 (혹은 특정 기간)
        # 여기서는 간단히 최근 1일치 뉴스를 가져온다고 가정
        yesterday = datetime.now() - timedelta(days=1)
        raw_news_df = self.db_feed.fetch_news_raw(yesterday, datetime.now())

        if raw_news_df.empty:
            logger.info("No new raw news to process for NLP.")
            return

        processed_summaries = []
        thematic_stocks_data = []
        
        for index, row in raw_news_df.iterrows():
            original_news_id = row['id']
            news_title = row['title']
            news_content = row['content']
            news_datetime = row['datetime']

            # 1. 뉴스 요약 및 감성 분석 (daily_news_summarizer의 summarize_news_with_ai 함수를 모방)
            # 실제 AI 호출 로직은 여기에 구현하거나 외부 모듈로 위임
            summary_text = f"요약: {news_title[:50]}..." # 예시 요약
            sentiment_score = 0.5 # 예시 감성 점수 (실제로는 AI 모델 결과)

            processed_summaries.append({
                'original_news_id': original_news_id,
                'summary': summary_text,
                'sentiment_score': sentiment_score,
                'processed_at': datetime.now()
            })

            # 2. 테마 분석 및 종목 발굴 (news_theme_analyzer, thematic_stocks 모방)
            # news_theme_analyzer.py의 process_breaking_news 로직을 활용하여 관련 종목 추출
            # 여기서는 간단히 제목에서 종목명 매칭
            related_stock_codes = []
            for stock_name, stock_code in self.stock_names_map.items():
                if stock_name in news_title or stock_name in news_content:
                    related_stock_codes.append(stock_code)
            
            if related_stock_codes:
                # thematic_stocks.py의 run_daily_analysis_and_get_actionable_insights 로직 모방
                # 여기서는 관련 종목에 대한 테마 점수를 임시로 부여
                for stock_code in related_stock_codes:
                    thematic_stocks_data.append({
                        'theme_name': '뉴스_분석_테마', # 실제 테마는 NLP 모델이 생성
                        'stock_code': stock_code,
                        'analysis_date': news_datetime.date(),
                        'relevance_score': 0.7, # 예시 점수
                        'mention_count': 1
                    })
        
        # DB에 저장
        if processed_summaries:
            if self.db_feed.save_news_summaries(processed_summaries):
                logger.info(f"Saved {len(processed_summaries)} news summaries to DB.")
                self.publish_event('nlp_analysis_feed_events', {
                    'type': 'NEWS_SUMMARIZED',
                    'count': len(processed_summaries),
                    'timestamp': datetime.now().isoformat()
                })
            else:
                logger.error("Failed to save news summaries.")

        if thematic_stocks_data:
            if self.db_feed.save_thematic_stocks(thematic_stocks_data):
                logger.info(f"Saved {len(thematic_stocks_data)} thematic stock records to DB.")
                self.publish_event('nlp_analysis_feed_events', {
                    'type': 'THEMATIC_STOCKS_IDENTIFIED',
                    'count': len(thematic_stocks_data),
                    'timestamp': datetime.now().isoformat()
                })
            else:
                logger.error("Failed to save thematic stocks.")

    def _update_daily_universe(self):
        """
        최신 테마 및 종목 점수를 기반으로 daily_universe 테이블을 업데이트합니다.
        setup_daily_universe.py의 로직을 모방합니다.
        """
        logger.info("Updating daily universe based on latest analysis.")
        today = date.today()
        # 예시: 오늘 날짜의 테마 종목과 주가 데이터를 결합하여 유니버스 점수 계산
        # 실제 로직은 setup_manager.py의 _calculate_stock_scores 등을 참고하여 복잡하게 구현
        
        # 여기서는 간단히, 오늘 날짜의 thematic_stocks를 가져와 daily_universe에 반영
        thematic_df = self.db_feed.fetch_thematic_stocks(today)
        if thematic_df.empty:
            logger.info(f"No thematic stocks found for {today}. Daily universe not updated.")
            return

        universe_data = []
        for index, row in thematic_df.iterrows():
            stock_code = row['stock_code']
            total_score = row['relevance_score'] # 간단하게 관련성 점수를 총 점수로 사용
            score_detail = {'theme_relevance': float(row['relevance_score'])} # JSON 필드
            
            universe_data.append({
                'stock_code': stock_code,
                'date': today,
                'total_score': total_score,
                'score_detail': score_detail,
                'is_selected': True # 일단 모두 선택된 것으로 가정
            })
        
        if universe_data:
            if self.db_feed.save_daily_universe(universe_data):
                logger.info(f"Updated {len(universe_data)} records in daily_universe for {today}.")
                self.publish_event('nlp_analysis_feed_events', {
                    'type': 'DAILY_UNIVERSE_UPDATED',
                    'date': today.isoformat(),
                    'count': len(universe_data),
                    'timestamp': datetime.now().isoformat()
                })
            else:
                logger.error(f"Failed to update daily_universe for {today}.")

    def run(self):
        """
        NLPAnalysisFeed의 메인 실행 루프.
        주기적으로 뉴스 분석 및 테마/종목 발굴을 수행하고 DB에 저장합니다.
        """
        logger.info(f"NLPAnalysisFeed process started.")
        while not self.stop_event.is_set():
            try:
                self._process_raw_news_for_nlp() # 원본 뉴스 처리
                self._update_daily_universe() # 유니버스 업데이트
                self._wait_for_stop_signal(interval=300) # 5분마다 실행 (예시)
            except Exception as e:
                logger.error(f"[{self.feed_name}] Error in run loop: {e}", exc_info=True)
                time.sleep(5)

        logger.info(f"NLPAnalysisFeed process stopped.")
        self.db_feed.close()

# 실행 예시 (별도 프로세스로 실행될 때)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    nlp_feed = NLPAnalysisFeed()
    try:
        nlp_feed.run()
    except KeyboardInterrupt:
        logger.info("NLPAnalysisFeed interrupted by user.")
    finally:
        nlp_feed.stop()
