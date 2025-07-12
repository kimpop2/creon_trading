# tests/test_feeds_db_feed.py

import unittest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta, time
import sys
import os
import json
from decimal import Decimal

# 프로젝트 루트를 sys.path에 추가 (db_feed 임포트를 위함)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from feeds.db_feed import DBFeed # 테스트 대상 모듈
# config.settings에서 DB 설정을 로드한다고 가정
# 실제 테스트 환경에 맞게 config.settings 또는 환경 변수 설정 필요
# 예시:
# from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


class TestFeedsDBFeed(unittest.TestCase):
    """
    DBFeed 클래스의 데이터베이스 작업 메서드에 대한 단위 테스트.
    """
    def setUp(self):
        """테스트 설정: DBFeed 인스턴스 생성 및 테스트 데이터 준비"""
        self.db_feed = DBFeed()
        
        # 테스트 전 모든 Feed 관련 테이블을 삭제하고 다시 생성하여 클린 상태를 보장
        self._drop_all_feed_tables()
        self._create_all_feed_tables()

        # --- 테스트 데이터 준비 ---
        self.test_stock_info_data = [
            {'stock_code': 'A005930', 'stock_name': '삼성전자', 'market_type': 'KOSPI', 'sector': '반도체'},
            {'stock_code': 'A000660', 'stock_name': 'SK하이닉스', 'market_type': 'KOSPI', 'sector': '반도체'},
            {'stock_code': 'A293490', 'stock_name': '카카오게임즈', 'market_type': 'KOSDAQ', 'sector': '게임소프트웨어'}
        ]

        self.test_ohlcv_minute_data = [
            {'stock_code': 'A005930', 'datetime': datetime(2025, 7, 10, 9, 0, 0), 'open': 70000.0, 'high': 70100.0, 'low': 69900.0, 'close': 70050.0, 'volume': 100000},
            {'stock_code': 'A005930', 'datetime': datetime(2025, 7, 10, 9, 1, 0), 'open': 70050.0, 'high': 70200.0, 'low': 70000.0, 'close': 70150.0, 'volume': 80000},
            {'stock_code': 'A000660', 'datetime': datetime(2025, 7, 10, 9, 0, 0), 'open': 150000.0, 'high': 150200.0, 'low': 149900.0, 'close': 150100.0, 'volume': 50000}
        ]

        self.test_market_volume_data = [
            {'market_type': 'KOSPI', 'date': date(2025, 7, 10), 'time': time(9, 0, 0), 'total_amount': 100000000000},
            {'market_type': 'KOSDAQ', 'date': date(2025, 7, 10), 'time': time(9, 0, 0), 'total_amount': 50000000000},
            {'market_type': 'KOSPI', 'date': date(2025, 7, 10), 'time': time(9, 5, 0), 'total_amount': 105000000000}
        ]

        self.test_news_raw_data = [
            {'source': 'Creon News', 'datetime': datetime(2025, 7, 10, 10, 0, 0), 'title': '삼성전자, 신형 칩 개발 성공', 'content': '삼성전자가 혁신적인 신형 반도체 칩 개발에 성공했다고 발표했다.', 'url': 'http://news.samsung.com/1', 'related_stocks': ['A005930']},
            {'source': 'Telegram', 'datetime': datetime(2025, 7, 10, 10, 5, 0), 'title': '긴급속보: SK하이닉스 대규모 투자 발표', 'content': 'SK하이닉스가 파운드리 사업에 대규모 투자를 단행할 예정.', 'url': None, 'related_stocks': ['A000660']}
        ]

        self.test_investor_trends_data = [
            {'stock_code': 'A005930', 'date': date(2025, 7, 10), 'time': time(9, 30, 0), 'current_price': 70100, 'volume_total': 150000,
             'net_foreign': 10000, 'net_institutional': 5000, 'net_insurance_etc': 1000, 'net_trust': 2000,
             'net_bank': 500, 'net_pension': 1500, 'net_gov_local': 200, 'net_other_corp': 800, 'data_type': '수량'},
            {'stock_code': 'A000660', 'date': date(2025, 7, 10), 'time': time(9, 30, 0), 'current_price': 150100, 'volume_total': 70000,
             'net_foreign': -5000, 'net_institutional': -2000, 'net_insurance_etc': -500, 'net_trust': -1000,
             'net_bank': -200, 'net_pension': -800, 'net_gov_local': -100, 'net_other_corp': -400, 'data_type': '수량'}
        ]

        # news_summaries는 original_news_id가 필요하므로, news_raw 저장 후 ID를 얻어와야 함
        self.test_news_summaries_data = [] # 동적으로 채워질 예정

        self.test_thematic_stocks_data = [
            {'theme_name': '반도체', 'stock_code': 'A005930', 'analysis_date': date(2025, 7, 10), 'relevance_score': 0.95, 'mention_count': 50},
            {'theme_name': '반도체', 'stock_code': 'A000660', 'analysis_date': date(2025, 7, 10), 'relevance_score': 0.88, 'mention_count': 40},
            {'theme_name': '게임', 'stock_code': 'A293490', 'analysis_date': date(2025, 7, 10), 'relevance_score': 0.75, 'mention_count': 15}
        ]

        self.test_daily_universe_data = [
            {'stock_code': 'A005930', 'date': date(2025, 7, 10), 'total_score': 90.5, 'score_detail': {'price': 30, 'volume': 25, 'theme': 35.5}, 'is_selected': True},
            {'stock_code': 'A000660', 'date': date(2025, 7, 10), 'total_score': 85.0, 'score_detail': {'price': 28, 'volume': 22, 'theme': 35}, 'is_selected': True},
            {'stock_code': 'A293490', 'date': date(2025, 7, 10), 'total_score': 60.0, 'score_detail': {'price': 15, 'volume': 20, 'theme': 25}, 'is_selected': False}
        ]


    def tearDown(self):
        """테스트 정리: DB 연결 종료 및 테이블 정리"""
        self._drop_all_feed_tables() # 테스트 후 테이블 삭제
        self.db_feed.close()

    def test_01_create_and_drop_stock_tables(self):
        """종목 관련 테이블 생성 및 삭제 테스트"""
        # 테이블 삭제
        result = self.db_manager.drop_stock_tables()
        self.assertTrue(result)
        
        # 테이블 생성
        result = self.db_manager.create_stock_tables()
        self.assertTrue(result)

    # --- stock_info 테스트 ---
    def test_01_save_and_fetch_stock_info(self):
        """종목 기본 정보 저장 및 조회 테스트"""
        self.assertTrue(self.db_feed.save_stock_info(self.test_stock_info_data))

        fetched_map = self.db_feed.fetch_stock_info_map()
        self.assertEqual(len(fetched_map), 3)
        self.assertIn('A005930', fetched_map)
        self.assertEqual(fetched_map['A005930'], '삼성전자')

        # 업데이트 테스트
        updated_stock_info = [
            {'stock_code': 'A005930', 'stock_name': '삼성전자(업데이트)', 'market_type': 'KOSPI', 'sector': '반도체 및 장비'}
        ]
        self.assertTrue(self.db_feed.save_stock_info(updated_stock_info))
        fetched_map_updated = self.db_feed.fetch_stock_info_map()
        self.assertEqual(fetched_map_updated['A005930'], '삼성전자(업데이트)')

    # --- ohlcv_minute 테스트 ---
    def test_02_save_and_fetch_ohlcv_minute(self):
        """분봉 데이터 저장 및 조회 테스트"""
        self.db_feed.save_stock_info(self.test_stock_info_data) # FK 제약조건 만족
        self.assertTrue(self.db_feed.save_ohlcv_minute(self.test_ohlcv_minute_data))

        fetched_df = self.db_feed.fetch_ohlcv_minute('A005930', datetime(2025, 7, 10, 8, 0, 0), datetime(2025, 7, 10, 9, 30, 0))
        self.assertIsInstance(fetched_df, pd.DataFrame)
        self.assertEqual(len(fetched_df), 2)
        self.assertEqual(fetched_df.index.name, 'datetime')
        self.assertAlmostEqual(fetched_df.loc[datetime(2025, 7, 10, 9, 1, 0)]['close'], 70150.0)

        # 중복 데이터 업데이트 테스트 (close, volume 변경)
        updated_ohlcv = [
            {'stock_code': 'A005930', 'datetime': datetime(2025, 7, 10, 9, 0, 0), 'open': 70000.0, 'high': 70100.0, 'low': 69900.0, 'close': 70060.0, 'volume': 110000}
        ]
        self.assertTrue(self.db_feed.save_ohlcv_minute(updated_ohlcv))
        fetched_df_updated = self.db_feed.fetch_ohlcv_minute('A005930', datetime(2025, 7, 10, 9, 0, 0), datetime(2025, 7, 10, 9, 0, 0))
        self.assertAlmostEqual(fetched_df_updated.loc[datetime(2025, 7, 10, 9, 0, 0)]['close'], 70060.0)
        self.assertEqual(fetched_df_updated.loc[datetime(2025, 7, 10, 9, 0, 0)]['volume'], 110000)

    # --- market_volume 테스트 ---
    def test_03_save_and_fetch_market_volume(self):
        """시장별 거래대금 저장 및 조회 테스트"""
        self.assertTrue(self.db_feed.save_market_volume(self.test_market_volume_data))

        fetched_df = self.db_feed.fetch_market_volume('KOSPI', date(2025, 7, 10), date(2025, 7, 10))
        self.assertIsInstance(fetched_df, pd.DataFrame)
        self.assertEqual(len(fetched_df), 2)
        self.assertEqual(fetched_df.index.name, 'datetime')
        self.assertAlmostEqual(fetched_df.loc[datetime(2025, 7, 10, 9, 5, 0)]['total_amount'], 105000000000)

        # 중복 데이터 업데이트 테스트
        updated_volume = [
            {'market_type': 'KOSPI', 'date': date(2025, 7, 10), 'time': time(9, 0, 0), 'total_amount': 101000000000}
        ]
        self.assertTrue(self.db_feed.save_market_volume(updated_volume))
        fetched_df_updated = self.db_feed.fetch_market_volume('KOSPI', date(2025, 7, 10), date(2025, 7, 10))
        self.assertAlmostEqual(fetched_df_updated.loc[datetime(2025, 7, 10, 9, 0, 0)]['total_amount'], 101000000000)

    # --- news_raw 테스트 ---
    def test_04_save_and_fetch_news_raw(self):
        """원본 뉴스/메시지 저장 및 조회 테스트"""
        self.assertTrue(self.db_feed.save_news_raw(self.test_news_raw_data))

        fetched_df = self.db_feed.fetch_news_raw(datetime(2025, 7, 10, 9, 0, 0), datetime(2025, 7, 10, 11, 0, 0))
        self.assertIsInstance(fetched_df, pd.DataFrame)
        self.assertEqual(len(fetched_df), 2)
        self.assertEqual(fetched_df.iloc[0]['title'], '삼성전자, 신형 칩 개발 성공')
        self.assertEqual(fetched_df.iloc[1]['source'], 'Telegram')
        self.assertEqual(fetched_df.iloc[0]['related_stocks'], ['A005930']) # JSON 파싱 확인

        # 특정 source로 조회
        fetched_creon_news = self.db_feed.fetch_news_raw(datetime(2025, 7, 10, 9, 0, 0), datetime(2025, 7, 10, 11, 0, 0), source='Creon News')
        self.assertEqual(len(fetched_creon_news), 1)
        self.assertEqual(fetched_creon_news.iloc[0]['source'], 'Creon News')

    # --- investor_trends 테스트 ---
    def test_05_save_and_fetch_investor_trends(self):
        """투자자 매매 동향 저장 및 조회 테스트"""
        self.db_feed.save_stock_info(self.test_stock_info_data) # FK 제약조건 만족
        self.assertTrue(self.db_feed.save_investor_trends(self.test_investor_trends_data))

        fetched_df = self.db_feed.fetch_investor_trends('A005930', date(2025, 7, 10), date(2025, 7, 10), '수량')
        self.assertIsInstance(fetched_df, pd.DataFrame)
        self.assertEqual(len(fetched_df), 1)
        self.assertAlmostEqual(fetched_df.iloc[0]['net_foreign'], 10000)

        # 중복 데이터 업데이트 테스트 (net_foreign 변경)
        updated_trend = [
            {'stock_code': 'A005930', 'date': date(2025, 7, 10), 'time': time(9, 30, 0), 'current_price': 70100, 'volume_total': 150000,
             'net_foreign': 12000, 'net_institutional': 5000, 'net_insurance_etc': 1000, 'net_trust': 2000,
             'net_bank': 500, 'net_pension': 1500, 'net_gov_local': 200, 'net_other_corp': 800, 'data_type': '수량'}
        ]
        self.assertTrue(self.db_feed.save_investor_trends(updated_trend))
        fetched_df_updated = self.db_feed.fetch_investor_trends('A005930', date(2025, 7, 10), date(2025, 7, 10), '수량')
        self.assertAlmostEqual(fetched_df_updated.iloc[0]['net_foreign'], 12000)

    # --- news_summaries 테스트 ---
    def test_06_save_and_fetch_news_summaries(self):
        """뉴스 요약 및 감성 분석 결과 저장 및 조회 테스트"""
        # news_raw에 먼저 데이터 저장 및 ID 획득
        self.db_feed.save_news_raw(self.test_news_raw_data)
        raw_news_df = self.db_feed.fetch_news_raw(datetime(2025, 7, 10, 9, 0, 0), datetime(2025, 7, 10, 11, 0, 0))
        
        # original_news_id를 사용하여 test_news_summaries_data 구성
        self.test_news_summaries_data = [
            {'original_news_id': raw_news_df.loc[raw_news_df['title'].str.contains('삼성전자')].iloc[0]['id'],
             'summary': '삼성전자 신형 칩 개발 성공 소식.', 'sentiment_score': 0.8, 'processed_at': datetime(2025, 7, 10, 10, 10, 0)},
            {'original_news_id': raw_news_df.loc[raw_news_df['title'].str.contains('SK하이닉스')].iloc[0]['id'],
             'summary': 'SK하이닉스 대규모 투자 발표.', 'sentiment_score': 0.7, 'processed_at': datetime(2025, 7, 10, 10, 15, 0)}
        ]
        self.assertTrue(self.db_feed.save_news_summaries(self.test_news_summaries_data))

        fetched_df = self.db_feed.fetch_news_summaries(datetime(2025, 7, 10, 10, 0, 0), datetime(2025, 7, 10, 10, 20, 0))
        self.assertIsInstance(fetched_df, pd.DataFrame)
        self.assertEqual(len(fetched_df), 2)
        self.assertAlmostEqual(fetched_df.iloc[0]['sentiment_score'], 0.8)

    # --- thematic_stocks 테스트 ---
    def test_07_save_and_fetch_thematic_stocks(self):
        """테마별 관련 종목 저장 및 조회 테스트"""
        self.db_feed.save_stock_info(self.test_stock_info_data) # FK 제약조건 만족
        self.assertTrue(self.db_feed.save_thematic_stocks(self.test_thematic_stocks_data))

        fetched_df = self.db_feed.fetch_thematic_stocks(date(2025, 7, 10), theme_name='반도체')
        self.assertIsInstance(fetched_df, pd.DataFrame)
        self.assertEqual(len(fetched_df), 2)
        self.assertEqual(fetched_df.iloc[0]['stock_code'], 'A005930') # relevance_score 기준 내림차순 정렬 확인

        # 중복 데이터 업데이트 테스트 (relevance_score 변경)
        updated_thematic = [
            {'theme_name': '반도체', 'stock_code': 'A005930', 'analysis_date': date(2025, 7, 10), 'relevance_score': 0.98, 'mention_count': 55}
        ]
        self.assertTrue(self.db_feed.save_thematic_stocks(updated_thematic))
        fetched_df_updated = self.db_feed.fetch_thematic_stocks(date(2025, 7, 10), theme_name='반도체')
        self.assertAlmostEqual(fetched_df_updated.loc[fetched_df_updated['stock_code'] == 'A005930'].iloc[0]['relevance_score'], 0.98)

    # --- daily_universe 테스트 ---
    def test_08_save_and_fetch_daily_universe(self):
        """일일 매매 유니버스 저장 및 조회 테스트"""
        self.db_feed.save_stock_info(self.test_stock_info_data) # FK 제약조건 만족
        self.assertTrue(self.db_feed.save_daily_universe(self.test_daily_universe_data))

        fetched_df = self.db_feed.fetch_daily_universe(date(2025, 7, 10))
        self.assertIsInstance(fetched_df, pd.DataFrame)
        self.assertEqual(len(fetched_df), 3)
        self.assertEqual(fetched_df.iloc[0]['stock_code'], 'A005930') # total_score 기준 내림차순 정렬 확인
        self.assertTrue(fetched_df.iloc[0]['is_selected'])
        self.assertIsInstance(fetched_df.iloc[0]['score_detail'], dict) # JSON 파싱 확인

        # is_selected 필터링
        fetched_selected_df = self.db_feed.fetch_daily_universe(date(2025, 7, 10), is_selected=True)
        self.assertEqual(len(fetched_selected_df), 2)
        self.assertTrue(all(fetched_selected_df['is_selected']))

        # 중복 데이터 업데이트 테스트 (total_score, is_selected 변경)
        updated_universe = [
            {'stock_code': 'A005930', 'date': date(2025, 7, 10), 'total_score': 92.0, 'score_detail': {'price': 32, 'volume': 25, 'theme': 35}, 'is_selected': False}
        ]
        self.assertTrue(self.db_feed.save_daily_universe(updated_universe))
        fetched_df_updated = self.db_feed.fetch_daily_universe(date(2025, 7, 10))
        self.assertAlmostEqual(fetched_df_updated.loc[fetched_df_updated['stock_code'] == 'A005930'].iloc[0]['total_score'], 92.0)
        self.assertFalse(fetched_df_updated.loc[fetched_df_updated['stock_code'] == 'A005930'].iloc[0]['is_selected'])


if __name__ == '__main__':
    unittest.main()
