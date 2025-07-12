# feeds/db_feed.py

import pymysql
import logging
import pandas as pd
from datetime import datetime, date, timedelta, time
import os
import sys
import json
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple

# 로거 설정
logger = logging.getLogger(__name__)
from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
logger.debug("config/settings.py에서 DB 설정을 성공적으로 로드했습니다.")
# SQLAlchemy는 필요할 때 임포트 (Python 3.9 환경 호환성 고려)
try:
    from sqlalchemy import create_engine
except ImportError:
    create_engine = None
    logger.warning("SQLAlchemy가 설치되지 않았습니다. 'pip install sqlalchemy'를 실행해야 insert_df_to_db 메서드를 사용할 수 있습니다.")

# config.settings에서 DB 설정을 로드한다고 가정
# from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
# 예시 값 (실제 환경에 맞게 설정 필요)
DB_HOST = 'localhost'
DB_PORT = 3306
DB_USER = 'root'
DB_PASSWORD = 'admin'
DB_NAME = 'backtest_db'

try:
    from sqlalchemy import create_engine
except ImportError:
    create_engine = None
    logger.warning("SQLAlchemy is not installed. 'pip install sqlalchemy' is required for some DB operations.")

class DBFeed:
    """
    Feed 프로세스에서 사용할 데이터베이스 관리 클래스.
    Feed 관련 테이블에 대한 CRUD 작업을 담당합니다.
    """
    def __init__(self):
        self.host = DB_HOST
        self.port = DB_PORT
        self.user = DB_USER
        self.password = DB_PASSWORD
        self.db_name = DB_NAME
        self.conn = None
        self._connect()
        # SQLAlchemy Engine은 필요할 때 생성하도록 초기화 시점에는 생성하지 않음
        self._engine = None 

    def _connect(self):
        """데이터베이스에 연결합니다."""
        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db_name,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor # 딕셔너리 형태로 결과 반환
            )
            logger.info(f"DBFeed: Connected to database '{self.db_name}'.")
        except pymysql.err.MySQLError as e:
            logger.error(f"DBFeed: Database connection failed: {e}", exc_info=True)
            self.conn = None

    def get_db_connection(self):
        """현재 데이터베이스 연결 객체를 반환합니다. 연결이 끊어졌으면 재연결을 시도합니다."""
        if not self.conn or not self.conn.open:
            logger.warning("DBFeed: Database connection lost. Attempting to reconnect.")
            self._connect()
        return self.conn

    def close(self):
        """데이터베이스 연결을 닫습니다."""
        try:
            if self.conn and self.conn.open:
                self.conn.close()
            if self._engine:
                self._engine.dispose()
            logger.info("DBFeed: Database connection closed.")
        except pymysql.err.MySQLError as e:
            logger.error(f"DBFeed: Failed to close database connection: {e}", exc_info=True)
            self.conn = None

    def _get_db_engine(self):
        """SQLAlchemy Engine을 생성하거나 반환합니다."""
        if self._engine is None:
            if create_engine is None:
                logger.error("SQLAlchemy is not available. Cannot create DB engine.")
                return None
            try:
                db_url = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
                self._engine = create_engine(db_url, echo=False)
                logger.debug(f"DBFeed: SQLAlchemy Engine created.")
            except Exception as e:
                logger.error(f"DBFeed: Failed to create SQLAlchemy Engine: {e}", exc_info=True)
                self._engine = None
        return self._engine

    def execute_sql(self, sql: str, params: Optional[Any] = None) -> Optional[pymysql.cursors.DictCursor]:
        """
        SQL 쿼리를 실행하고 커밋하는 메서드.
        :param sql: 실행할 SQL 쿼리 문자열
        :param params: SQL 쿼리에 바인딩할 파라미터 (단일 튜플/딕셔너리 또는 executemany를 위한 리스트)
        :return: 쿼리 결과를 담은 커서 객체 (SELECT의 경우) 또는 None (오류 발생 시)
        """
        conn = self.get_db_connection()
        if not conn:
            logger.error("DBFeed: No DB connection to execute SQL.")
            return None

        try:
            with conn.cursor() as cur:
                if params:
                    if isinstance(params, list) and all(isinstance(p, tuple) for p in params):
                        cur.executemany(sql, params)
                    else:
                        cur.execute(sql, params)
                else:
                    cur.execute(sql)
                conn.commit()
                return cur
        except pymysql.MySQLError as e:
            logger.error(f"DBFeed: MariaDB Error during SQL execution: {e}. SQL: {sql}, Params: {params}", exc_info=True)
            conn.rollback()
            return None
        except Exception as e:
            logger.error(f"DBFeed: Unexpected error during SQL execution: {e}. SQL: {sql}, Params: {params}", exc_info=True)
            conn.rollback()
            return None

    def insert_df_to_db(self, table_name: str, df: pd.DataFrame, if_exists: str = "append", index: bool = False) -> bool:
        """
        DataFrame을 MariaDB에 삽입하는 메서드 (SQLAlchemy 사용).
        :param table_name: 데이터를 삽입할 테이블 이름
        :param df: 삽입할 데이터가 담긴 Pandas DataFrame
        :param if_exists: 테이블 존재 시 처리 방식 ('append', 'replace', 'fail')
        :param index: DataFrame의 인덱스를 DB 컬럼으로 저장할지 여부
        :return: 성공 시 True, 실패 시 False
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("DBFeed: Input data is not a pandas DataFrame.")
            return False

        engine = self._get_db_engine()
        if not engine:
            logger.error("DBFeed: Failed to get DB engine.")
            return False

        try:
            df.to_sql(table_name, con=engine, if_exists=if_exists, index=index)
            logger.debug(f"DBFeed: Successfully inserted {len(df)} rows into '{table_name}' with '{if_exists}' mode.")
            return True
        except Exception as e:
            logger.error(f"DBFeed: Error inserting DataFrame into '{table_name}': {e}", exc_info=True)
            return False

    # --- stock_info 테이블 관련 메서드 ---
    def save_stock_info(self, stock_info_list: List[Dict[str, Any]]) -> bool:
        """
        종목 기본 정보를 DB의 stock_info 테이블에 저장하거나 업데이트합니다.
        """
        sql = """
        INSERT INTO stock_info (stock_code, stock_name, market_type, sector)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            stock_name=VALUES(stock_name),
            market_type=VALUES(market_type),
            sector=VALUES(sector),
            updated_at=CURRENT_TIMESTAMP
        """
        data = [(s['stock_code'], s['stock_name'], s.get('market_type'), s.get('sector')) for s in stock_info_list]
        return self.execute_sql(sql, data) is not None

    def fetch_stock_info_map(self) -> Dict[str, str]:
        """DB에서 모든 종목 코드와 이름을 가져와 딕셔너리로 반환합니다."""
        sql = "SELECT stock_code, stock_name FROM stock_info"
        cursor = self.execute_sql(sql)
        if cursor:
            return {row['stock_code']: row['stock_name'] for row in cursor.fetchall()}
        return {}

    # --- ohlcv_minute 테이블 관련 메서드 ---
    def save_ohlcv_minute(self, ohlcv_data_list: List[Dict[str, Any]]) -> bool:
        """
        분봉 데이터를 DB의 ohlcv_minute 테이블에 저장하거나 업데이트합니다.
        """
        sql = """
        INSERT INTO ohlcv_minute (stock_code, datetime, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open=VALUES(open), high=VALUES(high), low=VALUES(low), close=VALUES(close),
            volume=VALUES(volume), updated_at=CURRENT_TIMESTAMP
        """
        data = [(d['stock_code'], d['datetime'], d['open'], d['high'],
                 d['low'], d['close'], d['volume']) for d in ohlcv_data_list]
        return self.execute_sql(sql, data) is not None

    def fetch_ohlcv_minute(self, stock_code: str, start_datetime: datetime, end_datetime: datetime) -> pd.DataFrame:
        """
        DB에서 특정 종목의 분봉 데이터를 조회합니다.
        """
        sql = """
        SELECT stock_code, datetime, open, high, low, close, volume
        FROM ohlcv_minute
        WHERE stock_code = %s AND datetime BETWEEN %s AND %s
        ORDER BY datetime ASC
        """
        params = (stock_code, start_datetime, end_datetime)
        cursor = self.execute_sql(sql, params)
        if cursor:
            df = pd.DataFrame(cursor.fetchall())
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, (Decimal, int, float)) else x)
            return df
        return pd.DataFrame()

    # --- market_volume 테이블 관련 메서드 ---
    def save_market_volume(self, market_volume_list: List[Dict[str, Any]]) -> bool:
        """
        시장별 거래대금 데이터를 DB의 market_volume 테이블에 저장하거나 업데이트합니다.
        """
        sql = """
        INSERT INTO market_volume (market_type, date, time, total_amount)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            total_amount=VALUES(total_amount), updated_at=CURRENT_TIMESTAMP
        """
        data = [(d['market_type'], d['date'], d['time'], d['total_amount']) for d in market_volume_list]
        return self.execute_sql(sql, data) is not None

    def fetch_market_volume(self, market_type: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        DB에서 시장별 거래대금 데이터를 조회합니다.
        """
        sql = """
        SELECT market_type, date, time, total_amount
        FROM market_volume
        WHERE market_type = %s AND date BETWEEN %s AND %s
        ORDER BY date ASC, time ASC
        """
        params = (market_type, start_date, end_date)
        cursor = self.execute_sql(sql, params)
        if cursor:
            df = pd.DataFrame(cursor.fetchall())
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
                df.set_index('datetime', inplace=True)
                df['total_amount'] = df['total_amount'].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
            return df
        return pd.DataFrame()

    # --- news_raw 테이블 관련 메서드 ---
    def save_news_raw(self, news_list: List[Dict[str, Any]]) -> bool:
        """
        원본 뉴스 및 텔레그램 메시지를 DB의 news_raw 테이블에 저장합니다.
        """
        sql = """
        INSERT INTO news_raw (source, datetime, title, content, url, related_stocks)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        data = []
        for n in news_list:
            related_stocks_json = json.dumps(n.get('related_stocks')) if n.get('related_stocks') else None
            data.append((
                n['source'], n['datetime'], n['title'], n.get('content'), n.get('url'), related_stocks_json
            ))
        return self.execute_sql(sql, data) is not None

    def fetch_news_raw(self, start_datetime: datetime, end_datetime: datetime, source: Optional[str] = None) -> pd.DataFrame:
        """
        DB에서 원본 뉴스 및 메시지 데이터를 조회합니다.
        """
        sql = "SELECT id, source, datetime, title, content, url, related_stocks FROM news_raw WHERE datetime BETWEEN %s AND %s"
        params = [start_datetime, end_datetime]
        if source:
            sql += " AND source = %s"
            params.append(source)
        sql += " ORDER BY datetime ASC"

        cursor = self.execute_sql(sql, params)
        if cursor:
            df = pd.DataFrame(cursor.fetchall())
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['related_stocks'] = df['related_stocks'].apply(lambda x: json.loads(x) if x else [])
            return df
        return pd.DataFrame()

    # --- investor_trends 테이블 관련 메서드 ---
    def save_investor_trends(self, trends_list: List[Dict[str, Any]]) -> bool:
        """
        투자자 매매 동향 데이터를 DB의 investor_trends 테이블에 저장하거나 업데이트합니다.
        """
        sql = """
        INSERT INTO investor_trends (stock_code, date, time, current_price, volume_total,
                                     net_foreign, net_institutional, net_insurance_etc, net_trust,
                                     net_bank, net_pension, net_gov_local, net_other_corp, data_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            current_price=VALUES(current_price), volume_total=VALUES(volume_total),
            net_foreign=VALUES(net_foreign), net_institutional=VALUES(net_institutional),
            net_insurance_etc=VALUES(net_insurance_etc), net_trust=VALUES(net_trust),
            net_bank=VALUES(net_bank), net_pension=VALUES(net_pension),
            net_gov_local=VALUES(net_gov_local), net_other_corp=VALUES(net_other_corp),
            updated_at=CURRENT_TIMESTAMP
        """
        data = []
        for t in trends_list:
            data.append((
                t['stock_code'], t['date'], t['time'], t.get('current_price'), t.get('volume_total'),
                t.get('net_foreign'), t.get('net_institutional'), t.get('net_insurance_etc'), t.get('net_trust'),
                t.get('net_bank'), t.get('net_pension'), t.get('net_gov_local'), t.get('net_other_corp'), t['data_type']
            ))
        return self.execute_sql(sql, data) is not None

    def fetch_investor_trends(self, stock_code: str, start_date: date, end_date: date, data_type: str) -> pd.DataFrame:
        """
        DB에서 특정 종목의 투자자 매매 동향 데이터를 조회합니다.
        """
        sql = """
        SELECT stock_code, date, time, current_price, volume_total,
               net_foreign, net_institutional, net_insurance_etc, net_trust,
               net_bank, net_pension, net_gov_local, net_other_corp, data_type
        FROM investor_trends
        WHERE stock_code = %s AND date BETWEEN %s AND %s AND data_type = %s
        ORDER BY date ASC, time ASC
        """
        params = (stock_code, start_date, end_date, data_type)
        cursor = self.execute_sql(sql, params)
        if cursor:
            df = pd.DataFrame(cursor.fetchall())
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
                df.set_index('datetime', inplace=True)
                for col in ['current_price', 'volume_total', 'net_foreign', 'net_institutional', 'net_insurance_etc', 'net_trust', 'net_bank', 'net_pension', 'net_gov_local', 'net_other_corp']:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
            return df
        return pd.DataFrame()

    # --- news_summaries 테이블 관련 메서드 ---
    def save_news_summaries(self, summaries_list: List[Dict[str, Any]]) -> bool:
        """
        NLP 분석을 통해 요약된 뉴스 및 감성 정보를 DB의 news_summaries 테이블에 저장합니다.
        """
        sql = """
        INSERT INTO news_summaries (original_news_id, summary, sentiment_score, processed_at)
        VALUES (%s, %s, %s, %s)
        """
        data = [(s['original_news_id'], s['summary'], s.get('sentiment_score'), s['processed_at']) for s in summaries_list]
        return self.execute_sql(sql, data) is not None

    def fetch_news_summaries(self, start_datetime: datetime, end_datetime: datetime, original_news_id: Optional[int] = None) -> pd.DataFrame:
        """
        DB에서 뉴스 요약 및 감성 분석 결과를 조회합니다.
        """
        sql = "SELECT id, original_news_id, summary, sentiment_score, processed_at FROM news_summaries WHERE processed_at BETWEEN %s AND %s"
        params = [start_datetime, end_datetime]
        if original_news_id:
            sql += " AND original_news_id = %s"
            params.append(original_news_id)
        sql += " ORDER BY processed_at ASC"

        cursor = self.execute_sql(sql, params)
        if cursor:
            df = pd.DataFrame(cursor.fetchall())
            if not df.empty:
                df['processed_at'] = pd.to_datetime(df['processed_at'])
                df['sentiment_score'] = df['sentiment_score'].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
            return df
        return pd.DataFrame()

    # --- thematic_stocks 테이블 관련 메서드 ---
    def save_thematic_stocks(self, thematic_list: List[Dict[str, Any]]) -> bool:
        """
        발굴된 테마 및 관련 종목 정보를 DB의 thematic_stocks 테이블에 저장하거나 업데이트합니다.
        """
        sql = """
        INSERT INTO thematic_stocks (theme_name, stock_code, analysis_date, relevance_score, mention_count)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            relevance_score=VALUES(relevance_score), mention_count=VALUES(mention_count),
            updated_at=CURRENT_TIMESTAMP
        """
        data = [(t['theme_name'], t['stock_code'], t['analysis_date'], t.get('relevance_score'), t.get('mention_count')) for t in thematic_list]
        return self.execute_sql(sql, data) is not None

    def fetch_thematic_stocks(self, analysis_date: date, theme_name: Optional[str] = None) -> pd.DataFrame:
        """
        DB에서 테마별 관련 종목 정보를 조회합니다.
        """
        sql = "SELECT theme_name, stock_code, analysis_date, relevance_score, mention_count FROM thematic_stocks WHERE analysis_date = %s"
        params = [analysis_date]
        if theme_name:
            sql += " AND theme_name = %s"
            params.append(theme_name)
        sql += " ORDER BY relevance_score DESC"

        cursor = self.execute_sql(sql, params)
        if cursor:
            df = pd.DataFrame(cursor.fetchall())
            if not df.empty:
                df['analysis_date'] = pd.to_datetime(df['analysis_date']).dt.date
                df['relevance_score'] = df['relevance_score'].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
            return df
        return pd.DataFrame()

    # --- daily_universe 테이블 관련 메서드 ---
    def save_daily_universe(self, universe_list: List[Dict[str, Any]]) -> bool:
        """
        일일 매매 유니버스 종목 및 점수를 DB의 daily_universe 테이블에 저장하거나 업데이트합니다.
        """
        sql = """
        INSERT INTO daily_universe (stock_code, date, total_score, score_detail, is_selected)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            total_score=VALUES(total_score), score_detail=VALUES(score_detail),
            is_selected=VALUES(is_selected), updated_at=CURRENT_TIMESTAMP
        """
        data = []
        for u in universe_list:
            score_detail_json = json.dumps(u.get('score_detail')) if u.get('score_detail') else None
            data.append((
                u['stock_code'], u['date'], u['total_score'], score_detail_json, u.get('is_selected', False)
            ))
        return self.execute_sql(sql, data) is not None

    def fetch_daily_universe(self, date: date, is_selected: Optional[bool] = None) -> pd.DataFrame:
        """
        DB에서 일일 매매 유니버스 선정 결과를 조회합니다.
        """
        sql = "SELECT stock_code, date, total_score, score_detail, is_selected FROM daily_universe WHERE date = %s"
        params = [date]
        if is_selected is not None:
            sql += " AND is_selected = %s"
            params.append(1 if is_selected else 0)
        sql += " ORDER BY total_score DESC"

        cursor = self.execute_sql(sql, params)
        if cursor:
            df = pd.DataFrame(cursor.fetchall())
            if not df.empty:
                df['date'] = pd.to_datetime(df['date']).dt.date
                df['total_score'] = df['total_score'].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                df['score_detail'] = df['score_detail'].apply(lambda x: json.loads(x) if x else {})
                df['is_selected'] = df['is_selected'].astype(bool)
            return df
        return pd.DataFrame()
