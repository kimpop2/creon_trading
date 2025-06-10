# backtesting/db/db_manager.py

import pymysql
import logging
import pandas as pd
from datetime import datetime, date, timedelta
import os
import sys

# sys.path에 프로젝트 루트 추가 (settings.py 임포트를 위함)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # 기본 로그 레벨 설정

class DBManager:
    def __init__(self):
        self.host = DB_HOST
        self.port = DB_PORT
        self.user = DB_USER
        self.password = DB_PASSWORD
        self.db_name = DB_NAME
        self.conn = None
        self._connect()

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
            logger.info(f"데이터베이스 '{self.db_name}'에 성공적으로 연결되었습니다.")
        except pymysql.err.MySQLError as e:
            logger.error(f"데이터베이스 연결 실패: {e}", exc_info=True)
            self.conn = None # 연결 실패 시 conn 초기화

    def get_db_connection(self):
        """현재 데이터베이스 연결 객체를 반환합니다. 연결이 끊어졌으면 재연결을 시도합니다."""
        if not self.conn or not self.conn.open:
            logger.warning("데이터베이스 연결이 끊어졌습니다. 재연결을 시도합니다.")
            self._connect()
        return self.conn

    def close(self):
        """데이터베이스 연결을 닫습니다."""
        if self.conn and self.conn.open:
            self.conn.close()
            logger.info("데이터베이스 연결이 닫혔습니다.")

    def create_all_tables(self):
        """schema.sql 파일의 SQL 쿼리를 실행하여 모든 테이블을 생성합니다."""
        conn = self.get_db_connection()
        if not conn:
            logger.error("DB 연결이 없어 테이블을 생성할 수 없습니다.")
            return

        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                sql_script = f.read()

            sql_commands = sql_script.split(';')
            with conn.cursor() as cursor:
                for command in sql_commands:
                    command = command.strip()
                    if command: # 빈 문자열이 아니면 실행
                        cursor.execute(command)
            conn.commit()
            logger.info("모든 테이블이 성공적으로 생성되었거나 이미 존재합니다.")
        except FileNotFoundError:
            logger.error(f"스키마 파일 '{schema_path}'을 찾을 수 없습니다.")
        except Exception as e:
            logger.error(f"테이블 생성 중 오류 발생: {e}", exc_info=True)
            conn.rollback()

    def drop_all_tables(self):
        """모든 테이블을 삭제합니다."""
        conn = self.get_db_connection()
        if not conn:
            logger.error("DB 연결이 없어 테이블을 삭제할 수 없습니다.")
            return

        # 외래 키 제약 조건이 있는 테이블부터 먼저 삭제
        tables_to_drop = ['minute_stock_data', 'daily_stock_data', 'stock_info'] # stock_finance 제거
        try:
            with conn.cursor() as cursor:
                for table_name in tables_to_drop:
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    logger.info(f"테이블 '{table_name}' 삭제 완료.")
            conn.commit()
            logger.info("모든 테이블이 성공적으로 삭제되었습니다.")
        except Exception as e:
            logger.error(f"테이블 삭제 중 오류 발생: {e}", exc_info=True)
            conn.rollback()

    def save_stock_info(self, stock_info_list):
        """
        종목 기본 정보 및 최신 재무 데이터를 DB의 stock_info 테이블에 저장하거나 업데이트합니다.
        :param stock_info_list: [{'stock_code': 'A005930', 'stock_name': '삼성전자', 'per': 10.5, ...}, ...]
        """
        conn = self.get_db_connection()
        if not conn: return False
        sql = """
        INSERT INTO stock_info
        (stock_code, stock_name, market_type, sector, per, pbr, eps, roe, debt_ratio, sales, operating_profit, net_profit, recent_financial_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            stock_name=VALUES(stock_name),
            market_type=VALUES(market_type),
            sector=VALUES(sector),
            per=VALUES(per),
            pbr=VALUES(pbr),
            eps=VALUES(eps),
            roe=VALUES(roe),
            debt_ratio=VALUES(debt_ratio),
            sales=VALUES(sales),
            operating_profit=VALUES(operating_profit),
            net_profit=VALUES(net_profit),
            recent_financial_date=VALUES(recent_financial_date),
            upd_date=CURRENT_TIMESTAMP
        """
        try:
            with conn.cursor() as cursor:
                data = []
                for info in stock_info_list:
                    # pbr은 Creon MarketEye에서 직접 제공되지 않을 수 있으므로, 기본값 0 또는 None
                    # schema.sql에 pbr 컬럼이 있으므로, 값을 넣어주거나 NULL 허용해야 함
                    pbr_value = info.get('pbr') # MarketEye에서 PBR 필드를 요청하지 않았다면 None이 될 것
                    if pbr_value is None: # 또는 0.0으로 초기화
                        pbr_value = 0.0 # 스키마가 DECIMAL(10,2)이고 NOT NULL이 아니므로 NULL도 가능

                    data.append((
                        info['stock_code'],
                        info['stock_name'],
                        info.get('market_type'),
                        info.get('sector'),
                        info.get('per'),
                        pbr_value, # pbr 값 처리
                        info.get('eps'),
                        info.get('roe'),
                        info.get('debt_ratio'),
                        info.get('sales'),
                        info.get('operating_profit'),
                        info.get('net_profit'),
                        info.get('recent_financial_date') # 새로운 재무 기준일 컬럼
                    ))
                cursor.executemany(sql, data)
            conn.commit()
            logger.debug(f"{len(stock_info_list)}개의 종목 정보를 저장/업데이트했습니다.")
            return True
        except Exception as e:
            logger.error(f"종목 정보 저장/업데이트 오류: {e}", exc_info=True)
            conn.rollback()
            return False

    def fetch_stock_info(self, stock_codes=None):
        """
        DB에서 종목 기본 정보 및 최신 재무 데이터를 조회합니다.
        :param stock_codes: 조회할 종목 코드 리스트 (없으면 전체 조회)
        :return: Pandas DataFrame
        """
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()
        sql = """
        SELECT stock_code, stock_name, market_type, sector, per, pbr, eps, roe, debt_ratio, sales, operating_profit, net_profit, recent_financial_date
        FROM stock_info
        """
        if stock_codes:
            placeholders = ','.join(['%s'] * len(stock_codes))
            sql += f" WHERE stock_code IN ({placeholders})"
        try:
            with conn.cursor() as cursor:
                if stock_codes:
                    cursor.execute(sql, stock_codes)
                else:
                    cursor.execute(sql)
                result = cursor.fetchall()
                return pd.DataFrame(result)
        except Exception as e:
            logger.error(f"종목 정보 조회 오류: {e}", exc_info=True)
            return pd.DataFrame()

    def save_daily_data(self, daily_data_list):
        """
        일봉 데이터를 DB에 저장하거나 업데이트합니다.
        :param daily_data_list: [{'stock_code': 'A005930', 'date': '2023-01-02', ...}, ...]
        """
        conn = self.get_db_connection()
        if not conn: return False
        sql = """
        INSERT INTO daily_stock_data
        (stock_code, date, open_price, high_price, low_price, close_price, volume, change_rate, trading_value)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open_price=VALUES(open_price),
            high_price=VALUES(high_price),
            low_price=VALUES(low_price),
            close_price=VALUES(close_price),
            volume=VALUES(volume),
            change_rate=VALUES(change_rate),
            trading_value=VALUES(trading_value)
        """
        try:
            with conn.cursor() as cursor:
                data = [(d['stock_code'], d['date'], d['open_price'], d['high_price'],
                         d['low_price'], d['close_price'], d['volume'],
                         d.get('change_rate'), d.get('trading_value'))
                        for d in daily_data_list]
                cursor.executemany(sql, data)
            conn.commit()
            logger.debug(f"{len(daily_data_list)}개의 일봉 데이터를 저장/업데이트했습니다.")
            return True
        except Exception as e:
            logger.error(f"일봉 데이터 저장/업데이트 오류: {e}", exc_info=True)
            conn.rollback()
            return False

    def fetch_daily_data(self, stock_code, start_date=None, end_date=None):
        """
        DB에서 특정 종목의 일봉 데이터를 조회합니다.
        :param stock_code: 조회할 종목 코드
        :param start_date: 시작 날짜 (YYYY-MM-DD 또는 date 객체)
        :param end_date: 종료 날짜 (YYYY-MM-DD 또는 date 객체)
        :return: Pandas DataFrame
        """
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()
        sql = """
        SELECT stock_code, date, open_price, high_price, low_price, close_price, volume, change_rate, trading_value
        FROM daily_stock_data
        WHERE stock_code = %s
        """
        params = [stock_code]
        if start_date:
            sql += " AND date >= %s"
            params.append(start_date)
        if end_date:
            sql += " AND date <= %s"
            params.append(end_date)
        sql += " ORDER BY date ASC" # 오래된 순서대로 정렬

        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, tuple(params))
                result = cursor.fetchall()
                return pd.DataFrame(result)
        except Exception as e:
            logger.error(f"일봉 데이터 조회 오류 ({stock_code}, {start_date}~{end_date}): {e}", exc_info=True)
            return pd.DataFrame()

    def get_latest_daily_data_date(self, stock_code):
        """
        특정 종목의 DB에 저장된 최신 일봉 데이터 날짜를 조회합니다.
        :param stock_code: 종목 코드
        :return: datetime.date 객체 또는 None
        """
        conn = self.get_db_connection()
        if not conn: return None
        sql = "SELECT MAX(date) AS latest_date FROM daily_stock_data WHERE stock_code = %s"
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, (stock_code,))
                result = cursor.fetchone()
                return result['latest_date'] if result and result['latest_date'] else None
        except Exception as e:
            logger.error(f"최신 일봉 날짜 조회 오류 ({stock_code}): {e}", exc_info=True)
            return None

    def save_minute_data(self, minute_data_list):
        """
        분봉 데이터를 DB에 저장하거나 업데이트합니다.
        :param minute_data_list: [{'stock_code': 'A005930', 'datetime': '2023-01-02 09:00', ...}, ...]
        """
        conn = self.get_db_connection()
        if not conn: return False
        sql = """
        INSERT INTO minute_stock_data
        (stock_code, datetime, open_price, high_price, low_price, close_price, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open_price=VALUES(open_price),
            high_price=VALUES(high_price),
            low_price=VALUES(low_price),
            close_price=VALUES(close_price),
            volume=VALUES(volume)
        """
        try:
            with conn.cursor() as cursor:
                data = [(d['stock_code'], d['datetime'], d['open_price'], d['high_price'],
                         d['low_price'], d['close_price'], d['volume'])
                        for d in minute_data_list]
                cursor.executemany(sql, data)
            conn.commit()
            logger.debug(f"{len(minute_data_list)}개의 분봉 데이터를 저장/업데이트했습니다.")
            return True
        except Exception as e:
            logger.error(f"분봉 데이터 저장/업데이트 오류: {e}", exc_info=True)
            conn.rollback()
            return False

    def fetch_minute_data(self, stock_code, date=None, start_datetime=None, end_datetime=None):
        """
        DB에서 특정 종목의 분봉 데이터를 조회합니다.
        :param stock_code: 조회할 종목 코드
        :param date: 특정 날짜의 데이터만 조회 (datetime.date 객체)
        :param start_datetime: 시작 시간 (datetime.datetime 객체)
        :param end_datetime: 종료 시간 (datetime.datetime 객체)
        :return: Pandas DataFrame
        """
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()
        sql = """
        SELECT stock_code, datetime, open_price, high_price, low_price, close_price, volume
        FROM minute_stock_data
        WHERE stock_code = %s
        """
        params = [stock_code]
        if date:
            sql += " AND DATE(datetime) = %s"
            params.append(date)
        if start_datetime:
            sql += " AND datetime >= %s"
            params.append(start_datetime)
        if end_datetime:
            sql += " AND datetime <= %s"
            params.append(end_datetime)
        sql += " ORDER BY datetime ASC" # 오래된 순서대로 정렬

        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, tuple(params))
                result = cursor.fetchall()
                return pd.DataFrame(result)
        except Exception as e:
            logger.error(f"분봉 데이터 조회 오류 ({stock_code}, {date}~{start_datetime}~{end_datetime}): {e}", exc_info=True)
            return pd.DataFrame()

    def get_latest_minute_data_datetime(self, stock_code):
        """
        특정 종목의 DB에 저장된 최신 분봉 데이터 시각을 조회합니다.
        :param stock_code: 종목 코드
        :return: datetime.datetime 객체 또는 None
        """
        conn = self.get_db_connection()
        if not conn: return None
        sql = "SELECT MAX(datetime) AS latest_datetime FROM minute_stock_data WHERE stock_code = %s"
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, (stock_code,))
                result = cursor.fetchone()
                return result['latest_datetime'] if result and result['latest_datetime'] else None
        except Exception as e:
            logger.error(f"최신 분봉 시각 조회 오류 ({stock_code}): {e}", exc_info=True)
            return None

    def get_all_stock_codes(self):
        """
        DB에 저장된 모든 종목 코드를 리스트로 반환합니다.
        """
        conn = self.get_db_connection()
        if not conn: return []
        sql = "SELECT stock_code FROM stock_info ORDER BY stock_code"
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()
                return [row['stock_code'] for row in results]
        except Exception as e:
            logger.error(f"모든 종목 코드 조회 오류: {e}", exc_info=True)
            return []