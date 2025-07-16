# manager/db_manager.py

import pymysql
import logging
import pandas as pd
from datetime import datetime, date, timedelta, time
import os
import sys
import json # JSON 직렬화를 위해 추가
import re
from decimal import Decimal # Decimal 타입 처리용
from typing import Dict, Any, Optional, List, Tuple
# --- 로거 설정 (스크립트 최상단에서 설정하여 항상 보이도록 함) ---
logger = logging.getLogger(__name__)
from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
logger.debug("config/settings.py에서 DB 설정을 성공적으로 로드했습니다.")
# SQLAlchemy는 필요할 때 임포트 (Python 3.9 환경 호환성 고려)
try:
    from sqlalchemy import create_engine
except ImportError:
    create_engine = None
    logger.warning("SQLAlchemy가 설치되지 않았습니다. 'pip install sqlalchemy'를 실행해야 insert_df_to_db 메서드를 사용할 수 있습니다.")

class DBManager:
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
            logger.info(f"데이터베이스 '{self.db_name}'에 성공적으로 연결되었습니다. (DBManager ID: {id(self)})")
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
        try:
            if self.conn and self.conn.open:
                self.conn.close()
            # SQLAlchemy Engine도 함께 닫아주는 것이 좋습니다.
            if self._engine:
                self._engine.dispose()
    
            logger.info(f"데이터베이스 연결이 닫혔습니다. (DBManager ID: {id(self)})")
        except pymysql.err.MySQLError as e:
            logger.error(f"데이터베이스 연결 닫기 실패: {e}", exc_info=True)
            self.conn = None # 연결 실패 시 conn 초기화

    def _get_db_engine(self):
        """SQLAlchemy Engine을 생성하거나 반환합니다."""
        if self._engine is None:
            try:
                db_url = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
                self._engine = create_engine(db_url, echo=False) # echo=True는 SQL 쿼리 로깅
                logger.debug(f"SQLAlchemy Engine 생성 완료: {db_url.split('@')[1]}")
            except Exception as e:
                logger.error(f"SQLAlchemy Engine 생성 실패: {e}", exc_info=True)
                self._engine = None
        return self._engine

    def check_table_exist(self, table_name):
        """MariaDB에서 테이블 존재 여부를 확인하는 메서드"""
        table_name = table_name.lower() # 테이블 이름을 소문자로 변환하여 비교
        conn = self.get_db_connection()
        if not conn:
            logger.error("DB 연결이 없어 테이블 존재 여부를 확인할 수 없습니다.")
            return False
        
        try:
            with conn.cursor() as cur:
                sql = "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = %s AND table_name = %s"
                cur.execute(sql, (self.db_name, table_name))
                result = cur.fetchone()
                exists = result['COUNT(*)'] > 0
                logger.debug(f"테이블 '{table_name}' 존재 여부 확인 결과: {exists}")
                return exists
        except pymysql.MySQLError as e:
            logger.error(f"MariaDB Error during table existence check for '{table_name}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during table existence check for '{table_name}': {e}", exc_info=True)
            return False

    def execute_script(self, sql_script):
        """SQL 스크립트를 실행하여 모든 테이블을 생성합니다."""
        conn = self.get_db_connection()
        if not conn:
            logger.error("DB 연결이 없어 SQL 스크립트를 실행할 수 없습니다.")
            return False

        cur_command = ''
        try:
            # Step 1: Remove all SQL comments (single-line -- and multi-line /* */)
            # This is a more robust approach to clean the script before splitting.
            cleaned_script = re.sub(r'--.*$', '', sql_script, flags=re.MULTILINE) # Remove single-line comments
            cleaned_script = re.sub(r'/\*.*?\*/', '', cleaned_script, flags=re.DOTALL) # Remove multi-line comments

            # Step 2: Split by semicolon and filter out empty strings and purely whitespace strings
            sql_commands = [
                cmd.strip() for cmd in cleaned_script.split(';') if cmd.strip()
            ]
            
            logger.debug(f"Parsed SQL commands ({len(sql_commands)} found):")
            for i, cmd in enumerate(sql_commands):
                logger.debug(f"  Command {i+1} (first 300 chars): {cmd[:300]}...") 

            if not sql_commands:
                logger.warning("SQL 스크립트에서 실행할 유효한 명령어를 찾을 수 없습니다. (스크립트가 비어있거나 모든 내용이 주석 처리되었나요?)")
                return False 

            with conn.cursor() as cursor:
                for i, command in enumerate(sql_commands):
                    cur_command = command
                    logger.debug(f"Executing SQL command {i+1}/{len(sql_commands)}: {cur_command[:100]}...") 
                    cursor.execute(cur_command)
            conn.commit()
            logger.info(f"SQL 스크립트 ({len(sql_commands)}개의 명령)가 정상적으로 실행되었습니다.")
            return True
        except Exception as e:
            logger.error(f"SQL 스크립트 실행 오류 발생: {e}", exc_info=True)
            logger.error(f"오류가 발생한 명령어: {cur_command}") 
            conn.rollback()
            return False

    def execute_sql_file(self, file_name):
        """특정 SQL 파일을 읽어 SQL 쿼리를 실행합니다."""
        # 이 스크립트 파일의 디렉토리를 기준으로 'sql' 하위 디렉토리를 찾습니다.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        script_dir = os.path.join(project_root, 'setup')
        sql_dir = os.path.join(script_dir, 'sql') 
        schema_path = os.path.join(sql_dir, file_name + '.sql')

        logger.info(f"SQL 파일 로드 및 실행 시도: {schema_path}")
        if not os.path.exists(schema_path): 
            logger.error(f"SQL 파일 '{schema_path}'을(를) 찾을 수 없습니다. 경로를 확인해주세요.")
            return False

        sql_script = ''
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            return self.execute_script(sql_script)
        except Exception as e:
            logger.error(f"SQL 파일 읽기 또는 실행 오류 발생: {e}", exc_info=True)
            return False
        

        
    def insert_df_to_db(self, table_name, df, option="append", is_index=False): # 기본값 append로 변경, index=False로 변경
        """DataFrame을 MariaDB에 삽입하는 메서드 (SQLAlchemy 사용)
        :param table_name: 데이터를 삽입할 테이블 이름
        :param df: 삽입할 데이터가 담긴 Pandas DataFrame
        :param option: 테이블 존재 시 처리 방식 ('append', 'replace', 'fail')
        :param is_index: DataFrame의 인덱스를 DB 컬럼으로 저장할지 여부
        """
        table_name = table_name.lower()
        if not isinstance(df, pd.DataFrame):
            logger.error("insert_df_to_db: 입력 데이터가 pandas DataFrame이 아닙니다.")
            return False

        if option not in ["replace", "append", "fail"]:
            logger.error(f"insert_df_to_db: option must be 'replace', 'append', or 'fail', but got '{option}'.")
            return False

        engine = self._get_db_engine()
        if not engine:
            logger.error("insert_df_to_db: DB 엔진 생성에 실패했습니다.")
            return False

        try:
            # index_label은 is_index가 True일 때만 의미가 있습니다.
            df.to_sql(table_name, con=engine, if_exists=option, index=is_index, 
                      index_label=df.index.name if is_index and df.index.name else None)
            logger.debug(f"DataFrame 데이터 {len(df)}행을 테이블 '{table_name}'에 '{option}' 모드로 성공적으로 삽입했습니다.")
            return True
        except Exception as e:
            logger.error(f"DataFrame 데이터를 테이블 '{table_name}'에 삽입 중 오류 발생: {e}", exc_info=True)
            return False

    def execute_sql(self, sql, param=None):
        """MariaDB에서 SQL 쿼리를 실행하고 커밋하는 메서드 (SELECT, INSERT, UPDATE, DELETE 등)
        :param sql: 실행할 SQL 쿼리 문자열
        :param param: SQL 쿼리에 바인딩할 파라미터 (단일 튜플/딕셔너리 또는 executemany를 위한 리스트)
        :return: 쿼리 결과를 담은 커서 객체 (SELECT의 경우) 또는 None (오류 발생 시)
        """
        conn = self.get_db_connection()
        if not conn:
            logger.error("DB 연결이 없어 SQL 쿼리를 실행할 수 없습니다.")
            return None

        try:
            with conn.cursor() as cur:
                if param:
                    logger.debug(f"SQL 실행: {sql} with params: {param}")
                    if isinstance(param, list): # 여러 개의 파라미터 리스트면 executemany
                        cur.executemany(sql, param)
                    else: # 단일 파라미터면 execute
                        cur.execute(sql, param)
                else: # 파라미터 없으면 execute
                    logger.debug(f"SQL 실행: {sql}")
                    cur.execute(sql)

                conn.commit()
                logger.debug("SQL 커밋 완료.")

                return cur # SELECT 쿼리의 경우 결과를 fetch하기 위해 커서 반환
        except pymysql.MySQLError as e:
            logger.error(f"MariaDB Error during SQL execution: {e}. SQL: {sql}, Params: {param}", exc_info=True)
            conn.rollback() 
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during SQL execution: {e}. SQL: {sql}, Params: {param}", exc_info=True)
            conn.rollback()
            return None
        
    def save_market_calendar(self, calendar_data_list: List[Dict[str, Any]]) -> bool:
        """
        시장 캘린더 데이터를 DB의 market_calendar 테이블에 저장하거나 업데이트합니다.
        'date' 컬럼을 UNIQUE KEY로 사용하여 중복 시 업데이트합니다.
        :param calendar_data_list: [{'date': datetime.date(2025, 1, 24), 'is_holiday': False, 'description': '거래일'}, ...]
        :return: 성공 시 True, 실패 시 False
        """
        conn = self.get_db_connection()
        if not conn:
            return False

        sql = """
        INSERT INTO market_calendar
        (date, is_holiday, description)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            is_holiday = VALUES(is_holiday),
            description = VALUES(description)
        """
        
        data = []
        for entry in calendar_data_list:
            date_to_save = entry['date']
            # Ensure date is a datetime.date object for the DB insertion
            # if isinstance(entry['date'], datetime.datetime):
            #     date_to_save = entry['date'].date()
            # elif isinstance(entry['date'], datetime.date):
            #     date_to_save = entry['date']
            # else:
            #     logger.warning(f"예상치 못한 'date' 타입: {type(entry['date'])}. datetime.date로 변환 시도.")
            #     try:
            #         date_to_save = datetime.datetime.strptime(str(entry['date']), '%Y-%m-%d').date()
            #     except ValueError:
            #         logger.error(f"날짜 변환 실패: {entry['date']}")
            #         continue # Skip this entry if date conversion fails

            data.append((
                date_to_save,
                entry['is_holiday'],
                entry['description']
            ))

        if not data:
            logger.warning("저장할 캘린더 데이터가 없습니다.")
            return True # No data to save, consider it a success

        try:
            cursor = self.execute_sql(sql, data) # Assumes execute_sql handles executemany
            if cursor:
                logger.info(f"{len(data)}개의 시장 캘린더 내역을 저장/업데이트했습니다.")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"시장 캘린더 내역 저장/업데이트 오류: {e}", exc_info=True)
            return False

    # ----------------------------------------------------------------------------
    # 종목/주가 관리 테이블
    # ----------------------------------------------------------------------------
    def create_stock_tables(self):
        return self.execute_sql_file('create_stock_tables')

    def drop_stock_tables(self):
        return self.execute_sql_file('drop_stock_tables')

    # --- stock_info 테이블 관련 메서드 ---
    def save_stock_info(self, stock_info_list: list):
        """
        종목 기본 정보 및 최신 재무 데이터를 DB의 stock_info 테이블에 저장하거나 업데이트합니다.
        :param stock_info_list: [{'stock_code': 'A005930', 'stock_name': '삼성전자', 'per': 10.5, ...}, ...]
        :return: 성공 시 True, 실패 시 False
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
            upd_date=CURRENT_TIMESTAMP() 
        """
        try:
            data = []
            for info in stock_info_list:
                pbr_value = info.get('pbr') 
                if pbr_value is None:
                    pbr_value = 0.0 

                data.append((
                    info['stock_code'],
                    info['stock_name'],
                    info.get('market_type'),
                    info.get('sector'),
                    info.get('per'),
                    pbr_value,
                    info.get('eps'),
                    info.get('roe'),
                    info.get('debt_ratio'),
                    info.get('sales'),
                    info.get('operating_profit'),
                    info.get('net_profit'),
                    info.get('recent_financial_date')
                ))
            
            cursor = self.execute_sql(sql, data)
            if cursor:
                logger.debug(f"{len(stock_info_list)}개의 종목 정보를 저장/업데이트했습니다.")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"save_stock_info 처리 중 예외 발생: {e}", exc_info=True)
            return False

    def fetch_stock_info(self, stock_codes: list = None):
        """
        DB에서 종목 기본 정보 및 최신 재무 데이터를 조회합니다.
        :param stock_codes: 조회할 종목 코드 리스트 (없으면 전체 조회)
        :return: Pandas DataFrame
        """
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()
        sql = """
        SELECT stock_code, stock_name, market_type, sector, per, pbr, eps, roe, debt_ratio, sales, operating_profit, net_profit, recent_financial_date, upd_date
        FROM stock_info
        """
        params = []
        if stock_codes:
            placeholders = ','.join(['%s'] * len(stock_codes))
            sql += f" WHERE stock_code IN ({placeholders})"
            params = stock_codes
        
        try:
            cursor = self.execute_sql(sql, tuple(params) if params else None)
            if cursor:
                result = cursor.fetchall()
                return pd.DataFrame(result)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"fetch_stock_info 처리 중 예외 발생: {e}", exc_info=True)
            return pd.DataFrame()
        
    # --- market_calendar 테이블에서 캘린더의 날짜를 가지고 온다 ---
    def get_all_trading_days(self, from_date: date, to_date: date) -> list[pd.Timestamp]: # <- 반환 타입 변경
        """
        DB의 market_calendar 테이블에서 지정된 기간의 모든 영업일 (is_holiday = FALSE) 날짜를 가져옵니다.
        :param from_date: 조회 시작일 (datetime.date 객체)
        :param to_date: 조회 종료일 (datetime.date 객체)
        :return: 영업일 날짜를 담은 list (pd.Timestamp 객체들) # <- 반환 타입 설명 변경
        """
        conn = self.get_db_connection()
        if not conn:
            logger.error("DB 연결 실패: 거래일 캘린더를 가져올 수 없습니다.")
            return []

        # is_holiday가 FALSE인 날짜만 선택하고, date 컬럼만 가져옴
        sql = """
        SELECT DISTINCT date 
        FROM market_calendar
        WHERE date BETWEEN %s AND %s
        ORDER BY date ASC
        """
        params = (from_date, to_date)
        
        try:
            cursor = self.execute_sql(sql, params)
            
            if cursor:
                # fetchall()은 딕셔너리 리스트를 반환하므로, 각 딕셔너리에서 'date' 값만 추출
                # 그리고 그 'date' 값 (datetime.date 객체)을 pd.Timestamp로 변환하고 normalize
                trading_days_ts = [pd.Timestamp(row['date']).normalize() for row in cursor.fetchall()] # <- 핵심 수정
                logger.debug(f"거래일 캘린더 로드 완료 ({from_date} ~ {to_date}): {len(trading_days_ts)}개 영업일")
                return trading_days_ts
            else:
                return []
        except Exception as e:
            logger.error(f"get_all_trading_days 처리 중 예외 발생: {e}", exc_info=True)
            return []

    def fetch_market_calendar(self, from_date: date, to_date: date) -> pd.DataFrame:
        """
        DB의 market_calendar 테이블에서 지정된 기간의 시장 캘린더 데이터를 조회합니다.
        :param from_date: 조회 시작일 (datetime.date 객체)
        :param to_date: 조회 종료일 (datetime.date 객체)
        :return: Pandas DataFrame (컬럼: 'date', 'is_holiday', 'description')
        """
        conn = self.get_db_connection()
        if not conn:
            logger.error("DB 연결 실패: 시장 캘린더를 가져올 수 없습니다.")
            return pd.DataFrame()

        sql = """
        SELECT date, is_holiday, description
        FROM market_calendar
        WHERE date BETWEEN %s AND %s
        ORDER BY date ASC
        """
        params = (from_date, to_date)
        
        try:
            cursor = self.execute_sql(sql, params)
            if cursor:
                result = cursor.fetchall()
                df = pd.DataFrame(result)
                if not df.empty:
                    #df['date'] = pd.to_datetime(df['date']).dt.date
                    df['date'] = pd.to_datetime(df['date']).dt.normalize()
                logger.debug(f"시장 캘린더 로드 완료 ({from_date} ~ {to_date}): {len(df)}개 날짜")
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"fetch_market_calendar 처리 중 예외 발생: {e}", exc_info=True)
            return pd.DataFrame()


    # --- daily_price 테이블 관련 메서드 ---
    def save_daily_price(self, daily_price_list: list):
        """
        일봉 데이터를 DB의 daily_price 테이블에 저장하거나 업데이트합니다.
        :param daily_price_list: [{'stock_code': 'A005930', 'date': '2023-01-02', 'open': ..., ...}, ...]
        :return: 성공 시 True, 실패 시 False
        """
        conn = self.get_db_connection()
        if not conn: return False
        
        sql = """
        INSERT INTO daily_price
        (stock_code, date, open, high, low, close, volume, change_rate, trading_value)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open=VALUES(open),
            high=VALUES(high),
            low=VALUES(low),
            close=VALUES(close),
            volume=VALUES(volume),
            change_rate=VALUES(change_rate),
            trading_value=VALUES(trading_value)
        """
        try:
            data = [(d['stock_code'], d['date'], d['open'], d['high'],
                     d['low'], d['close'], d['volume'],
                     d.get('change_rate'), d.get('trading_value'))
                    for d in daily_price_list]
            
            cursor = self.execute_sql(sql, data)
            if cursor:
                logger.info(f"{len(daily_price_list)}개의 일봉 데이터를 저장/업데이트했습니다.")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"save_daily_price 처리 중 예외 발생: {e}", exc_info=True)
            return False
        
    def fetch_daily_price(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
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
        SELECT stock_code, date, open, high, low, close, volume, change_rate, trading_value
        FROM daily_price
        WHERE stock_code = %s AND date BETWEEN %s AND %s 
        ORDER BY date ASC
        """
        params = (stock_code, start_date, end_date)
        
        try:
            cursor = self.execute_sql(sql, params)
            if cursor:
                result = cursor.fetchall()
                df = pd.DataFrame(result)
                if not df.empty:
                    # 핵심 수정: 'date' 컬럼을 pd.Timestamp로 변환하고 바로 인덱스로 설정
                    # .dt.date를 제거하여 datetime.date가 아닌 pd.Timestamp가 인덱스가 되도록 합니다.
                    df['date'] = pd.to_datetime(df['date']).dt.normalize()
                    df.set_index('date', inplace=True) # 인덱스 정규화
                       
                    # 숫자 컬럼을 float으로 명시적 변환
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'change_rate', 'trading_value']
                    for col in numeric_cols:
                        if col in df.columns:
                            # Decimal 객체가 올 수 있으므로 float()으로 변환
                            # None 값은 그대로 유지되도록 apply 사용
                            df[col] = df[col].apply(lambda x: float(x) if isinstance(x, (Decimal, int, float)) else x)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"fetch_daily_price 처리 중 예외 발생: {e}", exc_info=True)
            return pd.DataFrame()
        
    
    def get_latest_daily_price_date(self, stock_code: str):
        """
        특정 종목의 DB에 저장된 최신 일봉 데이터 날짜를 조회합니다.
        :param stock_code: 종목 코드
        :return: datetime.date 객체 또는 None
        """
        conn = self.get_db_connection()
        if not conn: return None
        sql = "SELECT MAX(date) AS latest_date FROM daily_price WHERE stock_code = %s"
        try:
            cursor = self.execute_sql(sql, (stock_code,))
            if cursor:
                result = cursor.fetchone()
                return result['latest_date'] if result and result['latest_date'] else None
            else:
                return None
        except Exception as e:
            logger.error(f"get_latest_daily_price_date 처리 중 예외 발생: {e}", exc_info=True)
            return None

    # --- minute_price 테이블 관련 메서드 ---
    def save_minute_price(self, minute_price_list: list):
        """
        일봉 데이터를 DB의 minute_price 테이블에 저장하거나 업데이트합니다.
        :param minute_price_list: [{'stock_code': 'A005930', 'datetime': '20230102090000', 'open': ..., ...}, ...]
        :return: 성공 시 True, 실패 시 False
        """
        conn = self.get_db_connection()
        if not conn: return False
        
        sql = """
        INSERT INTO minute_price
        (stock_code, datetime, open, high, low, close, volume, change_rate, trading_value)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open=VALUES(open),
            high=VALUES(high),
            low=VALUES(low),
            close=VALUES(close),
            volume=VALUES(volume),
            change_rate=VALUES(change_rate),
            trading_value=VALUES(trading_value)
        """
        try:
            data = [(d['stock_code'], d['datetime'], d['open'], d['high'],
                     d['low'], d['close'], d['volume'],
                     d.get('change_rate'), d.get('trading_value'))
                    for d in minute_price_list]
            
            cursor = self.execute_sql(sql, data)
            if cursor:
                logger.info(f"{len(minute_price_list)}개의 분봉 데이터를 저장/업데이트했습니다.")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"save_minute_price 처리 중 예외 발생: {e}", exc_info=True)
            return False
        
    def fetch_minute_price(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        DB에서 특정 종목의 분봉 데이터를 조회합니다.
        :param stock_code: 조회할 종목 코드
        :param start_date: 시작 날짜 (YYYY-MM-DD 또는 date 객체)
        :param end_date: 종료 날짜 (YYYY-MM-DD 또는 date 객체)
        :return: Pandas DataFrame
        """
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()
        
        # start_date와 end_date를 datetime으로 변환하여 정확한 시간 범위 지정
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        sql = """
        SELECT stock_code, datetime, open, high, low, close, volume, change_rate, trading_value
        FROM minute_price
        WHERE stock_code = %s 
        AND datetime >= %s 
        AND datetime <= %s
        ORDER BY datetime ASC
        """
        params = (stock_code, start_datetime, end_datetime)
        
        try:
            cursor = self.execute_sql(sql, params)
            if cursor:
                result = cursor.fetchall()
                df = pd.DataFrame(result)
                if not df.empty:
                    # 'datetime' 컬럼을 Pandas datetime 객체로 변환하고 인덱스로 설정
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)

                    # 숫자 컬럼을 float으로 명시적 변환
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'change_rate', 'trading_value']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = df[col].apply(lambda x: float(x) if isinstance(x, (Decimal, int, float)) else x)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"fetch_minute_price 처리 중 예외 발생: {e}", exc_info=True)
            return pd.DataFrame()

    # --- stock_info 관련 추가 메서드 ---
    def fetch_stock_codes_by_criteria(self, market_type: str = None, sector: str = None, per_max: float = None, 
                                     pbr_max: float = None, eps_min: float = None, roe_min: float = None, 
                                     debt_ratio_max: float = None, sales_min: int = None, 
                                     operating_profit_min: int = None, net_profit_min: int = None):
        """
        주어진 재무 지표 및 시장/섹터 조건에 따라 종목 코드를 조회합니다.
        :return: 조건에 맞는 종목 코드 리스트 (list[str])
        """
        conn = self.get_db_connection()
        if not conn: return []

        sql = "SELECT stock_code FROM stock_info WHERE 1=1"
        params = []

        if market_type:
            sql += " AND market_type = %s"
            params.append(market_type)
        if sector:
            sql += " AND sector = %s"
            params.append(sector)
        if per_max is not None:
            sql += " AND per <= %s"
            params.append(per_max)
        if pbr_max is not None:
            sql += " AND pbr <= %s"
            params.append(pbr_max)
        if eps_min is not None:
            sql += " AND eps >= %s"
            params.append(eps_min)
        if roe_min is not None:
            sql += " AND roe >= %s"
            params.append(roe_min)
        if debt_ratio_max is not None:
            sql += " AND debt_ratio <= %s"
            params.append(debt_ratio_max)
        if sales_min is not None:
            sql += " AND sales >= %s"
            params.append(sales_min)
        if operating_profit_min is not None:
            sql += " AND operating_profit >= %s"
            params.append(operating_profit_min)
        if net_profit_min is not None:
            sql += " AND net_profit >= %s"
            params.append(net_profit_min)
        
        sql += " ORDER BY stock_code"

        try:
            cursor = self.execute_sql(sql, tuple(params) if params else None)
            if cursor:
                results = cursor.fetchall()
                return [row['stock_code'] for row in results]
            else:
                return []
        except Exception as e:
            logger.error(f"조건부 종목 코드 조회 오류: {e}", exc_info=True)
            return []

    def get_all_stock_codes(self):
        """
        DB에 저장된 모든 종목 코드를 리스트로 반환합니다.
        :return: 모든 종목 코드 리스트 (list[str])
        """
        conn = self.get_db_connection()
        if not conn: return []
        sql = "SELECT stock_code FROM stock_info ORDER BY stock_code"
        try:
            cursor = self.execute_sql(sql)
            if cursor:
                results = cursor.fetchall()
                return [row['stock_code'] for row in results]
            else:
                return []
        except Exception as e:
            logger.error(f"모든 종목 코드 조회 오류: {e}", exc_info=True)
            return []



    # ----------------------------------------------------------------------------
    # 백테스트 관리 테이블
    # ----------------------------------------------------------------------------
    def create_backtest_tables(self):
        return self.execute_sql_file('create_backtest_tables')

    def drop_backtest_tables(self):
        return self.execute_sql_file('drop_backtest_tables')


    def fetch_backtest_performance(self, run_id: int):
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()

        query = "SELECT * FROM backtest_performance WHERE run_id = %s ORDER BY date ASC"
        try:
            result = self.execute_sql(query, (run_id,), fetch=True)
            if result:
                df = pd.DataFrame(result)
                # 숫자형 컬럼들을 명시적으로 float로 변환
                numeric_cols = ['end_capital', 'daily_return', 'cumulative_return', 'drawdown']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"일별 성과 정보 조회 오류 (run_id: {run_id}): {e}", exc_info=True)
            return pd.DataFrame()

    def fetch_backtest_trade(self, run_id: int):
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()

        query = "SELECT * FROM backtest_trade WHERE run_id = %s ORDER BY trade_datetime ASC"
        try:
            result = self.execute_sql(query, (run_id,), fetch=True)
            if result:
                df = pd.DataFrame(result)
                # 숫자형 컬럼들을 명시적으로 float로 변환
                numeric_cols = ['trade_price', 'trade_quantity', 'trade_amount', 'commission', 'realized_profit_loss']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"거래 로그 정보 조회 오류 (run_id: {run_id}): {e}", exc_info=True)
            return pd.DataFrame()
            
    # --- backtest_run 테이블 관련 메서드 ---
    def save_backtest_run(self, run_info: dict):
        """
        백테스트 실행 정보를 DB의 backtest_run 테이블에 저장하거나 업데이트합니다.
        :param run_info: 백테스트 실행 정보 딕셔너리
                         (run_id가 None이면 AUTO_INCREMENT, 있으면 해당 ID로 업데이트 시도)
        :return: 새로 삽입된 run_id (int) 또는 업데이트 성공 시 run_id (int), 실패 시 False
        """
        conn = self.get_db_connection()
        if not conn: return False
        
        sql = """
        INSERT INTO backtest_run
        (run_id, start_date, end_date, initial_capital, final_capital, total_profit_loss, 
         cumulative_return, max_drawdown, strategy_daily, strategy_minute, params_json_daily, params_json_minute)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            start_date=VALUES(start_date),
            end_date=VALUES(end_date),
            initial_capital=VALUES(initial_capital),
            final_capital=VALUES(final_capital),
            total_profit_loss=VALUES(total_profit_loss),
            cumulative_return=VALUES(cumulative_return),
            max_drawdown=VALUES(max_drawdown),
            strategy_daily=VALUES(strategy_daily),
            strategy_minute=VALUES(strategy_minute),
            params_json_daily=VALUES(params_json_daily),
            params_json_minute=VALUES(params_json_minute)
        """
        
        run_id_param = run_info.get('run_id') if run_info.get('run_id') is not None else None
        
        params = (
            run_id_param, 
            run_info.get('start_date'),
            run_info.get('end_date'),
            run_info.get('initial_capital'),
            run_info.get('final_capital'),
            run_info.get('total_profit_loss'),
            run_info.get('cumulative_return'),
            run_info.get('max_drawdown'),
            run_info.get('strategy_daily'),
            run_info.get('strategy_minute'),
            # --- 변경 부분: dict를 JSON 문자열로 직렬화 ---
            json.dumps(run_info.get('params_json_daily')) if run_info.get('params_json_daily') is not None else None, 
            json.dumps(run_info.get('params_json_minute')) if run_info.get('params_json_minute') is not None else None 
            # -----------------------------------------------
        )
        
        try:
            cursor = self.execute_sql(sql, params)
            if cursor:
                new_run_id = cursor.lastrowid if run_id_param is None else run_id_param
                logger.info(f"백테스트 실행 정보 저장/업데이트 성공. run_id: {new_run_id}")
                return new_run_id
            else:
                return False
        except Exception as e:
            logger.error(f"백테스트 실행 정보 저장/업데이트 오류: {e}", exc_info=True)
            return False

    def fetch_backtest_run(self, run_id: int = None, start_date: date = None, end_date: date = None):
        """
        DB에서 백테스트 실행 정보를 조회합니다.
        :param run_id: 조회할 백테스트 실행 ID
        :param start_date: 백테스트 시작일 필터링 (이상)
        :param end_date: 백테스트 종료일 필터링 (이하)
        :return: Pandas DataFrame
        """
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()
        
        sql = """
        SELECT run_id, start_date, end_date, initial_capital, final_capital, total_profit_loss, 
               cumulative_return, max_drawdown, strategy_daily, strategy_minute, 
               params_json_daily, params_json_minute, created_at
        FROM backtest_run
        WHERE 1=1
        """
        params = []
        if run_id is not None:
            sql += " AND run_id = %s"
            params.append(run_id)
        if start_date:
            sql += " AND start_date >= %s"
            params.append(start_date)
        if end_date:
            sql += " AND end_date <= %s"
            params.append(end_date)
        sql += " ORDER BY created_at DESC"

        try:
            cursor = self.execute_sql(sql, tuple(params) if params else None)
            if cursor:
                result = cursor.fetchall()
                return pd.DataFrame(result)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"백테스트 실행 정보 조회 오류 (run_id: {run_id}): {e}", exc_info=True)
            return pd.DataFrame()

    # --- backtest_trade 테이블 관련 메서드 ---
    def save_backtest_trade(self, trade_data_list: list):
        """
        백테스트 개별 거래 내역을 DB의 backtest_trade 테이블에 저장하거나 업데이트합니다.
        trade_id는 AUTO_INCREMENT이므로, 삽입 시에는 trade_id를 제외하고, 
        ON DUPLICATE KEY UPDATE 시에는 run_id, stock_code, trade_datetime UNIQUE KEY를 사용합니다.
        :param trade_data_list: [{'run_id': 1, 'stock_code': 'A005930', ...}, ...]
        :return: 성공 시 True, 실패 시 False
        """
        conn = self.get_db_connection()
        if not conn: return False
        
        # trade_id는 AUTO_INCREMENT이므로, INSERT 컬럼 리스트에서 제외
        sql = """
        INSERT INTO backtest_trade
        (run_id, stock_code, trade_type, trade_price, trade_quantity, trade_amount, 
         trade_datetime, commission, tax, realized_profit_loss, entry_trade_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            trade_type=VALUES(trade_type),
            trade_price=VALUES(trade_price),
            trade_quantity=VALUES(trade_quantity),
            trade_amount=VALUES(trade_amount),
            commission=VALUES(commission),
            tax=VALUES(tax),
            realized_profit_loss=VALUES(realized_profit_loss),
            entry_trade_id=VALUES(entry_trade_id)
        """
        data = []
        for trade in trade_data_list:
            data.append((
                trade['run_id'],
                trade['stock_code'],
                trade['trade_type'],
                trade['trade_price'],
                trade['trade_quantity'],
                trade['trade_amount'],
                trade['trade_datetime'],
                trade.get('commission'),
                trade.get('tax'),
                trade.get('realized_profit_loss'),
                trade.get('entry_trade_id')
            ))
        
        try:
            cursor = self.execute_sql(sql, data) # executemany를 위해 리스트 전달
            if cursor:
                logger.info(f"{len(trade_data_list)}개의 백테스트 거래 내역을 저장/업데이트했습니다.")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"백테스트 거래 내역 저장/업데이트 오류: {e}", exc_info=True)
            return False

    def fetch_backtest_trade(self, run_id: int, stock_code: str = None, start_datetime: datetime = None, end_datetime: datetime = None):
        """
        DB에서 백테스트 개별 거래 내역을 조회합니다.
        :param run_id: 조회할 백테스트 실행 ID (필수)
        :param stock_code: 조회할 종목 코드 (선택)
        :param start_datetime: 거래 시작 시각 (선택)
        :param end_datetime: 거래 종료 시각 (선택)
        :return: Pandas DataFrame
        """
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()
        
        sql = """
        SELECT trade_id, run_id, stock_code, trade_type, trade_price, trade_quantity, trade_amount, 
               trade_datetime, commission, tax, realized_profit_loss, entry_trade_id
        FROM backtest_trade
        WHERE run_id = %s
        """
        params = [run_id]
        if stock_code:
            sql += " AND stock_code = %s"
            params.append(stock_code)
        if start_datetime:
            sql += " AND trade_datetime >= %s"
            params.append(start_datetime)
        if end_datetime:
            sql += " AND trade_datetime <= %s"
            params.append(end_datetime)
        sql += " ORDER BY trade_datetime ASC"

        try:
            cursor = self.execute_sql(sql, tuple(params))
            if cursor:
                result = cursor.fetchall()
                return pd.DataFrame(result)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"백테스트 거래 내역 조회 오류 (run_id: {run_id}, stock_code: {stock_code}): {e}", exc_info=True)
            return pd.DataFrame()

    # --- backtest_performance 테이블 관련 메서드 ---
    def save_backtest_performance(self, performance_data_list: list):
        """
        백테스트 일별/기간별 성능 지표를 DB의 backtest_performance 테이블에 저장하거나 업데이트합니다.
        performance_id는 AUTO_INCREMENT이므로, 삽입 시에는 performance_id를 제외하고, 
        ON DUPLICATE KEY UPDATE 시에는 run_id, date UNIQUE KEY를 사용합니다.
        :param performance_data_list: [{'run_id': 1, 'date': '2023-01-02', ...}, ...]
        :return: 성공 시 True, 실패 시 False
        """
        conn = self.get_db_connection()
        if not conn: return False
        
        sql = """
        INSERT INTO backtest_performance
        (run_id, date, end_capital, daily_return, daily_profit_loss, cumulative_return, drawdown)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            end_capital=VALUES(end_capital),
            daily_return=VALUES(daily_return),
            daily_profit_loss=VALUES(daily_profit_loss),
            cumulative_return=VALUES(cumulative_return),
            drawdown=VALUES(drawdown)
        """
        data = []
        for perf in performance_data_list:
            data.append((
                perf['run_id'],
                perf['date'],
                perf['end_capital'],
                perf.get('daily_return'),
                perf.get('daily_profit_loss'),
                perf.get('cumulative_return'),
                perf.get('drawdown')
            ))
        
        try:
            cursor = self.execute_sql(sql, data) # executemany를 위해 리스트 전달
            if cursor:
                logger.info(f"{len(performance_data_list)}개의 백테스트 성능 지표를 저장/업데이트했습니다.")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"백테스트 성능 지표 저장/업데이트 오류: {e}", exc_info=True)
            return False

    def fetch_backtest_performance(self, run_id: int, start_date: date = None, end_date: date = None):
        """
        DB에서 백테스트 일별/기간별 성능 지표를 조회합니다.
        :param run_id: 조회할 백테스트 실행 ID (필수)
        :param start_date: 성능 기록 시작 날짜 (선택)
        :param end_date: 성능 기록 종료 날짜 (선택)
        :return: Pandas DataFrame
        """
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()
        
        sql = """
        SELECT performance_id, run_id, date, end_capital, daily_return, daily_profit_loss, 
               cumulative_return, drawdown
        FROM backtest_performance
        WHERE run_id = %s
        """
        params = [run_id]
        if start_date:
            sql += " AND date >= %s"
            params.append(start_date)
        if end_date:
            sql += " AND date <= %s"
            params.append(end_date)
        sql += " ORDER BY date ASC"

        try:
            cursor = self.execute_sql(sql, tuple(params))
            if cursor:
                result = cursor.fetchall()
                return pd.DataFrame(result)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"백테스트 성능 지표 조회 오류 (run_id: {run_id}): {e}", exc_info=True)
            return pd.DataFrame()
    # ----------------------------------------------------------------------------
    # 유니버스 관리 테이블
    # ----------------------------------------------------------------------------
    def save_daily_universe(self, daily_universe_data_list: list, target_date: date):
        """
        일별 유니버스 종목 목록을 DB의 daily_universe 테이블에 저장합니다.
        특정 날짜의 데이터가 이미 존재하면 기존 데이터를 삭제하고 새로 삽입합니다.
        :param daily_universe_data_list: [{'date': 'YYYY-MM-DD', 'stock_code': '005930', ...}, ...]
        :param target_date: 데이터를 저장할 날짜 (YYYY-MM-DD), 이 날짜의 기존 데이터는 삭제됩니다.
        :return: 성공 시 True, 실패 시 False
        """
        conn = self.get_db_connection()
        if not conn: return False

        try:
            # 1. 해당 날짜의 기존 데이터 삭제
            delete_sql = "DELETE FROM daily_universe WHERE date = %s"
            cursor = self.execute_sql(delete_sql, (target_date,)) # execute_sql 호출 후 cursor 반환받음
            logger.info(f"날짜 {target_date}의 기존 daily_universe 데이터 {cursor.rowcount if cursor else 0}개 삭제 완료.")

            if not daily_universe_data_list:
                logger.info(f"날짜 {target_date}에 저장할 daily_universe 데이터가 없습니다.")
                return True

            # 2. 새로운 데이터 삽입
            insert_sql = """
            INSERT INTO daily_universe
            (date, stock_code, stock_name, stock_score, 
             price_trend_score, trading_volume_score, volatility_score, theme_mention_score,
             theme_id, theme)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            data = []
            for item in daily_universe_data_list:
                data.append((
                    item['date'],
                    item['stock_code'],
                    item['stock_name'],
                    item['stock_score'],
                    item.get('price_trend_score'),
                    item.get('trading_volume_score'),
                    item.get('volatility_score'),
                    item.get('theme_mention_score'),
                    item.get('theme_id'),
                    item.get('theme')
                ))
            
            cursor = self.execute_sql(insert_sql, data) # executemany를 위해 리스트 전달
            if cursor:
                logger.info(f"{len(daily_universe_data_list)}개의 daily_universe 데이터를 저장했습니다.")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"daily_universe 데이터 저장 오류: {e}", exc_info=True)
            if conn:
                conn.rollback() # 오류 발생 시 롤백
            return False

    def fetch_daily_universe(self, target_date: date = None, stock_code: str = None) -> pd.DataFrame:
        """
        DB에서 daily_universe 데이터를 조회합니다.
        :param target_date: 조회할 날짜 (선택)
        :param stock_code: 조회할 종목코드 (선택)
        :return: Pandas DataFrame
        """
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()
        
        sql = """
        SELECT date, stock_code, stock_name, stock_score, 
               price_trend_score, trading_volume_score, volatility_score, theme_mention_score,
               theme_id, theme
        FROM daily_universe
        WHERE 1=1
        """
        params = []
        if target_date:
            sql += " AND date = %s"
            params.append(target_date)
        if stock_code:
            sql += " AND stock_code = %s"
            params.append(stock_code)
        
        sql += " ORDER BY date DESC, stock_score DESC"

        try:
            cursor = self.execute_sql(sql, tuple(params))
            if cursor:
                result = cursor.fetchall()
                # SELECT 문과 순서 일치하도록 컬럼 이름 직접 지정
                columns = [
                    'date', 'stock_code', 'stock_name', 'stock_score',
                    'price_trend_score', 'trading_volume_score', 'volatility_score', 'theme_mention_score',
                    'theme_id', 'theme'
                ]
                return pd.DataFrame(result, columns=columns)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"daily_universe 데이터 조회 오류 (date: {target_date}, stock_code: {stock_code}): {e}", exc_info=True)
            return pd.DataFrame()
        

    def fetch_daily_theme_stock(self, start_date: date, end_date: date) -> List[Tuple[str, str]]:
        """
        daily_universe 테이블에서 특정 기간에 해당하는 고유한 종목 코드를 가져옵니다.
        daily_universe 테이블이 daily_theme 테이블의 역할을 대체하므로 이 테이블에서 가져옵니다.
        """
        conn = self.get_db_connection()
        if not conn: return []
        
        sql = """
            SELECT DISTINCT
                stock_code,
                stock_name
            FROM (
                SELECT 
                    *,
                    -- 선택된 날짜 내에서 각 테마(theme_id)별로 stock_score가 높은 순서대로 순위를 매깁니다.
                    ROW_NUMBER() OVER (PARTITION BY theme_id ORDER BY stock_score DESC) as rn
                FROM
                    daily_universe
                WHERE
                    date BETWEEN %s AND %s
            ) AS ranked_daily_universe
            WHERE
                -- 각 테마별로 상위 3개 종목만 선택합니다.
                rn <= 3
        """
        params = (start_date, end_date)
        
        stocks = []
        try:
            cursor = self.execute_sql(sql, params)
            if cursor:
                results = cursor.fetchall()
                stocks = [(row['stock_code'], row['stock_name']) for row in results] 
                cursor.close()
        except Exception as e:
            logger.error(f"daily_universe에서 종목 코드를 가져오는 중 오류 발생 (기간: {start_date} ~ {end_date}): {e}", exc_info=True)
        return stocks
    

    # ----------------------------------------------------------------------------
    # 자동매매 관리 테이블
    # ----------------------------------------------------------------------------
    def create_trading_tables(self):
        return self.execute_sql_file('create_trading_tables')

    def drop_trading_tables(self):
        return self.execute_sql_file('drop_trading_tables')
    
    def save_trading_log(self, log_data: Dict[str, Any]) -> bool:
        """
        하나의 매매 로그를 trading_log 테이블에 저장합니다.
        log_data 딕셔너리는 trading_log 테이블의 컬럼과 매핑됩니다.
        """
        sql = """
            INSERT INTO trading_log (
                order_id, original_order_id, stock_code, stock_name, trading_date, trading_time,
                order_type, order_price, order_quantity, filled_price, filled_quantity,
                unfilled_quantity, order_status, commission, tax, net_amount, credit_type
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """
        params = (
            log_data.get('order_id'),
            log_data.get('original_order_id'),
            log_data.get('stock_code'),
            log_data.get('stock_name'),
            log_data.get('trading_date'),
            log_data.get('trading_time'),
            log_data.get('order_type'),
            log_data.get('order_price'),
            log_data.get('order_quantity'),
            log_data.get('filled_price'),
            log_data.get('filled_quantity'),
            log_data.get('unfilled_quantity'),
            log_data.get('order_status'),
            log_data.get('commission'),
            log_data.get('tax'),
            log_data.get('net_amount'),
            log_data.get('credit_type')
        )
        cursor = self.execute_sql(sql, params)
        if cursor:
            logger.info(f"거래 로그 저장 성공: {log_data.get('order_id')}")
            cursor.close()
            return True
        return False

    def fetch_trading_logs(self, start_date: date, end_date: date, stock_code: str = None) -> pd.DataFrame:
        """
        특정 기간의 매매 로그를 조회합니다.
        """
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()
        
        sql = """
            SELECT * FROM trading_log
            WHERE trading_date BETWEEN %s AND %s
        """
        params = (start_date, end_date)
        if stock_code:
            sql += " AND stock_code = %s"
            params = (start_date, end_date, stock_code)
        sql += " ORDER BY trading_date, trading_time"

        cursor = self.execute_sql(sql, params)
        if cursor:
            df = pd.DataFrame(cursor.fetchall())
            cursor.close()
            if not df.empty:
                df['trading_date'] = pd.to_datetime(df['trading_date'])
                # trading_time이 datetime.time 객체인 경우 datetime으로 조합
                df['trading_datetime'] = df.apply(
                    lambda row: datetime.combine(row['trading_date'].date(), row['trading_time']) if isinstance(row['trading_time'], time) else row['trading_date'],
                    axis=1
                )
                df.set_index('trading_datetime', inplace=True)
                logger.debug(f"거래 로그 {len(df)}건 조회 완료 (기간: {start_date} ~ {end_date}, 종목: {stock_code or '전체'})")
            else:
                logger.debug(f"조회된 거래 로그가 없습니다 (기간: {start_date} ~ {end_date}, 종목: {stock_code or '전체'})")
            return df
        return pd.DataFrame()

    def save_daily_portfolio(self, portfolio_data: Dict[str, Any]) -> bool:
        """
        일별 포트폴리오 스냅샷을 daily_portfolio 테이블에 저장합니다.
        record_date가 이미 존재하면 업데이트합니다.
        """
        sql = """
            INSERT INTO daily_portfolio (
                record_date, total_capital, cash_balance, total_asset_value,
                daily_profit_loss, daily_return_rate, cumulative_profit_loss, cumulative_return_rate, max_drawdown
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                total_capital = VALUES(total_capital),
                cash_balance = VALUES(cash_balance),
                total_asset_value = VALUES(total_asset_value),
                daily_profit_loss = VALUES(daily_profit_loss),
                daily_return_rate = VALUES(daily_return_rate),
                cumulative_profit_loss = VALUES(cumulative_profit_loss),
                cumulative_return_rate = VALUES(cumulative_return_rate),
                max_drawdown = VALUES(max_drawdown),
                updated_at = CURRENT_TIMESTAMP
        """
        params = (
            portfolio_data.get('record_date'),
            portfolio_data.get('total_capital'),
            portfolio_data.get('cash_balance'),
            portfolio_data.get('total_asset_value'),
            portfolio_data.get('daily_profit_loss'),
            portfolio_data.get('daily_return_rate'),
            portfolio_data.get('cumulative_profit_loss'),
            portfolio_data.get('cumulative_return_rate'),
            portfolio_data.get('max_drawdown')
        )
        cursor = self.execute_sql(sql, params)
        if cursor:
            logger.info(f"일별 포트폴리오 저장/업데이트 성공: {portfolio_data.get('record_date')}")
            cursor.close()
            return True
        return False

    def fetch_daily_portfolio(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        특정 기간의 일별 포트폴리오 스냅샷을 조회합니다.
        """
        sql = """
            SELECT * FROM daily_portfolio
            WHERE record_date BETWEEN %s AND %s
            ORDER BY record_date
        """
        params = (start_date, end_date)
        cursor = self.execute_sql(sql, params)
        if cursor:
            df = pd.DataFrame(cursor.fetchall())
            cursor.close()
            if not df.empty:
                df['record_date'] = pd.to_datetime(df['record_date']).dt.normalize()
                df.set_index('record_date', inplace=True)
                logger.debug(f"일별 포트폴리오 {len(df)}건 조회 완료 (기간: {start_date} ~ {end_date})")
            else:
                logger.debug(f"조회된 일별 포트폴리오가 없습니다 (기간: {start_date} ~ {end_date})")
            return df
        return pd.DataFrame()

    def fetch_latest_daily_portfolio(self) -> Optional[Dict[str, Any]]:
        """가장 최신 일별 포트폴리오 스냅샷을 조회합니다."""
        sql = """
            SELECT * FROM daily_portfolio
            ORDER BY record_date DESC
            LIMIT 1
        """
        cursor = self.execute_sql(sql)
        if cursor:
            result = cursor.fetchone()
            cursor.close()
            if result:
                logger.debug(f"최신 일별 포트폴리오 조회 완료: {result['record_date']}")
            else:
                logger.debug("최신 일별 포트폴리오가 없습니다.")
            return result
        return None

    def save_current_position(self, position_data: Dict[str, Any]) -> bool:
        """
        현재 보유 종목 정보를 current_positions 테이블에 저장/업데이트합니다.
        stock_code가 이미 존재하면 업데이트합니다.
        """
        sql = """
            INSERT INTO current_positions (
                stock_code, stock_name, quantity, sell_avail_qty, avg_price,
                eval_profit_loss, eval_return_rate, entry_date
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s 
            )
            ON DUPLICATE KEY UPDATE
                stock_name = VALUES(stock_name),
                quantity = VALUES(quantity),
                sell_avail_qty = VALUES(sell_avail_qty),
                avg_price = VALUES(avg_price),
                eval_profit_loss = VALUES(eval_profit_loss),
                eval_return_rate = VALUES(eval_return_rate),
                entry_date = VALUES(entry_date),
                last_update = CURRENT_TIMESTAMP
        """
        params = (
            position_data.get('stock_code'),
            position_data.get('stock_name'),
            position_data.get('quantity'),
            position_data.get('sell_avail_qty'),
            position_data.get('avg_price'),
            position_data.get('eval_profit_loss'),
            position_data.get('eval_return_rate'),
            position_data.get('entry_date')
        )
        cursor = self.execute_sql(sql, params)
        if cursor:
            logger.info(f"현재 보유 종목 저장/업데이트 성공: {position_data.get('stock_code')}")
            cursor.close()
            return True
        return False

    def delete_current_position(self, stock_code: str) -> bool:
        """
        현재 보유 종목 정보를 current_positions 테이블에서 삭제합니다.
        """
        sql = "DELETE FROM current_positions WHERE stock_code = %s"
        params = (stock_code,)
        cursor = self.execute_sql(sql, params)
        if cursor and cursor.rowcount > 0:
            logger.info(f"현재 보유 종목 삭제 성공: {stock_code}")
            cursor.close()
            return True
        elif cursor:
            logger.info(f"삭제할 현재 보유 종목이 없습니다: {stock_code}")
            cursor.close()
        return False

    def fetch_current_positions(self) -> List[Dict[str, Any]]:
        """
        현재 보유 중인 모든 종목 정보를 current_positions 테이블에서 조회합니다.
        """
        sql = "SELECT * FROM current_positions"
        cursor = self.execute_sql(sql)
        if cursor:
            results = cursor.fetchall()
            cursor.close()
            if results:
                logger.debug(f"현재 보유 종목 {len(results)}건 조회 완료.")
                return list(results)
            else:
                logger.debug("현재 보유 중인 종목이 없습니다.")
            return results
        return []

    def save_daily_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        하나의 매매 신호를 daily_signals 테이블에 저장/업데이트합니다.
        (signal_date, stock_code, signal_type)이 동일하면 업데이트합니다.
        """
        sql = """
            INSERT INTO daily_signals (
                signal_date, stock_code, stock_name, signal_type,
                strategy_name, target_price, target_quantity, is_executed, executed_order_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                stock_name = VALUES(stock_name),
                signal_type = VALUES(signal_type),
                strategy_name = VALUES(strategy_name),
                target_price = VALUES(target_price),
                target_quantity = VALUES(target_quantity),
                is_executed = VALUES(is_executed),
                executed_order_id = VALUES(executed_order_id),
                updated_at = CURRENT_TIMESTAMP
        """
        params = (
            signal_data.get('signal_date'),
            signal_data.get('stock_code'),
            signal_data.get('stock_name'),
            signal_data.get('signal_type'),
            signal_data.get('strategy_name'),
            signal_data.get('target_price'),
            signal_data.get('target_quantity'),
            signal_data.get('is_executed', False),
            signal_data.get('executed_order_id')
        )
        cursor = self.execute_sql(sql, params)
        if cursor:
            logger.info(f"매매 신호 저장/업데이트 성공: {signal_data.get('stock_code')} - {signal_data.get('signal_type')}")
            cursor.close()
            return True
        return False

    def fetch_daily_signals(self, signal_date: date, is_executed: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        특정 날짜의 매매 신호를 조회합니다. is_executed 필터링 가능.
        """
        sql = "SELECT * FROM daily_signals WHERE signal_date = %s"
        params = [signal_date]
        if is_executed is not None:
            sql += " AND is_executed = %s"
            params.append(1 if is_executed else 0)
        sql += " ORDER BY stock_code, signal_type"

        cursor = self.execute_sql(sql, tuple(params))
        if cursor:
            results = cursor.fetchall()
            cursor.close()
            if results:
                logger.debug(f"매매 신호 {len(results)}건 조회 완료 (날짜: {signal_date}, 실행 여부: {is_executed})")
            else:
                logger.debug(f"조회된 매매 신호가 없습니다 (날짜: {signal_date}, 실행 여부: {is_executed})")
            return results
        return []

    def update_daily_signal_status(self, signal_id: int, is_executed: bool, executed_order_id: Optional[str] = None) -> bool:
        """
        특정 매매 신호의 실행 상태를 업데이트합니다.
        """
        sql = """
            UPDATE daily_signals
            SET is_executed = %s, executed_order_id = %s, updated_at = CURRENT_TIMESTAMP
            WHERE signal_id = %s
        """
        params = (1 if is_executed else 0, executed_order_id, signal_id)
        cursor = self.execute_sql(sql, params)
        if cursor and cursor.rowcount > 0:
            logger.info(f"매매 신호 ID {signal_id} 상태 업데이트 성공 (실행 여부: {is_executed})")
            cursor.close()
            return True
        elif cursor:
            logger.warning(f"매매 신호 ID {signal_id}를 찾을 수 없거나 업데이트할 내용이 없습니다.")
            cursor.close()
        return False

    def clear_daily_signals(self, signal_date: date) -> bool:
        """
        특정 날짜의 모든 매매 신호를 삭제합니다.
        """
        sql = "DELETE FROM daily_signals WHERE signal_date = %s"
        params = (signal_date,)
        cursor = self.execute_sql(sql, params)
        if cursor:
            logger.info(f"{signal_date} 날짜의 매매 신호 {cursor.rowcount}건 삭제 완료.")
            cursor.close()
            return True
        return False

    # ----------------------------------------------------------------------------
    # Feed 관련 테이블 메서드 (db_feed.py에서 성공적으로 테스트된 기능들)
    # ----------------------------------------------------------------------------
    
    def create_feed_tables(self):
        """Feed 관련 테이블들을 생성합니다."""
        return self.execute_sql_file('create_feed_tables')

    def drop_feed_tables(self):
        """Feed 관련 테이블들을 삭제합니다."""
        return self.execute_sql_file('drop_feed_tables')
    
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
        cursor = self.execute_sql(sql, data)
        if cursor:
            logger.info(f"분봉 데이터 {len(ohlcv_data_list)}건 저장/업데이트 완료.")
            cursor.close()
            return True
        return False

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
            cursor.close()
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
            total_amount=VALUES(total_amount)
        """
        data = [(d['market_type'], d['date'], d['time'], d['total_amount']) for d in market_volume_list]
        cursor = self.execute_sql(sql, data)
        if cursor:
            logger.info(f"시장별 거래대금 데이터 {len(market_volume_list)}건 저장/업데이트 완료.")
            cursor.close()
            return True
        return False

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
            cursor.close()
            if not df.empty:
                # 'date'와 'time' 컬럼을 이용하여 'datetime' 컬럼 생성
                # time 컬럼이 Timedelta로 변환될 수 있으므로 안전하게 처리
                def combine_datetime(row):
                    try:
                        if hasattr(row['time'], 'time'):
                            return datetime.combine(row['date'], row['time'].time())
                        elif hasattr(row['time'], 'total_seconds'):
                            # Timedelta인 경우
                            seconds = int(row['time'].total_seconds())
                            hours = seconds // 3600
                            minutes = (seconds % 3600) // 60
                            seconds = seconds % 60
                            time_obj = time(hours, minutes, seconds)
                            return datetime.combine(row['date'], time_obj)
                        else:
                            return datetime.combine(row['date'], row['time'])
                    except:
                        return None
                
                df['datetime'] = df.apply(combine_datetime, axis=1)
                
                # 필요에 따라 다른 데이터 타입 변환 (예: 거래량/거래대금 float 변환)
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
        cursor = self.execute_sql(sql, data)
        if cursor:
            logger.info(f"뉴스 데이터 {len(news_list)}건 저장 완료.")
            cursor.close()
            return True
        return False

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

        cursor = self.execute_sql(sql, tuple(params))
        if cursor:
            df = pd.DataFrame(cursor.fetchall())
            cursor.close()
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
            net_gov_local=VALUES(net_gov_local), net_other_corp=VALUES(net_other_corp)
        """
        data = []
        for t in trends_list:
            data.append((
                t['stock_code'], t['date'], t['time'], t.get('current_price'), t.get('volume_total'),
                t.get('net_foreign'), t.get('net_institutional'), t.get('net_insurance_etc'), t.get('net_trust'),
                t.get('net_bank'), t.get('net_pension'), t.get('net_gov_local'), t.get('net_other_corp'), t['data_type']
            ))
        cursor = self.execute_sql(sql, data)
        if cursor:
            logger.info(f"투자자 매매 동향 데이터 {len(trends_list)}건 저장/업데이트 완료.")
            cursor.close()
            return True
        return False

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
            cursor.close()
            if not df.empty:
                # 'date'와 'time' 컬럼을 이용하여 'datetime' 컬럼 생성
                # time 컬럼이 Timedelta로 변환될 수 있으므로 안전하게 처리
                def combine_datetime(row):
                    try:
                        if hasattr(row['time'], 'time'):
                            return datetime.combine(row['date'], row['time'].time())
                        elif hasattr(row['time'], 'total_seconds'):
                            # Timedelta인 경우
                            seconds = int(row['time'].total_seconds())
                            hours = seconds // 3600
                            minutes = (seconds % 3600) // 60
                            seconds = seconds % 60
                            time_obj = time(hours, minutes, seconds)
                            return datetime.combine(row['date'], time_obj)
                        else:
                            return datetime.combine(row['date'], row['time'])
                    except:
                        return None
                
                df['datetime'] = df.apply(combine_datetime, axis=1)
                
                # 수치형 데이터 변환
                for col in ['current_price', 'volume_total', 'net_foreign', 'net_institutional', 'net_insurance_etc', 'net_trust', 'net_bank', 'net_pension', 'net_gov_local', 'net_other_corp']:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, (Decimal, int, float)) else x)
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
        cursor = self.execute_sql(sql, data)
        if cursor:
            logger.info(f"뉴스 요약 데이터 {len(summaries_list)}건 저장 완료.")
            cursor.close()
            return True
        return False

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

        cursor = self.execute_sql(sql, tuple(params))
        if cursor:
            df = pd.DataFrame(cursor.fetchall())
            cursor.close()
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
            relevance_score=VALUES(relevance_score), mention_count=VALUES(mention_count)
        """
        data = [(t['theme_name'], t['stock_code'], t['analysis_date'], t.get('relevance_score'), t.get('mention_count')) for t in thematic_list]
        cursor = self.execute_sql(sql, data)
        if cursor:
            logger.info(f"테마별 종목 데이터 {len(thematic_list)}건 저장/업데이트 완료.")
            cursor.close()
            return True
        return False

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

        cursor = self.execute_sql(sql, tuple(params))
        if cursor:
            df = pd.DataFrame(cursor.fetchall())
            cursor.close()
            if not df.empty:
                df['analysis_date'] = pd.to_datetime(df['analysis_date']).dt.date
                df['relevance_score'] = df['relevance_score'].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
            return df
        return pd.DataFrame() 