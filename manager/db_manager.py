# manager/db_manager.py

import pymysql
import logging
import pandas as pd
from datetime import datetime, date, timedelta
import os
import sys
import json # JSON 직렬화를 위해 추가
import re
from decimal import Decimal # Decimal 타입 처리용
# --- 로거 설정 (스크립트 최상단에서 설정하여 항상 보이도록 함) ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # 테스트 시 DEBUG로 설정하여 모든 로그 출력

# 콘솔 핸들러 추가 (이미 핸들러가 등록되어 있지 않다면)
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
# -----------------------------------------------------------
# sys.path에 프로젝트 루트 추가 (settings.py 임포트를 위함)
# 이 스크립트의 위치는 creon_trading/db/db_manager.py 이므로,
# project_root는 creon_trading 디렉토리가 되어야 합니다.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
sys.path.insert(0, project_root)
logger.debug(f"Project root added to sys.path: {project_root}")

from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
logger.info("config/settings.py에서 DB 설정을 성공적으로 로드했습니다.")
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
        # SQLAlchemy Engine도 함께 닫아주는 것이 좋습니다.
        if self._engine:
            self._engine.dispose()
            logger.info("SQLAlchemy Engine이 닫혔습니다.")

    def _get_db_engine(self):
        """SQLAlchemy Engine을 생성하거나 반환합니다."""
        if self._engine is None:
            try:
                db_url = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
                self._engine = create_engine(db_url, echo=False) # echo=True는 SQL 쿼리 로깅
                logger.info(f"SQLAlchemy Engine 생성 완료: {db_url.split('@')[1]}")
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
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
            logger.info(f"DataFrame 데이터 {len(df)}행을 테이블 '{table_name}'에 '{option}' 모드로 성공적으로 삽입했습니다.")
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
                logger.info(f"{len(stock_info_list)}개의 종목 정보를 저장/업데이트했습니다.")
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
        SELECT date 
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

    def save_market_calendar(self, df: pd.DataFrame, option: str = "append") -> bool:
        """
        Pandas DataFrame을 market_calendar 테이블에 저장합니다.
        :param df: 저장할 시장 캘린더 데이터 (컬럼: 'date', 'is_holiday', 'description')
        :param option: 테이블 존재 시 처리 방식 ('append', 'replace', 'fail')
        :return: 성공 여부 (True/False)
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("save_market_calendar: 입력 데이터가 pandas DataFrame이 아닙니다.")
            return False
        
        if df.empty:
            logger.warning("save_market_calendar: 저장할 데이터가 없습니다. 빈 DataFrame은 처리하지 않습니다.")
            return True

        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date']).dt.date

        if 'description' not in df.columns:
            df['description'] = ''
        
        if 'is_holiday' not in df.columns:
            df['is_holiday'] = False

        columns_to_save = ['date', 'is_holiday', 'description']
        df_to_save = df[columns_to_save]

        return self.insert_df_to_db('market_calendar', df_to_save, option=option, is_index=False)
            
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
    # 자동매매 관리 테이블
    # ----------------------------------------------------------------------------
    def create_trade_tables(self):
        return self.execute_sql_file('create_trade_tables')

    def drop_trade_tables(self):
        return self.execute_sql_file('drop_trade_tables')




    # ----------------------------------------------------------------------------
    # 아이디어 차원
    # ----------------------------------------------------------------------------
    # # --- insert_df_to_db 함수 수정 (upsert 옵션 추가) ---
    # def insert_df_to_db(self, table_name: str, df: pd.DataFrame, option: str = "append", is_index: bool = False):
    #     """
    #     DataFrame을 MariaDB에 삽입하는 메서드 (직접 SQL 쿼리 사용).
    #     'append', 'upsert' 모드를 지원합니다. 'replace' 또는 'fail' 모드는 직접 지원하지 않습니다.
        
    #     :param table_name: 데이터를 삽입할 테이블 이름
    #     :param df: 삽입할 데이터가 담긴 Pandas DataFrame
    #     :param option: 테이블 존재 시 처리 방식 ('append', 'upsert')
    #                    'append': 중복 시 오류 발생 (PRIMARY KEY/UNIQUE 위반 시)
    #                    'upsert': 중복 시 기존 레코드를 업데이트
    #     :param is_index: DataFrame의 인덱스를 DB 컬럼으로 저장할지 여부 (인덱스 이름이 컬럼명으로 사용됨)
    #     """
    #     table_name = table_name.lower()
    #     if not isinstance(df, pd.DataFrame):
    #         logger.error("insert_df_to_db: 입력 데이터가 pandas DataFrame이 아닙니다.")
    #         return False

    #     if option not in ["append", "upsert"]:
    #         logger.error(f"insert_df_to_db: option must be 'append' or 'upsert', but got '{option}'.")
    #         return False

    #     if df.empty:
    #         logger.info(f"DataFrame이 비어있어 테이블 '{table_name}'에 삽입할 데이터가 없습니다.")
    #         return True # 데이터 없으면 성공으로 간주

    #     # 1. 컬럼 목록 및 플레이스홀더 동적 생성
    #     columns = df.columns.tolist()
    #     if is_index and df.index.name:
    #         index_column_name = df.index.name
    #         columns.insert(0, index_column_name) # 인덱스를 첫 번째 컬럼으로 추가
    #     elif is_index and not df.index.name:
    #         logger.warning("insert_df_to_db: is_index=True 이지만 DataFrame 인덱스 이름이 없습니다. 'index'로 기본 설정.")
    #         index_column_name = 'index' # 기본 인덱스 이름
    #         columns.insert(0, index_column_name)
    #     else:
    #         index_column_name = None # 인덱스를 컬럼으로 사용하지 않음
            
    #     placeholders = ', '.join(['%s'] * len(columns))
    #     column_names = ', '.join(columns)

    #     # 2. SQL 쿼리 기본 부분 생성
    #     sql = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
        
    #     # 3. 'upsert' 옵션일 경우 ON DUPLICATE KEY UPDATE 절 추가
    #     if option == "upsert":
    #         update_clauses = []
    #         # 인덱스 컬럼을 제외한 모든 컬럼을 업데이트 대상으로 함
    #         # 주의: 여기서 인덱스 컬럼이 PRIMARY KEY 또는 UNIQUE KEY라고 가정합니다.
    #         # 만약 다른 컬럼들도 함께 키를 구성한다면, 해당 로직을 더 정교하게 만들어야 합니다.
    #         # 가장 간단한 경우, INSERT 대상 컬럼에서 키 컬럼을 제외한 나머지를 업데이트합니다.
            
    #         # 여기서 update_columns는 실제로 INSERT 되는 모든 컬럼 (인덱스 포함) 중
    #         # PRIMARY/UNIQUE KEY를 구성하는 컬럼을 제외한 컬럼들이어야 합니다.
    #         # 일반화된 함수에서는 어떤 컬럼이 KEY인지 알 수 없으므로, 모든 일반 컬럼을 대상으로 합니다.
            
    #         # DataFrame의 실제 컬럼명 (인덱스 컬럼은 제외)
    #         data_columns_for_update = df.columns.tolist()

    #         for col in data_columns_for_update:
    #             update_clauses.append(f"{col}=VALUES({col})")
            
    #         if update_clauses:
    #             sql += " ON DUPLICATE KEY UPDATE " + ", ".join(update_clauses)
    #         else:
    #             logger.warning(f"테이블 '{table_name}'에 업데이트할 컬럼이 없습니다. UPSERT가 제대로 작동하지 않을 수 있습니다.")

    #     # 4. DataFrame을 SQL에 적합한 튜플 리스트로 변환
    #     data_to_insert = []
    #     for index_val, row in df.iterrows():
    #         row_data = []
    #         if is_index:
    #             # 인덱스 값을 SQL에 맞게 변환 (DatetimeIndex -> datetime.datetime/date, 일반 인덱스 -> 그대로)
    #             if isinstance(index_val, pd.Timestamp):
    #                 if index_val.hour == 0 and index_val.minute == 0 and index_val.second == 0 and \
    #                    index_val.microsecond == 0: # 시간 정보가 없는 순수 날짜 Timestamp
    #                     row_data.append(index_val.date()) 
    #                 else: # 시간 정보가 있는 Timestamp
    #                     row_data.append(index_val.to_pydatetime()) 
    #             else:
    #                 row_data.append(index_val) # 다른 인덱스 타입은 그대로 추가

    #         for col in df.columns:
    #             val = row[col]
    #             # Pandas 기본 타입(numpy 타입 포함)을 Python 기본 타입으로 변환
    #             if pd.isna(val): # NaN 값은 None으로 변환
    #                 row_data.append(None)
    #             elif isinstance(val, (pd.Timestamp, datetime, date)):
    #                 # datetime/date 객체는 그대로
    #                 # Timestamp는 위 인덱스 처리에서 다루었으므로 여기서는 datetime.datetime/date를 주로 다룸
    #                 row_data.append(val)
    #             elif pd.api.types.is_integer_dtype(df[col]):
    #                 row_data.append(int(val)) # 정수형은 int로 변환
    #             elif pd.api.types.is_float_dtype(df[col]):
    #                 row_data.append(float(val)) # 실수형은 float로 변환
    #             else:
    #                 row_data.append(val) # 그 외는 그대로 (문자열 등)
            
    #         data_to_insert.append(tuple(row_data))

    #     try:
    #         cursor = self.execute_sql(sql, data_to_insert)
    #         if cursor:
    #             if option == "upsert":
    #                 logger.info(f"DataFrame 데이터 {len(df)}행을 테이블 '{table_name}'에 UPSERT 완료.")
    #             else: # append
    #                 logger.info(f"DataFrame 데이터 {len(df)}행을 테이블 '{table_name}'에 성공적으로 삽입했습니다.")
    #             return True
    #         else:
    #             return False
    #     except Exception as e:
    #         logger.error(f"DataFrame 데이터를 테이블 '{table_name}'에 삽입 중 오류 발생: {e}", exc_info=True)
    #         return False
# --- 테스트 코드 ---
if __name__ == "__main__":
    # 로깅 레벨을 DEBUG로 설정하여 SQL 실행 로그 확인
    logger.setLevel(logging.DEBUG) 
    
    # db_manager = DBManager()

    # # --- 1. 종목 및 주가 관련 테이블 테스트 ---
    # logger.info("--- 종목 및 주가 관련 테이블 테스트 시작 ---")
    
    # # 1.1 테이블 삭제 및 재생성 (stock_info, daily_price, minute_price, market_calendar)
    # logger.info("기존 stock_tables 삭제 중...")
    # if not db_manager.drop_stock_tables(): # <-- 반환값 확인
    #     logger.error("stock_tables 삭제 실패! 존재하지 않을 수 있음")
        
    # logger.info("stock_tables 생성 중...")
    # if not db_manager.create_stock_tables(): # <-- 반환값 확인
    #     logger.error("stock_tables 생성 실패! 테스트를 중단합니다. SQL 파일 경로 또는 SQL 문법을 확인하세요.")
    #     sys.exit(1) # 테스트 중단
    # logger.info("stock_tables 생성 완료.")

    # # 1.2 save_stock_info 테스트
    # logger.info("save_stock_info 테스트 시작...")
    # test_stock_info_data = [
    #     {
    #         'stock_code': 'A005930', 'stock_name': '삼성전자', 'market_type': 'KOSPI', 'sector': '반도체 및 반도체 장비',
    #         'per': 15.20, 'pbr': 1.60, 'eps': 5000.00, 'roe': 10.50, 'debt_ratio': 25.30,
    #         'sales': 280000000, 'operating_profit': 4000000, 'net_profit': 3000000,
    #         'recent_financial_date': date(2024, 3, 31)
    #     },
    #     {
    #         'stock_code': 'A000660', 'stock_name': 'SK하이닉스', 'market_type': 'KOSPI', 'sector': '반도체 및 반도체 장비',
    #         'per': 20.10, 'pbr': 2.50, 'eps': 3500.00, 'roe': 12.00, 'debt_ratio': 35.00,
    #         'sales': 35000000, 'operating_profit': 500000, 'net_profit': 400000,
    #         'recent_financial_date': date(2024, 3, 31)
    #     },
    #     {
    #         'stock_code': 'A035420', 'stock_name': 'NAVER', 'market_type': 'KOSPI', 'sector': '소프트웨어',
    #         'per': 30.50, 'pbr': 3.00, 'eps': 2000.00, 'roe': 8.00, 'debt_ratio': 15.00,
    #         'sales': 900000, 'operating_profit': 150000, 'net_profit': 100000,
    #         'recent_financial_date': date(2024, 3, 31)
    #     }
    # ]
    # db_manager.save_stock_info(test_stock_info_data)
    # logger.info("save_stock_info 테스트 완료.")

    # # 1.3 fetch_stock_info 테스트
    # logger.info("fetch_stock_info 테스트 시작 (전체 조회)...")
    # fetched_stock_info = db_manager.fetch_stock_info()
    # logger.info(f"조회된 stock_info 데이터:\n{fetched_stock_info}")

    # logger.info("fetch_stock_info 테스트 시작 (특정 종목 조회)...")
    # fetched_specific_stock_info = db_manager.fetch_stock_info(stock_codes=['A005930'])
    # logger.info(f"조회된 특정 종목 stock_info 데이터 (A005930):\n{fetched_specific_stock_info}")
    # logger.info("fetch_stock_info 테스트 완료.")

    # # 1.4 get_all_stock_codes 테스트
    # logger.info("get_all_stock_codes 테스트 시작...")
    # all_codes = db_manager.get_all_stock_codes()
    # logger.info(f"모든 종목 코드:\n{all_codes}")
    # logger.info("get_all_stock_codes 테스트 완료.")

    # # 1.5 fetch_stock_codes_by_criteria 테스트
    # logger.info("fetch_stock_codes_by_criteria 테스트 시작 (EPS 3000 이상)...")
    # filtered_codes = db_manager.fetch_stock_codes_by_criteria(eps_min=3000)
    # logger.info(f"EPS 3000 이상 종목 코드:\n{filtered_codes}")

    # logger.info("fetch_stock_codes_by_criteria 테스트 시작 (PBR 2.0 이하, ROE 10.0 이상)...")
    # filtered_codes_complex = db_manager.fetch_stock_codes_by_criteria(pbr_max=2.0, roe_min=10.0)
    # logger.info(f"PBR 2.0 이하, ROE 10.0 이상 종목 코드:\n{filtered_codes_complex}")
    # logger.info("fetch_stock_codes_by_criteria 테스트 완료.")

    # # 1.6 save_daily_price 테스트
    # logger.info("save_daily_price 테스트 시작...")
    # test_daily_price_data = [
    #     {
    #         'stock_code': 'A005930', 'date': date(2024, 6, 10), 'open': 78000.0, 'high': 78500.0,
    #         'low': 77500.0, 'close': 78200.0, 'volume': 10000000, 'trading_value': 782000000000, 'change_rate': 0.25
    #     },
    #     {
    #         'stock_code': 'A005930', 'date': date(2024, 6, 11), 'open': 78300.0, 'high': 79000.0,
    #         'low': 78000.0, 'close': 78800.0, 'volume': 12000000, 'trading_value': 945600000000, 'change_rate': 0.77
    #     },
    #     {
    #         'stock_code': 'A000660', 'date': date(2024, 6, 10), 'open': 180000.0, 'high': 181000.0,
    #         'low': 179000.0, 'close': 180500.0, 'volume': 5000000, 'trading_value': 902500000000, 'change_rate': 0.50
    #     }
    # ]
    # db_manager.save_daily_price(test_daily_price_data)
    # logger.info("save_daily_price 테스트 완료.")

    # # # 1.7 fetch_daily_price 테스트
    # logger.info("fetch_daily_price 테스트 시작 (A005930, 전체 기간)...")
    # fetched_price_data = db_manager.fetch_daily_price(stock_code='A005930')
    # logger.info(f"조회된 daily_price 데이터 (A005930):\n{fetched_price_data}")

    # logger.info("fetch_daily_price 테스트 시작 (A005930, 특정 기간)...")
    # fetched_price_data_period = db_manager.fetch_daily_price(
    #     stock_code='A005930', 
    #     start_date=date(2024, 6, 10), 
    #     end_date=date(2024, 6, 10)
    # )
    # logger.info(f"조회된 daily_price 데이터 (A005930, 2024-06-10~2024-06-10):\n{fetched_price_data_period}")
    # logger.info("fetch_daily_price 테스트 완료.")

    # # # 1.8 get_latest_daily_price_date 테스트
    # logger.info("get_latest_daily_price_date 테스트 시작 (A005930)...")
    # latest_date_5930 = db_manager.get_latest_daily_price_date('A005930')
    # logger.info(f"A005930 최신 일봉 날짜: {latest_date_5930}")

    # logger.info("get_latest_daily_price_date 테스트 시작 (A000660)...")
    # latest_date_660 = db_manager.get_latest_daily_price_date('A000660')
    # logger.info(f"A000660 최신 일봉 날짜: {latest_date_660}")
    # logger.info("get_latest_daily_price_date 테스트 완료.")

    # # # 1.9 save_minute_price 테스트 (새로 추가)
    # logger.info("save_minute_price 테스트 시작...")
    # test_minute_price_data = [
    #     {
    #         'stock_code': 'A005930', 'datetime': datetime(2024, 6, 10, 9, 0, 0), 'open': 78200.0, 
    #         'high': 78250.0, 'low': 78150.0, 'close': 78220.0, 'volume': 100000,
    #         'trading_value': 7822000000, 'change_rate': 0.01
    #     },
    #     {
    #         'stock_code': 'A005930', 'datetime': datetime(2024, 6, 10, 9, 1, 0), 'open': 78220.0, 
    #         'high': 78300.0, 'low': 78200.0, 'close': 78280.0, 'volume': 80000,
    #         'trading_value': 6262400000, 'change_rate': 0.07
    #     },
    #     {
    #         'stock_code': 'A000660', 'datetime': datetime(2024, 6, 10, 9, 0, 0), 'open': 180500.0, 
    #         'high': 180600.0, 'low': 180400.0, 'close': 180550.0, 'volume': 30000,
    #         'trading_value': 5416500000, 'change_rate': 0.03
    #     }
    # ]
    # db_manager.save_minute_price(test_minute_price_data)
    # logger.info("save_minute_price 테스트 완료.")

    # # # 1.10 fetch_minute_price 테스트 (새로 추가)
    # logger.info("fetch_minute_price 테스트 시작 (A005930, 전체 기간)...")
    # fetched_minute_data = db_manager.fetch_minute_price(stock_code='A005930')
    # logger.info(f"조회된 minute_price 데이터 (A005930):\n{fetched_minute_data}")

    # logger.info("fetch_minute_price 테스트 시작 (A005930, 특정 기간)...")
    # fetched_minute_data_period = db_manager.fetch_minute_price(
    #     stock_code='A005930', 
    #     start_date=date(2024, 6, 10), 
    #     end_date=date(2024, 6, 10)
    # )
    # logger.info(f"조회된 minute_price 데이터 (A005930, 2024-06-10 전체):\n{fetched_minute_data_period}")

    # logger.info("fetch_minute_price 테스트 시작 (A005930, 특정 시각 범위)...")
    # fetched_minute_data_time_range = db_manager.fetch_minute_price(
    #     stock_code='A005930', 
    #     start_date=datetime(2024, 6, 10, 9, 0, 0), 
    #     end_date=datetime(2024, 6, 10, 9, 0, 0)
    # )
    # logger.info(f"조회된 minute_price 데이터 (A005930, 2024-06-10 09:00:00):\n{fetched_minute_data_time_range}")
    # logger.info("fetch_minute_price 테스트 완료.")

    # # # 1.11 save_market_calendar 테스트 (새로 추가)
    # logger.info("save_market_calendar 테스트 시작...")
    # # 2024년 6월 10일(월)은 영업일, 6월 11일(화)도 영업일, 6월 12일(수)은 휴일(테스트용), 6월 13일(목)은 영업일
    # test_calendar_data = [
    #     {'date': date(2025, 6, 1), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 2), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 3), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 4), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 5), 'is_holiday': True, 'description': '가상공휴일'},
    #     {'date': date(2025, 6, 6), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 7), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 8), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 9), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 10), 'is_holiday': True, 'description': '가상공휴일'},
    #     {'date': date(2025, 6, 11), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 12), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 13), 'is_holiday': True, 'description': '주말(토)'}
    # ]
    # db_manager.save_market_calendar(pd.DataFrame(test_calendar_data))
    # logger.info("save_market_calendar 테스트 완료.")

    # # # 1.12 fetch_market_calendar 테스트 (새로 추가)
    # logger.info("fetch_market_calendar 테스트 시작 (2024-06-10 ~ 2024-06-15)...")
    # fetched_calendar_data = db_manager.fetch_market_calendar(date(2024, 6, 10), date(2024, 6, 15))
    # logger.info(f"조회된 market_calendar 데이터:\n{fetched_calendar_data}")
    # logger.info("fetch_market_calendar 테스트 완료.")

    # # 1.13 get_all_trading_days 테스트 (새로 추가)
    # logger.info("get_all_trading_days 테스트 시작 (2024-06-10 ~ 2024-06-15)...")
    # trading_days = db_manager.get_all_trading_days(date(2024, 6, 10), date(2025, 6, 15))
    # logger.info(f"조회된 영업일 데이터 (list):\n{trading_days}")
    # logger.info("get_all_trading_days 테스트 완료.")


    # logger.info("--- 종목 및 주가 관련 테이블 테스트 완료 ---")

    # # --- 2. 백테스팅 관련 테이블 테스트 ---
    # logger.info("--- 백테스팅 관련 테이블 테스트 시작 ---")

    # # 2.1 테이블 삭제 및 재생성 (backtest_run, backtest_trade, backtest_performance)
    # logger.info("기존 backtest_tables 삭제 중...")
    # if not db_manager.drop_backtest_tables(): # <-- 반환값 확인
    #     logger.error("backtest_tables 삭제 실패! 테스트를 중단합니다. SQL 파일 경로 또는 권한을 확인하세요.")
    #     sys.exit(1) # 테스트 중단
    # logger.info("backtest_tables 삭제 완료.")

    # logger.info("backtest_tables 생성 중...")
    # if not db_manager.create_backtest_tables(): # <-- 반환값 확인
    #     logger.error("backtest_tables 생성 실패! 테스트를 중단합니다. SQL 파일 경로 또는 SQL 문법을 확인하세요.")
    #     sys.exit(1) # 테스트 중단
    # logger.info("backtest_tables 생성 완료.")

    # # 2.2 save_backtest_run 테스트
    # logger.info("save_backtest_run 테스트 시작...")
    # test_run_info = {
    #     'start_date': date(2023, 1, 1),
    #     'end_date': date(2023, 12, 31),
    #     'initial_capital': 10000000.00,
    #     'final_capital': 12000000.00,
    #     'total_profit_loss': 2000000.00,
    #     'cumulative_return': 0.20,
    #     'max_drawdown': 0.05,
    #     'strategy_daily': 'DualMomentumDaily',
    #     'strategy_minute': 'RSIMinute',
    #     'params_json_daily': {'lookback_period': 12, 'num_top_stocks': 5},
    #     'params_json_minute': {'rsi_period': 14, 'buy_signal': 30, 'sell_signal': 70}
    # }
    # # 새로운 run_id 생성 및 저장
    # run_id = db_manager.save_backtest_run(test_run_info)
    # if run_id:
    #     logger.info(f"새로운 backtest_run 저장 완료. run_id: {run_id}")
    # else:
    #     logger.error("backtest_run 저장 실패!")
    #     run_id = 999999 # 테스트를 위해 임의의 ID 할당 (실패 시에도 계속 진행 위함)

    # # 기존 run_id로 업데이트 테스트
    # logger.info(f"기존 backtest_run 업데이트 테스트 (run_id: {run_id})...")
    # update_run_info = test_run_info.copy()
    # update_run_info['run_id'] = run_id
    # update_run_info['final_capital'] = 12500000.00
    # update_run_info['total_profit_loss'] = 2500000.00
    # update_run_info['cumulative_return'] = 0.25
    # db_manager.save_backtest_run(update_run_info)
    # logger.info("save_backtest_run 테스트 완료.")

    # # 2.3 fetch_backtest_run 테스트
    # logger.info(f"fetch_backtest_run 테스트 시작 (run_id: {run_id})...")
    # fetched_run_data = db_manager.fetch_backtest_run(run_id=run_id)
    # logger.info(f"조회된 backtest_run 데이터:\n{fetched_run_data}")

    # logger.info("fetch_backtest_run 테스트 시작 (기간 조회)...")
    # fetched_run_data_period = db_manager.fetch_backtest_run(
    #     start_date=date(2023, 1, 1), 
    #     end_date=date(2023, 12, 31)
    # )
    # logger.info(f"조회된 backtest_run 데이터 (2023년):\n{fetched_run_data_period}")
    # logger.info("fetch_backtest_run 테스트 완료.")

    # # 2.4 save_backtest_trade 테스트
    # logger.info("save_backtest_trade 테스트 시작...")
    # test_trade_data = [
    #     {
    #         'run_id': run_id, 'stock_code': 'A005930', 'trade_type': 'BUY', 'trade_price': 70000.0,
    #         'trade_quantity': 10, 'trade_amount': 700000.0, 'trade_datetime': datetime(2023, 1, 5, 9, 30, 0),
    #         'commission': 500.0, 'tax': 0.0, 'realized_profit_loss': 0.0, 'entry_trade_id': None
    #     },
    #     {
    #         'run_id': run_id, 'stock_code': 'A000660', 'trade_type': 'BUY', 'trade_price': 100000.0,
    #         'trade_quantity': 5, 'trade_amount': 500000.0, 'trade_datetime': datetime(2023, 1, 5, 10, 0, 0),
    #         'commission': 300.0, 'tax': 0.0, 'realized_profit_loss': 0.0, 'entry_trade_id': None
    #     }
    # ]
    # db_manager.save_backtest_trade(test_trade_data)

    # # 매도 거래 추가 (entry_trade_id를 위해 첫 번째 매수 거래의 trade_id가 필요)
    # # 실제 시스템에서는 매수 거래 저장 후 trade_id를 받아와 사용해야 하지만, 여기서는 임의로 가정
    # # 또는 fetch_backtest_trade로 매수 거래를 조회하여 trade_id를 얻어야 합니다.
    # fetched_trades_for_entry_id = db_manager.fetch_backtest_trade(run_id=run_id, stock_code='A005930', start_datetime=datetime(2023,1,5,9,29,0), end_datetime=datetime(2023,1,5,9,31,0))
    # entry_trade_id = fetched_trades_for_entry_id['trade_id'].iloc[0] if not fetched_trades_for_entry_id.empty else None

    # test_trade_sell_data = [
    #     {
    #         'run_id': run_id, 'stock_code': 'A005930', 'trade_type': 'SELL', 'trade_price': 75000.0,
    #         'trade_quantity': 10, 'trade_amount': 750000.0, 'trade_datetime': datetime(2023, 1, 10, 15, 0, 0),
    #         'commission': 500.0, 'tax': 75.0, 'realized_profit_loss': 49925.0, # (75000-70000)*10 - 500 - 75
    #         'entry_trade_id': entry_trade_id 
    #     }
    # ]
    # db_manager.save_backtest_trade(test_trade_sell_data)
    # logger.info("save_backtest_trade 테스트 완료.")

    # # 2.5 fetch_backtest_trade 테스트
    # logger.info(f"fetch_backtest_trade 테스트 시작 (run_id: {run_id})...")
    # fetched_trade_data = db_manager.fetch_backtest_trade(run_id=run_id)
    # logger.info(f"조회된 backtest_trade 데이터:\n{fetched_trade_data}")

    # logger.info(f"fetch_backtest_trade 테스트 시작 (run_id: {run_id}, 특정 종목 A005930)...")
    # fetched_trade_data_stock = db_manager.fetch_backtest_trade(run_id=run_id, stock_code='A005930')
    # logger.info(f"조회된 backtest_trade 데이터 (A005930):\n{fetched_trade_data_stock}")
    # logger.info("fetch_backtest_trade 테스트 완료.")

    # # 2.6 save_backtest_performance 테스트
    # logger.info("save_backtest_performance 테스트 시작...")
    # test_performance_data = [
    #     {
    #         'run_id': run_id, 'date': date(2023, 1, 1), 'end_capital': 10000000.0,
    #         'daily_return': 0.0, 'daily_profit_loss': 0.0, 'cumulative_return': 0.0, 'drawdown': 0.0
    #     },
    #     {
    #         'run_id': run_id, 'date': date(2023, 1, 2), 'end_capital': 10050000.0,
    #         'daily_return': 0.005, 'daily_profit_loss': 50000.0, 'cumulative_return': 0.005, 'drawdown': 0.0
    #     },
    #     {
    #         'run_id': run_id, 'date': date(2023, 1, 3), 'end_capital': 9900000.0,
    #         'daily_return': -0.0149, 'daily_profit_loss': -150000.0, 'cumulative_return': -0.01, 'drawdown': 0.015
    #     }
    # ]
    # db_manager.save_backtest_performance(test_performance_data)
    # logger.info("save_backtest_performance 테스트 완료.")

    # # 2.7 fetch_backtest_performance 테스트
    # logger.info(f"fetch_backtest_performance 테스트 시작 (run_id: {run_id})...")
    # fetched_performance_data = db_manager.fetch_backtest_performance(run_id=run_id)
    # logger.info(f"조회된 backtest_performance 데이터:\n{fetched_performance_data}")

    # logger.info(f"fetch_backtest_performance 테스트 시작 (run_id: {run_id}, 특정 기간)...")
    # fetched_performance_data_period = db_manager.fetch_backtest_performance(
    #     run_id=run_id, 
    #     start_date=date(2023, 1, 1), 
    #     end_date=date(2023, 1, 2)
    # )
    # logger.info(f"조회된 backtest_performance 데이터 (2023-01-01~2023-01-02):\n{fetched_performance_data_period}")
    # logger.info("fetch_backtest_performance 테스트 완료.")

    # logger.info("--- 백테스팅 관련 테이블 테스트 완료 ---")

    # # --- 클린업 ---
    # logger.info("--- 모든 테이블 삭제 시작 ---")
    # logger.info("backtest_tables 삭제 중...")
    # #db_manager.drop_backtest_tables()
    # logger.info("stock_tables 삭제 중...")
    # #db_manager.drop_stock_tables()
    # logger.info("--- 모든 테이블 삭제 완료 ---")

    # db_manager.close()
    # logger.info("DB 연결 종료.")
    # logger.info("모든 테스트 완료.")