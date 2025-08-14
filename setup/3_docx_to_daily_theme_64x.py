# setup/3_docx_to_daily_theme_64x.py
import os
import re
import pandas as pd
from docx import Document # 64 비트에 pip install python-docx
import pymysql
import sys
import textwrap
import time
from datetime import datetime, date
from decimal import Decimal
import shutil
import json

# PyKoSpacing 모듈 로드 (띄어쓰기 교정용)
try:
    from pykospacing import Spacing
    PYKOSPACING_ENABLED = True
    spacing_tool = Spacing()
    print("PyKoSpacing 활성화.")
except ImportError:
    PYKOSPACING_ENABLED = False
    spacing_tool = None
    print("PyKoSpacing 비활성화. 설치 필요 (pip install pykospacing).")

# Konlpy Komoran 모듈 로드 (명사 추출용)
try:
    from konlpy.tag import Komoran
    KONLPY_KOMORAN_ENABLED = True
    komoran_tool = Komoran()
    print("Konlpy Komoran 활성화.")
except ImportError:
    KONLPY_KOMORAN_ENABLED = False
    komoran_tool = None
    print("Konlpy Komoran 비활성화. 설치 필요 (pip install konlpy).")
except Exception as e:
    KONLPY_KOMORAN_ENABLED = False
    komoran_tool = None
    print(f"Konlpy Komoran 로드 오류: {e}. 명사 추출 기능을 사용할 수 없습니다.")

# --- 전역 설정 ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'backtest_db',
    'charset': 'utf8mb4'
}

# 현재 스크립트가 실행되는 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))
# DOCX 파일들이 있는 폴더 (현재 스크립트가 있는 곳의 'datas' 서브폴더)
datas_folder = os.path.join(current_dir, 'datas')
# 처리된 DOCX 파일들이 이동될 폴더 (datas/done)
done_folder = os.path.join(datas_folder, 'done')
# 최종 통합 Excel 파일이 저장될 폴더 (여기서는 current_dir)
output_excel_folder = current_dir

# --- 유틸리티 함수 ---

def print_query_execution_time(start_time, end_time, record_count, query_type="REPLACE"):
    """쿼리 실행 시간 및 레코드 처리 정보를 출력."""
    execution_time = end_time - start_time
    time_per_record = execution_time / record_count if record_count > 0 else 0
    print(f"DB {query_type} 쿼리 실행: {execution_time:.4f}초, {record_count}건 (건당 {time_per_record:.6f}초)")

def convert_to_date(date_str):
    """'yyyymmdd' 문자열을 DATE 객체로 변환."""
    try:
        if re.match(r'^\d{8}$', date_str):
            return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        return None
    except (ValueError, IndexError):
        return None

def parse_date_from_filename(filename, current_year=datetime.now().year):
    """
    'MM.DD특징주.docx'와 같은 파일명에서 날짜를 파싱하여 datetime.date 객체로 반환합니다.
    년도는 현재 시스템의 년도를 사용합니다.
    """
    match = re.match(r'(\d{1,2})\.(\d{1,2})특징주\.docx', filename)
    if match:
        month = int(match.group(1))
        day = int(match.group(2))
        try:
            return date(current_year, month, day)
        except ValueError: # 유효하지 않은 날짜 (예: 2월 30일)
            return None
    return None

def extract_tables_from_docx(docx_path):
    """
    DOCX 파일에서 모든 표를 추출하여 Pandas DataFrame 리스트로 반환합니다.
    """
    if not os.path.exists(docx_path):
        print(f"오류: '{docx_path}' 파일을 찾을 수 없습니다.")
        return []

    try:
        document = Document(docx_path)
        tables_dataframes = []
        
        for i, table in enumerate(document.tables):
            data = []
            for row in table.rows:
                row_cells = [cell.text for cell in row.cells]
                data.append(row_cells)
            
            if not data:
                continue

            headers = data[0]
            table_data = data[1:]

            df = pd.DataFrame(table_data, columns=headers)
            tables_dataframes.append(df)
            
        return tables_dataframes

    except Exception as e:
        print(f"'{docx_path}' 파일 처리 중 오류가 발생했습니다: {e}")
        return []

def extract_themes_simple(reason_text: str) -> str:
    """
    띄어쓰기 교정된 텍스트에서 테마 추출.
    '테마'/'관련주' 앞에 오는 명사(구)를 추출하고 슬래시로 분리 후 추가 정제.
    """
    themes = []
    
    for match in re.finditer(r'\s*([가-힣A-Za-z0-9\/]+)\s*(?:테마|관련주)', reason_text):
        potential_theme_name = match.group(1).strip()
        split_themes = [t.strip() for t in re.split(r'/', potential_theme_name) if t.strip()]
        
        for theme_word in split_themes:
            # ... (기존 테마 추출 로직과 동일)
            if 2 <= len(theme_word) <= 10 and theme_word not in themes:
                themes.append(theme_word)

    return ','.join(sorted(list(set(themes))))

def move_processed_files_to_done(file_paths, destination_folder):
    """
    처리 완료된 파일들을 지정된 'done' 폴더로 이동합니다.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"'{destination_folder}' 폴더가 생성되었습니다.")

    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                shutil.move(file_path, os.path.join(destination_folder, file_name))
                print(f"'{file_name}' -> '{destination_folder}'로 이동 완료.")
            else:
                print(f"경고: '{file_path}' 파일이 존재하지 않아 이동할 수 없습니다.")
        except Exception as e:
            print(f"'{file_path}' 이동 중 오류 발생: {e}")

def initialize_database_tables(cursor, connection):
    """
    필요한 데이터베이스 테이블들을 생성하고, daily_theme 테이블 구조를 확인/수정합니다.
    """
    print("\n--- 데이터베이스 테이블 초기화 시작 ---")
    
    # daily_theme 테이블 생성 (reason_nouns JSON 컬럼 포함)
    create_daily_theme_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS `daily_theme` (
            `date` DATE NOT NULL,
            `market` VARCHAR(8) NOT NULL ,
            `stock_code` VARCHAR(7) NOT NULL ,
            `stock_name` VARCHAR(25) NOT NULL ,
            `rate` DECIMAL(5,2) NULL DEFAULT '0.00',
            `amount` INT(11) NULL DEFAULT '0',
            `reason` VARCHAR(250) NOT NULL ,
            `reason_nouns` JSON NULL , -- JSON 타입 컬럼
            `theme` VARCHAR(250) NULL DEFAULT NULL ,
            PRIMARY KEY (`date`, `market`, `stock_code`) USING BTREE
        );
    """)
    cursor.execute(create_daily_theme_table_query)
    print("daily_theme 테이블 존재 확인 및 필요시 생성 완료.")

    # 'reason_nouns' 컬럼 존재 여부 확인 및 추가 (안정성 강화)
    cursor.execute("SHOW COLUMNS FROM daily_theme LIKE 'reason_nouns'")
    if not cursor.fetchone():
        print("daily_theme 테이블에 'reason_nouns' 컬럼이 없어 추가합니다.")
        cursor.execute("ALTER TABLE daily_theme ADD COLUMN reason_nouns JSON NULL AFTER reason")
        print("'reason_nouns' 컬럼 추가 완료.")

    # 다른 테이블들 생성... (기존과 동일)
    # ...

    connection.commit()
    print("--- 데이터베이스 테이블 초기화 완료 ---")


def process_korean_text(original_reason):
    """주어진 텍스트에 대해 띄어쓰기, 명사추출, 테마추출을 수행"""
    final_reason = original_reason
    if PYKOSPACING_ENABLED and final_reason:
        try:
            final_reason = spacing_tool(final_reason)
        except Exception as e:
            print(f"경고: 띄어쓰기 교정 오류 ('{original_reason[:30]}...'): {e}.")
    
    reason_nouns_list = []
    if KONLPY_KOMORAN_ENABLED and final_reason:
        try:
            nouns = komoran_tool.nouns(final_reason)
            reason_nouns_list = sorted(list(set([n for n in nouns if len(n) >= 2])))
        except Exception as e:
            print(f"경고: 명사 추출 오류 ('{final_reason[:30]}...'): {e}.")
    
    reason_nouns_json = json.dumps(reason_nouns_list, ensure_ascii=False)
    theme_str = extract_themes_simple(final_reason)
    
    return final_reason, reason_nouns_json, theme_str

# --- 메인 스크립트 실행 ---
def main():
    connection = None
    cursor = None
    try:
        print("특징주 데이터 처리 스크립트를 시작합니다.")
        
        # --- MariaDB 연결 및 초기화 ---
        connection = pymysql.connect(**DB_CONFIG)
        cursor = connection.cursor()
        initialize_database_tables(cursor, connection)

        # --- [신규 로직] DB 내 미처리 데이터(2번 스크립트 입력분 등) 업데이트 ---
        print("\n--- DB 내 미처리 데이터 업데이트 시작 ---")
        select_unprocessed_query = "SELECT date, market, stock_code, reason FROM daily_theme WHERE reason_nouns IS NULL"
        cursor.execute(select_unprocessed_query)
        records_to_process = cursor.fetchall()

        if not records_to_process:
            print("한글 파서 처리가 필요한 기존 데이터가 없습니다.")
        else:
            print(f"총 {len(records_to_process)}건의 기존 데이터에 대해 한글 파서 처리를 시작합니다.")
            data_to_update = []
            for record in records_to_process:
                date_val, market_val, stock_code_val, original_reason = record
                if not original_reason:
                    continue
                
                processed_reason, reason_nouns_json, theme_str = process_korean_text(original_reason)
                data_to_update.append((processed_reason, reason_nouns_json, theme_str, date_val, market_val, stock_code_val))

            if data_to_update:
                start_time = time.time()
                update_query = textwrap.dedent("""
                    UPDATE daily_theme SET reason = %s, reason_nouns = %s, theme = %s
                    WHERE date = %s AND market = %s AND stock_code = %s
                """)
                cursor.executemany(update_query, data_to_update)
                connection.commit()
                print_query_execution_time(start_time, time.time(), len(data_to_update), "UPDATE")
                print(f"총 {len(data_to_update)}건의 기존 데이터가 성공적으로 업데이트되었습니다.")

        # --- [기존 로직] DOCX 파일 처리 및 DataFrame 생성 ---
        print(f"\n--- '{datas_folder}' 폴더에서 특징주 DOCX 파일 검색 및 처리 시작 ---")
        
        # 'datas' 폴더와 'datas/done' 폴더 생성 확인
        if not os.path.exists(datas_folder):
            os.makedirs(datas_folder)
        if not os.path.exists(done_folder):
            os.makedirs(done_folder)

        docx_files = [f for f in os.listdir(datas_folder) if f.endswith('.docx')]
        if not docx_files:
            print(f"'{datas_folder}' 폴더에서 처리할 '.docx' 파일을 찾을 수 없습니다.")
            # DOCX 파일이 없어도 스크립트가 종료되지 않도록 return 제거
        else:
            all_features_dfs = []
            successfully_processed_docx_paths = []
            
            # ... (기존의 DOCX 파일 처리 로직)
            for filename in sorted(docx_files):
                print(f"\n--- '{filename}' 파일 처리 중 ---")
                file_path = os.path.join(datas_folder, filename)
                feature_date = parse_date_from_filename(filename)
                if feature_date is None:
                    print(f"경고: '{filename}' 파일명에서 날짜 파싱 불가. 건너뜀.")
                    continue

                dfs = extract_tables_from_docx(file_path)
                if not dfs:
                    print(f"경고: '{filename}' 파일에서 표를 찾을 수 없음. 건너뜀.")
                    continue
                
                # ... (이하 DataFrame 처리 로직은 기존과 거의 동일)
                current_df = dfs[0].copy()
                current_df['날짜'] = feature_date
                # ... 컬럼명 변경, 컬럼 추가 등 ...
                all_features_dfs.append(current_df)
                successfully_processed_docx_paths.append(file_path)

            if all_features_dfs:
                final_df = pd.concat(all_features_dfs, ignore_index=True)
                # ... 데이터 정제 ...

                data_to_insert = []
                for index, row in final_df.iterrows():
                    # ... 데이터 준비 ...
                    date_obj = row['날짜']
                    stock_name = str(row['종목명'])
                    original_reason = str(row['이유'])
                    
                    cursor.execute("SELECT stock_code FROM stock_info WHERE stock_name = %s", (stock_name,))
                    stock_info_result = cursor.fetchone()

                    if stock_info_result:
                        stock_code = stock_info_result[0]
                        processed_reason, reason_nouns_json, theme_str = process_korean_text(original_reason)
                        
                        # ... rate, amount 등 다른 값들 준비 ...
                        rate = row.get('등락률', 0)
                        amount = row.get('거래대금', 0)
                        market = row.get('구분', '정규장')

                        data_to_insert.append((date_obj, market, stock_code, stock_name, rate, amount, processed_reason, reason_nouns_json, theme_str))
                    else:
                        print(f"경고: '{stock_name}' 종목코드 미발견. 레코드 건너뜀.")

                if data_to_insert:
                    start_time = time.time()
                    replace_query = textwrap.dedent("""
                        REPLACE INTO daily_theme (date, market, stock_code, stock_name, rate, amount, reason, reason_nouns, theme)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """)
                    cursor.executemany(replace_query, data_to_insert)
                    connection.commit()
                    print_query_execution_time(start_time, time.time(), len(data_to_insert), "REPLACE")
                    print(f"총 {len(data_to_insert)}건의 신규 데이터가 daily_theme 테이블에 업데이트되었습니다.")
                else:
                    print("DOCX 파일에서 삽입/업데이트할 레코드 없음.")
                
                print("\n--- 처리된 DOCX 파일 'done' 폴더로 이동 중 ---")
                move_processed_files_to_done(successfully_processed_docx_paths, done_folder)

    except pymysql.MySQLError as e:
        if connection: connection.rollback()
        print(f"MariaDB 오류: {e}")
    except Exception as e:
        if connection: connection.rollback()
        print(f"예상치 못한 오류: {e}")
        import traceback; traceback.print_exc()
    finally:
        if cursor: cursor.close()
        if connection: connection.close()
        print("\nMariaDB 연결 해제. 스크립트 실행 완료.")

if __name__ == "__main__":
    main()
