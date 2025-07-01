import pandas as pd
import pymysql
import os.path
import sys
import re
import textwrap
import time
from datetime import datetime, date
# --- 설정 ---
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'backtest_db',
    'charset': 'utf8mb4'
}


# Excel file path
excel_file_name = 'judal_theme.xlsx' # 엑셀 파일명
excel_dir = 'datas' # 엑셀 파일이 있는 디렉토리
modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
input_excel_path = os.path.join(modpath, excel_dir, excel_file_name)

def print_query_execution_time(start_time, end_time, record_count, table_name=""):
    """쿼리 실행 시간, 레코드 건수, 레코드당 실행 시간을 출력하는 함수"""
    execution_time = end_time - start_time
    time_per_record = execution_time / record_count if record_count > 0 else 0
    print(f"[{table_name}] REPLACE 쿼리 실행 시간: {execution_time:.4f} 초")
    print(f"[{table_name}] 처리된 레코드 건수: {record_count} 건")
    print(f"[{table_name}] 레코드 1건당 실행 시간: {time_per_record:.6f} 초")
    print(f"[{table_name}] Data insertion/update completed")

def get_stock_code_by_name(cursor, stock_name):
    """종목명으로 종목코드를 조회하는 함수"""
    try:
        # stock_info 테이블에서 종목명으로 종목코드 조회
        query = "SELECT stock_code FROM stock_info WHERE stock_name = %s"
        cursor.execute(query, (stock_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0]  # 종목코드 반환
        else:
            print(f"경고: 종목명 '{stock_name}'에 해당하는 종목코드를 찾을 수 없습니다.")
            return None
    except Exception as e:
        print(f"종목코드 조회 중 오류 발생: {e}")
        return None

try:
    # Connect to MariaDB
    connection = pymysql.connect(**db_config)
    cursor = connection.cursor()

    # 1. theme_class 테이블 생성 (존재하지 않을 경우)
    # theme_id가 PRIMARY KEY이고, (theme, theme_class)는 UNIQUE KEY로 변경
    create_theme_class_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS theme_class (
            theme_id INT AUTO_INCREMENT PRIMARY KEY,
            theme VARCHAR(30) NOT NULL,
            theme_class VARCHAR(30) NOT NULL,
            theme_hit INT  DEFAULT 0,   
            theme_score INT DEFAULT 0,   
            momentum_score DECIMAL(10,4) DEFAULT 0,
            theme_desc VARCHAR(1000) NULL,
            UNIQUE KEY ak_theme_name_class (theme, theme_class), -- Alternate Key (대체 키)
            INDEX idx_theme (theme),
            INDEX idx_theme_class (theme_class)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    cursor.execute(create_theme_class_table_query)
    print("theme_class 테이블 존재 확인 및 필요시 생성 완료.")

    # 2. theme_stock 테이블 생성 (존재하지 않을 경우)
    # PK가 (theme_id, stock_code)로 변경됨
    create_theme_stock_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS theme_stock (
            theme_id INT NOT NULL,
            stock_code VARCHAR(7) NOT NULL,
            stock_score INT DEFAULT 0,                                        
            PRIMARY KEY (theme_id, stock_code),
            INDEX idx_ts_theme (theme_id),
            INDEX idx_ts_stock_code (stock_code)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    cursor.execute(create_theme_stock_table_query)
    print("theme_stock 테이블 존재 확인 및 필요시 생성 완료.")
    
    connection.commit()

    # Read Excel file (종목코드 컬럼 제거)
    df = pd.read_excel(input_excel_path, usecols=['테마', '종목명'])
    df = df.dropna(subset=['테마', '종목명']) # 테마나 종목명이 없는 행은 스킵
    print(f"엑셀 파일 '{input_excel_path}' 로드 완료. 총 {len(df)}개 레코드.")

    theme_class_data_to_insert = set() # 중복 방지를 위해 set 사용
    theme_stock_data_to_insert = set() # 중복 방지를 위해 set 사용

    # 테마 파싱을 위한 정규식 (괄호 처리 주석처리)
    # base_str: 괄호 바깥의 테마명 (예: '로봇', 'AI/챗봇', '교육/온라인 교육')
    # class_content: 괄호 안의 내용 (예: '산업용/협동로봇', '챗GPT 등')
    # theme_parser_regex = re.compile(r'(.+?)(?:\s*\(([^)]+)\))?$') # 수정: $ 앵커 추가, 마지막 괄호 매칭
    
    for index, row in df.iterrows():
        excel_theme_raw = str(row['테마']).strip()
        excel_stock_name_raw = str(row['종목명']).strip()

        # '신규상장' 테마는 처리하지 않음
        if '신규상장' in excel_theme_raw:
            # print(f"[DEBUG] '신규상장' 테마를 포함하므로 건너뜁니다: {excel_theme_raw}")
            continue

        # 종목명으로 종목코드 조회
        stock_code = get_stock_code_by_name(cursor, excel_stock_name_raw)
        if not stock_code:
            print(f"종목코드를 찾을 수 없어 건너뜁니다: {excel_stock_name_raw}")
            continue

        # 괄호 처리 로직 주석처리 - 테마명을 그대로 사용
        base_themes_str = excel_theme_raw
        # class_content = None

        # match_result = theme_parser_regex.search(excel_theme_raw)
        
        # if match_result:
        #     base_themes_str = match_result.group(1).strip()
        #     class_content = match_result.group(2) # 괄호 안의 내용 (None일 수 있음)

        #     if class_content:
        #         # 괄호 안 내용에서 ' 등' 제거 (예: '챗GPT 등' -> '챗GPT')
        #         class_content = class_content.replace(' 등', '').strip()
        
        # 기본 테마명 '/'로 분리 (예: 'AI/챗봇' -> ['AI', '챗봇'])
        base_themes = [t.strip() for t in base_themes_str.split('/') if t.strip()]
        
        # 클래스 내용 '/'로 분리 로직 주석처리
        # processed_classes = []
        # if class_content:
        #     processed_classes = [c.strip() for c in class_content.split('/') if c.strip()]

        # 디버깅을 위해 아래 주석을 해제하여 파싱 결과 확인 가능
        # print(f"\n[DEBUG] 원본 테마: '{excel_theme_raw}'")
        # print(f"[DEBUG] 추출된 기본 테마 리스트: {base_themes}")
        # print(f"[DEBUG] 추출된 분류 클래스 리스트: {processed_classes}")

        # theme_class 테이블에 삽입할 데이터 구성 (괄호 처리 없이 빈 문자열로 theme_class 저장)
        for theme in base_themes:
            theme_class_data_to_insert.add((theme, '')) # theme_class는 빈 문자열로 저장

        # theme_stock 테이블에 삽입할 데이터 구성 (모든 기본 테마에 대해 종목코드 매핑)
        for theme in base_themes:
            theme_stock_data_to_insert.add((theme, stock_code))

    # theme_class 데이터 삽입/업데이트
    if theme_class_data_to_insert:
        start_time_tc = time.time()
        replace_theme_class_query = textwrap.dedent("""
            REPLACE INTO theme_class (theme, theme_class, theme_desc)
            VALUES (%s, %s, NULL) -- description 컬럼은 현재 엑셀에 없으므로 NULL
        """)
        cursor.executemany(replace_theme_class_query, list(theme_class_data_to_insert))
        connection.commit()
        end_time_tc = time.time()
        print_query_execution_time(start_time_tc, end_time_tc, len(theme_class_data_to_insert), "theme_class")
    else:
        print("theme_class 테이블에 삽입/업데이트할 레코드가 없습니다.")

    # theme_stock 데이터 삽입/업데이트 (theme_id 조인하여 삽입)
    if theme_stock_data_to_insert:
        start_time_ts = time.time()
        # theme_id를 조인하여 가져오는 쿼리
        replace_theme_stock_query = textwrap.dedent("""
            REPLACE INTO theme_stock (theme_id, stock_code)
            SELECT tc.theme_id, %s
            FROM theme_class tc
            WHERE tc.theme = %s
        """)
        
        # 각 (theme, stock_code) 쌍에 대해 theme_id를 조인하여 삽입
        for theme, stock_code in theme_stock_data_to_insert:
            cursor.execute(replace_theme_stock_query, (stock_code, theme))
        
        connection.commit()
        end_time_ts = time.time()
        print_query_execution_time(start_time_ts, end_time_ts, len(theme_stock_data_to_insert), "theme_stock")
    else:
        print("theme_stock 테이블에 삽입/업데이트할 레코드가 없습니다.")

except pymysql.MySQLError as e:
    if 'connection' in locals() and connection:
        connection.rollback()
    print(f"MariaDB 오류 발생: {e}")
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {input_excel_path}")
except pd.errors.EmptyDataError:
    print(f"엑셀 파일 '{input_excel_path}'이(가) 비어 있습니다.")
except pd.errors.ParserError as e:
    print(f"엑셀 파싱 오류: {e}")
except Exception as e:
    if 'connection' in locals() and connection:
        connection.rollback()
    print(f"예상치 못한 오류 발생: {e}")
    import traceback
    traceback.print_exc() # 오류 스택 트레이스 출력
finally:
    if 'connection' in locals() and connection:
        cursor.close()
        connection.close()
        print("MariaDB 연결 해제.")