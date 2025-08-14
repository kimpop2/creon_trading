import pandas as pd
import pymysql
import os.path
import sys
import re
import textwrap
import time
from datetime import datetime, date
from decimal import Decimal # Decimal 타입 임포트 추가

# MariaDB connection configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'backtest_db',
    'charset': 'utf8mb4'
}

# Excel file path
excel_file_path = 'datas/특징주DB.xlsx'
modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
input_excel_path = os.path.join(modpath, excel_file_path)

def print_query_execution_time(start_time, end_time, record_count):
    """쿼리 실행 시간, 레코드 건수, 레코드당 실행 시간을 출력하는 함수"""
    execution_time = end_time - start_time
    time_per_record = execution_time / record_count if record_count > 0 else 0
    print(f"REPLACE 쿼리 실행 시간: {execution_time:.4f} 초")
    print(f"처리된 레코드 건수: {record_count} 건")
    print(f"레코드 1건당 실행 시간: {time_per_record:.6f} 초")
    print("Data insertion/update completed")

try:
    # Connect to MariaDB
    connection = pymysql.connect(**db_config)
    cursor = connection.cursor()

    # Create table if it doesn't exist
    # daily_theme 테이블의 컬럼 정보를 프롬프트 및 코드 사용에 맞게 수정
    create_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS daily_theme (
            date VARCHAR(8) NOT NULL,              -- 일자 (yyyymmdd)
            market VARCHAR(8) NOT NULL,      -- 구분 (정규장:1, 시간외:2, 또는 문자열)
            stock_code VARCHAR(7) NOT NULL,  -- 종목코드
            stock_name VARCHAR(25) NOT NULL, -- 종목명 (엑셀의 '종목명'에 대응)
            rate DECIMAL(5,2) NULL DEFAULT 0, -- 등락률 (DECIMAL(5,2)로 확장)
            amount INT(11) NULL DEFAULT 0,   -- 거래대금
            reason VARCHAR(250) NOT NULL,    -- 이유
            theme VARCHAR(250) NULL,          -- 테마명
            PRIMARY KEY (date, market, stock_code) USING BTREE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    cursor.execute(create_table_query)
    connection.commit()
    
    print("daily_theme 테이블 존재 확인 및 필요시 생성 완료.")

    # Read Excel file
    # 엑셀 컬럼 중 '일자', '구분', '종목명', '등락률', '거래대금', '이유'를 사용
    df = pd.read_excel(input_excel_path, usecols=['일자', '구분', '종목명', '등락률', '거래대금', '이유'])
    # Drop rows with missing values
    df = df.dropna()
    print(f"엑셀 파일 '{input_excel_path}' 로드 완료. 총 {len(df)}개 레코드.")

    data_to_insert = []

    for index, row in df.iterrows():
        date_str = str(row['일자']) # 날짜 컬럼을 문자열로 확실히 변환
        market = str(row['구분'])   # 구분 컬럼을 문자열로 확실히 변환
        stock_name = str(row['종목명']) # 종목명 (엑셀 컬럼)
        rate = row['등락률']
        amount = row['거래대금']
        reason = str(row['이유'])   # 이유 컬럼을 문자열로 확실히 변환
        
        # Check if date is in yyyymmdd format
        date_str = date_str[:8] # 앞에서 8자리까지만
        if not re.match(r'^\d{8}$', date_str):
            print(f"경고: 유효하지 않은 날짜 형식 (yyyymmdd)으로 행을 건너뜁니다: {row['일자']} (변환 후: {date_str})")
            continue

        # 등락률 교정
        if isinstance(rate, str): 
            rate = rate.replace('..', '.').replace(',', '.').replace('/', '.')
            parts = rate.split('.')
            if len(parts) > 1: # 소수점 포함
                # 소수점 뒤에 여러 개의 점이 있을 경우 첫 번째 점만 유효하게 처리
                rate = parts[0] + '.' + ''.join(parts[1:])
            
            # 숫자, -기호, . 제외한 문자 제거
            rate = re.sub(r'[^\d.-]', '', rate)
            try:
                rate = float(rate)
            except ValueError:
                print(f"경고: 유효하지 않은 등락률 값으로 행을 건너뜁니다: {row['등락률']} (변환 후: {rate})")
                continue
        
        # 오타에 의해 소수점이 없는 등락률 데이터 교정 (e.g., 1234 -> 12.34)
        if isinstance(rate, (int)) and (abs(rate) > 999) : # 등락률이 100%를 초과하는 큰 정수인 경우
            rate_str = str(rate)
            if len(rate_str) >= 3: # 최소 3자리여야 소수점 처리가 의미 있음 (e.g., 123 -> 1.23)
                rate = float(rate_str[:-2] + '.' + rate_str[-2:])
            else: # 너무 짧은 정수는 그대로 사용 (e.g., 12 -> 12.00)
                rate = float(rate_str)
        
        # 거래대금 자리수 콤마 제거 및 정수 변환
        if isinstance(amount, str):
            amount = re.sub(r'[^\d]', '', amount) # 숫자 외 문자 제거
            try:
                amount = int(amount)
            except ValueError:
                print(f"경고: 유효하지 않은 거래대금 값으로 행을 건너뜁니다: {row['거래대금']} (변환 후: {amount})")
                continue
        
        # 등락률 데이터 범위 유효성 검사 (매우 큰 등락률은 오류로 간주)
        if isinstance(rate, (int, float)) and (rate > 999.99 or rate < -999.99):
            print(f"경고: 범위를 벗어난 등락률 값으로 행을 건너뜁니다: {rate}")
            continue

        # Get stock_code from stock_info table using stock_name
        # '종목명'에 해당하는 stock_code를 stock_info 테이블에서 조회
        get_code_query = textwrap.dedent("""
            SELECT stock_code FROM stock_info WHERE stock_name = %s
        """)
        cursor.execute(get_code_query, (stock_name,)) # stock_name 변수 사용
        stock_info_result = cursor.fetchone()

        if stock_info_result:
            stock_code = stock_info_result[0]
            
            # Extract theme from reason
            themes = []
            # '테마'가 발견되면 그 앞 단어로 설정 (e.g., '로봇테마' -> '로봇')
            # 띄어쓰기가 있거나 없는 경우 모두 처리 (e.g., '로봇 테마', '로봇테마')
            # 중복 테마 방지
            for match in re.finditer(r'([0-9a-zA-Z가-힣]+)\s?테마', reason):
                theme = match.group(1)
                # 테마 길이가 2글자 이상이고, 아직 추가되지 않은 경우에만 추가
                if len(theme) >= 2 and theme not in themes:
                    themes.append(theme)
            theme_str = ','.join(themes) # 여러 테마는 쉼표로 구분하여 문자열로 저장

            
            
            # DB 삽입/업데이트를 위한 데이터 리스트에 추가
            data_to_insert.append((date_str, market, stock_code, stock_name, rate, amount, reason, theme_str))
        else:
            print(f"경고: 'stock_info' 테이블에서 종목명 '{stock_name}'에 해당하는 종목코드를 찾을 수 없습니다. 이 레코드를 건너뜁니다.")
    
    # Insert/update data using REPLACE
    if data_to_insert:
        start_time = time.time() # 쿼리 시작 시간 기록
        replace_query = textwrap.dedent("""
            REPLACE INTO daily_theme (date, market, stock_code, stock_name, rate, amount, reason, theme)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """)
        cursor.executemany(replace_query, data_to_insert)
        connection.commit()
        end_time = time.time() # 쿼리 종료 시간 기록
        
        # 쿼리 결과 출력
        print_query_execution_time(start_time, end_time, len(data_to_insert))
    else:
        print("삽입/업데이트할 레코드가 없습니다.")

except pymysql.MySQLError as e:
    if 'connection' in locals() and connection: # connection 객체가 존재하는지 확인
        connection.rollback()
    print(f"MariaDB 오류 발생: {e}")
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {input_excel_path}")
except pd.errors.EmptyDataError: # 빈 엑셀 파일일 경우 처리
    print(f"엑셀 파일 '{input_excel_path}'이(가) 비어 있습니다.")
except pd.errors.ParserError as e:
    print(f"엑셀 파싱 오류: {e}")
except Exception as e:
    if 'connection' in locals() and connection: # connection 객체가 존재하는지 확인
        connection.rollback()
    print(f"예상치 못한 오류 발생: {e}")
    import traceback
    traceback.print_exc() # 오류 스택 트레이스 출력
finally:
    if 'connection' in locals() and connection: # connection 객체가 존재하는지 확인
        cursor.close()
        connection.close()
        print("MariaDB 연결 해제.")