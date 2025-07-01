import pandas as pd
import pymysql
import os.path
import sys
import re
import textwrap
import time
from datetime import datetime, date
from decimal import Decimal

# PyKoSpacing 모듈 로드 (띄어쓰기 교정용)
# 32비트 환경 등에서 로드 실패 시 기능 비활성화
try:
    from pykospacing import Spacing
    PYKOSPACING_ENABLED = True
    spacing_tool = Spacing()
    print("PyKoSpacing 활성화.")
except ImportError:
    PYKOSPACING_ENABLED = False
    spacing_tool = None
    print("PyKoSpacing 비활성화. 설치 필요 또는 32비트 환경 미지원.")

# --- 전역 설정 ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'backtest_db',
    'charset': 'utf8mb4'
}

EXCEL_FILE_NAME = '특징주250611.xlsx'
MOD_PATH = os.path.dirname(os.path.abspath(sys.argv[0]))
INPUT_EXCEL_PATH = os.path.join(MOD_PATH, 'datas', EXCEL_FILE_NAME)

# --- 유틸리티 함수 ---

def print_query_execution_time(start_time, end_time, record_count):
    """쿼리 실행 시간 및 레코드 처리 정보를 출력."""
    execution_time = end_time - start_time
    time_per_record = execution_time / record_count if record_count > 0 else 0
    print(f"REPLACE 쿼리 실행: {execution_time:.4f}초, {record_count}건 (건당 {time_per_record:.6f}초)")

def convert_to_date(date_str):
    """'yyyymmdd' 문자열을 DATE 객체로 변환."""
    try:
        if re.match(r'^\d{8}$', date_str):
            return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        return None
    except (ValueError, IndexError):
        return None

def extract_themes_simple(reason_text: str) -> str:
    """
    띄어쓰기 교정된 텍스트에서 테마 추출.
    '테마'/'관련주' 앞에 오는 명사(구)를 추출하고 슬래시로 분리 후 추가 정제.
    """
    themes = []
    
    # '테마' 또는 '관련주' 직전의 단어(구) 캡처
    # (주의: PyKoSpacing 결과에 따라 붙어있는 단어 전체가 캡처될 수 있음)
    for match in re.finditer(r'\s*([가-힣A-Za-z0-9\/]+)\s*(?:테마|관련주)', reason_text):
        potential_theme_name = match.group(1).strip()
        
        # 슬래시로 구분된 여러 테마 처리 (예: "2차전지/배터리" -> ["2차전지", "배터리"])
        split_themes = [t.strip() for t in re.split(r'/', potential_theme_name) if t.strip()]
        
        for theme_word in split_themes:
            # theme_word는 현재 공백이 없는 단일 문자열 상태임

            # --- 5. '~에' 로 문자열 끝나면 처리 않음 (테마로 부적합) ---
            # 가장 먼저 처리하여 불필요한 연산 방지
            if theme_word.endswith('에') and len(theme_word) > 1: # '에'만 있는 경우는 제외
                continue

            # --- 불필요한 패턴 정의 및 제거 ---
            # 패턴과 그 앞의 모든 문자열을 제거
            
            # 1. 단일 불필요 단어
            # '속', '앞두고', '부각', '일부', '지속', '부상', '선고일' 등
            # 이 단어들이 나타나면 그 앞의 모든 문자열을 제거
            single_words_to_trim_before = [
                '속', '부각', '일부', '지속', '임박', '부상', '선고', '수혜', 
                '선고일', '앞두고', '수혜주', '재확산'
                
            ]
            
            # 2. '핵심단어' + '조사' 조합
            # '소식', '기대감', '전망', '격화', '수혜주', '최대', '협약', '상고', '급등'
            # + '에', '속', '로', '한'
            words_for_combined_pattern = [
                '소식', '기대감', '전망', '격화', '최대', '협약', '상고', '급등', '지연',
                '호실적', '최고조'
            ]
            post_positions = ['에', '속', '로', '한']
            
            combined_patterns_to_trim_before = []
            for word_part in words_for_combined_pattern:
                for post_pos in post_positions:
                    combined_patterns_to_trim_before.append(word_part + post_pos)
            
            # 모든 패턴을 하나의 리스트로 합치고, 길이가 긴 것부터 처리
            # 그래야 '최대소식에'가 '소식에'보다 먼저 매칭됨
            all_patterns_to_trim_before = sorted(
                single_words_to_trim_before + combined_patterns_to_trim_before,
                key=len,
                reverse=True
            )

            # --- 패턴을 포함해 그 앞의 문자열 제거 처리 ---
            original_theme_word = theme_word # 디버깅용
            
            found_and_trimmed = False
            for pattern in all_patterns_to_trim_before:
                # 정규식을 사용하여 패턴을 찾고, 그 패턴과 그 앞의 모든 문자열을 제거
                # r'.*?'는 비탐욕적으로 일치하는 가장 짧은 문자열
                # re.escape(pattern)은 패턴 내의 특수문자 처리를 위함
                match = re.search(rf'(.*?){re.escape(pattern)}(.*)', theme_word)
                if match:
                    # 패턴 뒤의 문자열만 취함
                    theme_word = match.group(2).strip()
                    found_and_trimmed = True
                    break # 하나라도 매칭되면 다음 패턴 검사 중단
            
            # 패턴에 의해 아무것도 제거되지 않았고, 여전히 불필요한 접두사가 있는 경우 대비
            # (예: '속'이 첫 단어로만 나타나야 하는 경우 등)
            # 이 부분은 주석처리하고, 위에서 모든 '제거해야 할 단어'를 패턴에 포함시키는 것을 우선함
            # if not found_and_trimmed:
            #    # 특정 접두사만 제거하는 로직이 필요하다면 여기에 추가
            #    pass

            if not theme_word: continue # 처리 후 빈 문자열이 되면 다음으로
            
            # 최종 유효성 검사 (2~15자 길이) 및 중복 방지
            if 2 <= len(theme_word) <= 10 and theme_word not in themes:
                themes.append(theme_word)

    return ','.join(sorted(list(set(themes))))

# (이하 main 함수는 동일하므로 생략)
# --- 메인 스크립트 실행 ---
def main():
    connection = None
    cursor = None
    try:
        connection = pymysql.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # daily_theme 테이블 생성 (필요시)
        create_table_query = textwrap.dedent("""
            CREATE TABLE IF NOT EXISTS daily_theme (
                date DATE NOT NULL, market VARCHAR(8) NOT NULL, stock_code VARCHAR(7) NOT NULL,
                stock_name VARCHAR(25) NOT NULL, rate DECIMAL(5,2) NULL DEFAULT 0,
                amount INT(11) NULL DEFAULT 0, reason VARCHAR(250) NOT NULL, theme VARCHAR(250) NULL,
                PRIMARY KEY (date, market, stock_code) USING BTREE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        cursor.execute(create_table_query)
        connection.commit()
        print("daily_theme 테이블 확인/생성 완료.")

        df = pd.read_excel(INPUT_EXCEL_PATH, usecols=['일자', '구분', '종목명', '등락률', '거래대금', '이유']).dropna()
        print(f"엑셀 파일 '{INPUT_EXCEL_PATH}' 로드 완료. {len(df)}개 레코드.")

        data_to_insert = []

        for index, row in df.iterrows():
            date_str = str(row['일자'])[:8]
            market = str(row['구분'])
            stock_name = str(row['종목명'])
            rate = row['등락률']
            amount = row['거래대금']
            original_reason = str(row['이유'])

            date_obj = convert_to_date(date_str)
            if not date_obj:
                print(f"경고: 유효하지 않은 날짜 형식 ({row['일자']}). 레코드 건너뜀.")
                continue

            # 등락률 및 거래대금 데이터 정제
            if isinstance(rate, str):
                rate = re.sub(r'[^0-9.-]', '', rate.replace('..', '.').replace(',', '.').replace('/', '.'))
                try: rate = float(rate)
                except ValueError: print(f"경고: 유효하지 않은 등락률 ({row['등락률']}). 레코드 건너뜀."); continue
            if isinstance(rate, (int)) and (abs(rate) > 999):
                rate = float(str(rate)[:-2] + '.' + str(rate)[-2:]) if len(str(rate)) >= 3 else float(str(rate))
            if isinstance(amount, str):
                amount = re.sub(r'[^\d]', '', amount)
                try: amount = int(amount)
                except ValueError: print(f"경고: 유효하지 않은 거래대금 ({row['거래대금']}). 레코드 건너뜀."); continue
            if isinstance(rate, (int, float)) and (rate > 999.99 or rate < -999.99):
                print(f"경고: 등락률 범위 초과 ({rate}). 레코드 건너뜀."); continue

            # 종목코드를 stock_info 테이블에서 조회
            cursor.execute("SELECT stock_code FROM stock_info WHERE stock_name = %s", (stock_name,))
            stock_info_result = cursor.fetchone()

            if stock_info_result:
                stock_code = stock_info_result[0]
                
                # DB에 저장할 최종 'reason' 값 결정 및 띄어쓰기 교정 적용 로직
                final_reason_for_db = original_reason
                if PYKOSPACING_ENABLED:
                    # DB에 기존 레코드가 있는지 조회
                    cursor.execute("SELECT reason FROM daily_theme WHERE date = %s AND market = %s AND stock_code = %s", (date_obj, market, stock_code))
                    existing_reason_in_db = cursor.fetchone()

                    # 1. DB에 레코드가 없거나 2. DB의 reason이 엑셀 원본과 다를 경우에만 띄어쓰기 교정 시도
                    # (즉, 새로운 데이터이거나, 기존 데이터가 아직 교정되지 않았을 경우)
                    if not existing_reason_in_db or existing_reason_in_db[0] != original_reason:
                        try:
                            final_reason_for_db = spacing_tool(original_reason)
                        except Exception as e:
                            print(f"경고: 띄어쓰기 교정 오류: {e}. 원본 이유 사용.")
                            final_reason_for_db = original_reason
                    else: # DB에 이미 (엑셀 원본과 일치하는) reason이 있다면, 이미 교정되었을 것으로 보고 DB 값 사용
                        final_reason_for_db = existing_reason_in_db[0]

                # 교정된(또는 원본) reason으로 테마 추출
                theme_str = extract_themes_simple(final_reason_for_db)
                
                data_to_insert.append((date_obj, market, stock_code, stock_name, rate, amount, final_reason_for_db, theme_str))
            else:
                print(f"경고: '{stock_name}' 종목코드 미발견. 레코드 건너뜀.")
        
        # 데이터 일괄 삽입/업데이트
        if data_to_insert:
            start_time = time.time()
            replace_query = textwrap.dedent("""
                REPLACE INTO daily_theme (date, market, stock_code, stock_name, rate, amount, reason, theme)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """)
            cursor.executemany(replace_query, data_to_insert)
            connection.commit()
            print_query_execution_time(start_time, time.time(), len(data_to_insert))
        else:
            print("삽입/업데이트할 레코드 없음.")

    except pymysql.MySQLError as e:
        if connection: connection.rollback()
        print(f"MariaDB 오류: {e}")
    except FileNotFoundError:
        print(f"파일을 찾을 수 없음: {INPUT_EXCEL_PATH}")
    except pd.errors.EmptyDataError:
        print(f"엑셀 파일 '{INPUT_EXCEL_PATH}' 비어 있음.")
    except pd.errors.ParserError as e:
        print(f"엑셀 파싱 오류: {e}")
    except Exception as e:
        if connection: connection.rollback()
        print(f"예상치 못한 오류: {e}")
        import traceback; traceback.print_exc()
    finally:
        if cursor: cursor.close()
        if connection: connection.close()
        print("MariaDB 연결 해제.")

if __name__ == "__main__":
    main()