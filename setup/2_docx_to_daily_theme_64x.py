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

def print_query_execution_time(start_time, end_time, record_count):
    """쿼리 실행 시간 및 레코드 처리 정보를 출력."""
    execution_time = end_time - start_time
    time_per_record = execution_time / record_count if record_count > 0 else 0
    print(f"DB 쿼리 실행: {execution_time:.4f}초, {record_count}건 (건당 {time_per_record:.6f}초)")

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
                row_cells = []
                for cell in row.cells:
                    row_cells.append(cell.text)
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
            if theme_word.endswith('에') and len(theme_word) > 1:
                continue

            single_words_to_trim_before = [
                '속', '부각', '일부', '지속', '임박', '부상', '선고', '수혜', 
                '선고일', '앞두고', '수혜주', '재확산'
                
            ]
            
            words_for_combined_pattern = [
                '소식', '기대감', '전망', '격화', '최대', '협약', '상고', '급등', '지연',
                '호실적', '최고조'
            ]
            post_positions = ['에', '속', '로', '한']
            
            combined_patterns_to_trim_before = []
            for word_part in words_for_combined_pattern:
                for post_pos in post_positions:
                    combined_patterns_to_trim_before.append(word_part + post_pos)
            
            all_patterns_to_trim_before = sorted(
                single_words_to_trim_before + combined_patterns_to_trim_before,
                key=len,
                reverse=True
            )

            found_and_trimmed = False
            for pattern in all_patterns_to_trim_before:
                match = re.search(rf'(.*?){re.escape(pattern)}(.*)', theme_word)
                if match:
                    theme_word = match.group(2).strip()
                    found_and_trimmed = True
                    break
            
            if not theme_word: continue
            
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
    필요한 데이터베이스 테이블(theme_class, theme_stock, daily_theme, word_dic, theme_word_relevance)을 생성합니다.
    """
    print("\n--- 데이터베이스 테이블 초기화 시작 ---")
    
    # 1. theme_class 테이블 생성 (사용자 제공 스키마 반영)
    create_theme_class_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS `theme_class` (
            `theme_id` INT(11) NOT NULL AUTO_INCREMENT,
            `theme` VARCHAR(30) NOT NULL ,
            `theme_class` VARCHAR(30) NULL ,
            `theme_synonyms` JSON NULL ,
            `theme_hit` INT(11) NULL DEFAULT '0',
            `theme_score` INT(11) NULL DEFAULT '0',
            `momentum_score` DECIMAL(10,4) NULL DEFAULT '0.00',
            `theme_desc` VARCHAR(200) NULL DEFAULT NULL ,
            PRIMARY KEY (`theme_id`) USING BTREE,
            INDEX `idx_theme` (`theme`) USING BTREE
        );
    """)
    cursor.execute(create_theme_class_table_query)
    print("theme_class 테이블 존재 확인 및 필요시 생성 완료.")

    # 2. theme_stock 테이블 생성
    create_theme_stock_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS `theme_stock` (
            `theme_id` INT(11) NOT NULL,
            `stock_code` VARCHAR(7) NOT NULL ,
            `stock_score` DECIMAL(10,4) NULL DEFAULT '0.0000', -- DECIMAL로 변경
            PRIMARY KEY (`theme_id`, `stock_code`) USING BTREE,
            INDEX `idx_ts_theme` (`theme_id`) USING BTREE,
            INDEX `idx_ts_stock_code` (`stock_code`) USING BTREE
        );
    """)
    cursor.execute(create_theme_stock_table_query)
    print("theme_stock 테이블 존재 확인 및 필요시 생성 완료.")

    # 3. daily_theme 테이블 생성 (reason_nouns JSON 컬럼 포함)
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

    # 4. word_dic 테이블 생성
    create_word_dic_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS word_dic (
            word VARCHAR(25) NOT NULL,
            freq INT NULL DEFAULT NULL,
            cumul_rate DECIMAL(10,2) NULL DEFAULT NULL,
            avg_rate DECIMAL(10,2) NULL DEFAULT NULL,
            PRIMARY KEY (word)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    cursor.execute(create_word_dic_table_query)
    print("word_dic 테이블 존재 확인 및 필요시 생성 완료.")
    
    # 5. theme_word_relevance 테이블 생성
    create_theme_word_relevance_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS theme_word_relevance (
            theme VARCHAR(30) NOT NULL,
            word VARCHAR(25) NOT NULL,
            relevance_score DECIMAL(10,4) NULL DEFAULT 0,
            num_occurrences INT NULL DEFAULT 0,
            avg_stock_rate_in_theme DECIMAL(10,2) NULL DEFAULT 0,
            PRIMARY KEY (theme, word),
            INDEX idx_twr_theme (theme),
            INDEX idx_twr_word (word)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    cursor.execute(create_theme_word_relevance_table_query)
    print("theme_word_relevance 테이블 존재 확인 및 필요시 생성 완료.")

    connection.commit()
    print("--- 데이터베이스 테이블 초기화 완료 ---")

# --- 메인 스크립트 실행 ---
def main():
    connection = None
    cursor = None
    try:
        print("특징주 데이터 처리 스크립트를 시작합니다.")
        print(f"대상 폴더: '{datas_folder}'")
        
        # 'datas' 폴더와 'datas/done' 폴더 생성 확인 (있으면 무시)
        if not os.path.exists(datas_folder):
            os.makedirs(datas_folder)
            print(f"'{datas_folder}' 폴더가 생성되었습니다. 특징주 DOCX 파일을 여기에 넣어주세요.")
        if not os.path.exists(done_folder):
            os.makedirs(done_folder)
            print(f"'{done_folder}' 폴더가 생성되었습니다.")

        # --- DOCX 파일 처리 및 DataFrame 생성 ---
        print(f"\n--- '{datas_folder}' 폴더에서 특징주 DOCX 파일 검색 및 처리 시작 ---")
        
        all_features_dfs = []
        processed_dates = []
        successfully_processed_docx_paths = [] # DB 및 Excel 생성 후 이동할 파일 목록

        docx_files = [f for f in os.listdir(datas_folder) if f.endswith('.docx')]
        docx_files.sort() # 날짜 순으로 처리하기 위해 정렬 (파일명 기반)

        if not docx_files:
            print(f"'{datas_folder}' 폴더에서 처리할 '.docx' 파일을 찾을 수 없습니다.")
            print("특징주 DOCX 파일을 폴더에 넣어주세요.")
            return

        for filename in docx_files:
            print(f"\n--- '{filename}' 파일 처리 중 ---")
            file_path = os.path.join(datas_folder, filename)
            
            feature_date = parse_date_from_filename(filename)
            if feature_date is None:
                print(f"경고: '{filename}' 파일명에서 유효한 날짜를 파싱할 수 없습니다. 이 파일을 건너뜀.")
                continue

            processed_dates.append(feature_date)

            dfs = extract_tables_from_docx(file_path)

            if not dfs:
                print(f"경고: '{filename}' 파일에서 표를 찾을 수 없거나 처리 오류가 발생했습니다. 건너뜀.")
                continue

            current_df = dfs[0].copy() 

            # '날짜' 컬럼 추가
            current_df['날짜'] = feature_date
            
            # 컬럼명 변경 딕셔너리
            # 이 부분은 실제 DOCX 파일의 헤더와 매칭되도록 조정 필요
            column_rename_map = {
                '거래대금(백만)': '거래대금',
            }
            # 실제 컬럼만 필터링하여 rename 적용
            current_df.rename(columns={k: v for k, v in column_rename_map.items() if k in current_df.columns}, inplace=True)

            # 필요한 컬럼 존재 여부 확인 및 추가
            required_cols = ['날짜', '종목명', '등락률', '거래대금', '이유']
            for col in required_cols:
                if col not in current_df.columns:
                    print(f"경고: '{col}' 컬럼이 '{filename}' 파일의 표에서 누락되었습니다. 빈 값으로 추가합니다.")
                    current_df[col] = pd.NA

            # '구분' 컬럼 추가 (현재 표에 없으므로 '정규장'으로 초기화)
            if '구분' not in current_df.columns:
                current_df['구분'] = '정규장'
            
            # 컬럼 순서 조정
            desired_order_cols = ['날짜', '구분', '종목명', '등락률', '거래대금', '이유']
            # 기존 컬럼 중 desired_order_cols에 없는 컬럼들을 뒤에 붙이기
            remaining_cols = [col for col in current_df.columns if col not in desired_order_cols]
            current_df = current_df[desired_order_cols + remaining_cols]

            # 데이터 타입 변환
            cols_to_numeric = ['등락률'] # 저가 추가 및 등락률 포함
            for col in cols_to_numeric:
                if col in current_df.columns:
                    current_df[col] = current_df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True) # 숫자, 점, 마이너스만 남김
                    current_df[col] = pd.to_numeric(current_df[col], errors='coerce')
            
            # 거래대금은 별도로 처리: 문자열 정제 후 int로 변환
            if '거래대금' in current_df.columns:
                current_df['거래대금'] = current_df['거래대금'].astype(str).str.replace(r'[^\d]', '', regex=True)
                current_df['거래대금'] = pd.to_numeric(current_df['거래대금'], errors='coerce').fillna(0).astype(int) # 결측치는 0으로 채우고 int

            all_features_dfs.append(current_df)
            successfully_processed_docx_paths.append(file_path)
            print(f"'{filename}' 처리 완료. DataFrame에 추가됨.")

        if not all_features_dfs:
            print("\n처리할 유효한 특징주 DOCX 파일이 없어 최종 데이터를 생성할 수 없습니다.")
            return

        final_df = pd.concat(all_features_dfs, ignore_index=True)
        if '종목명' in final_df.columns:
            final_df['종목명'] = final_df['종목명'].astype(str).str.strip()

        
        print(f"\n--- 모든 특징주 데이터 합치기 완료. 총 {len(final_df)}개 행 ---")
        print(final_df.info())
        print(final_df.head())

        # --- 합쳐진 데이터를 Excel로 저장 (선택 사항) ---
        # if processed_dates:
        #     start_date_excel = min(processed_dates).strftime('%Y%m%d')
        #     end_date_excel = max(processed_dates).strftime('%Y%m%d')
        #     excel_filename = f"{start_date_excel}-{end_date_excel}특징주_통합.xlsx"
        # else:
        #     excel_filename = "특징주_통합_데이터.xlsx"

        # output_excel_path = os.path.join(output_excel_folder, excel_filename)
        # try:
        #     final_df.to_excel(output_excel_path, index=False, encoding='utf-8-sig')
        #     print(f"\n성공: 통합 특징주 데이터가 '{output_excel_path}' (으)로 저장되었습니다.")
        # except Exception as e:
        #     print(f"\n오류: 통합 Excel 파일 저장 중 문제가 발생했습니다: {e}")


        # --- MariaDB 연결 및 데이터 업데이트 ---
        connection = pymysql.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # 데이터베이스 테이블 초기화
        initialize_database_tables(cursor, connection)
        
        data_to_insert = []

        # 최종 DataFrame을 순회하며 DB에 삽입할 데이터 준비
        for index, row in final_df.iterrows():
            date_obj = row['날짜']
            market = str(row['구분'])
            stock_name = str(row['종목명'])
            rate = row['등락률']
            amount = row['거래대금']
            original_reason = str(row['이유'])

            if pd.isna(date_obj) or pd.isna(stock_name) or not stock_name.strip():
                print(f"경고: 필수 컬럼(날짜 또는 종목명) 누락 또는 빈 값. 레코드 건너뜀: {row}")
                continue

            # 등락률 및 거래대금 데이터 정제 (float, int 타입 확인)
            # 이미 DataFrame 생성 단계에서 숫자형으로 변환되었으나, 마지막 유효성 검사
            if pd.isna(rate):
                rate = Decimal('0.00') # Decimal 타입으로 통일
            else:
                try:
                    rate = Decimal(str(rate))
                except Exception:
                    rate = Decimal('0.00')

            if pd.isna(amount):
                amount = 0
            else:
                try:
                    amount = int(amount)
                except Exception:
                    amount = 0
                    
            # DB의 DECIMAL(5,2) 범위에 맞게 등락률 조정
            if rate > Decimal('999.99'):
                rate = Decimal('999.99')
            elif rate < Decimal('-999.99'):
                rate = Decimal('-999.99')

            # 종목코드를 stock_info 테이블에서 조회
            cursor.execute("SELECT stock_code FROM stock_info WHERE stock_name = %s", (stock_name,))
            stock_info_result = cursor.fetchone()

            if stock_info_result:
                stock_code = stock_info_result[0]
                
                final_reason_for_db = original_reason if original_reason else ""
                # 띄어쓰기 교정 (PyKoSpacing)
                if PYKOSPACING_ENABLED and final_reason_for_db:
                    try:
                        final_reason_for_db = spacing_tool(final_reason_for_db)
                    except Exception as e:
                        print(f"경고: 띄어쓰기 교정 오류 ('{original_reason[:30]}...'): {e}. 원본 이유 사용.")
                
                # 명사 추출 (Komoran) 및 JSON 변환
                reason_nouns_list = []
                if KONLPY_KOMORAN_ENABLED and final_reason_for_db:
                    try:
                        nouns = komoran_tool.nouns(final_reason_for_db)
                        # 2글자 이상만 남기고, 중복 제거 및 정렬
                        reason_nouns_list = sorted(list(set([n for n in nouns if len(n) >= 2])))
                    except Exception as e:
                        print(f"경고: 명사 추출 오류 ('{final_reason_for_db[:30]}...'): {e}. 빈 명사 목록 사용.")
                
                reason_nouns_json = json.dumps(reason_nouns_list, ensure_ascii=False) # 한글 인코딩 문제 방지

                # 테마 추출
                theme_str = extract_themes_simple(final_reason_for_db)
                
                data_to_insert.append((date_obj, market, stock_code, stock_name, rate, amount, final_reason_for_db, reason_nouns_json, theme_str))
            else:
                print(f"경고: '{stock_name}' 종목코드 미발견. 레코드 건너뜀.")
        
        # 데이터 일괄 삽입/업데이트
        if data_to_insert:
            start_time = time.time()
            replace_query = textwrap.dedent("""
                REPLACE INTO daily_theme (date, market, stock_code, stock_name, rate, amount, reason, reason_nouns, theme)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """)
            cursor.executemany(replace_query, data_to_insert)
            connection.commit()
            print_query_execution_time(start_time, time.time(), len(data_to_insert))
            print(f"총 {len(data_to_insert)}건의 데이터가 daily_theme 테이블에 성공적으로 업데이트되었습니다.")
        else:
            print("삽입/업데이트할 레코드 없음.")

        # --- 처리된 파일들을 'done' 폴더로 이동 ---
        print("\n--- 처리된 DOCX 파일 'done' 폴더로 이동 중 ---")
        move_processed_files_to_done(successfully_processed_docx_paths, done_folder)

    except pymysql.MySQLError as e:
        if connection: connection.rollback()
        print(f"MariaDB 오류: {e}")
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없음: {e}")
    except pd.errors.EmptyDataError:
        print(f"데이터 파일이 비어 있습니다.")
    except pd.errors.ParserError as e:
        print(f"데이터 파싱 오류: {e}")
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