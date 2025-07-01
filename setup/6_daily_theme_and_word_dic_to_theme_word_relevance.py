import pandas as pd
import pymysql
from konlpy.tag import Komoran
from collections import defaultdict
import os.path
import sys
import re
import textwrap
import time
import datetime

# --- 설정 ---
# MariaDB 연결 설정
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'backtest_db',
    'charset': 'utf8mb4'
}

def print_processing_summary(start_time, end_time, total_items, process_name=""):
    """처리 시간, 항목 수, 그리고 항목당 실행 시간을 출력하는 함수"""
    processing_time = end_time - start_time
    time_per_item = processing_time / total_items if total_items > 0 else 0
    print(f"[{process_name}] 총 처리 시간: {processing_time:.4f} 초")
    print(f"[{process_name}] 처리된 항목 수: {total_items} 개")
    print(f"[{process_name}] 항목 1개당 실행 시간: {time_per_item:.6f} 초")
    print(f"[{process_name}] 작업 완료.")

def initialize_database_tables(cursor, connection):
    """필요한 데이터베이스 테이블(theme_stock, daily_theme, theme_word_relevance)을 생성합니다."""
    # NOTE: theme_class, word_dic 테이블 생성 로직은 다른 모듈에서 처리한다고 가정하여 제거했습니다.
    print("\n--- 데이터베이스 테이블 초기화 시작 ---")
    
    # 1. theme_stock 테이블 생성 (기존 로직 유지)

    # 2. theme_stock 테이블 생성 (존재하지 않을 경우)
    # 5. theme_word_relevance 테이블 생성 (NEW)
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

def load_stock_names(cursor):
    """stock_info 테이블에서 모든 종목명(stock_name)을 로드합니다."""
    print("\n--- stock_info 테이블에서 종목명 로드 중 ---")
    start_time = time.time()
    cursor.execute("SELECT stock_name FROM stock_info WHERE stock_name IS NOT NULL AND stock_name != ''")
    stock_names = {row[0] for row in cursor.fetchall()}
    end_time = time.time()
    print(f"총 {len(stock_names)}개 종목명 로드 완료 ({end_time - start_time:.4f} 초).")
    return stock_names

def load_theme_stock_mapping(cursor):
    """theme_stock 테이블에서 종목 코드와 테마 매핑을 로드합니다."""
    print("\n--- theme_stock 테이블에서 종목-테마 매핑 로드 중 ---")
    start_time = time.time()
    query = textwrap.dedent("""
        SELECT 
            ts.stock_code, 
            tc.theme AS theme_name
        FROM 
            theme_stock ts
        JOIN 
            theme_class tc ON ts.theme_id = tc.theme_id;
    """)
    cursor.execute(query)
    
    stock_to_themes = defaultdict(list)
    for stock_code, theme in cursor.fetchall():
        stock_to_themes[stock_code].append(theme)
    
    end_time = time.time()
    print(f"총 {len(stock_to_themes)}개 종목의 테마 매핑 로드 완료 ({end_time - start_time:.4f} 초).")
    return stock_to_themes


def calculate_and_update_theme_word_relevance(cursor, connection, stock_names_set, stock_to_themes_map,
                                               theme_word_relevance_min_occurrences=1, data_period_days=40): # data_period_days 파라미터 추가
    """
    daily_theme 테이블에서 데이터를 읽어 명사를 추출하고,
    theme_stock 매핑을 활용하여 theme_word_relevance 테이블을 계산하고 업데이트합니다.
    
    Args:
        cursor: MariaDB 커서 객체.
        connection: MariaDB 연결 객체.
        stock_names_set (set): stock_info에서 로드된 종목명 집합. (단어 필터링용)
        stock_to_themes_map (dict): theme_stock에서 로드된 종목 코드-테마 매핑.
        theme_word_relevance_min_occurrences (int): theme_word_relevance에 저장할 (테마, 단어) 쌍의 최소 발생 빈도.
        data_period_days (int): daily_theme 테이블에서 데이터를 로드할 기간 (단위: 일).
    """
    
    print(f"\n--- theme_word_relevance 계산 및 업데이트 시작 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    start_total_time = time.time()

    try:
        # --- 1. Load Data from daily_theme table with date filtering ---
        start_date = datetime.date.today() - datetime.timedelta(days=data_period_days)
        print(f"daily_theme 테이블에서 {start_date} 이후의 데이터 로드 중...")
        
        cursor.execute("""
            SELECT date, stock_code, rate, reason 
            FROM daily_theme
            WHERE date >= %s
        """, (start_date,))
        daily_theme_records = cursor.fetchall()
        total_records_loaded = len(daily_theme_records)
        print(f"daily_theme 테이블에서 총 {total_records_loaded}개 레코드 로드.")

        if not daily_theme_records:
            print("daily_theme 테이블에 유효한 데이터가 없어 theme_word_relevance 계산을 진행할 수 없습니다.")
            return False

        # --- 2. Initialize NLP Tools ---
        komoran = Komoran()
        
        # Data structure for theme_word_relevance aggregation
        # {theme: {word: {'count': N, 'sum_rate': R}}}
        theme_word_aggr = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'sum_rate': 0.0}))

        processed_record_count = 0
        
        # --- 3. Process Each Record from daily_theme Table ---
        print("daily_theme 레코드 분석 및 테마-단어 관련성 집계 중...")
        start_processing_records_time = time.time()

        for dt_date, dt_stock_code, dt_rate, dt_reason in daily_theme_records:
            current_reason = str(dt_reason).strip()
            
            # 'rate' 유효성 검사 및 필터링
            try:
                rate = float(dt_rate) 
                if rate > 30.0: # 30% 초과 등락률은 분석에서 제외
                    continue
            except (ValueError, TypeError):
                continue

            nouns = komoran.nouns(current_reason)
            
            # Get themes associated with this stock code
            associated_themes = stock_to_themes_map.get(dt_stock_code, [])

            if not associated_themes: # 연결된 테마가 없으면 건너뜀
                continue

            for noun in nouns:
                # 단어 내 공백 제거 (이전 요청 반영)
                cleaned_noun = noun.replace(" ", "") 
                
                # 필터링 조건: 한 글자 명사 제외, 숫자/음수로 시작하는 명사 제외, 종목명과 일치하는 명사 제외
                if (len(cleaned_noun) == 1 or
                    re.match(r'^[-\d]', cleaned_noun) or
                    cleaned_noun in stock_names_set):
                    continue

                # --- For theme_word_relevance ---
                for theme in associated_themes:
                    theme_word_aggr[theme][cleaned_noun]['count'] += 1
                    theme_word_aggr[theme][cleaned_noun]['sum_rate'] += rate

            processed_record_count += 1
        
        end_processing_records_time = time.time()
        print_processing_summary(start_processing_records_time, end_processing_records_time,
                                 processed_record_count, "테마-단어 관련성 데이터 집계")

        # --- 4. Prepare data for theme_word_relevance update ---
        data_to_update_theme_word_relevance = []
        for theme, words_data in theme_word_aggr.items():
            for word, metrics in words_data.items():
                num_occurrences = metrics['count']
                sum_rate = metrics['sum_rate']

                if num_occurrences < theme_word_relevance_min_occurrences: # Filter by minimum occurrences
                    continue
                
                avg_stock_rate_in_theme = sum_rate / num_occurrences
                
                # relevance_score 계산식
                # 예시: 빈도수 * (1 + 평균 등락률 절댓값 / 100)
                relevance_score = num_occurrences * (1 + abs(avg_stock_rate_in_theme / 100.0))
                
                data_to_update_theme_word_relevance.append((
                    theme, 
                    word, 
                    round(relevance_score, 4), # Round to 4 decimal places for DECIMAL(10,4)
                    num_occurrences, 
                    round(avg_stock_rate_in_theme, 2) # Round to 2 decimal places for DECIMAL(10,2)
                ))
        
        print(f"theme_word_relevance 업데이트를 위한 최종 데이터 {len(data_to_update_theme_word_relevance)}개 준비 완료.")

        # --- 5. Update/Insert into theme_word_relevance table ---
        if data_to_update_theme_word_relevance:
            print("\n--- theme_word_relevance 테이블 업데이트/삽입 시작 ---")
            start_db_update_time = time.time()
            # 기존 데이터를 모두 지우고 새로 삽입 (혹은 REPLACE INTO 사용)
            # 여기서는 명시적으로 TRUNCATE 후 INSERT를 제안합니다.
            # TRUNCATE TABLE은 매우 빠르게 테이블의 모든 데이터를 삭제합니다.
            cursor.execute("TRUNCATE TABLE theme_word_relevance")
            connection.commit()
            print("기존 theme_word_relevance 데이터 삭제 완료.")

            insert_theme_word_relevance_query = textwrap.dedent("""
                INSERT INTO theme_word_relevance (theme, word, relevance_score, num_occurrences, avg_stock_rate_in_theme)
                VALUES (%s, %s, %s, %s, %s)
            """)
            cursor.executemany(insert_theme_word_relevance_query, data_to_update_theme_word_relevance)
            connection.commit()
            end_db_update_time = time.time()
            print_processing_summary(start_db_update_time, end_db_update_time,
                                     len(data_to_update_theme_word_relevance), "theme_word_relevance DB 업데이트")
        else:
            print("theme_word_relevance 테이블에 업데이트/삽입할 데이터가 없습니다.")
            # 데이터가 없으면 기존 테이블을 비워둡니다.
            cursor.execute("TRUNCATE TABLE theme_word_relevance")
            connection.commit()
            print("theme_word_relevance 테이블이 비어있으므로 기존 데이터 삭제 완료.")


        end_total_time = time.time()
        print_processing_summary(start_total_time, end_total_time, total_records_loaded, "전체 theme_word_relevance 계산 및 업데이트")
        return True

    except pymysql.MySQLError as e:
        connection.rollback()
        print(f"MariaDB 오류 발생: {e}")
        return False
    except Exception as e:
        connection.rollback() 
        print(f"예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Main execution block ---
if __name__ == "__main__":
    conn = None
    try:
        # Establish DB connection
        conn = pymysql.connect(**db_config)
        cur = conn.cursor()

        # 1. Initialize DB tables (will now create theme_word_relevance if not exists)
        # Note: We keep `initialize_database_tables` for self-contained execution,
        # but it now only checks/creates `theme_stock`, `daily_theme`, `theme_word_relevance`.
        # `theme_class`, `word_dic` creation is assumed to be handled elsewhere.
        initialize_database_tables(cur, conn)

        # 2. Load stock names for filtering (still needed for noun filtering)
        stock_names = load_stock_names(cur)

        # 3. Load theme-stock mapping
        stock_to_themes = load_theme_stock_mapping(cur)

        # 4. Calculate and update theme_word_relevance
        success = calculate_and_update_theme_word_relevance(
            cur, conn,
            stock_names,
            stock_to_themes,
            theme_word_relevance_min_occurrences=4, # Minimum occurrences for (theme, word) pair in theme_word_relevance
            data_period_days=40 # <-- 이 부분 추가: word_dic과 동일하게 6개월 (180일)로 기간 제한
        )

        if not success:
            print("theme_word_relevance 계산 및 DB 업데이트 작업이 실패했거나 완료되지 않았습니다.")

        # NOTE: 금요일 클리닝 로직은 이 모듈의 목적(theme_word_relevance 계산)과 직접적인 관련이 없으므로 제거됩니다.
        # 이 클리닝 로직은 word_dic 관리 모듈에 있어야 합니다.

    except pymysql.MySQLError as e:
        print(f"MariaDB 연결 오류: {e}")
    except Exception as e:
        print(f"전체 스크립트 실행 중 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            cur.close()
            conn.close()
            print("MariaDB 연결 해제.")