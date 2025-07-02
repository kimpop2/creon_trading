import pandas as pd
import pymysql
# from konlpy.tag import Komoran # Komoran 모듈 사용 중지
from collections import defaultdict
import os.path
import sys
import re
import textwrap
import time
import datetime
import json # JSON 데이터 처리를 위해 추가

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
    """쿼리 실행 시간, 레코드 건수, 레코드당 실행 시간을 출력하는 함수"""
    processing_time = end_time - start_time
    time_per_item = processing_time / total_items if total_items > 0 else 0
    print(f"[{process_name}] 총 처리 시간: {processing_time:.4f} 초")
    print(f"[{process_name}] 처리된 항목 수: {total_items} 개")
    print(f"[{process_name}] 항목 1개당 실행 시간: {time_per_item:.6f} 초")
    print(f"[{process_name}] 작업 완료.")

def initialize_database_tables(cursor, connection):
    """필요한 데이터베이스 테이블(theme_class, theme_stock, daily_theme, word_dic)을 생성합니다."""
    print("\n--- 데이터베이스 테이블 초기화 시작 ---")
    
    # 1. theme_class 테이블 생성
    create_theme_class_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS theme_class (
            theme_id VARCHAR(36) NOT NULL,
            theme VARCHAR(30) NULL DEFAULT NULL,
            PRIMARY KEY (theme_id),
            UNIQUE INDEX theme_UNIQUE (theme)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    cursor.execute(create_theme_class_table_query)
    print("theme_class 테이블 존재 확인 및 필요시 생성 완료.")

    # 2. theme_stock 테이블 생성
    create_theme_stock_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS theme_stock (
            theme_id VARCHAR(36) NOT NULL,
            stock_code VARCHAR(7) NOT NULL,
            stock_score DECIMAL(10,4) DEFAULT 0.0000,
            PRIMARY KEY (theme_id, stock_code),
            INDEX idx_ts_stock_code (stock_code),
            INDEX idx_ts_theme_id (theme_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    cursor.execute(create_theme_stock_table_query)
    print("theme_stock 테이블 존재 확인 및 필요시 생성 완료.")

    # 3. daily_theme 테이블 생성 (reason_nouns 컬럼이 JSON 타입으로 존재함을 가정)
    create_daily_theme_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS daily_theme (
            date DATE NOT NULL,
            market VARCHAR(8) NOT NULL,
            stock_code VARCHAR(7) NOT NULL,
            stock_name VARCHAR(25) NOT NULL,
            rate DECIMAL(5,2) NULL DEFAULT '0.00',
            amount INT(11) NULL DEFAULT '0',
            reason VARCHAR(250) NOT NULL,
            reason_nouns JSON NULL, -- JSON 타입 컬럼
            theme VARCHAR(250) NULL DEFAULT NULL,
            PRIMARY KEY (date, market, stock_code) USING BTREE
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

def process_daily_theme_from_db_and_update_word_dic(cursor, connection, stock_names_set,
                                                     data_period_days=90, # 추가: word_dic 계산에 사용할 최근 데이터 기간 (일수)
                                                     freq_threshold=0, avg_rate_threshold=1):
    """
    daily_theme 테이블에서 데이터를 읽어 미리 추출된 명사를 활용하여 word_dic 테이블에 누적/업데이트합니다.
    'reason' 컬럼의 맞춤법 교정 로직은 제거되었으며, 추출된 명사 내의 공백이 제거됩니다.
    
    Args:
        cursor: MariaDB 커서 객체.
        connection: MariaDB 연결 객체.
        stock_names_set (set): stock_info에서 로드된 종목명 집합.
        data_period_days (int): word_dic 계산에 사용할 daily_theme 데이터의 최근 기간 (일수).
        freq_threshold (int): word_dic에 저장할 명사의 최소 빈도.
        avg_rate_threshold (int): word_dic에 저장할 명사의 최소 평균 등락률 절댓값.
    """
    
    print(f"\n--- daily_theme DB 데이터 분석 및 word_dic 업데이트 시작 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    start_total_time = time.time()

    try:
        # --- 1. Load Data from daily_theme table (기간 필터링 적용) ---
        print("daily_theme 테이블에서 데이터 로드 중...")
        # data_period_days에 따라 시작 날짜 계산
        start_date = datetime.date.today() - datetime.timedelta(days=data_period_days)
        print(f"조회 기간: {start_date} 부터 현재까지 ({data_period_days}일 데이터)")

        # reason_nouns 컬럼을 함께 조회하도록 쿼리 수정
        cursor.execute("""
            SELECT date, market, stock_code, stock_name, rate, reason_nouns, theme
            FROM daily_theme
            WHERE date >= %s
        """, (start_date,))
        daily_theme_records = cursor.fetchall() 
        total_records_loaded = len(daily_theme_records)
        print(f"daily_theme 테이블에서 총 {total_records_loaded}개 레코드 로드.")

        if not daily_theme_records:
            print("daily_theme 테이블에 유효한 데이터가 없어 분석을 진행할 수 없습니다.")
            return False

        # Komoran 초기화 코드 제거
        
        noun_counts = defaultdict(int)
        noun_weighted_sum = defaultdict(float) 
        
        processed_reason_count = 0
        
        # --- 2. Process Each Record from daily_theme Table ---
        print("daily_theme 테이블 각 레코드의 'reason_nouns' 컬럼 명사 추출 중 (사전 추출된 명사 사용)...")
        start_processing_records_time = time.time()

        for dt_date, dt_market, dt_stock_code, dt_stock_name, dt_rate, dt_reason_nouns_json, dt_existing_theme in daily_theme_records:
            
            # 'rate' 유효성 검사 및 필터링
            try:
                rate = float(dt_rate) 
                if rate > 30.0: # 30% 초과 등락률은 분석에서 제외
                    continue
            except (ValueError, TypeError):
                # rate가 유효하지 않은 경우 스킵
                continue

            # reason_nouns JSON 컬럼에서 명사 리스트 파싱
            nouns = []
            if dt_reason_nouns_json:
                try:
                    nouns = json.loads(dt_reason_nouns_json)
                    if not isinstance(nouns, list): # JSON이 리스트 형태가 아닐 경우 대비
                        nouns = []
                except json.JSONDecodeError:
                    print(f"경고: reason_nouns JSON 파싱 오류. 레코드 건너김: {dt_reason_nouns_json[:50]}...")
                    continue
            
            if not nouns: # 추출된 명사가 없으면 건너뜀
                continue
            
            for noun in nouns:
                # --- 추가/수정된 로직: 단어 내의 공백 제거 ---
                cleaned_noun = str(noun).strip().replace(" ", "") # 파싱된 명사도 문자열로 변환 및 공백 제거
                
                # 필터링 조건: 한 글자 명사 제외, 숫자/음수로 시작하는 명사 제외, 종목명과 일치하는 명사 제외
                # 이제 cleaned_noun을 사용하여 필터링 및 통계 계산
                if (len(cleaned_noun) == 1 or
                    re.match(r'^[-\d]', cleaned_noun) or
                    cleaned_noun in stock_names_set):
                    continue

                noun_counts[cleaned_noun] += 1
                noun_weighted_sum[cleaned_noun] += rate
            processed_reason_count += 1
        
        end_processing_records_time = time.time()
        print_processing_summary(start_processing_records_time, end_processing_records_time,
                                 processed_reason_count, "명사 추출 및 통계 계산")

        # --- 3. Prepare data for word_dic update ---
        data_to_update_word_dic = []
        for noun, count in noun_counts.items():
            cumul_rate = noun_weighted_sum[noun]
            avg_rate = cumul_rate / count

            # 최종 필터링 조건 적용 (min_freq, min_avg_rate)
            if count > freq_threshold and abs(avg_rate) > avg_rate_threshold:
                # DB에 저장할 때 DECIMAL(10,2)에 맞춰 소수점 2자리까지 반올림
                data_to_update_word_dic.append((noun, count, round(cumul_rate, 2), round(avg_rate, 2)))

        print(f"word_dic 업데이트를 위한 최종 데이터 {len(data_to_update_word_dic)}개 준비 완료.")

        # --- 4. Update/Insert into word_dic table ---
        if data_to_update_word_dic:
            print("\n--- word_dic 테이블 업데이트/삽입 시작 ---")
            start_db_update_time = time.time()
            replace_word_dic_query = textwrap.dedent("""
                REPLACE INTO word_dic (word, freq, cumul_rate, avg_rate)
                VALUES (%s, %s, %s, %s)
            """)
            cursor.executemany(replace_word_dic_query, data_to_update_word_dic)
            connection.commit()
            end_db_update_time = time.time()
            print_processing_summary(start_db_update_time, end_db_update_time,
                                     len(data_to_update_word_dic), "word_dic DB 업데이트")
        else:
            print("word_dic 테이블에 업데이트/삽입할 데이터가 없습니다.")

        end_total_time = time.time()
        print_processing_summary(start_total_time, end_total_time, total_records_loaded, "전체 daily_theme DB 분석 및 word_dic 업데이트")
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

# --- word_dic_cleaner.py 의 clean_word_dic 함수를 여기에 통합 ---
# 이 함수는 금요일에만 호출됩니다.
def clean_word_dic(connection):
    print(f"\n--- word_dic 클리닝 시작 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    start_time = time.time()
    cursor = connection.cursor()

    # --- 클리닝 임계값 (clean_word_dic 함수 내부로 이동하여 독립성 유지) ---
    MIN_GLOBAL_FREQ = 5        # 이 값보다 적게 나타나는 단어는 제거됩니다
    MAX_GLOBAL_FREQ_PERCENTILE = 0.99 # 이 빈도 백분위수 이상의 단어는 제거될 수 있습니다 (상위 1%)
    MAX_COMMON_WORD_AVG_RATE_DEVIATION = 0.5 # 예: avg_rate가 -0.5와 +0.5 사이이고 빈도가 매우 높은 경우
    BLACKLIST_WORDS = ['기자', '사진', '뉴시스', '연합뉴스', '머니투데이', '코스피', '코스닥', '지수', '시장', '증시', '개장', '폐장', '마감', '시가총액'] # 발견되는 대로 더 추가하세요


    try:
        # 1단계: 수동 블랙리스트의 단어들 제거
        if BLACKLIST_WORDS:
            placeholders = ', '.join(['%s'] * len(BLACKLIST_WORDS))
            delete_blacklist_query = f"DELETE FROM word_dic WHERE word IN ({placeholders})"
            cursor.execute(delete_blacklist_query, BLACKLIST_WORDS)
            print(f"블랙리스트 단어 {cursor.rowcount}개 제거 완료.")
            connection.commit()

        # 2단계: 매우 낮은 빈도 단어 제거 (예: 오타, 정말 관련 없는 단어들)
        delete_low_freq_query = f"DELETE FROM word_dic WHERE freq < %s"
        cursor.execute(delete_low_freq_query, (MIN_GLOBAL_FREQ,))
        print(f"최소 빈도 {MIN_GLOBAL_FREQ} 미만 단어 {cursor.rowcount}개 제거 완료.")
        connection.commit()

        # 3단계: 매우 높은 빈도이지만 낮은 영향력을 가진 단어 식별 및 제거 (잠재적 불용어)
        # 먼저 총 빈도를 계산하여 백분위수 방법을 사용하는 경우 결정
        cursor.execute("SELECT SUM(freq) FROM word_dic")
        total_freq = cursor.fetchone()[0]
        if total_freq is None or total_freq == 0:
            print("word_dic이 비어있거나 데이터가 없어 고빈도 단어 처리를 건너낍니다.")
            return

        # 백분위수에 기반한 고빈도 임계값 계산 (또는 고정값 사용)
        cursor.execute("SELECT word, freq, avg_rate FROM word_dic ORDER BY freq DESC")
        all_words_sorted_by_freq = cursor.fetchall()

        # 상위 X%의 빈도 임계값 찾기
        freq_cutoff_index = int(len(all_words_sorted_by_freq) * (1 - MAX_GLOBAL_FREQ_PERCENTILE))
        if freq_cutoff_index >= len(all_words_sorted_by_freq): 
            freq_cutoff_index = len(all_words_sorted_by_freq) - 1
        
        # 해당 절단점에서의 빈도 값 결정
        high_freq_threshold = all_words_sorted_by_freq[freq_cutoff_index][1] if all_words_sorted_by_freq else 0
        
        # 이 빈도 이상의 단어들에 대해 avg_rate가 0에 가까운지 확인
        deleted_high_freq_count = 0
        words_to_delete = []
        for word, freq, avg_rate in all_words_sorted_by_freq:
            # avg_rate를 float으로 명시적으로 변환하여 비교
            if freq >= high_freq_threshold and abs(float(avg_rate)) < MAX_COMMON_WORD_AVG_RATE_DEVIATION:
                words_to_delete.append(word)
        
        if words_to_delete:
            placeholders = ', '.join(['%s'] * len(words_to_delete))
            delete_high_freq_query = f"DELETE FROM word_dic WHERE word IN ({placeholders})"
            cursor.execute(delete_high_freq_query, words_to_delete)
            deleted_high_freq_count = cursor.rowcount
            connection.commit()
        print(f"최고 빈도 & 낮은 영향력 단어 {deleted_high_freq_count}개 제거 완료.")
        connection.commit()

    except pymysql.MySQLError as e:
        connection.rollback()
        print(f"MariaDB 오류 발생 중 word_dic 클리닝: {e}")
    except Exception as e:
        connection.rollback()
        print(f"예상치 못한 오류 발생 중 word_dic 클리닝: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        end_time = time.time()
        print(f"--- word_dic 클리닝 완료 ({end_time - start_time:.4f} 초) ---")


# --- Main execution block ---
if __name__ == "__main__":
    conn = None
    try:
        # Establish DB connection
        conn = pymysql.connect(**db_config)
        cur = conn.cursor()

        # 1. Initialize DB tables
        initialize_database_tables(cur, conn)

        # 2. Load stock names for filtering
        stock_names = load_stock_names(cur)

        # 3. Process daily_theme directly from DB and update word_dic
        # word_dic 계산 기간을 최근 6개월(180일)로 설정
        success = process_daily_theme_from_db_and_update_word_dic(
            cur, conn,
            stock_names,
            data_period_days=90, # word_dic 계산에 사용할 기간을 90일로 설정 (원래 스크립트와 동일)
            freq_threshold=0,
            avg_rate_threshold=1
        )

        if not success:
            print("데이터 처리 및 DB 업데이트 작업이 실패했거나 완료되지 않았습니다.")
        
        # --- 추가된 로직: 오늘이 금요일인 경우에만 word_dic 클리닝 수행 ---
        today = datetime.date.today()
        # weekday() 함수는 월요일을 0으로, 일요일을 6으로 반환합니다. 금요일은 4입니다.
        if today.weekday() == 4: # 4는 금요일을 의미합니다.
            print(f"\n[알림] 오늘은 금요일이므로 word_dic 클리닝을 시작합니다.")
            clean_word_dic(conn)
        else:
            print(f"\n[알림] 오늘은 금요일이 아니므로 word_dic 클리닝을 건너킵니다. (오늘 요일: {today.weekday()})")

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