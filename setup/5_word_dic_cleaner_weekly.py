# word_dic_cleaner.py

import pymysql
from time import time
import datetime

# --- 설정 ---
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'backtest_db',
    'charset': 'utf8mb4'
}

# --- 클리닝 임계값 ---
MIN_GLOBAL_FREQ = 5       # 이 값보다 적게 나타나는 단어는 제거됩니다
MAX_GLOBAL_FREQ_PERCENTILE = 0.95 # 이 빈도 백분위수 이상의 단어는 제거될 수 있습니다 (상위 1%)
                                  # 또는 고정 숫자를 사용할 수도 있습니다, 예: MAX_GLOBAL_FREQ_ABS = 100000

# 선택사항: 고빈도, 낮은 영향력 단어 제거를 위한 임계값
MAX_COMMON_WORD_AVG_RATE_DEVIATION = 0.5 # 예: avg_rate가 -0.5와 +0.5 사이이고 빈도가 매우 높은 경우

# 수동 블랙리스트 (항상 제거할 단어들)
BLACKLIST_WORDS = ['기자', '사진', '뉴시스', '연합뉴스', '머니투데이', '코스피', '코스닥', '지수', '시장', '증시', '개장', '폐장', '마감', '시가총액'] # 발견되는 대로 더 추가하세요

def clean_word_dic(connection):
    print(f"\n--- word_dic 클리닝 시작 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    start_time = time()
    cursor = connection.cursor()

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
            print("word_dic이 비어있거나 데이터가 없어 고빈도 단어 처리를 건너뜁니다.")
            return

        # 백분위수에 기반한 고빈도 임계값 계산 (또는 고정값 사용)
        cursor.execute("SELECT word, freq, avg_rate FROM word_dic ORDER BY freq DESC")
        all_words_sorted_by_freq = cursor.fetchall()

        # 상위 X%의 빈도 임계값 찾기
        freq_cutoff_index = int(len(all_words_sorted_by_freq) * (1 - MAX_GLOBAL_FREQ_PERCENTILE))
        if freq_cutoff_index >= len(all_words_sorted_by_freq): # 작은 데이터셋 처리
            freq_cutoff_index = len(all_words_sorted_by_freq) - 1
        
        # 해당 절단점에서의 빈도 값 결정
        high_freq_threshold = all_words_sorted_by_freq[freq_cutoff_index][1] if all_words_sorted_by_freq else 0
        
        # 이 빈도 이상의 단어들에 대해 avg_rate가 0에 가까운지 확인
        deleted_high_freq_count = 0
        for word, freq, avg_rate in all_words_sorted_by_freq:
            if freq >= high_freq_threshold and abs(float(avg_rate)) < MAX_COMMON_WORD_AVG_RATE_DEVIATION:
                cursor.execute("DELETE FROM word_dic WHERE word = %s", (word,))
                deleted_high_freq_count += 1
        print(f"최고 빈도 & 낮은 영향력 단어 {deleted_high_freq_count}개 제거 완료.")
        connection.commit()

        # 4단계: (선택사항 - 더 고급) theme_word_relevance 재평가
        # word_dic에서 단어가 제거되면 theme_word_relevance에서도 제거해야 할 수 있습니다
        # 또는 관련성 점수를 재계산해야 할 수 있습니다.
        # 이 부분은 변경사항을 연쇄적으로 적용하거나 주기적으로 process_daily_theme_data를
        # 다시 실행하는 것에만 의존하는 더 복잡한 로직이 필요합니다.
        # 시간이 지나면서 자연스럽게 관련 없는 단어들이 제거될 것입니다.
        # 현재로서는 클리닝 후 process_daily_theme_data를 실행하면
        # theme_word_relevance가 자연스럽게 정리될 것이라고 가정합니다.

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
        end_time = time()
        print(f"--- word_dic 클리닝 완료 ({end_time - start_time:.4f} 초) ---")

# --- 메인 실행 ---
if __name__ == "__main__":
    conn = None
    try:
        conn = pymysql.connect(**db_config)
        clean_word_dic(conn)
    except pymysql.MySQLError as e:
        print(f"MariaDB 연결 오류: {e}")
    except Exception as e:
        print(f"스크립트 실행 중 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            print("MariaDB 연결 해제.")