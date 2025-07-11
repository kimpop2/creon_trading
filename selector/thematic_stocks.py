# daily_thematic_runner_modify.py

import pymysql
import datetime
from collections import defaultdict
from time import time
import re, sys, textwrap
import json # JSON 파싱을 위해 추가
# from konlpy.tag import Komoran # 제거됨
# from pykospacing import Spacing # 제거됨

# --- 설정 ---
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'backtest_db',
    'charset': 'utf8mb4'
}

# 잠재적 신규 테마/단어 식별을 위한 임계값
NEW_WORD_MIN_FREQ = 10      # 신규 단어 후보로 고려할 word_dic의 최소 빈도
NEW_WORD_MIN_AVG_RATE = 3 # 신규 단어 후보의 최소 평균 등락률 절댓값
NEW_WORD_MIN_THEME_ASSOCIATIONS = 3 # '신규'로 간주하기 위한 기존 테마와의 최대 연관 수

# --- 유틸리티 함수 (공유 모듈에 없는 경우 이전 스크립트에서 복사) ---
def print_processing_summary(start_time, end_time, total_items, process_name=""):
    """처리 시간, 항목 수, 그리고 항목당 실행 시간을 출력하는 함수"""
    processing_time = end_time - start_time
    time_per_item = processing_time / total_items if total_items > 0 else 0
    print(f"[{process_name}] 총 처리 시간: {processing_time:.4f} 초")
    print(f"[{process_name}] 처리된 항목 수: {total_items} 개")
    print(f"[{process_name}] 항목 1개당 실행 시간: {time_per_item:.6f} 초")
    print(f"[{process_name}] 작업 완료.")

# initialize_database_tables 함수 제거됨: 이 프로그램은 테이블 추가/업데이트를 수행하지 않음

def load_stock_names(cursor):
    """stock_info 테이블에서 모든 종목명(stock_name)을 로드합니다."""
    print("\n--- stock_info 테이블에서 종목명 로드 중 ---")
    start_time = time()
    cursor.execute("SELECT stock_name FROM stock_info WHERE stock_name IS NOT NULL AND stock_name != ''")
    stock_names = {row[0] for row in cursor.fetchall()}
    end_time = time()
    print(f"총 {len(stock_names)}개 종목명 로드 완료 ({end_time - start_time:.4f} 초).")
    return stock_names

def load_theme_stock_mapping(cursor):
    """
    theme_stock 테이블에서 종목 코드와 가장 세분화된 테마(theme_class) 매핑을 로드합니다.
    theme_stock의 'theme_id'와 theme_class.theme_id를 조인하여 theme_class 이름을 가져옵니다.
    """
    print("\n--- theme_stock 테이블에서 종목-세부테마 매핑 로드 중 ---")
    start_time = time()
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
    
    stock_to_sub_themes = defaultdict(list)
    for stock_code, sub_theme in cursor.fetchall():
        stock_to_sub_themes[stock_code].append(sub_theme)
    
    end_time = time()
    print(f"총 {len(stock_to_sub_themes)}개 종목의 세부테마 매핑 로드 완료 ({end_time - start_time:.4f} 초).")
    return stock_to_sub_themes

def process_daily_theme_data(cursor, connection, stock_names_set, stock_to_themes_map,
                             data_period_days=40): # data_period_days 파라미터 유지
    """
    daily_theme 테이블에서 데이터를 읽어 명사를 추출하여 theme_class momentum_score 계산에 필요한 데이터를 집계합니다.
    이 함수는 daily_theme, word_dic, theme_word_relevance 테이블을 직접 업데이트하지 않습니다.
    단지 theme_class 테이블의 momentum_score 컬럼만 업데이트합니다.
    """
    print(f"\n--- daily_theme DB 데이터 분석 및 theme_class momentum_score 집계 시작 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    start_total_time = time()

    try:
        # 직접 매칭 우선순위를 위해 모든 기존 theme_class 이름 로드
        cursor.execute("SELECT theme FROM theme_class")
        all_theme_class_names = {row[0] for row in cursor.fetchall()}

        # --- 1. daily_theme 테이블에서 데이터 로드 (기간 제한 적용) ---
        start_date = datetime.date.today() - datetime.timedelta(days=data_period_days)
        print(f"daily_theme 테이블에서 {start_date} 이후의 데이터 로드 중...")

        # reason_nouns 컬럼 추가
        cursor.execute("""
            SELECT date, market, stock_code, stock_name, rate, reason, theme, reason_nouns
            FROM daily_theme
            WHERE date >= %s
        """, (start_date,))
        daily_theme_records = cursor.fetchall()
        total_records_loaded = len(daily_theme_records)
        print(f"daily_theme 테이블에서 총 {total_records_loaded}개 레코드 로드.")

        if not daily_theme_records:
            print("daily_theme 테이블에 유효한 데이터가 없어 분석을 진행할 수 없습니다.")
            return False

        # Komoran 및 Spacing 관련 객체 제거

        # word_dic 관련 집계는 남겨둠 (momentum_score 및 신규 테마 식별에 간접적으로 사용될 수 있으므로)
        word_dic_noun_counts = defaultdict(int)
        word_dic_noun_weighted_sum = defaultdict(float) 

        # theme_word_relevance 관련 집계는 남겨둠 (momentum_score 계산에 사용)
        theme_word_aggr = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'sum_rate': 0.0}))
        
        processed_reason_count = 0
        
        print("daily_theme 테이블 각 레코드의 'reason_nouns' 컬럼 분석 중...")
        start_processing_records_time = time()

        for dt_date, dt_market, dt_stock_code, dt_stock_name, dt_rate, dt_reason, dt_processed_themes, dt_reason_nouns in daily_theme_records:
            # Komoran 명사 추출 대신 reason_nouns 컬럼 사용
            parsed_reason_nouns = []
            if dt_reason_nouns: # reason_nouns 컬럼이 비어있지 않은지 확인
                try:
                    # reason_nouns는 JSON 문자열 형태의 명사 리스트로 가정
                    parsed_reason_nouns = json.loads(dt_reason_nouns)
                except json.JSONDecodeError as e:
                    print(f"경고: reason_nouns JSON 파싱 오류 for {dt_stock_code}, {dt_date}: {e}")
                    parsed_reason_nouns = []
            
            all_nouns_for_processing = set(parsed_reason_nouns)

            # daily_theme.theme 컬럼에서 콤마로 구분된 대표 테마명들을 파싱
            processed_themes_for_entry = [t.strip() for t in str(dt_processed_themes).split(',') if t.strip()]

            # 등락률 유효성 검사
            try:
                rate = float(dt_rate)
                if rate > 30.0: # 30% 초과 등락률은 분석에서 제외
                    continue
            except (ValueError, TypeError):
                continue

            associated_themes_for_stock = stock_to_themes_map.get(dt_stock_code, [])

            # theme_word_relevance용 집계 로직 변경
            # 이제 associated_themes_for_stock 대신 processed_themes_for_entry를 사용
            for noun in all_nouns_for_processing:
                # word_dic 집계는 기존과 동일 (reason_nouns에서 추출된 모든 명사에 대해)
                word_dic_noun_counts[noun] += 1
                word_dic_noun_weighted_sum[noun] += rate

                # theme_word_relevance 집계: 해당 daily_theme 레코드에 연결된 (미리 처리된) 테마들을 사용
                if processed_themes_for_entry:
                    for theme_name_from_daily_theme_column in processed_themes_for_entry:
                        theme_word_aggr[theme_name_from_daily_theme_column][noun]['count'] += 1
                        theme_word_aggr[theme_name_from_daily_theme_column][noun]['sum_rate'] += rate
            processed_reason_count += 1
        
        end_processing_records_time = time()
        print_processing_summary(start_processing_records_time, end_processing_records_time,
                                 processed_reason_count, "reason_nouns 분석 및 집계")

        # theme_class momentum_score 계산 및 업데이트
        theme_momentum_updates = []
        for theme_class_name, words_data in theme_word_aggr.items():
            # 수정된 momentum_score 계산 로직: 빈도와 평균 등락률 절댓값의 곱을 합산
            momentum_score_sum = 0.0
            for word, metrics in words_data.items():
                if metrics['count'] > 0:
                    avg_rate_for_word_in_theme = metrics['sum_rate'] / metrics['count']
                    # 각 단어의 '등장 횟수'와 '평균 등락률의 절댓값'을 곱하여 합산
                    momentum_score_sum += metrics['count'] * abs(avg_rate_for_word_in_theme)
            
            # 최종 momentum_score로 사용 (원하는 경우 정규화 가능)
            momentum_score = momentum_score_sum 
            
            # 단어가 전혀 없는 테마는 점수 업데이트에서 제외
            if momentum_score > 0: 
                theme_momentum_updates.append((round(momentum_score, 4), theme_class_name))

        if theme_momentum_updates:
            print("\n--- theme_class momentum_score 업데이트 시작 ---")
            update_momentum_query = textwrap.dedent("""
                UPDATE theme_class
                SET momentum_score = %s
                WHERE theme = %s
            """)
            cursor.executemany(update_momentum_query, theme_momentum_updates)
            connection.commit()
            print_processing_summary(0, 0, len(theme_momentum_updates), "theme_class momentum_score 업데이트")
        else:
            print("theme_class momentum_score 업데이트할 내용이 없습니다.")

        end_total_time = time()
        print_processing_summary(start_total_time, end_total_time, total_records_loaded, "전체 DB 데이터 분석 및 theme_class momentum_score 업데이트")
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


def identify_potential_new_themes(cursor):
    """
    word_dic에서 높은 영향력을 가지지만 기존 테마와의 연관성이 낮은 단어를 찾아 새로운 테마 후보로 제시합니다.
    이 함수는 word_dic과 theme_word_relevance 테이블이 이미 업데이트되어 있다고 가정하고 조회합니다.
    """
    print("\n--- 새로운 테마 후보 식별 시작 ---")
    start_time = time()

    # 높은 빈도/평균 등락률 단어를 위해 word_dic 쿼리
    query_word_dic = textwrap.dedent(f"""
        SELECT word, freq, avg_rate
        FROM word_dic
        WHERE freq >= %s AND ABS(avg_rate) >= %s
    """)
    cursor.execute(query_word_dic, (NEW_WORD_MIN_FREQ, NEW_WORD_MIN_AVG_RATE))
    high_impact_words = cursor.fetchall()

    potential_new_themes = []
    if high_impact_words:
        # theme_word_relevance에서 기존 테마와의 연관성 확인
        cursor_dict = cursor.connection.cursor(pymysql.cursors.DictCursor) 
        
        for word, freq, avg_rate in high_impact_words:
            query_theme_association = textwrap.dedent("""
                SELECT COUNT(DISTINCT theme) AS distinct_themes_count
                FROM theme_word_relevance
                WHERE word = %s
            """)
            cursor_dict.execute(query_theme_association, (word,))
            
            result = cursor_dict.fetchone()
            distinct_themes_count = result['distinct_themes_count'] if result else 0

            if distinct_themes_count <= NEW_WORD_MIN_THEME_ASSOCIATIONS:
                potential_new_themes.append({
                    'word': word,
                    'global_freq': freq,
                    'global_avg_rate': avg_rate,
                    'existing_theme_count': distinct_themes_count
                })
        cursor_dict.close()

    end_time = time()
    print_processing_summary(start_time, end_time, len(potential_new_themes), "새로운 테마 후보 식별")
    
    if potential_new_themes:
        print("\n--- 새로운 테마/키워드 후보 (트레이더 검토 필요) ---")
        for candidate in potential_new_themes:
            print(f"    단어: {candidate['word']}, 전체 빈도: {candidate['global_freq']}, 평균 등락률: {candidate['global_avg_rate']:.2f}, 기존 테마 연관 수: {candidate['existing_theme_count']}")
        print("위 단어들을 검토하여 theme_class에 추가하거나 기존 테마를 강화할지 결정하십시오.")
    else:
        print("새로운 테마/키워드 후보를 찾지 못했습니다.")

    return potential_new_themes


def get_actionable_insights(cursor, limit_themes=5, limit_stocks_per_theme=3):
    """
    최신 모멘텀 점수를 기준으로 상위 테마와 그에 속한 관련 종목을 추천합니다.
    """
    print("\n--- 행동 가능한 통찰력 생성 시작 ---")
    start_time = time()

    # 1. momentum_score 기준으로 상위 N개 테마 가져오기
    # theme_id와 대표 테마명(theme)을 함께 가져옵니다.
    query_top_themes = textwrap.dedent("""
        SELECT theme_id, theme, momentum_score 
        FROM theme_class
        ORDER BY momentum_score DESC
        LIMIT %s
    """)
    cursor.execute(query_top_themes, (limit_themes,))
    # top_themes는 이제 (theme_id, theme_name, momentum_score) 튜플을 포함합니다.
    top_themes = cursor.fetchall() 

    actionable_results = []
    if not top_themes:
        print("모멘텀이 있는 테마를 찾을 수 없습니다.")
        return []

    # 2. 각 상위 테마에 대해 관련 상위 종목 찾기
    # 루프에서 theme_id와 대표 테마명을 직접 사용합니다.
    for theme_id_val, display_theme_name, momentum_score in top_themes:
        # 출력 메시지에서 비어있는 'theme_class' 대신 실제 대표 테마명 사용
        print(f"\n--- 테마: {display_theme_name} (모멘텀 스코어: {momentum_score:.2f}) ---")
        
        # 테마(theme_id)와 관련된 종목 찾기
        query_theme_stocks = textwrap.dedent("""
            SELECT ts.stock_code, si.stock_name AS stock_name, ts.stock_score
            FROM theme_stock ts
            JOIN stock_info si ON ts.stock_code = si.stock_code
            WHERE ts.theme_id = %s  -- 서브쿼리 대신 직접 theme_id 사용
            ORDER BY ts.stock_score DESC
            LIMIT %s
        """)
        # 쿼리 실행 시 해당 테마의 theme_id를 파라미터로 전달
        cursor.execute(query_theme_stocks, (theme_id_val, limit_stocks_per_theme))
        related_stocks = cursor.fetchall() # (stock_code, stock_name, stock_score)

        theme_result = {
            'theme_id': theme_id_val,         # 테마 ID 저장 (필요시)
            'theme_class': display_theme_name, # 'theme_class' 키에 대표 테마명 저장
            'broad_theme': display_theme_name, # 'broad_theme' 키에도 동일하게 저장 (광의/세부 구분이 필요없다면)
            'momentum_score': momentum_score,
            'recommended_stocks': []
        }

        if related_stocks:
            for stock_code, stock_name, stock_score in related_stocks:
                # 선택사항: 이 종목들의 최근 daily_theme 데이터를 가져와서 당일 성과 확인
                query_recent_daily_theme = textwrap.dedent("""
                    SELECT rate, reason
                    FROM daily_theme
                    WHERE stock_code = %s
                    ORDER BY date DESC
                    LIMIT 1
                """)
                cursor.execute(query_recent_daily_theme, (stock_code,))
                recent_data = cursor.fetchone()

                recent_rate = float(recent_data[0]) if recent_data and recent_data[0] else 0.0
                recent_reason = recent_data[1] if recent_data and recent_data[1] else "N/A"

                print(f"    - 종목: {stock_name} ({stock_code}), 테마 기여 점수: {stock_score}, 최근 등락률: {recent_rate:.2f}%, \n      이유: {recent_reason}")
                theme_result['recommended_stocks'].append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'stock_score': stock_score,
                    'recent_rate': recent_rate,
                    'recent_reason': recent_reason
                })
        else:
            print(f"    - 이 테마 ({display_theme_name})에 연결된 종목을 찾을 수 없습니다.") # 출력 메시지 변경
        
        actionable_results.append(theme_result)

    end_time = time()
    print_processing_summary(start_time, end_time, len(top_themes), "행동 가능한 통찰력 생성")
    return actionable_results

# 배치 러너를 위한 새로운 메인 호출 가능 함수
def run_daily_analysis_and_get_actionable_insights(conn, data_period_days=40): # data_period_days 파라미터 추가
    cur = conn.cursor() # 배치 러너에서 전달된 연결 사용

    # 0단계: 테이블 초기화 로직 제거됨.
    # 이 스크립트 실행 전에 필요한 모든 DB 테이블이 생성되어 있고, 
    # word_dic 및 theme_word_relevance가 다른 모듈에 의해 업데이트되었다고 가정합니다.

    # 1단계: 처리에 필요한 필수 데이터 로드
    stock_names_set = load_stock_names(cur)
    stock_to_themes_map = load_theme_stock_mapping(cur)

    # 2단계: 일별 데이터 수집 시뮬레이션
    # 이 부분은 실제 데이터 소스에 따라 구현해야 합니다.
    # 이 스크립트의 경우 시장 데이터 스크래퍼를 실행한 후 `daily_theme`에 오늘의 관련 데이터가 이미 포함되어 있다고 가정합니다.
    print("\n--- 일별 daily_theme 데이터 수집 (외부 소스 연동 필요) ---")
    print("daily_theme 테이블에 당일 데이터가 업데이트되었다고 가정합니다.")

    # 3단계: daily_theme 데이터를 기반으로 theme_class momentum_score를 위한 핵심 처리 실행 
    success = process_daily_theme_data(
        cur, conn,
        stock_names_set,
        stock_to_themes_map,
        data_period_days=data_period_days # 외부에서 받은 data_period_days를 전달
    )

    if not success:
        print("데이터 처리 및 DB 업데이트 작업이 실패했습니다.")
        cur.close()
        return [] # 실패 시 빈 리스트 반환

    # 4단계: 잠재적 신규 테마/키워드 식별 (콘솔에 출력)
    # 이 기능은 정보를 '읽어' 추천 후보를 제시하므로 유지합니다.
    identify_potential_new_themes(cur)

    # 5단계: 행동 가능한 통찰력 생성 및 반환
    actionable_results = get_actionable_insights(cur, limit_themes=5, limit_stocks_per_theme=10) # 충분한 후보 가져오기

    cur.close() # 이 함수에서 열린 커서 닫기
    return actionable_results

if __name__ == "__main__":
    # 이 블록은 이제 주로 이 특정 스크립트의 독립 실행 테스트를 위해 사용됩니다
    conn = None
    # 테스트를 위한 data_period_days 설정 (원하는 값으로 변경 가능)
    # 예를 들어, 최근 30일 데이터만 보려면 `test_data_period_days = 30`으로 변경
    test_data_period_days = 40 # 사용자의 요청에 따라 20일로 설정

    try:
        conn = pymysql.connect(**db_config)
        results = run_daily_analysis_and_get_actionable_insights(conn, data_period_days=test_data_period_days) # 기간 인수를 전달
        print("\n--- 독립 실행 daily_thematic_runner 테스트 결과 ---")
        if results:
            for res in results:
                print(f"\n테마: {res['theme_class']} (모멘텀: {res['momentum_score']:.2f})")
                for stock in res['recommended_stocks']:
                    print(f"    - {stock['stock_name']} ({stock['stock_code']}), 등락률: {stock['recent_rate']:.2f}%")
        else:
            print("추천할 테마나 종목이 없습니다.")

    except Exception as e:
        print(f"독립 실행 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            print("독립 실행 테스트를 위한 MariaDB 연결 종료.")