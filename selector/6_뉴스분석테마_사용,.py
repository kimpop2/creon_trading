import pymysql
from konlpy.tag import Komoran
from pykospacing import Spacing
import re
import textwrap
import datetime
from collections import defaultdict
from time import time # 'time' 모듈에서 'time'을 임포트했는지 확인하세요

# Assume db_config and load_stock_names are defined as in your existing script
# You'd likely load stock_names_set once when your application starts.
# MariaDB connection configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'backtest_db', # Actual DB name
    'charset': 'utf8mb4'
}

def load_stock_names(cursor):
    """stock_info 테이블에서 모든 종목명(stock_name)을 로드합니다."""
    print("\n--- stock_info 테이블에서 종목명 로드 중 ---")
    start_time = time()  # 경과 시간 측정에는 time() 사용
    cursor.execute("SELECT stock_name FROM stock_info WHERE stock_name IS NOT NULL AND stock_name != ''")
    stock_names = {row[0] for row in cursor.fetchall()}
    end_time = time()
    print(f"총 {len(stock_names)}개 종목명 로드 완료 ({end_time - start_time:.4f} 초).")
    return stock_names

def print_metric_explanations():
    """분석 결과에 사용되는 주요 지표에 대한 설명을 출력합니다."""
    print("\n--- 분석 결과 용어 설명 ---")
    print("  * 테마 점수: 특정 키워드가 해당 테마와 얼마나 강하게 연결되어 있는지를 나타내는 점수입니다. (theme_word_relevance.relevance_score)")
    print("  * 테마 등락률: 해당 키워드를 포함하는 뉴스와 연관된 종목들이 속한 '특정 테마 내에서' 기록한 평균 등락률입니다. (theme_word_relevance.avg_stock_rate_in_theme)")
    print("  * 키워드 등락률: 특정 키워드가 나타난 '모든 종목의 모든 등락률을 평균'한 값입니다. (word_dic.avg_rate)")
    print("  * 출현 빈도: 특정 키워드가 전체 기간 동안 뉴스에 나타난 총 횟수입니다. (word_dic.freq)")
    print("----------------------------")

def process_breaking_news(news_text, db_connection, stock_names_set):
    komoran = Komoran()
    spacing_tool = Spacing()

    # 1단계: 속보 텍스트 처리
    print(f"\n--- 속보 분석 시작 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    print(f"뉴스 텍스트: {news_text[:100]}...") # 처음 100자 출력

    # 띄어쓰기 교정 적용
    corrected_news_text = spacing_tool(news_text) if spacing_tool else news_text
    print(f"교정된 텍스트: {corrected_news_text[:100]}...")

    # 명사 추출
    raw_nouns = komoran.nouns(corrected_news_text)
    
    # 명사 필터링
    filtered_keywords = []
    for noun in raw_nouns:
        if (len(noun) == 1 or
            re.match(r'^[-\d]', noun) or
            noun in stock_names_set):
            continue
        filtered_keywords.append(noun)
    
    if not filtered_keywords:
        print("뉴스에서 관련 키워드를 찾지 못했습니다.")
        return None

    print(f"추출 및 필터링된 키워드: {filtered_keywords}")

    # 2단계: theme_word_relevance 및 word_dic 쿼리
    try:
        cursor = db_connection.cursor(pymysql.cursors.DictCursor) # 컬럼 접근을 쉽게 하기 위해 DictCursor 사용
        
        # SQL 쿼리의 IN 절을 준비합니다.
        # 이렇게 하면 "'word1', 'word2', 'word3'"와 같은 문자열이 생성됩니다.
        keywords_sql_placeholder = ', '.join(['%s'] * len(filtered_keywords))

        query = textwrap.dedent(f"""
            SELECT
                twr.theme,
                twr.word,
                twr.relevance_score,
                twr.num_occurrences AS theme_word_occurrences,
                twr.avg_stock_rate_in_theme,
                wd.freq AS word_global_freq,
                wd.avg_rate AS word_global_avg_rate
            FROM
                theme_word_relevance twr
            JOIN
                word_dic wd ON twr.word = wd.word
            WHERE
                twr.word IN ({keywords_sql_placeholder})
            ORDER BY
                twr.relevance_score DESC, twr.theme, twr.word;
        """)
        
        cursor.execute(query, filtered_keywords)
        results = cursor.fetchall()
        
        if not results:
            print("이 키워드에 대한 관련 테마-단어 관계를 찾지 못했습니다.")
            return None
            
        # 용어 설명 출력
        print_metric_explanations()

        # 3단계: 결과 집계 및 해석
        print("\n--- 쿼리 결과 (상위 10개) ---")
        for i, row in enumerate(results[:10]):
            print(f"  {i+1}. 테마: {row['theme']}, 단어: {row['word']}, 테마 점수: {float(row['relevance_score']):.2f}, 테마 등락률: {float(row['avg_stock_rate_in_theme']):.2f}, 키워드 등락률: {float(row['word_global_avg_rate']):.2f}, 출현 빈도: {row['word_global_freq']}회")

        # 추가 분석 (예: 집계된 관련성 점수별 상위 3개 테마 식별)
        theme_scores = defaultdict(float)
        theme_details = defaultdict(lambda: {'total_relevance': 0.0, 'word_count': 0, 'avg_rate_sum': 0.0})

        for row in results:
            # Decimal.Decimal을 명시적으로 float으로 변환하여 더하기
            theme_scores[row['theme']] += float(row['relevance_score'])
            theme_details[row['theme']]['total_relevance'] += float(row['relevance_score'])
            theme_details[row['theme']]['word_count'] += 1
            theme_details[row['theme']]['avg_rate_sum'] += float(row['avg_stock_rate_in_theme'])

        sorted_themes = sorted(theme_scores.items(), key=lambda item: item[1], reverse=True)

        print("\n--- 뉴스에 영향을 받은 상위 테마 ---")
        for theme, score in sorted_themes[:5]: # 상위 5개 테마
            details = theme_details[theme]
            avg_theme_rate_overall = details['avg_rate_sum'] / details['word_count'] if details['word_count'] > 0 else 0
            print(f"  테마: {theme}, 테마 점수 합계: {score:.2f}, 테마(키워드) 등락률: {avg_theme_rate_overall:.2f}")

        print("\n--- 속보 분석 완료 ---")
        return results

    except pymysql.MySQLError as e:
        print(f"MariaDB 쿼리 오류: {e}")
        return None
    except Exception as e:
        print(f"뉴스 처리 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()

# --- Example Usage ---
if __name__ == "__main__":
    conn = None
    try:
        # DB 연결 설정
        conn = pymysql.connect(**db_config)
        cursor_for_setup = conn.cursor()

        # 중요: stock_info와 theme_stock이 채워져 있고 theme_word_relevance에 데이터가 있는지 확인하세요.
        # 이 테이블들을 채우려면 이전에 작성한 (통합된) 스크립트를 먼저 실행해야 할 수 있습니다.
        
        # 종목명 로드 (필터링에 필수)
        # 참고: stock_info 테이블에는 'code_name' 또는 'stock_name'이 있을 수 있습니다. 여기서는 정의에 따라 'stock_name'으로 가정합니다.
        stock_names_set = load_stock_names(cursor_for_setup)
        cursor_for_setup.close() # 설정 커서 닫기

        # 예시 속보
        breaking_news_item_1 = "삼성전자, 신형 파운드리 공정 개발 성공으로 반도체 시장 주도권 강화 예상."
        breaking_news_item_2 = "LG에너지솔루션, 전기차 배터리 생산량 확대 발표. 2차전지 관련주 급등."
        breaking_news_item_3 = "증시 전체적인 하락세 지속, 투자 심리 위축."
        breaking_news_item_4 = "엔비디아, 신형 AI 반도체 칩 출시로 자율주행 시장 선점 기대"
        breaking_news_item_5 = "원화 스테이블코인에 IT·게임 업계도 후끈... 네이버페이 대표도 \"주도적 역할 할 것\""

        print("\n--- 속보 항목 1 처리 중 ---")
        process_breaking_news(breaking_news_item_1, conn, stock_names_set)

        print("\n--- 속보 항목 2 처리 중 ---")
        process_breaking_news(breaking_news_item_2, conn, stock_names_set)

        print("\n--- 속보 항목 3 처리 중 ---")
        process_breaking_news(breaking_news_item_3, conn, stock_names_set)

        print("\n--- 속보 항목 4 처리 중 ---")
        process_breaking_news(breaking_news_item_4, conn, stock_names_set)

        print("\n--- 속보 항목 5 처리 중 ---")
        process_breaking_news(breaking_news_item_5, conn, stock_names_set)

    except pymysql.MySQLError as e:
        print(f"MariaDB 연결 오류: {e}")
    except Exception as e:
        print(f"메인 실행 중 예상치 못한 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            print("MariaDB 연결이 종료되었습니다.")