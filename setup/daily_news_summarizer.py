# daily_news_summarizer.py

import pymysql
import datetime
from collections import defaultdict
import time
import os
import textwrap

# --- 설정 ---
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'backtest_db',
    'charset': 'utf8mb4'
}

# 생성형 AI API 설정
# 중요: 선택한 AI 모델의 실제 API 키와 엔드포인트로 교체하세요
# Google Gemini API의 경우 다음과 같이 사용합니다:
# import google.generativeai as genai
# genai.configure(api_key=os.environ.get("GEMINI_API_KEY")) # 또는 직접 API 키
# model = genai.GenerativeModel('gemini-pro')

# 이 의사 코드에서는 AI 호출을 위한 플레이스홀더 함수를 사용합니다.
# 선택한 AI 서비스에 따라 실제 API 호출을 구현해야 합니다.
# 데모 목적으로 실제 API가 설정되지 않은 경우 간단한 목업 AI 응답을 사용합니다.

# AI 요약 매개변수
MAX_SUMMARY_CHARS = 250
AI_API_CALL_DELAY_SECONDS = 2 # API 속도 제한에 따라 조정 (예: 호출당 2초)
NEWS_RETENTION_MONTHS = 2

# --- 헬퍼 함수 (이전 스크립트와 유사하게 적절히 정의되었다고 가정) ---
def print_processing_summary(start_time, end_time, total_items, process_name=""):
    # ... (이전과 동일) ...
    pass

# 실제 AI 요약 함수를 위한 플레이스홀더
def summarize_text_with_ai(news_titles_list):
    """
    생성형 AI API에 연결하여 뉴스 제목 목록을 요약합니다.
    요약된 텍스트(str)를 반환하거나 실패 시 None을 반환합니다.
    """
    combined_titles = "\n".join(news_titles_list)
    prompt = f"다음 뉴스 제목들을 {MAX_SUMMARY_CHARS}자 이내의 한 문장으로 요약해 줘:\n{combined_titles}"

    print(f"  [AI] 요약 요청: {combined_titles[:80]}... (총 {len(combined_titles)}자)")

    try:
        # --- 이 섹션을 실제 AI API 호출로 교체하세요 ---
        # Google Gemini 예시:
        # response = model.generate_content(prompt)
        # summary = response.text.strip()

        # 의사 코드 데모를 위한 목업 AI 응답
        time.sleep(AI_API_CALL_DELAY_SECONDS) # API 호출 지연 시뮬레이션
        mock_summary = f"[{news_titles_list[0][:20]} 등] 관련 뉴스가 발생했으며, 이는 주식 시장에 큰 영향을 미칠 것으로 예상됩니다."
        if len(mock_summary) > MAX_SUMMARY_CHARS:
             mock_summary = mock_summary[:MAX_SUMMARY_CHARS-3] + "..." # 너무 길면 잘라냄

        summary = mock_summary
        # --- AI API 호출 섹션 끝 ---

        return summary
    except Exception as e:
        print(f"  [AI Error] 요약 중 오류 발생: {e}")
        return None

# --- 메인 로직 ---
def run_daily_news_summarizer(connection):
    print(f"\n--- 일별 뉴스 요약 및 daily_theme 업데이트 시작 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    start_total_time = time.time()
    cursor = connection.cursor()

    try:
        # 1. daily_theme에서 최신 시장 날짜 결정
        # 이는 daily_theme이 이미 주식 등락률 정보로 채워져 있다고 가정합니다.
        cursor.execute("SELECT MAX(date) FROM daily_theme")
        latest_market_date_str = cursor.fetchone()[0]
        if not latest_market_date_str:
            print("daily_theme에 최신 날짜 데이터가 없습니다. 요약 프로세스를 건너_니다.")
            return

        print(f"최신 시장 데이터 날짜: {latest_market_date_str}")
        
        # 2. 최신 시장 날짜와 관련 종목에 대한 뉴스 제목 검색
        # daily_theme과 조인하여 일일 시장 데이터가 있는 종목의 뉴스만 가져옵니다.
        query_raw_news = textwrap.dedent("""
            SELECT dn.stock_code, dn.news_title
            FROM daily_news dn
            JOIN daily_theme dt ON dn.date = dt.date AND dn.stock_code = dt.stock_code
            WHERE dn.date = %s
            ORDER BY dn.stock_code, dn.seq ASC;
        """)
        cursor.execute(query_raw_news, (latest_market_date_str,))
        raw_news_records = cursor.fetchall()

        if not raw_news_records:
            print(f"날짜 {latest_market_date_str} 에 요약할 뉴스 데이터가 없습니다.")
            return

        # stock_code별로 뉴스 그룹화
        news_by_stock = defaultdict(list)
        for stock_code, news_title in raw_news_records:
            news_by_stock[stock_code].append(news_title)
        
        print(f"총 {len(news_by_stock)}개 종목에 대한 뉴스 요약 진행 예정.")

        summarized_count = 0
        failed_summaries = 0
        for stock_code, titles_list in news_by_stock.items():
            print(f"종목 {stock_code}의 뉴스 {len(titles_list)}개 요약 중...")
            summarized_reason = summarize_text_with_ai(titles_list)
            
            if summarized_reason:
                # 요약된 이유로 daily_theme 업데이트
                update_query = textwrap.dedent("""
                    UPDATE daily_theme
                    SET reason = %s
                    WHERE date = %s AND stock_code = %s;
                """)
                cursor.execute(update_query, (summarized_reason, latest_market_date_str, stock_code))
                summarized_count += 1
                print(f"  -> 종목 {stock_code} 요약 완료.")
            else:
                failed_summaries += 1
                print(f"  -> 종목 {stock_code} 요약 실패. 기존 이유 유지.")
            
            # 더 나은 진행 상황 추적/복구를 위해 각 종목 또는 배치 후 커밋
            connection.commit() 
            
        print(f"\n총 {summarized_count}개 종목 뉴스 요약 완료, {failed_summaries}개 실패.")

        # 3. 오래된 daily_news 데이터 정리
        cutoff_date = datetime.date.today() - datetime.timedelta(days=NEWS_RETENTION_MONTHs * 30)
        print(f"daily_news 테이블에서 {cutoff_date} 이전 데이터 삭제 중...")
        delete_old_news_query = "DELETE FROM daily_news WHERE date < %s"
        cursor.execute(delete_old_news_query, (cutoff_date.strftime('%Y%m%d'),))
        print(f"오래된 뉴스 {cursor.rowcount}개 삭제 완료.")
        connection.commit()

    except pymysql.MySQLError as e:
        connection.rollback()
        print(f"MariaDB 오류 발생 중 뉴스 요약: {e}")
    except Exception as e:
        connection.rollback()
        print(f"예상치 못한 오류 발생 중 뉴스 요약: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        end_total_time = time.time()
        print_processing_summary(start_total_time, end_total_time, summarized_count, "일별 뉴스 요약")

# --- 메인 실행 ---
if __name__ == "__main__":
    conn = None
    try:
        conn = pymysql.connect(**db_config)
        # daily_news 테이블이 존재하는지 확인 (아직 없다면 initialize_database_tables에서 정의해야 함)
        # CREATE TABLE daily_news (
        #    seq INT AUTO_INCREMENT PRIMARY KEY,
        #    date VARCHAR(8) NOT NULL,
        #    news_title VARCHAR(500) NOT NULL,
        #    stock_code VARCHAR(6) NOT NULL,
        #    theme_class VARCHAR(50), -- 선택사항: 초기 태깅용
        #    media VARCHAR(50),
        #    news_url VARCHAR(255),
        #    KEY idx_date_stock (date, stock_code)
        # );

        run_daily_news_summarizer(conn)

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