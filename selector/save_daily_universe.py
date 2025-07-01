# save_daily_universe.py

import pymysql
import datetime
import textwrap
from time import time

# --- 설정 (db_config는 사용 가능하거나 전달된다고 가정) ---
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'backtest_db',
    'charset': 'utf8mb4'
}

# --- 유틸리티 함수 (이전 스크립트의 print_processing_summary) ---
def print_processing_summary(start_time, end_time, total_items, process_name=""):
    processing_time = end_time - start_time
    time_per_item = processing_time / total_items if total_items > 0 else 0
    print(f"[{process_name}] 총 처리 시간: {processing_time:.4f} 초")
    print(f"[{process_name}] 처리된 항목 수: {total_items} 개")
    print(f"[{process_name}] 항목 1개당 실행 시간: {time_per_item:.6f} 초")
    print(f"[{process_name}] 작업 완료.")

def save_daily_stock_universe(connection, recommended_themes_and_stocks, target_num_stocks=100):
    """
    제공된 추천 테마 및 종목 정보를 바탕으로 daily_stock_universe 테이블에 매매 대상 종목을 저장합니다.
    target_num_stocks에 맞춰 상위 종목들을 선정합니다.
    """
    print(f"\n--- daily_stock_universe 저장 시작 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    start_time = time()
    cursor = connection.cursor()

    # YYYYMMDD 형식으로 오늘 날짜 가져오기
    today_str = datetime.datetime.now().strftime('%Y%m%d')

    # daily_stock_universe 테이블이 존재하는지 확인
    # 이 DDL은 아직 없다면 메인 DB 초기화의 일부여야 합니다
    create_table_ddl = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS daily_stock_universe (
            date VARCHAR(8) NOT NULL,
            stock_code VARCHAR(6) NOT NULL,
            stock_name VARCHAR(50) NOT NULL,
            broad_theme VARCHAR(50) NULL,
            theme_class VARCHAR(50) NULL,
            theme_momentum_score DECIMAL(10,4) DEFAULT 0,
            stock_theme_score DECIMAL(10,4) DEFAULT 0,
            daily_rate DECIMAL(6,2) DEFAULT 0,
            daily_reason_summary VARCHAR(250) NULL,
            selection_reason VARCHAR(200) NULL,
            rank_overall INT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, stock_code),
            KEY idx_theme_class (theme_class)
        );
    """)
    cursor.execute(create_table_ddl)
    connection.commit()
    print("daily_stock_universe 테이블 존재 확인 및 필요시 생성 완료.")

    # 전체 순위를 위한 점수 메커니즘으로 모든 후보 종목 준비
    all_candidate_stocks = []

    # 추천 테마와 그 종목들을 반복
    # recommended_themes_and_stocks는 get_actionable_insights의 출력입니다
    # 이는 딕셔너리 리스트이며, 각 딕셔너리는 'theme_class', 'broad_theme', 'momentum_score', 'recommended_stocks'를 가집니다
    
    # 간단한 점수 로직: theme_momentum_score * stock_theme_score * (1 + daily_rate/100 if positive, else 1)
    # 이는 높은 모멘텀 테마, 높은 테마 기여 종목, 긍정적인 일일 성과를 우선시합니다.
    for theme_data in recommended_themes_and_stocks:
        theme_class = theme_data['theme_class']
        broad_theme = theme_data['broad_theme']
        theme_momentum_score = float(theme_data['momentum_score']) # float 변환 보장

        for stock_info in theme_data['recommended_stocks']:
            stock_code = stock_info['stock_code']
            stock_name = stock_info['stock_name']
            stock_theme_score = float(stock_info['stock_score']) # float 변환 보장
            daily_rate = float(stock_info['recent_rate']) # float 변환 보장
            daily_reason_summary = stock_info['recent_reason']

            # 각 종목에 대한 전체 순위 점수 계산
            overall_stock_score = (
                theme_momentum_score * stock_theme_score * (1 + (daily_rate / 100.0 if daily_rate > 0 else 0)) # 긍정적인 등락률에 대한 부스트
            )
            # daily_rate 영향에 상한을 두거나 음수 등락률을 다르게 처리하고 싶을 수 있습니다
            # 단순화를 위해 음수 등락률은 부스트하지 않지만 여기서는 페널티도 주지 않습니다.
            # 원한다면 페널티를 추가할 수 있습니다: (1 + daily_rate / 100.0)

            all_candidate_stocks.append({
                'date': today_str,
                'stock_code': stock_code,
                'stock_name': stock_name,
                'broad_theme': broad_theme,
                'theme_class': theme_class,
                'theme_momentum_score': theme_momentum_score,
                'stock_theme_score': stock_theme_score,
                'daily_rate': daily_rate,
                'daily_reason_summary': daily_reason_summary,
                'selection_reason': f"높은 모멘텀 테마 ({theme_class}) 관련 종목", # 기본 이유
                'overall_score': overall_stock_score
            })
    
    # 전체 점수 기준으로 내림차순 정렬
    all_candidate_stocks.sort(key=lambda x: x['overall_score'], reverse=True)

    # 상위 N개 종목 선택
    selected_stocks_for_universe = all_candidate_stocks[:target_num_stocks]
    
    if not selected_stocks_for_universe:
        print("선정된 매매 대상 종목이 없습니다.")
        return

    # 배치 삽입을 위한 데이터 준비
    insert_data = []
    for rank, stock_data in enumerate(selected_stocks_for_universe):
        insert_data.append((
            stock_data['date'],
            stock_data['stock_code'],
            stock_data['stock_name'],
            stock_data['broad_theme'],
            stock_data['theme_class'],
            stock_data['theme_momentum_score'],
            stock_data['stock_theme_score'],
            stock_data['daily_rate'],
            stock_data['daily_reason_summary'],
            stock_data['selection_reason'],
            rank + 1 # 순위는 1부터 시작
        ))

    # 새 항목 삽입 전에 오늘 날짜의 기존 항목 삭제 (재실행 처리용)
    delete_today_query = "DELETE FROM daily_stock_universe WHERE date = %s"
    cursor.execute(delete_today_query, (today_str,))
    connection.commit()
    print(f"오늘 ({today_str})의 기존 daily_stock_universe 레코드 {cursor.rowcount}개 삭제 완료.")


    insert_query = textwrap.dedent("""
        INSERT INTO daily_stock_universe (
            date, stock_code, stock_name, broad_theme, theme_class,
            theme_momentum_score, stock_theme_score, daily_rate,
            daily_reason_summary, selection_reason, rank_overall
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """)

    cursor.executemany(insert_query, insert_data)
    connection.commit()
    
    end_time = time()
    print_processing_summary(start_time, end_time, len(selected_stocks_for_universe), "daily_stock_universe 저장")
    print(f"최종 선정된 매매 대상 종목 수: {len(selected_stocks_for_universe)}개.")


if __name__ == "__main__":
    # 사용 예시:
    # 이 부분은 일반적으로 메인 배치 스크립트에서 호출됩니다.
    # 독립 실행 시 recommended_themes_and_stocks를 위한 목업 데이터가 필요할 수 있습니다.
    print("save_daily_universe.py는 일반적으로 배치 스크립트의 일부로 실행됩니다.")
    print("독립 실행 시 테스트를 위한 목업 데이터가 필요할 수 있습니다.")
    
    # 테스트용 목업 데이터 (get_actionable_insights의 실제 출력으로 대체)
    mock_recommended_themes_and_stocks = [
        {
            'theme_class': 'HBM', 
            'broad_theme': '반도체', 
            'momentum_score': 85.50, 
            'recommended_stocks': [
                {'stock_code': '000660', 'stock_name': 'SK하이닉스', 'stock_score': 0.95, 'recent_rate': 7.20, 'recent_reason': 'HBM3E 양산 성공, 엔비디아 공급 확대.'},
                {'stock_code': '005930', 'stock_name': '삼성전자', 'stock_score': 0.88, 'recent_rate': 3.10, 'recent_reason': '차세대 HBM 개발 가속화 소식.'},
                {'stock_code': '003550', 'stock_name': 'LG', 'stock_score': 0.70, 'recent_rate': 1.50, 'recent_reason': '반도체 장비 관련 자회사 수혜 기대.'},
            ]
        },
        {
            'theme_class': '로보택시', 
            'broad_theme': '자율주행', 
            'momentum_score': 78.20, 
            'recommended_stocks': [
                {'stock_code': '000270', 'stock_name': '기아', 'stock_score': 0.92, 'recent_rate': 5.80, 'recent_reason': '로보택시 시범 서비스 추진 계획 발표.'},
                {'stock_code': '005380', 'stock_name': '현대차', 'stock_score': 0.85, 'recent_rate': 4.50, 'recent_reason': '자율주행 랩스 투자 확대 소식.'},
            ]
        },
        # target_num_stocks 테스트를 위해 더 많은 목업 테마/종목 추가
    ]

    try:
        conn = pymysql.connect(**db_config)
        save_daily_stock_universe(conn, mock_recommended_themes_and_stocks, target_num_stocks=10) # 10개 종목으로 테스트
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