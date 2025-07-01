# daily_batch_runner.py

import subprocess
import sys
import datetime
import time

# --- 설정 ---
# 각 스크립트의 경로를 정확히 지정해주세요.
# 이 스크립트와 동일한 디렉토리에 있다면 파일명만 적어도 됩니다.
DAILY_NEWS_SUMMARIZER_SCRIPT = 'daily_news_summarizer.py'
DAILY_THEMATIC_RUNNER_SCRIPT = 'daily_thematic_runner.py'
SAVE_DAILY_UNIVERSE_SCRIPT = 'save_daily_universe.py' # 이 스크립트에서는 함수를 직접 임포트할 예정

# --- 데이터베이스 설정 (직접 임포트 및 함수 호출용) ---
# db_config가 daily_thematic_runner.py에 정의되어 있거나 전달될 수 있다고 가정합니다.
# 공통 설정을 통합하거나 환경 변수/설정 파일을 통해 전달하는 것이 가장 좋습니다.
# 단순화를 위해, 스크립트가 이를 허용한다면 db_config를 임포트합니다.
# 또는 각 스크립트가 자체적으로 내부 db_config를 사용하여 DB 연결을 관리한다고 가정합니다.

# --- 더 나은 제어와 데이터 전달을 위한 핵심 함수 임포트 ---
# subprocess 호출을 피하고 직접 데이터 전달을 가능하게 하려면
# daily_thematic_runner.py와 save_daily_universe.py의 주요 로직을 함수로 리팩터링하여 노출하는 것이 좋습니다 (예: run_thematic_analysis, save_universe).
# 현재로서는 특정 함수를 임포트할 수 있다고 가정합니다.
# NameError 또는 임포트 문제가 발생하면 아래의 subprocess.run 방식을 사용하세요.

# run_daily_thematic_analysis와 save_daily_stock_universe 함수가 있다고 가정한 구조 예시
# from daily_thematic_runner import run_thematic_analysis, get_actionable_insights, db_config # db_config가 노출된 경우
# from save_daily_universe import save_daily_stock_universe

# 직접 임포트가 불가능하거나 순환 참조가 발생하면, 각 스크립트에 대해 subprocess.run을 사용하세요.
# 이 예시 코드는 별도의 파일에 대해 subprocess.run을 사용하는 더 단순한 방법을 보여줍니다.

def run_script(script_name, description):
    """지정된 Python 스크립트를 실행하고 로그를 남깁니다."""
    print(f"\n[{description}] 실행 시작: {datetime.datetime.now()}")
    try:
        # sys.executable은 현재 파이썬 인터프리터 경로를 나타냅니다.
        # check=True는 명령어 실패 시 CalledProcessError를 발생시킵니다.
        # capture_output=False (기본값)는 자식 프로세스의 출력이 부모 프로세스에 직접 출력되도록 합니다.
        process = subprocess.run([sys.executable, script_name], check=True, capture_output=False)
        print(f"[{description}] 실행 완료: {datetime.datetime.now()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{description}] 오류 발생: 스크립트 실행 실패. {e}")
        print(f"표준 출력: {e.stdout.decode()}")
        print(f"표준 에러: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"[{description}] 예상치 못한 오류 발생: {e}")
        return False

def main_daily_batch_process():
    print("=" * 50)
    print(f"일별 주식 테마 분석 및 매매 유니버스 생성 배치 시작: {datetime.datetime.now()}")
    print("=" * 50)

    # 1단계: 원시 뉴스 요약
    # (사전 조건: daily_news 테이블이 외부 스크래퍼에 의해 채워져 있어야 함)
    # 이 단계는 daily_theme.reason 컬럼을 업데이트합니다.
    if not run_script(DAILY_NEWS_SUMMARIZER_SCRIPT, "뉴스 타이틀 요약"):
        print("뉴스 요약 단계에서 오류 발생. 배치 프로세스 중단.")
        return

    # 2단계: 테마 분석 (Word_dic, Theme_word_relevance, Theme_class momentum_score 업데이트)
    # 이 단계는 새로 요약된 daily_theme.reason 데이터를 사용합니다.
    # 또한 테마 모멘텀 점수를 계산하고 잠재적 신규 테마를 식별합니다.
    # 참고: daily_thematic_runner.py의 get_actionable_insights 함수가
    # 데이터를 직접 반환한다면, 여기서 해당 데이터를 받을 수 있습니다.
    # subprocess.run을 사용할 경우 전체 스크립트를 실행합니다.
    if not run_script(DAILY_THEMATIC_RUNNER_SCRIPT, "일별 테마 분석 및 DB 업데이트"):
        print("일별 테마 분석 단계에서 오류 발생. 배치 프로세스 중단.")
        return
    
    # --- 3단계에 대한 중요 참고사항 ---
    # subprocess.run을 사용할 경우, 복잡한 데이터 구조(예: recommended_themes_and_stocks)를
    # 스크립트 간에 직접 전달하기 어렵습니다.
    # 옵션 A (더 단순하지만 덜 직접적): daily_thematic_runner.py가 actionable_insights를
    # 임시 파일(예: JSON)로 저장하고, save_daily_universe.py가 이를 로드합니다.
    # 옵션 B (더 견고함): 스크립트를 함수로 리팩터링하여 임포트해서 사용합니다.
    # 현재로서는 save_daily_universe.py가 DB에서 최신 인사이트를 다시 쿼리하거나
    # daily_thematic_runner.py가 이미 actionable_insights_today 테이블에 인사이트를 저장했다고 가정합니다.
    # 더 깔끔한 방법은 get_actionable_insights를 daily_thematic_runner.py의 함수로 만들어
    # 여기서 직접 호출한 뒤, 그 반환값을 save_daily_stock_universe에 전달하는 것입니다.

    # 함수 임포트 방식으로 데이터 흐름을 개선합니다.
    # daily_thematic_runner.py에 run_daily_analysis_and_get_insights()와 같은 메인 함수가 있다고 가정하고
    # save_daily_universe.py에는 save_daily_stock_universe(connection, insights_data, target_num) 함수가 있다고 가정합니다.
    
    # 리팩터링 가정:
    # 1. daily_thematic_runner.py의 if __name__ == "__main__": 블록이
    #    run_daily_analysis_and_get_actionable_insights(conn)와 같은 함수로 대체되어 actionable_results를 반환합니다.
    # 2. save_daily_universe.py의 if __name__ == "__main__": 블록이 제거되고,
    #    save_daily_stock_universe(conn, recommended_data, target_num_stocks)가 메인 함수로 사용됩니다.

    try:
        from daily_thematic_runner import db_config, initialize_database_tables, load_stock_names, load_theme_stock_mapping, process_daily_theme_data, identify_potential_new_themes, get_actionable_insights
        from save_daily_universe import save_daily_stock_universe # save_daily_universe.py가 같은 디렉토리에 있다고 가정

        # 이 마스터 스크립트에서 사용할 연결을 새로 생성하여 함수에 전달합니다.
        # (또는 각 함수가 자체적으로 연결을 관리해도 됩니다.)
        conn = pymysql.connect(**db_config)
        cur = conn.cursor()

        print("\n--- 일별 테마 분석 핵심 로직 직접 실행 ---")
        # 2.1단계: 테이블 초기화 (매일 실행 시 중복이지만, 첫 실행 시 안전)
        initialize_database_tables(cur, conn)

        # 2.2단계: 필수 데이터 로드 (종목명, 테마-종목 매핑)
        stock_names_set = load_stock_names(cur)
        stock_to_themes_map = load_theme_stock_mapping(cur)

        # 2.3단계: daily_theme 데이터 처리 (word_dic, theme_word_relevance, theme_class momentum_score 업데이트)
        # 이 함수는 내부적으로 DB 커밋을 처리해야 합니다.
        success_processing_data = process_daily_theme_data(
            cur, conn, stock_names_set, stock_to_themes_map,
            word_freq_threshold=0, word_avg_rate_threshold=1,
            theme_word_relevance_min_occurrences=1
        )
        if not success_processing_data:
            print("process_daily_theme_data 단계에서 오류 발생. 배치 프로세스 중단.")
            conn.close()
            return
        
        # 2.4단계: 잠재적 신규 테마 식별 (트레이더 검토용, 콘솔 출력)
        identify_potential_new_themes(cur)

        # 2.5단계: 액션 가능한 인사이트 도출 (다음 단계에 사용할 데이터 반환)
        recommended_themes_and_stocks = get_actionable_insights(cur, limit_themes=5, limit_stocks_per_theme=10)
        # 여기서 테마별로 더 많은 종목을 가져오고, save_daily_universe에서 상위 100개만 선택합니다.
        
        if not recommended_themes_and_stocks:
            print("액션 가능한 통찰력을 얻지 못했습니다. 매매 유니버스 생성 건너_니다.")
            conn.close()
            return

        print("\n--- 일별 매매 유니버스 저장 ---")
        # 3단계: 일별 매매 유니버스 저장
        # 인사이트와 연결을 저장 함수에 직접 전달합니다.
        save_daily_stock_universe(conn, recommended_themes_and_stocks, target_num_stocks=100) # 100개 종목 목표

    except ImportError as e:
        print(f"필요한 모듈을 임포트할 수 없습니다. 스크립트 파일 경로를 확인하거나 함수 노출이 제대로 되었는지 확인하세요: {e}")
        print("subprocess.run 방식으로 대체 실행 시도...")
        # 직접 임포트가 실패하면 subprocess 방식으로 대체 (데이터 전달에는 덜 이상적임)
        if not run_script(DAILY_THEMATIC_RUNNER_SCRIPT, "일별 테마 분석 및 DB 업데이트 (Subprocess)"):
             print("일별 테마 분석 단계 (Subprocess)에서 오류 발생. 배치 프로세스 중단.")
             return
        # subprocess를 사용할 경우, save_daily_universe는 인사이트를 DB에서 다시 쿼리해야 하거나
        # daily_thematic_runner.py가 결과를 임시 테이블/파일에 저장해야 합니다.
        # 단순화를 위해 직접 임포트 방식을 선호하며, 이 부분은 대체 경고입니다.
        # 실제로 이 대체 방식을 제대로 동작시키려면 save_daily_universe가 특정 임시 테이블을 읽도록 수정해야 합니다.
        # 현재 상태에서는 save_daily_universe가 recommended_themes_and_stocks 데이터를 필요로 합니다.
        print("\n[경고] subprocess 방식은 데이터 연동이 원활하지 않을 수 있습니다. 수동 검토 필요.")
        print("매매 유니버스 저장은 건너_니다. (`daily_thematic_runner.py`가 직접 데이터를 반환하지 않음)")

    except pymysql.MySQLError as e:
        print(f"MariaDB 연결 또는 쿼리 오류: {e}")
    except Exception as e:
        print(f"배치 스크립트 실행 중 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            print("MariaDB 연결 해제.")
        print("=" * 50)
        print(f"일별 주식 테마 분석 및 매매 유니버스 생성 배치 종료: {datetime.datetime.now()}")
        print("=" * 50)

if __name__ == "__main__":
    main_daily_batch_process()