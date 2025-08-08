# run_profiling.py

import logging
import sys
import os
from datetime import date # [추가] 날짜 타입 힌트를 위해 임포트

# --- 프로젝트 경로 설정 (기존과 동일) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from manager.db_manager import DBManager
from analyzer.strategy_profiler import StrategyProfiler
from config.settings import LIVE_HMM_MODEL_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProfilingRunner")

# [수정] main 함수의 핵심 로직을 별도 함수로 분리하고, 파라미터를 받도록 변경
def run_strategy_profiling(model_name: str, start_date: date, end_date: date):
    """
    [수정됨] 특정 HMM 모델과 특정 기간의 백테스트 결과에 대해서만 전략 프로파일을 생성하고 저장합니다.
    """
    logger.info(f"====== 전략 프로파일 생성을 시작합니다 ({model_name}, {start_date} ~ {end_date}) ======")
    db_manager = DBManager()
    
    try:
        # 1. 프로파일링 기준 모델 ID 조회
        model_info = db_manager.fetch_hmm_model_by_name(model_name)
        if not model_info:
            logger.error(f"'{model_name}' 모델을 DB에서 찾을 수 없습니다.")
            return False
        model_id = model_info['model_id']
        logger.info(f"프로파일링 기준 모델: '{model_name}' (ID: {model_id})")

        # 2. [수정] 특정 기간의 데이터만 불러오도록 변경
        logger.info(f"DB에서 {start_date} ~ {end_date} 기간의 백테스트 성과, 국면, 실행 정보를 로딩합니다...")
        
        # 참고: 이 부분이 동작하려면 DBManager의 해당 메서드들이 날짜 필터링을 지원해야 합니다.
        run_info_df = db_manager.fetch_backtest_run(start_date=start_date, end_date=end_date)
        performance_df = db_manager.fetch_backtest_performance(start_date=start_date, end_date=end_date)
        regime_data_df = db_manager.fetch_daily_regimes(model_id=model_id) # 국면 데이터는 모델에 종속되므로 기간 필터 불필요

        if run_info_df.empty or performance_df.empty:
            logger.warning("해당 기간의 백테스트 실행 정보 또는 성과 데이터가 DB에 없습니다. 이전 단계를 확인하세요.")
            # 국면 데이터는 없을 수 있으므로 경고만 하고 진행
            if regime_data_df.empty:
                logger.warning(f"모델 ID {model_id}에 대한 국면 데이터가 없습니다.")

        # 3. StrategyProfiler 인스턴스 생성 및 프로파일 생성 실행 (기존과 동일)
        profiler = StrategyProfiler()
        profiles_to_save = profiler.generate_profiles(
            performance_df=performance_df,
            regime_data_df=regime_data_df,
            run_info_df=run_info_df,
            model_id=model_id
        )

        # 4. 생성된 프로파일을 DB에 저장
        if profiles_to_save:
            logger.info(f"총 {len(profiles_to_save)}개의 전략 프로파일을 DB에 저장합니다.")
            # [수정] 저장 시 기존 프로파일을 덮어쓰도록 model_id와 함께 전달
            if db_manager.save_strategy_profiles(profiles_to_save):
                logger.info("✅ 전략 프로파일을 DB에 성공적으로 저장했습니다.")
                return True
            else:
                logger.error("전략 프로파일 DB 저장에 실패했습니다.")
                return False
        else:
            logger.warning("생성된 프로파일이 없습니다. 저장할 데이터가 없습니다.")
            return True # 프로파일이 없는 것도 성공적인 실행으로 간주

    except Exception as e:
        logger.critical(f"프로파일링 실행 중 심각한 오류 발생: {e}", exc_info=True)
        return False
    finally:
        db_manager.close()
        logger.info(f"====== 전략 프로파일 생성을 종료합니다 ======")

# [수정] __main__ 블록은 함수 테스트 용도로 변경
if __name__ == "__main__":
    # 이 스크립트를 직접 실행하면, 아래 파라미터로 함수의 동작을 테스트합니다.
    # 롤링 윈도우의 첫 번째 루프를 가정
    test_model_name = "wf_model_2024_09" # walk-forward model for 2024-09
    test_start_date = date(2023, 7, 1)  # 학습 기간 시작
    test_end_date = date(2024, 6, 30)    # 학습 기간 종료
    
    logger.info(f"모듈 테스트를 시작합니다: '{test_model_name}'")
    success = run_strategy_profiling(
        model_name=test_model_name,
        start_date=test_start_date,
        end_date=test_end_date
    )

    if success:
        logger.info("✅ 프로파일링 모듈 테스트 성공.")
    else:
        logger.error("❌ 프로파일링 모듈 테스트 실패.")