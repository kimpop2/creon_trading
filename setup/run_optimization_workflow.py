# setup/run_optimization_workflow.py

import logging
from datetime import datetime, timedelta
import sys
import os
# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- 필요한 모든 컴포넌트 임포트 ---
from manager.backtest_manager import BacktestManager
from manager.db_manager import DBManager
from api.creon_api import CreonAPIClient
from optimizer.system_optimizer import SystemOptimizer
from config.settings import INITIAL_CASH

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def execute_full_optimization_workflow():
    """
    HMM 기반 동적 자산배분 시스템의 전체 최적화 워크플로우를 실행합니다.
    """
    logger.info("====== 전체 최적화 워크플로우 시작 ======")

    # 1. 핵심 컴포넌트 초기화
    api_client = CreonAPIClient()
    db_manager = DBManager()
    backtest_manager = BacktestManager(api_client, db_manager)
    optimizer = SystemOptimizer(backtest_manager=backtest_manager, initial_cash=INITIAL_CASH)

    # 2. 기간 설정 (예: 최근 1년)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    
    # 3. [1단계] 개별 전략 최적화 실행 (전략 리스트는 설정 파일 등에서 관리)
    logger.info("--- [1단계] 개별 전략 최적화를 시작합니다. ---")
    strategies_to_optimize = ['SMADaily', 'TripleScreenDaily'] # 최적화할 전략 목록
    best_strategy_params = {}

    for strategy_name in strategies_to_optimize:
        results = optimizer.run_hybrid_optimization(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            mode='strategy'
        )
        if results:
            best_strategy_params[strategy_name] = results.get('params')

    # 4. [2단계] 국면별 전략 프로파일 DB 구축 (이 부분은 별도 스크립트로 구현 필요)
    logger.info("--- [2단계] 국면별 전략 프로파일 DB 구축을 시작합니다. (구현 필요) ---")
    # build_strategy_profiles(best_strategy_params, start_date, end_date)

    # 5. [3단계] HMM 시스템 최적화 실행
    logger.info("--- [3단계] HMM 전체 시스템 최적화를 시작합니다. ---")
    final_system_results = optimizer.run_hybrid_optimization(
        strategy_name='SMADaily', # HMM 시스템에 포함될 기반 전략 그룹
        start_date=start_date,
        end_date=end_date,
        mode='hmm'
    )

    # 6. 최종 결과 요약 및 저장
    if final_system_results:
        logger.info("\n====== 최종 최적화 결과 ======")
        logger.info(f"최고 성과 파라미터: {final_system_results.get('params')}")
        logger.info(f"최고 성과 지표: {final_system_results.get('metrics')}")
        # (결과를 DB나 파일에 저장하는 로직 추가)

    logger.info("====== 전체 최적화 워크플로우 종료 ======")


if __name__ == '__main__':
    execute_full_optimization_workflow()