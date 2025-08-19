# run_regimes_profiling.py

import logging
from datetime import date
import json
import sys
import os
import time
import concurrent.futures

# --- 프로젝트 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 필요한 모듈 임포트 ---
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from optimizer.regimes_optimizer import RegimesOptimizer
from config.settings import LIVE_HMM_MODEL_NAME, INITIAL_CASH, STRATEGY_CONFIGS

logging.basicConfig(level=logging.info, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimize_and_save_single_strategy(strategy_name: str, model_name: str, start_date: date, end_date: date):
    """한 개의 전략에 대한 최적화 및 DB 저장 작업을 수행하는 함수 (프로세스별로 실행됨)"""
    
    # 각 프로세스는 독립적인 DB/API 연결을 가져야 함
    api_client = CreonAPIClient()
    db_manager = DBManager()
    backtest_manager = BacktestManager(api_client=api_client, db_manager=db_manager)
    
    # --- ▼ [핵심 수정] 연결 안정화를 위한 1초 지연 추가 ▼ ---
    logger.info(f"[{strategy_name}] DB/API 연결 안정화를 위해 1초 대기합니다.")
    time.sleep(3)
    # --- ▲ [핵심 수정] 종료 ▲ ---
    
    try:
        logger.info(f"[{strategy_name}] 프로파일링 프로세스 시작...")
        
        model_info = db_manager.fetch_hmm_model_by_name(model_name)
        if not model_info:
            logging.error(f"[{strategy_name}] 모델 '{model_name}'을 찾을 수 없어 중단합니다.")
            return f"실패: 모델({model_name}) 없음"

        model_id = model_info['model_id']
        regime_map = db_manager.fetch_daily_regimes(model_id)

        optimizer = RegimesOptimizer(backtest_manager, initial_cash=10_000_000)
        
        champion_params_by_regime = optimizer.run_optimization_for_strategy(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            regime_map=regime_map,
            model_name=model_name
        )
        profiles_to_save = []
        # DB에 저장할 형태로 결과 가공
        for regime_id, result in champion_params_by_regime.items():
            metrics = result.get('metrics', {})
            params = result.get('params', {}).get('strategy_params', {})
            
            # --- ▼ [핵심 수정] strategy_profiler.py와 동일한 방식으로 DB 저장 데이터 생성 ---
            profiles_to_save.append({
                'strategy_name': strategy_name,
                'model_id': model_id,
                'regime_id': regime_id,
                'sharpe_ratio': metrics.get('sharpe_ratio'),
                'mdd': metrics.get('mdd'),
                'total_return': metrics.get('total_return'),
                'win_rate': metrics.get('win_rate'),       # 이제 정상적으로 값을 가져옴
                'num_trades': metrics.get('num_trades'),   # 이제 정상적으로 값을 가져옴
                'profiling_start_date': start_date,
                'profiling_end_date': end_date,
                'params_json': json.dumps(params)          # 'parameters_json' -> 'params_json'
            })
            # --- ▲ 수정 종료 ---

        if db_manager.save_strategy_profiles(profiles_to_save):
            return f"성공: 프로파일 {len(profiles_to_save)}개 저장"
        else:
            return f"실패: DB 저장 오류"
            
    except Exception as e:
        logging.critical(f"[{strategy_name}] 프로세스 실행 중 오류 발생: {e}", exc_info=True)
        return f"실패: {e}"
    finally:
        db_manager.close()

def run_strategy_profiling(model_name: str, start_date: date, end_date: date):
    """
    ProcessPoolExecutor를 사용하여 여러 전략의 프로파일링을 병렬로 실행합니다.
    """
    active_strategies = [name for name, config in STRATEGY_CONFIGS.items() if config.get('strategy_status')]
    if not active_strategies:
        logger.warning("활성화된 전략이 없습니다. settings.py에서 'strategy_status': True인 전략을 확인하세요.")
        return
    
    logger.info(f"====== 병렬 전략 프로파일링 시작 (대상: {active_strategies}) ======")
    
    futures = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for strategy_name in active_strategies:
            future = executor.submit(
                optimize_and_save_single_strategy, 
                strategy_name, model_name, start_date, end_date
            )
            futures[future] = strategy_name

        for future in concurrent.futures.as_completed(futures):
            strategy_name = futures[future]
            try:
                result = future.result()
                logger.info(f"✅ 완료: '{strategy_name}' -> 결과: {result}")
            except Exception as exc:
                logging.error(f"'{strategy_name}' 실행 중 예외 발생: {exc}")

    logger.info(f"====== 모든 전략 프로파일링 작업 완료 ======")

if __name__ == '__main__':
    # 이 스크립트를 직접 실행하면, 아래 파라미터로 함수의 동작을 테스트합니다.
    test_model_name = LIVE_HMM_MODEL_NAME
    test_start_date = date(2024, 1, 1)
    test_end_date = date(2024, 8, 30)
    
    logger.info(f"모듈 테스트를 시작합니다: '{test_model_name}'")
    run_strategy_profiling(
        model_name=test_model_name,
        start_date=test_start_date,
        end_date=test_end_date
    )