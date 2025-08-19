# scripts/optimization/run_strategy_optimization.py (신규 파일)

import logging
import json
import numpy as np
from datetime import date
from typing import Dict, List, Any
import sys
import os

# --- 프로젝트 경로 추가 및 모듈 임포트 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from optimizer.strategy_optimizer import StrategyOptimizer
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from config.settings import INITIAL_CASH, STRATEGY_CONFIGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """[신규] Numpy 타입을 JSON 직렬화 가능한 Python 기본 타입으로 재귀적으로 변환합니다."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj


def run_strategy_optimization(start_date: date, end_date: date, backtest_manager: BacktestManager) -> Dict[str, Dict[str, Any]]:
    """
    settings.py의 STRATEGIES_INFO 에 정의된 모든 전략에 대해 최적화를 수행하고,
    전략별 '챔피언 파라미터' 딕셔너리를 반환합니다.
    """
    logger.info(f"===== 시스템 전체 전략 최적화 시작: {start_date} ~ {end_date} =====")
    
    optimizer = StrategyOptimizer(backtest_manager=backtest_manager, initial_cash=INITIAL_CASH)
    champion_params_all = {}

    # [수정] STRATEGY_CONFIGS 리스트를 직접 사용
    for strategy_name, config in STRATEGY_CONFIGS.items():
        if config.get('strategy_status') is True: # 'strategy_status'가 True인 전략만
            logger.info(f"\n--- 전략 최적화 중: {strategy_name} ---")
            final_results = optimizer.run_hybrid_optimization(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
            )
        
        if final_results:
            best_params = final_results.get('params', {}).get('strategy_params', {})
            champion_params_all[strategy_name] = best_params
            logger.info(f"✅ {strategy_name}의 챔피언 파라미터: {best_params}")
        else:
            logger.warning(f"{strategy_name}의 최적 파라미터를 찾지 못했습니다. 기본값을 사용합니다.")
            # 최적화 실패 시 settings.py의 기본값 사용
            champion_params_all[strategy_name] = STRATEGY_CONFIGS.get(strategy_name, {}).get('default_params', {})
            
    logger.info("===== 시스템 전체 전략 최적화 완료 =====")
    return champion_params_all

if __name__ == '__main__':
    api_client = CreonAPIClient()
    db_manager = DBManager()
    backtest_manager = BacktestManager(api_client, db_manager)

    start_date = date(2025, 6, 1)
    end_date = date(2025, 6, 15)

    all_champion_params = run_strategy_optimization(start_date, end_date, backtest_manager)
    
    logger.info("\n" + "="*50 + "\n--- 최종 챔피언 파라미터 요약 ---\n" + "="*50)
    serializable_params = convert_numpy_types(all_champion_params)
    logger.info(json.dumps(serializable_params, indent=4, ensure_ascii=False))