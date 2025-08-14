# scripts/optimization/run_portfolio_optimizer.py

import logging
import json
from datetime import date
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from optimizer.portfolio_optimizer import PortfolioOptimizer
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from config.settings import INITIAL_CASH, LIVE_HMM_MODEL_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/portfolio_optimizer_run.log", encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

def run_portfolio_optimization(model_name: str, start_date: date, end_date: date, backtest_manager: BacktestManager) -> dict:
    """단일 기간에 대한 포트폴리오 최적화를 실행하고 최적의 정책(policy) 딕셔너리를 반환합니다."""
    logger.info(f"====== 포트폴리오 최적화 시작: {model_name} ({start_date} ~ {end_date}) ======")
    
    optimizer = PortfolioOptimizer(
        backtest_manager=backtest_manager,
        initial_cash=INITIAL_CASH
    )
    
    final_results = optimizer.run_hybrid_optimization(
        start_date=start_date,
        end_date=end_date,
        model_name=model_name
    )

    if final_results:
        best_hmm_params = final_results.get('params', {}).get('hmm_params', {})
        policy_rules = {
            "regime_to_principal_ratio": {
                "0": 1.0,
                "1": best_hmm_params.get('policy_bear_ratio', 0.5),
                "2": best_hmm_params.get('policy_crisis_ratio', 0.2),
                "3": 1.0
            },
            "default_principal_ratio": 1.0
        }
        logger.info(f"✅ 최적 정책 발견: {json.dumps(policy_rules, indent=2)}")
        return policy_rules
    else:
        logger.error("최적 정책을 찾지 못했습니다.")
        return {}

if __name__ == '__main__':
    api_client = CreonAPIClient() 
    db_manager = DBManager()
    backtest_manager = BacktestManager(api_client, db_manager)

    start_date = date(2024, 12, 2)
    end_date = date(2025, 5, 30)
    
    # 데이터 사전 준비 (최적화 실행 전 한 번만)
    backtest_manager.prepare_pykrx_data_for_period(start_date, end_date)

    optimal_policy = run_portfolio_optimization(
        model_name='wf_model_202508',
        start_date=start_date,
        end_date=end_date,
        backtest_manager=backtest_manager
    )

    if optimal_policy:
        result_filename = "policy.json"
        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump(optimal_policy, f, ensure_ascii=False, indent=4)
        logger.info(f"✅ 최적 HMM 정책을 '{result_filename}' 파일에 저장했습니다. config/policy.json으로 활용하세요.")