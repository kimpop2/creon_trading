"""
베이지안 최적화 실행 스크립트
"""

import datetime
import logging
import sys
import os
import json

# 현재 스크립트의 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.creon_api import CreonAPIClient
from manager.data_manager import DataManager
from manager.db_manager import DBManager
from trader.reporter import Reporter
from selector.stock_selector import StockSelector
from optimizer.bayesian_optimizer import BayesianOptimizationStrategy
from trader.backtester import Backtester
from config.sector_config import sector_stocks  # 공통 설정 파일에서 import

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/bayesian_optimizer_run.log", encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger(__name__)

def main():
    """베이지안 최적화 실행"""
    logger.info("베이지안 최적화를 시작합니다.")
    
    # 컴포넌트 초기화
    api_client = CreonAPIClient()
    if not api_client.connected:
        logger.error("Creon API에 연결할 수 없습니다.")
        return
    
    data_manager = DataManager()
    db_manager = DBManager()
    reporter = Reporter(db_manager=db_manager)
    
    # 공통 설정 파일에서 sector_stocks 가져오기
    stock_selector = StockSelector(
        data_manager=data_manager, 
        api_client=api_client, 
        sector_stocks_config=sector_stocks
    )
    
    # 베이지안 최적화 전략 설정
    bayesian_strategy = BayesianOptimizationStrategy(
        n_initial_points=10,
        n_iterations=20
    )
    
    # 백테스터 초기화 - DB 저장 비활성화 (최적화 시 DB 저장 비활성화)
    backtester_instance = Backtester(
        data_manager=data_manager, 
        api_client=api_client, 
        reporter=reporter, 
        stock_selector=stock_selector,
        initial_cash=10_000_000,
        save_to_db=False  # 최적화 시 DB 저장 비활성화
    )
    
    # 백테스트 기간 설정
    start_date = datetime.datetime(2024, 12, 1).date()
    end_date = datetime.datetime(2025, 4, 1).date()
    
    logger.info(f"최적화 기간: {start_date} ~ {end_date}")
    
    # 베이지안 최적화 실행
    from optimizer.progressive_refinement_optimizer import ProgressiveRefinementOptimizer
    
    optimizer = ProgressiveRefinementOptimizer(
        strategy=bayesian_strategy,
        api_client=api_client,
        data_manager=data_manager,
        reporter=reporter,
        stock_selector=stock_selector,
        initial_cash=10_000_000
    )
    
    results = optimizer.run_progressive_optimization(
        start_date=start_date,
        end_date=end_date,
        sector_stocks=sector_stocks,
        refinement_levels=1,  # 베이지안은 1단계만
        initial_combinations=None,
        daily_strategy_name='sma_daily',
        minute_strategy_name='open_minute'
    )
    
    # 결과 저장 및 출력
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bayesian_optimization_results_{timestamp}.json"
    
    # optimizer/results 폴더에 저장
    results_dir = os.path.join("optimizer", "results")
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    optimizer.save_results(results, filepath)
    optimizer.print_summary(results)

if __name__ == "__main__":
    main() 