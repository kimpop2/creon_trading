"""
하이브리드 최적화 실행 스크립트
그리드서치 + 베이지안 최적화 조합
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
from backtest.reporter import Reporter
from selector.stock_selector import StockSelector
from optimizer.progressive_refinement_optimizer import ProgressiveRefinementOptimizer, GridSearchStrategy
from optimizer.bayesian_optimizer import BayesianOptimizationStrategy
from backtest.backtester import Backtester
from config.sector_config import sector_stocks  # 공통 설정 파일에서 import

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/hybrid_optimizer_run.log", encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger(__name__)

class HybridOptimizer:
    """하이브리드 최적화 클래스"""
    
    def __init__(self, api_client, data_manager, reporter, stock_selector):
        self.api_client = api_client
        self.data_manager = data_manager
        self.reporter = reporter
        self.stock_selector = stock_selector
        
    def run_hybrid_optimization(self, start_date, end_date, sector_stocks):
        """하이브리드 최적화 실행"""
        logger.info("=== 하이브리드 최적화 시작 ===")
        
        # 1단계: 그리드서치로 대략적인 최적 영역 찾기
        logger.info("1단계: 그리드서치 최적화 시작")
        grid_results = self._run_grid_search(start_date, end_date, sector_stocks)
        
        if not grid_results or not grid_results.get('best_params'):
            logger.error("그리드서치에서 최적 파라미터를 찾지 못했습니다.")
            return None
        
        # 2단계: 베이지안으로 최적점 주변 세밀 탐색
        logger.info("2단계: 베이지안 세밀 최적화 시작")
        bayesian_results = self._run_bayesian_refinement(
            grid_results['best_params'], start_date, end_date, sector_stocks
        )
        
        # 3단계: 결과 비교 및 최종 선택
        final_results = self._compare_and_select_best(grid_results, bayesian_results)
        
        return final_results
    
    def _run_grid_search(self, start_date, end_date, sector_stocks):
        """그리드서치 최적화 실행"""
        grid_strategy = GridSearchStrategy()
        optimizer = ProgressiveRefinementOptimizer(
            strategy=grid_strategy,
            api_client=self.api_client,
            data_manager=self.data_manager,
            reporter=self.reporter,
            stock_selector=self.stock_selector,
            initial_cash=10_000_000
        )
        
        results = optimizer.run_progressive_optimization(
            start_date=start_date,
            end_date=end_date,
            sector_stocks=sector_stocks,
            refinement_levels=2,
            initial_combinations=30,
            daily_strategy_name='sma_daily',
            minute_strategy_name='open_minute'
        )
        
        return results
    
    def _run_bayesian_refinement(self, best_params, start_date, end_date, sector_stocks):
        """베이지안 세밀 최적화 실행"""
        # 최적 파라미터 주변으로 범위 설정
        refined_ranges = self._create_refined_ranges(best_params)
        
        # 베이지안 전략에 세밀화된 범위 적용
        bayesian_strategy = BayesianOptimizationStrategy(
            n_initial_points=8,
            n_iterations=15
        )
        
        # 세밀화된 범위로 베이지안 최적화 실행
        optimizer = ProgressiveRefinementOptimizer(
            strategy=bayesian_strategy,
            api_client=self.api_client,
            data_manager=self.data_manager,
            reporter=self.reporter,
            stock_selector=self.stock_selector,
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
        
        return results
    
    def _create_refined_ranges(self, best_params):
        """최적 파라미터 주변으로 세밀화된 범위 생성"""
        refined_ranges = {
            'strategy_params': {
                'sma_daily': {
                    'parameter_ranges': {
                        'short_sma_period': [
                            max(1, best_params['sma_params']['short_sma_period'] - 1),
                            best_params['sma_params']['short_sma_period'] + 1
                        ],
                        'long_sma_period': [
                            max(5, best_params['sma_params']['long_sma_period'] - 2),
                            best_params['sma_params']['long_sma_period'] + 2
                        ],
                        'volume_ma_period': [
                            max(2, best_params['sma_params']['volume_ma_period'] - 1),
                            best_params['sma_params']['volume_ma_period'] + 1
                        ],
                        'num_top_stocks': [
                            max(2, best_params['sma_params']['num_top_stocks'] - 1),
                            best_params['sma_params']['num_top_stocks'] + 1
                        ],
                    }
                }
            },
            'common_params': {
                'stop_loss': [
                    best_params['stop_loss_params']['stop_loss_ratio'] - 1.0,
                    best_params['stop_loss_params']['stop_loss_ratio'] + 1.0
                ],
                'trailing_stop': [
                    best_params['stop_loss_params']['trailing_stop_ratio'] - 1.0,
                    best_params['stop_loss_params']['trailing_stop_ratio'] + 1.0
                ],
                'max_losing_positions': [
                    max(1, best_params['stop_loss_params']['max_losing_positions'] - 1),
                    best_params['stop_loss_params']['max_losing_positions'] + 1
                ]
            }
        }
        
        return refined_ranges
    
    def _compare_and_select_best(self, grid_results, bayesian_results):
        """결과 비교 및 최종 선택"""
        grid_score = grid_results.get('best_metrics', {}).get('sharpe_ratio', 0)
        bayesian_score = bayesian_results.get('best_metrics', {}).get('sharpe_ratio', 0)
        
        logger.info(f"그리드서치 최고 샤프지수: {grid_score:.3f}")
        logger.info(f"베이지안 최고 샤프지수: {bayesian_score:.3f}")
        
        if bayesian_score > grid_score:
            logger.info("베이지안 최적화 결과가 더 우수합니다.")
            final_results = bayesian_results
            final_results['optimization_method'] = 'Bayesian'
        else:
            logger.info("그리드서치 결과가 더 우수합니다.")
            final_results = grid_results
            final_results['optimization_method'] = 'Grid Search'
        
        return final_results

def main():
    """하이브리드 최적화 실행"""
    logger.info("하이브리드 최적화를 시작합니다.")
    
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
    
    # 백테스터 초기화 - DB 저장 비활성화 (최적화 시 DB 저장 비활성화)
    backtester_instance = Backtester(
        data_manager=data_manager, 
        api_client=api_client, 
        reporter=reporter, 
        stock_selector=stock_selector,
        initial_cash=10_000_000,
        save_to_db=False  # 최적화 시 DB 저장 비활성화
    )
    
    # 하이브리드 최적화 실행
    hybrid_optimizer = HybridOptimizer(
        api_client=api_client,
        data_manager=data_manager,
        reporter=reporter,
        stock_selector=stock_selector
    )
    
    # 백테스트 기간 설정
    start_date = datetime.datetime(2025, 5, 1).date()
    end_date = datetime.datetime(2025, 6, 15).date()
    
    logger.info(f"최적화 기간: {start_date} ~ {end_date}")
    
    # 하이브리드 최적화 실행
    results = hybrid_optimizer.run_hybrid_optimization(
        start_date=start_date,
        end_date=end_date,
        sector_stocks=sector_stocks
    )
    
    if results:
        # 결과 저장 및 출력
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hybrid_optimization_results_{timestamp}.json"
        
        # optimizer/results 폴더에 저장
        results_dir = os.path.join("optimizer", "results")
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"결과 저장 완료: {filepath}")
        
        # 결과 요약 출력
        print("\n" + "="*60)
        print(f"하이브리드 최적화 결과 요약")
        print("="*60)
        print(f"최적화 방법: {results.get('optimization_method', 'Unknown')}")
        
        # 날짜 계산 수정
        days_diff = (end_date - start_date).days
        print(f"백테스팅 기간: {start_date} ~ {end_date} ({days_diff}일)")
        
        # 대상 종목수 계산 (공통 설정 파일 사용)
        from config.sector_config import get_total_stock_count, get_sector_names
        total_stocks = get_total_stock_count()
        sector_names = get_sector_names()
        print(f"대상 종목수: {total_stocks}개 ({', '.join(sector_names)} 섹터)")
        
        if results.get('best_metrics'):
            metrics = results['best_metrics']
            print(f"수익률: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"샤프지수: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"승률: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"MDD: {metrics.get('mdd', 0)*100:.2f}%")
        
        # 최적 파라미터 출력
        if results.get('best_params'):
            print(f"\n최적 파라미터:")
            best_params = results['best_params']
            
            # SMA 파라미터 출력
            if 'sma_params' in best_params:
                sma = best_params['sma_params']
                print(f"  SMA: {sma['short_sma_period']}일/{sma['long_sma_period']}일")
                print(f"  거래량MA: {sma['volume_ma_period']}일")
                print(f"  종목수: {sma['num_top_stocks']}개")
            
            # 손절매 파라미터 출력 (소수점 1자리로 제한)
            if 'stop_loss_params' in best_params:
                stop_loss = best_params['stop_loss_params']
                print(f"  손절매: {stop_loss['stop_loss_ratio']:.1f}%")
                print(f"  트레일링스탑: {stop_loss['trailing_stop_ratio']:.1f}%")
                print(f"  최대 손실 포지션: {stop_loss['max_losing_positions']}개")
    
    logger.info("하이브리드 최적화가 완료되었습니다.")

if __name__ == "__main__":
    main() 