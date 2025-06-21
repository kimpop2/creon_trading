"""
삼중창 전략 최적화 테스트 스크립트
"""

import datetime
import logging
import sys
import os

# 현재 스크립트의 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.creon_api import CreonAPIClient
from manager.data_manager import DataManager
from manager.db_manager import DBManager
from backtest.reporter import Reporter
from selector.stock_selector import StockSelector
from optimizer.grid_search_optimizer import GridSearchOptimizer
from optimizer.progressive_refinement_optimizer import ProgressiveRefinementOptimizer, GridSearchStrategy
from config.sector_config import sector_stocks

# 로깅 설정
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/triple_screen_optimization_test.log", encoding='utf-8'),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)

def test_triple_screen_grid_search():
    """삼중창 전략 그리드서치 테스트"""
    logger.info("=== 삼중창 전략 그리드서치 테스트 시작 ===")
    
    # 컴포넌트 초기화
    api_client = CreonAPIClient()
    data_manager = DataManager()
    db_manager = DBManager()
    reporter = Reporter(db_manager=db_manager)
    stock_selector = StockSelector(data_manager=data_manager, api_client=api_client, sector_stocks_config=sector_stocks)
    
    # 백테스트 기간 설정
    start_date = datetime.datetime(2025, 3, 1).date()
    end_date = datetime.datetime(2025, 6, 20).date()
    
    # 그리드서치 최적화 실행
    optimizer = GridSearchOptimizer(
        api_client=api_client,
        data_manager=data_manager,
        reporter=reporter,
        stock_selector=stock_selector,
        initial_cash=10_000_000
    )
    
    # 삼중창 전략 파라미터 조합 생성 테스트
    combinations = optimizer.generate_parameter_combinations()
    logger.info(f"생성된 파라미터 조합 수: {len(combinations)}")
    
    # 삼중창 전략 조합 필터링
    triple_screen_combinations = [c for c in combinations if 'triple_screen_params' in c]
    logger.info(f"삼중창 전략 조합 수: {len(triple_screen_combinations)}")
    
    if triple_screen_combinations:
        logger.info("첫 번째 삼중창 조합 예시:")
        first_combo = triple_screen_combinations[0]
        logger.info(f"  추세MA: {first_combo['triple_screen_params']['trend_ma_period']}일")
        logger.info(f"  RSI: {first_combo['triple_screen_params']['momentum_rsi_period']}일")
        logger.info(f"  RSI 범위: {first_combo['triple_screen_params']['momentum_rsi_oversold']}-{first_combo['triple_screen_params']['momentum_rsi_overbought']}")
        logger.info(f"  거래량MA: {first_combo['triple_screen_params']['volume_ma_period']}일")
        logger.info(f"  종목수: {first_combo['triple_screen_params']['num_top_stocks']}개")
        logger.info(f"  최소추세강도: {first_combo['triple_screen_params']['min_trend_strength']}")
    
    # 제한된 조합으로 테스트 실행
    test_combinations = triple_screen_combinations[:5]  # 처음 5개만 테스트
    logger.info(f"테스트할 조합 수: {len(test_combinations)}")
    
    successful_results = []
    failed_results = []
    
    for i, params in enumerate(test_combinations):
        logger.info(f"테스트 {i+1}/{len(test_combinations)}")
        try:
            result = optimizer.run_single_backtest(params, start_date, end_date, sector_stocks)
            if result['success']:
                successful_results.append(result)
                logger.info(f"  성공: 수익률 {result['metrics'].get('total_return', 0)*100:.2f}%, "
                           f"샤프지수 {result['metrics'].get('sharpe_ratio', 0):.2f}")
            else:
                failed_results.append(result)
                logger.error(f"  실패: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"  예외 발생: {str(e)}")
            failed_results.append({'params': params, 'error': str(e)})
    
    logger.info(f"테스트 완료: 성공 {len(successful_results)}개, 실패 {len(failed_results)}개")
    
    if successful_results:
        # 최고 성과 결과 찾기
        best_result = max(successful_results, key=lambda x: x['metrics'].get('sharpe_ratio', -999))
        logger.info("최고 성과 결과:")
        logger.info(f"  수익률: {best_result['metrics'].get('total_return', 0)*100:.2f}%")
        logger.info(f"  샤프지수: {best_result['metrics'].get('sharpe_ratio', 0):.2f}")
        logger.info(f"  승률: {best_result['metrics'].get('win_rate', 0)*100:.1f}%")
        logger.info(f"  MDD: {best_result['metrics'].get('mdd', 0)*100:.2f}%")

def test_triple_screen_progressive_optimization():
    """삼중창 전략 점진적 세밀화 테스트"""
    logger.info("=== 삼중창 전략 점진적 세밀화 테스트 시작 ===")
    
    # 컴포넌트 초기화
    api_client = CreonAPIClient()
    data_manager = DataManager()
    db_manager = DBManager()
    reporter = Reporter(db_manager=db_manager)
    stock_selector = StockSelector(data_manager=data_manager, api_client=api_client, sector_stocks_config=sector_stocks)
    
    # 백테스트 기간 설정
    start_date = datetime.datetime(2025, 3, 1).date()
    end_date = datetime.datetime(2025, 4, 1).date()
    
    # 점진적 세밀화 최적화 실행
    grid_strategy = GridSearchStrategy()
    optimizer = ProgressiveRefinementOptimizer(
        strategy=grid_strategy,
        api_client=api_client,
        data_manager=data_manager,
        reporter=reporter,
        stock_selector=stock_selector,
        initial_cash=10_000_000
    )
    
    # 삼중창 전략으로 점진적 세밀화 실행
    results = optimizer.run_progressive_optimization(
        start_date=start_date,
        end_date=end_date,
        sector_stocks=sector_stocks,
        refinement_levels=2,  # 2단계 세밀화
        initial_combinations=20,  # 초기 20개 조합
        daily_strategy_name='triple_screen_daily',  # 삼중창 전략
        minute_strategy_name='open_minute'
    )
    
    if results and results.get('best_metrics'):
        logger.info("점진적 세밀화 최적화 완료:")
        logger.info(f"  수익률: {results['best_metrics'].get('total_return', 0)*100:.2f}%")
        logger.info(f"  샤프지수: {results['best_metrics'].get('sharpe_ratio', 0):.2f}")
        logger.info(f"  승률: {results['best_metrics'].get('win_rate', 0)*100:.1f}%")
        logger.info(f"  MDD: {results['best_metrics'].get('mdd', 0)*100:.2f}%")
        
        # 최적 파라미터 출력
        if results.get('best_params'):
            params = results['best_params']
            if 'triple_screen_params' in params:
                logger.info("최적 파라미터:")
                logger.info(f"  추세MA: {params['triple_screen_params']['trend_ma_period']}일")
                logger.info(f"  RSI: {params['triple_screen_params']['momentum_rsi_period']}일")
                logger.info(f"  RSI 범위: {params['triple_screen_params']['momentum_rsi_oversold']}-{params['triple_screen_params']['momentum_rsi_overbought']}")
                logger.info(f"  거래량MA: {params['triple_screen_params']['volume_ma_period']}일")
                logger.info(f"  종목수: {params['triple_screen_params']['num_top_stocks']}개")
                logger.info(f"  최소추세강도: {params['triple_screen_params']['min_trend_strength']}")

if __name__ == "__main__":
    try:
        # 그리드서치 테스트
        test_triple_screen_grid_search()
        
        print("\n" + "="*60)
        
        # 점진적 세밀화 테스트
        test_triple_screen_progressive_optimization()
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc() 