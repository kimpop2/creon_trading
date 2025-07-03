"""
하이브리드 최적화 실행 스크립트
그리드서치 + 베이지안 최적화 조합
"""

import datetime
import logging
import sys
import os
import json
from dateutil.relativedelta import relativedelta

# 현재 스크립트의 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trade.backtest import Backtest
from api.creon_api import CreonAPIClient
from manager.backtest_manager import BacktestManager
from manager.db_manager import DBManager
from trade.backtest_report import BacktestReport
from optimizer.progressive_refinement_optimizer import ProgressiveRefinementOptimizer, GridSearchStrategy
from optimizer.bayesian_optimizer import BayesianOptimizationStrategy

# from config.sector_config import sector_stocks  # 더 이상 필요 없으므로 삭제

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
    
    def __init__(self, api_client, backtest_manager, report, db_manager):
        self.api_client = api_client
        self.backtest_manager = backtest_manager
        self.report = report
        self.db_manager = db_manager
        
    # run_hybrid_optimization, _run_grid_search, _run_bayesian_refinement 메서드에서 sector_stocks 인자 제거
    def run_hybrid_optimization(self, start_date, end_date, daily_strategy_name='sma_daily'):
        """하이브리드 최적화 실행"""
        logger.info("=== 하이브리드 최적화 시작 ===")
        
        # 1단계: 그리드서치로 대략적인 최적 영역 찾기
        logger.info("1단계: 그리드서치 최적화 시작")
        # sector_stocks 인자 제거
        grid_results = self._run_grid_search(start_date, end_date, daily_strategy_name) 
        
        if not grid_results or not grid_results.get('best_params'):
            logger.error("그리드서치에서 최적 파라미터를 찾지 못했습니다.")
            return None
        
        # 2단계: 베이지안으로 최적점 주변 세밀 탐색
        logger.info("2단계: 베이지안 세밀 최적화 시작")
        # sector_stocks 인자 제거
        bayesian_results = self._run_bayesian_refinement(
            grid_results['best_params'], start_date, end_date, daily_strategy_name
        )
        
        # 3단계: 결과 비교 및 최종 선택
        final_results = self._compare_and_select_best(grid_results, bayesian_results)
        
        return final_results
    
    # sector_stocks 인자 제거
    def _run_grid_search(self, start_date, end_date, daily_strategy_name='sma_daily'):
        """그리드서치 최적화 실행"""
        grid_strategy = GridSearchStrategy()
        optimizer = ProgressiveRefinementOptimizer(
            strategy=grid_strategy,
            api_client=self.api_client,
            backtest_manager=self.backtest_manager,
            report=self.report,
            db_manager=self.db_manager, # db_manager를 전달하여 내부에서 종목 조회 가능
            initial_cash=10_000_000
        )
        
        results = optimizer.run_progressive_optimization(
            start_date=start_date,
            end_date=end_date,
            # sector_stocks=sector_stocks, # <-- 이 인자는 더 이상 필요 없으므로 제거
            daily_strategy_name=daily_strategy_name,
            minute_strategy_name='open_minute'
        )
        
        return results
    
    # sector_stocks 인자 제거
    def _run_bayesian_refinement(self, best_params, start_date, end_date, daily_strategy_name='sma_daily'):
        """베이지안 세밀 최적화 실행"""
        # 최적 파라미터 주변으로 범위 설정
        refined_ranges = self._create_refined_ranges(best_params, daily_strategy_name)
        
        # 베이지안 전략에 세밀화된 범위 적용
        bayesian_strategy = BayesianOptimizationStrategy(
            n_initial_points=8,
            n_iterations=15
        )
        
        # 세밀화된 범위로 베이지안 최적화 실행
        optimizer = ProgressiveRefinementOptimizer(
            strategy=bayesian_strategy,
            api_client=self.api_client,
            backtest_manager=self.backtest_manager,
            report=self.report,
            db_manager=self.db_manager, # db_manager를 전달하여 내부에서 종목 조회 가능
            initial_cash=10_000_000
        )
        
        results = optimizer.run_progressive_optimization(
            start_date=start_date,
            end_date=end_date,
            # sector_stocks=sector_stocks, # <-- 이 인자는 더 이상 필요 없으므로 제거
            refinement_levels=1,  # 베이지안은 1단계만
            initial_combinations=None,
            daily_strategy_name=daily_strategy_name,
            minute_strategy_name='open_minute'
        )
        
        return results
    
    def _create_refined_ranges(self, best_params, daily_strategy_name='sma_daily'):
        """최적 파라미터 주변으로 세밀화된 범위 생성"""
        # 이 함수는 파라미터 범위만 생성하므로, sector_stocks와 직접적인 관련 없음
        # 기존 로직 유지
        refined_ranges = {
            'strategy_params': {
                'sma_daily': {
                    'parameter_ranges': {
                        'short_sma_period': [
                            max(1, best_params.get('sma_params', {}).get('short_sma_period', 5) - 1),
                            best_params.get('sma_params', {}).get('short_sma_period', 5) + 1
                        ],
                        'long_sma_period': [
                            max(5, best_params.get('sma_params', {}).get('long_sma_period', 15) - 2),
                            best_params.get('sma_params', {}).get('long_sma_period', 15) + 2
                        ],
                        'volume_ma_period': [
                            max(2, best_params.get('sma_params', {}).get('volume_ma_period', 5) - 1),
                            best_params.get('sma_params', {}).get('volume_ma_period', 5) + 1
                        ],
                        # 'num_top_stocks'는 이제 daily_universe에서 결정되므로,
                        # 전략 파라미터에서 제거하거나, 최적화 대상이 아니라면 고정값으로 처리
                        'num_top_stocks': [ # 이 부분은 전략의 'num_top_stocks'가 여전히 파라미터로 필요하다면 유지
                            max(2, best_params.get('sma_params', {}).get('num_top_stocks', 5) - 1),
                            best_params.get('sma_params', {}).get('num_top_stocks', 5) + 1
                        ],
                    }
                },
                'dual_momentum_daily': {
                    'parameter_ranges': {
                        'momentum_period': [
                            max(5, best_params.get('dual_momentum_params', {}).get('momentum_period', 15) - 2),
                            best_params.get('dual_momentum_params', {}).get('momentum_period', 15) + 2
                        ],
                        'rebalance_weekday': [
                            max(0, best_params.get('dual_momentum_params', {}).get('rebalance_weekday', 1) - 1),
                            min(4, best_params.get('dual_momentum_params', {}).get('rebalance_weekday', 1) + 1)
                        ],
                        'num_top_stocks': [
                            max(2, best_params.get('dual_momentum_params', {}).get('num_top_stocks', 5) - 1),
                            best_params.get('dual_momentum_params', {}).get('num_top_stocks', 5) + 1
                        ],
                    }
                },
                'triple_screen_daily': {
                    'parameter_ranges': {
                        'trend_ma_period': [
                            max(15, best_params.get('triple_screen_params', {}).get('trend_ma_period', 30) - 5),
                            best_params.get('triple_screen_params', {}).get('trend_ma_period', 30) + 5
                        ],
                        'momentum_rsi_period': [
                            max(10, best_params.get('triple_screen_params', {}).get('momentum_rsi_period', 14) - 2),
                            best_params.get('triple_screen_params', {}).get('momentum_rsi_period', 14) + 2
                        ],
                        'momentum_rsi_oversold': [
                            max(20, best_params.get('triple_screen_params', {}).get('momentum_rsi_oversold', 25) - 2),
                            best_params.get('triple_screen_params', {}).get('momentum_rsi_oversold', 25) + 2
                        ],
                        'momentum_rsi_overbought': [
                            max(65, best_params.get('triple_screen_params', {}).get('momentum_rsi_overbought', 75) - 3),
                            best_params.get('triple_screen_params', {}).get('momentum_rsi_overbought', 75) + 3
                        ],
                        'volume_ma_period': [
                            max(5, best_params.get('triple_screen_params', {}).get('volume_ma_period', 10) - 2),
                            best_params.get('triple_screen_params', {}).get('volume_ma_period', 10) + 2
                        ],
                        'num_top_stocks': [
                            max(3, best_params.get('triple_screen_params', {}).get('num_top_stocks', 5) - 1),
                            best_params.get('triple_screen_params', {}).get('num_top_stocks', 5) + 1
                        ],
                        'min_trend_strength': [
                            max(0.01, best_params.get('triple_screen_params', {}).get('min_trend_strength', 0.03) - 0.01),
                            best_params.get('triple_screen_params', {}).get('min_trend_strength', 0.03) + 0.01
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
        # 기존 로직 유지
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
    
    backtest_manager = BacktestManager()
    db_manager = DBManager()
    report = BacktestReport(db_manager=db_manager)
    

    # 백테스터 초기화 - DB 저장 비활성화 (최적화 시 DB 저장 비활성화)
    # Backtest 클래스도 내부적으로 daily_universe를 사용하도록 수정 필요
    backtest_instance = Backtest(
        backtest_manager=backtest_manager, 
        api_client=api_client, 
        backtest_report=report, 
        db_manager=db_manager, # db_manager를 Backtest에 전달하여 종목 조회에 사용
        initial_cash=10_000_000,
        save_to_db=False  # 최적화 시 DB 저장 비활성화
    )
    
    # 하이브리드 최적화 실행
    hybrid_optimizer = HybridOptimizer(
        api_client=api_client,
        backtest_manager=backtest_manager,
        report=report,
        db_manager=db_manager
    )
    
    # ===========================================================
    # 백테스트 기간 설정
    start_date = datetime.datetime(2025, 5, 1).date()
    end_date = datetime.datetime(2025, 6, 15).date()
    
    logger.info(f"최적화 기간: {start_date} ~ {end_date}")

    # run_hybrid_optimization 호출 시 sector_stocks 인자 제거
    results = hybrid_optimizer.run_hybrid_optimization(
        start_date=start_date,
        end_date=end_date,
        # sector_stocks=sector_stocks, # <-- 이 인자 제거
        daily_strategy_name='sma_daily'  ################ 전략 사용
    )
    # ===========================================================
    
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
        
        # 대상 종목수 계산 (daily_universe에서 동적으로 가져오도록 변경)
        # from config.sector_config import get_total_stock_count, get_sector_names # <-- 이 줄 삭제
        # total_stocks = get_total_stock_count() # <-- 이 부분도 삭제
        # sector_names = get_sector_names() # <-- 이 부분도 삭제
        # print(f"대상 종목수: {total_stocks}개 ({', '.join(sector_names)} 섹터)") # <-- 이 부분도 수정
        
        # 백테스트 기간 동안 daily_universe에 등장한 고유 종목 수를 집계 (예시)
        # 이 부분은 실제 백테스트 결과 객체(results)에 포함되도록 백테스트 로직을 수정해야 합니다.
        # 현재는 임시로 "DB의 daily_universe에서 동적으로 선택됨"으로 표시
        print(f"대상 종목: DB의 daily_universe에서 동적으로 선택됨") 
        
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
            
            # 듀얼모멘텀 파라미터 출력
            if 'dual_momentum_params' in best_params:
                dual = best_params['dual_momentum_params']
                weekday_names = ['월', '화', '수', '목', '금']
                weekday_name = weekday_names[dual['rebalance_weekday']]
                print(f"  모멘텀 기간: {dual['momentum_period']}일")
                print(f"  리밸런싱 요일: {weekday_name}요일")
                print(f"  종목수: {dual['num_top_stocks']}개")
            
            # 삼중창 파라미터 출력
            if 'triple_screen_params' in best_params:
                triple = best_params['triple_screen_params']
                print(f"  추세MA: {triple['trend_ma_period']}일")
                print(f"  RSI 기간: {triple['momentum_rsi_period']}일")
                print(f"  RSI 과매도: {triple['momentum_rsi_oversold']}")
                print(f"  RSI 과매수: {triple['momentum_rsi_overbought']}")
                print(f"  거래량MA: {triple['volume_ma_period']}일")
                print(f"  종목수: {triple['num_top_stocks']}개")
                print(f"  최소 추세강도: {triple['min_trend_strength']:.3f}")
            
            # 손절매 파라미터 출력 (소수점 1자리로 제한)
            if 'stop_loss_params' in best_params:
                stop_loss = best_params['stop_loss_params']
                print(f"  손절매: {stop_loss['stop_loss_ratio']:.1f}%")
                print(f"  트레일링스탑: {stop_loss['trailing_stop_ratio']:.1f}%")
                print(f"  최대 손실 포지션: {stop_loss['max_losing_positions']}개")
    
    logger.info("하이브리드 최적화가 완료되었습니다.")

if __name__ == "__main__":
    main()
