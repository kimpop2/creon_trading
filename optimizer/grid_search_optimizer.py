"""
Grid Search Optimizer
그리드서치 기법을 이용한 전략 파라미터 최적화 프로그램
새로운 백테스트 구조에 맞춰 설계됨
"""

import itertools
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import json
import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from backtest.backtester import Backtester
from strategies.sma_daily import SMADaily
from manager.data_manager import DataManager
from manager.db_manager import DBManager
from backtest.reporter import Reporter
from selector.stock_selector import StockSelector
from strategies.open_minute import OpenMinute

# 로거 설정
logger = logging.getLogger(__name__)

class GridSearchOptimizer:
    """
    그리드서치 기법을 이용한 전략 파라미터 최적화 클래스
    새로운 백테스트 구조에 맞춰 설계됨
    """
    
    def __init__(self, 
                 api_client: CreonAPIClient,
                 data_manager: DataManager,
                 reporter: Reporter,
                 stock_selector: StockSelector,
                 initial_cash: float = 10_000_000):
        """
        GridSearchOptimizer 초기화
        
        Args:
            api_client: Creon API 클라이언트
            data_manager: 데이터 매니저
            reporter: 리포터
            stock_selector: 종목 선택기
            initial_cash: 초기 자본금
        """
        self.api_client = api_client
        self.data_manager = data_manager
        self.reporter = reporter
        self.stock_selector = stock_selector
        self.initial_cash = initial_cash
        
        # 최적화 결과 저장
        self.optimization_results = []
        self.best_result = None
        
        logger.info("GridSearchOptimizer 초기화 완료")
    
    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """
        최적화할 파라미터 조합을 생성합니다.
        
        Returns:
            파라미터 조합 리스트
        """
        # SMA 전략 파라미터 - 최적점 주변 세밀 조정
        sma_short_periods = [1, 2, 3, 4]        # 2일 주변 세밀 조정
        sma_long_periods = [8, 10, 12, 15]      # 10일 주변 세밀 조정
        volume_ma_periods = [2, 3, 4, 5]        # 3일 주변 세밀 조정
        num_top_stocks = [2, 3, 4, 5]           # 3개 주변 세밀 조정
        
        # 손절매 파라미터 - 최적점 주변 세밀 조정
        stop_loss_ratios = [-2.5, -3.0, -3.5, -4.0]    # -3.0% 주변 0.5% 단위
        trailing_stop_ratios = [-2.5, -3.0, -3.5]      # -3.0% 주변 세밀 조정
        max_losing_positions = [2, 3, 4]               # 3개 주변 조정
        
        combinations = []
        
        # SMA + 손절매 조합 생성 (OpenMinute는 RSI 파라미터가 필요하지 않음)
        for short_period, long_period, volume_period, num_stocks in itertools.product(
            sma_short_periods, sma_long_periods, volume_ma_periods, num_top_stocks):
            
            # 유효한 조합만 필터링 (단기 < 장기)
            if short_period >= long_period:
                continue
                
            for stop_loss, trailing_stop, max_losing in itertools.product(
                stop_loss_ratios, trailing_stop_ratios, max_losing_positions):
                
                combination = {
                    'sma_params': {
                        'short_sma_period': short_period,
                        'long_sma_period': long_period,
                        'volume_ma_period': volume_period,
                        'num_top_stocks': num_stocks,
                        'safe_asset_code': 'A439870'
                    },
                    'stop_loss_params': {
                        'stop_loss_ratio': stop_loss,
                        'trailing_stop_ratio': trailing_stop,
                        'portfolio_stop_loss': stop_loss,
                        'early_stop_loss': stop_loss,
                        'max_losing_positions': max_losing
                    }
                }
                combinations.append(combination)
        
        logger.info(f"총 {len(combinations)}개의 파라미터 조합 생성 완료")
        return combinations
    
    def run_single_backtest(self, 
                           params: Dict[str, Any], 
                           start_date: datetime.date, 
                           end_date: datetime.date,
                           sector_stocks: Dict[str, List[Tuple[str, str]]]) -> Dict[str, Any]:
        """
        단일 파라미터 조합으로 백테스트를 실행합니다.
        
        Args:
            params: 파라미터 조합
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            sector_stocks: 섹터별 종목 정보
            
        Returns:
            백테스트 결과
        """
        try:
            # 백테스터 초기화
            backtester = Backtester(
                data_manager=self.data_manager,
                api_client=self.api_client,
                reporter=self.reporter,
                stock_selector=self.stock_selector,
                initial_cash=self.initial_cash
            )
            
            # 전략 생성
            sma_strategy = SMADaily(
                data_store=backtester.data_store,
                strategy_params=params['sma_params'],
                broker=backtester.broker
            )
            
            # OpenMinute 전략 생성 (RSI 파라미터가 필요하지 않음)
            minute_params = {
                'num_top_stocks': params['sma_params']['num_top_stocks']
            }
            open_minute_strategy = OpenMinute(
                data_store=backtester.data_store,
                strategy_params=minute_params,
                broker=backtester.broker
            )
            
            # 전략 설정
            backtester.set_strategies(
                daily_strategy=sma_strategy,
                minute_strategy=open_minute_strategy
            )
            
            # 손절매 파라미터 설정
            backtester.set_broker_stop_loss_params(params['stop_loss_params'])
            
            # 데이터 로딩
            self._load_backtest_data(backtester, start_date, end_date, sector_stocks)
            
            # 백테스트 실행
            portfolio_values, metrics = backtester.run(start_date, end_date)
            
            # 결과 정리
            result = {
                'params': params,
                'metrics': metrics,
                'portfolio_values': portfolio_values,
                'success': True
            }
            
            logger.info(f"백테스트 성공: 수익률 {metrics.get('total_return', 0)*100:.2f}%, "
                       f"샤프지수 {metrics.get('sharpe_ratio', 0):.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"백테스트 실패: {str(e)}")
            return {
                'params': params,
                'metrics': {},
                'portfolio_values': pd.DataFrame(),
                'success': False,
                'error': str(e)
            }
    
    def _load_backtest_data(self, 
                           backtester: Backtester, 
                           start_date: datetime.date, 
                           end_date: datetime.date,
                           sector_stocks: Dict[str, List[Tuple[str, str]]]):
        """
        백테스트에 필요한 데이터를 로딩합니다.
        """
        # 데이터 가져오기 시작일 (백테스트 시작일 1개월 전)
        data_fetch_start = (start_date - timedelta(days=30)).replace(day=1)
        
        # 안전자산 데이터 로딩
        safe_asset_code = 'A439870'
        daily_df = self.data_manager.cache_daily_ohlcv(safe_asset_code, data_fetch_start, end_date)
        backtester.add_daily_data(safe_asset_code, daily_df)
        
        # 모든 종목 데이터 로딩
        stock_names = []
        for sector, stocks in sector_stocks.items():
            for stock_name, _ in stocks:
                stock_names.append(stock_name)
        
        for name in stock_names:
            code = self.api_client.get_stock_code(name)
            if code:
                daily_df = self.data_manager.cache_daily_ohlcv(code, data_fetch_start, end_date)
                if not daily_df.empty:
                    backtester.add_daily_data(code, daily_df)
    
    def run_grid_search(self, 
                       start_date: datetime.date, 
                       end_date: datetime.date,
                       sector_stocks: Dict[str, List[Tuple[str, str]]],
                       max_combinations: int = None) -> Dict[str, Any]:
        """
        그리드서치 최적화를 실행합니다.
        
        Args:
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            sector_stocks: 섹터별 종목 정보
            max_combinations: 최대 테스트할 조합 수 (None이면 모든 조합)
            
        Returns:
            최적화 결과
        """
        logger.info("그리드서치 최적화 시작")
        
        # 파라미터 조합 생성
        combinations = self.generate_parameter_combinations()
        
        # 최대 조합 수 제한
        if max_combinations and len(combinations) > max_combinations:
            combinations = combinations[:max_combinations]
            logger.info(f"최대 {max_combinations}개 조합으로 제한")
        
        # 각 조합에 대해 백테스트 실행
        successful_results = []
        failed_results = []
        
        for i, params in enumerate(combinations):
            logger.info(f"진행률: {i+1}/{len(combinations)} ({((i+1)/len(combinations)*100):.1f}%)")
            
            result = self.run_single_backtest(params, start_date, end_date, sector_stocks)
            
            if result['success']:
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        # 결과 분석
        optimization_result = self._analyze_results(successful_results, failed_results)
        
        logger.info(f"최적화 완료: 성공 {len(successful_results)}개, 실패 {len(failed_results)}개")
        
        return optimization_result
    
    def _analyze_results(self, 
                        successful_results: List[Dict], 
                        failed_results: List[Dict]) -> Dict[str, Any]:
        """
        최적화 결과를 분석합니다.
        """
        if not successful_results:
            logger.warning("성공한 백테스트 결과가 없습니다.")
            return {'error': 'No successful results'}
        
        # 성과 지표별 최고 결과 찾기
        best_by_sharpe = max(successful_results, 
                           key=lambda x: x['metrics'].get('sharpe_ratio', -999))
        best_by_return = max(successful_results, 
                           key=lambda x: x['metrics'].get('total_return', -999))
        best_by_win_rate = max(successful_results, 
                             key=lambda x: x['metrics'].get('win_rate', -999))
        best_by_profit_factor = max(successful_results, 
                                  key=lambda x: x['metrics'].get('profit_factor', -999))
        
        # 전체 결과를 DataFrame으로 변환
        results_df = pd.DataFrame([
            {
                'sma_short': r['params']['sma_params']['short_sma_period'],
                'sma_long': r['params']['sma_params']['long_sma_period'],
                'volume_ma': r['params']['sma_params']['volume_ma_period'],
                'num_stocks': r['params']['sma_params']['num_top_stocks'],
                'stop_loss': r['params']['stop_loss_params']['stop_loss_ratio'],
                'trailing_stop': r['params']['stop_loss_params']['trailing_stop_ratio'],
                'max_losing': r['params']['stop_loss_params']['max_losing_positions'],
                'total_return': r['metrics'].get('total_return', 0),
                'sharpe_ratio': r['metrics'].get('sharpe_ratio', 0),
                'win_rate': r['metrics'].get('win_rate', 0),
                'profit_factor': r['metrics'].get('profit_factor', 0),
                'max_drawdown': r['metrics'].get('mdd', 0),
                'annual_volatility': r['metrics'].get('annual_volatility', 0)
            }
            for r in successful_results
        ])
        
        # 통계 정보
        stats = {
            'total_combinations': len(successful_results) + len(failed_results),
            'successful_combinations': len(successful_results),
            'failed_combinations': len(failed_results),
            'success_rate': len(successful_results) / (len(successful_results) + len(failed_results)),
            'avg_return': results_df['total_return'].mean(),
            'avg_sharpe': results_df['sharpe_ratio'].mean(),
            'avg_win_rate': results_df['win_rate'].mean(),
            'std_return': results_df['total_return'].std(),
            'std_sharpe': results_df['sharpe_ratio'].std()
        }
        
        return {
            'best_by_sharpe': best_by_sharpe,
            'best_by_return': best_by_return,
            'best_by_win_rate': best_by_win_rate,
            'best_by_profit_factor': best_by_profit_factor,
            'results_dataframe': results_df,
            'statistics': stats,
            'all_results': successful_results,
            'failed_results': failed_results
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """
        최적화 결과를 파일로 저장합니다.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"grid_search_results_{timestamp}.json"
        
        # optimizer/results 폴더에 저장
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)
        csv_filepath = os.path.join(results_dir, filename.replace('.json', '.csv'))
        
        # JSON으로 저장할 수 있는 형태로 변환
        save_data = {
            'statistics': results['statistics'],
            'best_by_sharpe': {
                'params': results['best_by_sharpe']['params'],
                'metrics': results['best_by_sharpe']['metrics']
            },
            'best_by_return': {
                'params': results['best_by_return']['params'],
                'metrics': results['best_by_return']['metrics']
            },
            'best_by_win_rate': {
                'params': results['best_by_win_rate']['params'],
                'metrics': results['best_by_win_rate']['metrics']
            },
            'best_by_profit_factor': {
                'params': results['best_by_profit_factor']['params'],
                'metrics': results['best_by_profit_factor']['metrics']
            }
        }
        
        # DataFrame을 CSV로 저장
        results['results_dataframe'].to_csv(csv_filepath, index=False, encoding='utf-8-sig')
        
        # JSON 파일 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"결과 저장 완료: {filepath}, {csv_filepath}")
    
    def print_summary(self, results: Dict[str, Any]):
        """
        최적화 결과 요약을 출력합니다.
        """
        print("\n" + "="*60)
        print("그리드서치 최적화 결과 요약")
        print("="*60)
        
        stats = results['statistics']
        print(f"총 조합 수: {stats['total_combinations']}")
        print(f"성공한 조합: {stats['successful_combinations']}")
        print(f"실패한 조합: {stats['failed_combinations']}")
        print(f"성공률: {stats['success_rate']*100:.1f}%")
        print(f"평균 수익률: {stats['avg_return']*100:.2f}%")
        print(f"평균 샤프지수: {stats['avg_sharpe']:.2f}")
        print(f"평균 승률: {stats['avg_win_rate']*100:.1f}%")
        
        print("\n" + "-"*60)
        print("최고 성과 조합")
        print("-"*60)
        
        print(f"최고 샤프지수:")
        self._print_best_result(results['best_by_sharpe'], 'sharpe_ratio')
        
        print(f"\n최고 수익률:")
        self._print_best_result(results['best_by_return'], 'total_return')
        
        print(f"\n최고 승률:")
        self._print_best_result(results['best_by_win_rate'], 'win_rate')
        
        print(f"\n최고 수익비:")
        self._print_best_result(results['best_by_profit_factor'], 'profit_factor')
    
    def _print_best_result(self, result: Dict, metric_name: str):
        """
        최고 성과 결과를 출력합니다.
        """
        params = result['params']
        metrics = result['metrics']
        
        print(f"  SMA: {params['sma_params']['short_sma_period']}일/{params['sma_params']['long_sma_period']}일")
        print(f"  거래량MA: {params['sma_params']['volume_ma_period']}일")
        print(f"  종목수: {params['sma_params']['num_top_stocks']}개")
        print(f"  손절매: {params['stop_loss_params']['stop_loss_ratio']}%")
        print(f"  트레일링스탑: {params['stop_loss_params']['trailing_stop_ratio']}%")
        
        metric_value = metrics.get(metric_name, 0)
        if metric_name == 'total_return':
            print(f"  {metric_name}: {metric_value*100:.2f}%")
        elif metric_name == 'win_rate':
            print(f"  {metric_name}: {metric_value*100:.1f}%")
        else:
            print(f"  {metric_name}: {metric_value:.2f}")
        
        print(f"  샤프지수: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  MDD: {metrics.get('mdd', 0)*100:.2f}%")

if __name__ == "__main__":
    import logging
    from datetime import datetime

    # 로깅 설정 (콘솔 + 파일)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("logs/grid_search_optimizer_run.log", encoding='utf-8'),
                            logging.StreamHandler(sys.stdout)
                        ])

    # Creon API, DataManager, Reporter, StockSelector 등 초기화
    api_client = CreonAPIClient()
    data_manager = DataManager()
    db_manager = DBManager()
    reporter = Reporter(db_manager=db_manager)
    
    # 공통 설정 파일에서 sector_stocks 가져오기
    from config.sector_config import sector_stocks
    stock_selector = StockSelector(data_manager=data_manager, api_client=api_client, sector_stocks_config=sector_stocks)

    # 백테스트 기간 설정
    start_date = datetime(2025, 3, 1).date()
    end_date = datetime(2025, 4, 1).date()

    optimizer = GridSearchOptimizer(
        api_client=api_client,
        data_manager=data_manager,
        reporter=reporter,
        stock_selector=stock_selector,
        initial_cash=10_000_000
    )

    # 그리드서치 실행 (세밀 조정 테스트)
    results = optimizer.run_grid_search(
        start_date=start_date,
        end_date=end_date,
        sector_stocks=sector_stocks,
        max_combinations=100  # 50개 → 100개로 증가 (세밀 조정)
    )

    optimizer.save_results(results)
    optimizer.print_summary(results) 