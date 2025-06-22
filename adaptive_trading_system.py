"""
실시간 적응형 매매 시스템
매일 최적화를 수행하여 최적 파라미터로 실시간 매매 실행
"""

import datetime
import logging
import sys
import os
import json
import schedule
import time
from typing import Dict, Any, Optional

# 현재 스크립트의 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.creon_api import CreonAPIClient
from manager.data_manager import DataManager
from manager.db_manager import DBManager
from trader.reporter import Reporter
from selector.stock_selector import StockSelector
from optimizer.progressive_refinement_optimizer import ProgressiveRefinementOptimizer, GridSearchStrategy
from strategies.sma_daily import SMADaily
from strategies.rsi_minute import RSIMinute
from trader.backtester import Backtester
from strategies.open_minute import OpenMinute

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("adaptive_trading_system.log", encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger(__name__)

class AdaptiveTradingSystem:
    """실시간 적응형 매매 시스템"""
    
    def __init__(self):
        self.api_client = None
        self.data_manager = None
        self.reporter = None
        self.stock_selector = None
        self.current_params = None
        self.optimization_history = []
        
    def initialize(self):
        """시스템 초기화"""
        logger.info("실시간 적응형 매매 시스템 초기화 중...")
        
        # 컴포넌트 초기화
        self.api_client = CreonAPIClient()
        if not self.api_client.connected:
            raise ConnectionError("Creon API 연결 실패")
        
        self.data_manager = DataManager()
        db_manager = DBManager()
        self.reporter = Reporter(db_manager=db_manager)
        
        # 섹터별 종목 설정
        sector_stocks = {
            '반도체': [
                ('삼성전자', 'IT'), ('SK하이닉스', 'IT'), ('DB하이텍', 'IT'),
                ('네패스아크', 'IT')
            ],
            '2차전지': [
                ('LG에너지솔루션', '2차전지'), ('삼성SDI', '2차전지'), ('SK이노베이션', '2차전지'),
                ('에코프로비엠', '2차전지'), ('포스코퓨처엠', '2차전지')
            ],
            '바이오': [
                ('삼성바이오로직스', '바이오'), ('셀트리온', '바이오'), ('SK바이오사이언스', '바이오'),
                ('유한양행', '바이오'), ('한미약품', '바이오')
            ],
            'IT': [
                ('NAVER', 'IT'), ('카카오', 'IT'), ('크래프톤', 'IT'),
                ('엔씨소프트', 'IT'), ('넷마블', 'IT')
            ],
            '자동차': [
                ('현대차', '자동차'), ('기아', '자동차'), ('현대모비스', '자동차'),
                ('한온시스템', '자동차')
            ]
        }
        
        self.stock_selector = StockSelector(
            data_manager=self.data_manager,
            api_client=self.api_client,
            sector_stocks_config=sector_stocks
        )
        
        # 기본 파라미터 설정 (시스템 시작 시)
        self.current_params = self._get_default_params()
        
        logger.info("실시간 적응형 매매 시스템 초기화 완료")
    
    def _get_default_params(self) -> Dict[str, Any]:
        """기본 파라미터 설정"""
        return {
            'sma_params': {
                'short_sma_period': 2,
                'long_sma_period': 10,
                'volume_ma_period': 3,
                'num_top_stocks': 4,
                'safe_asset_code': 'A439870'
            },
            'rsi_params': {
                'minute_rsi_period': 45,
                'minute_rsi_oversold': 30,
                'minute_rsi_overbought': 70
            },
            'stop_loss_params': {
                'stop_loss_ratio': -5.0,
                'trailing_stop_ratio': -3.0,
                'portfolio_stop_loss': -5.0,
                'early_stop_loss': -5.0,
                'max_losing_positions': 3
            }
        }
    
    def daily_optimization(self):
        """매일 최적화 수행"""
        try:
            logger.info("=== 일일 최적화 시작 ===")
            
            # 최적화 기간 설정 (지난 1개월)
            end_date = datetime.datetime.now().date()
            start_date = end_date - datetime.timedelta(days=30)
            
            logger.info(f"최적화 기간: {start_date} ~ {end_date}")
            
            # 섹터별 종목 설정
            sector_stocks = {
                '반도체': [
                    ('삼성전자', 'IT'), ('SK하이닉스', 'IT'), ('DB하이텍', 'IT')
                ],
                '2차전지': [
                    ('LG에너지솔루션', '2차전지'), ('삼성SDI', '2차전지'), ('SK이노베이션', '2차전지')
                ],
                '바이오': [
                    ('삼성바이오로직스', '바이오'), ('셀트리온', '바이오'), ('SK바이오사이언스', '바이오')
                ],
                'IT': [
                    ('NAVER', 'IT'), ('카카오', 'IT'), ('크래프톤', 'IT')
                ]
            }
            
            # 그리드서치 최적화 실행
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
                initial_combinations=30,  # 빠른 최적화를 위해 조합 수 줄임
                daily_strategy_name='sma_daily',
                minute_strategy_name='open_minute'
            )
            
            if results and results.get('best_params'):
                # 최적 파라미터 업데이트
                self.current_params = results['best_params']
                
                # 최적화 히스토리 저장
                optimization_record = {
                    'date': datetime.datetime.now().isoformat(),
                    'optimization_period': f"{start_date} ~ {end_date}",
                    'best_params': self.current_params,
                    'best_metrics': results.get('best_metrics', {}),
                    'optimization_method': 'Grid Search'
                }
                self.optimization_history.append(optimization_record)
                
                # 최적화 결과 저장
                self._save_optimization_results(results)
                
                logger.info("일일 최적화 완료 - 파라미터 업데이트됨")
                logger.info(f"샤프지수: {results.get('best_metrics', {}).get('sharpe_ratio', 0):.3f}")
                logger.info(f"수익률: {results.get('best_metrics', {}).get('total_return', 0)*100:.2f}%")
                
            else:
                logger.warning("최적화에서 유효한 결과를 찾지 못했습니다. 기존 파라미터 유지")
                
        except Exception as e:
            logger.error(f"일일 최적화 중 오류 발생: {str(e)}")
            logger.info("기존 파라미터로 계속 진행")
    
    def _save_optimization_results(self, results: Dict[str, Any]):
        """최적화 결과 저장"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{timestamp}.json"
        
        save_data = {
            'optimization_date': datetime.datetime.now().isoformat(),
            'best_params': results.get('best_params', {}),
            'best_metrics': results.get('best_metrics', {}),
            'optimization_history': results.get('optimization_history', [])
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"최적화 결과 저장: {filename}")
    
    def execute_trading(self):
        """실시간 매매 실행"""
        try:
            logger.info("=== 실시간 매매 실행 ===")
            
            # 백테스터 초기화 (실시간 매매용)
            backtester = Backtester(
                data_manager=self.data_manager,
                api_client=self.api_client,
                reporter=self.reporter,
                stock_selector=self.stock_selector,
                initial_cash=10_000_000
            )
            
            # 최적화된 파라미터로 전략 생성
            sma_strategy = SMADaily(
                data_store=backtester.data_store,
                strategy_params=self.current_params['sma_params'],
                broker=backtester.broker
            )
            
            open_minute_strategy = OpenMinute(
                data_store=backtester.data_store,
                strategy_params=self.current_params['rsi_params'],
                broker=backtester.broker
            )
            
            # 전략 설정
            backtester.set_strategies(
                daily_strategy=sma_strategy,
                minute_strategy=open_minute_strategy
            )
            
            # 손절매 파라미터 설정
            backtester.set_broker_stop_loss_params(self.current_params['stop_loss_params'])
            
            # 섹터별 종목 설정
            sector_stocks = {
                '반도체': [
                    ('삼성전자', 'IT'), ('SK하이닉스', 'IT'), ('DB하이텍', 'IT')
                ],
                '2차전지': [
                    ('LG에너지솔루션', '2차전지'), ('삼성SDI', '2차전지'), ('SK이노베이션', '2차전지')
                ],
                '바이오': [
                    ('삼성바이오로직스', '바이오'), ('셀트리온', '바이오'), ('SK바이오사이언스', '바이오')
                ],
                'IT': [
                    ('NAVER', 'IT'), ('카카오', 'IT'), ('크래프톤', 'IT')
                ]
            }
            
            # 데이터 로딩
            self._load_trading_data(backtester, sector_stocks)
            
            # 당일 매매 실행
            today = datetime.datetime.now().date()
            portfolio_values, metrics = backtester.run(today, today)
            
            # 매매 결과 저장
            self._save_trading_results(today, metrics)
            
            logger.info("실시간 매매 실행 완료")
            
        except Exception as e:
            logger.error(f"실시간 매매 실행 중 오류 발생: {str(e)}")
    
    def _load_trading_data(self, backtester: Backtester, sector_stocks: Dict[str, list]):
        """매매용 데이터 로딩"""
        # 데이터 가져오기 시작일 (1개월 전)
        end_date = datetime.datetime.now().date()
        start_date = (end_date - datetime.timedelta(days=30)).replace(day=1)
        
        # 안전자산 데이터 로딩
        safe_asset_code = 'A439870'
        daily_df = self.data_manager.cache_daily_ohlcv(safe_asset_code, start_date, end_date)
        backtester.add_daily_data(safe_asset_code, daily_df)
        
        # 모든 종목 데이터 로딩
        stock_names = []
        for sector, stocks in sector_stocks.items():
            for stock_name, _ in stocks:
                stock_names.append(stock_name)
        
        for name in stock_names:
            code = self.api_client.get_stock_code(name)
            if code:
                daily_df = self.data_manager.cache_daily_ohlcv(code, start_date, end_date)
                if not daily_df.empty:
                    backtester.add_daily_data(code, daily_df)
    
    def _save_trading_results(self, trading_date: datetime.date, metrics: Dict[str, Any]):
        """매매 결과 저장"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trading_results_{trading_date}_{timestamp}.json"
        
        save_data = {
            'trading_date': trading_date.isoformat(),
            'current_params': self.current_params,
            'metrics': metrics,
            'optimization_history_count': len(self.optimization_history)
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"매매 결과 저장: {filename}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """현재 시스템 상태 조회"""
        return {
            'current_params': self.current_params,
            'optimization_history_count': len(self.optimization_history),
            'last_optimization': self.optimization_history[-1] if self.optimization_history else None,
            'system_status': 'Running'
        }
    
    def schedule_daily_tasks(self):
        """일일 작업 스케줄링"""
        # 매일 오전 8:30에 최적화 수행
        schedule.every().day.at("08:30").do(self.daily_optimization)
        
        # 매일 오전 9:00에 매매 실행
        schedule.every().day.at("09:00").do(self.execute_trading)
        
        logger.info("일일 작업 스케줄링 완료")
        logger.info("- 매일 08:30: 최적화 수행")
        logger.info("- 매일 09:00: 매매 실행")
    
    def run_scheduler(self):
        """스케줄러 실행"""
        logger.info("스케줄러 시작...")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 체크

def main():
    """실시간 적응형 매매 시스템 실행"""
    logger.info("실시간 적응형 매매 시스템을 시작합니다.")
    
    # 시스템 초기화
    trading_system = AdaptiveTradingSystem()
    trading_system.initialize()
    
    # 일일 작업 스케줄링
    trading_system.schedule_daily_tasks()
    
    # 스케줄러 실행
    trading_system.run_scheduler()

if __name__ == "__main__":
    main() 