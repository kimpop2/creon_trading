"""
OpenMinute 전략 테스트 스크립트
분봉 데이터 로딩 없이 최적화 성능 향상 테스트
"""

import datetime
import logging
import sys
import os
from typing import Dict, Any

# 현재 스크립트의 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.creon_api import CreonAPIClient
from manager.data_manager import DataManager
from manager.db_manager import DBManager
from backtest.reporter import Reporter
from selector.stock_selector import StockSelector
from backtest.backtester import Backtester
from strategies.sma_daily import SMADaily
from strategies.open_minute import OpenMinute
from strategy_params_config import get_strategy_params

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("open_minute_test.log", encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger(__name__)

class OpenMinuteTester:
    """OpenMinute 전략 테스트 클래스"""
    
    def __init__(self):
        self.api_client = None
        self.data_manager = None
        self.reporter = None
        self.stock_selector = None
        
    def initialize(self):
        """시스템 초기화"""
        logger.info("OpenMinute 전략 테스트 시스템 초기화 중...")
        
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
            ]
        }
        
        self.stock_selector = StockSelector(
            data_manager=self.data_manager,
            api_client=self.api_client,
            sector_stocks_config=sector_stocks
        )
        
        logger.info("OpenMinute 전략 테스트 시스템 초기화 완료")
    
    def test_open_minute_strategy(self, start_date: datetime.date, end_date: datetime.date):
        """OpenMinute 전략 테스트"""
        logger.info("=== OpenMinute 전략 테스트 시작 ===")
        
        # 백테스터 초기화
        backtester = Backtester(
            data_manager=self.data_manager,
            api_client=self.api_client,
            reporter=self.reporter,
            stock_selector=self.stock_selector,
            initial_cash=10_000_000
        )
        
        # SMA 일봉 전략 생성
        sma_params = get_strategy_params('sma_daily')
        sma_strategy = SMADaily(
            data_store=backtester.data_store,
            strategy_params=sma_params,
            broker=backtester.broker
        )
        
        # OpenMinute 분봉 전략 생성
        rsi_params = get_strategy_params('rsi_minute')
        open_minute_strategy = OpenMinute(
            data_store=backtester.data_store,
            strategy_params=rsi_params,
            broker=backtester.broker
        )
        
        # 전략 설정
        backtester.set_strategies(
            daily_strategy=sma_strategy,
            minute_strategy=open_minute_strategy
        )
        
        # 데이터 로딩 (일봉 데이터만)
        self._load_test_data(backtester, start_date, end_date)
        
        # 백테스트 실행
        portfolio_values, metrics = backtester.run(start_date, end_date)
        
        # 결과 출력
        self._print_test_results("OpenMinute 전략", metrics)
        
        return metrics
    
    def _load_test_data(self, backtester: Backtester, start_date: datetime.date, end_date: datetime.date):
        """테스트용 데이터 로딩 (일봉 데이터만)"""
        # 안전자산 데이터 로딩
        safe_asset_code = 'A439870'
        daily_df = self.data_manager.cache_daily_ohlcv(safe_asset_code, start_date, end_date)
        backtester.add_daily_data(safe_asset_code, daily_df)
        
        # 모든 종목 데이터 로딩 (일봉만)
        stock_names = []
        for sector, stocks in self.stock_selector.sector_stocks_config.items():
            for stock_name, _ in stocks:
                stock_names.append(stock_name)
        
        for name in stock_names:
            code = self.api_client.get_stock_code(name)
            if code:
                daily_df = self.data_manager.cache_daily_ohlcv(code, start_date, end_date)
                if not daily_df.empty:
                    backtester.add_daily_data(code, daily_df)
        
        logger.info(f"테스트 데이터 로딩 완료: {len(backtester.data_store['daily'])}개 종목 (일봉 데이터만)")
    
    def _print_test_results(self, strategy_name: str, metrics: Dict[str, Any]):
        """테스트 결과 출력"""
        logger.info(f"\n=== {strategy_name} 테스트 결과 ===")
        logger.info(f"총 수익률: {metrics.get('total_return', 0)*100:.2f}%")
        logger.info(f"연간 수익률: {metrics.get('annual_return', 0)*100:.2f}%")
        logger.info(f"샤프 지수: {metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"최대 낙폭: {metrics.get('max_drawdown', 0)*100:.2f}%")
        logger.info(f"승률: {metrics.get('win_rate', 0)*100:.1f}%")
        logger.info(f"총 거래 횟수: {metrics.get('total_trades', 0)}")
        logger.info(f"평균 보유 기간: {metrics.get('avg_holding_period', 0):.1f}일")
        logger.info("=" * 50)

def main():
    """메인 실행 함수"""
    logger.info("OpenMinute 전략 테스트를 시작합니다.")
    
    # 테스터 초기화
    tester = OpenMinuteTester()
    tester.initialize()
    
    # 테스트 기간 설정 (최근 1개월)
    end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=30)
    
    logger.info(f"테스트 기간: {start_date} ~ {end_date}")
    
    # OpenMinute 전략 테스트
    results = tester.test_open_minute_strategy(start_date, end_date)
    
    logger.info("OpenMinute 전략 테스트 완료!")

if __name__ == "__main__":
    main() 