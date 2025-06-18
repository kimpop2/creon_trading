"""
새로운 전략들 테스트 스크립트
듀얼 모멘텀, 볼린저+RSI, 섹터 로테이션, 삼중창 시스템 전략 테스트
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
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.bollinger_rsi_daily import BollingerRSIDaily
from strategies.sector_rotation_daily import SectorRotationDaily
from strategies.triple_screen_daily import TripleScreenDaily
from strategy_params_config import get_strategy_params

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("new_strategies_test.log", encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger(__name__)

class NewStrategiesTester:
    """새로운 전략들 테스트 클래스"""
    
    def __init__(self):
        self.api_client = None
        self.data_manager = None
        self.reporter = None
        self.stock_selector = None
        
    def initialize(self):
        """시스템 초기화"""
        logger.info("새로운 전략 테스트 시스템 초기화 중...")
        
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
        
        logger.info("새로운 전략 테스트 시스템 초기화 완료")
    
    def test_dual_momentum_strategy(self, start_date: datetime.date, end_date: datetime.date):
        """듀얼 모멘텀 전략 테스트"""
        logger.info("=== 듀얼 모멘텀 전략 테스트 시작 ===")
        
        # 백테스터 초기화
        backtester = Backtester(
            data_manager=self.data_manager,
            api_client=self.api_client,
            reporter=self.reporter,
            stock_selector=self.stock_selector,
            initial_cash=10_000_000
        )
        
        # 듀얼 모멘텀 전략 생성
        dual_momentum_params = get_strategy_params('dual_momentum_daily')
        dual_momentum_strategy = DualMomentumDaily(
            data_store=backtester.data_store,
            strategy_params=dual_momentum_params,
            broker=backtester.broker
        )
        
        # 전략 설정
        backtester.set_strategies(
            daily_strategy=dual_momentum_strategy,
            minute_strategy=None
        )
        
        # 데이터 로딩
        self._load_test_data(backtester, start_date, end_date)
        
        # 백테스트 실행
        portfolio_values, metrics = backtester.run(start_date, end_date)
        
        # 결과 출력
        self._print_test_results("듀얼 모멘텀 전략", metrics)
        
        return metrics
    
    def test_bollinger_rsi_strategy(self, start_date: datetime.date, end_date: datetime.date):
        """볼린저 밴드 + RSI 전략 테스트"""
        logger.info("=== 볼린저 밴드 + RSI 전략 테스트 시작 ===")
        
        # 백테스터 초기화
        backtester = Backtester(
            data_manager=self.data_manager,
            api_client=self.api_client,
            reporter=self.reporter,
            stock_selector=self.stock_selector,
            initial_cash=10_000_000
        )
        
        # 볼린저 + RSI 전략 생성
        bollinger_rsi_params = get_strategy_params('bollinger_rsi_daily')
        bollinger_rsi_strategy = BollingerRSIDaily(
            data_store=backtester.data_store,
            strategy_params=bollinger_rsi_params,
            broker=backtester.broker
        )
        
        # 전략 설정
        backtester.set_strategies(
            daily_strategy=bollinger_rsi_strategy,
            minute_strategy=None
        )
        
        # 데이터 로딩
        self._load_test_data(backtester, start_date, end_date)
        
        # 백테스트 실행
        portfolio_values, metrics = backtester.run(start_date, end_date)
        
        # 결과 출력
        self._print_test_results("볼린저 밴드 + RSI 전략", metrics)
        
        return metrics
    
    def test_sector_rotation_strategy(self, start_date: datetime.date, end_date: datetime.date):
        """섹터 로테이션 전략 테스트"""
        logger.info("=== 섹터 로테이션 전략 테스트 시작 ===")
        
        # 백테스터 초기화
        backtester = Backtester(
            data_manager=self.data_manager,
            api_client=self.api_client,
            reporter=self.reporter,
            stock_selector=self.stock_selector,
            initial_cash=10_000_000
        )
        
        # 섹터 로테이션 전략 생성
        sector_rotation_params = get_strategy_params('sector_rotation_daily')
        sector_rotation_strategy = SectorRotationDaily(
            data_store=backtester.data_store,
            strategy_params=sector_rotation_params,
            broker=backtester.broker
        )
        
        # 전략 설정
        backtester.set_strategies(
            daily_strategy=sector_rotation_strategy,
            minute_strategy=None
        )
        
        # 데이터 로딩
        self._load_test_data(backtester, start_date, end_date)
        
        # 백테스트 실행
        portfolio_values, metrics = backtester.run(start_date, end_date)
        
        # 결과 출력
        self._print_test_results("섹터 로테이션 전략", metrics)
        
        return metrics
    
    def test_triple_screen_strategy(self, start_date: datetime.date, end_date: datetime.date):
        """삼중창 시스템 전략 테스트"""
        logger.info("=== 삼중창 시스템 전략 테스트 시작 ===")
        
        # 백테스터 초기화
        backtester = Backtester(
            data_manager=self.data_manager,
            api_client=self.api_client,
            reporter=self.reporter,
            stock_selector=self.stock_selector,
            initial_cash=10_000_000
        )
        
        # 삼중창 시스템 전략 생성
        triple_screen_params = get_strategy_params('triple_screen_daily')
        triple_screen_strategy = TripleScreenDaily(
            data_store=backtester.data_store,
            strategy_params=triple_screen_params,
            broker=backtester.broker
        )
        
        # 전략 설정
        backtester.set_strategies(
            daily_strategy=triple_screen_strategy,
            minute_strategy=None
        )
        
        # 데이터 로딩
        self._load_test_data(backtester, start_date, end_date)
        
        # 백테스트 실행
        portfolio_values, metrics = backtester.run(start_date, end_date)
        
        # 결과 출력
        self._print_test_results("삼중창 시스템 전략", metrics)
        
        return metrics
    
    def _load_test_data(self, backtester: Backtester, start_date: datetime.date, end_date: datetime.date):
        """테스트용 데이터 로딩"""
        # 안전자산 데이터 로딩
        safe_asset_code = 'A439870'
        daily_df = self.data_manager.cache_daily_ohlcv(safe_asset_code, start_date, end_date)
        backtester.add_daily_data(safe_asset_code, daily_df)
        
        # KOSPI 지수 데이터 로딩 (듀얼 모멘텀 전략용)
        kospi_code = 'A001'
        kospi_df = self.data_manager.cache_daily_ohlcv(kospi_code, start_date, end_date)
        backtester.add_daily_data(kospi_code, kospi_df)
        
        # 모든 종목 데이터 로딩
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
        
        logger.info(f"테스트 데이터 로딩 완료: {len(backtester.data_store['daily'])}개 종목")
    
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
    
    def compare_all_strategies(self, start_date: datetime.date, end_date: datetime.date):
        """모든 전략 비교 테스트"""
        logger.info("=== 모든 전략 비교 테스트 시작 ===")
        
        results = {}
        
        # 각 전략 테스트
        try:
            results['dual_momentum'] = self.test_dual_momentum_strategy(start_date, end_date)
        except Exception as e:
            logger.error(f"듀얼 모멘텀 전략 테스트 실패: {str(e)}")
            results['dual_momentum'] = {}
        
        try:
            results['bollinger_rsi'] = self.test_bollinger_rsi_strategy(start_date, end_date)
        except Exception as e:
            logger.error(f"볼린저+RSI 전략 테스트 실패: {str(e)}")
            results['bollinger_rsi'] = {}
        
        try:
            results['sector_rotation'] = self.test_sector_rotation_strategy(start_date, end_date)
        except Exception as e:
            logger.error(f"섹터 로테이션 전략 테스트 실패: {str(e)}")
            results['sector_rotation'] = {}
        
        try:
            results['triple_screen'] = self.test_triple_screen_strategy(start_date, end_date)
        except Exception as e:
            logger.error(f"삼중창 시스템 전략 테스트 실패: {str(e)}")
            results['triple_screen'] = {}
        
        # 결과 비교
        self._print_comparison_results(results)
        
        return results
    
    def _print_comparison_results(self, results: Dict[str, Dict[str, Any]]):
        """전략 비교 결과 출력"""
        logger.info("\n=== 전략 비교 결과 ===")
        logger.info(f"{'전략명':<20} {'수익률':<10} {'샤프지수':<10} {'최대낙폭':<10} {'승률':<10}")
        logger.info("-" * 70)
        
        for strategy_name, metrics in results.items():
            if metrics:
                total_return = metrics.get('total_return', 0) * 100
                sharpe = metrics.get('sharpe_ratio', 0)
                max_dd = metrics.get('max_drawdown', 0) * 100
                win_rate = metrics.get('win_rate', 0) * 100
                
                logger.info(f"{strategy_name:<20} {total_return:>8.2f}% {sharpe:>8.3f} "
                           f"{max_dd:>8.2f}% {win_rate:>8.1f}%")
        
        logger.info("=" * 70)

def main():
    """메인 실행 함수"""
    logger.info("새로운 전략 테스트를 시작합니다.")
    
    # 테스터 초기화
    tester = NewStrategiesTester()
    tester.initialize()
    
    # 테스트 기간 설정 (최근 3개월)
    end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=90)
    
    logger.info(f"테스트 기간: {start_date} ~ {end_date}")
    
    # 모든 전략 비교 테스트
    results = tester.compare_all_strategies(start_date, end_date)
    
    logger.info("새로운 전략 테스트 완료!")

if __name__ == "__main__":
    main() 