# report_handler.py (신규 파일)

import abc
import logging
import pandas as pd
from datetime import datetime
from manager.db_manager import DBManager
from util.strategies_util import calculate_performance_metrics

logger = logging.getLogger(__name__)

# 1. 데이터 저장 전략 인터페이스 (사용자가 제안한 구조)
class AbstractReport(abc.ABC):
    """리포트 데이터를 특정 목적지(DB 테이블, 파일 등)에 저장하는 방식에 대한 '규격'을 정의합니다."""
    @abc.abstractmethod
    def save(self, report_data: dict, **kwargs):
        pass

# 2. 구체적인 저장 클래스 구현
class BacktestDB(AbstractReport):
    """백테스트 결과를 DB의 backtest_* 테이블에 저장하는 클래스입니다."""
    def __init__(self, db_manager: DBManager):
        self.db_manager = db_manager

    def save(self, report_data: dict, **kwargs):
        """계산된 리포트 데이터를 받아 백테스트용 테이블에 저장합니다."""
        logger.info("BacktestDBStorage: 백테스트 결과 저장을 시작합니다.")
        
        run_summary = report_data.get('summary')
        trades = report_data.get('trades')
        daily_performance = report_data.get('daily_performance')

        # [기존 backtest_report.py 로직 추출]
        # 2-1. 백테스트 실행 요약 정보 저장
        run_id = self.db_manager.save_backtest_run(run_summary)
        if not run_id:
            logger.error("백테스트 실행 정보(summary) 저장 실패!")
            return

        logger.info(f"백테스트 실행 요약 (run_id: {run_id}) DB 저장 완료.")

        # 2-2. 거래 내역 저장
        if trades:
            # 모든 거래 기록에 run_id 추가
            for trade in trades:
                trade['run_id'] = run_id
            self.db_manager.save_backtest_trade(trades)
            logger.info(f"{len(trades)}건의 거래 내역을 DB에 저장했습니다.")

        # 2-3. 일별 성과 저장
        if daily_performance:
            for performance in daily_performance:
                performance['run_id'] = run_id
            self.db_manager.save_backtest_performance(daily_performance)
            logger.info(f"{len(daily_performance)}건의 일별 성과를 DB에 저장했습니다.")
        
        return run_id


class TradingDB(AbstractReport):
    """자동매매 결과를 DB의 trading_log, daily_portfolio 테이블에 저장하는 클래스입니다."""
    def __init__(self, db_manager: DBManager):
        self.db_manager = db_manager

    def save(self, report_data: dict, **kwargs):
        """계산된 리포트 데이터를 받아 자동매매용 테이블에 저장합니다."""
        logger.info("LiveTradeDBStorage: 자동매매 결과 저장을 시작합니다.")
        
        portfolio_summary = report_data.get('summary')
        
        # [기존 trading_report.py의 역할 수행]
        # 자동매매 환경에서는 매일 장 마감 후, 하루의 최종 성과를 기록합니다.
        if portfolio_summary:
            # ReportGenerator가 생성한 summary 딕셔너리를 DB 테이블 컬럼에 맞게 변환
            live_portfolio_data = {
                'record_date': portfolio_summary.get('end_date'),
                'total_capital': portfolio_summary.get('final_capital'),
                'cash_balance': kwargs.get('cash_balance', 0), # 추가 정보는 kwargs로 전달
                'total_asset_value': portfolio_summary.get('final_capital') - kwargs.get('cash_balance', 0),
                'daily_profit_loss': portfolio_summary.get('daily_profit_loss', 0), # ReportGenerator에서 계산 필요
                'daily_return_rate': portfolio_summary.get('daily_return', 0), # ReportGenerator에서 계산 필요
                'cumulative_profit_loss': portfolio_summary.get('total_profit_loss'),
                'cumulative_return_rate': portfolio_summary.get('cumulative_return')
            }
            self.db_manager.save_daily_portfolio(live_portfolio_data)
            logger.info(f"{live_portfolio_data['record_date']}의 일일 포트폴리오 최종 상태를 DB에 저장했습니다.")

# 3. 통합된 리포트 생성기
class ReportGenerator:
    """
    원본 데이터를 받아 성능을 계산하고, 지정된 저장 전략에 따라 결과를 저장하는 통합 클래스.
    """
    def __init__(self, storage_strategy: AbstractReport):
        self.storage = storage_strategy

    def generate(self,
                 start_date: datetime.date,
                 end_date: datetime.date,
                 initial_cash: float,
                 portfolio_value_series: pd.Series,
                 transaction_log: list,
                 strategy_info: dict):
        """
        [수정됨] 리포트를 생성하고, 누락된 필드를 모두 포함하여 저장합니다.
        """
        if portfolio_value_series.empty:
            logger.warning("포트폴리오 가치 데이터가 없어 보고서를 생성할 수 없습니다.")
            return None

        # [공통 로직] 1. 성능 지표 계산
        metrics = calculate_performance_metrics(portfolio_value_series)

        # [공통 로직] 2. 저장할 데이터 객체 생성
        final_capital = portfolio_value_series.iloc[-1]
        summary_data = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_cash,
            'final_capital': final_capital,
            # [추가] total_profit_loss 필드 복원
            'total_profit_loss': final_capital - initial_cash,
            'cumulative_return': metrics.get('total_return', 0),
            'max_drawdown': metrics.get('mdd', 0),
            **strategy_info
        }

        trade_records = transaction_log

        # [수정] 일별 성과 기록 시 누락된 필드 추가
        performance_records = []
        previous_value = initial_cash
        for date, value in portfolio_value_series.items():
            daily_profit_loss = value - previous_value
            daily_return = daily_profit_loss / previous_value if previous_value != 0 else 0
            
            # [추가] 일별 누적 수익률 및 최대 낙폭 계산
            temp_series = portfolio_value_series[:date]
            cumulative_return_to_date = (value - initial_cash) / initial_cash if initial_cash != 0 else 0
            # MDD는 전체 시리즈를 기반으로 계산하는 것이 더 정확하지만,
            # 기존 로직과 동일하게 일별로 계산하려면 아래와 같이 할 수 있습니다.
            rolling_max = temp_series.expanding(min_periods=1).max()
            daily_drawdown = (temp_series - rolling_max) / rolling_max
            mdd_to_date = daily_drawdown.min() if not daily_drawdown.empty else 0
            
            performance_records.append({
                'date': date.date(),
                'end_capital': value,
                'daily_return': daily_return,
                # [추가] daily_profit_loss, cumulative_return, drawdown 필드 복원
                'daily_profit_loss': daily_profit_loss,
                'cumulative_return': cumulative_return_to_date,
                'drawdown': mdd_to_date
            })
            previous_value = value
            
        report_data = {
            'summary': summary_data,
            'trades': trade_records,
            'daily_performance': performance_records
        }
        
        # 3. 주입받은 저장 전략을 통해 저장 실행
        run_id = self.storage.save(report_data)

        # 4. 최종 결과 로깅 (기존과 동일)
        logger.info("\n=== 최종 결과 요약 ===")
        logger.info(f"기간: {start_date} ~ {end_date}")
        logger.info(f"초기자금: {initial_cash:,.0f}원 | 최종자산: {summary_data['final_capital']:,.0f}원")
        logger.info(f"총 수익률: {summary_data['cumulative_return']*100:.2f}% | 최대 낙폭 (MDD): {summary_data['max_drawdown']*100:.2f}%")
        logger.info(f"연간 수익률: {metrics.get('annual_return', 0)*100:.2f}% | 샤프 지수: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info("\n====================")
        return run_id