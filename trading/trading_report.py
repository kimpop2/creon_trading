# trading/trading_reporter.py
# 설명: 백테스팅 결과 보고서 생성 및 DB 저장 관리
# 작성일: 2025-06-17 (수정일: 2025-07-06)

import logging
import datetime
import pandas as pd
import json
import uuid # run_id 생성을 위해 추가 (db_manager에서 생성하도록 변경됨)

from manager.db_manager import DBManager
from util.strategies_util import calculate_performance_metrics # 성능 지표 계산 함수 임포트

logger = logging.getLogger(__name__)

class TradingReport:
    """
    백테스팅 결과를 취합하고, 성능 지표를 계산하며,
    결과를 데이터베이스에 저장하는 역할을 담당합니다.
    """
    def __init__(self, db_manager: DBManager):
        self.db_manager = db_manager
        logger.info("TradingReport 초기화 완료.")

    def generate_and_save_report(self,
                                 start_date: datetime.date,
                                 end_date: datetime.date,
                                 initial_cash: float,
                                 portfolio_value_series: pd.Series,
                                 transaction_log: list,
                                 daily_strategy_name: str,
                                 minute_strategy_name: str,
                                 daily_strategy_params: dict,
                                 minute_strategy_params: dict) -> str:
        """
        백테스트의 최종 결과를 생성하고 데이터베이스에 저장합니다.
        
        Args:
            start_date (datetime.date): 백테스트 시작일.
            end_date (datetime.date): 백테스트 종료일.
            initial_cash (float): 초기 자본.
            portfolio_value_series (pd.Series): 날짜별 포트폴리오 가치 시계열 데이터.
            transaction_log (list): Broker로부터 받은 모든 거래 내역 로그.
            daily_strategy_name (str): 사용된 일봉 전략의 이름.
            minute_strategy_name (str): 사용된 분봉 전략의 이름.
            daily_strategy_params (dict): 사용된 일봉 전략의 파라미터.
            minute_strategy_params (dict): 사용된 분봉 전략의 파라미터.

        Returns:
            str: 저장된 백테스트 run_id.
        """
        if portfolio_value_series.empty:
            logger.warning("포트폴리오 가치 데이터가 없어 보고서를 생성할 수 없습니다.")
            return None

        # 1. 성능 지표 계산
        metrics = calculate_performance_metrics(portfolio_value_series, risk_free_rate=0.03)

        # 2. 백테스트 실행 정보 (run_id)를 먼저 DB에 저장
        run_data = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_cash,
            'final_capital': portfolio_value_series.iloc[-1],
            'total_profit_loss': portfolio_value_series.iloc[-1] - initial_cash,
            'cumulative_return': metrics['total_return'],
            'max_drawdown': metrics['mdd'],
            'strategy_daily': daily_strategy_name,
            'strategy_minute': minute_strategy_name,
            'params_json_daily': json.dumps(daily_strategy_params) if daily_strategy_params else None,
            'params_json_minute': json.dumps(minute_strategy_params) if minute_strategy_params else None,
        }
        
        # db_manager의 save_backtest_run 메서드 호출
        run_id = self.db_manager.save_backtest_run(run_data)
        if not run_id:
            logger.error("백테스트 실행 정보 저장에 실패하여 보고서 생성을 중단합니다.")
            return None
        logger.info(f"백테스트 실행 요약 (run_id: {run_id}) DB 저장 완료.")
        
        # 3. 거래 내역 (transaction_log) DB에 저장
        trade_records = []
        for trade in transaction_log:
            trade_type_str = trade[2].upper() # 'buy' 또는 'sell'
            trade_amount = trade[4] * trade[3] # quantity * price
            
            # TODO: realized_profit_loss와 entry_trade_id는 실제 구현 시 브로커에서 계산 및 추적 필요
            realized_profit_loss = 0 # Placeholder
            entry_trade_id = None # Placeholder

            trade_records.append({
                'run_id': run_id,
                'stock_code': trade[1],
                'trade_type': trade_type_str,
                'trade_price': trade[3],
                'trade_quantity': trade[4],
                'trade_amount': trade_amount, 
                'trade_datetime': trade[0],
                'commission': trade[5],
                'tax': 0, # 현재는 tax 계산 없음 (백테스트 Broker에서 구현 필요)
                'realized_profit_loss': realized_profit_loss,
                'entry_trade_id': entry_trade_id
            })
        if trade_records:
            # db_manager의 save_backtest_trade 메서드 호출
            self.db_manager.save_backtest_trade(trade_records)
            logger.info(f"{len(trade_records)}개의 거래 내역을 DB에 저장했습니다.")
        else:
            logger.info("저장할 거래 내역이 없습니다.")

        # 4. 일별 성능 지표 (performance_records) DB에 저장
        performance_records = []
        previous_day_portfolio_value = initial_cash

        for date_index, value in portfolio_value_series.items():
            current_date = date_index.date() # Pandas DatetimeIndex에서 date 객체 추출
            daily_profit_loss = value - previous_day_portfolio_value
            daily_return = daily_profit_loss / previous_day_portfolio_value if previous_day_portfolio_value != 0 else 0
            
            # MDD는 전체 시리즈에서 계산되므로, 일별 MDD는 임시로 현재까지의 최대 낙폭으로
            temp_portfolio_series = portfolio_value_series[:date_index] # 현재 날짜까지의 포트폴리오 가치
            temp_metrics = calculate_performance_metrics(temp_portfolio_series, risk_free_rate=0.03)

            performance_records.append({
                'run_id': run_id,
                'date': current_date,
                'end_capital': value,
                'daily_return': daily_return, 
                'daily_profit_loss': daily_profit_loss,
                'cumulative_return': temp_metrics['total_return'], 
                'drawdown': temp_metrics['mdd'] 
            })
            previous_day_portfolio_value = value

        if performance_records:
            # db_manager의 save_backtest_performance 메서드 호출
            self.db_manager.save_backtest_performance(performance_records)
            logger.info(f"{len(performance_records)}개의 일별 성능 지표를 DB에 저장했습니다.")
        else:
            logger.info("저장할 일별 성능 지표가 없습니다.")

        logger.info("\n=== 백테스트 최종 결과 요약 ===")
        logger.info(f"시작일: {start_date.isoformat()}")
        logger.info(f"종료일: {end_date.isoformat()}")
        logger.info(f"초기자금: {initial_cash:,.0f}원")
        logger.info(f"최종 포트폴리오 가치: {portfolio_value_series.iloc[-1]:,.0f}원")
        logger.info(f"총 수익률: {metrics['total_return']*100:.2f}%")
        logger.info(f"연간 수익률: {metrics['annual_return']*100:.2f}%")
        logger.info(f"연간 변동성: {metrics['annual_volatility']*100:.2f}%")
        logger.info(f"샤프 지수: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"최대 낙폭 (MDD): {metrics['mdd']*100:.2f}%")
        logger.info(f"승률: {metrics['win_rate']*100:.2f}%")
        logger.info(f"수익비 (Profit Factor): {metrics['profit_factor']:.2f}")

        return run_id