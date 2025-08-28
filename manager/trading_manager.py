# manager/trading_manager.py (리팩토링 후)

import logging
from datetime import date, datetime
from typing import Dict, List, Set, Optional, Any
import pandas as pd
import json
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from manager.data_manager import DataManager
from util.indicators import calculate_performance_metrics
logger = logging.getLogger(__name__)

class TradingManager(DataManager):
    """
    실시간 거래 환경의 데이터 관리를 담당합니다.
    DataManager의 모든 기능을 그대로 상속받아 사용합니다.
    """
    def __init__(self, api_client, db_manager):
        super().__init__(api_client, db_manager)
        self.current_model_id: Optional[int] = None
        logger.info("TradingManager 초기화 완료.")

    def save_trading_trade(self, trade_data: Dict[str, Any]) -> bool:
        """거래 로그 저장시 trading_run, trading_performance 도 업데이트 """
        if self.db_manager.save_trading_trade(trade_data):
            #self.update_trading_run_performance(datetime.now().date(), trade_data)
            return True
        return False

    # def update_trading_run_performance(self, current_date: date, trade_data: Dict[str, Any]) -> bool:
    #     """
    #     [최종 수정본] 장 마감 후, DB 데이터를 기반으로 누적 성과를 집계하여
    #     trading_run, trading_performance 테이블을 업데이트합니다.
    #     """
    #     logger.info(f"--- {current_date} 자동매매 결과 집계 및 저장 시작 ---")
    #     try:

    #         # ▼▼▼ [수정] 누적 성과 계산을 위한 로직 변경 ▼▼▼
    #         # 2. 기존 누적 'run' 정보 조회
    #         model_id = trade_data['model_id']
    #         existing_run_df = self.db_manager.fetch_trading_run(model_id=model_id)

    #         # 3. 자본금 계산
    #         current_prices = self.api_client.get_current_prices_bulk(list(self.broker.get_current_positions().keys()))
    #         #final_capital = self.broker.get_portfolio_value(current_prices)

    #         # 4. 시작일, 최초/일일 투자금 결정
    #         if not existing_run_df.empty:
    #             # 기존 기록이 있는 경우: 최초 투자금과 시작일은 기존 값을 사용
    #             existing_run = existing_run_df.iloc[0]
    #             initial_capital_for_run = float(existing_run['initial_capital'])
    #             start_date_for_run = existing_run['start_date']
    #             # 어제의 최종 자본을 오늘의 시작 자본으로 사용
    #             daily_initial_capital = float(existing_run['final_capital'])
    #         else:
    #             # 최초 실행인 경우: 모든 값을 새로 설정
    #             initial_capital_for_run = self.broker.initial_cash
    #             start_date_for_run = current_date
    #             daily_initial_capital = self.broker.initial_cash

    #         # 5. 일일 및 누적 성과 지표 계산
    #         daily_profit_loss = final_capital - daily_initial_capital
    #         daily_return = daily_profit_loss / daily_initial_capital if daily_initial_capital > 0 else 0.0
            
    #         # 누적 손익 및 수익률은 '최초 투자금' 대비 '현재 최종 자본'으로 계산
    #         total_profit_loss_cumulative = final_capital - initial_capital_for_run
    #         cumulative_return = total_profit_loss_cumulative / initial_capital_for_run if initial_capital_for_run > 0 else 0.0

    #         # MDD 계산 (전체 자산 곡선 기준)
    #         performance_history_df = self.db_manager.fetch_trading_performance(model_id=model_id, end_date=current_date)
    #         equity_curve = pd.Series(dtype=float)
    #         if not performance_history_df.empty:
    #             # DB에서 조회한 과거 데이터로 Series 생성
    #             equity_curve = performance_history_df.set_index('date')['end_capital']
    #         # 오늘의 최종 자본을 자산 곡선에 추가
    #         equity_curve[pd.Timestamp(current_date).date()] = final_capital
            
    #         metrics = calculate_performance_metrics(equity_curve)
    #         max_drawdown = metrics.get('mdd', 0.0)
            
    #         # 6. 사용된 전략 정보 요약
    #         daily_strategy_names = ', '.join([s.__class__.__name__ for s in self.daily_strategies])
    #         daily_strategy_params_json = json.dumps({s.__class__.__name__: s.strategy_params for s in self.daily_strategies})

    #         # 7. trading_run 테이블에 저장할 '누적' 데이터 구성
    #         run_data = {
    #             'model_id': model_id,
    #             'start_date': start_date_for_run,       # 최초 시작일
    #             'end_date': current_date,               # 최종 거래일 (오늘)
    #             'initial_capital': initial_capital_for_run, # 최초 투자금
    #             'final_capital': final_capital,         # 현재 최종 자본
    #             'total_profit_loss': total_profit_loss_cumulative, # 누적 손익
    #             'cumulative_return': cumulative_return, # 누적 수익률
    #             'max_drawdown': max_drawdown,
    #             'strategy_daily': daily_strategy_names,
    #             'params_json_daily': daily_strategy_params_json,
    #             'trading_date': current_date # save_trading_run 내부에서 start/end date 설정에 사용
    #         }
    #         # save_trading_run은 내부적으로 start_date를 업데이트하지 않음
    #         self.db_manager.save_trading_run(run_data)

    #         # 8. trading_performance 테이블에 저장할 '일일' 데이터 구성
    #         performance_data = {
    #             'model_id': model_id,
    #             'date': current_date,
    #             'end_capital': final_capital,
    #             'daily_return': daily_return,
    #             'daily_profit_loss': daily_profit_loss,
    #             'cumulative_return': cumulative_return, # 그날까지의 누적 수익률
    #             'drawdown': max_drawdown # 그날까지의 MDD
    #         }
    #         # ▲▲▲ 수정 완료 ▲▲▲
    #         self.db_manager.save_trading_performance(performance_data)

    #         logger.info(f"--- {current_date} 자동매매 결과 저장 완료 ---")
    #         self.notifier.send_message(
    #             f"📈 {current_date} 장 마감\n"
    #             f" - 최종 자산: {final_capital:,.0f}원\n"
    #             f" - 당일 손익: {daily_profit_loss:,.0f}원 ({daily_return:.2%})\n"
    #             f" - 누적 수익률: {cumulative_return:.2%}\n"
    #             f" - MDD: {max_drawdown:.2%}"
    #         )

    #     except Exception as e:
    #         logger.error(f"일일 성과 기록 중 오류 발생: {e}", exc_info=True)
    #         self.notifier.send_message("🚨 일일 성과 기록 중 심각한 오류가 발생했습니다.")


    # TradingManager에만 특화된 기능이 필요할 경우 여기에 추가
    def save_trading_log(self, log_data: Dict[str, Any]) -> bool:
        """[래퍼] 거래 로그를 DB에 저장합니다."""
        return self.db_manager.save_trading_log(log_data)

    def fetch_trading_logs(self, start_date: date, end_date: date, stock_code: str = None) -> pd.DataFrame:
        """[래퍼] 특정 기간의 매매 로그를 조회합니다."""
        return self.db_manager.fetch_trading_logs(start_date, end_date, stock_code)

    def save_daily_portfolio(self, portfolio_data: Dict[str, Any]) -> bool:
        """[래퍼] 일별 포트폴리오 스냅샷을 DB에 저장합니다."""
        return self.db_manager.save_daily_portfolio(portfolio_data)
        
    def fetch_latest_daily_portfolio(self) -> Optional[Dict[str, Any]]:
        """[래퍼] 가장 최신 일별 포트폴리오 스냅샷을 조회합니다."""
        return self.db_manager.fetch_latest_daily_portfolio()

    # def save_current_position(self, position_data: Dict[str, Any]) -> bool:
    #     """[래퍼] 현재 보유 종목 정보를 DB에 저장/업데이트합니다."""
    #     return self.db_manager.save_current_position(position_data)

    def save_current_position(self, position_data: Dict[str, Any]) -> bool:
        """
        [최종 수정] 현재 보유 종목 정보를 저장/업데이트합니다.
        entry_date가 없으면 trading_trade 테이블에서 조회하여 채웁니다.
        """
        if not position_data.get('entry_date'):
            stock_code = position_data.get('stock_code')
            if stock_code:
                # DB에서 가장 최근의 'BUY' 거래 날짜를 조회합니다.
                latest_buy_date = self.db_manager.fetch_latest_buy_trade_date(stock_code)
                
                if latest_buy_date:
                    # 매수 기록이 있으면 해당 날짜로 entry_date를 설정합니다.
                    position_data['entry_date'] = latest_buy_date
                    logger.info(f"[{stock_code}]의 누락된 entry_date를 DB 기록({latest_buy_date})으로 복원했습니다.")
                else:
                    # 매수 기록이 없으면 (예: 시스템 도입 전 보유 종목), 오늘 날짜로 설정합니다.
                    position_data['entry_date'] = date.today()
        # ▲▲▲ 로직 적용 완료 ▲▲▲
        
        # 최종적으로 완성된 position_data를 DB에 저장합니다.
        return self.db_manager.save_current_position(position_data)