# manager/trading_manager.py (리팩토링 후)

import logging
from datetime import date
from typing import Dict, List, Set, Optional, Any
import pandas as pd
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from manager.data_manager import DataManager

logger = logging.getLogger(__name__)

class TradingManager(DataManager):
    """
    실시간 거래 환경의 데이터 관리를 담당합니다.
    DataManager의 모든 기능을 그대로 상속받아 사용합니다.
    """
    def __init__(self, api_client, db_manager):
        super().__init__(api_client, db_manager)
        logger.info("TradingManager 초기화 완료.")

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

    def save_current_position(self, position_data: Dict[str, Any]) -> bool:
        """[래퍼] 현재 보유 종목 정보를 DB에 저장/업데이트합니다."""
        return self.db_manager.save_current_position(position_data)