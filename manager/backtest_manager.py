# manager/backtest_manager.py (리팩토링 후)

import logging
import pandas as pd

from datetime import datetime, date, timedelta
from typing import Dict, List, Set, Optional, Any

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from manager.data_manager import DataManager
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager

logger = logging.getLogger(__name__)

class BacktestManager(DataManager):
    """
    백테스팅 환경의 데이터 관리를 담당하며, API 사용 가능 여부에 따라 동작을 달리합니다.
    """
    def __init__(self, api_client: CreonAPIClient, db_manager: DBManager):
        super().__init__(api_client, db_manager)
        self.pykrx_master_df = None
        logger.info("BacktestManager 초기화 완료.")


    def _fetch_and_store_daily_range(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """[오버라이딩] API 사용이 가능할 때만 부모 클래스의 API 호출 로직을 사용합니다."""
        if self.api_client and self.api_client.is_connected():
            logger.debug(f"[{stock_code}] API 사용 가능. 부모 DataManager의 API 호출 로직을 사용합니다.")
            return super()._fetch_and_store_daily_range(stock_code, start_date, end_date)
        else:
            logger.warning(f"[{stock_code}] API 사용 불가. DB 데이터만으로 동작합니다. (신규 데이터 조회 없음)")
            return pd.DataFrame()


    def _fetch_and_store_minute_range(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """[오버라이딩] API 사용이 가능할 때만 부모 클래스의 API 호출 로직을 사용합니다."""
        if self.api_client and self.api_client.is_connected():
            return super()._fetch_and_store_minute_range(stock_code, start_date, end_date)
        else:
            logger.warning(f"[{stock_code}] API 사용 불가. DB 데이터만으로 동작합니다. (신규 데이터 조회 없음)")
            return pd.DataFrame()




    def save_backtest_run(self, run_info: dict) -> int:
        """[래퍼] 백테스트 실행 요약 정보를 DB에 저장합니다."""
        return self.db_manager.save_backtest_run(run_info)


    def save_backtest_trade(self, trade_data_list: list) -> bool:
        """[래퍼] 백테스트 개별 거래 내역을 DB에 저장합니다."""
        return self.db_manager.save_backtest_trade(trade_data_list)


    def save_backtest_performance(self, performance_data_list: list) -> bool:
        """[래퍼] 백테스트 일별 성과를 DB에 저장합니다."""
        return self.db_manager.save_backtest_performance(performance_data_list)

    def save_strategy_profiles(self, profiles_data_list: list) -> bool:
        """[래퍼] 백테스트 일별 성과를 DB에 저장합니다."""
        return self.db_manager.save_strategy_profiles(profiles_data_list)


    def fetch_backtest_run(self, run_id: int = None, start_date: date = None, end_date: date = None) -> pd.DataFrame:
        """[래퍼] DB에서 백테스트 실행 정보를 조회합니다."""
        return self.db_manager.fetch_backtest_run(run_id, start_date, end_date)
    
    def fetch_backtest_performance(self, run_id: int = None, start_date: date = None, end_date: date = None) -> pd.DataFrame:
        """[래퍼] DB에서 백테스트 일별 성능능 정보를 조회합니다."""
        return self.db_manager.fetch_backtest_performance(run_id, start_date, end_date)
    
    def fetch_daily_regimes(self, model_id: int = None) -> pd.DataFrame:
        """[래퍼] DB에서 백테스트 실행 정보를 조회합니다."""
        return self.db_manager.fetch_daily_regimes(model_id)

