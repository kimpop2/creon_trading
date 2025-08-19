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
from config.settings import STRATEGY_CONFIGS, FETCH_DAILY_PERIOD, FETCH_MINUTE_PERIOD, COMMON_PARAMS
logger = logging.getLogger(__name__)

class BacktestManager(DataManager):
    """
    백테스팅 환경의 데이터 관리를 담당하며, API 사용 가능 여부에 따라 동작을 달리합니다.
    """
    def __init__(self, api_client: CreonAPIClient, db_manager: DBManager):
        super().__init__(api_client, db_manager)
        self.pykrx_master_df = None
        self.indicator_cache = {}
        logger.info("BacktestManager 초기화 완료.")

    # --- ▼ [신규 추가] 데이터 로딩을 총괄하는 메서드 ▼ ---
    def prepare_data_for_backtest(self, start_date: date, end_date: date) -> dict:
        """
        백테스팅 전체 기간에 필요한 모든 종목의 가격 데이터를 미리 로딩하여
        data_store 딕셔너리 형태로 반환합니다.
        """
        logging.info(f"--- 백테스트 데이터 사전 로딩 시작 ({start_date} ~ {end_date}) ---")
        
        data_store = {'daily': {}, 'minute': {}}
        
        # 1. 유니버스 종목 코드 결정
        universe_codes = set(self.get_universe_codes())
        if COMMON_PARAMS.get('market_index_code'):
            universe_codes.add(COMMON_PARAMS['market_index_code'])
        if COMMON_PARAMS.get('safe_asset_code'):
            universe_codes.add(COMMON_PARAMS['safe_asset_code'])

        # 2. 전체 기간 데이터 사전 로딩
        daily_start = start_date - timedelta(days=FETCH_DAILY_PERIOD)
        minute_start = start_date - timedelta(days=FETCH_MINUTE_PERIOD)
        
        all_trading_dates_set = set(self.get_all_trading_days(daily_start, end_date))
        
        for code in list(universe_codes):
            logging.info(f"데이터 로딩: {code} 일봉: ({daily_start} ~ {end_date})")
            daily_df = self.cache_daily_ohlcv(code, daily_start, end_date, all_trading_dates_set)
            if not daily_df.empty:
                data_store['daily'][code] = daily_df
            
            # 분봉 데이터는 필요할 경우에만 로드 (PassMinute가 아닐 때)
            # 여기서는 최적화를 위해 일봉만 로드하는 것으로 단순화할 수 있습니다.
            # minute_df = self.cache_minute_ohlcv(...)
            # if not minute_df.empty: ...

        logging.info(f"--- 모든 데이터 준비 완료 ---")
        return data_store

    # [신규 추가] 보조지표 사전 계산 메서드
    def precalculate_all_indicators_for_period(self, start_date: date, end_date: date):
        """
        주어진 기간에 대해 모든 유니버스 종목의 보조지표를 미리 계산하여 캐시에 저장합니다.
        '데이터 연속성' 문제를 해결하는 핵심적인 역할을 합니다.
        """
        logger.info(f"보조지표 사전 계산 시작 ({start_date} ~ {end_date})...")
        
        # settings.py에 정의된 모든 전략의 파라미터를 기반으로 필요한 SMA 기간들을 모두 수집
        required_sma_periods = set([5, 10, 17, 20, 60]) # 기본값
        for config in STRATEGY_CONFIGS.values():
            for param, value in config.get('default_params', {}).items():
                if 'period' in param:
                    required_sma_periods.add(value)
            for param, p_config in config.get('optimization_params', {}).items():
                 if 'period' in param:
                    required_sma_periods.add(p_config['min'])
                    required_sma_periods.add(p_config['max'])
        
        logger.info(f"필요한 SMA 기간: {sorted(list(required_sma_periods))}")
        
        # 지표 계산에 필요한 최대 기간만큼 과거 데이터를 더 불러옴
        max_lookback = max(required_sma_periods) if required_sma_periods else 200
        data_start_date = start_date - timedelta(days=max_lookback * 1.5) # 여유분 확보
        
        all_trading_dates_set = set(self.get_all_trading_days(data_start_date, end_date))
        universe_codes = self.get_universe_codes()

        for code in universe_codes:
            # 연속된 전체 기간의 데이터를 불러옴
            daily_df = self.cache_daily_ohlcv(code, data_start_date, end_date, all_trading_dates_set)
            if daily_df.empty:
                continue

            # 필요한 모든 SMA 지표 계산
            for period in required_sma_periods:
                daily_df[f'sma_{period}'] = daily_df['close'].rolling(window=period).mean()
                daily_df[f'volume_sma_{period}'] = daily_df['volume'].rolling(window=period).mean()

            # 계산된 데이터프레임을 캐시에 저장
            self.indicator_cache[code] = daily_df
        
        logger.info(f"총 {len(self.indicator_cache)}개 종목의 보조지표 사전 계산 완료.")

    def get_precalculated_data(self, stock_code: str) -> pd.DataFrame:
        """캐시된 보조지표 데이터를 반환합니다."""
        return self.indicator_cache.get(stock_code, pd.DataFrame())
    
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

