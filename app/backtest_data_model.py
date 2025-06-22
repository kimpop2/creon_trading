# app/backtest_data_model.py

import pandas as pd
from datetime import date
import sys
import os

# project_root를 sys.path에 추가하여 모듈 임포트 가능하게 함
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from manager.backtest_manager import BacktestManager

class BacktestDataModel:
    """
    백테스팅 관련 데이터 로직을 처리하는 모델 클래스입니다.
    BacktestManager를 통해 데이터베이스와 상호작용합니다.
    """
    def __init__(self, backtest_manager: BacktestManager):
        self.backtest_manager = backtest_manager
        
        # 전체 백테스트 런 정보
        self.all_backtest_runs = pd.DataFrame()
        self.run_strategy_params = {}

        # 현재 선택된 정보
        self.current_run_id = None
        self.current_stock_code = None
        self.current_trade_date = None
        
        # 현재 선택된 run_id에 종속된 데이터
        self.current_trades = pd.DataFrame()

    def load_all_backtest_runs(self, progress_callback=None):
        """모든 백테스트 실행 정보를 로드하고 캐싱합니다."""
        if progress_callback:
            progress_callback(25, "백테스트 실행 정보를 데이터베이스에서 조회하는 중입니다...")
        
        self.all_backtest_runs = self.backtest_manager.fetch_backtest_runs()
        
        if progress_callback:
            progress_callback(70, "백테스트 실행 목록을 로딩하는 중입니다...")
        
        return self.all_backtest_runs

    def _get_strategy_params(self, run_id: int):
        """특정 run_id의 전략 파라미터를 지연 로딩으로 가져옵니다."""
        if run_id not in self.run_strategy_params:
            run_info = self.all_backtest_runs[self.all_backtest_runs['run_id'] == run_id]
            if not run_info.empty:
                row = run_info.iloc[0]
                
                daily_params = row.get('params_json_daily', {})
                minute_params = row.get('params_json_minute', {})
                
                if not isinstance(daily_params, dict): 
                    daily_params = {}
                if not isinstance(minute_params, dict): 
                    minute_params = {}
                
                daily_params['strategy_name'] = row.get('strategy_daily', '')
                minute_params['strategy_name'] = row.get('strategy_minute', '')
                
                self.run_strategy_params[run_id] = {
                    'daily': daily_params,
                    'minute': minute_params
                }
        
        return self.run_strategy_params.get(run_id, {'daily': {}, 'minute': {}})

    def set_selected_run_id(self, run_id: int):
        """선택된 백테스트 run_id를 설정하고 관련 거래 데이터를 미리 로드합니다."""
        if self.current_run_id != run_id:
            self.current_run_id = run_id
            self.current_stock_code = None
            self.current_trade_date = None
            # run_id가 변경되면 거래내역을 새로 로드
            self.current_trades = self.backtest_manager.fetch_backtest_trades(run_id)

    def set_selected_stock_code(self, stock_code: str):
        """선택된 종목 코드를 설정합니다."""
        self.current_stock_code = stock_code
        self.current_trade_date = None

    def set_selected_daily_date(self, trade_date: date):
        """선택된 일봉 날짜를 설정합니다."""
        self.current_trade_date = trade_date

    def load_performance_data(self, run_id: int) -> pd.DataFrame:
        """선택된 run_id에 대한 일별 성능 데이터를 로드합니다."""
        return self.backtest_manager.fetch_backtest_performance(run_id)
        
    def load_traded_stocks_summary(self, run_id: int) -> pd.DataFrame:
        """선택된 run_id에 대한 종목별 매매 요약 정보를 로드합니다."""
        # 이 함수는 내부적으로 fetch_backtest_trades를 호출하지만,
        # DataManager의 캐시 덕분에 DB 부하는 적습니다.
        return self.backtest_manager.get_traded_stocks_summary(run_id)

    def load_daily_chart_data(self, stock_code: str) -> tuple:
        """선택된 종목의 일봉 차트 데이터를 로드합니다."""
        if not self.current_run_id:
            return pd.DataFrame(), pd.DataFrame(), {}
        
        run_info = self.all_backtest_runs[self.all_backtest_runs['run_id'] == self.current_run_id]
        if run_info.empty:
            return pd.DataFrame(), pd.DataFrame(), {}
            
        start_date = run_info['start_date'].iloc[0]
        end_date = run_info['end_date'].iloc[0]
        daily_params = self._get_strategy_params(self.current_run_id)['daily']
        
        daily_ohlcv = self.backtest_manager.fetch_daily_ohlcv_with_indicators(
            stock_code, start_date, end_date, daily_params
        )
        
        trades_for_stock = self.current_trades[self.current_trades['stock_code'] == stock_code]
        
        return daily_ohlcv, trades_for_stock, daily_params

    def load_minute_chart_data(self, stock_code: str, trade_date: date) -> tuple:
        """선택된 날짜의 분봉 차트 데이터를 로드합니다."""
        if not self.current_run_id or not trade_date:
            return pd.DataFrame(), pd.DataFrame(), {}

        minute_params = self._get_strategy_params(self.current_run_id)['minute']
        
        minute_ohlcv = self.backtest_manager.fetch_minute_ohlcv_with_indicators(
            stock_code, trade_date, minute_params
        )
        
        trades_for_stock_and_date = self.current_trades[
            (self.current_trades['stock_code'] == stock_code) &
            (pd.to_datetime(self.current_trades['trade_datetime']).dt.date == trade_date)
        ]
        
        return minute_ohlcv, trades_for_stock_and_date, minute_params

    def search_backtest_runs(self, search_text: str) -> pd.DataFrame:
        """전략 이름으로 백테스트 실행 목록을 검색합니다."""
        if not search_text:
            return self.all_backtest_runs

        search_text = search_text.lower()
        
        # 'strategy_daily' 또는 'strategy_minute' 컬럼에서 검색
        # pd.Series.str.contains()는 NaN 값을 처리하기 위해 na=False 옵션을 사용
        filtered_df = self.all_backtest_runs[
            self.all_backtest_runs['strategy_daily'].str.lower().str.contains(search_text, na=False) |
            self.all_backtest_runs['strategy_minute'].str.lower().str.contains(search_text, na=False)
        ]
        return filtered_df