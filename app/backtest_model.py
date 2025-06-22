import pandas as pd
from datetime import date
import sys
import os
import json
import logging
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt5.QtGui import QColor

# project_root를 sys.path에 추가하여 모듈 임포트 가능하게 함
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from manager.backtest_manager import BacktestManager
from manager.db_manager import DBManager
from manager.data_manager import DataManager

logger = logging.getLogger(__name__)

class BacktestModel(QAbstractTableModel):
    """
    백테스팅 관련 데이터 로직을 처리하는 통합 모델 클래스입니다.
    UI 표시와 비즈니스 로직을 모두 담당합니다.
    """
    def __init__(self, backtest_manager: BacktestManager = None, data=None, headers=None, display_headers=None):
        super().__init__()
        
        # UI 모델 관련 속성
        self._data = pd.DataFrame()
        self._headers = [] 
        self._display_headers = []
        if data is not None:
            self.set_data(data, headers, display_headers)
        
        # 비즈니스 로직 관련 속성
        self.backtest_manager = backtest_manager
        self.db_manager = DBManager()
        self.data_manager = DataManager()
        
        # 전체 백테스트 런 정보
        self.all_backtest_runs = pd.DataFrame()
        self.run_strategy_params = {}

        # 현재 선택된 정보
        self.current_run_id = None
        self.current_stock_code = None
        self.current_trade_date = None
        
        # 현재 선택된 run_id에 종속된 데이터
        self.current_trades = pd.DataFrame()

        self.stock_dic = self._load_stock_info_map()

    # ==================== UI 모델 메서드들 ====================
    
    def set_data(self, data: pd.DataFrame, headers: list = None, display_headers: list = None):
        """DataFrame을 모델 데이터로 설정합니다."""
        self.beginResetModel()
        self._data = data.copy() if data is not None else pd.DataFrame()
        self._headers = headers if headers else []
        if display_headers:
            self._display_headers = display_headers
        elif not self._data.empty:
            self._display_headers = self._data.columns.tolist()
        else:
            self._display_headers = []
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        # self._headers가 튜플 리스트 형태일 때만 2줄 레이아웃으로 간주
        if self._headers and isinstance(self._headers[0], tuple):
            return len(self._data) * 2
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._display_headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        # 2줄 레이아웃인지 확인하고, 실제 데이터의 인덱스(logical_row) 계산
        is_two_line_layout = self._headers and isinstance(self._headers[0], tuple)
        logical_row = index.row() // 2 if is_two_line_layout else index.row()

        # 역할(role)에 따라 다른 데이터 반환
        if role == Qt.BackgroundRole:
            # 홀수번째 데이터 묶음(1, 3, 5...)에 옅은 회색 배경 적용
            if logical_row % 2 == 1:
                return QColor("#f0f0f0")
            return None

        # 값(value)과 컬럼 이름(col_name) 계산
        col = index.column()
        col_name = None
        value = None
        
        try:
            if is_two_line_layout:
                row_type = index.row() % 2  # 0: 첫번째 줄, 1: 두번째 줄
                col_name = self._headers[col][row_type]
                if not col_name: return None
                value = self._data.iloc[logical_row][col_name]
            else: # 일반 1줄 레이아웃
                if self._headers:
                    col_name = self._headers[col]
                elif not self._data.empty:
                    col_name = self._data.columns[col]
                else: # 데이터가 없으면 아무것도 하지 않음
                    return None
                value = self._data.iloc[logical_row][col_name]
        except (KeyError, IndexError):
            return None

        # 데이터 포맷팅
        if role == Qt.DisplayRole:
            if col_name in ['strategy_daily', 'strategy_minute'] and isinstance(value, str):
                return value.replace('(', '\n(')

            if isinstance(value, (date, pd.Timestamp)):
                return value.strftime('%Y-%m-%d')
            
            if col_name in ['cumulative_return', 'max_drawdown', 'annualized_return', 'daily_return', 'drawdown', 'avg_return_per_trade']:
                if pd.isna(value): return ""
                return f"{value:.2f}%"
            elif col_name in ['initial_capital', 'final_capital', 'total_profit_loss', 'end_capital', 'daily_profit_loss', 'trade_price', 'trade_amount', 'commission', 'tax', 'realized_profit_loss', 'total_realized_profit_loss']:
                if pd.isna(value): return ""
                return f"{value:,.0f}"
            else:
                return str(value)
        
        elif role == Qt.TextAlignmentRole:
            if is_two_line_layout:
                return Qt.AlignCenter
            if col_name in ['run_id', 'initial_capital', 'final_capital', 'total_profit_loss', 'cumulative_return', 'max_drawdown',
                            'end_capital', 'daily_return', 'daily_profit_loss', 'drawdown',
                            'trade_count', 'total_realized_profit_loss', 'avg_return_per_trade',
                            'trade_price', 'trade_quantity', 'trade_amount', 'commission', 'tax', 'realized_profit_loss', 'performance_id']:
                return Qt.AlignRight | Qt.AlignVCenter
            else:
                return Qt.AlignLeft | Qt.AlignVCenter

        elif role == Qt.ForegroundRole:
            if col_name in ['cumulative_return', 'total_profit_loss', 'annualized_return', 'daily_return', 'daily_profit_loss', 'total_realized_profit_loss', 'avg_return_per_trade', 'realized_profit_loss']:
                if pd.notna(value):
                    return QColor(Qt.darkGreen) if value > 0 else QColor(Qt.darkRed)
            elif col_name in ['max_drawdown', 'drawdown']:
                if pd.notna(value) and value > 0:
                    return QColor(Qt.darkRed)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._display_headers[section]
        return None

    def get_row_data(self, row_index):
        is_two_line_layout = self._headers and isinstance(self._headers[0], tuple)
        logical_row = row_index // 2 if is_two_line_layout else row_index
        if 0 <= logical_row < len(self._data):
            return self._data.iloc[logical_row]
        return None

    def append_data(self, new_data: pd.DataFrame):
        """기존 데이터에 새로운 데이터를 추가합니다."""
        if new_data.empty:
            return

        is_two_line_layout = self._headers and isinstance(self._headers[0], tuple)
        
        current_row_count = len(self._data)
        additional_row_count = len(new_data)
        
        start_row_view = self.rowCount()
        end_row_view = start_row_view + (additional_row_count * 2 if is_two_line_layout else additional_row_count) - 1

        self.beginInsertRows(QModelIndex(), start_row_view, end_row_view)
        
        self._data = pd.concat([self._data, new_data], ignore_index=True)
        
        self.endInsertRows()

    # ==================== 비즈니스 로직 메서드들 ====================

    def _load_stock_info_map(self):
        """DataManager를 통해 종목코드-종목명 맵을 로드합니다."""
        logger.info("종목 정보 맵(딕셔너리)을 초기화합니다.")
        return self.data_manager.get_stock_info_map()

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
        """
        특정 백테스트 실행(run_id)에 대해 거래된 모든 종목의 요약 정보를 반환합니다.
        종목명을 포함하여 반환합니다.
        """
        # backtest_manager를 통해 요약 정보를 가져옵니다.
        # 이 함수는 내부에 'return_per_trade' 계산 로직을 포함하고 있습니다.
        summary_df = self.backtest_manager.get_traded_stocks_summary(run_id)
        
        # 종목명을 매핑
        if not summary_df.empty and self.stock_dic:
            summary_df['stock_name'] = summary_df['stock_code'].map(self.stock_dic).fillna(summary_df['stock_code'])
        elif not summary_df.empty:
            summary_df['stock_name'] = summary_df['stock_code']

        return summary_df

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