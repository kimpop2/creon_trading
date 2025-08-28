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

from manager.app_manager import AppManager
logger = logging.getLogger(__name__)

class TradingModel(QAbstractTableModel):
    """
    AppManager를 사용하여 데이터 로직을 처리하는 UI 모델 클래스.
    """
    def __init__(self, app_manager: AppManager, data=None, headers=None, display_headers=None):
        super().__init__()
        self.app_manager = app_manager
        # UI 모델 관련 속성
        self._data = pd.DataFrame()
        self._headers = [] 
        self._display_headers = []
        if data is not None:
            self.set_data(data, headers, display_headers)
        
        # 전체 백테스트 런 정보
        self.all_trading_runs = pd.DataFrame()
        self.run_strategy_params = {}

        # ▼▼▼ [수정] 식별자를 model_id와 trading_date로 변경 ▼▼▼
        self.current_model_id = None
        self.current_trading_date = None
        self.current_stock_code = None
        self.current_selected_date = None # 일봉차트에서 선택된 날짜
        self.current_trades = pd.DataFrame()
        # ▲▲▲ 수정 완료 ▲▲▲


        self.stock_dic = self.app_manager.get_stock_info_map()

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

    # def _load_stock_info_map(self):
    #     """TradingManager를 통해 종목코드-종목명 맵을 로드합니다."""
    #     logger.info("종목 정보 맵(딕셔너리)을 초기화합니다.")
    #     return self.trading_manager._load_stock_info_map()

    def load_all_trading_runs(self, progress_callback=None):
        if progress_callback: progress_callback(25, "자동매매 실행 정보를 조회하는 중입니다...")
        self.all_trading_runs = self.app_manager.get_trading_runs() # AppManager의 새 메서드 호출
        if progress_callback: progress_callback(70, "자동매매 실행 목록을 로딩하는 중입니다...")
        return self.all_trading_runs

    # ▼▼▼ [수정] JSON 문자열을 파싱하는 로직 추가 ▼▼▼
    def _get_strategy_params(self, model_id: int, trading_date: date):
        key = (model_id, trading_date)
        if key not in self.run_strategy_params:
            run_info = self.all_trading_runs[
                (self.all_trading_runs['model_id'] == model_id) & 
                (pd.to_datetime(self.all_trading_runs['trading_date']).dt.date == trading_date)
            ]
            if not run_info.empty:
                row = run_info.iloc[0]
                
                # --- JSON 문자열 파싱 로직 ---
                daily_params_str = row.get('params_json_daily', '{}')
                minute_params_str = row.get('params_json_minute', '{}')
                
                try:
                    # None이거나 비어있지 않은 문자열일 경우에만 파싱 시도
                    daily_params = json.loads(daily_params_str) if daily_params_str and isinstance(daily_params_str, str) else {}
                    minute_params = json.loads(minute_params_str) if minute_params_str and isinstance(minute_params_str, str) else {}
                except (json.JSONDecodeError, TypeError):
                    daily_params = {}
                    minute_params = {}
                # --- 파싱 로직 종료 ---

                self.run_strategy_params[key] = {
                    'daily': {'strategy_name': row.get('strategy_daily', ''), **daily_params},
                    'minute': {'strategy_name': row.get('strategy_minute', ''), **minute_params}
                }
        return self.run_strategy_params.get(key, {'daily': {}, 'minute': {}})
    # ▲▲▲ 수정 완료 ▲▲▲

    def set_selected_run(self, model_id: int, trading_date: date):
        if self.current_model_id != model_id or self.current_trading_date != trading_date:
            self.current_model_id = model_id
            self.current_trading_date = trading_date
            self.current_stock_code = None
            self.current_selected_date = None
            self.current_trades = self.app_manager.get_trading_trades(start_date=trading_date, end_date=trading_date)

    def set_selected_stock_code(self, stock_code: str):
        self.current_stock_code = stock_code
        self.current_selected_date = None

    def set_selected_daily_date(self, trade_date: date):
        self.current_selected_date = trade_date

    def load_performance_data(self, model_id: int, start_date: date, end_date: date) -> pd.DataFrame:
        return self.app_manager.get_trading_performance(model_id=model_id, start_date=start_date, end_date=end_date)
        
    def load_traded_stocks_summary(self, trading_date: date) -> pd.DataFrame:
        return self.app_manager.get_traded_stocks_summary_for_date(trading_date)

    def load_daily_chart_data(self, stock_code: str) -> tuple:
        if not self.current_model_id or not self.current_trading_date: return pd.DataFrame(), pd.DataFrame(), {}
        
        start_date = self.current_trading_date - pd.Timedelta(days=60)
        end_date = self.current_trading_date
        daily_params = self._get_strategy_params(self.current_model_id, self.current_trading_date)['daily']
        
        daily_ohlcv = self.app_manager.get_daily_ohlcv_with_indicators(stock_code, start_date, end_date, daily_params)
        trades_for_stock = self.current_trades[self.current_trades['stock_code'] == stock_code]
        return daily_ohlcv, trades_for_stock, daily_params

    def load_minute_chart_data(self, stock_code: str, trade_date: date) -> tuple:
        if not self.current_model_id or not self.current_trading_date: return pd.DataFrame(), pd.DataFrame(), {}
        minute_params = self._get_strategy_params(self.current_model_id, self.current_trading_date)['minute']
        minute_ohlcv = self.app_manager.get_minute_ohlcv_with_indicators(stock_code, trade_date, minute_params)
        
        trades_for_stock_and_date = self.current_trades[
            (self.current_trades['stock_code'] == stock_code) &
            (pd.to_datetime(self.current_trades['trade_datetime']).dt.date == trade_date)
        ]
        return minute_ohlcv, trades_for_stock_and_date, minute_params

    def search_trading_runs(self, search_text: str) -> pd.DataFrame:
        if not search_text: return self.all_trading_runs
        search_text = search_text.lower()
        return self.all_trading_runs[
            self.all_trading_runs['strategy_daily'].str.lower().str.contains(search_text, na=False)
        ]