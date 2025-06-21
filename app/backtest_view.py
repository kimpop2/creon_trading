# app/backtest_view.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QHeaderView, QLabel,
    QSplitter, QLineEdit, QPushButton, QSizePolicy, QMessageBox
)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt5.QtGui import QColor, QPalette

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from datetime import datetime, date, time
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator, AutoDateLocator
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import logging

# Matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class BacktestModel(QAbstractTableModel):
    """
    QTableView에 데이터를 표시하기 위한 모델 (2줄 레이아웃 지원)
    """
    def __init__(self, data=None, headers=None, display_headers=None):
        super().__init__()
        self._data = pd.DataFrame()
        # headers: 실제 데이터 컬럼명. 1줄 레이아웃에서는 리스트, 2줄 레이아웃에서는 튜플 리스트.
        self._headers = [] 
        self._display_headers = [] # View에 표시될 헤더
        if data is not None:
            self.set_data(data, headers, display_headers)

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

class BacktestView(QWidget):
    def __init__(self):
        super().__init__()
        # 영구적인 데이터 모델을 먼저 생성합니다.
        self.run_list_model = BacktestModel()
        self.performance_model = BacktestModel()
        self.traded_stocks_model = BacktestModel()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("백테스팅 결과 시각화 프로그램")
        self.setGeometry(100, 100, 1600, 1200) # 1600x1200 크기로 초기 창 크기 설정

        # Main Layout (Horizontal Splitter)
        main_splitter = QSplitter(Qt.Horizontal)
        self.setLayout(QHBoxLayout()) # Use a basic layout for the main window

        # Left Panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Left Top: Backtest Run List
        self.run_list_label = QLabel("백테스트 실행 목록")
        self.run_search_input = QLineEdit()
        self.run_search_input.setPlaceholderText("전략 이름으로 검색...")
        self.run_search_button = QPushButton("검색")
        
        search_layout = QHBoxLayout()
        search_layout.addWidget(self.run_search_input)
        search_layout.addWidget(self.run_search_button)

        self.run_table_view = QTableView()
        self.run_table_view.setModel(self.run_list_model) # 영구 모델 설정
        self.run_table_view.setSelectionBehavior(QTableView.SelectRows)
        self.run_table_view.setSelectionMode(QTableView.SingleSelection)
        self.run_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents) # 내용에 맞게 너비 조절
        self.run_table_view.verticalHeader().hide() # 행 번호 숨기기

        # 포커스를 잃어도 선택 색상 유지
        run_palette = self.run_table_view.palette()
        run_palette.setColor(QPalette.Inactive, QPalette.Highlight, run_palette.color(QPalette.Active, QPalette.Highlight))
        run_palette.setColor(QPalette.Inactive, QPalette.HighlightedText, run_palette.color(QPalette.Active, QPalette.HighlightedText))
        self.run_table_view.setPalette(run_palette)

        # 누적 수익률 차트를 백테스트 실행 목록 영역으로 이동
        self.performance_figure = Figure()
        self.performance_canvas = FigureCanvas(self.performance_figure)
        self.performance_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.performance_canvas.updateGeometry()

        left_layout.addWidget(self.run_list_label)
        left_layout.addLayout(search_layout)
        left_layout.addWidget(self.run_table_view, stretch=3)
        left_layout.addWidget(self.performance_canvas, stretch=2) # 차트를 목록 아래에 추가

        # Left Bottom: Backtest Performance List (테이블만 남김)
        left_bottom_panel = QWidget()
        left_bottom_layout = QVBoxLayout(left_bottom_panel)

        self.performance_label = QLabel("선택된 백테스트 일별 성능 (Run ID: N/A)")
        
        self.performance_table_view = QTableView()
        self.performance_table_view.setModel(self.performance_model) # 영구 모델 설정
        self.performance_table_view.setSelectionBehavior(QTableView.SelectRows)
        self.performance_table_view.setSelectionMode(QTableView.SingleSelection)
        self.performance_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.performance_table_view.verticalHeader().hide()

        # 포커스를 잃어도 선택 색상 유지
        perf_palette = self.performance_table_view.palette()
        perf_palette.setColor(QPalette.Inactive, QPalette.Highlight, perf_palette.color(QPalette.Active, QPalette.Highlight))
        perf_palette.setColor(QPalette.Inactive, QPalette.HighlightedText, perf_palette.color(QPalette.Active, QPalette.HighlightedText))
        self.performance_table_view.setPalette(perf_palette)

        left_bottom_layout.addWidget(self.performance_label)
        left_bottom_layout.addWidget(self.performance_table_view) # 차트는 제거
        
        # Left Panel Splitter
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.addWidget(left_panel)
        left_splitter.addWidget(left_bottom_panel)
        left_splitter.setSizes([self.height() * 2 // 3, self.height() // 3]) # 상단(목록+차트)과 하단(테이블) 비율 2:1

        main_splitter.addWidget(left_splitter)

        # Right Panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Right Top: Stock List & Daily Chart
        right_top_panel = QWidget()
        right_top_layout = QHBoxLayout(right_top_panel)
        right_top_layout.setContentsMargins(0, 0, 0, 0)

        # Right Top Left: Traded Stock List
        self.traded_stocks_label = QLabel("매매 종목 목록")
        self.traded_stocks_table_view = QTableView()
        self.traded_stocks_table_view.setModel(self.traded_stocks_model) # 영구 모델 설정
        self.traded_stocks_table_view.setSelectionBehavior(QTableView.SelectRows)
        self.traded_stocks_table_view.setSelectionMode(QTableView.SingleSelection)
        self.traded_stocks_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.traded_stocks_table_view.verticalHeader().hide()

        # 포커스를 잃어도 선택 색상 유지
        stock_palette = self.traded_stocks_table_view.palette()
        stock_palette.setColor(QPalette.Inactive, QPalette.Highlight, stock_palette.color(QPalette.Active, QPalette.Highlight))
        stock_palette.setColor(QPalette.Inactive, QPalette.HighlightedText, stock_palette.color(QPalette.Active, QPalette.HighlightedText))
        self.traded_stocks_table_view.setPalette(stock_palette)

        traded_stocks_sub_panel = QWidget()
        traded_stocks_sub_layout = QVBoxLayout(traded_stocks_sub_panel)
        traded_stocks_sub_layout.addWidget(self.traded_stocks_label)
        traded_stocks_sub_layout.addWidget(self.traded_stocks_table_view)
        
        right_top_layout.addWidget(traded_stocks_sub_panel)

        # Right Top Right: Daily Chart
        self.daily_chart_label = QLabel("종목 일봉 차트 (N/A)")
        self.daily_chart_figure = Figure()
        self.daily_chart_canvas = FigureCanvas(self.daily_chart_figure)
        self.daily_chart_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.daily_chart_canvas.updateGeometry()

        daily_chart_sub_panel = QWidget()
        daily_chart_sub_layout = QVBoxLayout(daily_chart_sub_panel)
        daily_chart_sub_layout.addWidget(self.daily_chart_label)
        daily_chart_sub_layout.addWidget(self.daily_chart_canvas)

        right_top_layout.addWidget(daily_chart_sub_panel)
        
        right_top_splitter = QSplitter(Qt.Horizontal)
        right_top_splitter.addWidget(traded_stocks_sub_panel)
        right_top_splitter.addWidget(daily_chart_sub_panel)
        # 종목 목록(index 0)은 늘어나지 않고, 차트(index 1)만 남은 공간을 모두 차지하도록 설정
        right_top_splitter.setStretchFactor(0, 0)
        right_top_splitter.setStretchFactor(1, 1)

        # Right Bottom: Minute Chart
        self.minute_chart_label = QLabel("선택 일자 분봉 차트 (N/A)")
        self.minute_chart_figure = Figure()
        self.minute_chart_canvas = FigureCanvas(self.minute_chart_figure)
        self.minute_chart_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.minute_chart_canvas.updateGeometry()

        minute_chart_sub_panel = QWidget()
        minute_chart_sub_layout = QVBoxLayout(minute_chart_sub_panel)
        minute_chart_sub_layout.addWidget(self.minute_chart_label)
        minute_chart_sub_layout.addWidget(self.minute_chart_canvas)

        # 오른쪽 패널을 수직 스플리터로 변경
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(right_top_splitter)
        right_splitter.addWidget(minute_chart_sub_panel)
        right_splitter.setSizes([self.height() // 2, self.height() // 2]) # 일봉/분봉 차트 비율 1:1

        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([self.width() // 3, self.width() * 2 // 3]) # 초기 비율 1:2

        self.layout().addWidget(main_splitter) # Set the main splitter to the window's layout

        self.show() # 지정된 크기로 실행

    def update_run_list(self, df: pd.DataFrame):
        """백테스트 실행 목록 테이블을 업데이트합니다."""
        # 2줄 헤더 설정
        headers = [
            ('start_date', 'end_date'),
            ('initial_capital', 'final_capital'),
            ('total_profit_loss', 'max_drawdown'),
            ('cumulative_return', 'annualized_return'),
            ('strategy_daily', 'strategy_minute')
        ]
        display_headers = [
            "일자\n(시작/종료)", "자본\n(초기/최종)", "손익\n최대낙폭", 
            "수익률\n연수익률", "전략\n(일봉/분봉)"
        ]
        
        # 테이블의 모델에 새로운 데이터를 설정하여 화면을 갱신
        self.run_list_model.set_data(df, headers, display_headers)
        self.run_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.run_table_view.resizeRowsToContents()

    def update_performance_list(self, df: pd.DataFrame, run_id: int):
        """일별 성능 목록 테이블을 업데이트하고 그래프를 그립니다."""
        self.performance_label.setText(f"선택된 백테스트 일별 성능 (Run ID: {run_id})")
        
        headers = [
            "performance_id", "run_id", "date", "end_capital", "daily_return", "daily_profit_loss", 
            "cumulative_return", "drawdown"
        ]
        display_headers = [
            "성능 ID", "실행 ID", "날짜", "일일 자본", "일일 수익률", "일일 손익", 
            "누적 수익률", "낙폭"
        ]
        self.performance_model.set_data(df, headers=headers, display_headers=display_headers)
        self.performance_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.performance_table_view.resizeRowsToContents()

        # 누적 수익률 그래프 업데이트
        self.performance_figure.clear()
        ax = self.performance_figure.add_subplot(111)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            ax.plot(df['date'], df['cumulative_return'], label='누적 수익률 (%)', color='blue')
            ax.set_title(f"Run ID {run_id} - 누적 수익률")
            ax.set_xlabel("날짜")
            ax.set_ylabel("누적 수익률 (%)")
            ax.grid(True)
            ax.legend()
            self.performance_figure.autofmt_xdate() # X축 날짜 라벨 자동 포맷
        self.performance_canvas.draw()

    def update_traded_stocks_list(self, df: pd.DataFrame):
        """매매 종목 목록 테이블을 업데이트합니다."""
        headers = [
            "stock_code", "trade_count", "total_realized_profit_loss", "avg_return_per_trade"
        ]
        display_headers = [
            "종목코드", "매매횟수", "총 실현손익", "평균수익률"
        ]
        self.traded_stocks_model.set_data(df, headers=headers, display_headers=display_headers)
        self.traded_stocks_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def update_daily_chart(self, ohlcv_df: pd.DataFrame, trades_df: pd.DataFrame, stock_code: str, daily_strategy_params: dict):
        """
        일봉 차트(가격+거래량)를 matplotlib을 사용하여 업데이트합니다.
        """
        logger.info(f"일봉 차트 업데이트 시작: 종목코드={stock_code}, 전달된 파라미터={daily_strategy_params}")
        logger.info(f"전달된 OHLCV 데이터 컬럼: {ohlcv_df.columns.tolist() if not ohlcv_df.empty else '비어 있음'}")

        self.daily_chart_label.setText(f"종목 일봉 차트 ({stock_code})")
        self.daily_chart_figure.clear()

        if ohlcv_df.empty:
            self.daily_chart_canvas.draw()
            return

        # 1. 차트 영역 분할
        if 'macd' in ohlcv_df.columns:
            # MACD가 있을 경우 3분할 (가격:MACD:거래량 = 5:2:1)
            gs = gridspec.GridSpec(3, 1, height_ratios=[5, 2, 1])
            ax_macd = self.daily_chart_figure.add_subplot(gs[1])
            ax_vol = self.daily_chart_figure.add_subplot(gs[2], sharex=ax_macd)
            ax_macd.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            ax_macd.set_ylabel("MACD", fontsize=10)
        else:
            # MACD가 없을 경우 2분할 (가격:거래량 = 5:1)
            gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
            ax_vol = self.daily_chart_figure.add_subplot(gs[1])
        
        ax1 = self.daily_chart_figure.add_subplot(gs[0], sharex=ax_vol)
        ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax1.set_ylabel("주가", fontsize=10)
        ax_vol.set_ylabel("거래량", fontsize=10)

        # 2. 가격 차트 그리기 (ax1)
        from mplfinance.original_flavor import candlestick_ohlc
        ohlc_data = ohlcv_df[['open', 'high', 'low', 'close']].copy()
        ohlc_data.reset_index(inplace=True)
        ohlc_data['Date'] = ohlc_data['Date'].map(plt.matplotlib.dates.date2num)
        candlestick_ohlc(ax1, ohlc_data.values, width=0.6, colorup='red', colordown='blue', alpha=0.8)

        # 3. 거래량 차트 그리기 (ax_vol)
        volume_colors = ['red' if c >= o else 'blue' for o, c in zip(ohlcv_df['open'], ohlcv_df['close'])]
        ax_vol.bar(ohlcv_df.index, ohlcv_df['volume'], color=volume_colors, width=0.8, alpha=0.7)
        ax_vol.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{int(x/1000):,}k'))

        # 4. 보조 지표 및 매매 시점 그리기 (ax1)
        strategy_name = daily_strategy_params.get('strategy_name')
        if strategy_name == 'SMADaily':
            short_period = daily_strategy_params.get('short_sma_period')
            long_period = daily_strategy_params.get('long_sma_period')
            if f'SMA_{short_period}' in ohlcv_df.columns:
                ax1.plot(ohlcv_df.index, ohlcv_df[f'SMA_{short_period}'], color='orange', linewidth=1, label=f'SMA {short_period}')
            if f'SMA_{long_period}' in ohlcv_df.columns:
                ax1.plot(ohlcv_df.index, ohlcv_df[f'SMA_{long_period}'], color='purple', linewidth=1, label=f'SMA {long_period}')
        elif strategy_name == 'TripleScreenDaily':
            ema_short_period = daily_strategy_params.get('ema_short_period', 12)
            ema_long_period = daily_strategy_params.get('ema_long_period', 26)
            if f'EMA_{ema_short_period}' in ohlcv_df.columns:
                ax1.plot(ohlcv_df.index, ohlcv_df[f'EMA_{ema_short_period}'], color='orange', linewidth=1, label=f'EMA {ema_short_period}')
            if f'EMA_{ema_long_period}' in ohlcv_df.columns:
                ax1.plot(ohlcv_df.index, ohlcv_df[f'EMA_{ema_long_period}'], color='purple', linewidth=1, label=f'EMA {ema_long_period}')
            
            # MACD 차트 그리기 (ax_macd)
            if 'macd' in ohlcv_df.columns and ax_macd:
                ax_macd.plot(ohlcv_df.index, ohlcv_df['macd'], color='blue', linewidth=1, label='MACD')
                ax_macd.plot(ohlcv_df.index, ohlcv_df['macd_signal'], color='red', linestyle='--', linewidth=1, label='Signal')
                histogram_colors = ['red' if h >= 0 else 'blue' for h in ohlcv_df['macd_histogram']]
                ax_macd.bar(ohlcv_df.index, ohlcv_df['macd_histogram'], color=histogram_colors, width=0.8, alpha=0.7, label='Histogram')
                ax_macd.axhline(0, color='gray', linestyle='-', linewidth=0.5)
                ax_macd.legend(loc='upper left', fontsize='small')
                ax_macd.grid(True, alpha=0.3)
        
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['trade_type'] == 'BUY']
            sell_trades = trades_df[trades_df['trade_type'] == 'SELL']
            buy_dates = [d.date() for d in buy_trades['trade_datetime']]
            sell_dates = [d.date() for d in sell_trades['trade_datetime']]
            for trade_date in buy_dates:
                ohlc_row = ohlcv_df[ohlcv_df.index.date == trade_date]
                if not ohlc_row.empty:
                    price = ohlc_row['low'].iloc[0] * 0.98
                    ax1.scatter(ohlc_row.index[0], price, marker='^', s=100, color='green', zorder=5)
            for trade_date in sell_dates:
                ohlc_row = ohlcv_df[ohlcv_df.index.date == trade_date]
                if not ohlc_row.empty:
                    price = ohlc_row['high'].iloc[0] * 1.02
                    ax1.scatter(ohlc_row.index[0], price, marker='v', s=100, color='red', zorder=5)

        # 5. 차트 스타일 및 레이아웃 설정
        ax1.set_title(f"{stock_code} 일봉 차트", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax_vol.grid(True, alpha=0.3)
        
        handles, labels = ax1.get_legend_handles_labels()
        if handles: ax1.legend()
        
        # X축 포맷터는 공유 X축의 최종인 ax_vol에 설정
        locator = AutoDateLocator(minticks=5, maxticks=10)
        ax_vol.xaxis.set_major_locator(locator)
        ax_vol.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        
        # 주가(%) Y축 설정 (ax2)
        ax2 = ax1.twinx()
        ref_price = ohlcv_df['open'].iloc[0]
        y1_lim = ax1.get_ylim()
        y2_lim = ((y1_lim[0] - ref_price) / ref_price * 100, (y1_lim[1] - ref_price) / ref_price * 100)
        ax2.set_ylim(y2_lim)
        ax2.set_ylabel("주가 (%)", fontsize=10)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
        
        ax1.yaxis.set_label_position('right')
        ax1.yaxis.tick_right()
        ax2.yaxis.set_label_position('left')
        ax2.yaxis.tick_left()
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))

        self.daily_chart_figure.autofmt_xdate()
        self.daily_chart_figure.tight_layout()
        self.daily_chart_figure.subplots_adjust(hspace=0.05) # 서브플롯 간격 조정
        self.daily_chart_canvas.draw()

    def update_minute_chart(self, ohlcv_df: pd.DataFrame, trades_df: pd.DataFrame, stock_code: str, trade_date: date, minute_strategy_params: dict):
        """
        분봉 차트(가격+거래량)를 matplotlib을 사용하여 업데이트합니다.
        """
        logger.info(f"분봉 차트 업데이트 시작: 종목코드={stock_code}, 날짜={trade_date}, 전달된 파라미터={minute_strategy_params}")
        logger.info(f"전달된 OHLCV 데이터 컬럼: {ohlcv_df.columns.tolist() if not ohlcv_df.empty else '비어 있음'}")
        
        self.minute_chart_label.setText(f"선택 일자 분봉 차트 ({stock_code} - {trade_date.strftime('%Y-%m-%d')})")
        self.minute_chart_figure.clear()

        if ohlcv_df.empty:
            self.minute_chart_canvas.draw()
            return

        # 1. 차트 영역 분할 (가격:거래량 = 5:1)
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
        ax1 = self.minute_chart_figure.add_subplot(gs[0])
        ax_vol = self.minute_chart_figure.add_subplot(gs[1], sharex=ax1) # X축 공유

        ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax1.set_ylabel("주가", fontsize=10)
        ax_vol.set_ylabel("거래량", fontsize=10)
        
        # 2. 가격 차트 그리기 (ax1)
        from mplfinance.original_flavor import candlestick_ohlc
        ohlc_data_numeric = ohlcv_df[['open', 'high', 'low', 'close', 'volume']].copy()
        ohlc_data_numeric.reset_index(inplace=True)
        ohlc_data_numeric['idx'] = range(len(ohlc_data_numeric))
        plot_data = ohlc_data_numeric[['idx', 'open', 'high', 'low', 'close']].values
        candlestick_ohlc(ax1, plot_data, width=0.6, colorup='red', colordown='blue', alpha=0.8)

        # 3. 거래량 차트 그리기 (ax_vol)
        volume_colors = ['red' if c >= o else 'blue' for o, c in zip(ohlc_data_numeric['open'], ohlc_data_numeric['close'])]
        ax_vol.bar(ohlc_data_numeric['idx'], ohlc_data_numeric['volume'], color=volume_colors, width=0.8, alpha=0.7)
        ax_vol.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{int(x/1000):,}k'))

        # 날짜 변경 지점 수직선
        date_col = 'Datetime' if 'Datetime' in ohlc_data_numeric.columns else 'index'
        ohlc_data_numeric[date_col] = pd.to_datetime(ohlc_data_numeric[date_col])
        dates = ohlc_data_numeric[date_col].dt.date
        unique_dates = dates.drop_duplicates()
        if len(unique_dates) > 1:
            second_day_start_index = dates[dates == unique_dates.iloc[1]].index[0]
            separator_idx = second_day_start_index - 0.5
            ax1.axvline(separator_idx, color='gray', linestyle='--', linewidth=1)
            ax_vol.axvline(separator_idx, color='gray', linestyle='--', linewidth=1)

        # 4. 보조 지표 및 매매 시점 그리기 (ax1)
        ax2 = None # RSI용 Y축
        strategy_name = minute_strategy_params.get('strategy_name')
        if strategy_name == 'RSIMinute' and 'RSI' in ohlcv_df.columns:
            ax2 = ax1.twinx()
            ax2.plot(range(len(ohlcv_df)), ohlcv_df['RSI'].values, color='magenta', linewidth=1, label=f'RSI')
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.axhline(minute_strategy_params.get('minute_rsi_oversold', 30), color='grey', linestyle='--', lw=0.8)
            ax2.axhline(minute_strategy_params.get('minute_rsi_overbought', 70), color='grey', linestyle='--', lw=0.8)

        if not trades_df.empty:
            trade_indices = []
            for dt in trades_df['trade_datetime']:
                time_diff = (ohlc_data_numeric[date_col] - dt).abs()
                match_idx = time_diff.idxmin()
                trade_indices.append(ohlc_data_numeric.loc[match_idx, 'idx'])
            trades_df['idx'] = trade_indices
            buy_trades = trades_df[trades_df['trade_type'] == 'BUY']
            sell_trades = trades_df[trades_df['trade_type'] == 'SELL']
            ax1.scatter(buy_trades['idx'], buy_trades['trade_price'] * 0.99, marker='^', s=150, color='green', zorder=5)
            ax1.scatter(sell_trades['idx'], sell_trades['trade_price'] * 1.01, marker='v', s=150, color='red', zorder=5)

        # 5. 차트 스타일 및 레이아웃 설정
        ax1.set_title(f"{stock_code} 분봉 차트", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax_vol.grid(True, alpha=0.3)

        # X축 레이블은 ax_vol에 설정
        tick_indices = [i for i, dt in enumerate(ohlc_data_numeric[date_col]) if dt.minute == 0 and dt.hour % 2 == 1]
        tick_labels = [ohlc_data_numeric[date_col].iloc[i].strftime('%m-%d\n%H:%M') for i in tick_indices]
        ax_vol.set_xticks(tick_indices)
        ax_vol.set_xticklabels(tick_labels)
        ax_vol.set_xlim(-1, len(ohlc_data_numeric))

        handles, labels = ax1.get_legend_handles_labels()
        if ax2:
            h2, l2 = ax2.get_legend_handles_labels()
            handles.extend(h2)
            labels.extend(l2)
        if handles: ax1.legend(handles, labels, loc='upper left')

        # 주가(%) Y축 설정 (ax3)
        ax3 = ax1.twinx()
        ref_price = ohlcv_df['open'].iloc[0]
        y1_lim = ax1.get_ylim()
        y3_lim = ((y1_lim[0] - ref_price) / ref_price * 100, (y1_lim[1] - ref_price) / ref_price * 100)
        ax3.set_ylim(y3_lim)
        ax3.set_ylabel("주가 (%)", fontsize=10)
        ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f%%'))

        # Y축 위치 정리
        ax1.yaxis.set_label_position('right')
        ax1.yaxis.tick_right()
        ax3.yaxis.set_label_position('left')
        ax3.yaxis.tick_left()
        if ax2: # RSI 축이 있다면
            ax2.yaxis.set_label_position('right')
            ax2.yaxis.tick_right()
            ax2.spines['right'].set_position(('outward', 50)) # 주가 축과 겹치지 않게 이동

        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
        
        self.minute_chart_figure.tight_layout()
        self.minute_chart_figure.subplots_adjust(hspace=0.05) # 서브플롯 간격 조정
        self.minute_chart_canvas.draw()

    def show_message_box(self, title, message):
        """사용자에게 메시지 박스를 표시합니다."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()