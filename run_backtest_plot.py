import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QComboBox, QTableView, QAbstractItemView, QLabel,
                             QGridLayout, QSplitter, QHeaderView)
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QColor
from PyQt5.QtCore import Qt, QModelIndex
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
import pandas as pd
import numpy as np
import datetime
import json
from decimal import Decimal # Decimal 타입 임포트
from manager.db_manager import DBManager

class BacktestVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Backtest Visualizer")
        self.setGeometry(100, 100, 1400, 900)

        self.set_matplotlib_font()

        self.db_manager = DBManager()

        self.init_ui()
        self.load_backtest_runs()

    def set_matplotlib_font(self):
        font_name = 'Malgun Gothic'
        if any(font.name == font_name for font in fm.fontManager.ttflist):
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Matplotlib font set to: {font_name}")
        else:
            print(f"Font '{font_name}' not found. Trying default font...")
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False

    def init_ui(self):
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        self.run_selector_label = QLabel("Select Backtest Run:")
        self.run_selector_combo = QComboBox(self)
        self.run_selector_combo.currentIndexChanged.connect(self.load_selected_run_data)
        top_layout.addWidget(self.run_selector_label)
        top_layout.addWidget(self.run_selector_combo)
        top_layout.addStretch(1)

        main_layout.addLayout(top_layout)

        main_splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        self.metrics_group_box = QWidget()
        self.metrics_layout = QGridLayout()
        self.metrics_group_box.setLayout(self.metrics_layout)
        self.metrics_labels = {}
        metrics_to_display = [
            "Run ID", "Start Date", "End Date", "Initial Capital", "Final Capital",
            "Total Profit/Loss", "Cumulative Return", "Max Drawdown",
            "Daily Strategy", "Minute Strategy", "Daily Strategy Params", "Minute Strategy Params"
        ]
        for i, metric in enumerate(metrics_to_display):
            row = i // 2
            col = (i % 2) * 2
            label_name = QLabel(f"{metric}:")
            label_value = QLabel("N/A")
            label_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.metrics_layout.addWidget(label_name, row, col)
            self.metrics_layout.addWidget(label_value, row, col + 1)
            self.metrics_labels[metric] = label_value
        left_layout.addWidget(self.metrics_group_box)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        left_layout.addWidget(self.canvas)

        main_splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        right_splitter = QSplitter(Qt.Vertical)

        self.daily_performance_label = QLabel("Daily Performance:")
        self.daily_performance_table = QTableView()
        self.daily_performance_model = QStandardItemModel()
        self.daily_performance_table.setModel(self.daily_performance_model)
        self.daily_performance_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.daily_performance_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.daily_performance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_splitter.addWidget(self.daily_performance_label)
        right_splitter.addWidget(self.daily_performance_table)

        self.trade_log_label = QLabel("Trade Log:")
        self.trade_log_table = QTableView()
        self.trade_log_model = QStandardItemModel()
        self.trade_log_table.setModel(self.trade_log_model)
        self.trade_log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.trade_log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.trade_log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_splitter.addWidget(self.trade_log_label)
        right_splitter.addWidget(self.trade_log_table)

        right_layout.addWidget(right_splitter)

        main_splitter.addWidget(right_panel)
        main_layout.addWidget(main_splitter)

        self.setLayout(main_layout)

        main_splitter.setSizes([int(self.width() * 0.5), int(self.width() * 0.5)])
        right_splitter.setSizes([int(self.height() * 0.5), int(self.height() * 0.5)])


    def load_backtest_runs(self):
        runs_df = self.db_manager.fetch_backtest_run()
        self.run_selector_combo.clear()
        if not runs_df.empty:
            for index, run in runs_df.iterrows():
                run_id = run['run_id']
                # DataFrame에서 datetime.date 객체를 다룰 때, strftime 이전에 pd.Timestamp로 명시적 변환
                # 또는 .to_pydatetime() 사용 (일부 버전에서 date 객체가 아닌 timestamp 객체로 넘어올 수 있음)
                start_date = run['start_date'].strftime('%Y-%m-%d') if pd.notna(run['start_date']) and isinstance(run['start_date'], (datetime.date, pd.Timestamp)) else "N/A"
                end_date = run['end_date'].strftime('%Y-%m-%d') if pd.notna(run['end_date']) and isinstance(run['end_date'], (datetime.date, pd.Timestamp)) else "N/A"
                daily_strat = run['strategy_daily'] if pd.notna(run['strategy_daily']) else ""
                minute_strat = run['strategy_minute'] if pd.notna(run['strategy_minute']) else ""

                display_text = f"Run ID: {run_id} | {start_date} ~ {end_date}"
                if daily_strat:
                    display_text += f" | Daily: {daily_strat}"
                if minute_strat:
                    display_text += f" | Minute: {minute_strat}"
                self.run_selector_combo.addItem(display_text, run_id)
        else:
            self.run_selector_combo.addItem("No backtest runs found.", None)

    def load_selected_run_data(self):
        selected_run_id = self.run_selector_combo.currentData()
        if selected_run_id is None:
            self.clear_display()
            return

        print(f"Loading data for Run ID: {selected_run_id}")

        run_df = self.db_manager.fetch_backtest_run(run_id=selected_run_id)
        if not run_df.empty:
            run_info = run_df.iloc[0].to_dict()
            self.update_summary_metrics(run_info)
        else:
            print(f"No summary data found for Run ID: {selected_run_id}")
            self.clear_summary_metrics()

        daily_perf_df = self.db_manager.fetch_backtest_performance(run_id=selected_run_id)
        self.update_daily_performance_table(daily_perf_df)
        self.plot_portfolio_value(daily_perf_df)

        trade_log_df = self.db_manager.fetch_backtest_trade(run_id=selected_run_id)
        self.update_trade_log_table(trade_log_df)

    def _format_numeric_value(self, value, format_spec, suffix=""):
        """
        숫자 값을 안전하게 포맷팅하고, NaN, None, 또는 유효하지 않은 숫자 문자열일 경우 'N/A'를 반환합니다.
        Decimal 타입이 들어와도 float으로 변환하여 처리합니다.
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        # Decimal 객체일 경우 float으로 명시적으로 변환 (Decimal은 float로의 직접 포맷팅이 까다로울 수 있음)
        if isinstance(value, Decimal):
            value = float(value) # Decimal을 float으로 변환
            
        try:
            # value가 문자열일 경우 float으로 변환을 시도 (예: "808712.11" 같은 문자열)
            # 이미 Decimal이 float으로 변환되었거나, 원래 float/int인 경우 이 부분은 통과
            numeric_value = float(value) if isinstance(value, str) else value
            
            return format_spec.format(numeric_value) + suffix
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert value '{repr(value)}' (Type: {type(value)}) to number or format it with '{format_spec}'. Error: {e}. Returning 'N/A'.")
            return "N/A"
        
    def update_summary_metrics(self, run_info: dict):
        start_date_str = run_info['start_date'].strftime('%Y-%m-%d') if 'start_date' in run_info and pd.notna(run_info['start_date']) and isinstance(run_info['start_date'], (datetime.date, pd.Timestamp)) else "N/A"
        end_date_str = run_info['end_date'].strftime('%Y-%m-%d') if 'end_date' in run_info and pd.notna(run_info['end_date']) and isinstance(run_info['end_date'], (datetime.date, pd.Timestamp)) else "N/A"

        daily_params = "N/A"
        # json.loads는 None이나 NaN을 처리하지 못하므로, 명시적으로 문자열인지 확인
        if isinstance(run_info.get('params_json_daily'), str) and pd.notna(run_info.get('params_json_daily')):
            try:
                decoded_params = json.loads(run_info['params_json_daily'])
                daily_params = json.dumps(decoded_params, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                daily_params = "Invalid JSON"

        minute_params = "N/A"
        if isinstance(run_info.get('params_json_minute'), str) and pd.notna(run_info.get('params_json_minute')):
            try:
                decoded_params = json.loads(run_info['params_json_minute'])
                minute_params = json.dumps(decoded_params, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                minute_params = "Invalid JSON"

        metrics_map = {
            "Run ID": run_info.get('run_id', 'N/A'),
            "Start Date": start_date_str,
            "End Date": end_date_str,
            "Initial Capital": self._format_numeric_value(run_info.get('initial_capital'), "{:,.0f}", "원"),
            "Final Capital": self._format_numeric_value(run_info.get('final_capital'), "{:,.0f}", "원"),
            "Total Profit/Loss": self._format_numeric_value(run_info.get('total_profit_loss'), "{:+,,.0f}", "원"),
            # 수익률과 MDD는 100을 곱하기 전에 먼저 숫자로 변환해야 함
            "Cumulative Return": self._format_numeric_value(run_info.get('cumulative_return', 0) * 100, "{:.2f}", "%"),
            "Max Drawdown": self._format_numeric_value(run_info.get('max_drawdown', 0) * 100, "{:.2f}", "%"),
            "Daily Strategy": run_info.get('strategy_daily', 'N/A'),
            "Minute Strategy": run_info.get('strategy_minute', 'N/A'),
            "Daily Strategy Params": daily_params,
            "Minute Strategy Params": minute_params,
        }

        for metric, value in metrics_map.items():
            if metric in self.metrics_labels:
                self.metrics_labels[metric].setText(str(value))

    def update_daily_performance_table(self, df: pd.DataFrame):
        self.daily_performance_model.clear()
        if df.empty:
            self.daily_performance_model.setHorizontalHeaderLabels(["No data available."])
            return

        headers = ["Date", "End Capital", "Daily Return", "Cumulative Return", "Drawdown"]
        self.daily_performance_model.setHorizontalHeaderLabels(headers)

        for row_idx, row_data in df.iterrows():
            date_str = row_data['date'].strftime('%Y-%m-%d') if pd.notna(row_data['date']) and isinstance(row_data['date'], (datetime.date, pd.Timestamp)) else "N/A"
            end_capital = self._format_numeric_value(row_data['end_capital'], "{:,.0f}", "원")
            daily_return = self._format_numeric_value(row_data['daily_return'] * 100, "{:.2f}", "%") # 100을 곱한 후 포맷팅
            cumulative_return = self._format_numeric_value(row_data['cumulative_return'] * 100, "{:.2f}", "%")
            drawdown = self._format_numeric_value(row_data['drawdown'] * 100, "{:.2f}", "%")

            items = [
                QStandardItem(date_str),
                QStandardItem(end_capital),
                QStandardItem(daily_return),
                QStandardItem(cumulative_return),
                QStandardItem(drawdown)
            ]
            self.daily_performance_model.appendRow(items)

    def update_trade_log_table(self, df: pd.DataFrame):
        self.trade_log_model.clear()
        if df.empty:
            self.trade_log_model.setHorizontalHeaderLabels(["No data available."])
            return

        headers = ["Date/Time", "Stock Code", "Type", "Price", "Quantity", "Amount", "Commission", "Profit/Loss"]
        self.trade_log_model.setHorizontalHeaderLabels(headers)

        for row_idx, row_data in df.iterrows():
            trade_datetime_str = row_data['trade_datetime'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row_data['trade_datetime']) and isinstance(row_data['trade_datetime'], (datetime.datetime, pd.Timestamp)) else "N/A"
            stock_code = row_data['stock_code'] if pd.notna(row_data['stock_code']) else ""
            trade_type = row_data['trade_type'] if pd.notna(row_data['trade_type']) else ""
            trade_price = self._format_numeric_value(row_data['trade_price'], "{:,.0f}")
            trade_quantity = self._format_numeric_value(row_data['trade_quantity'], "{:,.0f}")
            trade_amount = self._format_numeric_value(row_data['trade_amount'], "{:,.0f}")
            commission = self._format_numeric_value(row_data['commission'], "{:,.0f}")
            realized_profit_loss = self._format_numeric_value(row_data['realized_profit_loss'], "{:+,,.0f}")

            items = [
                QStandardItem(trade_datetime_str),
                QStandardItem(stock_code),
                QStandardItem(trade_type),
                QStandardItem(trade_price),
                QStandardItem(trade_quantity),
                QStandardItem(trade_amount),
                QStandardItem(commission),
                QStandardItem(realized_profit_loss)
            ]
            if trade_type.upper() == 'BUY':
                for item in items:
                    item.setBackground(QColor(230, 255, 230))
            elif trade_type.upper() == 'SELL':
                for item in items:
                    item.setBackground(QColor(255, 230, 230))
            self.trade_log_model.appendRow(items)

    def plot_portfolio_value(self, df: pd.DataFrame):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not df.empty and 'date' in df.columns and 'end_capital' in df.columns:
            # 'date'와 'end_capital' 컬럼에 NaN 값이 있는 행을 필터링
            valid_data = df.dropna(subset=['date', 'end_capital']).copy() # .copy()를 추가하여 SettingWithCopyWarning 방지

            # 'date' 컬럼이 문자열일 경우 datetime 객체로 변환 시도
            # DBManager에서 DictCursor 사용 시 datetime.date 객체로 오는 것이 일반적이나, 만약을 대비
            if not valid_data.empty:
                # 'date' 컬럼이 이미 datetime.date 또는 pd.Timestamp 객체인지 확인
                if not isinstance(valid_data['date'].iloc[0], (datetime.date, pd.Timestamp)):
                    valid_data['date'] = pd.to_datetime(valid_data['date'])
                
                # 'end_capital' 컬럼이 숫자가 아닐 경우 숫자로 변환 시도
                if not pd.api.types.is_numeric_dtype(valid_data['end_capital']):
                    valid_data['end_capital'] = pd.to_numeric(valid_data['end_capital'], errors='coerce')
                    valid_data.dropna(subset=['end_capital'], inplace=True) # 변환 후 NaN이 된 값 제거

            if not valid_data.empty:
                dates = valid_data['date'].tolist()
                portfolio_values = valid_data['end_capital'].tolist()

                ax.plot(dates, portfolio_values, label="Portfolio Value", color='blue')
                ax.set_title("Portfolio Value Over Time")
                ax.set_xlabel("Date")
                ax.set_ylabel("Portfolio Value (KRW)")
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()

                self.figure.autofmt_xdate()
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            else:
                ax.text(0.5, 0.5, "No valid portfolio value data to display after cleansing.",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
        else:
            ax.text(0.5, 0.5, "No portfolio value data (or missing columns) to display.",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray')

        self.canvas.draw()

    def clear_display(self):
        self.clear_summary_metrics()
        self.daily_performance_model.clear()
        self.daily_performance_model.setHorizontalHeaderLabels(["No data available."])
        self.trade_log_model.clear()
        self.trade_log_model.setHorizontalHeaderLabels(["No data available."])
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Select a backtest run to view data.",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        self.canvas.draw()

    def clear_summary_metrics(self):
        for label_widget in self.metrics_labels.values():
            label_widget.setText("N/A")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    plt.switch_backend('Qt5Agg')
    viewer = BacktestVisualizer()
    viewer.show()
    sys.exit(app.exec_())