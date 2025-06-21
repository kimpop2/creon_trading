# app/backtest_controller.py

import pandas as pd
from datetime import date
import logging
from PyQt5.QtCore import QModelIndex, Qt

logger = logging.getLogger(__name__)

class BacktestController:
    def __init__(self, view, model):
        self.view = view
        self.model = model
        
        # 시그널을 먼저 연결하고, 그 다음에 데이터를 로드하여
        # 초기 선택 이벤트가 정상적으로 발생하도록 순서를 변경합니다.
        self._connect_signals()
        self.load_initial_data()

    def _connect_signals(self):
        """UI의 시그널을 컨트롤러의 슬롯에 연결합니다."""
        self.view.run_search_button.clicked.connect(self.on_run_search)
        self.view.run_table_view.selectionModel().selectionChanged.connect(self.on_run_selected)
        self.view.traded_stocks_table_view.selectionModel().selectionChanged.connect(self.on_stock_selected)
        self.view.performance_table_view.selectionModel().selectionChanged.connect(self.on_performance_date_selected)
        # 일봉 차트 클릭 이벤트 연결
        self.view.daily_chart_canvas.mpl_connect('button_press_event', self.on_daily_chart_click)

    def load_initial_data(self):
        """애플리케이션 시작 시 초기 데이터를 로드합니다."""
        all_runs_df = self.model.load_all_backtest_runs()
        self.view.update_run_list(all_runs_df)
        if not all_runs_df.empty:
            # 첫 번째 행을 자동으로 선택
            self.view.run_table_view.selectRow(0)

    def on_run_search(self):
        """백테스트 실행 목록 검색 버튼 클릭 시 호출됩니다."""
        search_text = self.view.run_search_input.text()
        filtered_df = self.model.search_backtest_runs(search_text)
        self.view.update_run_list(filtered_df)
        
        # 검색 결과에 따라 뷰 상태 초기화
        if filtered_df.empty:
            # 검색 결과가 없으면 모든 뷰를 초기화
            self.view.update_performance_list(pd.DataFrame(), -1)
            self.view.update_traded_stocks_list(pd.DataFrame())
            self.view.update_daily_chart(pd.DataFrame(), pd.DataFrame(), "N/A", {})
            self.view.update_minute_chart(pd.DataFrame(), pd.DataFrame(), "N/A", date.today(), {})

    def on_run_selected(self):
        """백테스트 실행 목록에서 특정 항목을 선택했을 때 호출됩니다."""
        selected_indexes = self.view.run_table_view.selectionModel().selectedRows()
        if not selected_indexes:
            return

        selected_row_index = selected_indexes[0].row()
        run_data = self.view.run_table_view.model().get_row_data(selected_row_index)
        
        if run_data is None:
            logger.warning(f"선택된 행({selected_row_index})의 데이터를 가져올 수 없습니다.")
            return

        run_id = run_data['run_id']
        self.model.set_selected_run_id(run_id)

        # 일별 성능 데이터 로드 및 뷰 업데이트
        performance_df = self.model.load_performance_data(run_id)
        self.view.update_performance_list(performance_df, run_id)

        # 매매 종목 요약 정보 로드 및 뷰 업데이트
        traded_stocks_df = self.model.load_traded_stocks_summary(run_id)
        self.view.update_traded_stocks_list(traded_stocks_df)
        
        if not traded_stocks_df.empty:
            # 첫 번째 종목을 자동으로 선택
            self.view.traded_stocks_table_view.selectRow(0)
        else:
            # 매매 종목이 없으면 차트 초기화
            self.view.update_daily_chart(pd.DataFrame(), pd.DataFrame(), "N/A", {})
            self.view.update_minute_chart(pd.DataFrame(), pd.DataFrame(), "N/A", date.today(), {})

    def on_stock_selected(self):
        """매매 종목 목록에서 특정 종목을 선택했을 때 호출됩니다."""
        selected_indexes = self.view.traded_stocks_table_view.selectionModel().selectedRows()
        
        stock_data = None
        if selected_indexes:
            # 사용자가 직접 선택한 경우
            selected_row_index = selected_indexes[0].row()
            stock_data = self.view.traded_stocks_table_view.model().get_row_data(selected_row_index)
        else:
            # 프로그램에 의해 첫 행이 선택되었으나 selectionModel에 반영되기 전일 경우, 첫 번째 행 데이터를 직접 가져옴
            model = self.view.traded_stocks_table_view.model()
            if model and model.rowCount() > 0:
                stock_data = model.get_row_data(0)

        if stock_data is None:
            # 차트 초기화 또는 오류 처리
            self.view.update_daily_chart(pd.DataFrame(), pd.DataFrame(), "N/A", {})
            self.view.update_minute_chart(pd.DataFrame(), pd.DataFrame(), "N/A", date.today(), {})
            return

        stock_code = stock_data['stock_code']
        self.model.set_selected_stock_code(stock_code)

        # 일봉 차트 데이터 로드 및 뷰 업데이트
        daily_df, trades_df, daily_params = self.model.load_daily_chart_data(stock_code)
        self.view.update_daily_chart(daily_df, trades_df, stock_code, daily_params)
        
        # 첫 번째 일별 성능 데이터를 기준으로 분봉 차트 업데이트
        performance_model = self.view.performance_table_view.model()
        if performance_model and performance_model.rowCount() > 0:
            self.view.performance_table_view.selectRow(0)
            self.on_performance_date_selected() # 슬롯 강제 호출
        else:
            self.view.update_minute_chart(pd.DataFrame(), pd.DataFrame(), "N/A", date.today(), {})

    def on_performance_date_selected(self):
        """일별 성능 목록에서 특정 날짜를 선택했을 때 호출됩니다."""
        selected_indexes = self.view.performance_table_view.selectionModel().selectedRows()
        
        performance_data = None
        if selected_indexes:
            # 사용자가 직접 선택한 경우
            selected_row_index = selected_indexes[0].row()
            performance_data = self.view.performance_table_view.model().get_row_data(selected_row_index)
        else:
            # 프로그램에 의해 첫 행이 선택되었으나 selectionModel에 반영되기 전일 경우, 첫 번째 행 데이터를 직접 가져옴
            model = self.view.performance_table_view.model()
            if model and model.rowCount() > 0:
                performance_data = model.get_row_data(0)

        if performance_data is None or self.model.current_stock_code is None:
            return
            
        trade_date = pd.to_datetime(performance_data['date']).date()
        self.model.set_selected_daily_date(trade_date)
        
        self.update_minute_chart_for_date(trade_date)

    def on_daily_chart_click(self, event):
        """일봉 차트에서 특정 날짜를 클릭했을 때 호출됩니다."""
        if event.xdata is None or self.model.current_stock_code is None:
            return
        
        # Matplotlib 날짜 숫자를 datetime 객체로 변환
        from matplotlib.dates import num2date
        trade_date = num2date(event.xdata).date()
        
        self.model.set_selected_daily_date(trade_date)
        self.update_minute_chart_for_date(trade_date)
        
        # 해당 날짜를 performance_table_view에서 찾아 선택
        model = self.view.performance_table_view.model()
        if model:
            for row in range(model.rowCount()):
                index = model.index(row, 2) # 'date' 컬럼
                date_in_table = pd.to_datetime(model.data(index, Qt.DisplayRole)).date()
                if date_in_table == trade_date:
                    self.view.performance_table_view.selectRow(row)
                    break

    def update_minute_chart_for_date(self, trade_date):
        """주어진 날짜에 대한 분봉 차트를 로드하고 업데이트합니다."""
        if self.model.current_stock_code is None:
            return

        minute_df, trades_df, minute_params = self.model.load_minute_chart_data(self.model.current_stock_code, trade_date)
        self.view.update_minute_chart(minute_df, trades_df, self.model.current_stock_code, trade_date, minute_params)