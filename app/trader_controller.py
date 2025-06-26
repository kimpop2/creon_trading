# app/trader_controller.py

import pandas as pd
from datetime import date
import logging
from PyQt5.QtCore import QModelIndex, Qt, QTimer, QElapsedTimer
from PyQt5.QtWidgets import QProgressDialog

logger = logging.getLogger(__name__)

class TraderController:
    def __init__(self, view, model):
        self.view = view
        self.model = model
        self.loading_dialog = None
        self.is_loading = False  # 로딩 중 상태 플래그 추가
        
        # view의 stock_dic을 model의 stock_dic으로 설정
        self.view.stock_dic = self.model.stock_dic
        
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

    def show_loading_dialog(self, message="데이터를 로딩하는 중입니다..."):
        """로딩 다이얼로그를 표시합니다."""
        if self.loading_dialog is None:
            self.loading_dialog = QProgressDialog(message, None, 0, 100, self.view)
            self.loading_dialog.setWindowTitle("데이터 로딩 중...")
            self.loading_dialog.setWindowModality(Qt.WindowModal)
            self.loading_dialog.setCancelButton(None)
            self.loading_dialog.setMinimumDuration(0)
            
            # 창 크기 조정
            self.loading_dialog.resize(350, 120)
            
            # 타이머를 사용하여 프로그레스 바 애니메이션
            self.loading_timer = QTimer(self.loading_dialog)
            self.progress_value = 0

            def advance_progress():
                self.progress_value = (self.progress_value + 5) % 101 # 0~100 반복
                self.loading_dialog.setValue(self.progress_value)

            self.loading_timer.timeout.connect(advance_progress)
            self.loading_timer.start(200) # 0.2초마다 업데이트
        
        self.loading_dialog.setLabelText(message)
        self.loading_dialog.setValue(0)
        self.loading_dialog.show()
        from PyQt5.QtWidgets import QApplication
        QApplication.instance().processEvents()

    def hide_loading_dialog(self):
        """로딩 다이얼로그를 숨깁니다."""
        if self.loading_dialog:
            self.loading_timer.stop()
            self.loading_dialog.close()
            self.loading_dialog = None

    def load_initial_data(self):
        """애플리케이션 시작 시 초기 데이터를 로드합니다. 성능을 위해 점진적으로 로드합니다."""
        self.show_loading_dialog("백테스트 실행 목록을 로딩하는 중입니다...")
        try:
            def update_progress(value, message):
                if self.loading_dialog:
                    self.loading_dialog.setValue(value)
                    self.loading_dialog.setLabelText(message)
                    from PyQt5.QtWidgets import QApplication
                    QApplication.instance().processEvents()
            
            all_runs_df = self.model.load_all_trader_runs(progress_callback=update_progress)

            if all_runs_df.empty:
                self.view.update_run_list(all_runs_df)
                self.hide_loading_dialog()
                return

            # 1. 최초 50건만 먼저 로드하여 UI 반응성 확보
            initial_load_df = all_runs_df.iloc[:50]
            self.view.update_run_list(initial_load_df)
            
            self.hide_loading_dialog()

            # 2. 첫 번째 행을 자동으로 선택하여 다른 UI 요소들의 로딩을 유발
            if not initial_load_df.empty:
                self.view.run_table_view.selectRow(0)

            # 3. 나머지 데이터를 백그라운드에서 로드하도록 스케줄링
            def load_remaining_data():
                logger.info("나머지 백테스트 실행 목록을 백그라운드에서 로드합니다.")
                
                # 현재 선택된 run_id를 저장
                selected_indexes = self.view.run_table_view.selectionModel().selectedRows()
                selected_run_id = None
                if selected_indexes:
                    run_data = self.view.run_table_view.model().get_row_data(selected_indexes[0].row())
                    if run_data is not None:
                        selected_run_id = run_data['run_id']

                # 전체 데이터로 모델을 업데이트
                self.view.update_run_list(all_runs_df)

                # 이전 선택을 복원
                if selected_run_id is not None:
                    matches = all_runs_df.index[all_runs_df['run_id'] == selected_run_id].tolist()
                    if matches:
                        model = self.view.run_table_view.model()
                        try:
                            logical_row = all_runs_df.index.get_loc(matches[0])
                            is_two_line_layout = model._headers and isinstance(model._headers[0], tuple)
                            view_row = logical_row * 2 if is_two_line_layout else logical_row
                            self.view.run_table_view.selectRow(view_row)
                            self.view.run_table_view.scrollTo(model.index(view_row, 0))
                        except (KeyError, IndexError):
                            pass

            remaining_df = all_runs_df.iloc[50:]
            if not remaining_df.empty:
                QTimer.singleShot(100, load_remaining_data)

        except Exception as e:
            logger.error(f"초기 데이터 로딩 중 오류 발생: {e}")
            self.hide_loading_dialog()

    def on_run_search(self):
        """백테스트 실행 목록 검색 버튼 클릭 시 호출됩니다."""
        search_text = self.view.run_search_input.text()
        self.show_loading_dialog("검색 결과를 로딩하는 중입니다...")
        try:
            filtered_df = self.model.search_trader_runs(search_text)
            self.view.update_run_list(filtered_df)
            
            # 검색 결과에 따라 뷰 상태 초기화
            if filtered_df.empty:
                # 검색 결과가 없으면 모든 뷰를 초기화
                self.view.update_performance_list(pd.DataFrame(), -1)
                self.view.update_traded_stocks_list(pd.DataFrame())
                self.view.update_daily_chart(pd.DataFrame(), pd.DataFrame(), "N/A", {})
                self.view.update_minute_chart(pd.DataFrame(), pd.DataFrame(), "N/A", date.today(), {})
        finally:
            self.hide_loading_dialog()

    def on_run_selected(self):
        """백테스트 실행 목록에서 특정 항목을 선택했을 때 호출됩니다."""
        if self.is_loading:
            return  # 이미 로딩 중이면 무시
        
        self.is_loading = True
        self.show_loading_dialog("백테스트 상세 데이터를 로딩하는 중입니다...")
        
        try:
            selected_indexes = self.view.run_table_view.selectionModel().selectedRows()
            if not selected_indexes:
                return

            run_data = self.view.run_table_view.model().get_row_data(selected_indexes[0].row())
            if run_data is None:
                return

            run_id = run_data['run_id']
            self.model.set_selected_run_id(run_id)

            # 데이터 로딩
            self.loading_dialog.setLabelText("일별 성능 및 종목 요약 로딩 중...")
            performance_df = self.model.load_performance_data(run_id)
            traded_stocks_df = self.model.load_traded_stocks_summary(run_id)

            # 뷰 업데이트
            self.view.update_performance_list(performance_df, run_id)
            self.view.update_traded_stocks_list(traded_stocks_df)
            
            if not traded_stocks_df.empty:
                # 첫 종목 차트 로딩
                self.loading_dialog.setLabelText("첫 종목 차트 데이터 로딩 중...")
                stock_code = traded_stocks_df.iloc[0]['stock_code']
                self.model.set_selected_stock_code(stock_code)
                
                daily_df, daily_trades, daily_params = self.model.load_daily_chart_data(stock_code)
                self.view.update_daily_chart(daily_df, daily_trades, stock_code, daily_params)

                # 첫 매매일자에 수직선, 없으면 첫 봉
                selected_date = None
                if not daily_trades.empty and 'trade_datetime' in daily_trades.columns:
                    first_trade_date = pd.to_datetime(daily_trades['trade_datetime']).dt.date.min()
                    if first_trade_date in daily_df.index.date:
                        selected_date = first_trade_date
                if selected_date is None and not daily_df.empty:
                    selected_date = daily_df.index[0].date()
                
                if selected_date:
                    self.model.set_selected_daily_date(selected_date)
                    minute_df, minute_trades, minute_params = self.model.load_minute_chart_data(stock_code, selected_date)
                    self.view.update_minute_chart(minute_df, minute_trades, stock_code, selected_date, minute_params)
                    self.view.draw_selected_date_line(selected_date)

                # '매매 종목 목록'의 첫 행만 자동 선택하여 관련 차트를 로드합니다.
                self.view.traded_stocks_table_view.selectRow(0)
                # '일별 성능 목록'의 자동 선택은 주석 처리하여 수직선 이동 문제를 방지합니다.
                # self.view.performance_table_view.selectRow(0)
            else:
                self.view.update_daily_chart(pd.DataFrame(), pd.DataFrame(), "N/A", {})
                self.view.update_minute_chart(pd.DataFrame(), pd.DataFrame(), "N/A", date.today(), {})
                self.view.draw_selected_date_line(None)
        finally:
            self.hide_loading_dialog()
            self.is_loading = False

    def on_stock_selected(self):
        """매매 종목 목록에서 특정 종목을 선택했을 때 호출됩니다."""
        if self.is_loading:
            return  # 로딩 중에는 새 선택 무시
            
        selected_indexes = self.view.traded_stocks_table_view.selectionModel().selectedRows()
        if not selected_indexes:
            return

        stock_data = self.view.traded_stocks_table_view.model().get_row_data(selected_indexes[0].row())
        if stock_data is None:
            return
            
        stock_code = stock_data['stock_code']
        if self.model.current_stock_code == stock_code:
            return  # 이미 선택된 종목이면 무시

        self.is_loading = True
        self.show_loading_dialog(f"{stock_code} 종목 차트를 로딩하는 중입니다...")
        
        try:
            self.model.set_selected_stock_code(stock_code)
            
            # 차트 데이터 로드 및 업데이트
            daily_df, daily_trades, daily_params = self.model.load_daily_chart_data(stock_code)
            self.view.update_daily_chart(daily_df, daily_trades, stock_code, daily_params)

            # 첫 매매일자에 수직선, 없으면 첫 봉
            selected_date = None
            if not daily_trades.empty and 'trade_datetime' in daily_trades.columns:
                first_trade_date = pd.to_datetime(daily_trades['trade_datetime']).dt.date.min()
                if first_trade_date in daily_df.index.date:
                    selected_date = first_trade_date
            if selected_date is None and not daily_df.empty:
                selected_date = daily_df.index[0].date()

            if selected_date:
                self.model.set_selected_daily_date(selected_date)
                minute_df, minute_trades, minute_params = self.model.load_minute_chart_data(stock_code, selected_date)
                self.view.update_minute_chart(minute_df, minute_trades, stock_code, selected_date, minute_params)
                self.view.draw_selected_date_line(selected_date)
                # 자동 선택 코드를 제거하여 2차 호출 방지
                # self.view.performance_table_view.selectRow(0)
            else:
                self.view.update_minute_chart(pd.DataFrame(), pd.DataFrame(), "N/A", date.today(), {})
                self.view.draw_selected_date_line(None)
        finally:
            self.hide_loading_dialog()
            self.is_loading = False

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
        
        # 선 그리기 및 분봉 차트 업데이트
        self.view.draw_selected_date_line(trade_date)
        self.update_minute_chart_for_date(trade_date)

    def on_daily_chart_click(self, event):
        """일봉 차트에서 특정 날짜를 클릭했을 때 호출됩니다."""
        if event.xdata is None or self.model.current_stock_code is None:
            return
        
        # 캔들 중심으로 스냅
        snapped_date = self.view.snap_to_nearest_date(event.xdata)
        if snapped_date is None:
            return
        trade_date = pd.to_datetime(snapped_date).date()
        
        self.model.set_selected_daily_date(trade_date)
        
        # 선 그리기 및 분봉/테이블 업데이트
        self.view.draw_selected_date_line(trade_date)
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

        self.show_loading_dialog("분봉 캔들을 로딩하는 중입니다...")
        try:
            self.loading_dialog.setValue(0)
            minute_df, trades_df, minute_params = self.model.load_minute_chart_data(self.model.current_stock_code, trade_date)
            self.view.update_minute_chart(minute_df, trades_df, self.model.current_stock_code, trade_date, minute_params)
            self.loading_dialog.setValue(100)
        finally:
            self.hide_loading_dialog()