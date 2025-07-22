# Phoenix-Eye: 자동매매 모니터링 GUI (V2)
#
# 이 코드는 PyQt5와 pyqtgraph 라이브러리를 사용하여 생성되었습니다.
# 실행 전, 아래 라이브러리가 설치되어 있는지 확인해주세요.
# pip install PyQt5 pyqtgraph
#
import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                             QHeaderView, QSplitter, QFrame, QSizePolicy)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt
import pyqtgraph as pg

# -----------------------------------------------------------------------------
# 메인 윈도우 클래스
# -----------------------------------------------------------------------------
class PhoenixEyeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Phoenix-Eye 자동매매 모니터링 시스템')
        # 화면 크기 변경
        self.setGeometry(100, 100, 1200, 1000)
        # 기본 Qt5 테마 사용 (스타일시트 제거)

        # --- 메인 레이아웃 설정 ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # --- 1. 글로벌 헤더 생성 ---
        header_widget = self.create_header_panel()
        main_layout.addWidget(header_widget)

        # --- 2. 메인 컨텐츠 영역 (4단 컬럼) ---
        main_splitter = QSplitter(Qt.Horizontal)

        col1 = self.create_universe_panel()
        col2 = self.create_strategy_panel()
        col3 = self.create_chart_panel()
        col4 = self.create_position_panel()

        main_splitter.addWidget(col1)
        main_splitter.addWidget(col2)
        main_splitter.addWidget(col3)
        main_splitter.addWidget(col4)

        # 각 컬럼의 초기 크기 비율 변경 (1:2:4:2)
        main_splitter.setSizes([120, 240, 480, 240])

        main_layout.addWidget(main_splitter)

        # --- 목업 데이터 로드 ---
        self.load_mock_data()

    def create_panel_widget(self, title_text):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        title_label = QLabel(title_text)
        title_label.setFont(QFont('Malgun Gothic', 12, QFont.Bold))
        layout.addWidget(title_label)
        
        return panel, layout

    def create_header_panel(self):
        header_frame = QFrame()
        header_frame.setFixedHeight(60)
        header_frame.setStyleSheet("border-bottom: 1px solid #c0c0c0;")
        
        layout = QHBoxLayout(header_frame)
        layout.setContentsMargins(10, 5, 10, 5)

        status_label = QLabel("시스템 상태: ● 운영 중")
        status_label.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(status_label)
        layout.addStretch(1)

        summary_layout = QGridLayout()
        summary_layout.addWidget(QLabel("총자산:"), 0, 0)
        self.total_assets_label = QLabel("10,550,000 원")
        summary_layout.addWidget(self.total_assets_label, 0, 1)
        
        summary_layout.addWidget(QLabel("주식 평가액:"), 0, 2)
        self.equity_value_label = QLabel("5,550,000 원")
        summary_layout.addWidget(self.equity_value_label, 0, 3)

        summary_layout.addWidget(QLabel("가용 현금:"), 1, 0)
        self.cash_label = QLabel("5,000,000 원")
        summary_layout.addWidget(self.cash_label, 1, 1)

        summary_layout.addWidget(QLabel("당일 손익:"), 1, 2)
        self.pnl_label = QLabel("+50,000 원 (+0.50%)")
        self.pnl_label.setStyleSheet("color: red;")
        summary_layout.addWidget(self.pnl_label, 1, 3)
        layout.addLayout(summary_layout)
        layout.addStretch(2)

        layout.addWidget(QPushButton("▶ 자동매매 시작"))
        layout.addWidget(QPushButton("■ 자동매매 정지"))
        emergency_button = QPushButton("🔥 포트폴리오 즉시 청산")
        emergency_button.setStyleSheet("background-color: #C13E3E; color: white;")
        layout.addWidget(emergency_button)
        layout.addWidget(QPushButton("📜 상세 리포트 보기"))

        return header_frame

    def create_universe_panel(self):
        panel, layout = self.create_panel_widget("동적 유니버스")
        
        self.universe_table = QTableWidget()
        self.universe_table.setColumnCount(3)
        self.universe_table.setHorizontalHeaderLabels(["종목명", "현재가", "등락률"])
        self.universe_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.universe_table.verticalHeader().setVisible(False)
        self.universe_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        layout.addWidget(self.universe_table)
        return panel

    def create_strategy_panel(self):
        panel, layout = self.create_panel_widget("전략 및 활성 신호")
        
        layout.addWidget(QLabel("전략 상태"))
        self.strategy_table = QTableWidget()
        self.strategy_table.setColumnCount(3)
        self.strategy_table.setHorizontalHeaderLabels(["전략명", "상태", "할당 자본금"])
        self.strategy_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.strategy_table.verticalHeader().setVisible(False)
        self.strategy_table.setFixedHeight(100)
        layout.addWidget(self.strategy_table)

        layout.addWidget(QLabel("활성 신호"))
        self.signal_table = QTableWidget()
        self.signal_table.setColumnCount(4)
        self.signal_table.setHorizontalHeaderLabels(["종목명", "신호", "목표가", "실행"])
        self.signal_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.signal_table.setColumnWidth(3, 80)
        self.signal_table.verticalHeader().setVisible(False)
        layout.addWidget(self.signal_table)

        self.perf_plot = pg.PlotWidget()
        self.perf_plot.showGrid(x=True, y=True, alpha=0.3)
        self.perf_plot.addLegend()
        # 폭과 높이를 같게 설정
        self.perf_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.perf_plot.setAspectLocked(True)
        layout.addWidget(self.perf_plot)

        return panel

    def create_chart_panel(self):
        """(중앙 우측) Column 3: 4개의 상세 차트 패널을 생성합니다."""
        panel, layout = self.create_panel_widget("상세 차트")
        
        self.charts = []
        for i in range(4):
            # GraphicsLayoutWidget을 사용하여 가격과 거래량 차트를 함께 배치
            win = pg.GraphicsLayoutWidget()
            layout.addWidget(win)
            self.charts.append(win)

        return panel

    def create_position_panel(self):
        panel, layout = self.create_panel_widget("보유 종목 및 성과")

        layout.addWidget(QLabel("현재 포지션"))
        self.position_table = QTableWidget()
        self.position_table.setColumnCount(6)
        self.position_table.setHorizontalHeaderLabels(["종목명", "수량", "평단가", "현재가", "평가손익", "수익률"])
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.position_table.verticalHeader().setVisible(False)
        layout.addWidget(self.position_table)

        layout.addWidget(QLabel("완료된 거래"))
        self.trade_log_table = QTableWidget()
        self.trade_log_table.setColumnCount(2)
        self.trade_log_table.setHorizontalHeaderLabels(["종목명", "실현 손익"])
        self.trade_log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.trade_log_table.verticalHeader().setVisible(False)
        layout.addWidget(self.trade_log_table)

        self.portfolio_plot = pg.PlotWidget()
        self.portfolio_plot.showGrid(x=True, y=True, alpha=0.3)
        self.portfolio_plot.addLegend()
        # 폭과 높이를 같게 설정
        self.portfolio_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.portfolio_plot.setAspectLocked(True)
        layout.addWidget(self.portfolio_plot)
        
        return panel

    def load_mock_data(self):
        """GUI를 채우기 위한 목업(가상) 데이터를 로드합니다."""
        # 테이블 데이터 80% 이상 채우기
        # 1. 유니버스
        universe_data = [
            (f"종목 A{i:02d}", f"{np.random.randint(100, 500)*100:,}", f"{np.random.uniform(-5, 5):+.2f}%") for i in range(20)
        ]
        self.universe_table.setRowCount(len(universe_data))
        for row, data in enumerate(universe_data):
            for col, item in enumerate(data):
                table_item = QTableWidgetItem(item)
                table_item.setTextAlignment(Qt.AlignCenter)
                if col == 2:
                    color = QColor('blue') if '+' in item else QColor('red')
                    table_item.setForeground(color)
                self.universe_table.setItem(row, col, table_item)

        # 2. 전략 상태 (생략)
        
        # 3. 활성 신호
        signal_data = [
            (f"종목 B{i:02d}", "매수", f"{np.random.randint(200, 600)*100:,}") for i in range(8)
        ]
        self.signal_table.setRowCount(len(signal_data))
        for row, data in enumerate(signal_data):
            for col, item in enumerate(data):
                 self.signal_table.setItem(row, col, QTableWidgetItem(item))
            exec_button = QPushButton("즉시 실행")
            self.signal_table.setCellWidget(row, 3, exec_button)

        # 4. 현재 포지션
        position_data = [
            (f"종목 C{i:02d}", f"{np.random.randint(1, 20)}", f"{np.random.randint(100, 300)*100:,}", f"{np.random.randint(100, 300)*100:,}", f"{np.random.randint(-50000, 50000):+,}", f"{np.random.uniform(-10, 10):+.2f}%") for i in range(10)
        ]
        self.position_table.setRowCount(len(position_data))
        for row, data in enumerate(position_data):
            for col, item in enumerate(data):
                table_item = QTableWidgetItem(item)
                if col >= 4:
                    color = QColor('blue') if '+' in item else QColor('red')
                    table_item.setForeground(color)
                self.position_table.setItem(row, col, table_item)

        # 5. 완료된 거래
        trade_log_data = [
            (f"종목 D{i:02d}", f"{np.random.randint(-100000, 100000):+,}") for i in range(10)
        ]
        self.trade_log_table.setRowCount(len(trade_log_data))
        for row, data in enumerate(trade_log_data):
            for col, item in enumerate(data):
                table_item = QTableWidgetItem(item)
                if col == 1:
                    color = QColor('blue') if '+' in item else QColor('red')
                    table_item.setForeground(color)
                self.trade_log_table.setItem(row, col, table_item)

        # 6. 그래프 데이터
        dates = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=30))
        timestamps = [d.timestamp() for d in dates]
        
        strategy_perf = 1 + np.random.randn(30).cumsum() / 50
        kospi_perf = 1 + np.random.randn(30).cumsum() / 80
        self.perf_plot.plot(timestamps, strategy_perf, pen=pg.mkPen('b', width=2), name='SMADaily')
        self.perf_plot.plot(timestamps, kospi_perf, pen=pg.mkPen('g', width=2), name='KOSPI')
        
        portfolio_value = 10000000 * (1 + np.random.randn(30).cumsum() / 40)
        cash_value = 5000000 * (1 - np.sin(np.linspace(0, 2*np.pi, 30))/2)
        self.portfolio_plot.plot(timestamps, portfolio_value, pen=pg.mkPen('b', width=2), name='총자산')
        self.portfolio_plot.plot(timestamps, cash_value, pen=pg.mkPen(color=(100,100,100), style=Qt.DotLine), name='현금')

        # 7. 4개의 캔들스틱 차트 그리기
        for chart_widget in self.charts:
            self.draw_candlestick_on_plot(chart_widget)

    def draw_candlestick_on_plot(self, graphics_layout_widget):
        """지정된 GraphicsLayoutWidget에 캔들스틱과 거래량 차트를 그립니다."""
        # 가격 차트와 거래량 차트 추가
        price_plot = graphics_layout_widget.addPlot(row=0, col=0)
        volume_plot = graphics_layout_widget.addPlot(row=1, col=0)
        
        price_plot.showGrid(x=True, y=True, alpha=0.3)
        volume_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # X축 연결
        volume_plot.setXLink(price_plot)
        
        # 목업 데이터 생성
        n = 60
        x = np.arange(n)
        opens = np.random.uniform(100, 102, n) + np.sin(np.arange(n)/5)
        highs = opens + np.random.uniform(0, 2, n)
        lows = opens - np.random.uniform(0, 2, n)
        closes = opens + np.random.uniform(-1.5, 1.5, n)
        volumes = np.random.randint(1000, 5000, n)

        # 캔들스틱 아이템 (내부 클래스로 정의)
        class CandlestickItem(pg.GraphicsObject):
            def __init__(self, data):
                pg.GraphicsObject.__init__(self)
                self.data = data
                self.generatePicture()

            def generatePicture(self):
                self.picture = pg.QtGui.QPicture()
                p = pg.QtGui.QPainter(self.picture)
                w = 0.4
                for (t, open, high, low, close) in self.data:
                    p.setPen(pg.mkPen('k'))
                    p.drawLine(pg.QtCore.QPointF(t, low), pg.QtCore.QPointF(t, high))
                    if open > close:
                        p.setBrush(pg.mkBrush('r'))
                    else:
                        p.setBrush(pg.mkBrush('g'))
                    p.drawRect(pg.QtCore.QRectF(t-w, open, w*2, close-open))
                p.end()

            def paint(self, p, *args):
                p.drawPicture(0, 0, self.picture)

            def boundingRect(self):
                return pg.QtCore.QRectF(self.picture.boundingRect())

        chart_data = [(i, opens[i], highs[i], lows[i], closes[i]) for i in range(n)]
        item = CandlestickItem(chart_data)
        price_plot.addItem(item)
        
        # 거래량 차트
        volume_item = pg.BarGraphItem(x=x, height=volumes, width=0.8, brush='w')
        volume_plot.addItem(volume_item)
        
        # 이동평균선
        ma5 = pd.Series(closes).rolling(window=5).mean()
        ma20 = pd.Series(closes).rolling(window=20).mean()
        price_plot.plot(x, ma5, pen=pg.mkPen('y', width=1))
        price_plot.plot(x, ma20, pen=pg.mkPen('c', width=1))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PhoenixEyeApp()
    window.show()
    sys.exit(app.exec_())
