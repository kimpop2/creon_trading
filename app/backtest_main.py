# main.py
import sys
import os
from PyQt5.QtWidgets import QApplication, QProgressDialog, QLabel
from PyQt5.QtCore import Qt, QTimer, QElapsedTimer
import logging

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.backtest_view import BacktestView
from app.backtest_controller import BacktestController
from app.backtest_model import BacktestModel
from manager.business_manager import BusinessManager
from manager.data_manager import DataManager

def setup_logging():
    """로깅 기본 설정을 수행합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 특정 라이브러리의 로그 레벨 조정 (필요시)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.WARNING)

def show_loading_dialog(app, message="백테스트 시각화 프로그램을 초기화하는 중입니다..."):
    """로딩 다이얼로그를 표시합니다."""
    progress = QProgressDialog(message, None, 0, 100, None)
    progress.setWindowTitle("백테스트 시각화 프로그램 초기화")
    progress.setWindowModality(Qt.WindowModal)
    progress.setCancelButton(None)  # 취소 버튼 제거
    progress.setMinimumDuration(0)  # 즉시 표시
    progress.setValue(0)
    
    # 창 크기 조정
    progress.resize(400, 150)
    
    # 애니메이션 효과를 위한 타이머
    timer = QTimer()
    dots = 0
    start_time = QElapsedTimer()
    start_time.start()
    
    def update_dots():
        nonlocal dots
        dots = (dots + 1) % 4
        elapsed = start_time.elapsed() / 1000.0
        label = progress.labelText().split('\n')[0] + ('.' * dots) + "\n\n경과 시간: {:.1f}초".format(elapsed)
        progress.setLabelText(label)
    
    timer.timeout.connect(update_dots)
    timer.start(500)  # 0.5초마다 업데이트
    
    app.processEvents()  # UI 업데이트 강제 실행
    return progress, timer

def main():
    """애플리케이션의 메인 진입점입니다."""
    setup_logging()
    
    app = QApplication(sys.argv)
    
    # 로딩 다이얼로그 표시
    progress, timer = show_loading_dialog(app)
    
    # 1. 공유할 핵심 인스턴스 생성
    logging.info("애플리케이션 시작: 공유 인스턴스 생성 중...")
    try:
        progress.setLabelText("데이터 매니저를 초기화하는 중입니다...")
        progress.setValue(5)
        app.processEvents()
        data_manager = DataManager()
        
        progress.setLabelText("백테스트 매니저를 초기화하는 중입니다...")
        progress.setValue(10)
        app.processEvents()
        business_manager = BusinessManager(data_manager)
        
        progress.setLabelText("데이터 모델을 초기화하는 중입니다...")
        progress.setValue(15)
        app.processEvents()
        model = BacktestModel(business_manager)
        
        # --- 실행목록 로딩 (전체 70%까지 진행) ---
        progress.setLabelText("백테스트 실행목록을 로딩하는 중입니다...")
        progress.setValue(20)
        app.processEvents()
        
        def update_progress(value, message):
            progress.setValue(value)
            progress.setLabelText(message)
            app.processEvents()
        
        all_runs_df = model.load_all_backtest_runs(progress_callback=update_progress)
        # 전략 파라미터 파싱이 제거되어 70%에서 바로 완료
        # progress.setValue(70)  # 이미 load_all_backtest_runs에서 70%로 설정됨
        app.processEvents()
        # --------------------------------------
        
    except Exception as e:
        logging.critical(f"초기화 중 심각한 오류 발생: {e}", exc_info=True)
        progress.close()
        timer.stop()
        # 사용자에게 오류 메시지 박스를 보여주고 종료할 수 있음
        sys.exit(1)
    
    logging.info("공유 인스턴스 생성 완료.")

    # 2. View와 Controller 생성 및 연결
    progress.setLabelText("사용자 인터페이스를 구성하는 중입니다...")
    progress.setValue(85)
    app.processEvents()
    view = BacktestView()
    
    progress.setLabelText("컨트롤러를 초기화하는 중입니다...")
    progress.setValue(95)
    app.processEvents()
    controller = BacktestController(view, model)
    
    # 로딩 다이얼로그 닫기
    progress.setLabelText("초기화 완료!")
    progress.setValue(100)
    app.processEvents()
    timer.stop()
    progress.close()
    
    view.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()