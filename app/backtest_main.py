# main.py
import sys
import os
from PyQt5.QtWidgets import QApplication
import logging

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.backtest_view import BacktestView
from app.backtest_controller import BacktestController
from app.backtest_data_model import BacktestDataModel
from manager.backtest_manager import BacktestManager
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

def main():
    """애플리케이션의 메인 진입점입니다."""
    setup_logging()
    
    app = QApplication(sys.argv)
    
    # 1. 공유할 핵심 인스턴스 생성
    logging.info("애플리케이션 시작: 공유 인스턴스 생성 중...")
    try:
        data_manager = DataManager()
        backtest_manager = BacktestManager(data_manager)
        model = BacktestDataModel(backtest_manager)
    except Exception as e:
        logging.critical(f"초기화 중 심각한 오류 발생: {e}", exc_info=True)
        # 사용자에게 오류 메시지 박스를 보여주고 종료할 수 있음
        sys.exit(1)
    logging.info("공유 인스턴스 생성 완료.")

    # 2. View와 Controller 생성 및 연결
    view = BacktestView()
    controller = BacktestController(view, model)
    
    view.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()