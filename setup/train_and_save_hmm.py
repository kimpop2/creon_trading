# setup/train_and_save_hmm.py

import logging
from datetime import datetime
import sys
import os

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 필요한 모듈 임포트 ---
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from api.creon_api import CreonAPIClient
from analyzer.hmm_model import RegimeAnalysisModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_save_production_model():
    """실거래에 사용할 HMM 모델을 학습하고 DB에 저장합니다."""
    
    MODEL_NAME = "Test_HMM_v1" # settings.py의 LIVE_HMM_MODEL_NAME과 일치해야 함
    NUM_STATES = 4  # 최적화 결과 가장 좋았던 상태 개수 (예: 4개)
    
    logger.info(f"'{MODEL_NAME}' 모델 생성을 시작합니다 (상태 개수: {NUM_STATES}).")

    # 1. 컴포넌트 초기화
    api_client = CreonAPIClient()
    db_manager = DBManager()
    backtest_manager = BacktestManager(api_client, db_manager)
    
    # 2. HMM 학습에 필요한 데이터 가져오기
    # 기준 날짜는 최근 날짜로 설정
    training_data = backtest_manager.get_market_data_for_hmm(current_date=datetime.now().date(), days=730) # 예: 최근 2년 데이터
    
    if training_data.empty:
        logger.error("HMM 모델 학습용 데이터를 가져올 수 없어 중단합니다.")
        return

    # 3. 모델 생성 및 학습
    hmm_model = RegimeAnalysisModel(n_states=NUM_STATES)
    hmm_model.fit(training_data)
    
    # 4. 학습된 모델 파라미터를 DB에 저장
    model_params = hmm_model.get_params()
    
    start_date_str = training_data.index.min().strftime('%Y-%m-%d')
    end_date_str = training_data.index.max().strftime('%Y-%m-%d')

    db_manager.save_hmm_model(
        model_name=MODEL_NAME,
        n_states=NUM_STATES,
        observation_vars=['daily_return'], # get_market_data_for_hmm에서 사용하는 변수
        model_params=model_params,
        training_start_date=start_date_str,
        training_end_date=end_date_str
    )
    
    db_manager.close()
    logger.info(f"'{MODEL_NAME}' 모델을 DB에 성공적으로 저장했습니다.")


if __name__ == '__main__':
    train_and_save_production_model()