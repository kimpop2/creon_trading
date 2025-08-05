# analyzer/train_and_save_hmm.py

import logging
from datetime import date
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from analyzer.hmm_model import RegimeAnalysisModel
from config.settings import LIVE_HMM_MODEL_NAME
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_save_production_model(model_name_override: str = None, days: int = 730):
    """[최종 수정] HMM 모델을 저장/업데이트하고, model_id를 다시 조회하여 국면 데이터를 저장합니다."""
    
    MODEL_NAME = model_name_override or LIVE_HMM_MODEL_NAME
    NUM_STATES = 4
    
    logger.info(f"'{MODEL_NAME}' 최종 모델 생성을 시작합니다 (상태 개수: {NUM_STATES}).")

    db_manager = DBManager()
    backtest_manager = BacktestManager(None, db_manager)
    
    try:
        training_data = backtest_manager.get_market_data_for_hmm(days=days)
        if training_data.empty:
            logger.error("HMM 모델 학습용 데이터를 가져올 수 없어 중단합니다.")
            return

        hmm_model = RegimeAnalysisModel(n_states=NUM_STATES, covariance_type="diag")
        hmm_model.fit(training_data)
        
        model_params = hmm_model.get_params()
        start_date_str = training_data.index.min().strftime('%Y-%m-%d')
        end_date_str = training_data.index.max().strftime('%Y-%m-%d')

        # 1. 모델 파라미터 저장/업데이트
        success = db_manager.save_hmm_model(
            model_name=MODEL_NAME, n_states=NUM_STATES,
            observation_vars=list(training_data.columns), model_params=model_params,
            training_start_date=start_date_str, training_end_date=end_date_str
        )
        if not success:
            logger.error(f"'{MODEL_NAME}' 모델을 DB에 저장/업데이트 실패.")
            return

        # 2. 저장된 모델의 정확한 ID를 다시 조회
        model_info = db_manager.fetch_hmm_model_by_name(MODEL_NAME)
        if not model_info:
            logger.error(f"'{MODEL_NAME}' 모델 정보를 DB에서 찾을 수 없습니다.")
            return
        model_id = model_info['model_id']
        logger.info(f"'{MODEL_NAME}' 모델(ID: {model_id}) 처리 완료.")

        # 3. 정확한 ID로 국면 데이터 저장
        predicted_regimes = hmm_model.predict(training_data)
        regime_data_to_save = [{'date': date_val.date(), 'model_id': model_id, 'regime_id': int(regime_id)}
                               for date_val, regime_id in zip(training_data.index, predicted_regimes)]
        
        if db_manager.save_daily_regimes(regime_data_to_save):
            logger.info(f"총 {len(regime_data_to_save)}개의 일별 국면 데이터를 DB에 성공적으로 저장했습니다.")
        else:
            logger.error("일별 국면 데이터 저장에 실패했습니다.")
    finally:
        db_manager.close()

if __name__ == '__main__':
    train_and_save_production_model()