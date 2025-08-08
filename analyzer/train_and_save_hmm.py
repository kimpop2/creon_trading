# analyzer/train_and_save_hmm.py

import logging
from datetime import date
import sys
import os
import numpy as np # 데이터 검사를 위해 numpy 임포트

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from analyzer.hmm_model import RegimeAnalysisModel
from config.settings import LIVE_HMM_MODEL_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# [수정] 주요 파라미터를 함수 인자로 받도록 변경
def run_hmm_training(model_name: str, start_date: date, end_date: date, backtest_manager: BacktestManager, n_states: int = 4):
    """
    [수정됨] 특정 기간과 이름으로 HMM 모델을 학습/저장하고, 국면 데이터를 생성합니다.
    롤링 윈도우 분석의 각 단계에서 호출될 재사용 가능한 함수입니다.
    """
    
    logger.info(f"'{model_name}' 모델 생성을 시작합니다 (상태: {n_states}개, 기간: {start_date} ~ {end_date}).")

    
    try:
        # [수정] get_market_data_for_hmm 호출 시 days 대신 start_date, end_date 전달
        training_data = backtest_manager.get_market_data_for_hmm(start_date=start_date, end_date=end_date)
        
        if training_data.empty:
            logger.error("HMM 모델 학습용 데이터를 가져올 수 없어 중단합니다.")
            return False # [수정] 실패 시 False 반환

        # --- 데이터 유효성 검사 (기존과 동일) ---
        if training_data.isnull().sum().sum() > 0:
            logger.error("HMM 학습 데이터에 NaN 값이 포함되어 있어 학습을 중단합니다.")
            return False
        if np.isinf(training_data).sum().sum() > 0:
            logger.error("HMM 학습 데이터에 무한대(inf) 값이 포함되어 있어 학습을 중단합니다.")
            return False
        logger.info("학습 데이터 유효성 검사 통과.")
        
        hmm_model = RegimeAnalysisModel(n_states=n_states, covariance_type="diag")
        hmm_model.fit(training_data)
        
        # --- 모델 학습 수렴 여부 확인 (기존과 동일) ---
        if not hmm_model.model.monitor_.converged:
            logger.warning(f"'{model_name}' 모델이 최적값에 수렴하지 않았을 수 있습니다.")
        else:
            logger.info("모델 학습이 정상적으로 최적값에 수렴했습니다.")
        
        model_params = hmm_model.get_params()
        start_date_str = training_data.index.min().strftime('%Y-%m-%d')
        end_date_str = training_data.index.max().strftime('%Y-%m-%d')

        # --- 모델 파라미터 저장/업데이트 (기존과 동일) ---
        success = backtest_manager.db_manager.save_hmm_model(
            model_name=model_name, n_states=n_states,
            observation_vars=list(training_data.columns), model_params=model_params,
            training_start_date=start_date_str, training_end_date=end_date_str
        )
        if not success:
            logger.error(f"'{model_name}' 모델을 DB에 저장/업데이트 실패.")
            return False

        model_info = backtest_manager.db_manager.fetch_hmm_model_by_name(model_name)
        if not model_info:
            logger.error(f"'{model_name}' 모델 정보를 DB에서 찾을 수 없습니다.")
            return False
        model_id = model_info['model_id']
        logger.info(f"'{model_name}' 모델(ID: {model_id}) 처리 완료.")

        predicted_regimes = hmm_model.predict(training_data)
        regime_data_to_save = [{'date': date_val.date(), 'model_id': model_id, 'regime_id': int(regime_id)}
                               for date_val, regime_id in zip(training_data.index, predicted_regimes)]
        
        if backtest_manager.db_manager.save_daily_regimes(regime_data_to_save):
            logger.info(f"총 {len(regime_data_to_save)}개의 일별 국면 데이터를 DB에 성공적으로 저장했습니다.")
        else:
            logger.error("일별 국면 데이터 저장에 실패했습니다.")
            return False
        
        return True # [수정] 성공 시 True 반환
    finally:
        backtest_manager.db_manager.close()


if __name__ == '__main__':
    # 테스트용 파라미터
    test_model_name = "test_model_2025_08"
    test_start_date = date(2024, 8, 1)
    test_end_date = date(2025, 7, 31)    
    logger.info(f"모듈 테스트를 시작합니다: '{test_model_name}'")
    
    # 함수 실행
    success = run_hmm_training(
        model_name=test_model_name,
        start_date=test_start_date,
        end_date=test_end_date,
        n_states=4
    )
    
    if success:
        logger.info(f"✅ 모듈 테스트 성공: '{test_model_name}'")
    else:
        logger.error(f"❌ 모듈 테스트 실패.")    # 기본값(4개)으로 생성
