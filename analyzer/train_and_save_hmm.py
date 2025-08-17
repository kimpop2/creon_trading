# analyzer/train_and_save_hmm.py

import logging
from datetime import date, datetime
import calendar # 월의 마지막 날을 계산하기 위해 추가
import re # 모델명에서 날짜를 안정적으로 추출하기 위해 추가
import numpy as np # 데이터 검사를 위해 numpy 임포트
import pandas as pd
import json
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from analyzer.hmm_model import RegimeAnalysisModel
from config.settings import LIVE_HMM_MODEL_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- 1. 국면 분석 및 정책 생성 함수 (신규 추가) ---
def generate_policy_file(backtest_manager: BacktestManager, model_name: str, output_path: str = 'policy.json'):
    """
    학습된 HMM 모델의 국면별 성과를 분석하여 policy.json 파일을 생성합니다.
    """
    analysis_df = backtest_manager.prepare_analysis_data(model_name)
    analysis_df['daily_return'] = analysis_df['종가'].pct_change().fillna(0)

    # 국면별 성과(평균 수익률, 변동성) 계산
    regime_stats = []
    for regime_id in sorted(analysis_df['regime_id'].unique()):
        regime_data = analysis_df[analysis_df['regime_id'] == regime_id]
        mean_return = regime_data['daily_return'].mean()
        volatility = regime_data['daily_return'].std()
        regime_stats.append({
            'regime_id': regime_id,
            'mean_return': mean_return,
            'volatility': volatility
        })
        logger.info(f"[분석] 국면 {regime_id}: 평균수익률 {mean_return:.4%}, 변동성 {volatility:.4%}")

    # ✨ [핵심] 평균 수익률 기준으로 국면을 정렬하여 의미 부여
    # HMM이 부여한 국면 번호(0,1,2,3)는 임의적이므로, 성과에 따라 재정렬해야 일관된 정책 적용 가능
    sorted_regimes = sorted(regime_stats, key=lambda x: x['mean_return'], reverse=True)
    
    # 정렬된 순서에 따라 투자 비중(가중치) 할당
    # 0: 가장 좋음(강세장), 1: 좋음, 2: 나쁨, 3: 가장 나쁨(위기)
    policy_weights = [1.0, 0.7, 0.4, 0.1] 
    
    regime_to_principal_ratio = {}
    sorted_regime_map = {} # 최종 국면 맵핑 정보 로깅용
    
    for i, regime_stat in enumerate(sorted_regimes):
        original_regime_id = regime_stat['regime_id']
        assigned_weight = policy_weights[i]
        regime_to_principal_ratio[str(original_regime_id)] = assigned_weight
        sorted_regime_map[i] = f"원래 국면 {original_regime_id} (수익률: {regime_stat['mean_return']:.3%}) -> 가중치: {assigned_weight}"
        
    logger.info("성과 기반 국면 정렬 및 가중치 할당 완료:")
    for sorted_id, description in sorted_regime_map.items():
        logger.info(f"  - {sorted_id}번: {description}")
        
    # JSON 파일 생성
    policy_data = {
        "regime_to_principal_ratio": regime_to_principal_ratio,
        "default_principal_ratio": 0.5 # 예외 발생 시 사용할 기본값
    }
    policy_path = os.path.join(project_root, 'config', 'policy.json')
    try:
        with open(policy_path, 'w', encoding='utf-8') as f:
            json.dump(policy_data, f, indent=4)
        logger.info(f"✅ 정책 파일 '{policy_path}' 생성을 완료했습니다.")
        return True
    except Exception as e:
        logger.error(f"정책 파일 '{policy_path}' 생성 중 오류 발생: {e}")
        return False

# --- ▼ [신규 추가] 모델명 파싱을 위한 헬퍼 함수 ---
def _parse_model_name(model_name: str) -> tuple[str, int]:
    """
    모델명 'EKLMNO_4s_2108-2508_v1' 에서 특징 이니셜과 상태 개수를 추출합니다.
    """
    parts = model_name.split('_')
    
    # 1. 특징 이니셜 추출 (예: 'EKLMNO')
    feature_initials = parts[0]
    
    # 2. 상태 개수 추출 (예: '4s' -> 4)
    n_states = 4 # 기본값
    for part in parts:
        if part.endswith('s'):
            try:
                # 's'를 제외하고 숫자로 변환
                n_states = int(part[:-1])
                break
            except ValueError:
                continue # 숫자로 변환 실패 시 다음 파트로 이동
                
    logger.info(f"모델명 분석 완료: 사용할 특징 이니셜='{feature_initials}', 상태 개수={n_states}")
    return feature_initials, n_states
    
# [수정] 주요 파라미터를 함수 인자로 받도록 변경
def run_hmm_training(model_name: str, start_date: date, end_date: date, backtest_manager: BacktestManager):
    """
    [개선됨] 모델명을 분석하여 필요한 특징만 선택하고, 동적으로 HMM 모델을 학습/저장합니다.
    """

    # 1. 모델명 분석하여 사용할 특징과 상태 개수 결정
    feature_initials, n_states = _parse_model_name(model_name)
    required_initials = set(feature_initials) # 빠른 조회를 위해 set으로 변환

    logger.info(f"'{model_name}' 모델 생성을 시작합니다 (상태: {n_states}개, 기간: {start_date} ~ {end_date}).")

    # 2. 모든 가능한 특징 데이터를 우선 생성
    backtest_manager.prepare_pykrx_data_for_period(start_date, end_date)
    all_features_data = backtest_manager.get_market_data_for_hmm(start_date=start_date, end_date=end_date)
    
    if all_features_data.empty:
        logger.error("HMM 모델 학습용 데이터를 가져올 수 없어 중단합니다.")
        return False

    # 3. 모델명에 지정된 이니셜을 가진 특징들만 필터링
    selected_columns = [
        col for col in all_features_data.columns 
        if col[0] in required_initials
    ]
    
    # 필터링된 컬럼이 실제로 존재하는지, 요청한 만큼 있는지 확인
    if len(selected_columns) != len(required_initials):
        logger.error(f"모델명에 요청된 특징 중 일부를 데이터에서 찾을 수 없습니다.")
        logger.error(f" - 요청된 이니셜: {sorted(list(required_initials))}")
        logger.error(f" - 실제 생성된 특징: {all_features_data.columns.tolist()}")
        return False
        
    training_data = all_features_data[selected_columns]
    logger.info(f"총 {len(all_features_data.columns)}개 중 {len(training_data.columns)}개 특징 선택 완료: {selected_columns}")

    # --- 4. 이후 과정은 필터링된 training_data로 동일하게 진행 ---
    if training_data.isnull().sum().sum() > 0 or np.isinf(training_data).sum().sum() > 0:
        logger.error("필터링된 학습 데이터에 NaN 또는 inf 값이 포함되어 있어 중단합니다.")
        return False
    
    hmm_model = RegimeAnalysisModel(n_states=n_states, covariance_type="diag")
    hmm_model.fit(training_data) # 필터링된 데이터로 학습
    
    if not hmm_model.model.monitor_.converged:
        logger.warning(f"'{model_name}' 모델이 최적값에 수렴하지 않았을 수 있습니다.")
    else:
        logger.info("모델 학습이 정상적으로 최적값에 수렴했습니다.")
    
    model_params = hmm_model.get_params()
    start_date_str = training_data.index.min().strftime('%Y-%m-%d')
    end_date_str = training_data.index.max().strftime('%Y-%m-%d')

    # [중요] DB에 저장 시, 실제 사용된 특징 목록(observation_vars)을 정확히 전달
    success = backtest_manager.db_manager.save_hmm_model(
        model_name=model_name, n_states=n_states,
        observation_vars=list(training_data.columns), # training_data.columns 사용
        model_params=model_params,
        training_start_date=start_date_str,
        training_end_date=end_date_str
    )
    if not success:
        # ... (이하 로직은 기존과 거의 동일) ...
        logger.error(f"'{model_name}' 모델을 DB에 저장/업데이트 실패.")
        return False

    model_info = backtest_manager.fetch_hmm_model_by_name(model_name)
    if not model_info: return False
    
    model_id = model_info['model_id']
    predicted_regimes = hmm_model.predict(training_data) # 필터링된 데이터로 예측
    
    regime_data_to_save = [{'date': date_val.date(), 'model_id': model_id, 'regime_id': int(regime_id)}
                            for date_val, regime_id in zip(training_data.index, predicted_regimes)]
    
    if backtest_manager.db_manager.save_daily_regimes(regime_data_to_save):
        logger.info(f"총 {len(regime_data_to_save)}개의 일별 국면 데이터를 DB에 성공적으로 저장했습니다.")
    else:
        logger.error("일별 국면 데이터 저장에 실패했습니다.")
        return False, None
    # [핵심] 생성된 regime_map을 직접 반환하도록 변경
    regime_map_df = pd.DataFrame(regime_data_to_save).set_index('date')
    return True, model_info, regime_map_df


if __name__ == '__main__':
    # --- ▼ [핵심 수정] 모델명 하나만 정의하면 모든 설정이 자동으로 결정됨 ---
    test_model_name = 'EKLMNO_4s_2208-2508'

    # --- 1. 모델명에서 학습 기간(YYMM-YYMM) 문자열 추출 ---
    # 정규표현식을 사용하여 'YYMM-YYMM' 패턴을 정확히 찾아냄
    date_pattern = re.search(r'(\d{4})-(\d{4})', test_model_name)
    if not date_pattern:
        raise ValueError("모델명에서 'YYMM-YYMM' 형식의 날짜 정보를 찾을 수 없습니다.")
    
    start_yymm_str, end_yymm_str = date_pattern.groups()

    # --- 2. 시작일(해당 월의 1일) 및 종료일(해당 월의 마지막 날) 자동 계산 ---
    start_dt = datetime.strptime(start_yymm_str, '%y%m')
    
    end_dt_raw = datetime.strptime(end_yymm_str, '%y%m')
    # calendar.monthrange()를 사용해 해당 월의 마지막 날짜를 구함 (예: (3, 31) -> 31)
    _, last_day = calendar.monthrange(end_dt_raw.year, end_dt_raw.month)
    end_dt = datetime(end_dt_raw.year, end_dt_raw.month, last_day)

    # 최종 date 객체로 변환
    test_start_date = start_dt.date()
    test_end_date = end_dt.date()
    
    logger.info(f"'{test_model_name}' 모델 테스트를 시작합니다.")
    logger.info(f" -> 모델명에서 추출된 학습 기간: {test_start_date} ~ {test_end_date}")
    
    # --- 3. 기존 로직 실행 (변수명만 통일) ---
    db_manager = None
    try:
        api_client = CreonAPIClient()
        db_manager = DBManager()
        backtest_manager = BacktestManager(api_client, db_manager)
        
        backtest_manager.prepare_pykrx_data_for_period(test_start_date, test_end_date)
        
        training_success, model_info = run_hmm_training(
            model_name=test_model_name,
            start_date=test_start_date,
            end_date=test_end_date,
            backtest_manager=backtest_manager
        )
        
        if training_success:
            logger.info("HMM 모델 학습 성공. 정책 파일 생성을 시작합니다.")
            policy_success = generate_policy_file(backtest_manager, test_model_name)
            if policy_success:
                logger.info(f"✅ 모듈 테스트 성공: '{test_model_name}' 모델 학습 및 정책 파일 생성 완료.")
            else:
                logger.error(f"❌ 정책 파일 생성 실패.")
        else:
            logger.error(f"❌ 모듈 테스트 실패: HMM 모델 학습 실패.")
            
    except Exception as e:
        logger.critical(f"테스트 실행 중 오류 발생: {e}", exc_info=True)
    finally:
        if db_manager:
            db_manager.close()