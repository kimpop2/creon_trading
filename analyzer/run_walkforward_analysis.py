# analyzer/run_walkforward_analysis.py

import logging
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import sys
import os
import json
from multiprocessing import Pool, freeze_support # 병렬 처리를 위한 Pool, freeze_support 임포트

# --- 상대 경로 임포트 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.hmm_backtest import run_backtest_and_save_to_db, run_final_backtest
from analyzer.train_and_save_hmm import run_hmm_training 
from analyzer.run_profiling import run_strategy_profiling
from optimizer.run_portfolio_optimization import run_portfolio_optimization
from optimizer.run_system_optimization import run_system_optimization
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
# --- 로깅 설정 ---
# 1. 로그 파일을 저장할 절대 경로를 계산합니다.
log_dir = os.path.join(project_root, 'logs')
# 3. 절대 경로를 사용하여 로그 파일 핸들러를 설정합니다.
log_file_path = os.path.join(log_dir, 'walkforward_analysis.log')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path, encoding='utf-8'), # <--- 절대 경로로 수정
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger("WalkForwardOrchestrator")

# --- 최종 성과 분석을 위한 헬퍼 함수 ---
def calculate_performance_metrics(portfolio_value_series: pd.Series) -> dict:
    if portfolio_value_series.empty or len(portfolio_value_series) < 2:
        return {'cagr': 0, 'mdd': 0, 'sharpe_ratio': 0}
    
    days = (portfolio_value_series.index[-1] - portfolio_value_series.index[0]).days
    cagr = (portfolio_value_series.iloc[-1] / portfolio_value_series.iloc[0]) ** (365.0 / days) - 1 if days > 0 else 0
    
    roll_max = portfolio_value_series.cummax()
    daily_drawdown = portfolio_value_series / roll_max - 1.0
    mdd = daily_drawdown.cummin().iloc[-1]
    
    daily_returns = portfolio_value_series.pct_change().dropna()
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) != 0 else 0
    
    return {
        'cagr': f"{cagr:.2%}",
        'mdd': f"{mdd:.2%}",
        'sharpe_ratio': f"{sharpe_ratio:.2f}"
    }

# --- 헬퍼 함수: 단일 윈도우 처리 ---
def run_single_window(args: tuple):
    """
    하나의 롤링 윈도우에 대한 전체 워크플로우(1~6단계)를 실행하는 함수입니다.
    이 함수가 각 병렬 프로세스에서 실행될 작업 단위입니다.
    """
    try:
        training_start, training_end, test_start, test_end, model_name = args
        
        logger.info("\n" + "#"*80)
        logger.info(f"### 윈도우 처리 시작 | 모델명: {model_name} | PID: {os.getpid()} ###")
        logger.info(f"학습 기간: {training_start} ~ {training_end}")
        logger.info(f"검증 기간: {test_start} ~ {test_end}")
        logger.info("#"*80)

        # --- [핵심] 공유 컴포넌트 생성 및 데이터 사전 로딩 ---
        api_client = CreonAPIClient()
        db_manager = DBManager()
        backtest_manager = BacktestManager(api_client, db_manager)
        backtest_manager.prepare_pykrx_data_for_period(training_start, test_end) # 전체 필요 기간 로드

        # 1. 개별 전략 최적화
        champion_params = run_system_optimization(training_start, training_end, backtest_manager)
        
        # 2. 챔피언 전략 백테스트 및 DB 저장
        for strategy_name, params in champion_params.items():
            run_backtest_and_save_to_db(strategy_name, params, training_start, training_end, backtest_manager)
            
        # 3. HMM 모델 학습
        run_hmm_training(model_name, training_start, training_end, backtest_manager)
        
        # 4. 전략 프로파일링
        run_strategy_profiling(model_name, training_start, training_end)
        
        #5. 포트폴리오 운영 정책 최적화
        optimal_policy = run_portfolio_optimization(model_name, training_start, training_end, backtest_manager)
        if not optimal_policy:
            logger.warning(f"최적 정책을 찾지 못해 [{model_name}]의 최종 검증을 건너뜁니다.")
            return pd.Series(dtype=float)
            
        if optimal_policy:
            result_filename = "policy.json"
            with open(result_filename, 'w', encoding='utf-8') as f:
                json.dump(optimal_policy, f, ensure_ascii=False, indent=4)
            logger.info(f"✅ 최적 HMM 정책을 '{result_filename}' 파일에 저장했습니다. config/policy.json으로 활용하세요.")
            # 6. 최종 성과 검증 (Out-of-Sample)
            monthly_result_series = run_final_backtest(model_name, test_start, test_end, optimal_policy, backtest_manager)
            
        logger.info(f"### 윈도우 처리 완료: {model_name} ###")
        return monthly_result_series

    except Exception as e:
        model_name_for_log = args[4] if len(args) > 4 else "Unknown Window"
        logger.critical(f"'{model_name_for_log}' 윈도우 처리 중 심각한 오류 발생: {e}", exc_info=True)
        return pd.Series(dtype=float) # 오류 발생 시에도 빈 시리즈 반환

# --- 메인 실행 함수 ---
def main():
    logger.info("="*80)
    logger.info("======= 롤링 윈도우 워크 포워드 분석 (병렬 처리) 시작 =======")
    logger.info("="*80)

    # 1. 핵심 설정값 정의
    TOTAL_MONTHS = 12      # [수정] 총 12번의 롤링 테스트 수행
    TRAINING_MONTHS = 9  # 학습 기간 (12개월)
    GAP_MONTHS = 2        # 학습과 검증 사이의 공백 기간 (1개월)
    TEST_MONTHS = 1       # 검증 기간 (1개월)
    # TOTAL_MONTHS = 1      # [수정] 총 6번의 롤링 테스트 수행
    # TRAINING_MONTHS = 6  # 학습 기간 (12개월)
    # GAP_MONTHS = 2        # 학습과 검증 사이의 공백 기간 (2개월)
    # TEST_MONTHS = 1       # 검증 기간 (1개월)
    NUM_PROCESSES = 4     # [추가] 동시에 실행할 프로세스 수

    today = date.today()
    
    # 2. 병렬 처리할 작업 목록 생성
    window_args = []
    for i in range(TOTAL_MONTHS):
        test_end_date = today - relativedelta(months=i, day=31)
        test_start_date = test_end_date - relativedelta(months=TEST_MONTHS - 1, day=1)
        
        training_end_date = test_start_date - relativedelta(months=GAP_MONTHS, days=1)
        training_start_date = training_end_date - relativedelta(months=TRAINING_MONTHS - 1, day=1)
        
        model_name = f"wf_model_{test_start_date.strftime('%Y%m')}"
        window_args.append((training_start_date, training_end_date, test_start_date, test_end_date, model_name))

    # 3. [핵심] 병렬 처리 실행
    logger.info(f"{NUM_PROCESSES}개의 프로세스를 사용하여 병렬 분석을 시작합니다...")
    
    # Windows 환경에서 multiprocessing 사용 시 freeze_support() 호출은 필수
    freeze_support()
    with Pool(processes=NUM_PROCESSES) as pool:
        all_monthly_results = pool.map(run_single_window, window_args)
    
    # 4. 최종 결과 취합 및 분석
    logger.info("\n" + "="*80)
    logger.info("======= 모든 롤링 윈도우 분석이 완료되었습니다. 최종 결과를 취합합니다. =======")
    
    valid_results = [s for s in all_monthly_results if s is not None and not s.empty]
    
    if not valid_results:
        logger.warning("유효한 월별 테스트 결과가 없어 최종 성과를 계산할 수 없습니다.")
    else:
        valid_results.sort(key=lambda s: s.index[0])
        final_equity_curve = pd.concat(valid_results)
        
        logger.info("\n--- 최종 통합 자산 곡선 (Out-of-Sample) ---")
        logger.info(final_equity_curve.to_string())
        
        final_metrics = calculate_performance_metrics(final_equity_curve)
        logger.info("\n--- 최종 통합 성과 지표 ---")
        logger.info(f"CAGR (연복리수익률): {final_metrics['cagr']}")
        logger.info(f"MDD (최대낙폭): {final_metrics['mdd']}")
        logger.info(f"Sharpe Ratio (샤프 지수): {final_metrics['sharpe_ratio']}")
        
        try:
            import matplotlib.pyplot as plt
            plt.style.use('ggplot')
            final_equity_curve.plot(figsize=(15, 8), title='Walk-Forward Analysis - Out-of-Sample Equity Curve')
            plt.savefig('walkforward_equity_curve.png')
            logger.info("최종 자산 곡선 그래프를 'walkforward_equity_curve.png' 파일로 저장했습니다.")
        except ImportError:
            logger.warning("Matplotlib이 설치되지 않아 그래프를 저장할 수 없습니다. (pip install matplotlib)")

    logger.info("="*80)

if __name__ == "__main__":
    main()