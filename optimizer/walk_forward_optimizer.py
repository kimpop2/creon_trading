# optimizer/walk_forward_optimizer.py

import json
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Tuple
import logging
import sys
import os
import time
import concurrent.futures # --- [신규] 병렬 처리를 위한 모듈 임포트 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 필요한 모듈들을 모두 임포트
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from analyzer.train_and_save_hmm import run_hmm_training
from optimizer.regimes_optimizer import RegimesOptimizer
from trading.hmm_backtest import HMMBacktest
from strategies.pass_minute import PassMinute
from config.settings import INITIAL_CASH, STRATEGY_CONFIGS
# 동적 로딩을 위해 모든 전략 클래스 임포트
from strategies.sma_daily import SMADaily
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.breakout_daily import BreakoutDaily
from strategies.closing_bet_daily import ClosingBetDaily

logger = logging.getLogger(__name__)

# --- ▼ [신규] 병렬 프로세스에서 실행될 작업 함수 ---
def _run_single_strategy_optimization_worker(args: Dict) -> List[Dict]:
    """
    단일 전략에 대한 최적화 및 프로파일 생성을 수행하는 독립적인 작업자 함수.
    """
    # 각 프로세스는 독립적인 리소스를 생성해야 함
    strategy_name = args['strategy_name']
    optim_start, optim_end = args['optim_start'], args['optim_end']
    model_id, model_name = args['model_id'], args['model_name']
    train_start, train_end = args['train_start'], args['train_end']
    
    # DB/API 연결 안정화를 위한 지연
    time.sleep(1) 
    api_client = CreonAPIClient()
    db_manager = DBManager()
    backtest_manager = BacktestManager(api_client=api_client, db_manager=db_manager)
    optimizer = RegimesOptimizer(backtest_manager, INITIAL_CASH)
    
    logger.info(f"  - [Process:{os.getpid()}] '{strategy_name}' 최적화 시작 (기간: {optim_start}~{optim_end})")
    
    try:
        regime_data_df = backtest_manager.fetch_daily_regimes(model_id)
        
        champion_params_by_regime = optimizer.run_optimization_for_strategy(
            strategy_name=strategy_name,
            start_date=optim_start,
            end_date=optim_end,
            regime_map=regime_data_df,
            model_name=f"temp_model_for_optim_{model_id}"
        )
        
        if not champion_params_by_regime:
            return []

        # 결과를 프로파일 형태로 가공
        profiles = []
        for regime_id, result in champion_params_by_regime.items():
            metrics = result.get('metrics', {})
            params = result.get('params', {}).get('strategy_params', {})
            profile = {
                'strategy_name': strategy_name, 'model_id': model_id, 'regime_id': regime_id,
                'sharpe_ratio': metrics.get('sharpe_ratio'), 'mdd': metrics.get('mdd'),
                'total_return': metrics.get('total_return'), 'win_rate': metrics.get('win_rate'),
                'num_trades': metrics.get('num_trades'), 'profiling_start_date': train_start,
                'profiling_end_date': train_end, 'params_json': json.dumps(params)
            }
            profiles.append(profile)
        
        return profiles
    except Exception as e:
        logger.error(f"  - [Process:{os.getpid()}] '{strategy_name}' 처리 중 오류: {e}", exc_info=True)
        return []
    finally:
        db_manager.close()


class WalkForwardOptimizer:
    def __init__(self, backtest_manager: BacktestManager):
        self.backtest_manager = backtest_manager
        # --- [수정] param_optimizer는 각 워커 프로세스가 생성하므로 제거 ---
        # self.param_optimizer = RegimesOptimizer(
        #     backtest_manager=self.backtest_manager,
        #     initial_cash=INITIAL_CASH
        # )

    def generate_walk_forward_periods(self, total_start_date: date, total_end_date: date, training_years: int, testing_years: int) -> List[Dict[str, date]]:
        periods = []
        current_start_date = total_start_date
        while True:
            train_start = current_start_date
            # train_end: 학습 시작일로부터 training_years 후의 날짜에서 하루를 뺌
            train_end = date(train_start.year + training_years, train_start.month, train_start.day) - timedelta(days=1)
            
            test_start = train_end + timedelta(days=1)
            # test_end: 테스트 시작일로부터 testing_years 후의 날짜에서 하루를 뺌
            test_end = date(test_start.year + testing_years, test_start.month, test_start.day) - timedelta(days=1)

            if test_end >= total_end_date:
                test_end = total_end_date
                periods.append({'train_start': train_start, 'train_end': train_end, 'test_start': test_start, 'test_end': test_end})
                break
            
            periods.append({'train_start': train_start, 'train_end': train_end, 'test_start': test_start, 'test_end': test_end})
            # 다음 윈도우의 시작일은 현재 테스트 시작일
            current_start_date = test_start
            
        logger.info(f"총 {len(periods)}개의 Walk-Forward 기간이 생성되었습니다.")
        return periods

    # --- ▼ [최종 구현] WFA 전체 프로세스를 실행하는 메인 메서드 ---
    def run_optimization(self, total_start_date: date, total_end_date: date, training_years: int, testing_years: int, model_prefix: str, policy_dict: dict = None) -> pd.Series:
        periods = self.generate_walk_forward_periods(total_start_date, total_end_date, training_years, testing_years)
        all_oos_results = []

        for i, period in enumerate(periods):
            train_start, train_end = period['train_start'], period['train_end']
            test_start, test_end = period['test_start'], period['test_end']

            logger.info(f"\n===== WFO Window #{i + 1}/{len(periods)} 처리 중 =====")
            logger.info(f"  [학습 기간]: {train_start} ~ {train_end}")
            logger.info(f"  [검증 기간]: {test_start} ~ {test_end}")

            # 1. HMM 학습 단계 호출
            model_name, model_id = self._run_hmm_training_step(train_start, train_end, model_prefix)
            if not model_id:
                logger.warning(f"Window #{i + 1} HMM 학습 실패. 다음 Window로 건너뜁니다.")
                continue

            # 2. 프로파일링 단계 호출
            profiling_success = self._run_in_sample_profiling_step(train_start, train_end, model_id)
            if not profiling_success:
                logger.warning(f"Window #{i + 1} 프로파일링 실패. 다음 Window로 건너뜁니다.")
                continue
            
            # 3. 최종 검증 단계 호출
            oos_series, run_id = self._run_out_of_sample_validation_step(test_start, test_end, model_name, model_id)
            if not oos_series.empty:
                all_oos_results.append(oos_series)
        
        logger.info("\n===== 모든 WFO Window 처리 완료 =====")
        return pd.concat(all_oos_results) if all_oos_results else pd.Series(dtype=float)

    # --- 내부 헬퍼 메서드들 (이미 TDD로 검증 완료) ---
    def run_hmm_training_step(self, train_start: date, train_end: date, model_prefix: str) -> Tuple[str, int]:
        model_name = f"{model_prefix}_{train_start.strftime('%y%m')}-{train_end.strftime('%y%m')}"
        logger.info(f"HMM 모델 학습 시작 (모델명: {model_name})")
        success, model_info, _ = run_hmm_training(model_name=model_name, start_date=train_start, end_date=train_end, backtest_manager=self.backtest_manager)
        if not success or not model_info:
            logger.error("HMM 모델 학습에 실패했습니다.")
            return None, None
        model_id = model_info.get('model_id')
        logger.info(f"HMM 모델 학습 성공 (model_id: {model_id})")
        return model_name, model_id
    
    
    # --- ▼ [핵심 수정] 프로파일링 단계를 병렬 처리 로직으로 전면 교체 ---
    def run_in_sample_profiling_step(self, train_start: date, train_end: date, model_id: int, model_name: str) -> bool:
        try:
            logger.info(f"--- In-Sample 프로파일링 시작 (model_id: {model_id}) ---")

            # 1. 최적화 기간을 학습 기간의 마지막 4개월로 설정
            # timedelta는 month 연산을 지원하지 않으므로, year와 month를 직접 계산
            optim_end_date = train_end
            
            start_month = optim_end_date.month - 3 # 4개월 전은 현재 달 포함 3달을 빼는 것
            start_year = optim_end_date.year
            if start_month <= 0:
                start_month += 12
                start_year -= 1
            optim_start_date = date(start_year, start_month, 1)

            logger.info(f"파라미터 최적화 기간: {optim_start_date} ~ {optim_end_date} (HMM 학습 기간: {train_start} ~ {train_end})")

            active_strategies = [name for name, config in STRATEGY_CONFIGS.items() if config.get("strategy_status")]
            
            # 2. 병렬 처리 실행 (이하 로직은 변경 없음)
            all_profiles_for_window = []
            tasks = []
            for strategy_name in active_strategies:
                task_args = {
                    'strategy_name': strategy_name, 'optim_start': optim_start_date, 'optim_end': optim_end_date,
                    'model_id': model_id, 'model_name': model_name, 'train_start': train_start, 'train_end': train_end
                }
                tasks.append(task_args)

            # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            #     future_to_strategy = {executor.submit(_run_single_strategy_optimization_worker, task): task['strategy_name'] for task in tasks}
            #     for future in concurrent.futures.as_completed(future_to_strategy):
            #         strategy_name = future_to_strategy[future]
            #         try:
            #             profiles = future.result()
            #             if profiles:
            #                 all_profiles_for_window.extend(profiles)
            #                 logger.info(f"  - '{strategy_name}' 최적화 완료, 프로파일 {len(profiles)}개 생성.")
            #             else:
            #                 logger.warning(f"  - '{strategy_name}' 최적화 결과 없음.")
            #         except Exception as exc:
            #             logger.error(f"  - '{strategy_name}' 작업 중 예외 발생: {exc}")

            # [신규 순차 처리 코드]
            logger.info("병렬 처리 비활성화. 순차적으로 전략 최적화를 실행합니다...")
            all_profiles_for_window = []
            for task in tasks:
                strategy_name = task['strategy_name']
                try:
                    # 작업자 함수를 직접 호출
                    profiles = _run_single_strategy_optimization_worker(task)
                    if profiles:
                        # 폴백 로직을 여기에 직접 적용하거나, 간단히 추가
                        all_profiles_for_window.extend(profiles)
                        logger.info(f"  - '{strategy_name}' 최적화 완료, 프로파일 {len(profiles)}개 생성.")
                    else:
                        logger.warning(f"  - '{strategy_name}' 최적화 결과 없음.")
                except Exception as exc:
                    logger.error(f"  - '{strategy_name}' 작업 중 예외 발생: {exc}")
            # --- ▲ 변경 완료 ---

            if not all_profiles_for_window:
                logger.warning("생성된 프로파일이 없어 DB에 저장하지 않습니다.")
                return False
            
            save_success = self.backtest_manager.save_strategy_profiles(all_profiles_for_window)
            if save_success:
                logger.info(f"--- In-Sample 프로파일링 완료: 총 {len(all_profiles_for_window)}개 프로파일 DB 저장 ---")
                return True
            else:
                logger.error("프로파일 DB 저장에 실패했습니다.")
                return False
                
        except Exception as e:
            logger.error(f"프로파일링 단계에서 예외 발생: {e}", exc_info=True)
            return False

    # =================== 차후 적용, creon api 로 데이터 가져오는 것을 먼저 처리하고, 계산처리만 병렬로 수행행
    # def run_in_sample_profiling_step(self, train_start: date, train_end: date, model_id: int, model_name: str) -> bool:
    #     try:
    #         logger.info(f"--- In-Sample 프로파일링 시작 (model_id: {model_id}) ---")

    #         # 1. 최적화 기간 설정 (학습 기간 마지막 4개월)
    #         optim_end_date = train_end
    #         start_month = optim_end_date.month - 3
    #         start_year = optim_end_date.year
    #         if start_month <= 0:
    #             start_month += 12
    #             start_year -= 1
    #         optim_start_date = date(start_year, start_month, 1)

    #         # 2. [선 데이터 준비] 메인 프로세스에서 API 의존적인 모든 데이터를 미리 로드
    #         logger.info(f"병렬 처리를 위한 데이터 사전 로딩 시작 (기간: {optim_start_date} ~ {optim_end_date})")
    #         # prepare_data_for_backtest가 data_store 딕셔너리를 반환한다고 가정
    #         preloaded_data_store = self.backtest_manager.prepare_data_for_backtest(optim_start_date, optim_end_date)
    #         regime_data_df = self.backtest_manager.fetch_daily_regimes(model_id)
            
    #         # 3. [후 병렬 계산] 병렬 처리 실행
    #         active_strategies = [name for name, config in STRATEGY_CONFIGS.items() if config.get("strategy_status")]
    #         tasks = []
    #         for strategy_name in active_strategies:
    #             tasks.append({
    #                 'strategy_name': strategy_name, 'optim_start': optim_start_date, 'optim_end': optim_end_date,
    #                 'model_id': model_id, 'train_start': train_start, 'train_end': train_end,
    #                 'data_store': preloaded_data_store, 'regime_data': regime_data_df
    #             })

    #         all_profiles_for_window = []
    #         with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    #             # ... (이후 결과 취합 및 폴백 로직은 이전과 동일)
    #             future_to_strategy = {executor.submit(_run_single_strategy_optimization_worker, task): task['strategy_name'] for task in tasks}
    #             for future in concurrent.futures.as_completed(future_to_strategy):
    #                 # ... 결과 취합 ...
    #                 all_profiles_for_window.extend(future.result())

    #         if not all_profiles_for_window:
    #             logger.warning("생성된 프로파일이 없어 DB에 저장하지 않습니다.")
    #             return False
            
    #         save_success = self.backtest_manager.save_strategy_profiles(all_profiles_for_window)
    #         return save_success

    #     except Exception as e:
    #         logger.error(f"프로파일링 단계에서 예외 발생: {e}", exc_info=True)
    #         return False



    def run_out_of_sample_validation_step(self, test_start: date, test_end: date, model_name: str, model_id: int, initial_cash: float) -> Tuple[pd.Series, int]:
        try:
            logger.info(f"--- Out-of-Sample 최종 검증 시작 (model_name: {model_name}, 초기자본: {initial_cash:,.0f}원) ---")
            
            # 1. 백테스트 인스턴스 생성
            oos_tester = HMMBacktest(
                manager=self.backtest_manager,
                initial_cash=initial_cash,
                start_date=test_start,
                end_date=test_end,
                save_to_db=True
            )

            # 2. 활성화된 모든 전략 인스턴스 생성
            daily_strategies_list = []
            for name, config in STRATEGY_CONFIGS.items():
                if config.get("strategy_status"):
                    strategy_class = globals().get(name)
                    if strategy_class:
                        instance = strategy_class(
                            broker=oos_tester.broker,
                            data_store=oos_tester.data_store # data_store 참조를 넘겨줌
                        )
                        daily_strategies_list.append(instance)

            minute_strategy = PassMinute(broker=oos_tester.broker, data_store=oos_tester.data_store)
            
            # --- ▼ [핵심 추가] 백테스트 실행 전 데이터 준비 단계 호출 ---
            logger.info("검증 기간에 사용할 유니버스 데이터 준비를 시작합니다...")
            oos_tester.set_strategies(daily_strategies_list, minute_strategy)
            oos_tester.prepare_for_system()
            # --- ▲ 추가 완료 ---

            # 3. HMM 모드로 백테스트 실행
            portfolio_series, _, _, run_id = oos_tester.reset_and_rerun(
                daily_strategies=daily_strategies_list,
                minute_strategy=minute_strategy,
                mode='hmm',
                model_name=model_name,
                policy_dict={}
            )
            
            logger.info(f"--- Out-of-Sample 최종 검증 완료 (run_id: {run_id}) ---")
            return portfolio_series, run_id

        except Exception as e:
            logger.error(f"최종 검증 백테스트 단계에서 예외 발생: {e}", exc_info=True)
            return pd.Series(dtype=float), None
        

    # # --- ▼ [핵심 수정] 프로파일링 단계를 병렬 처리 로직으로 전면 교체 ---
    # def run_in_sample_profiling_step(self, train_start: date, train_end: date, model_id: int, model_name: str) -> bool:
    #     try:
    #         logger.info(f"--- In-Sample 프로파일링 시작 (model_id: {model_id}) ---")

    #         # 1. 최적화 기간을 학습 기간의 마지막 4개월로 설정
    #         # timedelta는 month 연산을 지원하지 않으므로, year와 month를 직접 계산
    #         optim_end_date = train_end
            
    #         start_month = optim_end_date.month - 3 # 4개월 전은 현재 달 포함 3달을 빼는 것
    #         start_year = optim_end_date.year
    #         if start_month <= 0:
    #             start_month += 12
    #             start_year -= 1
    #         optim_start_date = date(start_year, start_month, 1)

    #         logger.info(f"파라미터 최적화 기간: {optim_start_date} ~ {optim_end_date} (HMM 학습 기간: {train_start} ~ {train_end})")

    #         active_strategies = [name for name, config in STRATEGY_CONFIGS.items() if config.get("strategy_status")]
            
    #         # 2. 병렬 처리 실행 (이하 로직은 변경 없음)
    #         all_profiles_for_window = []
    #         tasks = []
    #         for strategy_name in active_strategies:
    #             task_args = {
    #                 'strategy_name': strategy_name, 'optim_start': optim_start_date, 'optim_end': optim_end_date,
    #                 'model_id': model_id, 'model_name': model_name, 'train_start': train_start, 'train_end': train_end
    #             }
    #             tasks.append(task_args)

    #         with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    #             future_to_strategy = {executor.submit(_run_single_strategy_optimization_worker, task): task['strategy_name'] for task in tasks}
                
    #             # --- ▼ [핵심 수정] 결과 취합 및 폴백 로직 변경 ---
    #             for future in concurrent.futures.as_completed(future_to_strategy):
    #                 strategy_name = future_to_strategy[future]
    #                 try:
    #                     # optimizer로부터 받은 국면별 최적화 결과 (딕셔너리 형태)
    #                     # 예: {0: {'metrics': ..., 'params': ...}, 2: {'metrics': ..., 'params': ...}}
    #                     optimization_results = future.result()
                        
    #                     # 최종적으로 모든 국면(0,1,2,3)의 프로파일을 담을 딕셔너리
    #                     profiles_for_strategy = {}
    #                     best_sharpe_profile = None

    #                     if optimization_results:
    #                         # 1. 성공적으로 최적화된 국면의 프로파일 생성
    #                         for regime_id, result in optimization_results.items():
    #                             metrics = result.get('metrics', {})
    #                             params = result.get('params', {}).get('strategy_params', {})
    #                             profile = {
    #                                 'strategy_name': strategy_name, 'model_id': model_id, 'regime_id': regime_id,
    #                                 'sharpe_ratio': metrics.get('sharpe_ratio', 0), 'mdd': metrics.get('mdd', 0),
    #                                 'total_return': metrics.get('total_return', 0), 'win_rate': metrics.get('win_rate', 0),
    #                                 'num_trades': metrics.get('num_trades', 0), 'profiling_start_date': train_start,
    #                                 'profiling_end_date': train_end, 'params_json': json.dumps(params)
    #                             }
    #                             profiles_for_strategy[regime_id] = profile

    #                             # 가장 샤프 지수가 높은 프로파일을 찾기 위해 업데이트
    #                             if best_sharpe_profile is None or profile['sharpe_ratio'] > best_sharpe_profile['sharpe_ratio']:
    #                                 best_sharpe_profile = profile
                        
    #                     # 2. 프로파일이 없는 국면에 대한 '기본 프로파일' 생성
    #                     existing_regimes = set(profiles_for_strategy.keys())
    #                     all_regimes = {0, 1, 2, 3}
    #                     missing_regimes = all_regimes - existing_regimes

    #                     if missing_regimes:
    #                         fallback_params_json = None
    #                         if best_sharpe_profile:
    #                             # 2-1. 가장 성과가 좋았던 국면의 파라미터를 폴백으로 사용
    #                             fallback_params_json = best_sharpe_profile['params_json']
    #                             logger.warning(f"  - '{strategy_name}'의 누락 국면({missing_regimes})을 Best 국면(#{best_sharpe_profile['regime_id']})의 파라미터로 채웁니다.")
    #                         else:
    #                             # 2-2. (예외) 유효한 프로파일이 하나도 없으면 settings.py의 기본값 사용
    #                             logger.warning(f"  - '{strategy_name}'는 모든 국면 최적화 실패. settings.py 기본 파라미터로 채웁니다.")
                                
    #                             default_params = STRATEGY_CONFIGS.get(strategy_name, {}).get('default_params', {})
    #                             fallback_params_json = json.dumps(default_params)

    #                         for regime_id in missing_regimes:
    #                             default_profile = {
    #                                 'strategy_name': strategy_name, 'model_id': model_id, 'regime_id': regime_id,
    #                                 'sharpe_ratio': 0, 'mdd': 0, 'total_return': 0, 'win_rate': 0, 'num_trades': 0,
    #                                 'profiling_start_date': train_start, 'profiling_end_date': train_end,
    #                                 'params_json': fallback_params_json
    #                             }
    #                             profiles_for_strategy[regime_id] = default_profile
                        
    #                     # 최종적으로 모든 국면에 대한 프로파일을 전체 리스트에 추가
    #                     all_profiles_for_window.extend(profiles_for_strategy.values())

    #                 except Exception as exc:
    #                     logger.error(f"  - '{strategy_name}' 작업 중 예외 발생: {exc}")
    #             # --- ▲ 수정 완료 ---

    #         if not all_profiles_for_window:
    #             logger.warning("생성된 프로파일이 없어 DB에 저장하지 않습니다.")
    #             return False
            
    #         save_success = self.backtest_manager.save_strategy_profiles(all_profiles_for_window)
    #         if save_success:
    #             logger.info(f"--- In-Sample 프로파일링 완료: 총 {len(all_profiles_for_window)}개 프로파일 DB 저장 ---")
    #             return True
    #         else:
    #             logger.error("프로파일 DB 저장에 실패했습니다.")
    #             return False
                
    #     except Exception as e:
    #         logger.error(f"프로파일링 단계에서 예외 발생: {e}", exc_info=True)
    #         return False

    # def run_out_of_sample_validation_step(self, test_start: date, test_end: date, model_name: str, model_id: int, initial_cash: float) -> Tuple[pd.Series, int]:
    #     try:
    #         logger.info(f"--- Out-of-Sample 최종 검증 시작 (model_name: {model_name}, 초기자본: {initial_cash:,.0f}원) ---")
            
    #         # 1. 백테스트 인스턴스 생성
    #         oos_tester = HMMBacktest(
    #             manager=self.backtest_manager,
    #             initial_cash=initial_cash,
    #             start_date=test_start,
    #             end_date=test_end,
    #             save_to_db=True
    #         )

    #         # 2. 활성화된 모든 전략 인스턴스 생성
    #         daily_strategies_list = []
    #         for name, config in STRATEGY_CONFIGS.items():
    #             if config.get("strategy_status"):
    #                 strategy_class = globals().get(name)
    #                 if strategy_class:
    #                     instance = strategy_class(
    #                         broker=oos_tester.broker,
    #                         data_store=oos_tester.data_store # data_store 참조를 넘겨줌
    #                     )
    #                     daily_strategies_list.append(instance)

    #         minute_strategy = PassMinute(broker=oos_tester.broker, data_store=oos_tester.data_store)
            
    #         # --- ▼ [핵심 추가] 백테스트 실행 전 데이터 준비 단계 호출 ---
    #         logger.info("검증 기간에 사용할 유니버스 데이터 준비를 시작합니다...")
    #         oos_tester.set_strategies(daily_strategies_list, minute_strategy)
    #         oos_tester.prepare_for_system()
    #         # --- ▲ 추가 완료 ---

    #         # 3. HMM 모드로 백테스트 실행
    #         portfolio_series, _, _, run_id = oos_tester.reset_and_rerun(
    #             daily_strategies=daily_strategies_list,
    #             minute_strategy=minute_strategy,
    #             mode='hmm',
    #             model_name=model_name,
    #             policy_dict={}
    #         )
            
    #         logger.info(f"--- Out-of-Sample 최종 검증 완료 (run_id: {run_id}) ---")
    #         return portfolio_series, run_id

    #     except Exception as e:
    #         logger.error(f"최종 검증 백테스트 단계에서 예외 발생: {e}", exc_info=True)
    #         return pd.Series(dtype=float), None