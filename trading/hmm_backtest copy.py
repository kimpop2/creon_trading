# trading/hmm_backtest.py

import logging
import pandas as pd
import numpy as np
from datetime import datetime as dt

# --- 프로젝트의 다른 모듈들을 임포트 ---
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from trading.backtest import Backtest # 핵심 엔진으로 Backtest 클래스를 재사용
from analyzer.hmm_model import RegimeAnalysisModel
from analyzer.inference_service import RegimeInferenceService
from analyzer.policy_map import PolicyMap
from manager.portfolio_manager import PortfolioManager
from strategies.sma_daily import SMADaily # 기본 전략으로 사용
from strategies.pass_minute import PassMinute
from config.settings import INITIAL_CASH, STRATEGY_CONFIGS, COMMON_OPTIMIZATION_PARAMS

logger = logging.getLogger(__name__)


class HMMBacktest:
    """HMM 전체 시스템의 단일 백테스트 실행을 담당하는 클래스"""

    def __init__(self, backtest_manager: BacktestManager, initial_cash: float):
        self.backtest_manager = backtest_manager
        self.initial_cash = initial_cash

    def run(self, start_date: dt.date, end_date: dt.date, 
            hmm_params: dict, strategy_params: dict) -> dict:
        """
        옵티마이저로부터 파라미터를 받아 HMM 백테스트를 실행하고 최종 성과(metrics)를 반환합니다.
        """
        try:
            # 1. 백테스트 엔진 초기화
            backtester = Backtest(
                manager=self.backtest_manager,
                initial_cash=self.initial_cash,
                start_date=start_date,
                end_date=end_date,
                save_to_db=False # 최적화 중에는 DB 저장 안 함
            )

            # 2. HMM 두뇌 모듈을 동적 파라미터로 설정
            # 2.1 HMM 모델 학습 (실제로는 미리 학습된 모델을 로드해야 함)
            #     여기서는 파라미터(n_states)에 따라 매번 새로 학습한다고 가정
            market_data = self.backtest_manager.get_market_data_for_hmm(start_date) # HMM 학습용 데이터 로드
            hmm_model = RegimeAnalysisModel(n_states=hmm_params['hmm_n_states'])
            hmm_model.fit(market_data)
            inference_service = RegimeInferenceService(hmm_model)
            
            # 2.2 정책 맵 동적 생성
            policy_map = PolicyMap()
            policy_map.rules = { # 파일 대신 파라미터로 규칙 생성
                "regime_to_principal_ratio": {
                    "0": 1.0, # (예시) 강세장
                    "1": hmm_params.get('policy_bear_ratio', 0.5), # (예시) 약세장
                    "2": hmm_params.get('policy_crisis_ratio', 0.2) # (예시) 위기장
                }, "default_principal_ratio": 1.0
            }

            # 3. 전략 프로파일 DB에서 로드 (실제 구현 필요)
            # profiles_df = self.backtest_manager.db_manager.fetch_strategy_profiles_by_model(...)
            # strategy_profiles = ...
            strategy_profiles = {} # 임시

            # 4. HMM 두뇌를 탑재한 PortfolioManager 생성
            portfolio_manager = PortfolioManager(
                backtester.broker, STRATEGY_CONFIGS, inference_service, policy_map
            )
            # backtester가 최신 portfolio_manager를 사용하도록 연결
            backtester.portfolio_manager = portfolio_manager

            # 5. 전략 인스턴스 생성
            final_strategy_params = {**COMMON_OPTIMIZATION_PARAMS, **strategy_params}
            daily_strategy = SMADaily(broker=backtester.broker, data_store=backtester.data_store, strategy_params=final_strategy_params)
            minute_strategy = PassMinute(broker=backtester.broker, data_store=backtester.data_store, strategy_params=final_strategy_params)
            
            # 6. 백테스트 준비 및 실행
            backtester.set_strategies(daily_strategies=[daily_strategy], minute_strategy=minute_strategy)
            backtester.prepare_for_system() # 데이터는 한 번만 로드됨
            _, metrics = backtester.run() # run 대신 reset_and_rerun을 사용하지 않음 (Backtest 객체를 매번 새로 생성하므로)

            return metrics

        except Exception as e:
            logger.error(f"HMM 백테스트 실행 중 오류 발생: {e}", exc_info=True)
            return {} # 실패 시 빈 딕셔너리 반환