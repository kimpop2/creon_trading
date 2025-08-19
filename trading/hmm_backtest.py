# backtest/backtester.py
from datetime import datetime as dt, date, time, timedelta 
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import sys
import os
# 프로젝트 루트 경로를 sys.path에 추가 (manager 디렉토리에서 실행 가능하도록)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.broker import Broker
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager # BacktestManager 타입 힌트를 위해 남겨둠
from util.indicators import *
from strategies.strategy import DailyStrategy, MinuteStrategy
from trading.report_generator import ReportGenerator, BacktestDB
from manager.capital_manager import CapitalManager
from manager.portfolio_manager import PortfolioManager
from analyzer.hmm_model import RegimeAnalysisModel
from analyzer.inference_service import RegimeInferenceService
from analyzer.policy_map import PolicyMap
from analyzer.strategy_profiler import StrategyProfiler
# --- ▼ [수정] 전략 클래스 임포트 추가 ▼ ---
from strategies.sma_daily import SMADaily
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.breakout_daily import BreakoutDaily
from strategies.pullback_daily import PullbackDaily
from strategies.closing_bet_daily import ClosingBetDaily
from strategies.pass_minute import PassMinute
from strategies.target_price_minute import TargetPriceMinute
# --- ▲ [수정] 종료 ▲ ---

# 설정 파일 로드
from config.settings import (
    FETCH_DAILY_PERIOD, FETCH_MINUTE_PERIOD,
    MARKET_OPEN_TIME, MARKET_CLOSE_TIME,
    INITIAL_CASH, COMMON_PARAMS,
    STRATEGY_CONFIGS,               # [수정] 통합 설정 임포트
    STOP_LOSS_PARAMS,
    PRINCIPAL_RATIO,
    LIVE_HMM_MODEL_NAME
)     
logger = logging.getLogger(__name__)

class HMMBacktest:
    # __init__ 메서드를 외부에서 필요한 인스턴스를 주입받도록 변경
    def __init__(self, 
                 manager: BacktestManager,
                 initial_cash: float, 
                 start_date: dt.date,
                 end_date: dt.date,
                 save_to_db: bool = True # DB 저장 여부
        ):
        self.manager = manager
        self.initial_cash = initial_cash
        self.original_initial_cash = initial_cash # [추가] 리셋을 위한 원본 초기 자본금 저장
        self.broker = Broker(manager=manager, initial_cash=initial_cash)
        self.capital_manager = CapitalManager(self.broker)
        self.portfolio_manager = None
        self.start_date = start_date
        self.end_date = end_date
        self.daily_strategies: List[DailyStrategy] = []
        self.minute_strategy: MinuteStrategy = None
        self.data_store = {'daily': {}, 'minute': {}} 
        self.save_to_db = save_to_db

        self.portfolio_values = [] # self.broker.get_portfolio_value #########
        self.custom_universe: Optional[List[str]] = None

        self.market_open_time = dt.strptime(MARKET_OPEN_TIME, '%H:%M:%S').time()
        self.market_close_time = dt.strptime(MARKET_CLOSE_TIME, '%H:%M:%S').time()
        
        self.last_portfolio_check = None # 포트폴리오 손절 체크 시간을 추적하기 위한 변수
        self.portfolio_stop_flag = False # 포트폴리오 손절 발생 시 당일 매매 중단 플래그

        self._minute_data_cache = {}   # {stock_code: DataFrame (오늘치 분봉 데이터)}
        self.price_cache = {}           # 가격 캐시를 Backtest 클래스가 직접 소유
        # [추가] 학습된 HMM 모델을 저장할 인스턴스 변수
        self.hmm_model = None
        self.inference_service = None

        logging.info(f"백테스터 초기화 완료. 초기 현금: {self.initial_cash:,.0f}원, DB저장: {self.save_to_db}")

    def set_strategies(self, daily_strategies: List[DailyStrategy], minute_strategy: MinuteStrategy, stop_loss_params: dict = None):
        self.daily_strategies = daily_strategies
        self.minute_strategy = minute_strategy
        if stop_loss_params:
            self.broker.set_stop_loss_params(stop_loss_params)
        daily_strategy_names = ', '.join([s.__class__.__name__ for s in self.daily_strategies])
        logger.info(f"전략 설정 완료: Daily(s)='{daily_strategy_names}', Minute='{self.minute_strategy.__class__.__name__}'")

    # --- ▼ [2. 추가] 사용자 정의 유니버스를 설정하는 세터(setter) 메서드 ---
    def set_custom_universe(self, universe_codes: Optional[List[str]]):
        """
        최적화 등 특정 목적을 위해 사용할 사용자 정의 유니버스를 설정합니다.
        """
        self.custom_universe = universe_codes
        if universe_codes:
            logger.info(f"사용자 정의 유니버스({len(universe_codes)}개 종목)가 설정되었습니다.")
    # --- ▲ 추가 완료 ---

    def prepare_for_system(self) -> bool:
        """백테스팅 기간내 일봉 분봉 데이터를 미리 data_store 에 저장 합니다."""
        start_date = self.start_date
        end_date = self.end_date
        logging.info(f"--- 백테스트 데이터 준비 시작 ({start_date} ~ {end_date}) ---")
        
        # 1. 유니버스 종목 코드 결정 (모든 전략이 필요로 하는 종목 총망라)
        universe_codes = set()
        # --- ▼ [3. 수정] 유니버스 결정 로직 변경 ---
        # 1. 사용할 유니버스 종목 코드 결정
        if self.custom_universe is not None:
            logger.info("사용자 정의 유니버스를 사용하여 데이터를 준비합니다.")
            universe_codes = set(self.custom_universe)
        else:
            logger.info("기본 전체 유니버스를 사용하여 데이터를 준비합니다.")
            universe_codes = set(self.manager.get_universe_codes())
        # --- ▲ 수정 완료 ---
        
        # COMMON_PARAMS의 필수 코드 추가
        market_code = COMMON_PARAMS.get('market_index_code')
        if market_code: universe_codes.add(market_code)
        safe_asset_code = COMMON_PARAMS.get('safe_asset_code')
        if safe_asset_code: universe_codes.add(safe_asset_code)

        # 2. 전체 기간 데이터 사전 로딩
        daily_start = start_date - timedelta(days=FETCH_DAILY_PERIOD) # 지표 계산을 위한 충분한 과거 데이터
        minute_start = start_date - timedelta(days=FETCH_MINUTE_PERIOD) # 지표 계산을 위한 충분한 과거 데이터
        
        all_trading_dates_list = self.manager.get_all_trading_days(daily_start, end_date)
        all_trading_dates_set = set(all_trading_dates_list)
        
        for code in list(universe_codes):
            logging.info(f"데이터 로딩: {code} 일봉: ({daily_start} ~ {end_date}), 분봉: ({minute_start} ~ {end_date})")
            # 일봉 데이터 로드
            daily_df = self.manager.cache_daily_ohlcv(code, daily_start, end_date, all_trading_dates_set)
            if not daily_df.empty: self.data_store['daily'][code] = daily_df
            
            if self.minute_strategy.strategy_name != 'PassMinute':
                # 분봉 데이터 로드
                minute_df = self.manager.cache_minute_ohlcv(code, minute_start, end_date, all_trading_dates_set)
                if not minute_df.empty:
                    # data_store에 해당 종목 코드의 딕셔너리가 없으면 생성
                    self.data_store['minute'].setdefault(code, {})
                    # 불러온 분봉 데이터를 날짜별로 그룹화하여 저장
                    for group_date, group_df in minute_df.groupby(minute_df.index.date):
                        # group_date는 dt.date 객체
                        self.data_store['minute'][code][group_date] = group_df
        
        logging.info(f"--- 모든 데이터 준비 완료 ---")
        return True        

    def reset_and_rerun(self, daily_strategies: List[DailyStrategy], minute_strategy: MinuteStrategy, 
                        mode: str = 'strategy', hmm_params: dict = None,
                        model_name: str = None, policy_dict: dict = None) -> tuple:
        """
        [최종 리팩토링] data_store는 유지한 채, 브로커와 전략만 리셋하고 백테스트를 재실행합니다.
        HMM 모드에서는 반드시 model_name을 받아, 사전 학습된 모델과 프로파일을 DB에서 로드합니다.
        """
        logger.info(f"--- 백테스트 리셋 및 재실행 시작 (Mode: {mode}, Model: {model_name}) ---")
        
        self.broker = Broker(manager=self.manager, initial_cash=self.original_initial_cash)
        self.portfolio_values = []

        if mode == 'hmm':
            if not model_name:
                raise ValueError("HMM 모드에서는 'model_name'이 반드시 필요합니다.")

            model_info = self.manager.fetch_hmm_model_by_name(model_name)
            if not model_info:
                raise ValueError(f"DB에서 HMM 모델 '{model_name}'을 찾을 수 없습니다.")
            
            self.model_id = model_info['model_id']
            hmm_model = RegimeAnalysisModel.load_from_params(model_info['model_params'])
            self.inference_service = RegimeInferenceService(hmm_model)
            # --- ▼ [1. 핵심 수정] HMM 모델에서 전이 행렬을 추출합니다. ---
            transition_matrix = hmm_model.model.transmat_

            profiles_df = self.manager.fetch_strategy_profiles_by_model(self.model_id)
            profiles_for_pm = {}
            if not profiles_df.empty:
                for _, row in profiles_df.iterrows():
                    strategy_name = row['strategy_name']
                    regime_id = row['regime_id']
                    if strategy_name not in profiles_for_pm:
                        profiles_for_pm[strategy_name] = {}
                    profiles_for_pm[strategy_name][regime_id] = row.to_dict()
            
            # --- ▼ [핵심 수정] 누락된 PortfolioManager 초기화 코드 추가 ▼ ---
            self.policy_map = PolicyMap()
            if policy_dict:
                self.policy_map.rules = policy_dict
            elif hmm_params: # 옵티마이저에서 동적으로 생성된 규칙 적용
                self.policy_map.rules = {
                    "regime_to_principal_ratio": {
                        "0": 1.0, 
                        "1": hmm_params.get('policy_bear_ratio', 0.5),
                        "2": hmm_params.get('policy_crisis_ratio', 0.2), 
                        "3": 1.0
                    }, "default_principal_ratio": 1.0
                }
            else: # 기본 policy.json 파일 사용
                policy_path = os.path.join(project_root, 'config', 'policy.json')
                self.policy_map.load_rules(policy_path)

            # [수정] STRATEGY_CONFIGS를 스캔하여 활성화된 전략 목록을 동적으로 생성
            active_strategies_for_pm = []
            for name, config in STRATEGY_CONFIGS.items():
                if config.get("strategy_status") is True:
                    active_strategies_for_pm.append({
                        "name": name,
                        "weight": config.get("strategy_weight", 0)
                    })
            
            # --- ▼ [2. 핵심 수정] PortfolioManager 생성자에 transition_matrix를 전달합니다. ---
            self.capital_manager = PortfolioManager(
                broker=self.broker,
                portfolio_configs=active_strategies_for_pm,
                inference_service=self.inference_service,
                policy_map=self.policy_map,
                strategy_profiles=profiles_for_pm,
                transition_matrix=transition_matrix # <-- 이 인자를 추가
            )
            # --- ▲ [핵심 수정] 종료 ▲ ---

        else: # 'strategy' 모드
            self.capital_manager = CapitalManager(self.broker)

        # 전략 객체들의 브로커 참조 업데이트 (기존과 동일)
        for strategy in daily_strategies:
            strategy.broker = self.broker
        minute_strategy.broker = self.broker
        self.set_strategies(daily_strategies, minute_strategy)

        # =================================================================
        # 백테스트 메인 루프 실행
        # =================================================================
        start_date = self.start_date
        end_date = self.end_date      
        market_calendar_df = self.manager.fetch_market_calendar(start_date, end_date)
        trading_dates = market_calendar_df[market_calendar_df['is_holiday'] == 0]['date'].dt.date.tolist()
        # --- ▼ [핵심 추가] 매매 제외 대상 종목 리스트를 미리 정의 ---
        non_tradeable_codes = {
            COMMON_PARAMS.get('market_index_code'),
            COMMON_PARAMS.get('safe_asset_code')
        }
        non_tradeable_codes.discard(None) # 설정 파일에 값이 없는 경우를 대비해 None 제거
        # --- ▲ 추가 완료 ---

        for i, current_date in enumerate(trading_dates):
            prev_date = trading_dates[i - 1] if i > 0 else self.start_date - timedelta(days=1)
            prev_prices = self._get_prices_for_date(prev_date)

            if mode == 'hmm':
                # --- ▼ [수정 2] get_strategy_capitals 호출 방식 수정 ▼ ---
                account_equity = self.capital_manager.get_account_equity(prev_prices)
                # get_market_data_for_hmm 호출 시, start_date와 end_date를 명시하여 현재 날짜까지의 데이터만 사용하도록 함
                hmm_input_data = self.manager.get_market_data_for_hmm(start_date=self.start_date, end_date=current_date)
                total_principal, regime_probs = self.capital_manager.get_total_principal(account_equity, hmm_input_data)
                
                # PortfolioManager가 이미 프로파일을 가지고 있으므로, 인자로 넘겨줄 필요 없음
                strategy_capitals = self.capital_manager.get_strategy_capitals(total_principal, regime_probs)
                # --- ▲ [수정 2] 종료 ▲ ---
            else: # 'strategy' 모드
                account_equity = self.capital_manager.get_account_equity(prev_prices)
                total_principal, regime_probs = self.capital_manager.get_total_principal(account_equity, PRINCIPAL_RATIO)

            # 복수 일봉 전략 실행 및 신호 통합
            signals_from_all = []
            for strategy in self.daily_strategies:
                if mode == 'hmm':
                    strategy_capital = strategy_capitals.get(strategy.strategy_name, 0)
                else:
                    strategy_capital = self.capital_manager.get_strategy_capital(strategy.strategy_name, total_principal)

                strategy.run_daily_logic(current_date, strategy_capital)

                # --- ▼ [핵심 수정] 생성된 신호에서 매매 제외 대상을 필터링 ---
                filtered_signals = {
                    code: signal for code, signal in strategy.signals.items()
                    if code not in non_tradeable_codes
                }
                signals_from_all.append(filtered_signals)
                # --- ▲ 수정 완료 ---                
            
            final_signals = self._aggregate_signals(signals_from_all)
            self.minute_strategy.update_signals(final_signals)

            
            market_open_dt = dt.combine(current_date, self.market_open_time)
            stocks_to_trade = set(self.broker.get_current_positions().keys()) | set(final_signals.keys())

            if self.minute_strategy.strategy_name == 'PassMinute':
                current_dt = market_open_dt + timedelta(minutes=1)
                for stock_code in list(stocks_to_trade):
                            self.minute_strategy.run_minute_logic(current_dt, stock_code)
                
            else:
                # 분봉 루프 실행
                for minute_offset in range(391):
                    current_dt = market_open_dt + timedelta(minutes=minute_offset)
                    if self.market_open_time < current_dt.time() <= self.market_close_time:
                        if current_dt.time() < time(9, 1) or time(15, 20) < current_dt.time() < time(15, 30): 
                            continue
                        current_prices = self.get_all_current_prices(current_dt)
                        self.broker.check_and_execute_stop_loss(current_prices, current_dt)
                        for stock_code in list(stocks_to_trade):
                            self.minute_strategy.run_minute_logic(current_dt, stock_code)

            # 5. 하루 종료 후 포트폴리오 가치 기록
            close_prices = self.get_all_current_prices(dt.combine(current_date, self.market_close_time))
            daily_portfolio_value = self.broker.get_portfolio_value(close_prices)
            self.portfolio_values.append((current_date, daily_portfolio_value))
            logging.info(f"[{current_date.isoformat()}] 장 마감 포트폴리오 가치: {daily_portfolio_value:,.0f}원")

        # =================================================================
        # 4. 결과 반환 로직 (기존 run() 메서드와 동일)
        # =================================================================
        if len(self.portfolio_values) > 1:
            portfolio_df = pd.DataFrame(self.portfolio_values[1:], columns=['Date', 'PortfolioValue'])
            portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
            portfolio_value_series = portfolio_df.set_index('Date')['PortfolioValue']
        else:
            portfolio_value_series = pd.Series(dtype=float)

        # calculate_performance_metrics 함수가 외부에 있다고 가정
        final_metrics = calculate_performance_metrics(portfolio_value_series)
        trade_log = self.broker.transaction_log # Broker로부터 거래 로그를 가져옴
        run_id = None # run_id 초기화

        # --- ▼ [핵심 추가] save_to_db가 True일 때만 DB에 저장하고 run_id를 받아옴 ---
        if self.save_to_db and not portfolio_value_series.empty:
            storage = BacktestDB(self.manager.get_db_manager())
            reporter = ReportGenerator(storage_strategy=storage)

            daily_strategy_names = ', '.join([s.__class__.__name__ for s in self.daily_strategies])
            daily_strategy_params = {s.__class__.__name__: s.strategy_params for s in self.daily_strategies}

            run_id = reporter.generate(
                start_date=self.start_date,
                end_date=self.end_date,
                initial_cash=self.initial_cash,
                portfolio_value_series=portfolio_value_series,
                transaction_log=trade_log,
                strategy_info={
                    'strategy_daily': daily_strategy_names,
                    'strategy_minute': self.minute_strategy.__class__.__name__,
                    'params_json_daily': daily_strategy_params,
                    'params_json_minute': self.minute_strategy.strategy_params,
                    'model_id': self.model_id if hasattr(self, 'model_id') else None # 모델 ID 추가
                }
            )
        elif portfolio_value_series.empty:
             logger.warning("포트폴리오 가치 데이터가 비어 있어 DB에 결과를 저장하지 않습니다.")
        else:
             logger.info("save_to_db=False로 설정되어 DB에 결과를 저장하지 않습니다.")
        
        # --- ▼ [핵심 수정] run_id를 함께 반환하도록 변경 ---
        return portfolio_value_series, final_metrics, trade_log, run_id

    def run(self):
        start_date = self.start_date
        end_date = self.end_date
        # [수정] prepare_for_system()을 run 메서드 외부에서 호출하도록 변경 (최적화 시 유리)
        # self.prepare_for_system() 

        logging.info(f"백테스트 시작: {start_date} 부터 {end_date} 까지")
        
        if not self.daily_strategies or not self.minute_strategy:
            logger.error("일봉 또는 분봉 전략이 설정되지 않았습니다. 백테스트를 중단합니다.")
            return

        market_calendar_df = self.manager.fetch_market_calendar(start_date, end_date)
        trading_dates = market_calendar_df[market_calendar_df['is_holiday'] == 0]['date'].dt.date.tolist()
        
        for i, current_date in enumerate(trading_dates):
            if not (start_date <= current_date <= end_date):
                continue

            logging.info(f"\n--- 현재 백테스트 날짜: {current_date.isoformat()} ---")
            
            # 1. 자금 할당 로직
            prev_date = trading_dates[i - 1] if i > 0 else start_date - timedelta(days=1)
            prev_prices = self._get_prices_for_date(prev_date)
            
            account_equity = self.capital_manager.get_account_equity(prev_prices)
            total_principal, regime_probs = self.capital_manager.get_total_principal(account_equity, PRINCIPAL_RATIO)

            # --- [핵심 수정] 신호 통합 로직 적용 ---
            # 2. 복수 일봉 전략 실행 및 신호 수집
            signals_from_all = []
            for strategy in self.daily_strategies:
                strategy_capital = self.capital_manager.get_strategy_capital(strategy.strategy_name, total_principal)
                strategy.run_daily_logic(current_date, strategy_capital)
                signals_from_all.append(strategy.signals)
            
            # 3. 신호 통합 정책을 적용하여 최종 신호를 결정합니다.
            final_signals = self._aggregate_signals(signals_from_all)
            self.minute_strategy.update_signals(final_signals)
            logging.info(f"신호 통합 완료. 최종 유효 신호: {len(final_signals)}개")
            # --- 수정 끝 ---
            # 4. 분봉 루프 실행
            market_open_dt = dt.combine(current_date, self.market_open_time)
            stocks_to_trade = set(self.broker.get_current_positions().keys()) | set(final_signals.keys())

            for minute_offset in range(391): # 9:00 ~ 15:30
                current_dt = market_open_dt + timedelta(minutes=minute_offset)
                
                if self.market_open_time < current_dt.time() <= self.market_close_time:
                    if current_dt.time() < time(9, 1) or time(15, 20) < current_dt.time() < time(15, 30): 
                        continue

                    current_prices = self.get_all_current_prices(current_dt)
                    
                    self.broker.check_and_execute_stop_loss(current_prices, current_dt)
                    
                    for stock_code in list(stocks_to_trade):
                        self.minute_strategy.run_minute_logic(current_dt, stock_code)

            # 5. 하루 종료 후 포트폴리오 가치 기록
            close_prices = self.get_all_current_prices(dt.combine(current_date, self.market_close_time))
            daily_portfolio_value = self.broker.get_portfolio_value(close_prices)
            self.portfolio_values.append((current_date, daily_portfolio_value))
            logging.info(f"[{current_date.isoformat()}] 장 마감 포트폴리오 가치: {daily_portfolio_value:,.0f}원")

        
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # 끝 매 영업일 반복
        ###################################
        logging.info("백테스트 완료. 최종 보고서 생성 중...")
        run_id = None # run_id 초기화
        # 백테스트 결과 보고서 생성 및 저장
        # 초기 포트폴리오 항목 (시작일 이전의 초기 자본)은 제외하고 실제 백테스트 기간 데이터만 사용
        daily_strategy_names = ', '.join([s.__class__.__name__ for s in self.daily_strategies])
        daily_strategy_params = {s.__class__.__name__: s.strategy_params for s in self.daily_strategies}
        if len(self.portfolio_values) > 1:
            portfolio_df = pd.DataFrame(self.portfolio_values[1:], columns=['Date', 'PortfolioValue'])
            portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
            portfolio_value_series = portfolio_df.set_index('Date')['PortfolioValue']
        else:
            # 백테스트 기간이 너무 짧아 포트폴리오 가치 데이터가 없는 경우
            portfolio_value_series = pd.Series(dtype=float)

        if self.save_to_db and not portfolio_value_series.empty:
            storage = BacktestDB(self.manager.get_db_manager())
            reporter = ReportGenerator(storage_strategy=storage)

            run_id = reporter.generate(
                start_date=start_date,
                end_date=end_date,
                initial_cash=self.initial_cash,
                portfolio_value_series=portfolio_value_series,
                transaction_log=self.broker.transaction_log,
                strategy_info={
                    # [수정] self.daily_strategy -> daily_strategy_names
                    'strategy_daily': daily_strategy_names,
                    'strategy_minute': self.minute_strategy.__class__.__name__,
                    # [수정] self.daily_strategy.strategy_params -> daily_strategy_params
                    'params_json_daily': daily_strategy_params,
                    'params_json_minute': self.minute_strategy.strategy_params
                }
            )
            # 5. [추가] 백테스트 완료 후, 얻은 run_id로 즉시 프로파일링 실행
            #self.run_profiling(run_id)

        elif portfolio_value_series.empty:
             logger.warning("포트폴리오 가치 데이터가 비어 있어 DB에 결과를 저장하지 않습니다.")
        else:
             logger.info("save_to_db=False로 설정되어 DB에 결과를 저장하지 않습니다.")
        
        final_metrics = calculate_performance_metrics(portfolio_value_series)
        return portfolio_value_series, final_metrics

    def _aggregate_signals(self, signals_from_all_strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        여러 전략에서 생성된 신호들을 제안된 정책에 따라 통합합니다.
        정책 우선순위: 1.매도 우선 -> 2.스코어 절대값 -> 3.수량
        """
        final_signals = {}
        
        for strategy_signals in signals_from_all_strategies:
            for stock_code, new_signal in strategy_signals.items():
                if stock_code not in final_signals:
                    # 새로운 종목의 신호는 바로 추가
                    final_signals[stock_code] = new_signal
                    continue

                # --- 신호 충돌 발생 시 ---
                existing_signal = final_signals[stock_code]

                # 1. 매도 우선 정책
                if existing_signal['signal_type'] == 'buy' and new_signal['signal_type'] == 'sell':
                    final_signals[stock_code] = new_signal # 매도 신호로 덮어쓰기
                    continue
                if existing_signal['signal_type'] == 'sell' and new_signal['signal_type'] == 'buy':
                    continue # 기존 매도 신호 유지

                # 신호 타입이 같은 경우 (buy vs buy, sell vs sell)
                # 2. 스코어 절대값 우선 정책
                existing_score = abs(existing_signal.get('score', 0))
                new_score = abs(new_signal.get('score', 0))
                if new_score > existing_score:
                    final_signals[stock_code] = new_signal
                    continue
                
                # 3. 수량 우선 정책 (스코어가 같을 경우)
                if new_score == existing_score:
                    existing_qty = existing_signal.get('target_quantity', 0)
                    new_qty = new_signal.get('target_quantity', 0)
                    if new_qty > existing_qty:
                        final_signals[stock_code] = new_signal
                        continue
        
        return final_signals

    # [신규] 특정 날짜의 종가를 가져오는 헬퍼 메서드
    def _get_prices_for_date(self, target_date: dt.date) -> Dict[str, float]:
        """data_store에서 특정 날짜의 모든 종목 종가를 조회합니다."""
        prices = {}
        for code, df in self.data_store['daily'].items():
            try:
                # Timestamp로 변환하여 정확히 해당 날짜의 데이터를 찾음
                price = df.loc[pd.Timestamp(target_date).normalize()]['close']
                prices[code] = price
            except KeyError:
                # 해당 날짜에 데이터가 없는 경우 (거래정지 등)
                continue
        return prices    

    def get_all_current_prices(self, current_dt):
        current_prices = {}
        for code in self.broker.positions.keys():
            price_data = self.minute_strategy._get_bar_at_time('minute', code, current_dt)
            # 가격 데이터가 있고, 종가가 NaN이 아닌 경우에만 추가
            if price_data is not None and not np.isnan(price_data['close']):
                current_prices[code] = price_data['close']  
        return current_prices
   
    def cleanup(self) -> None:
        """
        시스템 종료 시 필요한 리소스 정리 작업을 수행합니다.
        """
        logger.info("Backtest 시스템 cleanup 시작.")
        if self.broker:
            self.broker.cleanup()
        if self.manager:
            self.manager.close()
        logger.info("Backtest 시스템 cleanup 완료.") 

#     def run_profiling(self, run_id: int):
#         """
#         특정 run_id에 대한 프로파일링을 수행하는 헬퍼 함수.
#         """
#         logger.info(f"\n{'='*20} run_id: {run_id} 프로파일링 시작 {'='*20}")
        
#         try:
#             model_info = self.manager.fetch_hmm_model_by_name(LIVE_HMM_MODEL_NAME)
#             if not model_info:
#                 logger.error(f"'{LIVE_HMM_MODEL_NAME}' 모델이 DB에 없어 프로파일링을 중단합니다.")
#                 return

#             model_id = model_info['model_id']
#             run_info_df = self.manager.fetch_backtest_run(run_id=run_id)
#             performance_df = self.manager.fetch_backtest_performance(run_id=run_id)
#             regime_data_df = self.manager.fetch_daily_regimes(model_id=model_id)
#             # DB에서 Decimal 타입으로 가져온 데이터를 시스템 표준인 float으로 변환합니다.
#             if 'daily_return' in performance_df.columns:
#                 performance_df['daily_return'] = performance_df['daily_return'].astype(float)

#             if run_info_df.empty or performance_df.empty or regime_data_df.empty:
#                 logger.error("프로파일링에 필요한 데이터가 부족합니다.")
#                 return

#             profiler = StrategyProfiler()
#             profiles_to_save = profiler.generate_profiles(
#                 performance_df, regime_data_df, run_info_df, model_id
#             )

#             if profiles_to_save:
#                 if self.manager.save_strategy_profiles(profiles_to_save):
#                     logger.info(f"✅ run_id: {run_id}에 대한 프로파일을 DB에 성공적으로 저장했습니다.")
#                 else:
#                     logger.error("프로파일 DB 저장에 실패했습니다.")
#         finally:
#             self.manager.close()
#             logger.info(f"{'='*20} run_id: {run_id} 프로파일링 종료 {'='*20}")


# def run_backtest_and_save_to_db(strategy_name: str, strategy_params: dict, start_date: date, end_date: date, backtest_manager: BacktestManager):
#     """
#     주어진 파라미터로 단일 전략의 백테스트를 실행하고, 그 결과를 DB에 저장합니다.
#     """
#     logger.info(f"\n--- Running backtest for '{strategy_name}' and saving to DB ---")
#     logger.info(f"Period: {start_date} ~ {end_date}")
#     logger.info(f"Params: {strategy_params}")

#     # 1. 컴포넌트 초기화
    
#     # 2. Backtest 인스턴스 생성
#     backtest_system = HMMBacktest(
#         manager=backtest_manager,
#         initial_cash=INITIAL_CASH,
#         start_date=start_date,
#         end_date=end_date,
#         save_to_db=True # <-- DB 저장을 활성화
#     )
#     active_strategies = {name: config for name, config in STRATEGY_CONFIGS.items() if config.get("strategy_status")}
#     # 3. 전략 인스턴스 생성
#     strategy_config = STRATEGY_CONFIGS.get(strategy_name)
#     if not strategy_config or not strategy_config.get("strategy_status"):
#         logger.warning(f"'{strategy_name}' 전략은 현재 활성화(strategy_status=True) 상태가 아닙니다.")
#         # 필요에 따라 여기서 return할 수 있습니다.
#         # return 

#     # 2. globals()를 사용해 strategy_name(문자열)에 해당하는 실제 '클래스 객체'를 찾습니다.
#     strategy_class = globals().get(strategy_name)
#     if not strategy_class:
#         logger.error(f"전략 클래스 '{strategy_name}'를 찾을 수 없습니다. 파일 상단에 import 되었는지 확인하세요.")
#         return

#     # settings.py의 기본 파라미터와 최적화된 파라미터를 결합
#     base_params = {}
#     base_params = STRATEGY_CONFIGS.get(strategy_name, {}).get('default_params', {}).copy()
#     final_strategy_params = {**base_params, **strategy_params}

#     daily_strategy_instance = strategy_class(
#         broker=backtest_system.broker, 
#         data_store=backtest_system.data_store
#     )
#     # 분봉 전략은 최적화 대상이 아니므로 PassMinute 사용
#     minute_strategy_instance = PassMinute(
#         broker=backtest_system.broker,
#         data_store=backtest_system.data_store
#     )

#     # 4. 전략 설정 및 백테스트 실행
#     backtest_system.set_strategies(
#         daily_strategies=[daily_strategy_instance],
#         minute_strategy=minute_strategy_instance,
#         stop_loss_params=STOP_LOSS_PARAMS
#     )
    
#     backtest_system.prepare_for_system()
#     backtest_system.run()
#     backtest_system.cleanup()
#     logger.info(f"--- Finished backtest for '{strategy_name}' ---")


# # --- ▼ [신규] 최종 Out-of-Sample 테스트를 위한 래퍼 함수 ▼ ---

# def run_final_backtest(model_name: str, start_date: date, end_date: date, policy_dict: dict, backtest_manager: BacktestManager) -> pd.Series:
#     """
#     주어진 HMM 모델과 최적화된 정책으로 Out-of-Sample 기간에 대한 최종 백테스트를 실행하고,
#     일별 포트폴리오 가치 시리즈를 반환합니다.
#     """
#     logger.info(f"\n--- Running FINAL Out-of-Sample Test for '{model_name}' ---")
#     logger.info(f"Test Period: {start_date} ~ {end_date}")
#     logger.info(f"Policy: {policy_dict}")

#     # 1. 컴포넌트 초기화

#     # 2. Backtest 인스턴스 생성 (DB 저장은 False)
#     backtest_system = HMMBacktest(
#         manager=backtest_manager,
#         initial_cash=INITIAL_CASH,
#         start_date=start_date,
#         end_date=end_date,
#         save_to_db=False # <-- 최종 테스트 결과는 DB에 저장하지 않음
#     )

#     # 3. 전략 인스턴스 리스트 생성
#     daily_strategies_list = []
#     # STRATEGY_CONFIGS를 순회하며 'strategy_status'가 True인 전략만 객체로 생성
#     for name, config in STRATEGY_CONFIGS.items():
#         if config.get("strategy_status") is True:
#             # globals()에서 이름으로 실제 클래스를 찾습니다.
#             strategy_class = globals().get(name) 
#             if strategy_class:
#                 instance = strategy_class(
#                     broker=backtest_system.broker,
#                     data_store=backtest_system.data_store
#                 )
#                 daily_strategies_list.append(instance)
#             else:
#                 logger.warning(f"설정 파일에 있는 '{name}' 전략 클래스를 찾을 수 없어 건너뜁니다.")

#     minute_strategy_instance = PassMinute(broker=backtest_system.broker, data_store=backtest_system.data_store)
#     backtest_system.set_strategies(
#         daily_strategies=daily_strategies_list,
#         minute_strategy=minute_strategy_instance
#     )
#     # 4. 데이터 준비 및 백테스트 실행
#     backtest_system.prepare_for_system()

#     portfolio_series, _, _, _ = backtest_system.reset_and_rerun(
#         daily_strategies=daily_strategies_list,
#         minute_strategy=minute_strategy_instance,
#         mode='hmm',
#         model_name=model_name,
#         policy_dict=policy_dict # <-- 최적화된 정책 주입
#     )

#     backtest_system.cleanup()
#     logger.info(f"--- Finished FINAL Out-of-Sample Test for '{model_name}' ---")

#     return portfolio_series


if __name__ == "__main__":
    """
    Backtest 클래스 테스트 실행 코드
    """
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/backtest_run.log", encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])    
    try:
        # 1. 핵심 객체 생성
        api_client = CreonAPIClient()
        db_manager = DBManager()
        backtest_manager = BacktestManager(api_client, db_manager)
        # 5. 백테스트 준비 및 실행
        start_date = date(2024, 8, 1)
        end_date = date(2025, 7, 30)

        # end_date = datetime.now().date() - timedelta(days=1)
        # start_date = datetime.now().date() - timedelta(days=90)
        
        # 2. Backtest 인스턴스 생성
        backtest_system = HMMBacktest(
            manager=backtest_manager,
            initial_cash=INITIAL_CASH,
            start_date=start_date,
            end_date=end_date,
            save_to_db=True
        )
        
        daily_strategies_to_run = [
            # BreakoutDaily(
            #     broker=backtest_system.broker,
            #     data_store=backtest_system.data_store
            # ),
            # PullbackDaily(
            #     broker=backtest_system.broker,
            #     data_store=backtest_system.data_store
            # ),
            ClosingBetDaily(
                broker=backtest_system.broker,
                data_store=backtest_system.data_store
            )
            
        ]
        # 매도 전략
        # from strategies.target_price_minute import TargetPriceMinute
        # minute_strategy = TargetPriceMinute(broker=backtest_system.broker, data_store=backtest_system.data_store)
        from strategies.pass_minute import PassMinute
        minute_strategy = PassMinute(broker=backtest_system.broker, data_store=backtest_system.data_store)

        # 4. 전략 설정
        backtest_system.set_strategies(
            daily_strategies=daily_strategies_to_run, # 전략 1개만 넣으려면 [일봉전략]
            minute_strategy=minute_strategy,
            stop_loss_params=STOP_LOSS_PARAMS
        )
        # 5. 백테스트 실행
        backtest_system.prepare_for_system()
        backtest_system.run()

    except Exception as e:
        logger.error(f"Backtest 테스트 중 오류 발생: {e}", exc_info=True)
    finally:
        if 'backtest_system' in locals():
            backtest_system.cleanup()