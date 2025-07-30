# backtest/backtester.py
from datetime import datetime as dt, time, timedelta 
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
from manager.backtest_manager import BacktestManager # BacktestManager 타입 힌트를 위해 남겨둠
from manager.db_manager import DBManager    
from util.strategies_util import *
from strategies.strategy import DailyStrategy, MinuteStrategy
from trading.abstract_report import ReportGenerator, BacktestDB
from manager.capital_manager import CapitalManager
from analyzer.hmm_model import RegimeAnalysisModel
from analyzer.inference_service import RegimeInferenceService
from analyzer.policy_map import PolicyMap
from manager.portfolio_manager import PortfolioManager
from config.settings import PRINCIPAL_RATIO, STRATEGY_CONFIGS # 자금 관리 설정 임포트

# 설정 파일 로드
from config.settings import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 
    FETCH_DAILY_PERIOD, FETCH_MINUTE_PERIOD,
    MARKET_OPEN_TIME, MARKET_CLOSE_TIME,
    INITIAL_CASH, STOP_LOSS_PARAMS, COMMON_PARAMS,
    SMA_DAILY_PARAMS, DUAL_MOMENTUM_DAILY_PARAMS, TRIPLE_SCREEN_DAILY_PARAMS
)     
logger = logging.getLogger(__name__)

class Backtest:
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
        self.capital_manager = CapitalManager(self.broker, STRATEGY_CONFIGS)
        self.start_date = start_date
        self.end_date = end_date
        self.daily_strategies: List[DailyStrategy] = []
        self.minute_strategy: Optional[MinuteStrategy] = None
        self.portfolio_values = [] # self.broker.get_portfolio_value #########
        
        self.data_store = {'daily': {}, 'minute': {}} 
        
        self.save_to_db = save_to_db


        self.market_open_time = dt.strptime(MARKET_OPEN_TIME, '%H:%M:%S').time()
        self.market_close_time = dt.strptime(MARKET_CLOSE_TIME, '%H:%M:%S').time()
        
        self.last_portfolio_check = None # 포트폴리오 손절 체크 시간을 추적하기 위한 변수
        self.portfolio_stop_flag = False # 포트폴리오 손절 발생 시 당일 매매 중단 플래그

        self._minute_data_cache = {}   # {stock_code: DataFrame (오늘치 분봉 데이터)}
        self.price_cache = {}           # 가격 캐시를 Backtest 클래스가 직접 소유

        logging.info(f"백테스터 초기화 완료. 초기 현금: {self.initial_cash:,.0f}원, DB저장: {self.save_to_db}")

    def set_strategies(self, daily_strategies: List[DailyStrategy], minute_strategy: MinuteStrategy, stop_loss_params: dict = None):
        self.daily_strategies = daily_strategies
        self.minute_strategy = minute_strategy
        if stop_loss_params:
            self.broker.set_stop_loss_params(stop_loss_params)
        daily_strategy_names = ', '.join([s.__class__.__name__ for s in self.daily_strategies])
        logger.info(f"전략 설정 완료: Daily(s)='{daily_strategy_names}', Minute='{self.minute_strategy.__class__.__name__}'")

    def prepare_for_system(self) -> bool:
        """백테스팅 기간내 일봉 분봉 데이터를 미리 data_store 에 저장 합니다."""
        start_date = self.start_date
        end_date = self.end_date
        logging.info(f"--- 백테스트 데이터 준비 시작 ({start_date} ~ {end_date}) ---")
        
        # 1. 유니버스 종목 코드 결정 (모든 전략이 필요로 하는 종목 총망라)
        universe_codes = set()
        for strategy in self.daily_strategies:
            # 각 전략의 파라미터에서 유니버스 관련 정보를 가져오는 로직 (필요시 구현)
            # 여기서는 모든 종목을 로드한다고 가정
            pass 
        # 임시로 trading_manager와 유사하게 전체 유니버스 로드
        universe_codes.update(self.manager.get_universe_codes())
        
        # COMMON_PARAMS의 필수 코드 추가
        market_code = COMMON_PARAMS.get('market_index_code')
        if market_code: universe_codes.add(market_code)
        safe_asset_code = COMMON_PARAMS.get('safe_asset_code')
        if safe_asset_code: universe_codes.add(safe_asset_code)

        # 2. 전체 기간 데이터 사전 로딩
        daily_start = start_date - timedelta(days=FETCH_DAILY_PERIOD) # 지표 계산을 위한 충분한 과거 데이터
        minute_start = start_date - timedelta(days=FETCH_MINUTE_PERIOD) # 지표 계산을 위한 충분한 과거 데이터
        
        all_trading_dates_list = self.manager.db_manager.get_all_trading_days(daily_start, end_date)
        all_trading_dates_set = set(all_trading_dates_list)
        
        for code in list(universe_codes):
            logging.info(f"데이터 로딩: {code} 일봉: ({daily_start} ~ {end_date}), 분봉: ({minute_start} ~ {end_date})")
            # 일봉 데이터 로드
            daily_df = self.manager.cache_daily_ohlcv(code, daily_start, end_date, all_trading_dates_set)
            if not daily_df.empty: self.data_store['daily'][code] = daily_df
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


    def reset_and_rerun(self, daily_strategies: List[DailyStrategy], minute_strategy: MinuteStrategy, 
                        mode: str = 'strategy', hmm_params: dict = None) -> tuple:
        """
        [신규] data_store는 유지한 채, 브로커와 전략만 리셋하고 백테스트를 재실행합니다.
        Optimizer가 데이터 재로딩 없이 빠르게 반복 테스트를 수행하기 위해 사용됩니다.
        """
        logger.info("--- 백테스트 리셋 및 재실행 시작 ---")

        # if not self.daily_strategies or not self.minute_strategy:
        #     logger.error("일봉 또는 분봉 전략이 설정되지 않았습니다. 백테스트를 중단합니다.")
        #     return pd.Series(dtype=float), {}
                
        # 1. 브로커 및 포트폴리오 상태 초기화
        self.broker = Broker(manager=self.manager, initial_cash=self.original_initial_cash)
        self.portfolio_values = []

        # 2. HMM 모드일 경우, 동적 PortfolioManager 생성
        if mode == 'hmm':
            # --- [핵심 수정] ---
            # 미리 저장된 모델을 로드하는 대신, 파라미터에 따라 모델을 동적으로 훈련합니다.
            
            # 2.1 HMM 학습에 필요한 과거 시장 데이터 가져오기
            # start_date는 이 백테스트의 시작일이므로, 그 이전 데이터로 학습해야 합니다.
            hmm_training_data = self.manager.get_market_data_for_hmm(self.start_date)
            if hmm_training_data.empty:
                raise ValueError("HMM 모델 학습용 데이터를 가져올 수 없습니다.")

            # 2.2 옵티마이저가 전달한 n_states 파라미터로 HMM 모델 생성 및 학습
            n_states = hmm_params.get('hmm_n_states', 3) # 파라미터에서 상태 개수 가져오기
            hmm_model = RegimeAnalysisModel(n_states=n_states)
            hmm_model.fit(hmm_training_data)
            inference_service = RegimeInferenceService(hmm_model)
            
            # 2.3 정책 맵 동적 생성 (기존 코드는 올바름)
            policy_map = PolicyMap()
            policy_map.rules = {
                "regime_to_principal_ratio": {
                    "0": 1.0, 
                    "1": hmm_params.get('policy_bear_ratio', 0.5),
                    "2": hmm_params.get('policy_crisis_ratio', 0.2)
                }, "default_principal_ratio": 1.0
            }
            
            # 2.4 HMM 두뇌를 탑재한 PortfolioManager 생성
            self.portfolio_manager = PortfolioManager(
                self.broker, STRATEGY_CONFIGS, inference_service, policy_map
            )
            
            # 2.5 전략 프로파일 로드 (지금은 비워두어 정적 가중치 사용)
            # 추후 '최적화 단계 분리'를 적용할 때 이 부분을 활성화합니다.
            strategy_profiles = {} 
        else:
            # 'strategy' 모드일 경우, 기존의 정적 CapitalManager 사용
            self.capital_manager = CapitalManager(self.broker, STRATEGY_CONFIGS)

        # 3. 전략 객체들의 브로커 참조 업데이트 및 설정
        for strategy in daily_strategies:
            strategy.broker = self.broker

        minute_strategy.broker = self.broker

        self.set_strategies(daily_strategies, minute_strategy)

        # =================================================================
        # 3. 기존 run() 메서드의 핵심 실행 로직 (데이터 로딩 제외)
        # =================================================================
        # 4. 백테스트 메인 루프 실행
        start_date = self.start_date
        end_date = self.end_date      
        market_calendar_df = self.manager.fetch_market_calendar(start_date, end_date)
        trading_dates = market_calendar_df[market_calendar_df['is_holiday'] == 0]['date'].dt.date.tolist()

        for i, current_date in enumerate(trading_dates):
            # --- 자금 할당 로직 ---
            prev_date = trading_dates[i - 1] if i > 0 else self.start_date - timedelta(days=1)
            prev_prices = self._get_prices_for_date(prev_date)

            if mode == 'hmm':
                # HMM 모드: 동적 자산배분
                account_equity = self.portfolio_manager.get_account_equity(prev_prices)
                hmm_input_data = self.manager.get_market_data_for_hmm(current_date)
                total_principal, regime_probs = self.portfolio_manager.get_total_principal(account_equity, hmm_input_data)
                strategy_capitals = self.portfolio_manager.get_strategy_capitals(total_principal, regime_probs, strategy_profiles)
            else:
                # 전략 모드: 정적 자산배분
                account_equity = self.capital_manager.get_account_equity(prev_prices)
                total_principal = self.capital_manager.get_total_principal(account_equity, PRINCIPAL_RATIO)

            # 복수 일봉 전략 실행 및 신호 통합
            signals_from_all = []
            for strategy in self.daily_strategies:
                if mode == 'hmm':
                    strategy_capital = strategy_capitals.get(strategy.strategy_name, 0)
                else:
                    strategy_capital = self.capital_manager.get_strategy_capital(strategy.strategy_name, total_principal)

                strategy.run_daily_logic(current_date, strategy_capital)
                signals_from_all.append(strategy.signals)
            
            final_signals = self._aggregate_signals(signals_from_all)
            self.minute_strategy.update_signals(final_signals)

            # 분봉 루프 실행
            market_open_dt = dt.combine(current_date, self.market_open_time)
            stocks_to_trade = set(self.broker.get_current_positions().keys()) | set(final_signals.keys())

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
        return portfolio_value_series, final_metrics
                
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
            total_principal = self.capital_manager.get_total_principal(account_equity, PRINCIPAL_RATIO)

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

            reporter.generate(
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
        elif portfolio_value_series.empty:
             logger.warning("포트폴리오 가치 데이터가 비어 있어 DB에 결과를 저장하지 않습니다.")
        else:
             logger.info("save_to_db=False로 설정되어 DB에 결과를 저장하지 않습니다.")


        final_metrics = calculate_performance_metrics(portfolio_value_series)
        return portfolio_value_series, final_metrics
    
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


    # 분봉 데이터는 run 루프 내에서 cache_minute_ohlcv를 통해 직접 data_store['minute']에 저장됩니다.
    # def _should_check_portfolio(self, current_dt):
    #     """포트폴리오 체크가 필요한 시점인지 확인합니다."""
    #     if self.last_portfolio_check is None:
    #         self.last_portfolio_check = current_dt # 첫 체크 시점 기록
    #         return True
        
    #     current_time = current_dt.time()
    #     # 시간 비교를 정수로 변환하여 효율적으로 비교
    #     current_minutes = current_time.hour * 60 + current_time.minute
    #     check_minutes = [9 * 60, 15 * 60 + 20]  # 9:00, 15:20
        
    #     # 오늘 날짜이고, 체크 시간에 해당하며, 마지막 체크 시점이 오늘이 아니거나 해당 체크 시간이 아닌 경우
    #     if current_dt.date() == self.last_portfolio_check.date():
    #         if current_minutes in check_minutes and (self.last_portfolio_check.hour * 60 + self.last_portfolio_check.minute) != current_minutes:
    #             self.last_portfolio_check = current_dt
    #             return True
    #     elif current_dt.date() > self.last_portfolio_check.date(): # 날짜가 바뀌면 무조건 체크
    #         self.last_portfolio_check = current_dt
    #         return True
            
    #     return False

    # 오늘의 일봉 데이터가 미래정보를 가지지 않도록 제한한
    # def _update_daily_data_from_minute_bars(self, current_dt: dt.datetime):
    #     """
    #     매분 현재 시각까지의 1분봉 데이터를 집계하여 일봉 데이터를 생성하거나 업데이트합니다.
    #     _minute_data_cache를 활용하여 성능을 개선합니다.
    #     :param current_dt: 현재 시각 (datetime 객체)
    #     """
    #     current_date = current_dt.date()
        
    #     for stock_code in self.data_store['daily'].keys():
    #         # _minute_data_cache에서 해당 종목의 오늘치 전체 분봉 데이터를 가져옵니다.
    #         # 이 데이터는 run 함수 진입 시 cache_minute_ohlcv를 통해 이미 로드되어 있어야 합니다.
    #         today_minute_bars = self._minute_data_cache.get(stock_code)

    #         if today_minute_bars is not None and not today_minute_bars.empty:
    #             # 현재 시각까지의 분봉 데이터만 필터링 (슬라이싱)
    #             # 이 부분에서 불필요한 복사를 줄이기 위해 .loc을 사용합니다.
    #             filtered_minute_bars = today_minute_bars.loc[today_minute_bars.index <= current_dt]
                
    #             if not filtered_minute_bars.empty:
    #                 # 현재 시각까지의 일봉 데이터 계산
    #                 daily_open = filtered_minute_bars.iloc[0]['open']  # 첫 분봉 시가
    #                 daily_high = filtered_minute_bars['high'].max()    # 현재까지 최고가
    #                 daily_low = filtered_minute_bars['low'].min()      # 현재까지 최저가
    #                 daily_close = filtered_minute_bars.iloc[-1]['close']  # 현재 시각 종가
    #                 daily_volume = filtered_minute_bars['volume'].sum()   # 현재까지 누적 거래량

    #                 # 새로운 일봉 데이터 생성 (Series로 생성하여 성능 개선)
    #                 new_daily_bar = pd.Series({
    #                     'open': daily_open,
    #                     'high': daily_high,
    #                     'low': daily_low,
    #                     'close': daily_close,
    #                     'volume': daily_volume
    #                 }, name=pd.Timestamp(current_date)) # 인덱스를 날짜로 설정

    #                 # 일봉 데이터가 존재하면 업데이트, 없으면 추가
    #                 # .loc을 사용하여 직접 업데이트 (기존 DataFrame의 인덱스를 활용)
    #                 self.data_store['daily'][stock_code].loc[pd.Timestamp(current_date)] = new_daily_bar
                    
    #                 # 일봉 데이터가 추가되거나 업데이트될 때 인덱스 정렬은 필요 없음 (loc으로 특정 위치 업데이트)
    #                 # 단, 새로운 날짜가 추가될 경우 기존 DataFrame에 없던 인덱스가 추가되므로 sort_index는 필요할 수 있습니다.
    #                 # 하지만 백테스트에서는 날짜 순서대로 진행되므로 대부분의 경우 문제가 되지 않습니다.
    #         else:
    #             logging.debug(f"[{current_dt.isoformat()}] {stock_code}의 오늘치 분봉 데이터가 없거나 비어있어 일봉 업데이트를 건너킵니다.")


    # def _clear_daily_update_cache(self):
    #     """
    #     일봉 업데이트를 위한 분봉 데이터 캐시를 초기화합니다. 새로운 날짜로 넘어갈 때 호출됩니다.
    #     """
    #     self._minute_data_cache.clear()
    #     logging.debug("일봉 업데이트를 위한 분봉 데이터 캐시 초기화 완료.")


    # def add_daily_data(self, stock_code: str, df: pd.DataFrame):
    #     """백테스트를 위한 일봉 데이터를 추가합니다."""
    #     if not df.empty:
    #         self.data_store['daily'][stock_code] = df
    #         logging.debug(f"일봉 데이터 추가: {stock_code}, {len(df)}행")
    #     else:
    #         logging.warning(f"빈 데이터프레임이므로 {stock_code}의 일봉 데이터를 추가하지 않습니다.")

    # def load_stocks(self, start_date, end_date):
    #     from config.sector_stocks import sector_stocks

    #     fetch_start = start_date - timedelta(days=60) # 전략에 필요한 최대 기간 + 여유
    #     stock_codes_to_load = []
    #     for sector, stocks in sector_stocks.items():
    #         for stock_name, _ in stocks:
    #             code = self.api_client.get_stock_code(stock_name)
    #             if code:
    #                 stock_codes_to_load.append(code)
    #             else:
    #                 logging.warning(f"'{stock_name}' 종목의 코드를 찾을 수 없습니다. 해당 종목을 건너킵니다.")
    #     # 하드 코딩으로 추가 ###############
    #     stock_codes_to_load.append('U001')
    #     for code in stock_codes_to_load:
    #         stock_name = self.api_client.get_stock_name(code) # 종목명 다시 가져오기
    #         logging.info(f"'{stock_name}' (코드: {code}) 종목 일봉 데이터 로딩 중... (기간: {fetch_start.strftime('%Y%m%d')} ~ {end_date.strftime('%Y%m%d')})")
    #         daily_df = self.manager.cache_daily_ohlcv(code, fetch_start, end_date)
            
    #         if daily_df.empty:
    #             logging.warning(f"{stock_name} ({code}) 종목의 일봉 데이터를 가져올 수 없습니다. 해당 종목을 건너킵니다.")
    #             continue
    #         logging.debug(f"{stock_name} ({code}) 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
    #         self.add_daily_data(code, daily_df)
            
    #         # 분봉 데이터는 run 루프 내에서 필요한 시점에 로드되므로, 여기서는 미리 로드하지 않습니다.
    #         # 이전에 add_minute_data를 통해 전체 분봉을 로드하는 방식은 제거되었습니다.


if __name__ == "__main__":
    """
    Backtest 클래스 테스트 실행 코드
    """
    from datetime import date, datetime
    # 설정 파일 로드
    from config.settings import (
        INITIAL_CASH, STOP_LOSS_PARAMS, COMMON_PARAMS,
        SMA_DAILY_PARAMS, DUAL_MOMENTUM_DAILY_PARAMS, TRIPLE_SCREEN_DAILY_PARAMS,
        VOL_QUALITY_DAILY_PARAMS, RSI_REVERSION_DAILY_PARAMS, VOL_BREAKOUT_DAILY_PARAMS,
        PAIRS_TRADING_DAILY_PARAMS, INVERSE_DAILY_PARAMS
    )     

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
        start_date = date(2025, 6, 1)
        end_date = date(2025, 7, 1)
        # 2. Backtest 인스턴스 생성
        backtest_system = Backtest(
            manager=backtest_manager,
            initial_cash=INITIAL_CASH,
            start_date=start_date,
            end_date=end_date,
            save_to_db=True
        )

        # 3. 전략 인스턴스 생성 (복수 전략)

        from strategies.sma_daily import SMADaily
        sma_strategy = SMADaily(broker=backtest_system.broker, data_store=backtest_system.data_store, strategy_params=SMA_DAILY_PARAMS)

        from strategies.dual_momentum_daily import DualMomentumDaily
        dm_strategy = DualMomentumDaily(broker=backtest_system.broker, data_store=backtest_system.data_store, strategy_params=DUAL_MOMENTUM_DAILY_PARAMS)
        # 거래량 돌파 전략
        from strategies.vol_breakout_daily import VolBreakoutDaily
        vb_strategy = VolBreakoutDaily(broker=backtest_system.broker, data_store=backtest_system.data_store, strategy_params=VOL_BREAKOUT_DAILY_PARAMS)

        from strategies.triple_screen_daily import TripleScreenDaily
        ts_strategy = TripleScreenDaily(broker=backtest_system.broker, data_store=backtest_system.data_store, strategy_params=TRIPLE_SCREEN_DAILY_PARAMS)


        from strategies.vol_quality_daily import VolQualityDaily
        vq_strategy = VolQualityDaily(broker=backtest_system.broker, data_store=backtest_system.data_store, strategy_params=VOL_QUALITY_DAILY_PARAMS)
        # 
        from strategies.rsi_reversion_daily import RsiReversionDaily
        rsi_strategy = RsiReversionDaily(broker=backtest_system.broker, data_store=backtest_system.data_store, strategy_params=RSI_REVERSION_DAILY_PARAMS)
        
        # 페어 전략
        from strategies.pairs_trading_daily import PairsTradingDaily
        pt_strategy = PairsTradingDaily(broker=backtest_system.broker, data_store=backtest_system.data_store, strategy_params=PAIRS_TRADING_DAILY_PARAMS)
        # 역추세 전략
        from strategies.inverse_daily import InverseDaily
        inverse_strategy = InverseDaily(broker=backtest_system.broker, data_store=backtest_system.data_store, strategy_params=INVERSE_DAILY_PARAMS)
        
        # 매도 전략
        # from strategies.target_price_minute import TargetPriceMinute
        # minute_strategy = TargetPriceMinute(broker=backtest_system.broker, data_store=backtest_system.data_store, strategy_params=COMMON_PARAMS)
        from strategies.pass_minute import PassMinute
        minute_strategy = PassMinute(broker=backtest_system.broker, data_store=backtest_system.data_store, strategy_params=COMMON_PARAMS)

        # 4. 전략 설정
        backtest_system.set_strategies(
            #daily_strategies=[sma_strategy, dm_strategy, vb_strategy],
            daily_strategies=[sma_strategy],
            #daily_strategies=[sma_strategy, ts_strategy, dm_strategy, vq_strategy, rsi_strategy, vb_strategy, pt_strategy, inverse_strategy],
            #daily_strategies=[ts_strategy, pt_strategy, inverse_strategy],
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