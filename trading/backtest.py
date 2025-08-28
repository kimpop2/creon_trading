# trading/backtest.py
from datetime import datetime, time, timedelta
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.broker import Broker
from manager.backtest_manager import BacktestManager
from manager.db_manager import DBManager
from util.indicators import calculate_performance_metrics
from strategies.strategy import DailyStrategy, MinuteStrategy
from trading.report_generator import ReportGenerator, BacktestDB
from manager.capital_manager import CapitalManager
from config.settings import MARKET_OPEN_TIME, MARKET_CLOSE_TIME
# 설정 파일 로드
from config.settings import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 
    FETCH_DAILY_PERIOD, FETCH_MINUTE_PERIOD,
    MARKET_OPEN_TIME, MARKET_CLOSE_TIME,
    INITIAL_CASH, STOP_LOSS_PARAMS, COMMON_PARAMS,
    SMA_DAILY_PARAMS, DUAL_MOMENTUM_DAILY_PARAMS, TRIPLE_SCREEN_DAILY_PARAMS,
    PRINCIPAL_RATIO, PORTFOLIO_FOR_HMM_OPTIMIZATION
)  
logger = logging.getLogger(__name__)

class Backtest:
    def __init__(self, 
                 manager: BacktestManager,
                 initial_cash: float, 
                 start_date: datetime.date,
                 end_date: datetime.date,
                 save_to_db: bool = True # DB 저장 여부
        ):
        self.manager = manager
        self.initial_cash = initial_cash
        self.original_initial_cash = initial_cash #리셋을 위한 원본 초기 자본금 저장
        self.broker = Broker(manager=manager, initial_cash=initial_cash)
        self.capital_manager = CapitalManager(self.broker, PORTFOLIO_FOR_HMM_OPTIMIZATION)
        self.start_date = start_date
        self.end_date = end_date
        self.save_to_db = save_to_db
        self.daily_strategies: List[DailyStrategy] = []
        self.minute_strategy: Optional[MinuteStrategy] = None

        self.market_open_time = datetime.strptime(MARKET_OPEN_TIME, '%H:%M:%S').time()
        self.market_close_time = datetime.strptime(MARKET_CLOSE_TIME, '%H:%M:%S').time()

        self.data_store = {'daily': {}, 'minute': {}}
        
        self.is_data_prepared = False

    def set_strategies(self, daily_strategies: List[DailyStrategy], minute_strategy: MinuteStrategy, stop_loss_params: dict = None):
        self.daily_strategies = daily_strategies
        self.minute_strategy = minute_strategy
        if stop_loss_params:
            self.broker.set_stop_loss_params(stop_loss_params)
        
        # backtest.py 에서는 이 전략들을 반복해서 백테스팅 한다. 
        daily_strategy_names = ', '.join([s.__class__.__name__ for s in self.daily_strategies])
        logger.info(f"전략 설정 완료: Daily(s)='{daily_strategy_names}', Minute='{self.minute_strategy.__class__.__name__}'")


    # 전략 백테스팅, 최적화 할 때 유니버스는 동일하므로 최초 1번만 준비비
    def prepare_for_system(self) -> bool:
        """백테스팅 기간내 일봉 분봉 데이터를 미리 data_store 에 저장 합니다."""
        if self.is_data_prepared:
            return
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
            if not daily_df.empty: 
                self.data_store['daily'][code] = daily_df
                stock_name = self.manager.get_stock_name(code)
                logging.info(f"[{stock_name}({code})] 일봉 데이터 {len(daily_df)}건 로드 완료")

            # 본봉 전략이 PassMinute 가 아닐 때는 분봉까지 로드 해야 함함
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
                    logging.info(f"[{stock_name}({code})] 분봉 데이터 {len(minute_df)}건 로드 완료")
        
        self.is_data_prepared = True
        logging.info(f"--- 백테스트 데이터 준비 완료 ---")

    # 옵티마이저에서 전략 최적화 stragegy 모드로 호출
    def reset_and_rerun(self, daily_strategy: DailyStrategy, minute_strategy: MinuteStrategy, stop_loss_params: dict = None):
        """
        [옵티마이저 호환용]
        data_store는 유지한 채, 브로커와 전략만 리셋하고 백테스트를 재실행합니다.
        Optimizer가 데이터 재로딩 없이 빠르게 반복 테스트를 수행하기 위해 사용됩니다.
        """
        logger.info("--- 백테스트 리셋 및 재실행 시작 ---")

        # 1. 브로커 및 포트폴리오 상태 초기화
        self.broker = Broker(manager=self.manager, initial_cash=self.original_initial_cash)
        daily_strategy.broker = self.broker
        minute_strategy.broker = self.broker
        self.set_strategies(daily_strategy, minute_strategy)
        if stop_loss_params:
            self.set_broker_stop_loss_params(stop_loss_params)
        
        self.portfolio_values = []

        # CapitalManager 사용
        self.capital_manager = CapitalManager(self.broker, PORTFOLIO_FOR_HMM_OPTIMIZATION)

        self.last_portfolio_check = None    # 손절을 위한 마지막 포트폴리오 체크 시간
        logging.debug("백테스터 상태가 초기화되었습니다. (데이터는 유지됨)")

        return self.run(self.start_date, self.end_date, from_rerun=True)


    def reset_and_rerun(self, daily_strategies: List[DailyStrategy], minute_strategy: MinuteStrategy, 
                        mode: str = 'strategy', hmm_params: dict = None) -> tuple:
        """
        [신규] data_store는 유지한 채, 브로커와 전략만 리셋하고 백테스트를 재실행합니다.
        Optimizer가 데이터 재로딩 없이 빠르게 반복 테스트를 수행하기 위해 사용됩니다.
        """
        logger.info("--- 백테스트 리셋 및 재실행 시작 ---")

        if not self.daily_strategies or not self.minute_strategy:
            logger.error("일봉 또는 분봉 전략이 설정되지 않았습니다. 백테스트를 중단합니다.")
            return pd.Series(dtype=float), {}
                
        # 1. 브로커 및 포트폴리오 상태 초기화
        self.broker = Broker(manager=self.manager, initial_cash=self.original_initial_cash)
        self.portfolio_values = []

        # CapitalManager 사용
        self.capital_manager = CapitalManager(self.broker, PORTFOLIO_FOR_HMM_OPTIMIZATION)
        minute_strategy.broker = self.broker

        # 3. 전략 객체들의 브로커 참조 업데이트 및 설정
        for daily_strategy in daily_strategies:
            strategy.broker = self.broker
            self.set_strategies(daily_strategy, minute_strategy)
            logging.debug("백테스터 상태가 초기화되었습니다. (데이터는 유지됨)")
            return self.run(self.start_date, self.end_date, from_rerun=True)
        

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
            market_open_dt = datetime.combine(current_date, self.market_open_time)
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
            close_prices = self.get_all_current_prices(datetime.combine(current_date, self.market_close_time))
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


    def set_broker_stop_loss_params(self, params: dict = None):
        self.broker.set_stop_loss_params(params)
        logging.info("브로커 손절매 파라미터 설정 완료.")


    def run(self, start_date: datetime.date, end_date: datetime.date, from_rerun: bool = False):
        """
        메인 백테스트 실행 메서드.
        '초고속 모드'와 '일반 모드'를 분기하여 실행합니다.
        """
        if not from_rerun:
            self.prepare_data_for_run(start_date, end_date)

        if not self.daily_strategy or not self.minute_strategy:
            logger.error("전략이 설정되지 않았습니다.")
            return None, {}

        # [핵심] 초고속 모드 확인
        is_fast_mode = getattr(self.minute_strategy, 'is_fast_simulation_strategy', False)
        
        market_calendar_df = self.manager.fetch_market_calendar(start_date, end_date)
        trading_dates = sorted(market_calendar_df[market_calendar_df['is_holiday'] == 0]['date'].dt.date.tolist())

        # 초기 자산 기록
        prev_date = self.manager.get_previous_trading_day(trading_dates[0]) if trading_dates else start_date
        self.portfolio_values.append((prev_date, self.initial_cash))

        for current_date in trading_dates:
            logging.info(f"\n--- 현재 날짜: {current_date.isoformat()} | 모드: {'초고속' if is_fast_mode else '일반'} ---")
            
            # [수정] 전략에 할당할 자본금을 계산합니다.
            # 전일 종가 기준 포트폴리오 가치를 오늘의 시작 자본금으로 사용합니다.
            strategy_capital = self.portfolio_values[-1][1]
            
            # 1. 일봉 로직 실행 (공통)
            # [수정] strategy_capital 인자를 함께 전달합니다.
            self.daily_strategy.run_daily_logic(current_date, strategy_capital)
            self.minute_strategy.update_signals(self.daily_strategy.signals)
            
            # 2. 매매 실행 (모드 분기)
            if is_fast_mode:
                self._run_fast_mode(current_date)
            else:
                self._run_normal_mode(current_date)

            # 3. 일일 결산 (공통)
            self._daily_settlement(current_date)

        # 4. 최종 결과 반환
        return self._generate_final_report(start_date, end_date)

    def _run_fast_mode(self, current_date: datetime.date):
        """초고속 모드: 분봉 루프 없이 하루치 거래를 한 번에 시뮬레이션"""
        logging.debug(f"[{current_date}] 초고속 모드 실행...")
        
        # 매매 대상: 보유 종목 + 신규 신호 종목
        stocks_to_trade = set(self.broker.get_current_positions().keys()) | set(self.minute_strategy.signals.keys())

        for stock_code in stocks_to_trade:
            # PassMinute의 일일 시뮬레이션 메서드 호출
            self.minute_strategy.simulate_daily_execution(current_date, stock_code)

    def _run_normal_mode(self, current_date: datetime.date):
        """일반 모드: 1분봉을 순회하며 매매를 시뮬레이션합니다."""
        logging.debug(f"[{current_date}] 일반 모드 실행 (1분봉 순회)...")
        
        # 1. 거래 시간 생성 (09:00 ~ 15:30)
        start_time = datetime.combine(current_date, self.market_open_time)
        end_time = datetime.combine(current_date, self.market_close_time)
        trading_minutes = pd.date_range(start=start_time, end=end_time, freq='1min')

        # 2. 1분 단위 루프 실행
        for current_dt in trading_minutes:
            # 매매 및 리스크 관리 대상: 보유 종목 + 신규 신호 종목
            stocks_to_process = set(self.broker.get_current_positions().keys()) | set(self.minute_strategy.signals.keys())
            
            # 현재 분(minute)의 모든 종목 가격을 담을 딕셔너리
            current_prices_this_minute = {}

            for stock_code in stocks_to_process:
                # 해당 분의 가격 정보(분봉) 가져오기
                minute_bar = self.minute_strategy._get_bar_at_time('minute', stock_code, current_dt)
                
                if minute_bar is None or minute_bar.empty:
                    continue # 해당 분에 거래가 없으면 건너뛰기

                # 현재가를 이 분봉의 종가로 간주
                current_price = minute_bar['close']
                current_prices_this_minute[stock_code] = current_price

                # 3. 분봉 전략 로직 실행
                self.minute_strategy.run_minute_logic(current_dt, stock_code)

            # 4. 모든 종목에 대한 분봉 전략 실행 후, 손절매 로직 일괄 실행
            if current_prices_this_minute:
                self.broker.check_and_execute_stop_loss(current_prices_this_minute, current_dt)

    def _daily_settlement(self, current_date: datetime.date):
        """일일 포트폴리오 가치 계산 및 로깅"""
        current_prices = {}
        positions = self.broker.get_current_positions()
        for stock_code in positions.keys():
            daily_bar = self.minute_strategy._get_bar_at_time('daily', stock_code, datetime.combine(current_date, datetime.min.time()))
            current_prices[stock_code] = daily_bar['close'] if daily_bar is not None else positions[stock_code]['avg_price']
            
        portfolio_value = self.broker.get_portfolio_value(current_prices)
        self.portfolio_values.append((current_date, portfolio_value))
        logging.info(f"[{current_date}] 장 마감. 포트폴리오 가치: {portfolio_value:,.0f}원")
        
        # 다음 날을 위해 신호 초기화
        self.daily_strategy._reset_all_signals()
        self.minute_strategy.signals = {}

    def _generate_final_report(self, start_date, end_date):
        """백테스트 종료 후 최종 리포트 생성"""
        if len(self.portfolio_values) > 1:
            portfolio_df = pd.DataFrame(self.portfolio_values, columns=['Date', 'PortfolioValue'])
            portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
            portfolio_value_series = portfolio_df.set_index('Date')['PortfolioValue']
        else:
            portfolio_value_series = pd.Series(dtype=float)

        if self.save_to_db and not portfolio_value_series.empty:
            storage = BacktestDB(self.manager.get_db_manager())
            reporter = ReportGenerator(storage_strategy=storage)
            reporter.generate(
                start_date=start_date, end_date=end_date, initial_cash=self.initial_cash,
                portfolio_value_series=portfolio_value_series, transaction_log=self.broker.transaction_log,
                strategy_info={
                    'strategy_daily': self.daily_strategy.strategy_name if self.daily_strategy else 'N/A',
                    'strategy_minute': self.minute_strategy.strategy_name if self.minute_strategy else 'N/A',
                    'params_json_daily': self.daily_strategy.strategy_params if self.daily_strategy else {},
                    'params_json_minute': self.minute_strategy.strategy_params if self.minute_strategy else {}
                })

        final_metrics = calculate_performance_metrics(portfolio_value_series)
        return portfolio_value_series, final_metrics
    
if __name__ == '__main__':
    # --- 의존성 및 설정 파일 임포트 ---
    from strategies.sma_daily import SMADaily
    from strategies.dual_momentum_daily import DualMomentumDaily
    from strategies.pass_minute import PassMinute
    from api.creon_api import CreonAPIClient
    from config.settings import (
        INITIAL_CASH, STOP_LOSS_PARAMS,
        SMA_DAILY_PARAMS, DUAL_MOMENTUM_DAILY_PARAMS
    )

    # --- 1. 로깅 설정 ---
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("logs/backtest_main_run.log", encoding='utf-8'),
                            logging.StreamHandler(sys.stdout)
                        ])

    # --- 2. 핵심 컴포넌트 초기화 ---
    try:
        api_client = CreonAPIClient()
        db_manager = DBManager()
        backtest_manager = BacktestManager(api_client, db_manager)
        
        backtest_system = Backtest(
            manager=backtest_manager,
            initial_cash=INITIAL_CASH,
            save_to_db=True
        )
        
        # --- 3. 테스트할 전략과 사용할 분봉 전략 '클래스' 정의 ---
        strategies_to_test = [
            {'class': SMADaily, 'params': SMA_DAILY_PARAMS},
            {'class': DualMomentumDaily, 'params': DUAL_MOMENTUM_DAILY_PARAMS},
        ]
        
        # --- 4. 백테스트 기간 설정 ---
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=365)
        
        # --- 5. 데이터 사전 로딩 (사용할 분봉 전략 '클래스' 정보 전달) ---
        logging.info("===== 데이터 사전 로딩 시작 =====")
        backtest_system.prepare_data_for_run(start_date, end_date)
        logging.info("===== 데이터 사전 로딩 완료 =====")
        
        minute_strategy = PassMinute(
            broker=backtest_system.broker, 
            data_store=backtest_system.data_store, 
            strategy_params={}
        )
        # --- 6. 전략 순차 실행 루프 (개선된 구조) ---
        for strat_config in strategies_to_test:
            strategy_class = strat_config['class']
            strategy_params = strat_config['params']
            strategy_name = strategy_class.__name__

            logging.info(f"\n{'='*20} [{strategy_name}] 백테스트 시작 {'='*20}")

            # [개선] 루프 안에서 매번 새로운 전략 인스턴스를 생성
            # 상태가 초기화된 깨끗한 broker와 data_store를 전달
            daily_strategy = strategy_class(
                broker=backtest_system.broker, 
                data_store=backtest_system.data_store, 
                strategy_params=strategy_params
            )
            # 상태 초기화 후 백테스트 실행
            backtest_system.reset_and_rerun(
                daily_strategy=daily_strategy,
                minute_strategy=minute_strategy,
                stop_loss_params=STOP_LOSS_PARAMS
            )

            logging.info(f"{'='*20} [{strategy_name}] 백테스트 완료 {'='*20}")

    except Exception as e:
        logger.error(f"백테스트 실행 중 오류 발생: {e}", exc_info=True)
    finally:
        if 'backtest_manager' in locals() and backtest_manager:
            backtest_manager.close()
        logger.info("모든 백테스트 프로세스 종료.")