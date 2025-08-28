# trading/trading.py (최종 수정본)
import pandas as pd
import logging
from datetime import datetime, date, timedelta, time
import time as pytime
from typing import Dict, Any, List, Optional
import sys
import os
import json
import pythoncom

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from trading.brokerage import Brokerage
from trading.report_generator import ReportGenerator, TradingDB
from trading.hmm_brain import HMMBrain
from manager.trading_manager import TradingManager
from strategies.strategy import DailyStrategy, MinuteStrategy
from util.notifier import Notifier
from util.indicators import calculate_performance_metrics
from manager.capital_manager import CapitalManager
from manager.portfolio_manager import PortfolioManager
# --- 사용할 모든 전략 클래스를 임포트해야 함 ---
from strategies.sma_daily import SMADaily
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.target_price_minute import TargetPriceMinute

from config.settings import (
    MIN_STOCK_CAPITAL, PRINCIPAL_RATIO, STRATEGY_CONFIGS, COMMON_PARAMS,
    MARKET_OPEN_TIME, MARKET_CLOSE_TIME
)
from config.settings import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 
    MARKET_OPEN_TIME, MARKET_CLOSE_TIME,
    INITIAL_CASH, MIN_STOCK_CAPITAL, PRINCIPAL_RATIO, 
    COMMON_PARAMS, STOP_LOSS_PARAMS, 
    STRATEGY_CONFIGS,               # [수정] 통합 설정 임포트
    LIVE_HMM_MODEL_NAME
) 

logger = logging.getLogger(__name__)

class HMMTrading:
    def __init__(self, api_client: CreonAPIClient, manager: TradingManager, notifier: Notifier, initial_cash: float):
        self.api_client = api_client
        self.notifier = notifier

        self.manager = manager
        self.broker = Brokerage(self.api_client, self.manager, self.notifier, initial_cash=initial_cash)
        
        self.capital_manager: Optional[CapitalManager] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.principal_ratio = PRINCIPAL_RATIO # HMM 모드에서 동적으로 덮어쓸 값

        self.daily_strategies: List[DailyStrategy] = []
        self.minute_strategy: Optional[MinuteStrategy] = None
        
        self.data_store = {'daily': {}, 'minute': {}}
        
        self.is_running = True
        self.market_open_time = datetime.strptime(MARKET_OPEN_TIME, '%H:%M:%S').time()
        self.market_close_time = datetime.strptime(MARKET_CLOSE_TIME, '%H:%M:%S').time()
        self.last_daily_run_time = None
        self._last_update_log_time: Dict[str, float] = {}
        self._last_cumulative_volume: Dict[str, int] = {}
        
        self.api_client.set_conclusion_callback(self.broker.handle_order_conclusion)
        
        logger.info("Trading 시스템 초기화 완료.")

    def set_strategies(self, daily_strategies: List[DailyStrategy], minute_strategy: MinuteStrategy, stop_loss_params: dict = None):
        self.daily_strategies = daily_strategies
        self.minute_strategy = minute_strategy
        if stop_loss_params:
            self.broker.set_stop_loss_params(stop_loss_params)
        daily_strategy_names = ', '.join([s.__class__.__name__ for s in self.daily_strategies])
        logger.info(f"전략 설정 완료: Daily(s)='{daily_strategy_names}', Minute='{self.minute_strategy.__class__.__name__}'")

    def set_broker_stop_loss_params(self, params: dict = None):
        self.broker.set_stop_loss_params(params)
        logging.info("브로커 손절매 파라미터 설정 완료.")

    def handle_price_update(self, stock_code: str, current_price: float, volume: int, timestamp: float):
        self._update_realtime_data(stock_code, current_price, volume)

    def prepare_for_system(self) -> bool:
        trading_date = datetime.now().date()
        logger.info(f"--- {trading_date} 거래 준비 시작 ---")
        self.notifier.send_message(f"--- {trading_date} 거래 준비 시작 ---")

        self.broker.sync_account_status()
        logger.info("1. 증권사 계좌 상태 동기화 완료.")

        initial_universe_codes = self.manager.get_universe_codes()
        if not initial_universe_codes:
            logger.error("초기 유니버스 종목을 가져올 수 없습니다. 준비를 중단합니다.")
            return False
        logger.info(f"초기 유니버스 {len(initial_universe_codes)}개 종목 로드 완료.")

        logger.info(f"2. 유니버스 사전 필터링을 시작합니다 ('최소 투자금': {MIN_STOCK_CAPITAL:,.0f}원 기준).")
        initial_prices_data = self.manager.api_client.get_current_prices_bulk(initial_universe_codes)
        
        final_universe_codes = [code for code in initial_universe_codes if code.startswith('U')]
        
        for code in initial_universe_codes:
            if not code.startswith('A'): continue
            
            price_data = initial_prices_data.get(code)
            current_price = price_data.get('close', 0) if price_data else 0

            if 0 < current_price <= MIN_STOCK_CAPITAL:
                final_universe_codes.append(code)
            elif current_price > 0:
                logger.info(f"사전 필터링: [{code}] 제외 (현재가: {current_price:,.0f}원 > 최소 투자금)")
            else:
                logger.warning(f"사전 필터링: [{code}] 가격 정보를 가져올 수 없어 제외됩니다.")
        
        logger.info(f"사전 필터링 완료. 유니버스 종목 수: {len(initial_universe_codes)}개 -> {len(final_universe_codes)}개")

        current_positions = self.broker.get_current_positions().keys()
        
        required_codes_for_data = set(final_universe_codes) | set(current_positions)
        
        # 지수인텍스 코드 market_index_code 추가
        market_code = COMMON_PARAMS.get('market_index_code')
        required_codes_for_data.add(market_code)
        # 안전자산 코드 safe_asset_code 추가
        safe_asset_code = COMMON_PARAMS.get('safe_asset_code')
        required_codes_for_data.add(safe_asset_code)
        
        logger.info(f"3. 총 {len(required_codes_for_data)}개 종목에 대한 과거 데이터 로드를 시작합니다.")

        fetch_start_date = trading_date - timedelta(days=90)
        for code in required_codes_for_data:
            if code.startswith('U'):
                logger.info(f"일봉 데이터 로딩: 지수 코드({code})는 개별 종목 루프에서 건너뜁니다.")
                continue
            if not code.startswith('A'):
                logger.info(f"일봉 데이터 로딩: 비정상 코드({code})는 개별 종목 루프에서 건너뜁니다.")
                continue

            daily_df = self.manager.cache_daily_ohlcv(code, fetch_start_date, trading_date)
            if not daily_df.empty:
                self.data_store['daily'][code] = daily_df
            else:
                logger.warning(f"{code}의 일봉 데이터를 로드할 수 없습니다.")
        
        try:
            trading_date = datetime.now().date()
            pykrx_start_date = trading_date - timedelta(days=365 * 2)
            self.manager.prepare_pykrx_data_for_period(pykrx_start_date, trading_date)
            market_calendar_df = self.manager.fetch_market_calendar(trading_date - timedelta(days=10), trading_date)
            trading_days = market_calendar_df[market_calendar_df['is_holiday'] == 0]['date'].sort_values().tolist()

            N = 5
            start_fetch_date = trading_days[-N] if len(trading_days) >= N else trading_days[0]
            
            logger.info(f"분봉 데이터 따라잡기를 {start_fetch_date}부터 시작합니다. (최근 {N} 거래일)")
            for code in required_codes_for_data:
                if not code.startswith('A'):
                    logger.info(f"분봉 데이터 로딩: 비정상 코드({code})는 개별 종목 루프에서 건너뜁니다.")
                    continue

                minute_df = self.manager.cache_minute_ohlcv(code, start_fetch_date, trading_date)
                if not minute_df.empty:
                    self.data_store['minute'].setdefault(code, {})
                    for group_date, group_df in minute_df.groupby(minute_df.index.date):
                        self.data_store['minute'][code][group_date] = group_df
        except IndexError:
            logger.error("캘린더에서 직전 영업일을 확인할 수 없습니다. 분봉 데이터 따라잡기를 중단합니다.")
        except Exception as e:
            logger.error(f"분봉 데이터 따라잡기 중 오류 발생: {e}")

        logger.info("과거 데이터 로드 완료.")

        # self.broker._restore_positions_state(self.data_store)
        # logger.info("4. 보유 포지션 상태(최고가 등) 복원 완료.")
        
        logger.info(f"--- {trading_date} 모든 준비 완료. 장 시작 대기 ---")
        self.notifier.send_message(f"--- {trading_date} 모든 준비 완료. ---")
        return True
    
    def run(self) -> None:
        if not self.daily_strategies or not self.minute_strategy:
            logger.error("전략이 설정되지 않았습니다. 자동매매를 중단합니다.")
            return

        self.is_running = True
        self.notifier.send_message("🚀 장중 매매를 시작합니다!")
        
        last_heartbeat_time = pytime.time()
        while self.is_running:
            try:
                now = datetime.now()
                current_time = now.time()
                
                if pytime.time() - last_heartbeat_time > 30:
                    logger.info("❤️ [SYSTEM LIVE] 자동매매 시스템이 정상 동작 중입니다.")
                    last_heartbeat_time = pytime.time()
                
                if self.market_open_time <= current_time < self.market_close_time:
                    logger.info("="*50)
                    logger.info(f"[{now.strftime('%H:%M:%S')}] 장중 매매 루프 시작...")

                    if self.last_daily_run_time is None or (now - self.last_daily_run_time) >= timedelta(minutes=5):
                        logger.info("1. 모든 일일 전략 재실행 및 자금 재배분...")

                        codes_for_equity = set(self.data_store['daily'].keys()) | set(self.broker.get_current_positions().keys())
                        current_prices = self.manager.api_client.get_current_prices_bulk(list(codes_for_equity))
                        # --- ▼ [수정] HMM 모드(PortfolioManager)와 정적 모드(CapitalManager) 분기 처리 ▼ ---
                        if self.portfolio_manager:
                            # HMM 동적 모드
                            account_equity = self.portfolio_manager.get_account_equity(current_prices)
                            hmm_input_data = self.manager.get_market_data_for_hmm(start_date=(now.date() - timedelta(days=100)), end_date=now.date())
                            total_principal, regime_probs = self.portfolio_manager.get_total_principal(account_equity, hmm_input_data)
                            strategy_capitals = self.portfolio_manager.get_strategy_capitals(total_principal, regime_probs)
                        else:
                            # 정적 모드
                            account_equity = self.capital_manager.get_account_equity(current_prices)
                            total_principal = self.capital_manager.get_total_principal(account_equity, self.principal_ratio)
                            strategy_capitals = {
                                name: self.capital_manager.get_strategy_capital(name, total_principal)
                                for name in self.capital_manager.strategy_configs.keys()
                            }
                        # --- ▲ [수정] 종료 ▲ ---

                        signals_from_all = []
                        for strategy in self.daily_strategies:
                            strategy_name = strategy.strategy_name
                            
                            # 위에서 계산된 자본금을 할당
                            strategy_capital = strategy_capitals.get(strategy_name, 0)
                            logger.info(f"-> 전략 '{strategy_name}' 실행 (할당 자본: {strategy_capital:,.0f}원)")
                            
                            strategy.run_daily_logic(now.date(), strategy_capital)
                            signals_from_all.append(strategy.signals)
                        
                        final_signals = self._aggregate_signals(signals_from_all)
                        self.minute_strategy.update_signals(final_signals)
                        self.last_daily_run_time = now
                        
                        logger.info(f"-> 일일 전략 실행 및 신호 통합 완료. 최종 {len(final_signals)}개 신호 생성/업데이트.")
                    
                    stocks_to_process = set(self.data_store['daily'].keys()) | set(self.broker.get_current_positions().keys()) | set(self.minute_strategy.signals.keys())
                    logger.info(f"2. 처리 대상 종목 통합 완료: 총 {len(stocks_to_process)}개")
                    
                    if stocks_to_process:
                        codes_to_poll_stocks = [code for code in stocks_to_process]
                        # 지수인텍스 코드 market_index_code 추가
                        market_code = COMMON_PARAMS.get('market_index_code')
                        codes_to_poll_stocks.append(market_code)
                        # 안전자산 코드 safe_asset_code 추가
                        safe_asset_code = COMMON_PARAMS.get('safe_asset_code')
                        codes_to_poll_stocks.append(safe_asset_code)
                        
                        if codes_to_poll_stocks:
                            logger.info(f"종목 {len(codes_to_poll_stocks)}개 실시간 데이터 폴링...")
                            latest_stock_data = self.manager.api_client.get_current_prices_bulk(codes_to_poll_stocks)
                            for code, data in latest_stock_data.items():
                                self._update_data_store_from_poll(code, data)

                        logger.info("-> 데이터 폴링 및 업데이트 완료.")
                    
                    logger.info("4. 개별 종목 분봉 전략 및 리스크 관리 시작...")
                    for stock_code in list(stocks_to_process):
                        self._ensure_minute_data_exists(stock_code, now.date())
                        self.minute_strategy.run_minute_logic(now, stock_code)
                    
                    owned_codes = list(self.broker.get_current_positions().keys())
                    current_prices_for_positions = self.manager.api_client.get_current_prices_bulk(owned_codes)
                    self.broker.check_and_execute_stop_loss(current_prices_for_positions, now)
                    logger.info("-> 분봉 전략 및 리스크 관리 완료.")
                    
                    logger.info(f"루프 1회 실행 완료. 20초 후 다음 루프를 시작합니다.")
                    logger.info("="*50 + "\n")
                    pytime.sleep(20)
                
                elif current_time >= self.market_close_time:
                    logger.info("장 마감. 오늘의 모든 거래를 종료합니다.")
                    self.record_daily_performance(now.date())
                    self.stop_trading()
                
                else:
                    logger.info(f"장 시작({self.market_open_time.strftime('%H:%M')}) 대기 중...")
                    pytime.sleep(20)

            except KeyboardInterrupt:
                logger.info("사용자에 의해 시스템 종료 요청됨.")
                self.is_running = False
            except Exception as e:
                logger.error(f"매매 루프에서 예외 발생: {e}", exc_info=True)
                self.notifier.send_message(f"🚨 시스템 오류 발생: {e}")
                pytime.sleep(60)

    def stop_trading(self) -> None:
        self.is_running = False
        logger.info("자동매매 시스템 종료 요청 수신.")

    def cleanup(self) -> None:
        logger.info("Trading 시스템 cleanup 시작.")
        if self.broker:
            self.broker.cleanup()
        if self.api_client:
            self.api_client.cleanup()
        if self.manager:
            self.manager.close()
        logger.info("Trading 시스템 cleanup 완료.")

    def _update_data_store_from_poll(self, stock_code: str, market_data: Dict[str, Any]):
        api_time_str = market_data.get('time')
        if api_time_str is None:
            now = datetime.now()
        else:
            api_time = int(api_time_str)
            hour = api_time // 100
            minute = api_time % 100
            now = datetime.now().replace(hour=hour, minute=minute)

        current_minute = now.replace(second=0, microsecond=0)
        today = now.date()
        today_ts = pd.Timestamp(today)
        
        MINUTE_DF_COLUMNS = ['stock_code', 'open', 'high', 'low', 'close', 'volume', 'change_rate', 'trading_value']

        if stock_code in self.data_store['daily']:
            ohlcv_data = {k: v for k, v in market_data.items() if k in ['open', 'high', 'low', 'close', 'volume']}
            
            if len(ohlcv_data) == 5:

                # change_rate 및 trading_value 계산 로직 
                daily_df = self.data_store['daily'][stock_code]
                
                # 1. 등락률(change_rate) 계산
                change_rate = 0.0
                # DataFrame에 데이터가 최소 1줄 이상 있어야 전일 종가 조회 가능
                if not daily_df.empty:
                    # 마지막 행이 전일 데이터
                    yesterday_close = daily_df['close'].iloc[-1]
                    today_close = ohlcv_data['close']
                    
                    if yesterday_close > 0:
                        change_rate = ((today_close - yesterday_close) / yesterday_close) * 100
                
                # 2. 거래대금(trading_value) 계산
                trading_value = ohlcv_data['close'] * ohlcv_data['volume']

                # 데이터프레임의 모든 컬럼에 맞는 데이터를 딕셔너리 형태로 준비합니다.
                # 누락된 컬럼(change_rate, trading_value 등)에 기본값을 채워줍니다.
                new_row_data = {
                    'open': ohlcv_data['open'],
                    'high': ohlcv_data['high'],
                    'low': ohlcv_data['low'],
                    'close': ohlcv_data['close'],
                    'volume': ohlcv_data['volume'],
                    'stock_code': stock_code,
                    # [수정] 계산된 값으로 대체
                    'change_rate': change_rate,
                    'trading_value': trading_value
                }               
                
                self.data_store['daily'][stock_code].loc[today_ts] = new_row_data

        stock_minute_data = self.data_store['minute'].setdefault(stock_code, {})
        if today not in stock_minute_data:
            stock_minute_data[today] = pd.DataFrame(columns=MINUTE_DF_COLUMNS).set_index(pd.to_datetime([]))

        minute_df = stock_minute_data[today]
        
        current_price = market_data['close']
        cumulative_volume = market_data['volume']
        last_known_volume = self._last_cumulative_volume.get(stock_code, 0)
        
        minute_volume = cumulative_volume - last_known_volume
        if minute_volume < 0:
            minute_volume = cumulative_volume 

        if current_minute in minute_df.index:
            minute_df.loc[current_minute, 'high'] = max(minute_df.loc[current_minute, 'high'], current_price)
            minute_df.loc[current_minute, 'low'] = min(minute_df.loc[current_minute, 'low'], current_price)
            minute_df.loc[current_minute, 'close'] = current_price
            minute_df.loc[current_minute, 'volume'] += minute_volume
        else:
            new_row_data = [stock_code, current_price, current_price, current_price, current_price, minute_volume, 0.0, 0.0]
            new_row = pd.DataFrame([new_row_data], columns=MINUTE_DF_COLUMNS, index=[current_minute])
            new_row.index = pd.to_datetime(new_row.index)
            stock_minute_data[today] = pd.concat([minute_df, new_row])
        
        self._last_cumulative_volume[stock_code] = cumulative_volume

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

    def _ensure_minute_data_exists(self, stock_code: str, current_date: date):
        stock_minute_data = self.data_store['minute'].get(stock_code, {})
        if current_date not in stock_minute_data:
            logger.info(f"[{stock_code}] 종목의 당일 분봉 데이터가 없어 따라잡기를 실행합니다.")
            try:
                market_calendar_df = self.manager.fetch_market_calendar(current_date - timedelta(days=10), current_date)
                trading_days = market_calendar_df[market_calendar_df['is_holiday'] == 0]['date'].dt.date.sort_values().tolist()
                prev_trading_date = trading_days[-2] if len(trading_days) > 1 else current_date - timedelta(days=1)
                minute_df = self.manager.cache_minute_ohlcv(stock_code, prev_trading_date, current_date)
                if not minute_df.empty:
                    self.data_store['minute'].setdefault(stock_code, {})
                    for group_date, group_df in minute_df.groupby(minute_df.index.date):
                        self.data_store['minute'][stock_code][group_date] = group_df
                    logger.info(f"[{stock_code}] 분봉 데이터 따라잡기 완료.")
                else:
                    logger.warning(f"[{stock_code}] 분봉 데이터 따라잡기를 시도했으나 데이터를 가져오지 못했습니다.")
            except Exception as e:
                logger.error(f"[{stock_code}] 분봉 데이터 따라잡기 중 오류 발생: {e}")

    def record_daily_performance(self, current_date: date):
        """
        [최종 수정본] 장 마감 후, DB 데이터를 기반으로 누적 성과를 집계하여
        trading_run, trading_performance 테이블을 업데이트합니다.
        """
        logger.info(f"--- {current_date} 자동매매 결과 집계 및 저장 시작 ---")
        try:
            # 1. 현재 운영 모델 ID 조회
            model_info = self.manager.db_manager.fetch_hmm_model_by_name(LIVE_HMM_MODEL_NAME)
            if not model_info:
                logger.error(f"운영 모델({LIVE_HMM_MODEL_NAME}) 정보를 찾을 수 없어 결과 저장을 중단합니다.")
                self.notifier.send_message(f"🚨 중요: 운영 모델({LIVE_HMM_MODEL_NAME})을 DB에서 찾을 수 없습니다!")
                return
            model_id = model_info['model_id']

            # ▼▼▼ [수정] 누적 성과 계산을 위한 로직 변경 ▼▼▼
            # 2. 기존 누적 'run' 정보 조회
            existing_run_df = self.manager.db_manager.fetch_trading_run(model_id=model_id)

            # 3. 자본금 계산
            current_prices = self.api_client.get_current_prices_bulk(list(self.broker.get_current_positions().keys()))
            final_capital = self.broker.get_portfolio_value(current_prices)

            # 4. 시작일, 최초/일일 투자금 결정
            if not existing_run_df.empty:
                # 기존 기록이 있는 경우: 최초 투자금과 시작일은 기존 값을 사용
                existing_run = existing_run_df.iloc[0]
                initial_capital_for_run = float(existing_run['initial_capital'])
                start_date_for_run = existing_run['start_date']
                # 어제의 최종 자본을 오늘의 시작 자본으로 사용
                daily_initial_capital = float(existing_run['final_capital'])
            else:
                # 최초 실행인 경우: 모든 값을 새로 설정
                initial_capital_for_run = self.broker.initial_cash
                start_date_for_run = current_date
                daily_initial_capital = self.broker.initial_cash

            # 5. 일일 및 누적 성과 지표 계산
            daily_profit_loss = final_capital - daily_initial_capital
            daily_return = daily_profit_loss / daily_initial_capital if daily_initial_capital > 0 else 0.0
            
            # 누적 손익 및 수익률은 '최초 투자금' 대비 '현재 최종 자본'으로 계산
            total_profit_loss_cumulative = final_capital - initial_capital_for_run
            cumulative_return = total_profit_loss_cumulative / initial_capital_for_run if initial_capital_for_run > 0 else 0.0

            # MDD 계산 (전체 자산 곡선 기준)
            performance_history_df = self.manager.db_manager.fetch_trading_performance(model_id=model_id, end_date=current_date)
            equity_curve = pd.Series(dtype=float)
            if not performance_history_df.empty:
                # DB에서 조회한 과거 데이터로 Series 생성
                equity_curve = performance_history_df.set_index('date')['end_capital']
            # 오늘의 최종 자본을 자산 곡선에 추가
            equity_curve[pd.Timestamp(current_date).date()] = final_capital
            
            metrics = calculate_performance_metrics(equity_curve)
            max_drawdown = metrics.get('mdd', 0.0)
            
            # 6. 사용된 전략 정보 요약
            daily_strategy_names = ', '.join([s.__class__.__name__ for s in self.daily_strategies])
            daily_strategy_params_json = json.dumps({s.__class__.__name__: s.strategy_params for s in self.daily_strategies})

            # 7. trading_run 테이블에 저장할 '누적' 데이터 구성
            run_data = {
                'model_id': model_id,
                'start_date': start_date_for_run,       # 최초 시작일
                'end_date': current_date,               # 최종 거래일 (오늘)
                'initial_capital': initial_capital_for_run, # 최초 투자금
                'final_capital': final_capital,         # 현재 최종 자본
                'total_profit_loss': total_profit_loss_cumulative, # 누적 손익
                'cumulative_return': cumulative_return, # 누적 수익률
                'max_drawdown': max_drawdown,
                'strategy_daily': daily_strategy_names,
                'params_json_daily': daily_strategy_params_json,
                'trading_date': current_date # save_trading_run 내부에서 start/end date 설정에 사용
            }
            # save_trading_run은 내부적으로 start_date를 업데이트하지 않음
            self.manager.db_manager.save_trading_run(run_data)

            # 8. trading_performance 테이블에 저장할 '일일' 데이터 구성
            performance_data = {
                'model_id': model_id,
                'date': current_date,
                'end_capital': final_capital,
                'daily_return': daily_return,
                'daily_profit_loss': daily_profit_loss,
                'cumulative_return': cumulative_return, # 그날까지의 누적 수익률
                'drawdown': max_drawdown # 그날까지의 MDD
            }
            # ▲▲▲ 수정 완료 ▲▲▲
            self.manager.db_manager.save_trading_performance(performance_data)

            logger.info(f"--- {current_date} 자동매매 결과 저장 완료 ---")
            self.notifier.send_message(
                f"📈 {current_date} 장 마감\n"
                f" - 최종 자산: {final_capital:,.0f}원\n"
                f" - 당일 손익: {daily_profit_loss:,.0f}원 ({daily_return:.2%})\n"
                f" - 누적 수익률: {cumulative_return:.2%}\n"
                f" - MDD: {max_drawdown:.2%}"
            )

        except Exception as e:
            logger.error(f"일일 성과 기록 중 오류 발생: {e}", exc_info=True)
            self.notifier.send_message("🚨 일일 성과 기록 중 심각한 오류가 발생했습니다.")

if __name__ == "__main__":

    # 설정 파일 로드
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])

    sma_daily_logger = logging.getLogger('strategies.sma_daily')
    sma_daily_logger.setLevel(logging.DEBUG)
    try:
        api_client = CreonAPIClient()
        db_manager = DBManager()
        notifier = Notifier(telegram_token=TELEGRAM_BOT_TOKEN, telegram_chat_id=TELEGRAM_CHAT_ID)
        trading_manager = TradingManager(api_client, db_manager)

        trading_system = HMMTrading(
            api_client=api_client,
            manager=trading_manager,
            notifier=notifier,
            initial_cash=INITIAL_CASH
        )
        
        daily_strategies_to_run = []
        
        # --- ▼ [수정] HMM 작전 계획에 따른 Manager 분기 처리 ▼ ---
        today_str = datetime.now().strftime('%Y%m%d')
        file_name = f"{today_str}_directive.json"
        directive_path = os.path.join(project_root, 'trading', file_name)
        if os.path.exists(directive_path):
            logger.info(f"오늘의 HMM 작전 계획 파일('{file_name}') 발견. HMM 동적 모드로 실행.")
            with open(directive_path, 'r', encoding='utf-8') as f:
                directive = json.load(f)

            # HMM 두뇌를 생성하여 필요한 구성요소(추론기, 정책, 프로파일)를 가져옴
            brain = HMMBrain(db_manager, trading_manager)

            # PortfolioManager를 생성하고 trading_system에 장착
            # 1. DB에서 프로파일 데이터를 DataFrame으로 불러옵니다.
            all_profiles_df = trading_manager.fetch_strategy_profiles_by_model(brain.model_id)

            # 2. PortfolioManager가 사용할 이중 딕셔너리 형태로 가공합니다.
            profiles_dict = {}
            if not all_profiles_df.empty:
                for strategy_name, group in all_profiles_df.groupby('strategy_name'):
                    # 각 전략별로, regime_id를 키로 하는 딕셔너리를 생성
                    profiles_dict[strategy_name] = {
                        row['regime_id']: row.to_dict() for _, row in group.iterrows()
                    }

            # 3. PortfolioManager를 생성하고 trading_system에 장착
            trading_system.portfolio_manager = PortfolioManager(
                broker=trading_system.broker,
                portfolio_configs=directive['portfolio'],
                inference_service=brain.inference_service,
                policy_map=brain.policy_map,
                strategy_profiles=profiles_dict, # 가공된 딕셔너리를 전달
                transition_matrix=brain.inference_service.hmm_model.model.transmat_
            )
            
            # 작전 파일에 명시된 전략과 파라미터 로드
            for item in directive['portfolio']:
                strategy_class = globals().get(item['name'])
                if strategy_class:
                    strategy_instance = strategy_class(
                        broker=trading_system.broker,
                        data_store=trading_system.data_store
                    )
                    # 1. settings.py에서 기본 파라미터를 먼저 로드합니다.
                    default_params = STRATEGY_CONFIGS.get(item['name'], {}).get('default_params', {}).copy()
                    # 2. directive.json의 파라미터로 덮어쓰기(업데이트) 합니다.
                    default_params.update(item['params'])
                    # 3. 최종 병합된 파라미터를 할당합니다.
                    strategy_instance.strategy_params = default_params
                    daily_strategies_to_run.append(strategy_instance)
        else:
            logger.info(f"오늘의 HMM 작전 계획 파일 없음. 설정 파일(settings.py) 기반의 정적 모드로 실행.")
            # CapitalManager를 생성하고 trading_system에 장착
            trading_system.capital_manager = CapitalManager(trading_system.broker)
            # settings.py에 정의된 활성 전략을 동적으로 로드
            for name, config in STRATEGY_CONFIGS.items():
                if config.get("strategy_status") is True:
                    # globals()를 사용해 문자열 이름으로 클래스 객체를 동적으로 찾음
                    strategy_class = globals().get(name)
                    if strategy_class:
                        instance = strategy_class(
                            broker=trading_system.broker,
                            data_store=trading_system.data_store
                        )
                        daily_strategies_to_run.append(instance)
                    else:
                        logger.warning(f"전략 클래스 '{name}'를 찾을 수 없습니다. 임포트되었는지 확인하세요.")            
 

        # --- (이하 공통 실행 로직은 동일) ---
        minute_strategy = TargetPriceMinute(trading_system.broker, trading_system.data_store)
        
        trading_system.set_strategies(
            daily_strategies=daily_strategies_to_run,
            minute_strategy=minute_strategy
        )
        trading_system.set_broker_stop_loss_params(STOP_LOSS_PARAMS)
        
        if trading_system.prepare_for_system():
            pythoncom.CoInitialize()
            try:
                logger.info("=== 자동매매 시작 ===")
                trading_system.run()
            except KeyboardInterrupt:
                logger.info("사용자에 의해 시스템 종료 요청됨.")
            finally:
                trading_system.cleanup()
                pythoncom.CoUninitialize()
                logger.info("시스템 종료 완료.")

    except Exception as e:
        logger.error(f"Backtest 테스트 중 오류 발생: {e}", exc_info=True)