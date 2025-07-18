# trading/trading.py
import pandas as pd
import logging
from datetime import datetime, date, timedelta, time
import time as pytime
from typing import Dict, Any, List, Optional
import sys
import os
import pythoncom
# --- [수정] 필요한 모듈 임포트 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.abstract_broker import AbstractBroker
from trading.brokerage import Brokerage
from trading.abstract_report import ReportGenerator, TradingDB # Live 용 Storage 클래스 이름 사용

from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.trading_manager import TradingManager
from strategies.strategy import DailyStrategy, MinuteStrategy
from util.notifier import Notifier

logger = logging.getLogger(__name__) # <-- 이 코드가 있어야 합니다.

class Trading:
    """
    자동매매 시스템의 메인 오케스트레이터 클래스입니다.
    """
    def __init__(self, api_client: CreonAPIClient, db_manager: DBManager, notifier: Notifier, initial_cash: float):
        self.api_client = api_client
        self.db_manager = db_manager
        self.notifier = notifier

        # --- [수정] __init__ 생성자 정리 ---
        # 1. Manager를 먼저 생성합니다.
        self.manager = TradingManager(self.api_client, self.db_manager)
        # 2. Brokerage 생성 시 Manager를 주입하고, 타입 힌트는 AbstractBroker를 사용합니다.
        self.broker: AbstractBroker = Brokerage(self.api_client, self.manager, self.notifier, initial_cash=initial_cash)
        
        self.daily_strategy: Optional[DailyStrategy] = None
        self.minute_strategy: Optional[MinuteStrategy] = None
        
        self.data_store = {'daily': {}, 'minute': {}}
        
        self.is_running = True
        self.market_open_time = time(16, 0)
        self.market_close_time = time(18, 30)
        
        self.last_daily_run_time = None                   # 일일 전략의 마지막 실행 시간을 추적하기 위한 변수
        self._last_update_log_time: Dict[str, float] = {} # 실시간 데이터 업데이트 로그 출력을 제어하기 위한 변수
        self._last_cumulative_volume: Dict[str, int] = {} # 분봉 거래량 계산을 위해 마지막 누적 거래량을 저장하는 변수
        # Creon API의 실시간 체결 콜백을 Brokerage의 핸들러에 연결
        self.api_client.set_conclusion_callback(self.broker.handle_order_conclusion)
        #self.api_client.set_price_update_callback(self.handle_price_update) # 실시간 가격 콜백 핸들러 등록

        logger.info("Trading 시스템 초기화 완료.")

    def set_strategies(self, daily_strategy: DailyStrategy, minute_strategy: MinuteStrategy, stop_loss_params: dict = None):
        self.daily_strategy = daily_strategy
        self.minute_strategy = minute_strategy
        if stop_loss_params:
            self.broker.set_stop_loss_params(stop_loss_params)
        logger.info(f"전략 설정 완료: Daily='{daily_strategy.__class__.__name__}', Minute='{minute_strategy.__class__.__name__}'")

    def set_broker_stop_loss_params(self, params: dict = None):
        self.broker.set_stop_loss_params(params)
        logging.info("브로커 손절매 파라미터 설정 완료.")

    def handle_price_update(self, stock_code: str, current_price: float, volume: int, timestamp: float):
        """CreonAPIClient로부터 실시간 현재가 및 거래량 업데이트를 수신하는 콜백 함수."""
        self._update_realtime_data(stock_code, current_price, volume)


    def add_daily_data(self, stock_code: str, df: pd.DataFrame):
        """백테스트를 위한 일봉 데이터를 추가합니다."""
        if not df.empty:
            self.data_store['daily'][stock_code] = df
            logging.debug(f"일봉 데이터 추가: {stock_code}, {len(df)}행")
        else:
            logging.warning(f"빈 데이터프레임이므로 {stock_code}의 일봉 데이터를 추가하지 않습니다.")

    def prepare_for_trading(self) -> bool:
        """
        [신규] 시스템 시작 시, 거래에 필요한 모든 상태를 준비하고 복원합니다.
        성공 시 True, 실패 시 False를 반환합니다.
        """
        trading_date = datetime.now().date()
        logger.info(f"--- {trading_date} 거래 준비 시작 ---")
        self.notifier.send_message(f"--- {trading_date} 거래 준비 시작 ---")

        # --- 1. 초기 유니버스 종목 코드 로드 ---
        initial_universe_codes = self.manager.get_universe_codes()
        if not initial_universe_codes:
            logger.error("초기 유니버스 종목을 가져올 수 없습니다. 준비를 중단합니다.")
            return False
        
        # --- 2. 증권사 계좌 상태 동기화 (가용 현금 확보) ---
        self.broker.sync_account_status()
        logger.info("2. 증권사 계좌 상태 동기화 완료.")

        # --- [신규] 3. 유니버스 사전 필터링 (가격 기준) ---
        logger.info("3. 유니버스 사전 필터링을 시작합니다 (가격 기준).")
        available_cash = self.broker.get_current_cash_balance()
        num_top_stocks = self.daily_strategy.strategy_params['num_top_stocks']
        investment_per_stock = available_cash / num_top_stocks if num_top_stocks > 0 else 0

        # 필터링을 위해 초기 유니버스 전체의 현재가 정보를 가져옵니다.
        initial_prices_data = self.manager.api_client.get_current_prices_bulk(initial_universe_codes)
        
        final_universe_codes = []
        for code in initial_universe_codes:
            price_data = initial_prices_data.get(code)

            # [수정] 'price' 키를 'close' 키로 변경
            if price_data and price_data.get('close', 0) > 0:
                # 할당액이 최소 1주 가격보다 큰 종목만 최종 유니버스에 포함
                if investment_per_stock > price_data['close']:
                    final_universe_codes.append(code)
                else:
                    logger.info(f"사전 필터링: [{code}] 제외 (가격: {price_data['close']:,.0f}원 > 할당액: {investment_per_stock:,.0f}원)")
            elif code == 'U001':
                 final_universe_codes.append(code)
            else:
                 logger.warning(f"사전 필터링: [{code}] 가격 정보를 가져올 수 없어 제외됩니다.")

        logger.info(f"사전 필터링 완료. 유니버스 종목 수: {len(initial_universe_codes)}개 -> {len(final_universe_codes)}개")
        # --- 필터링 끝 ---
        # --- 4. 최종 유니버스에 대해서만 과거 데이터 로드 ---
        fetch_start_date = trading_date - timedelta(days=90)
        for code in final_universe_codes: # 'initial_universe_codes' 대신 'final_universe_codes' 사용
            daily_df = self.manager.cache_daily_ohlcv(code, fetch_start_date, trading_date)
            if not daily_df.empty:
                self.data_store['daily'][code] = daily_df
        logger.info("4. 최종 유니버스 데이터 로드 완료.")

        # --- [신규] 5. 오늘 날짜의 누락된 분봉 데이터 '따라잡기' ---
        logger.info("5. 누락된 분봉 데이터 따라잡기를 시작합니다.")
        trading_date = datetime.now().date()
        for code in final_universe_codes:
            if code.startswith('U'): continue # 지수 데이터는 분봉 조회에서 제외
            
            # manager를 통해 오늘 날짜의 모든 과거 분봉을 가져와 data_store에 채워넣음
            self.manager.cache_minute_ohlcv(code, trading_date, trading_date)
        logger.info("5. 누락된 분봉 데이터 따라잡기 완료.")
        # --- 따라잡기 끝 ---
        # --- 5. 증권사 포지션 정보 복원 ---
        self.broker.sync_account_status()
        logger.info("2. 증권사 계좌 상태 동기화 완료.")

        # --- 6. 비(非)유니버스 보유 종목 데이터 추가 복원 ---
        current_positions = self.broker.get_current_positions().keys()
        for stock_code in current_positions:
            if stock_code not in self.data_store['daily']:
                logger.info(f"비-유니버스 보유 종목 {stock_code} 데이터 추가 로드.")
                daily_df = self.manager.cache_daily_ohlcv(stock_code, fetch_start_date, trading_date)
                self.data_store['daily'][stock_code] = daily_df
        logger.info("3. 비-유니버스 보유 종목 데이터 준비 완료.")
        
        # --- 3.5. 포지션 상태(highest_price) 복원 ---
        self.broker._restore_positions_state(self.data_store)
        logger.info("3.5. 보유 포지션 상태(최고가 등) 복원 완료.")

        # --- 4. 일일 전략 실행 및 신호 생성 ---
        self.daily_strategy.run_daily_logic(trading_date)
        self.minute_strategy.update_signals(self.daily_strategy.signals)
        logger.info("4. 일일 전략 실행 및 신호 생성 완료.")

        logger.info(f"--- {trading_date} 모든 준비 완료. 장 시작 대기 ---")
        return True

    def run(self) -> None:
        """
        [수정] 장중 실시간 매매 루프만 담당합니다.
        """
        if not self.daily_strategy or not self.minute_strategy:
            logger.error("전략이 설정되지 않았습니다. 자동매매를 중단합니다.")
            return

        self.is_running = True
        self.notifier.send_message("🚀 장중 매매를 시작합니다!")
        
        last_heartbeat_time = pytime.time() # [추가] 시스템 상태 모니터링을 위한 변수
        while self.is_running:
            try:
                now = datetime.now()
                current_time = now.time()
                # --- [추가] 30초마다 시스템 Heartbeat 로그 출력 ---
                if pytime.time() - last_heartbeat_time > 30:
                    logger.info("❤️ [SYSTEM LIVE] 자동매매 시스템이 정상 동작 중입니다.")
                    last_heartbeat_time = pytime.time()
                # --- 추가 끝 ---
                if self.market_open_time <= current_time < self.market_close_time:
                    # --- [핵심] 1. 폴링으로 모든 종목의 현재가/거래량 일괄 업데이트 ---
                    stocks_to_monitor = set(self.data_store['daily'].keys()) | set(self.broker.get_current_positions().keys())
                    if stocks_to_monitor:
                        latest_market_data = self.manager.api_client.get_current_prices_bulk(list(stocks_to_monitor))
                        for code, data in latest_market_data.items():
                            self._update_data_store_from_poll(code, data)
                    # --- 폴링 끝 ---
                    # --- [추가] 5분마다 일일 전략 재실행 ---
                    if self.last_daily_run_time is None or (now - self.last_daily_run_time) >= timedelta(minutes=1):
                        logger.info(f"[{now.strftime('%H:%M:%S')}] 5분 주기로 일일 전략을 다시 실행합니다.")
                        # 4. 일일 전략 실행 및 신호 생성
                        self.daily_strategy.run_daily_logic(now.date())
                        self.minute_strategy.update_signals(self.daily_strategy.signals)
                        
                        self.last_daily_run_time = now # 마지막 실행 시간 업데이트
                    # --- [추가 끝] ---
                    
                    # 분봉 전략 및 실시간 리스크 관리 실행
                    stocks_to_trade = set(self.broker.get_current_positions().keys()) | set(self.minute_strategy.signals.keys())
                    
                    for stock_code in stocks_to_trade:
                        self.minute_strategy.run_minute_logic(now, stock_code)
                    
                    # 리스크 관리 (손절/익절)
                    current_prices = self.manager.get_current_prices(list(self.broker.get_current_positions().keys()))
                    self.broker.check_and_execute_stop_loss(current_prices, now)

                    pytime.sleep(10) # 10초마다 반복
                
                elif current_time >= self.market_close_time:
                    logger.info("장 마감. 오늘의 모든 거래를 종료합니다.")
                    # --- [핵심 추가] 장 마감 후 성과 기록 ---
                    self.record_daily_performance(now.date())
                    # --- 추가 끝 ---
                    self.stop_trading() # 루프 종료
                
                else: # 장 시작 전
                    logger.info(f"장 시작({self.market_open_time.strftime('%H:%M')}) 대기 중...")

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
        if self.db_manager:
            self.db_manager.close()
        logger.info("Trading 시스템 cleanup 완료.")


    def get_current_market_prices(self, stock_codes: List[str]) -> Dict[str, float]:
        """
        현재 시점의 시장 가격을 가져옵니다. 백테스트 시뮬레이션에서는 data_store에서 가져옵니다.
        """
        prices = {}
        for code in stock_codes:
            # 가장 최신 분봉의 종가를 현재 가격으로 간주
            if code in self.data_store['minute'] and self.data_store['minute'][code]:
                # 마지막 날짜의 마지막 분봉 데이터를 가져옴
                latest_date = max(self.data_store['minute'][code].keys())
                latest_minute_df = self.data_store['minute'][code][latest_date]
                if not latest_minute_df.empty:
                    prices[code] = latest_minute_df.iloc[-1]['close']
                else:
                    logging.warning(f"종목 {code}의 분봉 데이터에 유효한 가격이 없습니다.")
            else:
                logging.warning(f"종목 {code}의 분봉 데이터가 data_store에 없습니다. 가격을 0으로 설정합니다.")
                prices[code] = 0.0 # 데이터가 없을 경우 0으로 처리하거나 에러 처리
        return prices

    def _update_data_store_from_poll(self, stock_code: str, ohlcv_data: Dict[str, Any]):
        """[수정] 폴링 데이터로 현재 분봉과 일봉을 실시간 업데이트합니다."""
        now = datetime.now()
        current_minute = now.replace(second=0, microsecond=0)
        today = now.date()

        # --- 1. 실시간 일봉 데이터 직접 업데이트 ---
        if stock_code in self.data_store['daily']:
            self.data_store['daily'][stock_code].loc[pd.Timestamp(today)] = ohlcv_data

        # --- 2. 현재 진행 중인 마지막 분봉 업데이트 ---
        today_minute_df = self.data_store['minute'].get(stock_code, {}).get(today)
        if today_minute_df is None: return # 분봉 데이터가 없으면 스킵

        current_price = ohlcv_data['close']
        cumulative_volume = ohlcv_data['volume']

        last_known_volume = self._last_cumulative_volume.get(stock_code, 0)
        # 현재 1분 동안 발생한 거래량 계산
        minute_volume = cumulative_volume - last_known_volume
        if minute_volume < 0: minute_volume = cumulative_volume # 장 시작 등 초기화 경우
        
        if current_minute in today_minute_df.index: # 현재 분봉이 이미 존재하면 업데이트
            today_minute_df.loc[current_minute, 'high'] = max(today_minute_df.loc[current_minute, 'high'], current_price)
            today_minute_df.loc[current_minute, 'low'] = min(today_minute_df.loc[current_minute, 'low'], current_price)
            today_minute_df.loc[current_minute, 'close'] = current_price
            today_minute_df.loc[current_minute, 'volume'] += minute_volume
        else: # 새로운 분봉이면 생성
            new_row = {'open': current_price, 'high': current_price, 'low': current_price, 'close': current_price, 'volume': minute_volume}
            today_minute_df.loc[current_minute] = new_row
        
        self._last_cumulative_volume[stock_code] = cumulative_volume

    def _run_minute_strategy_and_realtime_checks(self, current_dt: datetime) -> None:
        """
        분봉 전략 및 실시간 데이터를 기반으로 매매 로직을 실행합니다.
        - 매수/매도 신호 처리
        - 보유 종목 손절매/익절매 체크
        """
        # 현재 활성화된 매수 신호들을 확인
        active_buy_signals = self.manager.load_daily_signals(current_dt.date(), is_executed=False, signal_type='BUY')

        # 모든 종목의 실시간 현재가 데이터를 업데이트 (TradingManager를 통해 CreonAPIClient 사용)
        # 이 함수는 TradingManager가 시장 데이터로부터 실시간으로 받아오거나, 필요시 조회하여 업데이트할 것임.
        # 실제로는 CreonAPIClient의 실시간 시세 구독을 통해 이루어짐.
        # 여기서는 편의상 TradingManager가 최신 현재가를 가져온다고 가정
        current_prices = self.get_current_market_prices(list(self.broker.get_current_positions().keys()) + \
                                                                        list(active_buy_signals.keys()))
        # 분봉 전략 실행
        if self.minute_strategy:
            for stock_code, signal_info in active_buy_signals.items():
                if signal_info['stock_code'] == stock_code and signal_info['is_executed'] == False: # 아직 체결되지 않은 매수 신호
                    try:
                        # 분봉 전략에 현재 시간과 종목 코드를 전달하여 매매 판단
                        self.minute_strategy.run_minute_logic(current_dt, stock_code)
                    except Exception as e:
                        logger.error(f"분봉 전략 '{self.minute_strategy.strategy_name}' 실행 중 오류 발생 ({stock_code}): {e}", exc_info=True)
                        self.notifier.send_message(f"❗ 분봉 전략 오류: {stock_code} - {e}")

        # 손절매/익절매 조건 체크 (보유 종목에 대해)
        self.broker.check_and_execute_stop_loss(current_prices, current_dt)
        
        # TODO: 미체결 주문 관리 로직 추가 (TradingManager에서 주기적으로 조회 및 갱신)
        # self.broker.get_unfilled_orders()를 통해 미체결 주문 상태를 확인하고,
        # 필요에 따라 정정/취소 로직을 호출할 수 있습니다.
        # 예: 특정 시간까지 미체결 시 취소 후 재주문 또는 타임컷 매도 등.

    def record_daily_performance(self, current_date: date):
        """[신규] 장 마감 후 ReportGenerator를 사용해 일일 성과를 기록하는 메서드"""
        
        # 1. 자동매매용 저장 전략 선택
        storage = TradingDB(self.db_manager)

        # 2. 리포트 생성기에 저장 전략 주입
        reporter = ReportGenerator(storage_strategy=storage)
        
        # 3. 리포트에 필요한 데이터 준비
        # 일일 포트폴리오 가치 시계열 데이터 (자동매매에서는 당일 종가 기준)
        end_value = self.broker.get_portfolio_value(self.manager.get_current_prices(list(self.broker.get_current_positions().keys())))
        
        # DB에서 전일자 포트폴리오 정보를 가져와 시계열 데이터 생성
        latest_portfolio = self.db_manager.fetch_latest_daily_portfolio()
        start_value = latest_portfolio.get('total_capital', self.broker.initial_cash) if latest_portfolio else self.broker.initial_cash
        
        portfolio_series = pd.Series([start_value, end_value], index=[pd.Timestamp(current_date - timedelta(days=1)), pd.Timestamp(current_date)])

        # 거래 로그는 DB에서 해당일의 로그를 다시 불러옴
        transaction_log = self.db_manager.fetch_trading_logs(current_date, current_date)

        # 4. 리포트 생성 및 저장
        reporter.generate(
            start_date=current_date,
            end_date=current_date,
            initial_cash=start_value,
            portfolio_value_series=portfolio_series,
            transaction_log=transaction_log.to_dict('records') if not transaction_log.empty else [],
            strategy_info={
                'strategy_daily': self.daily_strategy.__class__.__name__,
                'strategy_minute': self.minute_strategy.__class__.__name__,
                'params_json_daily': self.daily_strategy.strategy_params,
                'params_json_minute': self.minute_strategy.strategy_params
            },
            # LiveTradeDBStorage에 추가 정보 전달
            cash_balance=self.broker.get_current_cash_balance()
        )

    # def load_stocks(self, start_date, end_date):
    #     from config.sector_stocks import sector_stocks
    #     # 모든 종목 데이터 로딩: 하나의 리스트로 변환
    #     fetch_start = start_date - timedelta(days=30)
    #     stock_names = []
    #     for sector, stocks in sector_stocks.items():
    #         for stock_name, _ in stocks:
    #             stock_names.append(stock_name)

    #     all_target_stock_names = stock_names
    #     for name in all_target_stock_names:
    #         code = self.api_client.get_stock_code(name)
    #         if code:
    #             logging.info(f"'{name}' (코드: {code}) 종목 일봉 데이터 로딩 중... (기간: {fetch_start.strftime('%Y%m%d')} ~ {end_date.strftime('%Y%m%d')})")
    #             daily_df = self.manager.cache_daily_ohlcv(code, fetch_start, end_date)
                
    #             if daily_df.empty:
    #                 logging.warning(f"{name} ({code}) 종목의 일봉 데이터를 가져올 수 없습니다. 해당 종목을 건너뜁니다.")
    #                 continue
    #             logging.debug(f"{name} ({code}) 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
    #             self.add_daily_data(code, daily_df)
    #         else:
    #             logging.warning(f"'{name}' 종목의 코드를 찾을 수 없습니다. 해당 종목을 건너뜁니다.")


# TODO: 실제 사용 시 main 함수에서 Trading 객체를 생성하고 루프 시작
# 예시:
if __name__ == "__main__":
    from datetime import date, datetime
    # 설정 파일 로드
    from config.settings import (
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
        INITIAL_CASH,
        MARKET_OPEN_TIME, MARKET_CLOSE_TIME,
        DAILY_STRATEGY_RUN_TIME, PORTFOLIO_UPDATE_TIME,
        COMMON_PARAMS, SMA_DAILY_PARAMS, RSI_MINUTE_PARAMS, STOP_LOSS_PARAMS,
        LOG_LEVEL, LOG_FILE
    )   
    # 1. 전체 프로그램의 기본 로그 레벨을 INFO로 설정
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])

    # --- [추가] 특정 모듈(sma_daily)의 로그 레벨만 DEBUG로 설정 ---
    # sma_daily.py 파일 내의 로거를 이름으로 가져옵니다.
    sma_daily_logger = logging.getLogger('strategies.sma_daily')
    sma_daily_logger.setLevel(logging.DEBUG)
    # --- 추가 끝 ---
    try:
        # 1. 필요한 인스턴스들 생성
        api_client = CreonAPIClient()
        db_manager = DBManager()
        notifier = Notifier(telegram_token=TELEGRAM_BOT_TOKEN, telegram_chat_id=TELEGRAM_CHAT_ID)

        # 2. 자동매매 인스턴스 생성
        trading_system = Trading(
            api_client=api_client,
            db_manager=db_manager,
            notifier=notifier,
            initial_cash=INITIAL_CASH
        )
        
        # 3. 전략 설정
        # SMA 전략 설정
        from strategies.sma_daily import SMADaily
        daily_strategy = SMADaily(broker=trading_system.broker, data_store=trading_system.data_store, strategy_params=SMA_DAILY_PARAMS)

        # RSI 분봉 전략 설정
        # from strategies.rsi_minute import RSIMinute
        # minute_strategy = RSIMinute(broker=trading_system.broker, data_store=trading_system.data_store, strategy_params=RSI_MINUTE_PARAMS)        

        # 목표가 분봉 전략 설정 (최적화 결과 반영)
        from strategies.target_price_minute import TargetPriceMinute
        minute_strategy = TargetPriceMinute(broker=trading_system.broker, data_store=trading_system.data_store, strategy_params=COMMON_PARAMS)        

        # 일봉/분봉 전략 설정
        trading_system.set_strategies(daily_strategy=daily_strategy, minute_strategy=minute_strategy)
        # 손절매 파라미터 설정 (선택사항)
        trading_system.set_broker_stop_loss_params(STOP_LOSS_PARAMS)

        # 4. 일봉 데이터 로드
        end_date = date.today()
        start_date = end_date - timedelta(days=60)

        #trading_system.load_stocks(start_date, end_date)
        
        
        # 5. [수정] 거래 준비 단계 실행
        if trading_system.prepare_for_trading():
            # 6. [수정] 준비가 성공하면 매매 루프 실행
            # --- [추가] COM 환경 초기화 ---
            pythoncom.CoInitialize()

            try:
                logger.info("=== 자동매매 시작 ===")
                trading_system.run()
            except KeyboardInterrupt:
                logger.info("사용자에 의해 시스템 종료 요청됨.")
            finally:
                trading_system.cleanup()
                # --- [추가] COM 환경 정리 ---
                pythoncom.CoUninitialize()
                logger.info("시스템 종료 완료.")

    except Exception as e:
        logger.error(f"Backtest 테스트 중 오류 발생: {e}", exc_info=True)            