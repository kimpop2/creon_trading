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
        self.market_open_time = time(9, 0)
        self.market_close_time = time(15, 30)
        
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

    def prepare_for_system(self) -> bool:
        """
        [신규] 시스템 시작 시, 거래에 필요한 모든 상태를 준비하고 복원합니다.
        성공 시 True, 실패 시 False를 반환합니다.
        """
        trading_date = datetime.now().date()
        logger.info(f"--- {trading_date} 거래 준비 시작 ---")
        self.notifier.send_message(f"--- {trading_date} 거래 준비 시작 ---")

        # --- 1. 증권사 계좌 동기화 및 초기 유니버스 로드 ---
        # 가용 현금과 현재 보유 포지션을 파악하기 위해 계좌를 먼저 동기화합니다.
        self.broker.sync_account_status()
        logger.info("1. 증권사 계좌 상태 동기화 완료.")

        initial_universe_codes = self.manager.get_universe_codes()
        if not initial_universe_codes:
            logger.error("초기 유니버스 종목을 가져올 수 없습니다. 준비를 중단합니다.")
            return False
        logger.info(f"초기 유니버스 {len(initial_universe_codes)}개 종목 로드 완료.")

        # --- 2. 유니버스 사전 필터링 (가격 기준) ---
        # 할당된 자본으로 최소 1주도 매수할 수 없는 고가 종목을 제외합니다.
        logger.info("2. 유니버스 사전 필터링을 시작합니다 (가격 기준).")
        available_cash = self.broker.get_current_cash_balance()
        num_top_stocks = self.daily_strategy.strategy_params.get('num_top_stocks', 0)
        investment_per_stock = available_cash / num_top_stocks if num_top_stocks > 0 else 0

        if investment_per_stock > 0:
            initial_prices_data = self.manager.api_client.get_current_prices_bulk(initial_universe_codes)
            
            final_universe_codes = []
            for code in initial_universe_codes:
                # 코스피/코스닥 지수(U001)는 그대로 포함
                if code == 'U001':
                    final_universe_codes.append(code)
                    continue

                price_data = initial_prices_data.get(code)
                current_price = price_data.get('close', 0) if price_data else 0

                # 할당액이 최소 1주 가격보다 큰 종목만 최종 유니버스에 포함
                if current_price > 0 and investment_per_stock >= current_price:
                    final_universe_codes.append(code)
                elif current_price > 0:
                    logger.info(f"사전 필터링: [{code}] 제외 (가격: {current_price:,.0f}원 > 할당액: {investment_per_stock:,.0f}원)")
                else:
                    logger.warning(f"사전 필터링: [{code}] 가격 정보를 가져올 수 없어 제외됩니다.")
        else:
            # 종목당 투자금이 0이면 가격 필터링을 건너뜁니다.
            final_universe_codes = initial_universe_codes
            logger.warning("종목당 할당 투자금이 0원입니다. 가격 기반 필터링을 건너뜁니다.")
            
        logger.info(f"사전 필터링 완료. 유니버스 종목 수: {len(initial_universe_codes)}개 -> {len(final_universe_codes)}개")

        # --- 3. 필요 데이터 로드를 위한 최종 종목 리스트 통합 및 데이터 로드 ---
        # 필터링된 유니버스와 현재 보유 종목을 합쳐 필요한 모든 종목 리스트를 생성합니다.
        current_positions = self.broker.get_current_positions().keys()
        required_codes_for_data = set(final_universe_codes) | set(current_positions)
        logger.info(f"3. 총 {len(required_codes_for_data)}개 종목에 대한 과거 데이터 로드를 시작합니다.")

        # 일봉 데이터 로드
        fetch_start_date = trading_date - timedelta(days=90)
        for code in required_codes_for_data:
            daily_df = self.manager.cache_daily_ohlcv(code, fetch_start_date, trading_date)
            if not daily_df.empty:
                self.data_store['daily'][code] = daily_df
            else:
                logger.warning(f"{code}의 일봉 데이터를 로드할 수 없습니다.")
        
        # 누락된 분봉 데이터 '따라잡기'
        try:
            # 직전 영업일을 계산하기 위해 시장 캘린더를 조회합니다.
            market_calendar_df = self.db_manager.fetch_market_calendar(trading_date - timedelta(days=10), trading_date)
            trading_days = market_calendar_df[market_calendar_df['is_holiday'] == 0]['date'].dt.date.sort_values().tolist()

            # N일 평균 거래량 비교를 위해 필요한 만큼의 과거 데이터를 로드합니다.
            N = 5 # 5일 평균을 사용한다고 가정
            # N일 전 거래일을 계산합니다.
            start_fetch_date = trading_days[-N] if len(trading_days) >= N else trading_days[0]
            
            logger.info(f"분봉 데이터 따라잡기를 {start_fetch_date}부터 시작합니다. (최근 {N} 거래일)")
            for code in required_codes_for_data:
                
                # 수정된 시작 날짜를 사용하여 N일치의 데이터를 캐싱합니다.
                minute_df = self.manager.cache_minute_ohlcv(code, start_fetch_date, trading_date)
                
                # 불러온 데이터가 비어있지 않은 경우에만 처리
                if not minute_df.empty:
                    # 종목 코드에 해당하는 딕셔너리가 없으면 생성
                    self.data_store['minute'].setdefault(code, {})
                    
                    # 불러온 데이터를 날짜별로 그룹화하여 저장
                    for group_date, group_df in minute_df.groupby(minute_df.index.date):
                        self.data_store['minute'][code][group_date] = group_df
                # [수정 끝]

        except IndexError:
            logger.error("캘린더에서 직전 영업일을 확인할 수 없습니다. 분봉 데이터 따라잡기를 중단합니다.")
        except Exception as e:
            logger.error(f"분봉 데이터 따라잡기 중 오류 발생: {e}")

        logger.info("과거 데이터 로드 완료.")

        # --- 4. 포지션 상태 복원 및 트레이딩 신호 생성 ---
        # 현재 보유 중인 포지션의 상태(예: 매수 이후 최고가)를 복원합니다.
        self.broker._restore_positions_state(self.data_store)
        logger.info("4. 보유 포지션 상태(최고가 등) 복원 완료.")

        # 일일 전략을 실행하여 오늘의 매매 신호를 생성합니다.
        self.daily_strategy.run_daily_logic(trading_date)
        self.minute_strategy.update_signals(self.daily_strategy.signals)
        logger.info("일일 전략 실행 및 신호 생성 완료.")

        # --- 5. 준비 완료 ---
        logger.info(f"--- {trading_date} 모든 준비 완료. 장 시작 대기 ---")
        self.notifier.send_message(f"--- {trading_date} 모든 준비 완료. ---")
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
        
        last_heartbeat_time = pytime.time()
        while self.is_running:
            try:
                now = datetime.now()
                current_time = now.time()
                
                if pytime.time() - last_heartbeat_time > 30:
                    logger.info("❤️ [SYSTEM LIVE] 자동매매 시스템이 정상 동작 중입니다.")
                    last_heartbeat_time = pytime.time()
                
                if self.market_open_time <= current_time < self.market_close_time:
                    # [로그 추가] 루프의 시작을 명확히 표시
                    logger.info("="*50)
                    logger.info(f"[{now.strftime('%H:%M:%S')}] 장중 매매 루프 시작...")
                    
                    # 1. (필요시) 일일 전략을 다시 실행하여 최신 신호를 생성합니다.
                    if self.last_daily_run_time is None or (now - self.last_daily_run_time) >= timedelta(minutes=1):
                        logger.info("1. 일일 전략 재실행...")
                        self.daily_strategy.run_daily_logic(now.date())
                        self.minute_strategy.update_signals(self.daily_strategy.signals)
                        self.last_daily_run_time = now
                        # [로그 추가] 일일 전략 실행 결과
                        logger.info(f"-> 일일 전략 실행 완료. 총 {len(self.daily_strategy.signals)}개 신호 생성/업데이트.")
                    
                    # 2. 업데이트 및 전략 실행에 필요한 모든 종목 리스트를 통합합니다.
                    stocks_to_process = set(self.data_store['daily'].keys()) | set(self.broker.get_current_positions().keys()) | set(self.minute_strategy.signals.keys())
                    # [로그 추가] 처리 대상 종목 수
                    logger.info(f"2. 처리 대상 종목 통합 완료: 총 {len(stocks_to_process)}개")
                    
                    # 3. 통합된 리스트의 모든 종목에 대해 데이터를 폴링하고 업데이트합니다.
                    if stocks_to_process:
                        
                        # --- [핵심 수정] 시장 지수 조회와 일반 종목 조회를 분리 ---
                        market_index_code = self.daily_strategy.strategy_params.get('market_index_code')
                        
                        # 3-1. 일반 종목들만 먼저 폴링합니다.
                        codes_to_poll_stocks = [code for code in stocks_to_process if code != market_index_code]
                        if codes_to_poll_stocks:
                            logger.info(f"3-1. 일반 종목 {len(codes_to_poll_stocks)}개 실시간 데이터 폴링...")
                            latest_stock_data = self.manager.api_client.get_current_prices_bulk(codes_to_poll_stocks)
                            for code, data in latest_stock_data.items():
                                self._update_data_store_from_poll(code, data)

                        # 3-2. 시장 지수만 별도로 폴링하여 안정성을 높입니다.
                        if market_index_code:
                            logger.info(f"3-2. 시장 지수({market_index_code}) 데이터 폴링...")
                            latest_index_data = self.manager.api_client.get_current_prices_bulk([market_index_code])
                            if market_index_code in latest_index_data:
                                self._update_data_store_from_poll(market_index_code, latest_index_data[market_index_code])
                            else:
                                logger.warning(f"시장 지수({market_index_code}) 데이터 조회에 실패했습니다.")

                        logger.info("-> 데이터 폴링 및 업데이트 완료.")

                    # 4. 분봉 전략 및 리스크 관리를 실행합니다. (데이터가 최신 상태임이 보장됨)
                    logger.info("4. 개별 종목 분봉 전략 및 리스크 관리 시작...")
                    for stock_code in stocks_to_process:
                        self._ensure_minute_data_exists(stock_code, now.date())
                        self.minute_strategy.run_minute_logic(now, stock_code)
                    
                    # 리스크 관리 (손절/익절)
                    owned_codes = list(self.broker.get_current_positions().keys())
                    current_prices_for_positions = {code: latest_stock_data[code] for code in owned_codes if code in latest_stock_data}
                    self.broker.check_and_execute_stop_loss(current_prices_for_positions, now)
                    logger.info("-> 분봉 전략 및 리스크 관리 완료.")
                    
                    # [로그 추가] 루프의 끝과 대기 시간 안내
                    logger.info(f"루프 1회 실행 완료. 20초 후 다음 루프를 시작합니다.")
                    logger.info("="*50 + "\n")
                    pytime.sleep(20)
                
                elif current_time >= self.market_close_time:
                    logger.info("장 마감. 오늘의 모든 거래를 종료합니다.")
                    self.record_daily_performance(now.date())
                    self.stop_trading()
                
                else: # 장 시작 전
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


    def _update_data_store_from_poll(self, stock_code: str, market_data: Dict[str, Any]):
        """
        [수정] API에서 받은 시각 정보를 사용하여 분봉 데이터를 생성합니다.
        """
        api_time = market_data.get('time') # hhmm 형식 (예: 915)
        if api_time is None:
            logger.warning(f"[{stock_code}] 데이터에 시각 정보가 없어 시스템 시간으로 대체합니다.")
            now = datetime.now()
        else:
            hour = api_time // 100
            minute = api_time % 100
            now = datetime.now().replace(hour=hour, minute=minute)

        current_minute = now.replace(second=0, microsecond=0)
        today = now.date()
        today_ts = pd.Timestamp(today)
        # [수정] 다음 1분봉이 아닌, 현재 수신한 시각이 속한 분봉에 기록하도록 변경
        write_key = current_minute

        MINUTE_DF_COLUMNS = ['stock_code', 'open', 'high', 'low', 'close', 'volume', 'change_rate', 'trading_value']

        if stock_code in self.data_store['daily']:
            # [수정] market_data에서 OHLCV만 추출하여 일봉 데이터 업데이트
            ohlcv_data = {k: v for k, v in market_data.items() if k != 'time'}
            self.data_store['daily'][stock_code].loc[today_ts] = ohlcv_data

        stock_minute_data = self.data_store['minute'].setdefault(stock_code, {})

        if today not in stock_minute_data:
            stock_minute_data[today] = pd.DataFrame(columns=MINUTE_DF_COLUMNS).set_index(pd.to_datetime([]))

        minute_df = stock_minute_data[today]
        
        # [수정] market_data에서 직접 값을 가져옵니다.
        current_price = market_data['close']
        cumulative_volume = market_data['volume']
        last_known_volume = self._last_cumulative_volume.get(stock_code, 0)
        
        minute_volume = cumulative_volume - last_known_volume
        if minute_volume < 0: # 장 시작 등 누적 거래량이 초기화된 경우
            minute_volume = cumulative_volume 

        if write_key in minute_df.index:
            minute_df.loc[write_key, 'high'] = max(minute_df.loc[write_key, 'high'], current_price)
            minute_df.loc[write_key, 'low'] = min(minute_df.loc[write_key, 'low'], current_price)
            minute_df.loc[write_key, 'close'] = current_price
            minute_df.loc[write_key, 'volume'] += minute_volume
        else:
            new_row = {
                'stock_code': stock_code,
                'open': current_price, 
                'high': current_price, 
                'low': current_price, 
                'close': current_price, 
                'volume': minute_volume,
                'change_rate': 0.0,
                'trading_value': 0.0
            }
            # [수정] 새로운 행 추가 방식 변경
            new_df_row = pd.DataFrame([new_row], index=[write_key])
            stock_minute_data[today] = pd.concat([minute_df, new_df_row])
        
        self._last_cumulative_volume[stock_code] = cumulative_volume

    # def _run_minute_strategy_and_realtime_checks(self, current_dt: datetime) -> None:
    #     """
    #     분봉 전략 및 실시간 데이터를 기반으로 매매 로직을 실행합니다.
    #     - 매수/매도 신호 처리
    #     - 보유 종목 손절매/익절매 체크
    #     """
    #     # 현재 활성화된 매수 신호들을 확인
    #     active_buy_signals = self.manager.load_daily_signals(current_dt.date(), is_executed=False, signal_type='BUY')

    #     # 모든 종목의 실시간 현재가 데이터를 업데이트 (TradingManager를 통해 CreonAPIClient 사용)
    #     # 이 함수는 TradingManager가 시장 데이터로부터 실시간으로 받아오거나, 필요시 조회하여 업데이트할 것임.
    #     # 실제로는 CreonAPIClient의 실시간 시세 구독을 통해 이루어짐.
    #     # 여기서는 편의상 TradingManager가 최신 현재가를 가져온다고 가정
    #     current_prices = self.get_current_market_prices(list(self.broker.get_current_positions().keys()) + \
    #                                                                     list(active_buy_signals.keys()))
    #     # 분봉 전략 실행
    #     if self.minute_strategy:
    #         for stock_code, signal_info in active_buy_signals.items():
    #             if signal_info['stock_code'] == stock_code and signal_info['is_executed'] == False: # 아직 체결되지 않은 매수 신호
    #                 try:
    #                     # 분봉 전략에 현재 시간과 종목 코드를 전달하여 매매 판단
    #                     # [수정] '초'를 제거한 현재 분(minute)의 시작 시각을 전달합니다.
    #                     current_minute = current_dt.replace(second=0, microsecond=0)
    #                     self.minute_strategy.run_minute_logic(current_minute, stock_code)
    #                 except Exception as e:
    #                     logger.error(f"분봉 전략 '{self.minute_strategy.strategy_name}' 실행 중 오류 발생 ({stock_code}): {e}", exc_info=True)
    #                     self.notifier.send_message(f"❗ 분봉 전략 오류: {stock_code} - {e}")

    #     # 손절매/익절매 조건 체크 (보유 종목에 대해)
    #     self.broker.check_and_execute_stop_loss(current_prices, current_dt)
        
    def _ensure_minute_data_exists(self, stock_code: str, current_date: date):
        """특정 종목의 당일 분봉 데이터가 data_store에 없으면 DB/API에서 가져와 채웁니다."""
        
        stock_minute_data = self.data_store['minute'].get(stock_code, {})
        
        if current_date not in stock_minute_data:
            logger.info(f"[{stock_code}] 종목의 당일 분봉 데이터가 없어 따라잡기를 실행합니다.")
            
            try:
                # 직전 영업일 계산
                market_calendar_df = self.db_manager.fetch_market_calendar(current_date - timedelta(days=10), current_date)
                trading_days = market_calendar_df[market_calendar_df['is_holiday'] == 0]['date'].dt.date.sort_values().tolist()
                prev_trading_date = trading_days[-2] if len(trading_days) > 1 else current_date - timedelta(days=1)
                
                # 데이터 캐싱
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
        """[신규] 장 마감 후 ReportGenerator를 사용해 일일 성과를 기록하는 메서드"""
        
        # 1. 자동매매용 저장 전략 선택
        storage = TradingDB(self.db_manager)

        # 2. 리포트 생성기에 저장 전략 주입
        reporter = ReportGenerator(storage_strategy=storage)
        
        # 3. 리포트에 필요한 데이터 준비
        end_value = self.broker.get_portfolio_value(self.manager.get_current_prices(list(self.broker.get_current_positions().keys())))
        latest_portfolio = self.db_manager.fetch_latest_daily_portfolio()
        start_value = latest_portfolio.get('total_capital', self.broker.initial_cash) if latest_portfolio else self.broker.initial_cash
        
        # [수정] Series를 생성하기 전에 모든 값을 float으로 변환하여 타입을 통일합니다.
        portfolio_series = pd.Series(
            [float(start_value), float(end_value)], 
            index=[pd.Timestamp(current_date - timedelta(days=1)), pd.Timestamp(current_date)]
        )

        # 거래 로그는 DB에서 해당일의 로그를 다시 불러옴
        transaction_log = self.db_manager.fetch_trading_logs(current_date, current_date)

        # 4. 리포트 생성 및 저장
        reporter.generate(
            start_date=current_date,
            end_date=current_date,
            initial_cash=start_value,
            portfolio_value_series=portfolio_series, # 타입이 통일된 시리즈 전달
            transaction_log=transaction_log.to_dict('records') if not transaction_log.empty else [],
            strategy_info={
                'strategy_daily': self.daily_strategy.__class__.__name__,
                'strategy_minute': self.minute_strategy.__class__.__name__,
                'params_json_daily': self.daily_strategy.strategy_params,
                'params_json_minute': self.minute_strategy.strategy_params
            },
            cash_balance=self.broker.get_current_cash_balance()
        )



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
        
        # 5. [수정] 거래 준비 단계 실행
        if trading_system.prepare_for_system():
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