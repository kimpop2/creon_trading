# trading/trading.py

from datetime import datetime, date, timedelta, time
import logging
from typing import Dict, Any, List, Optional
import sys
import os
import threading

# 프로젝트 루트 경로를 sys.path에 추가 (다른 모듈 임포트를 위함)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.trading_manager import TradingManager
from trading.brokerage import Brokerage
from trading.trading_report import TradingReport # Reporter 타입 힌트를 위해 남겨둠
from strategies.trading_strategy import TradingStrategy
from util.notifier import Notifier
# --- 로거 설정 ---
logger = logging.getLogger(__name__)

class Trading:
    """
    자동매매 시스템의 메인 오케스트레이터 클래스입니다.
    자동매매 루프를 실행하고, 매매 전략을 조정하며, 실시간 데이터 및 주문 처리를 관리합니다.
    """
    def __init__(self,
                 api_client: CreonAPIClient,
                 db_manager: DBManager,
                 notifier: Notifier,
                 initial_cash: float = 10_000_000 # 초기 예수금 설정 (Brokerage 로 전달)
                 ):
        self.api_client = api_client
        self.db_manager = db_manager
        self.notifier = notifier
        self.initial_cash = initial_cash
        
        self.manager = TradingManager(self.api_client, self.db_manager) # initial_cash ???
        self.brokerage = Brokerage(self.api_client, self.manager, self.notifier, self.initial_cash)
        self.report = TradingReport(self.db_manager)
        self.strategy = None
        
        self.data_store = {'daily': {}, 'minute': {}} # {stock_code: DataFrame}
        
        self.is_running = False
        self.market_open_time = time(9, 0, 0)
        self.market_close_time = time(15, 30, 0)
        self.daily_strategy_run_time = time(8, 30, 0) # 일봉 전략 실행 시간 (장 시작 전)
        self.portfolio_update_time = time(16, 0, 0) # 포트폴리오 업데이트 시간 (장 마감 후)
        self.current_trading_date = datetime.now().date

        # Creon API의 실시간 체결/주문 응답 콜백 등록
        self.api_client.set_conclusion_callback(self.brokerage.handle_order_conclusion)
        
        logger.info("Trading 시스템 초기화 완료.")

    def set_strategies(self, strategy: Optional[TradingStrategy]) -> None:
        """
        사용할 일봉 및 분봉 전략을 설정합니다.
        """
        self.strategy = strategy
        
        if self.strategy:
            logger.info(f"매매 전략 설정: {self.strategy.strategy_name}")

    def set_broker_stop_loss_params(self, params: dict = None):
        self.broker.set_stop_loss_params(params)
        logging.info("브로커 손절매 파라미터 설정 완료.")

    # 필수 : 포트폴리오 손절시각 체크, 없으면 매분마다 보유종목 손절 체크로 비효율적
    def _should_check_portfolio(self, current_dt:datetime):
        """포트폴리오 체크가 필요한 시점인지 확인합니다."""
        if self.last_portfolio_check is None:
            return True
        
        current_time = current_dt.time()
        # 시간 비교를 정수로 변환하여 효율적으로 비교
        current_minutes = current_time.hour * 60 + current_time.minute
        check_minutes = [9 * 60, 15 * 60 + 20]  # 9:00, 15:20
        
        if current_minutes in check_minutes and (self.last_portfolio_check.date() != current_dt.date() or 
                                               (self.last_portfolio_check.hour * 60 + self.last_portfolio_check.minute) not in check_minutes):
            self.last_portfolio_check = current_dt
            return True
            
        return False

    def _update_daily_data_from_minute_bars(self, current_dt: datetime.datetime):
        """
        매분 현재 시각까지의 1분봉 데이터를 집계하여 일봉 데이터를 생성하거나 업데이트합니다.
        캐시를 사용하여 성능을 개선합니다.
        :param current_dt: 현재 시각 (datetime 객체)
        """
        current_date = current_dt.date()
        
        for stock_code in self.data_store['daily'].keys():
            minute_data_for_today = self.data_store['minute'].get(stock_code)

            if minute_data_for_today is not None:
                # 캐시 키 생성
                cache_key = f"{stock_code}_{current_date}"
                
                # 캐시된 데이터가 있고, 마지막 업데이트 시간이 현재 시각과 같거나 최신이면 스킵
                if cache_key in self._daily_update_cache:
                    last_update = self._daily_update_cache[cache_key]
                    if last_update >= current_dt:
                        continue
                
                # minute_data_for_today는 {date: DataFrame} dict 구조
                # current_date에 해당하는 DataFrame을 가져옴
                today_minute_bars = minute_data_for_today.get(current_date)

                if today_minute_bars is not None and not today_minute_bars.empty:
                    # 현재 시각까지의 분봉 데이터만 필터링
                    # current_dt 이하의 분봉 데이터만 사용
                    filtered_minute_bars = today_minute_bars[today_minute_bars.index <= current_dt]
                    
                    if not filtered_minute_bars.empty:
                        # 캐시된 필터링된 데이터가 있는지 확인
                        if cache_key in self._minute_data_cache:
                            cached_filtered_data = self._minute_data_cache[cache_key]
                            # 캐시된 데이터가 현재 시각까지의 데이터를 포함하는지 확인
                            if cached_filtered_data.index.max() >= current_dt:
                                filtered_minute_bars = cached_filtered_data[cached_filtered_data.index <= current_dt]
                            else:
                                # 캐시 업데이트
                                self._minute_data_cache[cache_key] = filtered_minute_bars
                        else:
                            # 캐시에 저장
                            self._minute_data_cache[cache_key] = filtered_minute_bars
                        
                        # 현재 시각까지의 일봉 데이터 계산
                        daily_open = filtered_minute_bars.iloc[0]['open']  # 첫 분봉 시가
                        daily_high = filtered_minute_bars['high'].max()    # 현재까지 최고가
                        daily_low = filtered_minute_bars['low'].min()      # 현재까지 최저가
                        daily_close = filtered_minute_bars.iloc[-1]['close']  # 현재 시각 종가
                        daily_volume = filtered_minute_bars['volume'].sum()   # 현재까지 누적 거래량

                        # 새로운 일봉 데이터 생성
                        new_daily_bar = pd.Series({
                            'open': daily_open,
                            'high': daily_high,
                            'low': daily_low,
                            'close': daily_close,
                            'volume': daily_volume
                        }, name=pd.Timestamp(current_date))

                        # 일봉 데이터가 존재하면 업데이트, 없으면 추가
                        if pd.Timestamp(current_date) in self.data_store['daily'][stock_code].index:
                            # 기존 일봉 데이터 업데이트
                            self.data_store['daily'][stock_code].loc[pd.Timestamp(current_date)] = new_daily_bar
                        else:
                            # 새로운 일봉 데이터 추가
                            self.data_store['daily'][stock_code] = pd.concat([
                                self.data_store['daily'][stock_code], 
                                pd.DataFrame([new_daily_bar])
                            ])
                            # 인덱스 정렬
                            self.data_store['daily'][stock_code].sort_index(inplace=True)
                        
                        # 업데이트 시간 캐시
                        self._daily_update_cache[cache_key] = current_dt

    def _clear_daily_update_cache(self):
        """
        일봉 업데이트 캐시를 초기화합니다. 새로운 날짜로 넘어갈 때 호출됩니다.
        """
        self._daily_update_cache.clear()
        self._minute_data_cache.clear()

    def start_trading_loop(self) -> None:
        """
        자동매매의 메인 루프를 시작합니다.
        이 함수는 블로킹 방식으로 실행되며, Ctrl+C 등으로 종료될 때까지 반복됩니다.
        """
        self.is_running = True
        self.notifier.send_message("🚀 자동매매 시스템이 시작되었습니다!")
        logger.info("자동매매 루프 시작...")

        while self.is_running:
            now = datetime.now()
            current_date = now.date()
            current_time = now.time()

            # 장 마감 시간 체크 및 자체 종료
            if current_time >= self.market_close_time:
                logger.info(f"[{now.strftime('%H:%M:%S')}] 장 마감 시간({self.market_close_time}) 도달. 시스템 자체 종료 시작.")
                self.is_running = False # 루프 종료 플래그 설정
                break # 즉시 루프 종료

            # 매매일이 변경되었는지 확인 및 초기화
            if self.current_trading_date != current_date:
                self.current_trading_date = current_date
                self._daily_reset_and_preparation(now)

            # 1. 전략 실행 (매일 장 시작 전)
            if current_time >= self.daily_strategy_run_time and \
               current_time < self.market_open_time and \
               not getattr(self, '_strategy_run_today', False): # 오늘 실행 여부 플래그
                logger.info(f"[{now.strftime('%H:%M:%S')}] 전략 로직 실행...")
                if self.strategy:
                    try:
                        self.strategy.run_strategy_logic(now)
                        logger.info(f"전략 '{self.strategy.strategy_name}' 실행 완료.")
                    except Exception as e:
                        logger.error(f"전략 실행 중 오류 발생: {e}", exc_info=True)
                        self.notifier.send_message(f"❗전략 오류: {self.strategy.strategy_name} - {e}")
                else:
                    logger.info("설정된 전략이 없습니다.")
                setattr(self, '_strategy_run_today', True) # 오늘 실행 완료 플래그 설정

            # 2. 장 중 분봉/실시간 데이터 기반 로직 (개장 시간 동안)
            if self.market_open_time <= current_time < self.market_close_time:
                # 장 시작 후 실시간 데이터 구독 및 분봉 로직 실행
                self._run_minute_strategy_and_realtime_checks(now)
                time.sleep(10) # 10초마다 체크 (실시간 데이터 처리량에 따라 조정)
            elif current_time >= self.market_close_time and \
                 current_time < self.portfolio_update_time and \
                 getattr(self, '_strategy_run_today', False) and \
                 not getattr(self, '_portfolio_updated_today', False):
                # 3. 장 마감 후 포트폴리오 업데이트 및 일일 결산
                logger.info(f"[{now.strftime('%H:%M:%S')}] 장 마감 후 포트폴리오 업데이트 및 결산...")
                self.brokerage.update_portfolio_status(now)
                setattr(self, '_portfolio_updated_today', True)
                setattr(self, '_strategy_run_today', False) # 다음날을 위해 초기화
            elif current_time >= self.portfolio_update_time:
                # 다음날을 위해 포트폴리오 업데이트 플래그 초기화
                setattr(self, '_portfolio_updated_today', False)

            # 비 영업시간 및 장 마감 후 대기
            if not (self.market_open_time <= current_time < self.market_close_time) and \
               not (self.daily_strategy_run_time <= current_time < self.market_open_time):
                # 다음 주요 시간까지 대기 (예: 다음 분 또는 다음 일봉 전략 실행 시간)
                next_check_time = now + timedelta(minutes=1)
                if current_time < self.daily_strategy_run_time:
                    next_check_time = datetime.combine(current_date, self.daily_strategy_run_time)
                elif current_time < self.market_open_time:
                    next_check_time = datetime.combine(current_date, self.market_open_time)
                elif current_time < self.market_close_time:
                    next_check_time = datetime.combine(current_date, self.market_close_time)
                elif current_time < self.portfolio_update_time:
                    next_check_time = datetime.combine(current_date, self.portfolio_update_time)
                else: # 오늘 모든 작업이 끝났다면 다음 날 장 시작 시간까지 대기
                    next_check_time = datetime.combine(current_date + timedelta(days=1), self.daily_strategy_run_time)

                wait_seconds = (next_check_time - now).total_seconds()
                if wait_seconds > 0:
                    logger.info(f"[{now.strftime('%H:%M:%S')}] 다음 주요 시간까지 대기: {int(wait_seconds)}초")
                    time.sleep(min(wait_seconds, 60)) # 최대 1분씩 대기하며 주기적으로 재확인
                else:
                    time.sleep(1) # 시간 역전 방지
            
            # 주말 체크 (실제 환경에서는 별도의 휴장일 API 연동 필요)
            if now.weekday() >= 5: # 토요일(5), 일요일(6)
                logger.info(f"[{now.strftime('%H:%M:%S')}] 주말입니다. 다음 월요일까지 대기...")
                # 다음 월요일까지 대기 (예시)
                days_until_monday = (7 - now.weekday()) % 7
                if days_until_monday == 0: # 현재가 일요일이면 다음 월요일은 1일 후
                    days_until_monday = 1
                next_monday = current_date + timedelta(days=days_until_monday)
                sleep_duration = (datetime.combine(next_monday, self.daily_strategy_run_time) - now).total_seconds()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                continue # 루프 재시작

            # 시스템 종료 조건 (예: 특정 시간, 외부 신호)
            # 여기서는 무한 루프이므로, 외부에서 self.is_running을 False로 설정해야 종료됨
            
        logger.info("자동매매 루프 종료.")
        self.notifier.send_message("🛑 자동매매 시스템이 종료되었습니다.")
        self.cleanup()

    def _daily_reset_and_preparation(self, current_date: date) -> None:
        """
        매일 새로운 거래일을 시작할 때 필요한 초기화 및 준비 작업을 수행합니다.
        """
        logger.info(f"--- {current_date} 새로운 거래일 준비 시작 ---")
        self.notifier.send_message(f"--- {current_date} 새로운 거래일 준비 시작 ---")
        
        # 매매 시스템 상태 플래그 초기화
        setattr(self, '_strategy_run_today', False)
        setattr(self, '_portfolio_updated_today', False)

        # Creon API 연결 상태 확인 및 재연결 시도
        if not self.api_client._check_creon_status():
            logger.warning("Creon API 연결이 끊어졌습니다. 재연결을 시도합니다...")
            if not self.api_client._check_creon_status():
                self.notifier.send_message("❌ Creon API 연결 실패. 시스템 종료 또는 수동 확인 필요.")
                logger.error("Creon API 연결 실패. 자동매매를 진행할 수 없습니다.")
                self.stop_trading() # 심각한 오류이므로 시스템 종료 고려
                return
            else:
                self.notifier.send_message("✅ Creon API 재연결 성공.")

        # Brokerage 계좌 상태 동기화 (전일 종가 및 장 마감 처리 후 최종 업데이트된 정보 반영)
        self.brokerage.sync_account_status()

        # 일봉/분봉 전략의 활성 신호 로드 (전일 미체결 신호 등)
        if self.strategy:
            self.strategy.load_active_signals(current_date)


        logger.info(f"--- {current_date} 새로운 거래일 준비 완료 ---")



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
        current_prices = self.manager.get_current_market_prices(list(self.brokerage.get_current_positions().keys()) + \
                                                                        list(active_buy_signals.keys()))
        # 분봉 전략 실행
        if self.strategy:
            for stock_code, signal_info in active_buy_signals.items():
                if signal_info['stock_code'] == stock_code and signal_info['is_executed'] == False: # 아직 체결되지 않은 매수 신호
                    try:
                        # 분봉 전략에 현재 시간과 종목 코드를 전달하여 매매 판단
                        self.strategy.run_trading_logic(current_dt, stock_code)
                    except Exception as e:
                        logger.error(f"분봉 전략 '{self.minute_strategy.strategy_name}' 실행 중 오류 발생 ({stock_code}): {e}", exc_info=True)
                        self.notifier.send_message(f"❗ 분봉 전략 오류: {stock_code} - {e}")

        # 손절매/익절매 조건 체크 (보유 종목에 대해)
        self.brokerage.check_and_execute_stop_loss(current_prices, current_dt)
        
        # TODO: 미체결 주문 관리 로직 추가 (TradingManager에서 주기적으로 조회 및 갱신)
        # self.brokerage.get_unfilled_orders()를 통해 미체결 주문 상태를 확인하고,
        # 필요에 따라 정정/취소 로직을 호출할 수 있습니다.
        # 예: 특정 시간까지 미체결 시 취소 후 재주문 또는 타임컷 매도 등.


    def stop_trading(self) -> None:
        """
        자동매매 루프를 안전하게 종료합니다.
        """
        logger.info("자동매매 시스템 종료 요청 수신.")
        self.is_running = False

    def cleanup(self) -> None:
        """
        시스템 종료 시 필요한 리소스 정리 작업을 수행합니다.
        """
        logger.info("Trading 시스템 cleanup 시작.")
        if self.brokerage:
            self.brokerage.cleanup()
        if self.api_client:
            self.api_client.cleanup()
        if self.db_manager:
            self.db_manager.close()
        logger.info("Trading 시스템 cleanup 완료.")


# TODO: 실제 사용 시 main 함수에서 Trading 객체를 생성하고 루프 시작
# 예시:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])

    # Creon API 연결
    api_client = CreonAPIClient()
    # DBManager, Notifier 초기화
    db_manager = DBManager()
    # 실제 텔레그램 토큰 및 채팅 ID 설정 필요
    telegram_notifier = Notifier(telegram_token="YOUR_TELEGRAM_BOT_TOKEN", telegram_chat_id="YOUR_TELEGRAM_CHAT_ID")

    # Trading 시스템 초기화
    trading_system = Trading(api_client, db_manager, telegram_notifier, initial_deposit=1_000_000)

    # 전략 설정 (예시) - 실제로는 config 등에서 로드하여 인스턴스 생성
    
    # SMA 전략 설정 (최적화 결과 반영)
    sma_strategy_params={
        'short_sma_period': 5,          #  4 → 5일 (더 안정적인 단기 이동평균)
        'long_sma_period': 20,          #  10 → 20일 (더 안정적인 장기 이동평균)
        'volume_ma_period': 10,         #  6 → 10일 (거래량 이동평균 기간 확장)
        'num_top_stocks': 5,            #  5 → 3 (집중 투자)
    }
    strategy_params = {
        'short_sma_period': 5, 
        'long_sma_period': 20, 
        'volume_ma_period': 20, 
        'num_top_stocks': 10
    }
    # 전략 인스턴스 생성
    from strategies.sma_strategy import SMAStrategy
    strategy_instance = SMAStrategy(trading_system.brokerage, trading_system.trading_manager, sma_strategy_params)

    trading_system.set_strategies(strategy=strategy_instance) # 임시로 전략 없음

    try:
        trading_system.start_trading_loop()
    except KeyboardInterrupt:
        logger.info("사용자에 의해 시스템 종료 요청됨.")
    finally:
        trading_system.cleanup()
        logger.info("시스템 종료 완료.")