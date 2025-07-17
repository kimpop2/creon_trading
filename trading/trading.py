# trading/trading.py
import pandas as pd
import logging
from datetime import datetime, date, timedelta, time
import time as pytime
from typing import Dict, Any, List, Optional
import sys
import os

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

logger = logging.getLogger(__name__)

class Trading:
    """
    자동매매 시스템의 메인 오케스트레이터 클래스입니다.
    """
    def __init__(self, api_client: CreonAPIClient, db_manager: DBManager, notifier: Notifier):
        self.api_client = api_client
        self.db_manager = db_manager
        self.notifier = notifier

        # --- [수정] __init__ 생성자 정리 ---
        # 1. Manager를 먼저 생성합니다.
        self.manager = TradingManager(self.api_client, self.db_manager)
        # 2. Brokerage 생성 시 Manager를 주입하고, 타입 힌트는 AbstractBroker를 사용합니다.
        self.broker: AbstractBroker = Brokerage(self.api_client, self.manager, self.notifier)
        
        self.daily_strategy: Optional[DailyStrategy] = None
        self.minute_strategy: Optional[MinuteStrategy] = None
        
        self.data_store = {'daily': {}, 'minute': {}}
        
        self.is_running = True
        self.market_open_time = time(9, 0)
        self.market_close_time = time(15, 30)
        self.last_strategy_run_time = datetime.min
        self._portfolio_updated_today = False
        self.current_trading_date: Optional[date] = None

        # 일봉 업데이트 캐시 변수들 (성능 개선용)
        self._daily_update_cache = {}  # {stock_code: {date: last_update_time}}
        self._minute_data_cache = {}   # {stock_code: {date: filtered_data}}

        # Creon API의 실시간 체결 콜백을 Brokerage의 핸들러에 연결
        self.api_client.set_conclusion_callback(self.broker.handle_order_conclusion)
        
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

    def add_daily_data(self, stock_code: str, df: pd.DataFrame):
        """백테스트를 위한 일봉 데이터를 추가합니다."""
        if not df.empty:
            self.data_store['daily'][stock_code] = df
            logging.debug(f"일봉 데이터 추가: {stock_code}, {len(df)}행")
        else:
            logging.warning(f"빈 데이터프레임이므로 {stock_code}의 일봉 데이터를 추가하지 않습니다.")

    # def add_minute_data(self, stock_code: str, df: pd.DataFrame):
    #     """백테스트를 위한 분봉 데이터를 추가합니다."""
    #     if not df.empty:
    #         self.data_store['minute'][stock_code] = df
    #         logging.debug(f"분봉 데이터 추가: {stock_code}, {len(df)}행")
    #     else:
    #         logging.warning(f"빈 데이터프레임이므로 {stock_code}의 분봉 데이터를 추가하지 않습니다.")


    # 자동매매 전용
    def add_signal(self, stock_code: str, signal_type: str, target_price: float, target_quantity: int, strategy_name: str) -> None:
        """
        새로운 매매 신호를 self.signals에 추가하고 DB에 저장합니다.
        """
        signal_data = {
            'stock_code': stock_code,
            'stock_name': self.manager.get_stock_name(stock_code),
            'signal_date': datetime.now().date(), # 신호 생성일
            'signal_type': signal_type,
            'target_price': target_price,
            'target_quantity': target_quantity,
            'strategy_name': strategy_name,
            'is_executed': False,
            'executed_order_id': None
        }

        success = self.manager.save_daily_signals(signal_data)
        if success:
            # DB 저장 성공 시, signal_id를 받아 signals 딕셔너리에 추가
            # TODO: save_daily_signals가 저장 후 signal_id를 반환하도록 수정 필요
            # 현재는 저장 성공 여부만 반환하므로, signals 딕셔너리에는 signal_id가 None으로 들어갈 수 있음.
            # 실제 사용 시에는 DB에서 signal_id를 다시 조회하거나, save_daily_signals에서 반환하도록 변경해야 함.
            self.signals[stock_code] = {**signal_data, 'signal_id': None} # 임시로 None
            logger.info(f"신호 추가: {stock_code}, 타입: {signal_type}, 가격: {target_price}, 수량: {target_quantity}")
        else:
            logger.error(f"신호 DB 저장 실패: {stock_code}, 타입: {signal_type}")

    # 자동매매 전용
    def load_active_signals(self, signal_date: date) -> None:
        """
        특정 날짜에 유효한(아직 실행되지 않은) 신호들을 DB에서 로드하여 self.signals에 설정합니다.
        주로 장 시작 시 호출됩니다.
        """
        active_signals = self.manager.load_daily_signals(signal_date, is_executed=False)
        self.signals = active_signals # trading_manager에서 딕셔너리 형태로 반환되므로 바로 할당
        if self.signals:
            logger.info(f"{signal_date}의 활성 신호 {len(self.signals)}건 로드 완료.")
        else:
            logger.info(f"{signal_date}에 로드할 활성 신호가 없습니다.")


    # 포트폴리오 손절시각 체크, 없으면 매분마다 보유종목 손절 체크로 비효율적
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

    # def check_portfolio_stop_loss(self, current_dt: datetime, current_prices: Dict[str, float]) -> bool:
    #     """
    #     포트폴리오 전체 손절매 조건을 확인하고 실행합니다.
    #     시뮬레이션에서는 이 함수가 매 분 호출될 수 있습니다.
    #     """
    #     if self.broker.stop_loss_params and self.broker.stop_loss_params.get('portfolio_stop_loss_enabled', False):
    #         # 특정 시간 이후에만 손절매 검사
    #         if current_dt.time() >= datetime.time(self.broker.stop_loss_params.get('portfolio_stop_loss_start_hour', 14), 0, 0):
    #             losing_positions_count = 0
    #             for stock_code, position in self.positions.items():
    #                 if position['size'] > 0 and stock_code in current_prices:
    #                     loss_ratio = self._calculate_loss_ratio(current_prices[stock_code], position['avg_price'])
    #                     if loss_ratio >= self.broker.stop_loss_params['stop_loss_ratio']: # 손실률이 기준 이상인 경우
    #                         losing_positions_count += 1

    #             if losing_positions_count >= self.broker.stop_loss_params['max_losing_positions']:
    #                 logger.info(f'[{current_dt.isoformat()}] [포트폴리오 손절] 손실 종목 수: {losing_positions_count}개 (기준: {self.broker.stop_loss_params["max_losing_positions"]}개 이상)')
    #                 self._execute_portfolio_sellout(current_prices, current_dt)
    #                 return True
    #     return False

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

    def _update_daily_data_from_minute_bars(self, current_dt: datetime):
        """
        매분 현재 시각까지의 1분봉 데이터를 집계하여 일봉 데이터를 생성하거나 업데이트합니다.
        _minute_data_cache를 활용하여 성능을 개선합니다.
        :param current_dt: 현재 시각 (datetime 객체)
        """
        current_date = current_dt.date()
        
        for stock_code in self.data_store['daily'].keys():
            # _minute_data_cache에서 해당 종목의 오늘치 전체 분봉 데이터를 가져옵니다.
            # 이 데이터는 run 함수 진입 시 cache_minute_ohlcv를 통해 이미 로드되어 있어야 합니다.
            
            today_minute_bars = self._minute_data_cache.get(stock_code)

            if today_minute_bars is not None and not today_minute_bars.empty:
                # 현재 시각까지의 분봉 데이터만 필터링 (슬라이싱)
                # 이 부분에서 불필요한 복사를 줄이기 위해 .loc을 사용합니다.
                filtered_minute_bars = today_minute_bars.loc[today_minute_bars.index <= current_dt]
                
                if not filtered_minute_bars.empty:
                    # 현재 시각까지의 일봉 데이터 계산
                    daily_open = filtered_minute_bars.iloc[0]['open']  # 첫 분봉 시가
                    daily_high = filtered_minute_bars['high'].max()    # 현재까지 최고가
                    daily_low = filtered_minute_bars['low'].min()      # 현재까지 최저가
                    daily_close = filtered_minute_bars.iloc[-1]['close']  # 현재 시각 종가
                    daily_volume = filtered_minute_bars['volume'].sum()   # 현재까지 누적 거래량

                    # 새로운 일봉 데이터 생성 (Series로 생성하여 성능 개선)
                    new_daily_bar = pd.Series({
                        'open': daily_open,
                        'high': daily_high,
                        'low': daily_low,
                        'close': daily_close,
                        'volume': daily_volume
                    }, name=pd.Timestamp(current_date)) # 인덱스를 날짜로 설정

                    # 일봉 데이터가 존재하면 업데이트, 없으면 추가
                    # .loc을 사용하여 직접 업데이트 (기존 DataFrame의 인덱스를 활용)
                    self.data_store['daily'][stock_code].loc[pd.Timestamp(current_date)] = new_daily_bar
                    
                    # 일봉 데이터가 추가되거나 업데이트될 때 인덱스 정렬은 필요 없음 (loc으로 특정 위치 업데이트)
                    # 단, 새로운 날짜가 추가될 경우 기존 DataFrame에 없던 인덱스가 추가되므로 sort_index는 필요할 수 있습니다.
                    # 하지만 백테스트에서는 날짜 순서대로 진행되므로 대부분의 경우 문제가 되지 않습니다.
            else:
                logging.debug(f"[{current_dt.isoformat()}] {stock_code}의 오늘치 분봉 데이터가 없거나 비어있어 일봉 업데이트를 건너킵니다.")

    def _clear_daily_update_cache(self):
        """
        일봉 업데이트 캐시를 초기화합니다. 새로운 날짜로 넘어갈 때 호출됩니다.
        """
        self._daily_update_cache.clear()
        self._minute_data_cache.clear()

    def run(self) -> None:
        """
        자동매매의 메인 루프를 시작합니다.
        (수정) 장중에 10분 간격으로 전략을 재실행합니다.
        """
        if not self.daily_strategy or not self.minute_strategy:
            logger.error("일봉 또는 분봉 전략이 설정되지 않았습니다. 자동매매를 중단합니다.")
            return

        self.is_running = True
        self.notifier.send_message("🚀 자동매매 시스템이 시작되었습니다! (10분 주기 스캔)")
        logger.info("자동매매 루프 시작...")
        
        self.current_trading_date = None

        while self.is_running:
            try:
                now = datetime.now()
                current_date = now.date()
                current_time = now.time()
                # 1. 장 마감 시간 이후 자동 종료
                if current_time > self.market_close_time and not getattr(self, '_portfolio_updated_today', False):
                    logger.info(f"[{now.strftime('%H:%M:%S')}] 장 마감. 포트폴리오 결산을 진행합니다.")
                    
                    # [수정] ReportGenerator를 사용하여 일일 성과 기록
                    self.record_daily_performance(now.date())
                    
                    setattr(self, '_portfolio_updated_today', True)                    
                    logger.info("오늘의 모든 작업을 종료합니다. 다음 거래일까지 대기합니다.")
                    pytime.sleep(60) # 1분 후 다음 날짜 체크
                    continue

                # 2. 새로운 거래일 준비
                if self.current_trading_date != current_date:
                    if now.weekday() >= 5: # 토, 일
                        logger.info(f"주말입니다. 다음 거래일까지 대기합니다.")
                        pytime.sleep(60)
                        continue
                    
                    self._daily_reset_and_preparation(current_date)
                    self.current_trading_date = current_date
                    self.last_strategy_run_time = None # 새 날짜가 되면 마지막 실행 시간 초기화

                # 💡 [수정된 핵심 로직] 장중(9:00 ~ 15:30)에 주기적으로 전략 실행
                if self.market_open_time <= current_time <= self.market_close_time:
                    
                    # 마지막 실행 후 10분이 경과했거나, 오늘 처음 실행하는 경우
                    should_run_strategy = (
                        self.last_strategy_run_time is None or
                        (now - self.last_strategy_run_time).total_seconds() >= 20 # 10분 = 600초
                    )

                    if should_run_strategy:
                        logger.info(f"[{now.strftime('%H:%M:%S')}] 10분 주기 도래. 일봉 전략 재실행 및 신호 업데이트...")
                        self._run_daily_strategy_and_prepare_data(current_date)
                        self.last_strategy_run_time = now # 마지막 실행 시간 기록
                    
                    # 분봉 전략 및 실시간 체크는 매 루프마다 실행
                    logger.debug(f"[{now.strftime('%H:%M:%S')}] 장중 매매 로직 실행...")
                    self._run_minute_strategy_and_realtime_checks(now)
                    
                    pytime.sleep(10) # 10초마다 장중 로직 반복
                    continue

                # 5. 그 외 시간 (대기)
                logger.debug(f"[{now.strftime('%H:%M:%S')}] 현재는 매매 관련 활동 시간이 아닙니다. 대기합니다.")
                pytime.sleep(30)

            except KeyboardInterrupt:
                logger.info("사용자에 의해 시스템 종료 요청됨.")
                self.is_running = False
            except Exception as e:
                logger.error(f"메인 루프에서 예외 발생: {e}", exc_info=True)
                self.notifier.send_message(f"🚨 시스템 오류 발생: {e}")
                pytime.sleep(60)

        logger.info("자동매매 루프가 정상적으로 종료되었습니다.")
        self.cleanup()

    def _daily_reset_and_preparation(self, current_date: date) -> None:
        """
        매일 새로운 거래일을 시작할 때 필요한 초기화 및 준비 작업을 수행합니다.
        """
        logger.info(f"--- {current_date} 새로운 거래일 준비 시작 ---")
        self.notifier.send_message(f"--- {current_date} 새로운 거래일 준비 시작 ---")
        
        setattr(self, '_strategy_run_today', False)
        setattr(self, '_portfolio_updated_today', False)

        # Creon API 연결 상태 확인
        if not self.api_client.is_connected():
            logger.error("Creon API 연결이 끊어졌습니다. 자동매매를 진행할 수 없습니다.")
            self.notifier.send_message("❌ Creon API 연결 실패. 시스템 종료 필요.")
            self.stop_trading()
            return

        self.broker.sync_account_status()
        self.load_active_signals(current_date)
        logger.info(f"--- {current_date} 새로운 거래일 준비 완료 ---")
    
    def _run_daily_strategy_and_prepare_data(self, current_date: date) -> None:
        """
        [최종 수정] 일봉 전략 실행 후, 유효한 신호만 필터링하여 DB에 저장하고 분봉 전략에 전달합니다.
        """
        now = datetime.now()
        logger.info(f"[{now.strftime('%H:%M:%S')}] 일봉 전략 실행 및 데이터 준비 시작...")

        if now.time() >= self.market_open_time:
            self._update_daily_data_from_minute_bars(now)

        # 1. 일봉 전략 실행하여 모든 분석 대상에 대한 신호 생성
        try:
            self.daily_strategy.run_daily_logic(current_date)
            logger.info(f"일봉 전략 '{self.daily_strategy.__class__.__name__}' 실행 완료.")
        except Exception as e:
            logger.error(f"일봉 전략 실행 중 오류 발생: {e}", exc_info=True)
            self.notifier.send_message(f"❗전략 오류: {self.daily_strategy.__class__.__name__} - {e}")
            return
            
        # 2. 💡 [핵심 수정] 유효한 신호('buy', 'sell', 'hold')만 필터링
        all_signals = self.daily_strategy.signals
        valid_signals = {
            code: info for code, info in all_signals.items()
            if info.get('signal_type') in ['buy', 'sell', 'hold']
        }

        # 3. 필터링된 유효 신호만 DB에 저장 --> 최소한 만 DB 에 저장
        # if valid_signals:
        #     logger.info(f"생성된 유효 신호 {len(valid_signals)}건을 DB에 저장합니다.")
        #     for stock_code, signal_info in valid_signals.items():
        #         if 'strategy_name' not in signal_info:
        #             signal_info['strategy_name'] = self.daily_strategy.strategy_name
        #         if 'stock_name' not in signal_info:
        #             signal_info['stock_name'] = self.manager.get_stock_name(stock_code)
                
        #         self.manager.save_daily_signals(signal_info)
        
        # 4. 필터링된 유효 신호만 분봉 전략으로 전달
        self.minute_strategy.update_signals(valid_signals)
        
        # 5. 분봉 데이터가 필요한 종목 목록 취합 (보유종목 + 유효 신호 종목)
        stocks_to_load = set(self.broker.get_current_positions().keys()) | set(valid_signals.keys())
        stocks_to_load.add('U001')
        if not stocks_to_load:
            logger.info("금일 거래 대상 종목이 없어 데이터 로드를 건너뜁니다.")
            return

        # 6. 필요한 종목들의 분봉 데이터 로드 및 캐시
        logger.info(f"총 {len(stocks_to_load)}개 종목의 분봉 데이터를 로드합니다: {list(stocks_to_load)}")
        prev_trading_day = current_date - timedelta(days=1) 
        
        for stock_code in stocks_to_load:
            minute_df = self.manager.cache_minute_ohlcv(stock_code, prev_trading_day, current_date)
            logger.warning(f"{stock_code}의 분봉 {len(minute_df)}데이터 로드.")
            if not minute_df.empty:
                if stock_code not in self.data_store['minute']:
                    self.data_store['minute'][stock_code] = {}
                for date_key in [prev_trading_day, current_date]:
                    date_data = minute_df[minute_df.index.date == date_key]
                    if not date_data.empty:
                        self.data_store['minute'][stock_code][date_key] = date_data

                today_minute_bars = minute_df[minute_df.index.date == current_date]
                if not today_minute_bars.empty:
                    self._minute_data_cache[stock_code] = today_minute_bars
            else:
                logger.warning(f"{stock_code}의 분봉 데이터 로드 실패.")
        logger.info("분봉 데이터 준비 완료.")

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
        self.broker.sync_account_status()

        # 일봉/분봉 전략의 활성 신호 로드 (전일 미체결 신호 등)
        self.load_active_signals(current_date)

        logger.info(f"--- {current_date} 새로운 거래일 준비 완료 ---")


    def update_realtime_minute_data(self, stock_code, current_price):
        """현재가를 받아 분봉 데이터를 업데이트합니다."""
        
        current_time = datetime.now()
        # 분 단위로 시간을 정규화 (예: 10:15:34 -> 10:15:00)
        current_minute = current_time.replace(second=0, microsecond=0)
        today_str = current_time.strftime('%Y-%m-%d')

        # 오늘 날짜의 분봉 데이터프레임이 없으면 새로 생성
        if today_str not in self.data_store['minute'][stock_code]:
            self.data_store['minute'][stock_code][today_str] = pd.DataFrame(
                columns=['open', 'high', 'low', 'close', 'volume']
            )

        today_df = self.data_store['minute'][stock_code][today_str]

        # 현재 분(minute)에 해당하는 데이터가 있는지 확인
        if current_minute in today_df.index:
            # 데이터가 있으면 high, low, close 업데이트
            today_df.loc[current_minute, 'high'] = max(today_df.loc[current_minute, 'high'], current_price)
            today_df.loc[current_minute, 'low'] = min(today_df.loc[current_minute, 'low'], current_price)
            today_df.loc[current_minute, 'close'] = current_price
            # (필요 시) 누적 거래량 추가
        else:
            # 데이터가 없으면 새로운 행(분봉) 추가
            new_row = {'open': current_price, 'high': current_price, 'low': current_price, 'close': current_price, 'volume': 0} # 초기 거래량은 0으로 설정
            today_df.loc[current_minute] = new_row


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

    def stop_trading(self) -> None:
        self.is_running = False
        logger.info("자동매매 시스템 종료 요청 수신.")

    def cleanup(self) -> None:
        """
        시스템 종료 시 필요한 리소스 정리 작업을 수행합니다.
        """
        logger.info("Trading 시스템 cleanup 시작.")
        if self.broker:
            self.broker.cleanup()
        if self.api_client:
            self.api_client.cleanup()
        if self.db_manager:
            self.db_manager.close()
        logger.info("Trading 시스템 cleanup 완료.")

    def load_stocks(self, start_date, end_date):
        from config.sector_stocks import sector_stocks
        # 모든 종목 데이터 로딩: 하나의 리스트로 변환
        fetch_start = start_date - timedelta(days=30)
        stock_names = []
        for sector, stocks in sector_stocks.items():
            for stock_name, _ in stocks:
                stock_names.append(stock_name)

        all_target_stock_names = stock_names
        for name in all_target_stock_names:
            code = self.api_client.get_stock_code(name)
            if code:
                logging.info(f"'{name}' (코드: {code}) 종목 일봉 데이터 로딩 중... (기간: {fetch_start.strftime('%Y%m%d')} ~ {end_date.strftime('%Y%m%d')})")
                daily_df = self.manager.cache_daily_ohlcv(code, fetch_start, end_date)
                
                if daily_df.empty:
                    logging.warning(f"{name} ({code}) 종목의 일봉 데이터를 가져올 수 없습니다. 해당 종목을 건너뜁니다.")
                    continue
                logging.debug(f"{name} ({code}) 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
                self.add_daily_data(code, daily_df)
            else:
                logging.warning(f"'{name}' 종목의 코드를 찾을 수 없습니다. 해당 종목을 건너뜁니다.")

# TODO: 실제 사용 시 main 함수에서 Trading 객체를 생성하고 루프 시작
# 예시:
if __name__ == "__main__":
    from datetime import date, datetime
    from strategies.sma_daily import SMADaily
    from strategies.rsi_minute import RSIMinute
    # 설정 파일 로드
    from config.settings import (
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
        INITIAL_CASH,
        MARKET_OPEN_TIME, MARKET_CLOSE_TIME,
        DAILY_STRATEGY_RUN_TIME, PORTFOLIO_UPDATE_TIME,
        SMA_DAILY_PARAMS, RSI_MINUTE_PARAMS, STOP_LOSS_PARAMS,
        LOG_LEVEL, LOG_FILE
    )   
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    try:
        # 1. 필요한 인스턴스들 생성
        api_client = CreonAPIClient()
        db_manager = DBManager()
        notifier = Notifier(telegram_token=TELEGRAM_BOT_TOKEN, telegram_chat_id=TELEGRAM_CHAT_ID)

        # 2. 자동매매 인스턴스 생성
        trading_system = Trading(
            api_client=api_client,
            db_manager=db_manager,
            notifier=notifier
        )
        
        # 전략 인스턴스 생성 broker와 data_store는 trading_system 내부의 것을 사용
        # SMA 전략 설정
        from strategies.sma_daily import SMADaily
        daily_strategy = SMADaily(broker=trading_system.broker, data_store=trading_system.data_store, strategy_params=SMA_DAILY_PARAMS)

        # RSI 분봉 전략 설정 (최적화 결과 반영)
        from strategies.rsi_minute import RSIMinute
        minute_strategy = RSIMinute(broker=trading_system.broker, data_store=trading_system.data_store, strategy_params=RSI_MINUTE_PARAMS)        
        
        # # RSI 가상 분봉 전략 설정 (최적화 결과 반영)
        # from strategies.open_minute import OpenMinute
        # minute_strategy = OpenMinute(broker=trading_system.broker, data_store=trading_system.data_store, strategy_params=RSI_MINUTE_PARAMS)        
        
        # 일봉/분봉 전략 설정
        trading_system.set_strategies(daily_strategy=daily_strategy, minute_strategy=minute_strategy)
        # 손절매 파라미터 설정 (선택사항)
        trading_system.set_broker_stop_loss_params(STOP_LOSS_PARAMS)

        # 일봉 데이터 로드
        end_date = date.today()
        start_date = end_date - timedelta(days=60)

        trading_system.load_stocks(start_date, end_date)

        try:
            logger.info("=== 자동매매 시작 ===")
            trading_system.run()
        except KeyboardInterrupt:
            logger.info("사용자에 의해 시스템 종료 요청됨.")
        finally:
            trading_system.cleanup()
            logger.info("시스템 종료 완료.")

    except Exception as e:
        logger.error(f"Backtest 테스트 중 오류 발생: {e}", exc_info=True)            