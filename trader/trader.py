# trade/trader.py

import logging
import pandas as pd
from datetime import datetime, date, time, timedelta
import time as time_module # time 모듈과 충돌 방지
import sys
import os
from typing import Dict, Any, Optional

# sys.path에 프로젝트 루트 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from trader.brokerage import Brokerage
from manager.data_manager import DataManager
from strategies.strategy import DailyStrategy, MinuteStrategy # 전략 추상 클래스 임포트
from util.strategies_util import get_next_weekday # 유틸리티 함수

logger = logging.getLogger(__name__)

class Trader:
    """
    실전 자동매매 시스템의 메인 실행 클래스.
    일일 및 분봉 전략을 실행하고, 브로커를 통해 실제 거래를 관리합니다.
    """
    def __init__(self, brokerage: Brokerage, data_manager: DataManager,
                 daily_strategy: DailyStrategy, minute_strategy: MinuteStrategy):
        
        self.brokerage = brokerage
        self.data_manager = data_manager
        self.daily_strategy = daily_strategy
        self.minute_strategy = minute_strategy

        self.trading_day_start_time = time(9, 0, 0)
        self.trading_day_end_time = time(15, 20, 0) # 장 마감 10분 전까지 거래
        self.minute_check_interval_seconds = 60 # 1분마다 체크

        # 포트폴리오 가치 추적용
        self.portfolio_values = [] # (datetime, portfolio_value)
        self.last_portfolio_log_time = None

        logger.info("Trader 초기화 완료.")

    def start_trading_day(self, current_date: date):
        """
        거래일 시작 시 필요한 초기화 및 데이터 로드를 수행합니다.
        """
        logger.info(f"\n--- {current_date.isoformat()} 실전 자동매매 시작 ---")
        
        # 1. 증권사 계좌 정보 동기화
        if not self.brokerage.sync_account_info():
            logger.error("계좌 정보 동기화 실패. 거래를 시작할 수 없습니다.")
            sys.exit(1)

        # 2. 손절매 파라미터 설정 (전략 객체에서 가져오거나 직접 설정)
        # 예시: 일봉 전략의 파라미터를 사용
        if hasattr(self.daily_strategy, 'params'):
            self.brokerage.set_stop_loss_params(self.daily_strategy.params)

        # 3. 초기 포트폴리오 가치 설정 (API 조회 값으로)
        # 초기 포지션의 현재가를 가져와야 정확한 계산 가능
        initial_prices = {}
        for code in self.brokerage.positions.keys():
            price = self.data_manager.get_realtime_price(code)
            initial_prices[code] = price if price else self.brokerage.positions[code]['avg_price']
        
        self.brokerage.initial_portfolio_value = self.brokerage.get_portfolio_value(initial_prices)
        logger.info(f"시작 현금: {self.brokerage.cash:,.0f}원, 시작 포트폴리오 가치: {self.brokerage.initial_portfolio_value:,.0f}원")
        
        # 4. 일봉 전략 실행 및 신호 생성
        prev_trading_day = self._get_previous_trading_day(current_date)
        if prev_trading_day:
            logger.info(f"전일({prev_trading_day.isoformat()}) 데이터 기반 일봉 전략 실행 및 신호 생성...")
            all_stock_codes = self.data_manager.get_all_stock_codes()

            daily_ohlcv_data = {}
            # 일봉 전략에 필요한 만큼의 데이터 로드
            lookback_period = self.daily_strategy.params.get('lookback_period', 200) + 50 # 버퍼
            from_date = prev_trading_day - timedelta(days=lookback_period)
            for code in all_stock_codes:
                df = self.data_manager.cache_daily_ohlcv(code, from_date, prev_trading_day)
                if df is not None and not df.empty:
                    daily_ohlcv_data[code] = df
            
            if daily_ohlcv_data:
                self.daily_strategy.data_store = daily_ohlcv_data
                daily_signals = self.daily_strategy.generate_signals(prev_trading_day)
                self.data_manager.save_daily_signals(daily_signals, current_date)
                logger.info(f"일봉 전략 신호 {len(daily_signals)}개 생성 및 저장 완료.")
            else:
                logger.warning("일봉 전략 실행을 위한 데이터가 부족합니다.")

        else:
            logger.warning(f"{current_date.isoformat()}의 이전 거래일을 찾을 수 없습니다. 일봉 전략 신호 생성을 건너뜝니다.")
        
        # 5. 분봉 전략에 일봉 전략 신호 전달
        signals_for_minute_strategy = self.data_manager.get_daily_signals(current_date)
        if signals_for_minute_strategy:
            self.minute_strategy.receive_daily_signals(signals_for_minute_strategy)
            logger.info(f"분봉 전략에 일봉 신호 {len(signals_for_minute_strategy)}개 전달 완료.")

        self.last_portfolio_log_time = datetime.now()
        # self.portfolio_values.append((self.last_portfolio_log_time, self.brokerage.initial_portfolio_value)) # 중복 제거

    def _get_previous_trading_day(self, current_date: date) -> Optional[date]:
        """주어진 날짜의 이전 거래일을 DB에서 조회합니다."""
        trading_days_ts = self.data_manager.db_manager.get_all_trading_days(current_date - timedelta(days=10), current_date)
        if not trading_days_ts:
            return None
        trading_dates = sorted([d.date() for d in trading_days_ts])
        
        try:
            # current_date가 거래일 목록에 있을 경우 그 전날을 반환
            current_date_index = trading_dates.index(current_date)
            if current_date_index > 0:
                return trading_dates[current_date_index - 1]
        except ValueError:
            # current_date가 거래일이 아닐 경우, 그 날짜보다 작은 가장 마지막 거래일을 반환
            trading_dates_before = [d for d in trading_dates if d < current_date]
            if trading_dates_before:
                return max(trading_dates_before)
        
        return None


    def run_trading_session(self, current_date: date):
        """
        실시간 거래 세션을 실행합니다.
        시장 개장 시간 동안 분봉 데이터를 처리하고 매매 로직을 수행합니다.
        """
        logger.info(f"--- {current_date.isoformat()} 실시간 거래 세션 시작 ---")
        
        while True:
            now = datetime.now()
            current_time = now.time()
            
            # 장 마감 후 종료
            if current_time > self.trading_day_end_time:
                logger.info("거래 세션 종료 시간 도달. 실시간 거래를 종료합니다.")
                break

            # 장 시작 전 대기
            if current_time < self.trading_day_start_time:
                wait_seconds = (datetime.combine(current_date, self.trading_day_start_time) - now).total_seconds()
                logger.info(f"장 시작까지 {int(wait_seconds)}초 남음. 1분 후 다시 확인.")
                time_module.sleep(60) 
                continue

            # --- 매 분마다 처리 로직 실행 ---
            
            logger.debug(f"현재 시각: {now.strftime('%H:%M:%S')} - 분봉 전략 및 손절매 조건 확인 시작")

            # 1. 현재 보유 종목의 실시간 현재가 조회
            current_prices = {}
            all_relevant_codes = set(self.brokerage.positions.keys())
            signals = self.data_manager.get_daily_signals(current_date)
            if signals: all_relevant_codes.update(signals.keys())

            for stock_code in all_relevant_codes:
                price = self.data_manager.get_realtime_price(stock_code)
                if price:
                    current_prices[stock_code] = price
            
            # 2. 포트폴리오 손절매 체크 (가장 먼저)
            if self.brokerage.check_and_execute_portfolio_stop_loss(current_prices, now):
                logger.warning("포트폴리오 전체 손절매 조건 충족. 거래 세션을 종료합니다.")
                break 

            # 3. 각 보유 종목 및 신호 종목에 대한 분봉 전략 및 개별 손절매 실행
            for stock_code in all_relevant_codes:
                price_for_check = current_prices.get(stock_code)
                if not price_for_check:
                    logger.debug(f"{stock_code}의 현재가를 조회할 수 없어 이번 틱은 건너뜁니다.")
                    continue

                # 개별 종목 손절매 체크 (보유 종목에 대해서만)
                if stock_code in self.brokerage.positions:
                    if self.brokerage.check_and_execute_stop_loss(stock_code, price_for_check, now):
                        logger.info(f"{stock_code} 개별 손절매 실행 완료.")
                        continue # 손절매 실행 시 해당 종목은 더 이상 분봉 전략 처리 불필요

                # 분봉 전략 실행
                minute_data_df = self.data_manager.cache_minute_ohlcv(stock_code, current_date, current_date)
                if minute_data_df is not None and not minute_data_df.empty:
                    self.minute_strategy.data_store = {stock_code: minute_data_df}
                    # 분봉 전략의 run_minute_logic이 주문 실행을 위해 Brokerage 객체를 직접 호출한다고 가정
                    self.minute_strategy.run_minute_logic(now, stock_code)
                else:
                    logger.warning(f"{stock_code}의 분봉 데이터를 가져올 수 없습니다. 분봉 전략 건너뜀.")
            
            # 4. 현재 포트폴리오 가치 기록
            current_portfolio_value = self.brokerage.get_portfolio_value(current_prices)
            self.portfolio_values.append((now, current_portfolio_value))
            logger.info(f"현재 포트폴리오 가치: {current_portfolio_value:,.0f}원")

            time_module.sleep(self.minute_check_interval_seconds) 

        logger.info(f"--- {current_date.isoformat()} 실시간 거래 세션 종료 ---")


    def end_trading_day(self, end_date: date):
        """
        거래일 종료 시 필요한 마무리 작업을 수행합니다.
        """
        logger.info(f"\n--- {end_date.isoformat()} 실전 자동매매 종료 ---")

        # 1. 최종 포트폴리오 스냅샷 저장
        final_current_prices = {}
        for stock_code in self.brokerage.positions.keys():
            price = self.data_manager.get_realtime_price(stock_code)
            if price:
                final_current_prices[stock_code] = price
            else: # 현재가 조회가 안되면 평단가로 저장
                final_current_prices[stock_code] = self.brokerage.positions[stock_code]['avg_price']

        final_portfolio_value = self.brokerage.get_portfolio_value(final_current_prices)
        self.data_manager.save_daily_portfolio_snapshot(
            snapshot_date=end_date,
            portfolio_value=final_portfolio_value,
            cash=self.brokerage.cash,
            positions=self.brokerage.positions # 백테스터와 호환되는 포맷으로 저장
        )
        logger.info(f"최종 포트폴리오 스냅샷 저장 완료: {end_date}, 가치: {final_portfolio_value:,.0f}원")

        # 2. DB 연결 종료
        self.data_manager.close()
        logger.info("Trader 작업 완료. DB 연결 종료.")

    # Reporter 관련 기능은 백테스팅에서 주로 사용하며, 실전에서는 BusinessManager를 통해 DB에 기록
    # 필요하다면 Reporter와 유사한 보고서 생성 기능을 BusinessManager에 추가하거나 별도 모듈로 분리 가능