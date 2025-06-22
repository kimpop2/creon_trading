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
from manager.business_manager import BusinessManager
from strategies.strategy import DailyStrategy, MinuteStrategy # 전략 추상 클래스 임포트
from util.strategies_util import get_next_weekday # 유틸리티 함수

logger = logging.getLogger(__name__)

class Trader:
    """
    실전 자동매매 시스템의 메인 실행 클래스.
    일일 및 분봉 전략을 실행하고, 브로커를 통해 실제 거래를 관리합니다.
    """
    def __init__(self, brokerage: Brokerage, business_manager: BusinessManager,
                 daily_strategy: DailyStrategy, minute_strategy: MinuteStrategy,
                 initial_cash: float):
        
        self.brokerage = brokerage
        self.business_manager = business_manager
        self.daily_strategy = daily_strategy
        self.minute_strategy = minute_strategy
        self.initial_cash = initial_cash # 초기 현금 (실제 계좌 현금과 동기화됨)

        self.trading_day_start_time = time(9, 0, 0)
        self.trading_day_end_time = time(15, 20, 0) # 장 마감 10분 전까지 거래
        self.minute_check_interval_seconds = 60 # 1분마다 체크 (실전에서는 API 호출 제한 고려)

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
            sys.exit(1) # 동기화 실패 시 프로그램 종료 (재시도 로직 추가 가능)

        # 2. 초기 현금 및 포트폴리오 가치 설정 (API 조회 값으로)
        self.brokerage.initial_portfolio_value = self.brokerage.cash + self.brokerage.get_portfolio_value({}) # 초기 현금 + 현재 보유 주식 가치
        logger.info(f"시작 현금: {self.brokerage.cash:,.0f}원, 시작 포트폴리오 가치: {self.brokerage.initial_portfolio_value:,.0f}원")
        
        # 3. 거래 로그 초기화 (일일 기준)
        self.brokerage.reset_daily_transactions()

        # 4. 일봉 전략 실행 및 신호 생성/로드
        # 실전에서는 전날 데이터 기준으로 일봉 신호를 생성하거나, DB에서 로드
        prev_trading_day = self._get_previous_trading_day(current_date)
        if prev_trading_day:
            logger.info(f"전일({prev_trading_day.isoformat()}) 데이터 기반 일봉 전략 실행 및 신호 생성...")
            # 데이터 매니저를 통해 필요한 과거 데이터 로드
            # get_historical_ohlcv는 pandas DataFrame을 반환해야 함
            all_stock_codes = self.business_manager.get_all_stock_codes() # 전체 종목 코드 가져오기

            # 일봉 전략을 위한 데이터 준비 (예: 200일치 일봉 데이터)
            daily_ohlcv_data = {}
            for code in all_stock_codes:
                df = self.business_manager.get_historical_ohlcv(code, 'D', 200) # 200일치 일봉
                if df is not None and not df.empty:
                    daily_ohlcv_data[code] = df
            
            self.daily_strategy.data_store = daily_ohlcv_data # DataManager 대신 DataFrame 딕셔너리 직접 할당
            daily_signals = self.daily_strategy.generate_signals(prev_trading_day) # 전일 데이터 기준으로 신호 생성
            self.business_manager.save_daily_signals(daily_signals, current_date) # 오늘 날짜로 신호 저장
            logger.info(f"일봉 전략 신호 {len(daily_signals)}개 생성 및 저장 완료.")
        else:
            logger.warning(f"{current_date.isoformat()}의 이전 거래일을 찾을 수 없습니다. 일봉 전략 신호 생성을 건너뜝니다.")
        
        # 5. 분봉 전략에 일봉 전략 신호 전달 (BusinessManager를 통해 로드)
        signals_for_minute_strategy = self.business_manager.load_daily_signals_for_today(current_date)
        self.minute_strategy.receive_daily_signals(signals_for_minute_strategy)
        logger.info(f"분봉 전략에 일봉 신호 {len(signals_for_minute_strategy)}개 전달 완료.")

        self.last_portfolio_log_time = datetime.now()
        self.portfolio_values.append((self.last_portfolio_log_time, self.brokerage.initial_portfolio_value))

    def _get_previous_trading_day(self, current_date: date) -> Optional[date]:
        """주어진 날짜의 이전 거래일을 DB에서 조회합니다."""
        # TODO: market_calendar 테이블에서 이전 거래일 조회 로직 구현 필요
        # 현재는 임시로 전날로 가정 (주말/공휴일 미고려)
        trading_calendar_df = self.business_manager.db_manager.fetch_data("SELECT trade_date FROM market_calendar ORDER BY trade_date DESC")
        if not trading_calendar_df.empty:
            trading_dates = sorted([d.date() for d in trading_calendar_df['trade_date'].unique()])
            try:
                current_date_index = trading_dates.index(current_date)
                if current_date_index > 0:
                    return trading_dates[current_date_index - 1]
            except ValueError:
                logger.warning(f"거래일 캘린더에 {current_date.isoformat()}가 없습니다.")
        
        # Fallback for testing or if calendar is empty
        return current_date - timedelta(days=1)


    def run_trading_session(self, current_date: date):
        """
        실시간 거래 세션을 실행합니다.
        시장 개장 시간 동안 분봉 데이터를 처리하고 매매 로직을 수행합니다.
        """
        logger.info(f"--- {current_date.isoformat()} 실시간 거래 세션 시작 ---")
        
        while True:
            now = datetime.now()
            current_time = now.time()
            current_dt = now # datetime 객체로 통일

            # 장 시작 전 대기
            if current_time < self.trading_day_start_time:
                wait_seconds = (datetime.combine(current_date, self.trading_day_start_time) - now).total_seconds()
                if wait_seconds > 0:
                    logger.info(f"장 시작 전 대기 중... {int(wait_seconds)}초 남음")
                    time_module.sleep(min(wait_seconds, 60)) # 최대 1분씩 대기
                continue

            # 장 마감 후 종료
            if current_time > self.trading_day_end_time:
                logger.info("거래 세션 종료 시간 도달. 실시간 거래를 종료합니다.")
                break

            # 매 분마다 처리 로직 실행
            if (now - self.last_portfolio_log_time).total_seconds() >= self.minute_check_interval_seconds:
                self.last_portfolio_log_time = now
                logger.info(f"현재 시각: {current_dt.strftime('%H:%M:%S')} - 분봉 전략 및 손절매 조건 확인 시작")

                # 1. 현재 보유 종목의 실시간 현재가 조회
                current_prices = {}
                for stock_code in self.brokerage.positions.keys():
                    price = self.business_manager.get_realtime_price(stock_code)
                    if price:
                        current_prices[stock_code] = price
                
                # 2. 포트폴리오 손절매 체크 (개별 종목 손절보다 먼저 확인)
                if self.brokerage.check_and_execute_portfolio_stop_loss(current_prices, current_dt):
                    logger.warning("포트폴리오 전체 손절매 조건 충족. 거래 세션을 종료합니다.")
                    break # 전체 손절 시 세션 종료

                # 3. 각 보유 종목 및 신호 종목에 대한 분봉 전략 및 개별 손절매 실행
                stocks_to_check = set(self.brokerage.positions.keys()) # 현재 보유 종목
                # 오늘 일봉 전략에서 받은 신호 종목 (매수 후보)도 포함하여 실시간 데이터 조회
                signals_for_minute_strategy = self.business_manager.load_daily_signals_for_today(current_date)
                stocks_to_check.update(signals_for_minute_strategy.keys())
                
                for stock_code in stocks_to_check:
                    # 개별 종목 손절매 체크 (보유 종목에 대해서만)
                    if stock_code in self.brokerage.positions:
                        price_for_stop_loss = current_prices.get(stock_code)
                        if price_for_stop_loss and self.brokerage.check_and_execute_stop_loss(stock_code, price_for_stop_loss, current_dt):
                            logger.info(f"{stock_code} 개별 손절매 실행 완료.")
                            continue # 손절매 실행 시 해당 종목은 더 이상 분봉 전략 처리 불필요

                    # 분봉 전략 실행
                    minute_data_df = self.business_manager.get_realtime_minute_data(stock_code)
                    if minute_data_df is not None and not minute_data_df.empty:
                        # 분봉 전략의 data_store를 현재 분봉 데이터로 설정
                        self.minute_strategy.data_store = {stock_code: minute_data_df}
                        self.minute_strategy.run_minute_logic(current_dt, stock_code)
                    else:
                        logger.warning(f"{stock_code}의 실시간 분봉 데이터를 가져올 수 없습니다. 분봉 전략 건너뜀.")
                
                # 4. 현재 포트폴리오 가치 기록
                current_portfolio_value = self.brokerage.get_portfolio_value(current_prices)
                self.portfolio_values.append((current_dt, current_portfolio_value))
                logger.info(f"현재 포트폴리오 가치: {current_portfolio_value:,.0f}원")

            time_module.sleep(self.minute_check_interval_seconds) # 다음 분봉 처리까지 대기

        logger.info(f"--- {current_date.isoformat()} 실시간 거래 세션 종료 ---")


    def end_trading_day(self, end_date: date):
        """
        거래일 종료 시 필요한 마무리 작업을 수행합니다.
        """
        logger.info(f"\n--- {end_date.isoformat()} 실전 자동매매 종료 ---")

        # 1. 최종 포트폴리오 스냅샷 저장
        # 마지막 포트폴리오 가치 기록 시점의 정보를 사용
        final_current_prices = {}
        for stock_code in self.brokerage.positions.keys():
            price = self.business_manager.get_realtime_price(stock_code)
            if price:
                final_current_prices[stock_code] = price

        final_portfolio_value = self.brokerage.get_portfolio_value(final_current_prices)
        self.business_manager.save_daily_portfolio_snapshot(
            snapshot_date=end_date,
            portfolio_value=final_portfolio_value,
            cash=self.brokerage.cash,
            positions=self.brokerage.positions
        )
        logger.info(f"최종 포트폴리오 스냅샷 저장 완료: {end_date}, 가치: {final_portfolio_value:,.0f}원")

        # 2. DB 연결 종료
        self.business_manager.close()
        logger.info("Trader 작업 완료. DB 연결 종료.")

    # Reporter 관련 기능은 백테스팅에서 주로 사용하며, 실전에서는 BusinessManager를 통해 DB에 기록
    # 필요하다면 Reporter와 유사한 보고서 생성 기능을 BusinessManager에 추가하거나 별도 모듈로 분리 가능