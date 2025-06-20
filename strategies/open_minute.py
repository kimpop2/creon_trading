"""
OpenMinute 전략
분봉 신호를 생성하지만 다음날 일봉 시가에 매매를 실행하는 전략
최적화 시 분봉 데이터 로딩을 피하여 성능 향상
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Any
from strategies.strategy import MinuteStrategy
from util.strategies_util import calculate_momentum, calculate_rsi

logger = logging.getLogger(__name__)

class OpenMinute(MinuteStrategy):
    """
    OpenMinute 전략
    - 분봉 데이터로 신호를 생성하지만 실제 매매는 다음날 일봉 시가에 실행
    - 최적화 시 분봉 데이터 로딩을 피하여 성능 향상
    - 일봉 데이터로부터 9:00~9:01의 1분봉을 생성하여 분봉 로직 활용
    """
    
    def __init__(self, data_store: Dict, strategy_params: Dict[str, Any], broker):
        super().__init__(data_store, strategy_params, broker)
        self.strategy_name = "OpenMinute"
        self.signals = {}  # DailyStrategy에서 업데이트 받을 시그널 저장
        self.last_portfolio_check = None  # 마지막 포트폴리오 체크 시간
        self.rsi_cache = {}  # RSI 캐시 추가
        self.last_prices = {}  # 마지막 가격 캐시 추가
        self.last_rsi_values = {}  # 마지막 RSI 값 캐시 추가
        # 가상 분봉 범위 캐시 추가
        self.virtual_range_cache = {}
        # 파라미터 검증
        self._validate_parameters()
        
    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = [
            'num_top_stocks',
            'minute_rsi_period',
            'minute_rsi_oversold',
            'minute_rsi_overbought'
        ]
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        logger.info(
            f"OpenMinute 전략 파라미터 검증 완료: "
            f"선택종목수={self.strategy_params['num_top_stocks']}, "
            f"RSI기간={self.strategy_params['minute_rsi_period']}, "
            f"과매도={self.strategy_params['minute_rsi_oversold']}, "
            f"과매수={self.strategy_params['minute_rsi_overbought']}"
        )

    def update_signals(self, signals):
        """
        DailyStrategy에서 생성된 신호들을 업데이트합니다.
        """
        self.signals = {
            stock_code: {**info, 'traded_today': False}
            for stock_code, info in signals.items()
        }

    def _calculate_rsi(self, historical_data, stock_code):
        """
        RSI를 계산하고 캐시합니다.
        캐시된 데이터를 활용하여 효율적으로 계산합니다.
        """
        if len(historical_data) < self.strategy_params['minute_rsi_period'] + 1:
            return None

        # 캐시된 RSI 값이 있고, 새로운 데이터가 1개만 추가된 경우
        if stock_code in self.last_rsi_values and len(historical_data) == len(self.rsi_cache.get(stock_code, [])) + 1:
            last_rsi = self.last_rsi_values[stock_code]
            last_prices = self.rsi_cache[stock_code]
            new_price = historical_data['close'].iloc[-1]
            
            # RSI 업데이트 계산
            price_change = new_price - last_prices[-1]
            if price_change > 0:
                gain = price_change
                loss = 0
            else:
                gain = 0
                loss = -price_change
                
            # 이동평균 업데이트
            avg_gain = (last_rsi['avg_gain'] * (self.strategy_params['minute_rsi_period'] - 1) + gain) / self.strategy_params['minute_rsi_period']
            avg_loss = (last_rsi['avg_loss'] * (self.strategy_params['minute_rsi_period'] - 1) + loss) / self.strategy_params['minute_rsi_period']
            
            # RSI 계산
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # 캐시 업데이트
            self.last_rsi_values[stock_code] = {'avg_gain': avg_gain, 'avg_loss': avg_loss}
            self.rsi_cache[stock_code] = historical_data['close'].values
            self.last_prices[stock_code] = new_price
            
            return rsi
        
        # 전체 RSI 계산 (캐시가 없는 경우)
        delta = historical_data['close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gain = gains.rolling(window=self.strategy_params['minute_rsi_period'], min_periods=1).mean().iloc[-1]
        avg_loss = losses.rolling(window=self.strategy_params['minute_rsi_period'], min_periods=1).mean().iloc[-1]
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # 캐시 업데이트
        self.last_rsi_values[stock_code] = {'avg_gain': avg_gain, 'avg_loss': avg_loss}
        self.rsi_cache[stock_code] = historical_data['close'].values
        self.last_prices[stock_code] = historical_data['close'].iloc[-1]
        
        return rsi

    def _should_check_portfolio(self, current_dt):
        """포트폴리오 체크가 필요한 시점인지 확인합니다."""
        if self.last_portfolio_check is None:
            return True
        
        current_time = current_dt.time()
        # 시간 비교를 정수로 변환하여 효율적으로 비교
        current_minutes = current_time.hour * 60 + current_time.minute
        check_minutes = [9 * 60]  # 9:00 (OpenMinute는 9:00에만 체크)
        
        if current_minutes in check_minutes and (self.last_portfolio_check.date() != current_dt.date() or 
                                               (self.last_portfolio_check.hour * 60 + self.last_portfolio_check.minute) not in check_minutes):
            self.last_portfolio_check = current_dt
            return True
            
        return False

    def _create_minute_bar_from_daily(self, stock_code, current_date):
        """
        일봉 데이터로부터 9:00의 1분봉을 생성합니다.
        """
        try:
            daily_df = self.data_store['daily'].get(stock_code)
            if daily_df is None or daily_df.empty:
                return None
            
            # 해당 날짜의 일봉 데이터 가져오기
            date_data = daily_df[daily_df.index.date == current_date]
            if date_data.empty:
                return None
            
            daily_bar = date_data.iloc[-1]
            
            # 9:00의 1분봉 생성 (시고저종 모두 동일)
            minute_time = datetime.combine(current_date, time(9, 0))
            minute_bar = pd.DataFrame({
                'open': [daily_bar['open']],
                'high': [daily_bar['high']],
                'low': [daily_bar['low']],
                'close': [daily_bar['close']],
                'volume': [daily_bar['volume']]
            }, index=[minute_time])
            
            return minute_bar
            
        except Exception as e:
            logger.error(f"분봉 생성 오류 ({stock_code}): {str(e)}")
            return None

    def _get_historical_minute_data(self, stock_code, current_date, lookback_period):
        """
        과거 분봉 데이터를 생성합니다 (일봉 데이터 기반).
        """
        try:
            daily_df = self.data_store['daily'].get(stock_code)
            if daily_df is None or daily_df.empty:
                return pd.DataFrame()
            
            # 현재 날짜까지의 일봉 데이터 가져오기
            historical_daily = daily_df[daily_df.index.date <= current_date].tail(lookback_period)
            
            if historical_daily.empty:
                return pd.DataFrame()
            
            # 각 일봉을 9:00 분봉으로 변환
            minute_bars = []
            for date, daily_bar in historical_daily.iterrows():
                minute_time = datetime.combine(date.date(), time(9, 0))
                minute_bar = pd.DataFrame({
                    'open': [daily_bar['open']],
                    'high': [daily_bar['high']],
                    'low': [daily_bar['low']],
                    'close': [daily_bar['close']],
                    'volume': [daily_bar['volume']]
                }, index=[minute_time])
                minute_bars.append(minute_bar)
            
            if minute_bars:
                return pd.concat(minute_bars)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"과거 분봉 데이터 생성 오류 ({stock_code}): {str(e)}")
            return pd.DataFrame()

    def run_minute_logic(self, current_dt, stock_code): # current_dt 현재날짜시각
        """
        OpenMinute 전략: 분봉 데이터 없이 일봉만으로 RSI 매매/강제매매 시뮬레이션
        하루에 1번만 매매가 가능하도록 traded_today 체크를 강화
        """
        # 조건문 단축: 신호 없거나 이미 거래했거나, 신호가 buy/sell이 아니면 바로 return
        if (
            stock_code not in self.signals or
            self.signals[stock_code].get('traded_today', False) or
            self.signals[stock_code].get('signal') not in ['buy', 'sell']
        ):
            return
        signal_info = self.signals[stock_code]
        momentum_signal = signal_info.get('signal')
        # 캐시 키
        cache_key = (stock_code, current_dt.date())
        if cache_key in self.virtual_range_cache:
            virtual_high, virtual_low = self.virtual_range_cache[cache_key]
        else:
            # 가상 분봉 범위 계산 (기존 코드)
            daily_df = self.data_store['daily'].get(stock_code)
            if daily_df is None or daily_df.empty:
                return
            date_data = daily_df[daily_df.index.date == current_dt.date()]
            if date_data.empty:
                return
            current_price = date_data['open'].iloc[0]
            prev_data = daily_df[daily_df.index.date < current_dt.date()]
            if prev_data.empty:
                return
            prev_close = prev_data['close'].iloc[-1]
            N = 5
            virtual_range_ratio = 0.3
            if len(prev_data) < N:
                return
            recent_highs = prev_data['high'].iloc[-N:]
            recent_lows = prev_data['low'].iloc[-N:]
            avg_range = (recent_highs - recent_lows).mean()
            virtual_range = avg_range * virtual_range_ratio
            virtual_high = prev_close + virtual_range
            virtual_low = prev_close - virtual_range
            self.virtual_range_cache[cache_key] = (virtual_high, virtual_low)

        # 매일 루프 시작 시 캐시 초기화 필요 (벡터화/메모리 관리)
        self.reset_virtual_range_cache()
        # 보유종목 수
        current_position_size = self.broker.get_position_size(stock_code)
        # 목표가 (전일 종가 기준)
        target_price = signal_info.get('target_price')
        # 오늘 일봉
        daily_df = self.data_store['daily'].get(stock_code)
        if daily_df is None or daily_df.empty:
            return
        date_data = daily_df[daily_df.index.date == current_dt.date()]
        if date_data.empty:
            return
        
        # 시가==현재가
        current_price = date_data['open'].iloc[0]
        today_high = date_data['high'].iloc[0]
        today_low = date_data['low'].iloc[0]
        
        # 전일 종가 필요
        prev_data = daily_df[daily_df.index.date < current_dt.date()]
        if prev_data.empty:
            return
        prev_close = prev_data['close'].iloc[-1]
        
        # 1. 손절매 (모든 보유 종목) - 하루에 1번만 실행
        if current_position_size > 0:
            stop_loss_ratio = self.broker.stop_loss_params.get('stop_loss_ratio', 0.05)
            trailing_stop_ratio = self.broker.stop_loss_params.get('trailing_stop_ratio', 0.02)

            position_info = self.broker.positions.get(stock_code, {})
            avg_price = position_info.get('avg_price', 0)
            fixed_stop_price = avg_price * (1 - stop_loss_ratio) if avg_price > 0 else 0
            highest_price = position_info.get('highest_price', current_price)
            trailing_stop_price = highest_price * (1 - trailing_stop_ratio)

            if current_price <= trailing_stop_price:
                logging.info(f'[트레일링 매도] {current_dt.isoformat()} - {stock_code} 현재가: {current_price:,.0f}원, 트레일링손절: {trailing_stop_price:,.0f}원, 수량: {current_position_size}주')
                if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt):
                    self.reset_signal(stock_code)
                
            elif current_price <= fixed_stop_price and fixed_stop_price > 0:
                logging.info(f'[손절매] {current_dt.isoformat()} - {stock_code} 현재가: {current_price:,.0f}원, 고정손절: {fixed_stop_price:,.0f}원, 수량: {current_position_size}주')
                if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt):
                    self.reset_signal(stock_code)
                
        # 2. 신호가 있는 종목만 RSI/강제매매 - 하루에 1번만 실행
        if stock_code not in self.signals:
            return
        
        # signal_info = self.signals[stock_code]
        momentum_signal = signal_info.get('signal')
        if momentum_signal not in ['buy', 'sell']:
            return
        
        # 추가 체크: 이미 거래했는지 다시 확인
        if signal_info.get('traded_today', False):
            logging.debug(f"[{current_dt.isoformat()}] {stock_code}: 이미 오늘 거래 완료. 신호 매매 건너뜀.")
            return
            
        # 3. RSI 매매 시뮬레이션 (가상 고가/저가 사용)
        if momentum_signal == 'buy' and current_position_size == 0:
            if virtual_low <= target_price <= virtual_high:
                target_quantity = signal_info.get('target_quantity', 0)
                
                if target_quantity > 0:
                    logging.info(f'[RSI 매수] {current_dt.isoformat()} - {stock_code} 가상저가: {virtual_low:,.0f}원, 시가: {current_price:,.0f}원, 수량: {target_quantity}주')
                    if self.broker.execute_order(stock_code, 'buy', virtual_low, target_quantity, current_dt):
                        self.reset_signal(stock_code)
            else:
                logging.info(f'[매수 취소] {current_dt.isoformat()} - {stock_code} 목표가: {target_price} {virtual_low:,.0f}~{virtual_high:,.0f}원 범위 이탈 (사유: 현금 부족/범위 이탈 등)')

        if momentum_signal == 'sell' and current_position_size > 0:
            if virtual_low <= target_price <= virtual_high:
                #target_quantity = signal_info.get('target_quantity', 0)
                logging.info(f'[RSI 매도] {current_dt.isoformat()} - {stock_code} 가상고가: {virtual_high:,.0f}원, 시가: {current_price:,.0f}원, 수량: {current_position_size}주')
                if self.broker.execute_order(stock_code, 'sell', virtual_high, current_position_size, current_dt):
                    self.reset_signal(stock_code)
            else:
                logging.info(f'[매도 취소] {current_dt.isoformat()} - {stock_code} 목표가: {target_price} {virtual_low:,.0f}~{virtual_high:,.0f}원 범위 이탈 (사유: 범위 이탈 등)')

    def reset_virtual_range_cache(self):
        self.virtual_range_cache = {} 