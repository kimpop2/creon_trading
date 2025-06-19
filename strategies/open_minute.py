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
        
        # 파라미터 검증
        self._validate_parameters()
        
    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = ['num_top_stocks']  # RSI 파라미터 제거
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        
        logger.info(f"OpenMinute 전략 파라미터 검증 완료: "
                   f"선택종목수={self.strategy_params['num_top_stocks']}")

    def update_signals(self, signals):
        """
        DailyStrategy에서 생성된 신호들을 업데이트합니다.
        """
        self.signals = {
            stock_code: {**info, 'traded_today': False}
            for stock_code, info in signals.items()
        }
        logging.debug(f"OpenMinute: {len(self.signals)}개의 시그널로 업데이트 완료. 첫 종목: {next(iter(self.signals), 'N/A')}")
        
        # 디버그: 신호 정보 상세 출력
        for stock_code, signal_info in self.signals.items():
            if signal_info.get('signal') in ['buy', 'sell']:
                logging.debug(f"OpenMinute 신호 업데이트 - {stock_code}: signal={signal_info.get('signal')}, target_quantity={signal_info.get('target_quantity', 'N/A')}")

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

    def _calculate_daily_rsi(self, stock_code, current_date):
        """
        일봉 데이터로 RSI를 계산합니다.
        """
        try:
            # 과거 일봉 데이터로 RSI 계산
            historical_daily = self._get_historical_data_up_to('daily', stock_code, current_date, lookback_period=14)
            
            if len(historical_daily) < 14:
                return None
            
            # RSI 계산
            delta = historical_daily['close'].diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            avg_gain = gains.rolling(window=14, min_periods=1).mean().iloc[-1]
            avg_loss = losses.rolling(window=14, min_periods=1).mean().iloc[-1]
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"일봉 RSI 계산 오류 ({stock_code}): {str(e)}")
            return None

    def _estimate_rsi_threshold_prices(self, prev_close, today_high, today_low, rsi_period=14, overbought=70, oversold=30):
        """
        전일 종가와 오늘 고가/저가를 이용해 RSI 과매수/과매도 가격을 근사적으로 계산합니다.
        (단순히 오늘 변동폭이 모두 상승/하락이라고 가정)
        """
        # 상승만 일어났을 때의 RSI
        gain = today_high - prev_close if today_high > prev_close else 0
        loss = prev_close - today_low if today_low < prev_close else 0
        # 평균상승/하락폭을 단일 변화로 근사
        avg_gain = gain / rsi_period
        avg_loss = loss / rsi_period
        # 과매수 가격: RSI=overbought가 되는 가격
        if avg_loss == 0:
            price_overbought = today_high
        else:
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))
            # 역산 불가, 그냥 고가 사용
            price_overbought = today_high
        # 과매도 가격: RSI=oversold가 되는 가격
        if avg_gain == 0:
            price_oversold = today_low
        else:
            rs = avg_gain / avg_loss if avg_loss > 0 else 0.01
            rsi = 100 - (100 / (1 + rs))
            # 역산 불가, 그냥 저가 사용
            price_oversold = today_low
        return price_overbought, price_oversold

    def run_minute_logic(self, current_dt, stock_code):
        """
        OpenMinute 전략: 분봉 데이터 없이 일봉만으로 RSI 매매/강제매매 시뮬레이션
        """
        current_minutes = current_dt.hour * 60 + current_dt.minute
        if current_minutes != 9 * 60 + 1:
            return
        current_position_size = self.broker.get_position_size(stock_code)
        daily_df = self.data_store['daily'].get(stock_code)
        if daily_df is None or daily_df.empty:
            return
        date_data = daily_df[daily_df.index.date == current_dt.date()]
        if date_data.empty:
            return
        current_price = date_data['open'].iloc[0]
        today_high = date_data['high'].iloc[0]
        today_low = date_data['low'].iloc[0]
        # 전일 종가 필요
        prev_data = daily_df[daily_df.index.date < current_dt.date()]
        if prev_data.empty:
            return
        prev_close = prev_data['close'].iloc[-1]
        # 최근 5일 변동폭 평균 계산 및 가상 고가/저가 비율 상수화
        N = 5
        virtual_range_ratio = 0.3  # 상수로 고정
        if len(prev_data) < N:
            return  # 데이터 부족 시 매매하지 않음
        recent_highs = prev_data['high'].iloc[-N:]
        recent_lows = prev_data['low'].iloc[-N:]
        avg_range = (recent_highs - recent_lows).mean()
        virtual_range = avg_range * virtual_range_ratio
        virtual_high = current_price + virtual_range
        virtual_low = current_price - virtual_range

        # 1. 손절매 (모든 보유 종목)
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
                    return
            elif current_price <= fixed_stop_price and fixed_stop_price > 0:
                logging.info(f'[손절매] {current_dt.isoformat()} - {stock_code} 현재가: {current_price:,.0f}원, 고정손절: {fixed_stop_price:,.0f}원, 수량: {current_position_size}주')
                if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt):
                    self.reset_signal(stock_code)
                    return
        # 2. 신호가 있는 종목만 RSI/강제매매
        if stock_code not in self.signals:
            return
        signal_info = self.signals[stock_code]
        momentum_signal = signal_info.get('signal')
        if momentum_signal not in ['buy', 'sell']:
            return
        if signal_info.get('traded_today', False):
            return
        # 3. RSI 매매 시뮬레이션 (가상 고가/저가 사용)
        if momentum_signal == 'buy' and current_position_size == 0:
            if virtual_low <= current_price <= virtual_high:
                target_quantity = signal_info.get('target_quantity', 0)
                if target_quantity > 0:
                    logging.info(f'[RSI 매수] {current_dt.isoformat()} - {stock_code} 가상저가: {virtual_low:,.0f}원, 시가: {current_price:,.0f}원, 수량: {target_quantity}주')
                    if self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt):
                        self.reset_signal(stock_code)
                        return
            else:
                if virtual_low > current_price or virtual_high < current_price:
                    logging.info(f'[미체결][매수 연기] {current_dt.isoformat()} - {stock_code} 목표가: {virtual_low:,.0f}~{virtual_high:,.0f}원 범위 이탈 (사유: 현금 부족/범위 이탈 등)')
        # 9:01에 신호 상태 로그 남기기
        if stock_code in self.signals:
            signal_info = self.signals[stock_code]
            logging.debug(f"[시그널체크] {current_dt.isoformat()} - {stock_code} signal={signal_info.get('signal')}, qty={signal_info.get('target_quantity')}, traded_today={signal_info.get('traded_today')}, position={current_position_size}")
        else:
            logging.debug(f"[시그널체크] {current_dt.isoformat()} - {stock_code} 신호 없음, position={current_position_size}") 