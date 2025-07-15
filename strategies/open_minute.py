"""
OpenMinute 전략
분봉 신호를 생성하지만 다음날 일봉 시가에 매매를 실행하는 전략
최적화 시 분봉 데이터 로딩을 피하여 성능 향상
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from strategies.strategy import MinuteStrategy
from util.strategies_util import *

logger = logging.getLogger(__name__)
# 디버깅을 위해 DEBUG 레벨로 설정
#logger.setLevel(logging.DEBUG)

class OpenMinute(MinuteStrategy):
    """
    OpenMinute 전략
    - 분봉 데이터로 신호를 생성하지만 실제 매매는 다음날 일봉 시가에 실행
    - 최적화 시 분봉 데이터 로딩을 피하여 성능 향상
    - 일봉 데이터로부터 9:00~9:01의 1분봉을 생성하여 분봉 로직 활용
    """
    
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self.strategy_name = "OpenMinute"
        
        # 전략 파라미터 검증
        self._validate_strategy_params()

        # RSI 캐시 추가
        self.rsi_cache = {}  # RSI 캐시 추가
        self.last_prices = {}  # 마지막 가격 캐시 추가
        self.last_rsi_values = {}  # 마지막 RSI 값 캐시 추가
        # 가상 분봉 범위 캐시 추가
        self.virtual_range_cache = {}
        
    def _validate_strategy_params(self):
        """전략 파라미터 검증"""
        required_params = ['num_top_stocks','minute_rsi_period','minute_rsi_oversold','minute_rsi_overbought']
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

    def update_signals(self, new_signals: Dict[str, Any]):
        """
        Trader로부터 새로운 시그널을 받아 self.signals를 업데이트합니다.
        """
        self.signals = new_signals
        logger.debug(f"OpenMinute: {len(new_signals)}개의 시그널 업데이트 받음. (buy: {sum(1 for s in new_signals.values() if s.get('signal') == 'buy')})")


    def _calculate_rsi(self, historical_data, stock_code):
        """
        RSI를 계산하고 캐시합니다.
        캐시된 데이터를 활용하여 효율적으로 계산합니다.
        """
        if len(historical_data) < self.strategy_params['minute_rsi_period'] + 1:
            logger.debug(f"[{stock_code}] RSI 계산을 위한 데이터 부족: {len(historical_data)}개, 최소 {self.strategy_params['minute_rsi_period'] + 1}개 필요.")
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
            
            logger.debug(f"[{stock_code}] RSI 업데이트 계산: {rsi:.2f}")
            return rsi
        
        # 전체 RSI 계산 (캐시가 없는 경우 또는 데이터가 많이 추가된 경우)
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
        
        logger.debug(f"[{stock_code}] 전체 RSI 계산: {rsi:.2f}")
        return rsi

    def _create_minute_bar_from_daily(self, stock_code, current_date):
        """
        일봉 데이터로부터 9:00의 1분봉을 생성합니다.
        """
        try:
            daily_df = self.data_store['daily'].get(stock_code)
            if daily_df is None or daily_df.empty:
                logger.debug(f"[{stock_code}] 분봉 생성 오류: 일봉 데이터 없음.")
                return None
            
            # 해당 날짜의 일봉 데이터 가져오기
            # 인덱스를 datetime.date 객체로 비교
            date_data = daily_df[daily_df.index.date == current_date]
            if date_data.empty:
                logger.debug(f"[{stock_code}] 분봉 생성 오류: {current_date} 날짜의 일봉 데이터 없음.")
                return None
            
            daily_bar = date_data.iloc[-1] # 해당 날짜의 마지막(유일한) 일봉 데이터
            
            # 9:00의 1분봉 생성 (시고저종 모두 동일)
            minute_time = datetime.combine(current_date, datetime.time(9, 0))
            minute_bar = pd.DataFrame({
                'open': [daily_bar['open']],
                'high': [daily_bar['high']],
                'low': [daily_bar['low']],
                'close': [daily_bar['close']],
                'volume': [daily_bar['volume']]
            }, index=[minute_time])
            
            logger.debug(f"[{stock_code}] {current_date} 일봉 기반 9:00 분봉 생성 완료.")
            return minute_bar
            
        except Exception as e:
            logger.error(f"분봉 생성 오류 ({stock_code}, {current_date}): {str(e)}", exc_info=True)
            return None

    def _get_historical_minute_data(self, stock_code, current_date, lookback_period):
        """
        과거 분봉 데이터를 생성합니다 (일봉 데이터 기반).
        """
        try:
            daily_df = self.data_store['daily'].get(stock_code)
            if daily_df is None or daily_df.empty:
                logger.debug(f"[{stock_code}] 과거 분봉 데이터 생성 오류: 일봉 데이터 없음.")
                return pd.DataFrame()
            
            # 현재 날짜까지의 일봉 데이터 가져오기
            # lookback_period는 일봉 개수이므로, tail() 사용
            historical_daily = daily_df[daily_df.index.date <= current_date].tail(lookback_period)
            
            if historical_daily.empty:
                logger.debug(f"[{stock_code}] 과거 분봉 데이터 생성 오류: {current_date} 이전 {lookback_period}일간 데이터 없음.")
                return pd.DataFrame()
            
            # 각 일봉을 9:00 분봉으로 변환
            minute_bars = []
            for date_idx, daily_bar in historical_daily.iterrows():
                minute_time = datetime.combine(date_idx.date(), datetime.time(9, 0))
                minute_bar = pd.DataFrame({
                    'open': [daily_bar['open']],
                    'high': [daily_bar['high']],
                    'low': [daily_bar['low']],
                    'close': [daily_bar['close']],
                    'volume': [daily_bar['volume']]
                }, index=[minute_time])
                minute_bars.append(minute_bar)
            
            if minute_bars:
                concatenated_df = pd.concat(minute_bars)
                logger.debug(f"[{stock_code}] 과거 분봉 데이터 {len(concatenated_df)}개 생성 완료.")
                return concatenated_df
            else:
                logger.debug(f"[{stock_code}] 과거 분봉 데이터 생성 실패: minute_bars가 비어있음.")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"과거 분봉 데이터 생성 오류 ({stock_code}, {current_date}): {str(e)}", exc_info=True)
            return pd.DataFrame()

    def run_minute_logic(self, current_dt: datetime, stock_code: str):
        """
        [수정됨] 일봉 전략의 신호를 받아 즉시 시장가로 주문을 실행합니다.
        """
        logger.debug(f"[{current_dt.isoformat()}] {stock_code}: OpenMinute.run_minute_logic 호출됨.")

        # 1. 신호 유효성 검사
        if stock_code not in self.signals:
            logger.debug(f"[{stock_code}]: 시그널 없음. 건너뜀.")
            return
        
        signal_info = self.signals[stock_code]
        order_signal = signal_info.get('signal_type')

        if signal_info.get('traded_today', False):
            logger.debug(f"[{stock_code}]: 이미 오늘 거래 완료. 건너뜀.")
            return

        if order_signal not in ['buy', 'sell']:
            logger.debug(f"[{stock_code}]: 매수/매도 신호 아님({order_signal}). 건너뜀.")
            return

        # 2. 신호에 따른 즉시 시장가 주문 실행
        current_position_size = self.broker.get_position_size(stock_code)
        
        # 2-1. 매수 신호 처리
        if order_signal == 'buy' and current_position_size == 0:
            target_quantity = signal_info.get('target_quantity', 0)
            if target_quantity <= 0:
                logger.warning(f"[{stock_code}]: 매수 신호는 있으나 목표 수량이 0 이하입니다.")
                return

            logger.info(f"✅ [시장가 매수 실행] {stock_code} / 수량: {target_quantity}주")
            
            # 시장가(price=0)로 즉시 주문 실행
            if self.broker.execute_order(stock_code, 'buy', 0, target_quantity, current_dt):
                self.signals[stock_code]['traded_today'] = True
                logger.info(f"[{stock_code}]: 시장가 매수 주문 요청 성공.")
            else:
                logger.error(f"[{stock_code}]: 시장가 매수 주문 요청 실패.")

        # 2-2. 매도 신호 처리
        elif order_signal == 'sell' and current_position_size > 0:
            logger.info(f"✅ [시장가 매도 실행] {stock_code} / 수량: {current_position_size}주")

            # 시장가(price=0)로 즉시 주문 실행
            if self.broker.execute_order(stock_code, 'sell', 0, current_position_size, current_dt):
                self.signals[stock_code]['traded_today'] = True
                self.reset_signal(stock_code) # 매도 후 신호 정리
                logger.info(f"[{stock_code}]: 시장가 매도 주문 요청 성공.")
            else:
                logger.error(f"[{stock_code}]: 시장가 매도 주문 요청 실패.")

    def reset_signal(self, stock_code: str):
        """
        특정 종목의 시그널을 초기화합니다.
        매매가 성공적으로 이루어진 후 호출되어 다음 매매 기회를 기다리게 합니다.
        """
        if stock_code in self.signals:
            # 매매가 이루어진 후에는 해당 종목의 시그널을 완전히 제거하는 것이 아니라,
            # 'traded_today' 플래그만 True로 설정하여 당일 추가 매매를 막는 것이 OpenMinute의 의도와 맞을 수 있습니다.
            # 다음 날이 되면 Trader에서 전체 signals 딕셔너리를 초기화할 것이므로.
            # 여기서는 매매 성공 시 시그널을 완전히 삭제하는 기존 로직을 유지합니다.
            del self.signals[stock_code]
        logger.debug(f"종목 {stock_code}의 시그널이 초기화되었습니다.")

    def reset_virtual_range_cache(self):
        """
        가상 범위 캐시를 초기화합니다.
        매일 새로운 거래일 시작 시 호출되어야 합니다.
        """
        self.virtual_range_cache = {}
        logger.debug("OpenMinute: 가상 범위 캐시가 초기화되었습니다.")

    # DailyStrategy가 OpenMinute의 signals 딕셔너리를 업데이트할 수 있도록 하는 메서드 (선택 사항)
    # Trader가 update_signals를 호출하므로 이 메서드는 OpenMinute 자체에는 필수는 아님.
    # 하지만 DailyStrategy에서 직접 OpenMinute의 signals를 조작하는 경우를 위해 남겨둘 수 있음.
    # def set_signals(self, signals: Dict[str, Any]):
    #     self.signals = signals
    #     logger.debug("OpenMinute의 시그널이 업데이트되었습니다.")
