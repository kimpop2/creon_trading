import datetime
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
        
        self._validate_strategy_params() # 전략 파라미터 검증

        self.rsi_cache = {}  # RSI 캐시 추가
        self.last_prices = {}  # 마지막 가격 캐시 추가
        self.last_rsi_values = {}  # 마지막 RSI 값 캐시 추가
        # 가상 분봉 범위 캐시 추가
        self.virtual_range_cache = {}
        self.strategy_name = "OpenMinute"

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

    def reset_virtual_range_cache(self):
        """
        가상 범위 캐시를 초기화합니다.
        매일 새로운 거래일 시작 시 호출되어야 합니다.
        """
        self.virtual_range_cache = {}
        logger.debug("OpenMinute: 가상 범위 캐시가 초기화되었습니다.")


    def run_minute_logic(self, current_dt, stock_code): # current_dt 현재날짜시각
        """
        OpenMinute 전략: 분봉 데이터 없이 일봉만으로 RSI 매매/강제매매 시뮬레이션
        하루에 1번만 매매가 가능하도록 traded_today 체크를 강화
        """
        logger.debug(f"[{current_dt.isoformat()}] {stock_code}: run_minute_logic 호출됨.")

        # 조건문 단축: 신호 없거나 이미 거래했거나, 신호가 buy/sell이 아니면 바로 return
        if stock_code not in self.signals:
            logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 시그널 없음. 매매 건너뜀.")
            return
        
        signal_info = self.signals[stock_code]
        order_signal = signal_info.get('signal')

        # 당일 이미 거래가 발생했으면 추가 거래 방지
        if signal_info.get('traded_today', False):
            logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 이미 오늘 거래 완료 (traded_today=True). 매매 건너뜀.")
            return

        if order_signal not in ['buy', 'sell']:
            logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 매수/매도 신호 아님 ({order_signal}). 매매 건너뜀.")
            return
        
        logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 시그널 정보: {signal_info}")

        # 캐시 키
        cache_key = (stock_code, current_dt.date())
        virtual_high, virtual_low = None, None

        if cache_key in self.virtual_range_cache:
            virtual_high, virtual_low = self.virtual_range_cache[cache_key]
            logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 가상 범위 캐시에서 로드. High: {virtual_high:,.0f}, Low: {virtual_low:,.0f}")
        else:
            # 가상 분봉 범위 계산
            daily_df = self.data_store['daily'].get(stock_code)
            if daily_df is None or daily_df.empty:
                logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 일봉 데이터 없음. 가상 범위 계산 불가.")
                return
            
            date_data = daily_df[daily_df.index.date == current_dt.date()]
            if date_data.empty:
                logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 오늘 날짜 일봉 데이터 없음. 가상 범위 계산 불가.")
                return
            
            today_open = date_data['open'].iloc[0]
            
            prev_data = daily_df[daily_df.index.date < current_dt.date()]
            if prev_data.empty:
                logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 전일 일봉 데이터 없음. 가상 범위 계산 불가.")
                return
            
            prev_close = prev_data['close'].iloc[-1]
            
            N = 5 # 가상 범위 계산에 사용할 과거 데이터 기간
            virtual_range_ratio = 0.3 # 가상 범위 비율
            
            if len(prev_data) < N:
                logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 가상 범위 계산을 위한 과거 데이터 부족 ({len(prev_data)}개). 최소 {N}개 필요.")
                return
            
            recent_highs = prev_data['high'].iloc[-N:]
            recent_lows = prev_data['low'].iloc[-N:]
            avg_range = (recent_highs - recent_lows).mean()
            
            virtual_range = avg_range * virtual_range_ratio
            virtual_high = prev_close + virtual_range
            virtual_low = prev_close - virtual_range
            
            self.virtual_range_cache[cache_key] = (virtual_high, virtual_low)
            logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 가상 범위 계산 완료. High: {virtual_high:,.0f}, Low: {virtual_low:,.0f}")

        # 보유종목 수
        current_position_size = self.broker.get_position_size(stock_code)
        # 목표가 (전일 종가 기준) - OpenMinute은 시가 매매이므로 target_price는 참고용
        target_price = signal_info.get('target_price') 
        
        # 오늘 일봉 데이터 (시가, 고가, 저가, 종가)
        daily_df_today = self.data_store['daily'].get(stock_code)
        if daily_df_today is None or daily_df_today.empty:
            logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 오늘 일봉 데이터 없음. 매매 건너뜀.")
            return
        date_data_today = daily_df_today[daily_df_today.index.date == current_dt.date()]
        if date_data_today.empty:
            logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 오늘 날짜 일봉 데이터 (date_data_today) 없음. 매매 건너뜀.")
            return
        
        today_open = date_data_today['open'].iloc[0]
        today_high = date_data_today['high'].iloc[0]
        today_low = date_data_today['low'].iloc[0]
        today_close = date_data_today['close'].iloc[0]
        logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 오늘 시가: {today_open:,.0f}, 고가: {today_high:,.0f}, 저가: {today_low:,.0f}, 종가: {today_close:,.0f}")
        
        # 1. 손절매 (모든 보유 종목) - 하루에 1번만 실행
        if self.broker.stop_loss_params is not None and current_position_size > 0:
            stop_loss_ratio = self.broker.stop_loss_params.get('stop_loss_ratio', 0.05)
            trailing_stop_ratio = self.broker.stop_loss_params.get('trailing_stop_ratio', 0.02)

            position_info = self.broker.positions.get(stock_code, {})
            avg_price = position_info.get('avg_price', 0)
            fixed_stop_price = avg_price * (1 - stop_loss_ratio) if avg_price > 0 else 0
            
            # 최고가 업데이트 (매일매일의 최고가 반영)
            # 백테스트에서는 일봉의 high를 사용하거나, 실제 트레이딩에서는 실시간 고가를 사용해야 함
            # 여기서는 일봉의 high를 최고가로 가정
            highest_price_recorded = position_info.get('highest_price', 0)
            if today_high > highest_price_recorded:
                self.broker.positions[stock_code]['highest_price'] = today_high # 브로커 포지션 정보 업데이트
                highest_price_recorded = today_high

            trailing_stop_price = highest_price_recorded * (1 - trailing_stop_ratio)
            
            logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 손절매 체크. 보유량: {current_position_size}, 평단: {avg_price:,.0f}, 최고가: {highest_price_recorded:,.0f}, 고정손절: {fixed_stop_price:,.0f}, 트레일링손절: {trailing_stop_price:,.0f}")

            # 트레일링 스탑 또는 고정 손절매 조건 확인 (시가 기준)
            if today_open <= trailing_stop_price and trailing_stop_price > 0: # 시가가 트레일링 스탑 가격 이하일 때
                logging.info(f'[트레일링 매도] {current_dt.isoformat()} - {stock_code} 시가: {today_open:,.0f}원 <= 트레일링 매도: {trailing_stop_price:,.0f}원, 수량: {current_position_size}주')
                if self.broker.execute_order(stock_code, 'sell', today_open, current_position_size, current_dt):
                    self.signals[stock_code]['traded_today'] = True # 오늘 거래 완료 표시
                    self.reset_signal(stock_code) # 매도 후 시그널 초기화
                return # 매도했으면 더 이상 매수 시도 안함
            
            if today_open <= fixed_stop_price and fixed_stop_price > 0: # 시가가 고정 손절매 가격 이하일 때
                logging.info(f'[손절매] {current_dt.isoformat()} - {stock_code} 시가: {today_open:,.0f}원 <= 손절: {fixed_stop_price:,.0f}원, 수량: {current_position_size}주')
                if self.broker.execute_order(stock_code, 'sell', today_open, current_position_size, current_dt):
                    self.signals[stock_code]['traded_today'] = True # 오늘 거래 완료 표시
                    self.reset_signal(stock_code) # 매도 후 시그널 초기화
                return # 매도했으면 더 이상 매수 시도 안함
                
        # 2. 신호에 따른 매매 시뮬레이션 (RSI 매매)
        # 이미 traded_today 플래그와 order_signal 유효성 검사는 위에서 했으므로 다시 할 필요 없음
        # ##########################
        # 2. 종목 매도
        # --------------------------
        if order_signal == 'sell' and current_position_size > 0:
            # 매도 전략 로직 시작 : (자유롭게 작성) ==============================
            # RSI 매도 로직 설명 :
            #
            # ----------------------------------------------------------------
            if virtual_low <= today_open <= virtual_high:
                logging.info(f'[RSI 매도] {current_dt.isoformat()} - {stock_code} 시가: {today_open:,.0f}원, 가상범위: [{virtual_low:,.0f}~{virtual_high:,.0f}], 수량: {current_position_size}주')
                if self.broker.execute_order(stock_code, 'sell', today_open, current_position_size, current_dt):
                    self.signals[stock_code]['traded_today'] = True # 오늘 거래 완료 표시
                    self.reset_signal(stock_code) # 매도 후 시그널 초기화
                    logger.info(f"[{current_dt.isoformat()}] {stock_code}: 매도 주문 성공.")
                else:
                    logger.warning(f"[{current_dt.isoformat()}] {stock_code}: 매도 주문 실패 (브로커 오류).")
            else:
                logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 시가 {today_open:,.0f}원이 가상 범위 [{virtual_low:,.0f}~{virtual_high:,.0f}] 이탈. 매도 취소.")
            # 끝 매도 전략 로직 =================================================

        # ##########################
        # 3. 종목 매수
        # --------------------------
        if order_signal == 'buy' and current_position_size == 0:
            # 매수 전략 로직 시작 : (자유롭게 작성) ==============================
            # RSI 매수 로직 설명 :
            #
            # ----------------------------------------------------------------            logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 매수 시그널 확인. 가상 범위: [{virtual_low:,.0f}~{virtual_high:,.0f}], 오늘 시가: {today_open:,.0f}")
            # 매수 조건: 시가가 가상 매수 범위 내에 있고, 목표 수량이 0보다 클 때
            if virtual_low <= today_open <= virtual_high:
                target_quantity = signal_info.get('target_quantity', 0)
                logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 매수 조건 충족 (시가 범위 내). 목표 수량: {target_quantity}주")
                
                if target_quantity > 0:
                    logging.info(f'[RSI 매수] {current_dt.isoformat()} - {stock_code} 시가: {today_open:,.0f}원, 가상범위: [{virtual_low:,.0f}~{virtual_high:,.0f}], 수량: {target_quantity}주')
                    if self.broker.execute_order(stock_code, 'buy', today_open, target_quantity, current_dt):
                        self.signals[stock_code]['traded_today'] = True # 오늘 거래 완료 표시
                        logger.info(f"[{current_dt.isoformat()}] {stock_code}: 매수 주문 성공.")
                    else:
                        logger.warning(f"[{current_dt.isoformat()}] {stock_code}: 매수 주문 실패 (브로커 오류 또는 현금 부족).")
                else:
                    logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 목표 수량 0. 매수 건너뜀.")
            else:
                logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 시가 {today_open:,.0f}원이 가상 범위 [{virtual_low:,.0f}~{virtual_high:,.0f}] 이탈. 매수 취소.")
            # 끝 매수 전략 로직 =================================================

