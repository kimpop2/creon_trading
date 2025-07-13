import datetime
import logging
import pandas as pd
import numpy as np 
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import sys
import os
# 프로젝트 루트 경로를 sys.path에 추가 (manager 디렉토리에서 실행 가능하도록)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strategies.strategy import MinuteStrategy
from util.strategies_util import *

logger = logging.getLogger(__name__)
# 디버깅을 위해 DEBUG 레벨로 설정
#logger.setLevel(logging.DEBUG)
class RSIMinute(MinuteStrategy): 
    """
    RSIMinute 전략
    - 
    - 
    - 
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        
        self._validate_strategy_params() # 전략 파라미터 검증

        self.rsi_cache = {}  # RSI 캐시 추가
        self.last_prices = {}  # 마지막 가격 캐시 추가
        self.last_rsi_values = {}  # 마지막 RSI 값 캐시 추가

        self.strategy_name = "RSIMinute"

    def _validate_strategy_params(self):
        """전략 파라미터의 유효성을 검증합니다."""
        required_params = ['minute_rsi_period', 'minute_rsi_oversold', 'minute_rsi_overbought']
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"RSI 분봉봉 전략에 필요한 파라미터 '{param}'이 설정되지 않았습니다.")
        
        # 과매수 과매도 점수 검증
        if self.strategy_params['minute_rsi_overbought'] <= self.strategy_params['minute_rsi_oversold']:
            raise ValueError("과매수 overbought 점수 는 과매도 점수 oversold 보다 커야 합니다.")
        
        # 파라미터 로그
        logging.info(f"RSI 분봉 전략 파라미터 검증 완료: "
                    f"RSI 분봉 기간={self.strategy_params['minute_rsi_period']}, "
                    f"과매수 점수={self.strategy_params['minute_rsi_oversold']}, "
                    f"과매도 점수={self.strategy_params['minute_rsi_overbought']} ")


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

    # 필수 : 반드시 구현
    def run_minute_logic(self, current_dt, stock_code):
        """
        분봉 데이터를 기반으로 RSI 매수/매도 로직을 실행합니다.

        """
        if stock_code not in self.signals:
            logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 시그널 없음. 매매 건너뜀.")
            return
        
        signal_info = self.signals[stock_code]
        order_signal = signal_info.get('signal')
        
        # 당일 이미 거래가 발생했으면 추가 거래 방지
        if signal_info.get('traded_today', False):
            logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 이미 오늘 거래 완료 (traded_today=True). 매매 건너뜀.")
            return

        logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 시그널 정보: {signal_info}")
 
        # 현재 시간의 분봉 데이터 가져오기 (한 번만 조회)
        minute_df = self._get_bar_at_time('minute', stock_code, current_dt)
        if minute_df is None or minute_df.empty:
            return
        
        current_price = minute_df['close']
        current_minute_time = current_dt.time()
        current_minutes = current_minute_time.hour * 60 + current_minute_time.minute

        # RSI 계산 (캐시 활용)
        historical_minute_data = self._get_historical_data_up_to(
            'minute',
            stock_code,
            current_dt,
            lookback_period=self.strategy_params['minute_rsi_period'] + 1
        )
        
        # 1. 손절 체크 ->broker 손절 처리
        current_position_size = self.broker.get_position_size(stock_code)
        if self.broker.stop_loss_params is not None and current_position_size > 0:
            if self.broker.check_and_execute_stop_loss(stock_code, current_price, current_dt):
                self.reset_signal(stock_code)
 

        # ##########################
        # 2. 종목 매도
        # --------------------------
        if order_signal == 'sell' and current_position_size > 0:
            # 매도 전략 로직 시작 : (자유롭게 작성) ==============================
            # RSI 매도 로직 설명 :
            #
            # ----------------------------------------------------------------
            # 2-1. RSI 지표 계산 : 매수와 중복되지만 코드를 보기 쉽게 중복 함
            current_rsi_value = self._calculate_rsi(historical_minute_data, stock_code)
            if current_rsi_value is None:
                return        

            if current_rsi_value >= self.strategy_params['minute_rsi_overbought']:
                logging.info(f'[RSI 매도] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 매도가: {current_price:,.0f}원, 수량: {current_position_size}주')
                self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
                self.reset_signal(stock_code)
            # 끝 매도 전략 로직 =================================================
            
            # 2-2. 강제매도 (타임컷) 전략에 강제매도가 필요하다면 다음 형식으로 추가
            if current_minutes >= (15 * 60 + 5) and current_minutes >= (15 * 60 + 20): # 오후 3:05 이후 강제매도 가능 (시간 범위 확장)
                today_data = minute_df[minute_df.index == pd.Timestamp(current_dt)]
                if today_data.empty:
                    return
                
                # 타임컷 강제매도 실행 (괴리율 조건 완화)
                self.execute_time_cut_sell(stock_code, current_dt, current_price, current_position_size, max_price_diff_ratio=0.7)
            # 끝 강제매도도 -------------------------------------------------------    

        # ##########################
        # 3. 종목 매수
        # --------------------------
        if order_signal == 'buy' and current_position_size == 0:
            target_quantity = signal_info.get('target_quantity', 0)
            if target_quantity <= 0:
                return

            # 매수 전략 로직 시작 : (자유롭게 작성) ==============================
            # RSI 매수 로직 설명 :
            #
            # ----------------------------------------------------------------
            # 3-1. RSI 지표 계산
            current_rsi_value = self._calculate_rsi(historical_minute_data, stock_code)
            if current_rsi_value is None:
                return        
    
            if current_rsi_value <= self.strategy_params['minute_rsi_oversold']:
                logging.info(f'[RSI 매수] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
                self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
                self.reset_signal(stock_code)
            
            # 끝 매수 전략 로직 =================================================
            
            # 3-2. 강제매수 (타임컷) 수정금지 -----------------------------------------
            if current_minutes >= (15 * 60 + 5) and current_minutes >= (15 * 60 + 20): # 오후 3:05 이후 강제매수 가능 (시간 범위 확장)
                today_data = minute_df[minute_df.index == pd.Timestamp(current_dt)]
                if today_data.empty:
                    return
                # 타임컷 강제매수 실행 (괴리율 조건 완화)
                self.execute_time_cut_buy(stock_code, current_dt, current_price, target_quantity, max_price_diff_ratio=0.7)
            # 끝 강제매수 -------------------------------------------------------

