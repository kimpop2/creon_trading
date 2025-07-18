import pandas as pd
import numpy as np 
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from strategies.strategy import MinuteStrategy
from util.strategies_util import *
import logging

logger = logging.getLogger(__name__)
# 디버깅을 위해 DEBUG 레벨로 설정
#logger.setLevel(logging.DEBUG)
class RSIMinute(MinuteStrategy): 
    """
    RSIMinute 전략
    - RSI 기반 분봉 매매 전략
    - 과매수/과매도 구간에서 매매 신호 생성
    - 타임컷 강제매매 기능 포함
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self.strategy_name = "RSIMinute"
        
        # 전략 파라미터 검증
        self._validate_strategy_params()

        # RSI 캐시 추가
        self.rsi_cache = {}  # RSI 캐시 추가
        self.last_prices = {}  # 마지막 가격 캐시 추가
        self.last_rsi_values = {}  # 마지막 RSI 값 캐시 추가

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

    # 분봉 매매 대상 : 매수신호, 매도신호, 보유 중 종목
    # 비교) 일봉 신호 대상은 모든 유니버스 종목
    def run_minute_logic(self, current_dt, stock_code):
        # 시그널 없음 건너 뜀
        if stock_code not in self.signals: return
        
        # 종목에 대한 신호
        signal_info = self.signals[stock_code]
        order_signal = signal_info.get('signal_type')
        
        if order_signal not in ['buy', 'sell']: return

        minute_df = self._get_bar_at_time('minute', stock_code, current_dt) 
        if minute_df is None or minute_df.empty: return
        
        current_price = minute_df['close']
        current_position_size = self.broker.get_position_size(stock_code)
        
        # 1. RSI 및 이격도 계산에 필요한 데이터 준비
        historical_data = self._get_historical_data_up_to('minute', stock_code, current_dt, self.strategy_params['minute_rsi_period'] + 1)
        current_rsi = self._calculate_rsi(historical_data, stock_code)
        if current_rsi is None: return
        
        max_deviation_ratio = self.strategy_params['max_deviation_ratio'] / 100
        
        # --- [수정] 매수 로직 (이격도 조건 복원) ---
        if order_signal == 'buy' and current_position_size == 0:
            target_quantity = signal_info.get('target_quantity', 0)
            target_price = signal_info.get('target_price')
            if target_quantity <= 0 or not target_price: return

            deviation = abs(current_price - target_price) / target_price
            
            # 이격도와 RSI 조건을 모두 만족할 때 매수
            if deviation <= max_deviation_ratio and current_rsi <= self.strategy_params['minute_rsi_oversold']:
                if self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, order_time=current_dt) is not None:
                    logging.info(f"✅ [RSI 매수 실행] {stock_code} (RSI: {current_rsi:.2f}, 가격 괴리율: {deviation:.2%})")
                    self.reset_signal(stock_code)

        # --- [수정] 매도 로직 (이격도 조건 복원) ---
        elif order_signal == 'sell' and current_position_size > 0:
            target_price = signal_info.get('target_price')
            
            # 경우 1: 리밸런싱 매도 (목표가가 없는 경우, RSI 조건만 확인)
            if not target_price and current_rsi >= self.strategy_params['minute_rsi_overbought']:
                if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, order_time=current_dt) is not None:
                    logging.info(f"✅ [RSI 리밸런싱 매도] {stock_code} (RSI: {current_rsi:.2f})")
                    self.reset_signal(stock_code)
            
            # 경우 2: 데드크로스 등 목표가가 있는 매도 (이격도와 RSI 조건 모두 확인)
            elif target_price:
                deviation = abs(current_price - target_price) / target_price
                if deviation <= max_deviation_ratio and current_rsi >= self.strategy_params['minute_rsi_overbought']:
                    if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, order_time=current_dt) is not None:
                        logging.info(f"✅ [RSI 목표가 매도] {stock_code} (RSI: {current_rsi:.2f}, 가격 괴리율: {deviation:.2%})")
                        self.reset_signal(stock_code)

        # --- 타임컷 로직은 필요시 여기에 추가 ---
        
        # # 매수 로직
        # if order_signal == 'buy' and current_position_size == 0:
        #     target_quantity = signal_info.get('target_quantity', 0)
        #     target_price = signal_info.get('target_price')
        #     if target_quantity <= 0 or not target_price: return

        #     deviation = abs(current_price - target_price) / target_price
        #     if deviation <= max_deviation_ratio and current_rsi <= self.strategy_params['minute_rsi_oversold']:
        #         logging.info(f"✅ [RSI 매수 실행] {stock_code} (RSI: {current_rsi:.2f}, 가격 괴리율: {deviation:.2%})")
        #         if self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, order_time=current_dt) is not None:
        #             self.reset_signal(stock_code)

        # # 매도 로직
        # elif order_signal == 'sell' and current_position_size > 0:
        #     target_price = signal_info.get('target_price')
        #     # 리밸런싱 매도 (목표가 없음)
        #     if not target_price and current_rsi >= self.strategy_params['minute_rsi_overbought']:
        #         logging.info(f"✅ [RSI 리밸런싱 매도] {stock_code} (RSI: {current_rsi:.2f})")
        #         if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, order_time=current_dt) is not None:
        #             self.reset_signal(stock_code)
        #     # 데드크로스 매도 (목표가 있음)
        #     elif target_price:
        #         deviation = abs(current_price - target_price) / target_price
        #         if deviation <= max_deviation_ratio and current_rsi >= self.strategy_params['minute_rsi_overbought']:
        #             logging.info(f"✅ [RSI 데드크로스 매도] {stock_code} (RSI: {current_rsi:.2f})")
        #             if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, order_time=current_dt) is not None:
        #                 self.reset_signal(stock_code)
        
        # --- [복원] 타임컷 (Time-cut) 강제 매매 로직 ---
        # 장 마감 부근(15:10 이후)이고, 아직 오늘 거래가 실행되지 않았다면 강제 실행
        # if not self.signals[stock_code].get('traded_today', False) and current_dt.time() >= datetime.time(15, 10):
        #     if order_signal == 'buy' and current_position_size == 0:
        #         target_quantity = signal_info.get('target_quantity', 0)
        #         self.execute_time_cut_buy(stock_code, current_dt, current_price, target_quantity, max_deviation_ratio=self.strategy_params['max_deviation_ratio']) # 타임컷 괴리율은 좀 더 여유있게
            
        #     elif order_signal == 'sell' and current_position_size > 0:
        #         self.execute_time_cut_sell(stock_code, current_dt, current_price, current_position_size, max_deviation_ratio=self.strategy_params['max_deviation_ratio'])