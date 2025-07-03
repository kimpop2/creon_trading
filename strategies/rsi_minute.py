import datetime
import logging
import pandas as pd
import numpy as np 
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from trade.trader import Trader
from strategies.strategy import MinuteStrategy 
from util.strategies_util import calculate_momentum, calculate_rsi 

logger = logging.getLogger(__name__)

class RSIMinute(MinuteStrategy): 
    def __init__(self, trade:Trader, strategy_params: Dict[str, Any]):
        super().__init__(trade, strategy_params)
        
        self._validate_strategy_params() # 전략 파라미터 검증

        self.rsi_cache = {}  # RSI 캐시 추가
        self.last_prices = {}  # 마지막 가격 캐시 추가
        self.last_rsi_values = {}  # 마지막 RSI 값 캐시 추가

        self.strategy_name = "RSIMinute"

    # 필수 : 파라미터 검증
    def _validate_strategy_params(self):
        """전략 파라미터의 유효성을 검증합니다."""
        required_params = ['minute_rsi_period', 'minute_rsi_oversold', 'minute_rsi_overbought', 'num_top_stocks']
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"RSI 분봉봉 전략에 필요한 파라미터 '{param}'이 설정되지 않았습니다.")
        
        # 과매수 과매도 점수 검증
        if self.strategy_params['minute_rsi_overbought'] <= self.strategy_params['minute_rsi_oversold']:
            raise ValueError("과매수 overbought 점수 는 과매도 점수 oversold 보다 커야 합니다.")
        
        # 파라미터 로그
        logging.info(f"RSI 분봉 전략 파라미터 검증 완료: RSI 분봉 기간={self.strategy_params['minute_rsi_period']}, "
                    f"과매수 점수={self.strategy_params['minute_rsi_oversold']}, "
                    f"과매도 점수={self.strategy_params['minute_rsi_overbought']} "
                    f"선택종목수={self.strategy_params['num_top_stocks']}")

    # # 필수 : 부모 클래스에 올릴 지 판다
    # def update_signals(self, signals):
    #     """
    #     DailyStrategy에서 생성된 신호들을 업데이트합니다.
    #     """
    #     self.signals = {
    #         stock_code: {**info, 'traded_today': False}
    #         for stock_code, info in signals.items()
    #     }

    # 필수 : 지표계산
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


    # 필수 : 반드시 구현
    def run_minute_logic(self, current_dt, stock_code):
        """
        분봉 데이터를 기반으로 RSI 매수/매도 로직을 실행합니다.
        """
        if stock_code not in self.signals:
            return
        
        signal_info = self.signals[stock_code]
        order_signal = signal_info.get('signal')
        
        # 당일 이미 거래가 발생했으면 추가 거래 방지
        if signal_info.get('traded_today', False):
            return

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

        # # ########################
        # # 종목 손절
        # # ########################
        current_position_size = self.broker.get_position_size(stock_code)
        if self.broker.stop_loss_params is not None and current_position_size > 0:
            if self.broker.check_and_execute_stop_loss(stock_code, current_price, current_dt):
                self.reset_signal(stock_code)

        # ########################
        # 매수 매도 공통
        # ########################
        target_range = 0.2 # 타임컷         

        current_rsi_value = self._calculate_rsi(historical_minute_data, stock_code)
        if current_rsi_value is None:
            return        

        # ##########################
        # 종목 매수
        # --------------------------
        # 
        # ##########################
        if order_signal == 'buy' and current_position_size == 0:
            target_quantity = signal_info.get('target_quantity', 0)
            if target_quantity <= 0:
                return

            # ########################
            # 지표를 이용한 매수 : RSI
            # ########################

            if current_rsi_value <= self.strategy_params['minute_rsi_oversold']:
                logging.info(f'[RSI 매수] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
                self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
                self.reset_signal(stock_code)
            
            # ########################
            # 강제매수 : 타임컷 
            # ########################
            if current_minutes == 15 * 60 + 5: # 오후 3:05 강제매수 가능
                today_data = minute_df[minute_df.index == pd.Timestamp(current_dt)]
                if today_data.empty:
                    return
                
                k = 0.3 # 가격 제한 범위
                today_high = today_data['high'].max() 
                today_low = today_data['low'].min()
                today_move = today_data['high'].max() - today_data['low'].min() # 변동폭
                high = today_high - today_move * target_range # 오늘 고가 k%  야래
                low = today_low + today_move * target_range   # 오늘 저가 k% 위
                # 목표가
                target_price = self.signals[stock_code].get('target_price', current_price)
                # 가격 제한 범위 내 목표가와 현재가 있어 매수가능
                if low <= target_price <= high and current_price <= high:
                    logging.info(f'[타임컷 강제매수] {current_dt.isoformat()} - {stock_code} 목표가: {target_price}, 매수가: {current_price}, 매수가능 저가~고가: {low}~{high}')
                    self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
                    self.reset_signal(stock_code)
                else:
                    logging.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 목표가: {target_price}, 매수가: {current_price}, 매수가능 저가~고가: {low}~{high}')

        # ##########################
        # 종목 매도
        # --------------------------
        # 
        # ##########################
        elif order_signal == 'sell' and current_position_size > 0:
            
            # ########################
            # 지표를 이용한 매도 : RSI
            # ########################
            if current_rsi_value >= self.strategy_params['minute_rsi_overbought']:
                logging.info(f'[RSI 매도] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 매도가: {current_price:,.0f}원, 수량: {current_position_size}주')
                self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
                self.reset_signal(stock_code)
            
            # ########################
            # 강제매도 : 타임컷 
            # ########################
            if current_minutes == 15 * 60 + 5: # 오후 3:05 강제매도 가능 
                today_data = minute_df[minute_df.index == pd.Timestamp(current_dt)]
                if today_data.empty:
                    return
                
                # 가격 제한 범위
                today_high = today_data['high'].max() 
                today_low = today_data['low'].min()
                today_move = today_data['high'].max() - today_data['low'].min()
                high = today_high - today_move * target_range
                low = today_low + today_move * target_range
                # 목표가
                target_price = self.signals[stock_code].get('target_price', current_price)
                # 가격 제한 범위 내 목표가와 현재가 있어 매수가능
                if low <= target_price <= high and low <= current_price:
                    logging.info(f'[타임컷 강제매도] {current_dt.isoformat()} - {stock_code} 목표가: {target_price}, 매도가: {current_price}, 오늘 저가~고가: {low}~{high}')
                    self.broker.execute_order(stock_code, 'sell', target_price, current_position_size, current_dt)
                    self.reset_signal(stock_code)
                else:
                    logging.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 목표가: {target_price}, 매도가: {current_price}, 오늘 저가~고가: {low}~{high}')
                

