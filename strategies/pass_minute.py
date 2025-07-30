# strategies/pass_minute.py

import pandas as pd
from datetime import datetime
from typing import Dict, Any
from strategies.strategy import MinuteStrategy
import logging

logger = logging.getLogger(__name__)

class PassMinute(MinuteStrategy):
    """
    OpenMinute 전략
    - DailyStrategy에서 생성된 target_price를 기반으로 분봉 매매를 실행합니다.
    - target_price가 해당 분봉의 고가와 저가 사이에 들어오면 즉시 주문을 실행합니다.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self.strategy_name = "PassMinute"
       

    def run_minute_logic(self, current_dt: datetime, stock_code: str):
        # 1. 실행할 신호가 있는지 확인
        if stock_code not in self.signals:
            return

        signal_info = self.signals[stock_code]
        order_signal = signal_info.get('signal_type')
        target_price = signal_info.get('target_price')

       
        # 2. 현재 분봉 데이터 가져오기
        daily_bar = self._get_bar_at_time('daily', stock_code, current_dt)
        if daily_bar is None or daily_bar.empty:
            return
        
        current_position_size = self.broker.get_position_size(stock_code)

        # 3. 매매 로직 분기
        # 경우 1: 리밸런싱 매도 (목표가 없음)
        if order_signal == 'sell' and not target_price:
            if current_position_size > 0:
                execution_price = (daily_bar['open'] + daily_bar['close']) / 2 # 시가와 종가의 중간 가격으로 매도
                logging.info(f"✅ [일일 리밸런싱 매도] {stock_code} at {execution_price:,.0f}")
                if self.broker.execute_order(stock_code, 'sell', execution_price, current_position_size, order_time=current_dt) is not None:
                    self.reset_signal(stock_code)

        # 경우 2: 목표가 기반 매매 (목표가 있음)
        elif target_price:
            # 목표가가 당일 가격 범위 안에 있었는지 확인
            if daily_bar['low'] <= target_price <= daily_bar['high']:
                # 매수 실행
                if order_signal == 'buy' and current_position_size == 0:
                    target_quantity = signal_info.get('target_quantity', 0)
                    if target_quantity > 0:
                        if self.broker.execute_order(stock_code, 'buy', target_price, target_quantity, order_time=current_dt) is not None:
                            logging.info(f"✅ [일일 목표가 도달 매수] {stock_code}: Target {target_price:,.0f}")
                            self.reset_signal(stock_code)
                # 매도 실행
                elif order_signal == 'sell' and current_position_size > 0:
                    if self.broker.execute_order(stock_code, 'sell', target_price, current_position_size, order_time=current_dt) is not None:
                        logging.info(f"✅ [일일 목표가 도달 매도] {stock_code}: Target {target_price:,.0f}")
                        self.reset_signal(stock_code)