# strategies/backtest_minute.py

import pandas as pd
from datetime import datetime
from typing import Dict, Any
from strategies.strategy import MinuteStrategy
import logging

logger = logging.getLogger(__name__)

class BacktestMinute(MinuteStrategy):
    """
    BacktestMinute 전략
    - DailyStrategy에서 생성된 target_price를 기반으로 분봉 매매를 실행합니다.
    - target_price가 해당 분봉의 고가와 저가 사이에 들어오면 즉시 주문을 실행합니다.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self.strategy_name = "BacktestMinute"

    def run_minute_logic(self, current_dt: datetime, stock_code: str):
        # 1. 실행할 신호가 있는지 확인
        if stock_code not in self.signals:
            return

        signal_info = self.signals[stock_code]
        order_signal = signal_info.get('signal_type')
        target_price = signal_info.get('target_price')

        # --- [신규] 리밸런싱 매도 로직 추가 ---
        # 경우 1: 리밸런싱 매도 (목표가가 없음)
        if order_signal == 'sell' and not target_price:
            current_position_size = self.broker.get_position_size(stock_code)
            if current_position_size > 0:
                minute_bar = self._get_bar_at_time('minute', stock_code, current_dt)
                if minute_bar is None or minute_bar.empty: return
                
                current_price = minute_bar['close'] # 해당 분봉의 종가로 즉시 매도
                logging.info(f"✅ [리밸런싱 매도 실행] {stock_code} at {current_price:,.0f}")
                if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, order_time=current_dt):
                    self.reset_signal(stock_code)
            return # 리밸런싱 매도 처리 후 함수 종료
        # --- 로직 추가 끝 ---

        # --- [수정] 기존 로직은 '목표가가 있는 경우'에만 실행되도록 변경 ---
        # 유효한 매수/매도 신호와 '목표가'가 모두 있어야 함
        if order_signal not in ['buy', 'sell'] or not target_price:
            return
        
        # 2. 현재 분봉 데이터 가져오기
        minute_bar = self._get_bar_at_time('minute', stock_code, current_dt)
        if minute_bar is None or minute_bar.empty:
            return

        # 3. 매매 조건 확인: 목표가가 현재 분봉의 저가와 고가 사이에 있는가?
        if minute_bar['low'] <= target_price <= minute_bar['high']:
            current_position_size = self.broker.get_position_size(stock_code)

            # 4. 매수 실행
            if order_signal == 'buy' and current_position_size == 0:
                target_quantity = signal_info.get('target_quantity', 0)
                if target_quantity > 0:
                    # 주문 실행 (실제 체결은 목표가로 되었다고 가정)
                    if self.broker.execute_order(stock_code, 'buy', target_price, target_quantity, order_time=current_dt):
                        logging.info(f"✅ [목표가 도달 매수] {stock_code}: Target {target_price:,.0f} / Quantity {target_quantity}")
                        self.reset_signal(stock_code) # 반복 주문 방지를 위해 신호 제거

            # 5. 매도 실행
            elif order_signal == 'sell' and current_position_size > 0:
                # 주문 실행 (실제 체결은 목표가로 되었다고 가정)
                if self.broker.execute_order(stock_code, 'sell', target_price, current_position_size, order_time=current_dt):
                    logging.info(f"✅ [목표가 도달 매도] {stock_code}: Target {target_price:,.0f} / Quantity {current_position_size}")
                    self.reset_signal(stock_code) # 반복 주문 방지를 위해 신호 제거