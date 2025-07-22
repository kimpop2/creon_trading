# strategies/target_price_minute.py

import pandas as pd
from datetime import datetime
from typing import Dict, Any
from strategies.strategy import MinuteStrategy
import logging

logger = logging.getLogger(__name__)

class TargetPriceMinute(MinuteStrategy):
    """
    TargetPriceMinute 전략
    - DailyStrategy에서 생성된 target_price를 기반으로 분봉 매매를 실행합니다.
    - target_price가 해당 분봉의 고가와 저가 사이에 들어오면 즉시 주문을 실행합니다.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self.strategy_name = "TargetPriceMinute"

    def run_minute_logic(self, current_dt: datetime, stock_code: str):
        current_minute_dt = current_dt.replace(second=0, microsecond=0)
        
        # 1. 실행할 신호가 있는지 확인
        if stock_code not in self.signals:
            return

        # [수정] minute_bar를 로직 초반에 한 번만 가져옵니다.
        minute_bar = self._get_bar_at_time('minute', stock_code, current_minute_dt)
        if minute_bar is None or minute_bar.empty:
            logging.info(f"[{stock_code}] 현재 시각({current_minute_dt})에 해당하는 분봉 데이터가 없습니다.")
            return

        signal_info = self.signals[stock_code]
        order_signal = signal_info.get('signal_type')
        target_price = signal_info.get('target_price')

        # --- 리밸런싱 매도 로직 (목표가 없음) ---
        if order_signal == 'sell' and not target_price:
            current_position_size = self.broker.get_position_size(stock_code)
            if current_position_size > 0:
                # 미리 가져온 minute_bar 사용
                current_price = minute_bar['close']
                logging.info(f"✅ [리밸런싱 매도 실행] {stock_code} at {current_price:,.0f}")
                if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, order_time=current_dt) is not None:
                    self.reset_signal(stock_code)
            return # 리밸런싱 매도 처리 후 함수 종료
        
        # --- [핵심 수정] 목표가 기반 매매 로직 (매수/매도 조건 분리) ---
        if order_signal not in ['buy', 'sell'] or not target_price:
            return
        
        current_price = minute_bar['close'] # 체결을 위해 현재 분봉의 종가를 기준으로 삼음
        current_position_size = self.broker.get_position_size(stock_code)

        # 1. 매수 로직: 현재가가 목표가보다 '같거나 낮으면' 즉시 매수
        if order_signal == 'buy' and current_position_size == 0 and current_price <= target_price:
            target_quantity = signal_info.get('target_quantity', 0)
            if target_quantity > 0:
                # 주문은 더 유리한 현재가(current_price)로 실행
                if self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, order_time=current_dt) is not None:
                    logging.info(f"✅ [유리한 가격 매수] {stock_code}: Target {target_price:,.0f}, Executed at {current_price:,.0f}, Qty {target_quantity}")
                    self.reset_signal(stock_code)

        # 2. 매도 로직: 현재가가 목표가보다 '같거나 높으면' 즉시 매도
        elif order_signal == 'sell' and current_position_size > 0 and current_price >= target_price:
            # 주문은 더 유리한 현재가(current_price)로 실행
            if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, order_time=current_dt) is not None:
                logging.info(f"✅ [유리한 가격 매도] {stock_code}: Target {target_price:,.0f}, Executed at {current_price:,.0f}, Qty {current_position_size}")
                self.reset_signal(stock_code)