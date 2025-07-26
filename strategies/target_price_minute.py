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
        
        if stock_code not in self.signals:
            return

        minute_bar = self._get_bar_at_time('minute', stock_code, current_minute_dt)
        if minute_bar is None or minute_bar.empty:
            return

        signal_info = self.signals[stock_code]
        order_signal = signal_info.get('signal_type')
        target_price = signal_info.get('target_price')
        
        # [핵심 수정] 신호에서 '실행 전술'을 가져옵니다.
        tactic = signal_info.get('execution_type', 'touch') # 기본값은 'touch'

        current_price = minute_bar['close']
        current_position_size = self.broker.get_position_size(stock_code)

        # --- 1. 전술(tactic)에 따른 매매 조건 설정 ---
        buy_condition_met = False
        sell_condition_met = False

        if order_signal == 'buy' and current_position_size == 0:
            if tactic == 'breakout':    # 돌파 매수
                buy_condition_met = (current_price >= target_price)
            elif tactic == 'pullback':  # 눌림목 매수
                buy_condition_met = (current_price <= target_price)
            elif tactic == 'touch':     # 일반 지정가 (기본값)
                buy_condition_met = (minute_bar['low'] <= target_price <= minute_bar['high'])
        
        elif order_signal == 'sell' and current_position_size > 0:
            if tactic == 'market':      # 리밸런싱 등 즉시 매도
                sell_condition_met = True
            elif target_price:
                # 'breakout'은 상향 돌파를 의미하므로, 매도 시에는 '하향 돌파(breakdown)'으로 해석
                if tactic in ['breakout', 'breakdown']: 
                    sell_condition_met = (current_price <= target_price)
                elif tactic == 'pullback': # 반등 시 매도
                    sell_condition_met = (current_price >= target_price)
                elif tactic == 'touch':
                    sell_condition_met = (minute_bar['low'] <= target_price <= minute_bar['high'])

        stock_name = self.get_stock_name(stock_code)
        target_price_str = f"{target_price:,.0f}" if target_price else "N/A"
        # 1. 매수 주문 실행
        if buy_condition_met:
            target_quantity = signal_info.get('target_quantity', 0)
            if target_quantity > 0:
                if self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, order_time=current_dt) is not None:
                    # [수정] 로그 형식에 종목명 추가
                    logging.info(f"✅ [매수 실행 ({tactic})] {stock_name}({stock_code}): Target {target_price_str}, Executed at {current_price:,.0f}")
                    self.reset_signal(stock_code)
        
        # 2. 매도 주문 실행
        elif sell_condition_met:
            price_to_sell = current_price
            if self.broker.execute_order(stock_code, 'sell', price_to_sell, current_position_size, order_time=current_dt) is not None:
                # [수정] 로그 형식에 종목명 추가
                logging.info(f"✅ [매도 실행 ({tactic})] {stock_name}({stock_code}): Target {target_price_str}, Executed at {price_to_sell:,.0f}")
                self.reset_signal(stock_code)