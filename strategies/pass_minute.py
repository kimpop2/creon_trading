# strategies/pass_minute.py

import pandas as pd
from datetime import datetime
from typing import Dict, Any
from strategies.strategy import MinuteStrategy
import logging

logger = logging.getLogger(__name__)

class PassMinute(MinuteStrategy):
    """
    [개선된 PassMinute 전략]
    - 분봉 데이터 없이 일봉 데이터(고가, 저가)만으로 매매를 시뮬레이션합니다.
    - 진입, 리밸런싱 매도, 손절, 익절, 트레일링 스탑을 모두 지원합니다.
    """
    def __init__(self, broker, data_store):
        super().__init__(broker, data_store)
        self._validate_strategy_params()
        self.is_fast_simulation_strategy = True

    def _validate_strategy_params(self):
        """전략 파라미터의 유효성을 검증합니다."""
        pass

    def run_minute_logic(self, current_dt: datetime, stock_code: str):
        """
        하루 동안의 모든 매매를 일봉 데이터로 시뮬레이션하는 핵심 메서드.
        backtest.py의 '초고속 모드'에서 하루에 한 번 호출됩니다.
        """
        # 1. 필요한 데이터 가져오기
        signal_info = self.signals.get(stock_code)
        daily_bar = self._get_bar_at_time('daily', stock_code, current_dt)
        
        if daily_bar is None or daily_bar.empty:
            return

        current_position_size = self.broker.get_position_size(stock_code)
        
        # 2. 보유 종목에 대한 매도/청산 로직 (손절 우선)
        if current_position_size > 0:
            position_info = self.broker.positions[stock_code]
            # 매수한 날짜와 현재 날짜가 같으면 매도 로직을 건너뜀
            
            
            # 2-1. 손절/익절/트레일링 가격 확인
            stop_loss_price = self.broker.get_stop_loss_price(stock_code)
            take_profit_price = self.broker.get_take_profit_price(stock_code)
            trailing_stop_price = self.broker.get_trailing_stop_price(stock_code, daily_bar['high'])

            # 2-2. 매도 우선순위: 손절 > 트레일링 스탑 > 익절 > 리밸런싱 매도 신호
            # (중요) 하나의 이벤트만 발생했다고 가정하고 시뮬레이션
            # 손절
            if stop_loss_price and daily_bar['low'] <= stop_loss_price:
                entry_dt = position_info.get('entry_date')
                if entry_dt and entry_dt == current_dt.date():
                    return # 또는 다른 로직으로 진입 방지                
                
                logging.info(f"📉 [PassMinute-손절] {stock_code} at {stop_loss_price:,.0f}")
                self.broker.execute_order(stock_code, 'sell', stop_loss_price, current_position_size, order_time=current_dt)
                self.reset_signal(stock_code)
                return # 매도 체결 후 당일 추가 거래 없음
            
            # 트레일링 스탑
            if trailing_stop_price and daily_bar['low'] <= trailing_stop_price:
                logging.info(f"📈 [PassMinute-트레일링] {stock_code} at {trailing_stop_price:,.0f}")
                self.broker.execute_order(stock_code, 'sell', trailing_stop_price, current_position_size, order_time=current_dt)
                self.reset_signal(stock_code)
                return

            # 익절
            if take_profit_price and daily_bar['high'] >= take_profit_price:
                logging.info(f"💰 [PassMinute-익절] {stock_code} at {take_profit_price:,.0f}")
                self.broker.execute_order(stock_code, 'sell', take_profit_price, current_position_size, order_time=current_dt)
                self.reset_signal(stock_code)
                return
            
            # 리밸런싱 매도 신호 (목표가 없음)
            if signal_info and signal_info.get('signal_type') == 'sell' and not signal_info.get('target_price'):
                # 시장가 매도이므로 시가에 체결되었다고 가정
                execution_price = daily_bar['open']
                logging.info(f"📉 [PassMinute-리밸런싱 매도] {stock_code} at {execution_price:,.0f}")
                self.broker.execute_order(stock_code, 'sell', execution_price, current_position_size, order_time=current_dt)
                self.reset_signal(stock_code)
                return

        # 3. 신규 매수 로직
        if signal_info and signal_info.get('signal_type') == 'buy' and current_position_size == 0:
            # --- ▼ [수정] 목표가 조건 삭제 및 시가 매수 로직으로 변경 ▼ ---
            execution_price = daily_bar['open'] # 다음 날 시가를 실행 가격으로 설정
            target_quantity = signal_info.get('target_quantity', 0)
            
            if target_quantity > 0:
                logging.info(f"✅ [PassMinute-시가 매수] {stock_code}: at {execution_price:,.0f}")
                self.broker.execute_order(stock_code, 'buy', execution_price, target_quantity, order_time=current_dt)
                self.reset_signal(stock_code)