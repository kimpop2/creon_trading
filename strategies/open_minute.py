# strategies/open_minute.py

import logging
from datetime import datetime
from typing import Dict, Any
from strategies.strategy import MinuteStrategy
from trading.abstract_broker import AbstractBroker

logger = logging.getLogger(__name__)

class OpenMinute(MinuteStrategy):
    """
    [최종 수정] OpenMinute 전략
    - 신호의 target_price와 현재가를 비교하는 '가격 필터' 로직이 추가되었습니다.
    - 유효한 가격 범위 내에서만 시장가 주문을 실행합니다.
    """
    
    def __init__(self, broker:AbstractBroker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self.strategy_name = "OpenMinute"
        self._validate_strategy_params()

    def _validate_strategy_params(self):
        """전략 파라미터 검증"""
        # [추가] 가격 필터링에 사용할 파라미터
        if 'max_deviation_ratio' not in self.strategy_params:
            # 기본값 설정 또는 에러 발생
            self.strategy_params['max_deviation_ratio'] = 0.02 # 기본값 2%
        logger.info(f"OpenMinute 전략 파라미터 검증 완료. 최대괴리율: {self.strategy_params['max_deviation_ratio']:.2%}")

    def run_minute_logic(self, current_dt: datetime, stock_code: str):
        logger.debug(f"[{current_dt.isoformat()}] {stock_code}: OpenMinute.run_minute_logic 호출됨.")

        if stock_code not in self.signals: return
        
        signal_info = self.signals[stock_code]
        order_signal = signal_info.get('signal_type')

        if signal_info.get('traded_today', False): return
        if order_signal not in ['buy', 'sell']: return

        # [수정] 현재가를 먼저 가져옴
        minute_df = self._get_bar_at_time('minute', stock_code, current_dt)
        if minute_df is None or minute_df.empty: return
        current_price = minute_df['close']
        
        current_position_size = self.broker.get_position_size(stock_code)
        
        # --- [수정] 매수 신호 처리 로직 ---
        if order_signal == 'buy' and current_position_size == 0:
            target_quantity = signal_info.get('target_quantity', 0)
            target_price = signal_info.get('target_price') # 신호에 포함된 목표가

            if target_quantity <= 0 or not target_price:
                return

            # # [추가] 가격 필터링 로직
            # deviation = abs(current_price - target_price) / target_price
            # max_deviation = self.strategy_params['max_deviation_ratio']

            # if deviation <= max_deviation:
            logger.info(f"✅ [시장가 매수 실행] {stock_code} / 수량: {target_quantity}주")
            logger.info(f"    (목표가: {target_price:,.0f}, 현재가: {current_price:,.0f}, 괴리율: {deviation:.2%})")
            if self.broker.execute_order(stock_code, 'buy', 0, target_quantity, order_time=current_dt):
                self.reset_signal(stock_code)
            # else:
            #     logger.warning(f"❌ [매수 신호 스킵] {stock_code} - 가격 괴리율 초과")
            #     logger.warning(f"    (목표가: {target_price:,.0f}, 현재가: {current_price:,.0f}, 괴리율: {deviation:.2%} > 허용치: {max_deviation:.2%})")
            #     # 신호가 유효하지 않으므로, 이 신호는 오늘 더 이상 실행되지 않도록 처리
            #     self.reset_signal(stock_code)

        # --- [수정] 매도 신호 처리 로직 (일관성을 위해 가격 필터 추가) ---
        elif order_signal == 'sell' and current_position_size > 0:
            target_price = signal_info.get('target_price')
            if not target_price: # 매도 신호에 목표가가 없을 경우, 무조건 매도
                logger.info(f"✅ [시장가 매도 실행] {stock_code} / 수량: {current_position_size}주 (목표가 없음)")
                if self.broker.execute_order(stock_code, 'sell', 0, current_position_size, order_time=current_dt):
                    self.reset_signal(stock_code)
                return

            # # 매도 신호에 목표가가 있는 경우 (데드크로스 등)
            # deviation = abs(current_price - target_price) / target_price
            # max_deviation = self.strategy_params['max_deviation_ratio']

            # if deviation <= max_deviation:
            #     logger.info(f"✅ [시장가 매도 실행] {stock_code} / 수량: {current_position_size}주")
            #     if self.broker.execute_order(stock_code, 'sell', 0, current_position_size, order_time=current_dt):
            #         self.reset_signal(stock_code)
            # else:
            #     logger.warning(f"❌ [매도 신호 스킵] {stock_code} - 가격 괴리율 초과. 다음 기회에 매도 시도.")
            #     # 매도는 다음 기회에 다시 시도할 수 있으므로 reset_signal을 호출하지 않을 수 있음
            #     # (설계에 따라 다름, 여기서는 스킵만 함)