# strategies/intelligent_minute.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from .strategy import MinuteStrategy
import logging

logger = logging.getLogger(__name__)

class IntelligentMinute(MinuteStrategy):
    """
    Avellaneda-Stoikov 모델을 사용하여 주문을 지능적으로 실행하는 실시간 매매 전용 분봉 전략.
    '상태 기계'로 동작하여 비동기 이벤트(체결, 시간)에 따라 행동을 결정합니다.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self.strategy_name = "IntelligentMinute"
        
        self.api_client = self.broker.api_client
        self.db_manager = self.broker.manager.db_manager
        
        self.trade_missions: Dict[str, Dict[str, Any]] = {}
        
        self._validate_parameters()

    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required = [
            'risk_aversion', 'order_flow_intensity', 'volatility_period',
            'chase_interval_seconds', 'max_chase_count'
        ]
        for param in required:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        logger.info("지능형 실행 전략 파라미터 검증 완료.")
    
    def update_signals(self, signals: Dict[str, Any]):
        """새로운 거래 임무를 'PENDING' 상태로 등록합니다."""
        for stock_code, signal_info in signals.items():
            if stock_code in self.trade_missions:
                logger.warning(f"[{stock_code}]에 대한 거래가 이미 진행 중입니다. 새로운 신호({signal_info['signal_type']})를 무시합니다.")
                continue
            
            self.trade_missions[stock_code] = {
                'status': 'PENDING',
                'signal_info': signal_info,
                'executed_quantity': 0,
                'last_update_time': datetime.fromtimestamp(0),
                'chase_count': 0
            }
            logger.info(f"[{stock_code}] 신규 거래 임무 등록 완료: {signal_info}")
            self.api_client.subscribe_realtime_bid(stock_code)

    def handle_conclusion(self, conclusion_data: Dict[str, Any]):
        """Brokerage로부터 체결/주문응답을 받아 상태를 업데이트합니다."""
        stock_code = conclusion_data.get('stock_code')
        if not stock_code or stock_code not in self.trade_missions:
            return

        mission = self.trade_missions[stock_code]
        
        if conclusion_data.get('order_status') in ['체결', '부분체결']:
            mission['executed_quantity'] += conclusion_data.get('quantity', 0)
            logger.info(f"[{stock_code}] 체결 수신. 누적 체결량: {mission['executed_quantity']}/{mission['signal_info']['target_quantity']}")
        
        if mission['executed_quantity'] >= mission['signal_info']['target_quantity']:
            mission['status'] = 'COMPLETED'
            self._cleanup_mission(stock_code)

    def _cleanup_mission(self, stock_code: str):
        """임무 완료 후 정리 작업을 수행합니다."""
        logger.info(f"[{stock_code}] 거래 임무 완료. 정리 작업을 시작합니다.")
        self.api_client.unsubscribe_realtime_bid(stock_code)
        if stock_code in self.trade_missions:
            del self.trade_missions[stock_code]
            
    def run_minute_logic(self, current_dt: datetime, stock_code: str, unfilled_orders: List[Dict[str, Any]]):
        """주기적으로 호출되어 상태에 따라 다음 행동을 결정합니다."""
        mission = self.trade_missions.get(stock_code)
        if not mission or mission['status'] == 'COMPLETED':
            return

        status = mission['status']
        current_inventory = self.broker.get_position_size(stock_code)
        active_order = next((o for o in unfilled_orders if o['stock_code'] == stock_code), None)
        
        if status == 'PENDING' and not active_order:
            self._execute_new_order(stock_code, mission, current_dt, current_inventory)
        
        elif status == 'EXECUTING' and active_order:
            if (current_dt - mission['last_update_time']).total_seconds() >= self.strategy_params['chase_interval_seconds']:
                self._execute_amend_order(stock_code, mission, active_order, current_dt, current_inventory)
        
        elif status == 'CHASING' and active_order:
            logger.info(f"[{stock_code}] 최대 추격 횟수 도달. 시장가 전환을 위해 기존 주문을 취소합니다.")
            self.broker.cancel_order(active_order['order_id'], stock_code)
            mission['status'] = 'FINALIZING_MARKET'
        
        elif status == 'FINALIZING_MARKET' and not active_order:
            logger.info(f"[{stock_code}] 시장가로 최종 주문을 실행합니다.")
            remaining_qty = mission['signal_info']['target_quantity'] - mission['executed_quantity']
            if remaining_qty > 0:
                self.broker.execute_order(stock_code, mission['signal_info']['signal_type'], 0, remaining_qty, current_dt)

    def _calculate_as_prices(self, stock_code: str, current_inventory: int, current_dt: datetime) -> Dict[str, float]:
        """Avellaneda-Stoikov 모델의 완전한 계산 로직입니다."""
        vol_period = self.strategy_params['volatility_period']
        
        # ✅ [수정] _get_historical_data_up_to를 사용하여 과거 분봉 데이터를 가져옵니다.
        price_data = self._get_historical_data_up_to('minute', stock_code, current_dt, lookback_period=vol_period)
        
        if price_data is None or len(price_data) < vol_period:
            logger.warning(f"[{stock_code}] 변동성 계산을 위한 데이터 부족. 기본 변동성을 사용합니다.")
            volatility = 0.2
        else:
            log_returns = np.log(price_data['close'] / price_data['close'].shift(1))
            minute_volatility = log_returns.std()
            volatility = minute_volatility * np.sqrt(252 * 390)

        price_info = self.api_client.get_current_price_and_quotes(stock_code)
        if not price_info or price_info['offer_prices'][0] == 0 or price_info['bid_prices'][0] == 0:
            return {}
        mid_price = (price_info['offer_prices'][0] + price_info['bid_prices'][0]) / 2

        gamma = self.strategy_params['risk_aversion']
        k = self.strategy_params.get('order_flow_intensity', 1.5)
        T = 1.0
        t = (current_dt - current_dt.replace(hour=9, minute=0, second=0)).total_seconds() / (6.5 * 3600)
        t = max(0, min(t, 1.0))

        time_left = T - t
        if time_left <= 0: time_left = 0.0001

        reservation_price = mid_price - current_inventory * gamma * (volatility**2) * time_left
        optimal_spread = (gamma * (volatility**2) * time_left) + (2/gamma) * np.log(1 + (gamma / k))

        ask_price = reservation_price + optimal_spread / 2
        bid_price = reservation_price - optimal_spread / 2
        
        return {
            'bid_price': self.api_client.round_to_tick(bid_price),
            'ask_price': self.api_client.round_to_tick(ask_price)
        }

    def _execute_new_order(self, stock_code, mission, current_dt, current_inventory):
        """신규 주문 제출 로직"""
        # ✅ [수정] current_dt를 _calculate_as_prices에 전달합니다.
        prices = self._calculate_as_prices(stock_code, current_inventory, current_dt)
        if not prices: return

        signal_type = mission['signal_info']['signal_type']
        remaining_qty = mission['signal_info']['target_quantity'] - mission['executed_quantity']
        price = prices['bid_price'] if signal_type == 'buy' else prices['ask_price']
        
        if remaining_qty > 0:
            self.broker.execute_order(stock_code, signal_type, price, remaining_qty, current_dt)
            mission['status'] = 'EXECUTING'
            mission['last_update_time'] = current_dt

    def _execute_amend_order(self, stock_code, mission, active_order, current_dt, current_inventory):
        """주문 정정 로직"""
        # ✅ [수정] current_dt를 _calculate_as_prices에 전달합니다.
        new_prices = self._calculate_as_prices(stock_code, current_inventory, current_dt)
        if not new_prices: return

        signal_type = mission['signal_info']['signal_type']
        current_price = active_order['order_price']
        new_price = new_prices['bid_price'] if signal_type == 'buy' else new_prices['ask_price']

        is_more_aggressive = (signal_type == 'buy' and new_price > current_price) or \
                            (signal_type == 'sell' and new_price < current_price)

        if is_more_aggressive:
            logger.info(f"[{stock_code}] 더 유리한 가격({new_price:,.0f})으로 주문을 정정합니다.")
            self.broker.amend_order(
                order_id=active_order['order_id'], 
                stock_code=stock_code, 
                new_price=new_price,
                new_quantity=active_order['unfilled_quantity']
            )
        else:
            logger.info(f"[{stock_code}] 가격 변동이 없어 주문 정정을 건너뜁니다. (현재가: {current_price}, 새 제안가: {new_price})")

        mission['chase_count'] += 1
        mission['last_update_time'] = current_dt
        logger.info(f"[{stock_code}] 추격 횟수 증가: {mission['chase_count']}/{self.strategy_params['max_chase_count']}")
        
        if mission['chase_count'] >= self.strategy_params['max_chase_count']:
            mission['status'] = 'CHASING'
            logger.info(f"[{stock_code}] 최대 추격 횟수에 도달하여 시장가 전환 상태로 변경합니다.")