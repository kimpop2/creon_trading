# strategies/intelligent_execution_minute.py

import pandas as pd
import numpy as np
from datetime import datetime, time
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
        
        # broker를 통해 api_client와 db_manager에 쉽게 접근
        self.api_client = self.broker.api_client
        self.db_manager = self.broker.manager.db_manager
        
        # 거래 임무 상태를 관리하는 딕셔너리
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
        """
        [수정] 새로운 거래 임무를 'PENDING' 상태로 등록합니다.
        이미 진행 중인 임무가 있는 종목의 신호는 무시합니다.
        """
        for stock_code, signal_info in signals.items():
            if stock_code in self.trade_missions:
                logger.warning(f"[{stock_code}]에 대한 거래가 이미 진행 중입니다. 새로운 신호({signal_info['signal_type']})를 무시합니다.")
                continue
            
            self.trade_missions[stock_code] = {
                'status': 'PENDING',
                'signal_info': signal_info,
                'executed_quantity': 0,
                'last_update_time': datetime.fromtimestamp(0), # 최초 실행을 위해 과거 시간으로 설정
                'chase_count': 0
            }
            logger.info(f"[{stock_code}] 신규 거래 임무 등록 완료: {signal_info}")
            # 지휘자(trading.py)에게 실시간 호가 구독 요청
            self.api_client.subscribe_realtime_bid(stock_code)

    def handle_conclusion(self, conclusion_data: Dict[str, Any]):
        """
        [이벤트 핸들러] Brokerage로부터 체결/주문응답을 받아 상태를 업데이트합니다.
        """
        stock_code = conclusion_data.get('stock_code')
        if not stock_code or stock_code not in self.trade_missions:
            return

        mission = self.trade_missions[stock_code]
        
        # 체결 이벤트일 경우, 체결 수량 업데이트
        if conclusion_data.get('order_status') in ['체결', '부분체결']:
            mission['executed_quantity'] += conclusion_data.get('quantity', 0)
            logger.info(f"[{stock_code}] 체결 수신. 누적 체결량: {mission['executed_quantity']}/{mission['signal_info']['target_quantity']}")
        
        # 임무 완료 여부 확인
        if mission['executed_quantity'] >= mission['signal_info']['target_quantity']:
            mission['status'] = 'COMPLETED'
            self._cleanup_mission(stock_code)

    def _cleanup_mission(self, stock_code: str):
        """임무 완료 후 정리 작업을 수행합니다."""
        logger.info(f"[{stock_code}] 거래 임무 완료. 정리 작업을 시작합니다.")
        # 지휘자(trading.py)에게 실시간 호가 구독 해지 요청
        self.api_client.unsubscribe_realtime_bid(stock_code)
        # 완료된 임무 제거
        if stock_code in self.trade_missions:
            del self.trade_missions[stock_code]
            
    def run_minute_logic(self, current_dt: datetime, stock_code: str):
        """
        [심장 박동] trading.py의 메인 루프에 의해 주기적으로 호출됩니다.
        현재 상태에 따라 다음 행동을 결정하고 즉시 종료됩니다. (절대 대기하지 않음)
        """
        mission = self.trade_missions.get(stock_code)
        if not mission or mission['status'] == 'COMPLETED':
            return

        # --- 상태별 행동 결정 (State Machine) ---
        status = mission['status']
        
        # 1. 미체결 주문 확인 (Brokerage의 상태를 조회)
        unfilled_orders = self.broker.get_unfilled_orders()
        active_order = next((o for o in unfilled_orders if o['stock_code'] == stock_code), None)
        
        # 2. 상태에 따른 로직 분기
        if status == 'PENDING' and not active_order:
            self._execute_new_order(stock_code, mission, current_dt)
        
        elif status == 'EXECUTING' and active_order:
            if (current_dt - mission['last_update_time']).total_seconds() >= self.strategy_params['chase_interval_seconds']:
                self._execute_amend_order(stock_code, mission, active_order, current_dt)
        
        elif status == 'CHASING' and active_order:
            logger.info(f"[{stock_code}] 최대 추격 횟수 도달. 시장가 전환을 위해 기존 주문을 취소합니다.")
            self.broker.cancel_order(active_order['order_id'], stock_code)
            mission['status'] = 'FINALIZING_MARKET'
        
        elif status == 'FINALIZING_MARKET' and not active_order:
            logger.info(f"[{stock_code}] 시장가로 최종 주문을 실행합니다.")
            remaining_qty = mission['signal_info']['target_quantity'] - mission['executed_quantity']
            if remaining_qty > 0:
                self.broker.execute_order(stock_code, mission['signal_info']['signal_type'], 0, remaining_qty, current_dt)
            # 시장가 주문 후에는 체결 콜백이 올 때까지 대기
    
    def _calculate_as_prices(self, stock_code: str, mission: Dict[str, Any], current_dt: datetime) -> Dict[str, float]:
        """Avellaneda-Stoikov 모델의 핵심 계산 로직"""
        # ... (이전 답변과 동일한 A-S 모델 계산 로직) ...
        # 1. 입력 변수 계산 (fair_price_s, inventory_q, volatility_sigma 등)
        # 2. A-S 공식 적용 (reservation_price_r, optimal_spread_delta)
        # 3. 최종 호가 반환 (bid_price, ask_price)
        # 이 예시에서는 계산 부분을 개념적으로 나타냅니다.
        price_info = self.api_client.get_current_price_and_quotes(stock_code)
        if not price_info: return {}
        fair_price = (price_info['offer_prices'][0] + price_info['bid_prices'][0]) / 2
        
        # 간단한 예시: 현재가 기준으로 일정 틱 아래/위에 주문
        bid_price = self.api_client.round_to_tick(fair_price * 0.998)
        ask_price = self.api_client.round_to_tick(fair_price * 1.002)
        
        return {'bid_price': bid_price, 'ask_price': ask_price}

    def _execute_new_order(self, stock_code, mission, current_dt):
        """신규 주문 제출 로직"""
        prices = self._calculate_as_prices(stock_code, mission, current_dt)
        if not prices: return

        signal_type = mission['signal_info']['signal_type']
        remaining_qty = mission['signal_info']['target_quantity'] - mission['executed_quantity']
        price = prices['bid_price'] if signal_type == 'buy' else prices['ask_price']
        
        # 호가 잔량을 고려하여 1회 주문 수량 결정 (예: 1차 호가 잔량의 50%)
        # ... (호가 잔량 조회 및 수량 계산 로직) ...
        order_qty = remaining_qty # 여기서는 잔량 전체를 주문

        if order_qty > 0:
            self.broker.execute_order(stock_code, signal_type, price, order_qty, current_dt)
            mission['status'] = 'EXECUTING'
            mission['last_update_time'] = current_dt

    def _execute_amend_order(self, stock_code, mission, active_order, current_dt):
        """주문 정정 로직"""
        new_prices = self._calculate_as_prices(stock_code, mission, current_dt)
        if not new_prices: return

        signal_type = mission['signal_info']['signal_type']
        current_price = active_order['order_price']
        new_price = new_prices['bid_price'] if signal_type == 'buy' else new_prices['ask_price']

        is_more_aggressive = (signal_type == 'buy' and new_price > current_price) or \
                             (signal_type == 'sell' and new_price < current_price)

        if is_more_aggressive:
            logger.info(f"[{stock_code}] 더 유리한 가격({new_price:,.0f})으로 주문을 정정합니다.")
            self.broker.amend_order(active_order['order_id'], stock_code, new_price=new_price)
            mission['chase_count'] += 1
            mission['last_update_time'] = current_dt
            
            if mission['chase_count'] >= self.strategy_params['max_chase_count']:
                mission['status'] = 'CHASING'