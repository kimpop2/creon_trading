# trade/broker.py

import datetime
import logging
import uuid # For generating unique order IDs
from typing import Dict, Any, List, Optional, Tuple
from collections import deque # For simulating a queue of events
import os, sys
# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
#from trade.abstract_broker import AbstractBroker

logger = logging.getLogger(__name__)

class Broker():
    def __init__(self, initial_cash, commission_rate=0.0016, slippage_rate=0.0004, simulated_fill_delay_seconds=3):
        self.cash = initial_cash
        self.positions = {}
        self.transaction_log = []
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.stop_loss_params = None
        self.initial_portfolio_value = initial_cash
        self.data_store = {'daily': {}, 'minute': {}} # {stock_code: DataFrame}
        # New for simulation: pending orders and a queue for simulated conclusions
        self.pending_orders: Dict[str, Dict[str, Any]] = {} # order_id -> order_details
        self.simulated_fill_delay_seconds = simulated_fill_delay_seconds
        # 이 큐는 (시뮬레이션_체결_시각, order_id, 종목코드, 수량, 가격, 주문유형)을 저장합니다.
        self.order_fill_queue: deque[Tuple[datetime.datetime, str, str, int, float, str]] = deque()
        
        logger.info(f"브로커 초기화: 초기 현금 {self.cash:,.0f}원, 수수료율 {self.commission_rate*100:.2f}%")

    def set_stop_loss_params(self, stop_loss_params: Dict[str, Any]):
        """손절매 관련 파라미터를 설정합니다."""
        if stop_loss_params is None:
            return
        self.stop_loss_params = stop_loss_params
        logger.info(f"손절매 파라미터 설정: {stop_loss_params}")

    def execute_order(self, stock_code: str, order_type: str, price: float, quantity: int, current_dt: datetime.datetime) -> Optional[str]:
        """
        주문을 시뮬레이션하여 접수하고, 체결을 위한 스케줄을 잡습니다.
        실시간 브로커의 send_order와 유사하게 작동합니다.
        """
        logger.info(f"[{current_dt.isoformat()}] 주문 시도: {order_type.upper()} {stock_code}, 수량: {quantity}, 가격: {price}")
        
        # 매수/매도 사전 잔고 확인
        if order_type.lower() == 'buy':
            required_cash = price * quantity * (1 + self.commission_rate)
            if self.cash < required_cash:
                logger.warning(f"[{current_dt.isoformat()}] 현금 부족: {stock_code} 매수 {quantity}개 @ {price}. 필요: {required_cash:,.0f}원, 현재: {self.cash:,.0f}원.")
                return None
        
        if order_type.lower() == 'sell':
            current_position_size = self.positions.get(stock_code, {}).get('size', 0)
            if current_position_size < quantity:
                logger.warning(f"[{current_dt.isoformat()}] 주식 부족: {stock_code} 매도 {quantity}개. 보유: {current_position_size}개.")
                return None

        # 주문 ID 생성 (고유성을 위해 datetime과 uuid를 조합)
        order_id = f"sim-{current_dt.strftime('%Y%m%d%H%M%S%f')}-{uuid.uuid4().hex[:8]}"
        
        # pending_orders에 주문 추가
        self.pending_orders[order_id] = {
            'stock_code': stock_code,
            'type': order_type,
            'requested_price': price,
            'requested_quantity': quantity,
            'filled_quantity': 0, # 현재 체결된 수량
            'status': 'PENDING',
            'order_time': current_dt # 주문 접수 시각
        }
        logger.info(f"[{current_dt.isoformat()}] 주문 접수 (시뮬레이션): ID={order_id}, {order_type} {quantity} {stock_code} @ {price}")

        # simulated_fill_delay_seconds 후에 체결되도록 스케줄링
        simulated_fill_time = current_dt + datetime.timedelta(seconds=self.simulated_fill_delay_seconds)
        self.order_fill_queue.append((simulated_fill_time, order_id, stock_code, quantity, price, order_type))
        self.order_fill_queue = deque(sorted(self.order_fill_queue, key=lambda x: x[0])) # 시간 순서대로 정렬

        return order_id
    
    def process_simulated_time_events(self, current_simulated_dt: datetime.datetime):
        """
        현재 시뮬레이션 시각에 체결되어야 할 보류 중인 주문을 처리합니다.
        메인 시뮬레이션 루프에서 주기적으로 호출됩니다.
        """
        while self.order_fill_queue and self.order_fill_queue[0][0] <= current_simulated_dt:
            sim_fill_time, order_id, stock_code, quantity, price, order_type = self.order_fill_queue.popleft()
            
            # 주문이 여전히 'PENDING' 상태인지 확인 (취소되거나 이미 체결 완료되지 않았는지)
            if order_id in self.pending_orders and self.pending_orders[order_id]['status'] != 'FILLED':
                logger.info(f"[{sim_fill_time.isoformat()}] 주문 체결 (시뮬레이션): ID={order_id}, {order_type} {quantity} {stock_code} @ {price}")
                # 실제 체결 처리 로직 호출
                self._process_conclusion(order_id, stock_code, order_type, quantity, price, sim_fill_time)
            else:
                logger.debug(f"[{sim_fill_time.isoformat()}] 이미 처리되었거나 취소된 주문 ID ({order_id}) 체결 시도 무시.")

    def _process_conclusion(self, order_id: str, stock_code: str, trade_type: str, 
                           filled_quantity: int, filled_price: float, trade_time: datetime.datetime):
        """
        거래 체결을 처리하고 브로커의 상태(현금, 포지션)를 업데이트하는 내부 메서드.
        LiveBroker의 on_conclusion 로직을 모방합니다.
        """
        if order_id not in self.pending_orders:
            logger.warning(f"[_process_conclusion] 체결 처리를 위한 알 수 없거나 이미 처리된 주문 ID: {order_id}")
            return
        
        pending_order = self.pending_orders[order_id]
        pending_order['filled_quantity'] += filled_quantity
        
        # 수수료 및 순 금액 계산
        transaction_value = filled_price * filled_quantity
        commission = transaction_value * self.commission_rate
        net_amount = transaction_value + commission if trade_type == 'buy' else transaction_value - commission

        # 현금 및 포지션 업데이트
        if trade_type == 'buy':
            self.cash -= net_amount
            if stock_code not in self.positions:
                self.positions[stock_code] = {
                    'size': 0, 
                    'avg_price': 0.0, 
                    'entry_date': trade_time.date(), 
                    'highest_price': filled_price # 손절매를 위한 최고가 기록
                }
            
            # 기존 포지션의 평균 단가 업데이트
            current_size = self.positions[stock_code]['size']
            current_value = current_size * self.positions[stock_code]['avg_price']
            new_size = current_size + filled_quantity
            new_value = current_value + transaction_value # 체결 단가 기준
            self.positions[stock_code]['size'] = new_size
            self.positions[stock_code]['avg_price'] = new_value / new_size if new_size > 0 else 0.0
            
            # 최고가 업데이트
            if filled_price > self.positions[stock_code].get('highest_price', 0):
                self.positions[stock_code]['highest_price'] = filled_price

        elif trade_type == 'sell':
            self.cash += net_amount
            if stock_code in self.positions:
                self.positions[stock_code]['size'] -= filled_quantity
                if self.positions[stock_code]['size'] <= 0: # 보유 수량이 0 이하면 포지션 삭제
                    del self.positions[stock_code]
            else: # 보유하지 않은 종목을 매도한 경우 (사전 체크에서 걸러지나, 혹시 모를 경우)
                logger.error(f"[_process_conclusion] 보유하지 않은 종목 {stock_code} 매도 체결 발생 (ID: {order_id}). 시뮬레이션 오류 가능성.")

        # 거래 내역 로그 기록
        self.transaction_log.append({
            'order_id': order_id,
            'time': trade_time,
            'stock_code': stock_code,
            'type': trade_type,
            'price': filled_price,
            'quantity': filled_quantity,
            'commission': commission,
            'net_cash_change': -net_amount if trade_type == 'buy' else net_amount,
            'cash_after': self.cash
        })
        logger.info(f"[{trade_time.isoformat()}] 잔고 업데이트: 현금: {self.cash:,.0f} KRW, {stock_code} 보유: {self.positions.get(stock_code, {}).get('size', 0)} 주.")

        # 주문이 완전히 체결되었는지 확인
        if pending_order['filled_quantity'] >= pending_order['requested_quantity']:
            pending_order['status'] = 'FILLED'
            if order_id in self.pending_orders: # pending_orders에서 제거
                del self.pending_orders[order_id]
            logger.info(f"[{trade_time.isoformat()}] 주문 {order_id} ({stock_code}) 완전 체결 완료.")
        else:
            pending_order['status'] = 'PARTIALLY_FILLED'
            logger.info(f"[{trade_time.isoformat()}] 주문 {order_id} ({stock_code}) 부분 체결. 남은 수량: {pending_order['requested_quantity'] - pending_order['filled_quantity']}")

    def _calculate_loss_ratio(self, current_price: float, avg_price: float) -> float:
        """손실률을 계산합니다."""
        if avg_price == 0:
            return 0.0
        return (avg_price - current_price) / avg_price

    def check_portfolio_stop_loss(self, current_dt: datetime.datetime, current_prices: Dict[str, float]) -> bool:
        """
        포트폴리오 전체 손절매 조건을 확인하고 실행합니다.
        시뮬레이션에서는 이 함수가 매 분 호출될 수 있습니다.
        """
        if self.stop_loss_params and self.stop_loss_params.get('portfolio_stop_loss_enabled', False):
            # 특정 시간 이후에만 손절매 검사
            if current_dt.time() >= datetime.time(self.stop_loss_params.get('portfolio_stop_loss_start_hour', 14), 0, 0):
                losing_positions_count = 0
                for stock_code, position in self.positions.items():
                    if position['size'] > 0 and stock_code in current_prices:
                        loss_ratio = self._calculate_loss_ratio(current_prices[stock_code], position['avg_price'])
                        if loss_ratio >= self.stop_loss_params['stop_loss_ratio']: # 손실률이 기준 이상인 경우
                            losing_positions_count += 1

                if losing_positions_count >= self.stop_loss_params['max_losing_positions']:
                    logger.info(f'[{current_dt.isoformat()}] [포트폴리오 손절] 손실 종목 수: {losing_positions_count}개 (기준: {self.stop_loss_params["max_losing_positions"]}개 이상)')
                    self._execute_portfolio_sellout(current_prices, current_dt)
                    return True
        return False

    def _execute_portfolio_sellout(self, current_prices: Dict[str, float], current_dt: datetime.datetime):
        """포트폴리오 전체 청산을 실행합니다."""
        logger.info(f"[{current_dt.isoformat()}] 포트폴리오 전체 청산 실행!")
        stocks_to_sell = list(self.positions.keys()) # 현재 보유 종목 리스트 복사
        for stock_code in stocks_to_sell:
            position_size = self.positions.get(stock_code, {}).get('size', 0)
            if position_size > 0 and stock_code in current_prices:
                # 시뮬레이션에서는 시장가로 즉시 매도된다고 가정
                self.execute_order(stock_code, 'sell', current_prices[stock_code], position_size, current_dt)
                logger.info(f"[{current_dt.isoformat()}] [포트폴리오 손절] {stock_code} 전량 매도 주문 발생. 수량: {position_size}, 가격: {current_prices[stock_code]:,.0f}")
            elif stock_code not in current_prices:
                logger.warning(f"[{current_dt.isoformat()}] [포트폴리오 손절] {stock_code} 현재가 정보를 찾을 수 없어 매도 불가. (보유수량: {position_size})")

    def reset_daily_transactions(self):
        """
        일별 거래 내역을 초기화 (필요시 사용).
        백테스트에서는 매일 초기화되지만, 라이브에서는 누적 관리됨.
        시뮬레이션에서는 특정 로직을 위해 초기화가 필요할 수 있습니다.
        """
        # 현재 시뮬레이션에서는 이 메서드가 하는 역할이 없음.
        # 필요하다면 여기에 일별 리셋 로직을 추가.
        pass

    def get_current_cash(self) -> float:
        return self.cash

    def get_current_positions(self) -> Dict[str, Dict[str, Any]]:
        return self.positions

    def get_transaction_log(self) -> List[Dict[str, Any]]:
        return self.transaction_log
    
    def get_position_size(self, stock_code):
        """특정 종목의 보유 수량을 반환합니다."""
        return self.positions.get(stock_code, {}).get('size', 0)

    def get_portfolio_value(self, current_prices):
        """현재 포트폴리오의 총 가치를 계산하여 반환합니다."""
        holdings_value = 0
        for stock_code, pos_info in self.positions.items():
            if stock_code in current_prices:
                holdings_value += pos_info['size'] * current_prices[stock_code]
            else:
                # 데이터가 없는 경우, 마지막 유효 가격 또는 평균 단가 사용
                logging.warning(f"경고: {stock_code}의 현재 가격 데이터가 없습니다. 보유 포지션 가치 계산에 문제 발생 가능.")
                holdings_value += pos_info['size'] * pos_info['avg_price'] # 대안으로 평균 단가 사용

        return self.cash + holdings_value

    def check_and_execute_stop_loss(self, stock_code, current_price, current_dt):
        """
        개별 종목에 대한 손절 조건을 체크하고, 조건 충족 시 매도 주문을 실행합니다.
        """
        if stock_code not in self.positions or self.positions[stock_code]['size'] <= 0:
            return False

        pos_info = self.positions[stock_code]
        avg_price = pos_info['avg_price']
        
        # 1. 포지션의 최고가 업데이트
        self._update_highest_price(stock_code, current_price)
        highest_price = pos_info['highest_price']

        loss_ratio = self._calculate_loss_ratio(current_price, avg_price)
        
        # 1. 단순 손절 (stop_loss_ratio)
        if self.stop_loss_params['stop_loss_ratio'] is not None and loss_ratio <= self.stop_loss_params['stop_loss_ratio']:
            logging.info(f"[개별 손절매 발생] {stock_code}: 현재 손실률 {loss_ratio:.2f}%가 기준치 {self.stop_loss_params['stop_loss_ratio']}%를 초과. {current_dt.isoformat()}")
            self.execute_order(stock_code, 'sell', current_price, pos_info['size'], current_dt)
            return True
        
        # 2. 트레일링 스탑 (trailing_stop_ratio)
        if self.stop_loss_params['trailing_stop_ratio'] is not None and highest_price > 0:
            trailing_loss_ratio = self._calculate_loss_ratio(current_price, highest_price)
            if trailing_loss_ratio <= self.stop_loss_params['trailing_stop_ratio']:
                logging.info(f"[트레일링 스탑 발생] {stock_code}: 현재가 {current_price:,.0f}원이 최고가 {highest_price:,.0f}원 대비 {trailing_loss_ratio:.2f}% 하락. {current_dt.isoformat()}")
                self.execute_order(stock_code, 'sell', current_price, pos_info['size'], current_dt)
                return True
        
        # 3. 보유 기간 기반 손절 (early_stop_loss)
        if self.stop_loss_params['early_stop_loss'] is not None and pos_info['entry_date'] is not None:
            holding_days = (current_dt.date() - pos_info['entry_date']).days
            if holding_days <= 5 and loss_ratio <= self.stop_loss_params['early_stop_loss']:
                logging.info(f"[조기 손절매 발생] {stock_code}: 매수 후 {holding_days}일 이내 손실률 {loss_ratio:.2f}%가 조기 손절 기준 {self.stop_loss_params['early_stop_loss']}% 초과. {current_dt.isoformat()}")
                self.execute_order(stock_code, 'sell', current_price, pos_info['size'], current_dt)
                return True
        
        return False
        
    def check_and_execute_portfolio_stop_loss(self, current_prices: Dict[str, float], current_dt: datetime) -> bool:
        """
        포트폴리오 전체의 손실률을 체크하고 손절 조건에 도달하면 모든 포지션을 청산합니다.
        1. 전체 손실폭 기준 (portfolio_stop_loss)
        2. 동시다발적 손실 기준 (max_losing_positions)
        """
        if not self.positions:
            return False

        # 1. 전체 손실폭 기준 (portfolio_stop_loss)
        if self.stop_loss_params['portfolio_stop_loss'] is not None:
            total_cost = 0
            total_current_value = 0
            
            for stock_code, position in self.positions.items():
                if position['size'] > 0:
                    total_cost += position['size'] * position['avg_price']
                    total_current_value += current_prices.get(stock_code, 0) * position['size']
            
            if total_cost > 0:
                total_loss_ratio = self._calculate_loss_ratio(total_current_value, total_cost)
                if total_loss_ratio <= self.stop_loss_params['portfolio_stop_loss']:
                    logging.info(f'[포트폴리오 손절] {current_dt.isoformat()} - 전체 손실률: {total_loss_ratio:.2f}%')
                    self._execute_portfolio_sellout(current_prices, current_dt)
                    return True

        # 2. 동시다발적 손실 기준 (max_losing_positions)
        if self.stop_loss_params['max_losing_positions'] is not None:
            if self.stop_loss_params['stop_loss_ratio'] is None:
                logging.warning("max_losing_positions 사용을 위해서는 stop_loss_ratio가 설정되어야 합니다.")
                return False

            losing_positions_count = 0
            for stock_code, position in self.positions.items():
                if position['size'] > 0 and stock_code in current_prices:
                    loss_ratio = self._calculate_loss_ratio(current_prices[stock_code], position['avg_price'])
                    if loss_ratio <= self.stop_loss_params['stop_loss_ratio']:
                        losing_positions_count += 1

            if losing_positions_count >= self.stop_loss_params['max_losing_positions']:
                logging.info(f'[포트폴리오 손절] {current_dt.isoformat()} - 손실 종목 수: {losing_positions_count}개 (기준: {self.stop_loss_params["max_losing_positions"]}개)')
                self._execute_portfolio_sellout(current_prices, current_dt)
                return True

        return False
    

