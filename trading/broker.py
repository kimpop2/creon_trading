# trading/broker.py

from datetime import datetime
import logging
from typing import Dict, Any, Optional
from trading.abstract_broker import AbstractBroker
from manager.db_manager import DBManager # DBManager 직접 사용
from manager.backtest_manager import BacktestManager

logger = logging.getLogger(__name__)

class Broker(AbstractBroker):
    """
    [최종 수정안] 백테스팅 환경을 위한 가상 브로커입니다.
    AbstractBroker 인터페이스를 따르며, 모든 손절/익절 로직을 포함합니다.
    """
    def __init__(self, manager: BacktestManager, initial_cash: float = 10_000_000):
        super().__init__()
        self.manager = manager

        self.initial_cash = initial_cash
        self.commission_rate = 0.0015
        self.tax_rate_sell = 0.002
        self._current_cash_balance: float = self.initial_cash
        
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.transaction_log: list = []
        self.order_counter = 0  # 주문 ID 생성을 위한 카운터 추가

        logging.info(f"백테스트 브로커 초기화: 초기 현금 {self.initial_cash:,.0f}원")

    def set_stop_loss_params(self, stop_loss_params: Optional[Dict[str, Any]]):
        self.stop_loss_params = stop_loss_params
        logging.info(f"백테스트 브로커 손절매 파라미터 설정 완료: {stop_loss_params}")

    def execute_order(self, stock_code: str, order_type: str, price: float, quantity: int, order_time: datetime, order_id: Optional[str] = None) -> Optional[str]:
        # --- 입력값 검증 (Defensive Programming) ---
        if not isinstance(price, (int, float)) or not isinstance(quantity, int):
            logging.error(f"주문 실패({stock_code}): 가격(price)과 수량(quantity)의 타입이 올바르지 않습니다.")
            return None

        if price <= 10 or quantity <= 0: # 10원 이하의 주문은 비정상으로 간주
            logging.error(f"주문 실패({stock_code}): 가격 또는 수량이 비정상적입니다. price={price}, quantity={quantity}")
            return None
        
        # (선택적) 상식적인 범위의 값인지 확인하여 경고 발생
        if price > 5_000_000: # 예: 500만원을 초과하는 가격
            logging.warning(f"주문 경고({stock_code}): 가격이 일반적인 범위를 벗어났습니다. price={price}")

        if quantity > 10000: # 예: 한번에 1만주 이상 주문 시
            logging.warning(f"주문 경고({stock_code}): 수량이 비정상적으로 많습니다. quantity={quantity}")
        # --- 검증 끝 ---
        
        commission = price * quantity * self.commission_rate
        trade_amount = price * quantity  # [수정] trade_amount 계산

        if order_type.lower() == 'buy':
            if self._current_cash_balance >= trade_amount + commission:
                self._current_cash_balance -= (trade_amount + commission)
                if stock_code in self.positions:
                    current_qty = self.positions[stock_code]['quantity']
                    current_avg_price = self.positions[stock_code]['avg_price']
                    new_qty = current_qty + quantity
                    new_avg_price = (current_avg_price * current_qty + price * quantity) / new_qty
                    self.positions[stock_code].update({'quantity': new_qty, 'avg_price': new_avg_price})
                else:
                    self.positions[stock_code] = {
                        'quantity': quantity, 
                        'sell_avail_qty': quantity, # 매수 직후이므로 보유수량과 동일
                        'avg_price': price, 
                        'entry_date': order_time, 
                        'highest_price': price
                    }

                self.order_counter += 1
                order_id = f"backtest_buy_{self.order_counter}"
                # [수정] transaction_log에 'trade_amount' 추가
                self.transaction_log.append(
                    {'trade_datetime': order_time, 
                     'stock_code': stock_code, 
                     'trade_type': 'BUY', 
                     'trade_price': price, 
                     'trade_quantity': quantity, 
                     'trade_amount': trade_amount, 
                     'commission': commission, 
                     'tax': 0, 
                     'realized_profit_loss': 0}
                )
                return order_id # 'success' 대신 생성된 ID 반환
            
            else: return None
        
        elif order_type.lower() == 'sell':
            if stock_code in self.positions and self.positions[stock_code]['quantity'] >= quantity:
                tax = trade_amount * self.tax_rate_sell
                self._current_cash_balance += (trade_amount - commission - tax)
                avg_price = self.positions[stock_code]['avg_price']
                profit = (price - avg_price) * quantity - commission - tax
                
                self.positions[stock_code]['quantity'] -= quantity
                self.positions[stock_code]['sell_avail_qty'] -= quantity # 매도 가능 수량도 차감

                if self.positions[stock_code]['quantity'] == 0: del self.positions[stock_code]
                self.order_counter += 1
                order_id = f"backtest_sell_{self.order_counter}"
                # [수정] transaction_log에 'trade_amount' 추가
                self.transaction_log.append(
                    {'trade_datetime': order_time,
                      'stock_code': stock_code, 
                      'trade_type': 'SELL', 
                      'trade_price': price, 
                      'trade_quantity': quantity, 
                      'trade_amount': trade_amount, 
                      'commission': commission, 
                      'tax': tax, 
                      'realized_profit_loss': profit}
                )
                return order_id # 'success' 대신 생성된 ID 반환
            
            else: return None
        
        return None

    def get_current_cash_balance(self) -> float:
        return self._current_cash_balance

    def get_current_positions(self) -> Dict[str, Dict[str, Any]]:
        return self.positions

    def get_position_size(self, stock_code: str) -> int:
        return self.positions.get(stock_code, {}).get('quantity', 0)

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        holdings_value = sum(pos_info['quantity'] * current_prices.get(stock_code, pos_info['avg_price']) for stock_code, pos_info in self.positions.items())
        return self._current_cash_balance + holdings_value
    
    def get_unfilled_stock_codes(self) -> set:
        """미체결 상태인 주문들의 종목 코드 집합을 반환합니다."""
        return set()
    def _restore_positions_state(self, data_store: Dict[str, Any]):
        """백테스트에서는 상태 복원이 불필요하므로 비워둡니다."""
        pass # trading.py와의 구조적 통일성을 위해 메서드 존재
    
    # 개별 종목 손절/익절/보유기간 손절/트레일링 스탑(매도) 와
    # 포토폴리오 손절을 함께 처리
    def check_and_execute_stop_loss(self, current_prices: Dict[str, float], current_dt: datetime) -> bool:
        if not self.stop_loss_params:
            return False
            
        executed_any = False
        # [복원] 1. 개별 종목 손절/익절 로직
        for stock_code in list(self.positions.keys()):
            if self._check_individual_stock_conditions(stock_code, current_prices, current_dt) == 'sucess':
                executed_any = True

        # [복원] 2. 포트폴리오 레벨 손절 로직
        if self._check_portfolio_conditions(current_prices, current_dt):
            executed_any = True
            
        return executed_any

    # 익절/손절/보유기반 손절/트레일링 매도 
    def _check_individual_stock_conditions(self, stock_code: str, current_prices: Dict[str, float], current_dt: datetime) -> bool:
        pos_info = self.positions.get(stock_code)
        current_price = current_prices.get(stock_code)
        if not pos_info or not current_price or pos_info['quantity'] <= 0:
            return False

        # 최고가 업데이트
        if current_price > pos_info.get('highest_price', 0):
            pos_info['highest_price'] = current_price

        avg_price = pos_info['avg_price']
        highest_price = pos_info['highest_price']
        profit_pct = (current_price - avg_price) * 100 / avg_price if avg_price > 0 else 0

        # 익절
        if profit_pct >= self.stop_loss_params.get('take_profit_ratio', float('inf')):
            logging.info(f"[익절] {stock_code}")
            return self.execute_order(stock_code, 'sell', current_price, pos_info['quantity'], current_dt)
        # 보유기간 기반 손절
        holding_days = (current_dt.date() - pos_info['entry_date'].date()).days
        if 1 <= holding_days <= 3 and profit_pct <= self.stop_loss_params.get('early_stop_loss', -float('inf')):
             logging.info(f"[조기손절] {stock_code}")
             return self.execute_order(stock_code, 'sell', current_price, pos_info['quantity'], current_dt)
        # 일반 손절
        if 1 <= holding_days and profit_pct <= self.stop_loss_params.get('stop_loss_ratio', -float('inf')):
            logging.info(f"[손절] {stock_code}")
            return self.execute_order(stock_code, 'sell', current_price, pos_info['quantity'], current_dt)
        # 트레일링 스탑
        if highest_price > 0:
            trailing_stop_pct = (current_price - highest_price) * 100 / highest_price
            if trailing_stop_pct <= self.stop_loss_params.get('trailing_stop_ratio', -float('inf')):
                logging.info(f"[가상 트레일링 스탑] {stock_code}")
                return self.execute_order(stock_code, 'sell', current_price, pos_info['quantity'], current_dt)
        
        return False

    # 포트폴리오 손절 판단 
    def _check_portfolio_conditions(self, current_prices: Dict[str, float], current_dt: datetime) -> bool:
        # [복원] 포트폴리오 전체 손실률 기준
        total_cost = sum(p['quantity'] * p['avg_price'] for p in self.positions.values())
        if total_cost == 0: return False
        
        total_current_value = sum(p['quantity'] * current_prices.get(code, p['avg_price']) for code, p in self.positions.items())
        total_profit_pct = (total_current_value - total_cost) * 100 / total_cost

        if total_profit_pct <= self.stop_loss_params.get('portfolio_stop_loss', -float('inf')):
            logging.info(f"[가상 포트폴리오 손절] 전체 손실률 {total_profit_pct:.2%}가 기준치 도달")
            self._execute_portfolio_sellout(current_prices, current_dt)
            return True

        # [복원] 동시다발적 손실 기준
        stop_loss_pct = self.stop_loss_params.get('stop_loss_ratio', -float('inf'))
        losing_positions_count = 0
        for code, pos in self.positions.items():
            price = current_prices.get(code)
            if price and ((price - pos['avg_price']) / pos['avg_price']) * 100 <= stop_loss_pct:
                losing_positions_count += 1
        
        if losing_positions_count >= self.stop_loss_params.get('max_losing_positions', float('inf')):
            logging.info(f"[가상 포트폴리오 손절] 손실 종목 수 {losing_positions_count}개가 기준치 도달")
            self._execute_portfolio_sellout(current_prices, current_dt)
            return True

        return False
    
    # 포토폴리오 손절 실행
    def _execute_portfolio_sellout(self, current_prices: Dict[str, float], current_dt: datetime):
        """포트폴리오 전체 청산을 실행합니다."""
        for stock_code in list(self.positions.keys()):
            pos_info = self.positions[stock_code]
            price = current_prices.get(stock_code, pos_info['avg_price'])
            self.execute_order(stock_code, 'sell', price, pos_info['quantity'], current_dt)

    def cleanup(self) -> None:
        logging.info("백테스트 브로커 리소스 정리 완료.")
        pass


    # PassMinute 를 위한 메서드
    def get_stop_loss_price(self, stock_code: str) -> Optional[float]:
        """단순 손절매 가격을 계산하여 반환합니다."""
        if not self.stop_loss_params:
            return None
        
        position = self.positions.get(stock_code)
        stop_loss_ratio = self.stop_loss_params.get('stop_loss_ratio')

        if position and stop_loss_ratio:
            avg_price = position['avg_price']
            # 손절 비율은 음수이므로 (예: -5.0) 그대로 더해줍니다.
            stop_loss_price = avg_price * (1 + stop_loss_ratio / 100.0)
            return stop_loss_price
        return None

    def get_take_profit_price(self, stock_code: str) -> Optional[float]:
        """단순 익절 가격을 계산하여 반환합니다."""
        if not self.stop_loss_params:
            return None

        position = self.positions.get(stock_code)
        take_profit_ratio = self.stop_loss_params.get('take_profit_ratio')

        if position and take_profit_ratio:
            avg_price = position['avg_price']
            take_profit_price = avg_price * (1 + take_profit_ratio / 100.0)
            return take_profit_price
        return None

    def get_trailing_stop_price(self, stock_code: str, current_high_price: float) -> Optional[float]:
        """
        트레일링 스탑 가격을 계산하여 반환합니다.
        (참고) 이 메서드를 호출하기 전에 포지션의 최고가(highest_price)가 업데이트되어야 합니다.
        """
        if not self.stop_loss_params:
            return None
            
        position = self.positions.get(stock_code)
        trailing_stop_ratio = self.stop_loss_params.get('trailing_stop_ratio')

        if position and trailing_stop_ratio:
            # 1. 포지션의 현재 최고가를 당일 고가와 비교하여 업데이트
            # check_and_execute_stop_loss 에서 이미 처리하고 있다면 이 라인은 생략 가능
            position['highest_price'] = max(position.get('highest_price', 0), current_high_price)
            
            # 2. 업데이트된 최고가 기준으로 트레일링 스탑 가격 계산
            highest_price = position['highest_price']
            trailing_stop_price = highest_price * (1 + trailing_stop_ratio / 100.0)
            return trailing_stop_price
        return None