import datetime
import logging
from typing import Dict
from trade.abstract_broker import AbstractBroker
# --- 로거 설정 (스크립트 최상단에서 설정하여 항상 보이도록 함) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 테스트 시 DEBUG로 설정하여 모든 로그 출력 - 제거
class Broker(AbstractBroker):
    def __init__(self, initial_cash, commission_rate=0.0016, slippage_rate=0.0004):
        self.cash = initial_cash
        self.positions = {}  # {stock_code: {'size': int, 'avg_price': float, 'entry_date': datetime.date, 'highest_price': float}}
        self.transaction_log = [] # (date, stock_code, type, price, quantity, commission, net_amount)
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.stop_loss_params = None
        self.initial_portfolio_value = initial_cash # 포트폴리오 손절을 위한 초기값
        logging.info(f"브로커 초기화: 초기 현금 {self.cash:,.0f}원, 수수료율 {self.commission_rate*100:.2f}%")

    def set_stop_loss_params(self, stop_loss_params):
        """손절매 관련 파라미터를 설정합니다."""
        if stop_loss_params is None:
            return
        self.stop_loss_params = stop_loss_params
        logging.info(f"브로커 손절매 파라미터 설정 완료: {stop_loss_params}")

    
    def execute_order(self, stock_code, order_type, price, quantity, current_dt):
        """매매 주문을 실행합니다."""
        if quantity <= 0:
            logging.warning(f"[{current_dt.isoformat()}] {stock_code}: {order_type} 수량 0. 주문 실행하지 않음.")
            return False

        effective_price = price * (1 + self.slippage_rate if order_type == 'buy' else 1 - self.slippage_rate)

        if order_type == 'buy':
            total_cost = effective_price * quantity  # 커미션 없이
            if self.cash >= total_cost:
                # 기존 포지션이 있으면 평균 단가 계산
                if stock_code in self.positions and self.positions[stock_code]['size'] > 0:
                    current_size = self.positions[stock_code]['size']
                    current_avg_price = self.positions[stock_code]['avg_price']
                    new_size = current_size + quantity
                    new_avg_price = (current_avg_price * current_size + effective_price * quantity) / new_size
                    self.positions[stock_code]['size'] = new_size
                    self.positions[stock_code]['avg_price'] = new_avg_price
                    self.positions[stock_code]['highest_price'] = effective_price
                else:
                    self.positions[stock_code] = {
                        'size': quantity,
                        'avg_price': effective_price,
                        'entry_date': current_dt.date(),
                        'highest_price': effective_price
                    }
                self.cash -= total_cost
                commission = 0  # 매수 시 커미션 없음
                self.transaction_log.append((current_dt, stock_code, 'buy', price, quantity, commission, total_cost))
                logging.info(f"[{current_dt.isoformat()}] {stock_code}: {quantity}주 매수. 실제 가격: {effective_price:,.0f}원, 수수료: {commission:,.0f}원, 매매대금: {total_cost:,.0f}원, ")
                return True
            else:
                logging.warning(f"[{current_dt.isoformat()}] {stock_code}: 현금 부족으로 매수 불가. 필요: {total_cost:,.0f}원, 현재: {self.cash:,.0f}원")
                return False
        
        elif order_type == 'sell':
            if stock_code in self.positions and self.positions[stock_code]['size'] > 0:
                actual_quantity = min(quantity, self.positions[stock_code]['size'])
                commission = effective_price * actual_quantity * self.commission_rate  # 매도 시에만 커미션
                revenue = effective_price * actual_quantity - commission
                # 수익금/수익률 계산
                avg_price = self.positions[stock_code]['avg_price']
                profit = (effective_price - avg_price) * actual_quantity
                profit_rate = ((effective_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
                self.cash += revenue
                self.transaction_log.append((current_dt, stock_code, 'sell', price, actual_quantity, commission, revenue))
                self.positions[stock_code]['size'] -= actual_quantity
                if self.positions[stock_code]['size'] == 0:
                    del self.positions[stock_code]
                logging.info(f"[{current_dt.isoformat()}] {stock_code}: {actual_quantity}주 매도. 실제 가격: {effective_price:,.0f}원, 수익금: {profit:,.0f}원, 수익률: {profit_rate:.2f}%, 수수료: {commission:,.0f}원. 매매대금: {revenue:,.0f}원")
                return True
            else:
                logging.warning(f"[{current_dt.isoformat()}] {stock_code}: 보유하지 않은 종목 또는 수량 부족으로 매도 불가.")
                return False
        return False

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

    def _update_highest_price(self, stock_code, current_price):
        """포지션의 최고가를 업데이트합니다 (트레일링 스탑을 위해)."""
        if stock_code in self.positions: # Broker의 positions를 직접 사용
            if current_price > self.positions[stock_code]['highest_price']:
                self.positions[stock_code]['highest_price'] = current_price

    def _calculate_loss_ratio(self, current_price: float, avg_price: float) -> float:
        """손실률을 계산합니다. 퍼센트 단위로 반환합니다."""
        return (current_price - avg_price) / avg_price * 100

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
        if self.stop_loss_params is not None and self.stop_loss_params['stop_loss_ratio'] is not None and loss_ratio <= self.stop_loss_params['stop_loss_ratio']:
            logging.info(f"[개별 손절매 발생] {stock_code}: 현재 손실률 {loss_ratio:.2f}%가 기준치 {self.stop_loss_params['stop_loss_ratio']}%를 초과. {current_dt.isoformat()}")
            self.execute_order(stock_code, 'sell', current_price, pos_info['size'], current_dt)
            return True
        
        # 2. 트레일링 스탑 (trailing_stop_ratio)
        if self.stop_loss_params is not None and self.stop_loss_params['trailing_stop_ratio'] is not None and highest_price > 0:
            trailing_loss_ratio = self._calculate_loss_ratio(current_price, highest_price)
            if trailing_loss_ratio <= self.stop_loss_params['trailing_stop_ratio']:
                logging.info(f"[트레일링 스탑 발생] {stock_code}: 현재가 {current_price:,.0f}원이 최고가 {highest_price:,.0f}원 대비 {trailing_loss_ratio:.2f}% 하락. {current_dt.isoformat()}")
                self.execute_order(stock_code, 'sell', current_price, pos_info['size'], current_dt)
                return True
        
        # 3. 보유 기간 기반 손절 (early_stop_loss)
        if self.stop_loss_params is not None and self.stop_loss_params['early_stop_loss'] is not None and pos_info['entry_date'] is not None:
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
        if self.stop_loss_params is not None and self.stop_loss_params['portfolio_stop_loss'] is not None:
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
                    

        # 2. 동시다발적 손실 기준 (max_losing_positions)
        if self.stop_loss_params is not None and self.stop_loss_params['max_losing_positions'] is not None:
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
                
        return False

    def _execute_portfolio_sellout(self, current_prices: Dict[str, float], current_dt: datetime):
        """포트폴리오 전체 청산을 실행합니다."""
        for stock_code in list(self.positions.keys()):
            self.execute_order(stock_code, 'sell', current_prices[stock_code], self.positions[stock_code]['size'], current_dt)

            # if self.positions[stock_code]['size'] > 0:
            #     if stock_code in current_prices: #################### 종목코드가 없는 경우가 있어서 안전 처리리
            #         self.execute_order(stock_code, 'sell', current_prices[stock_code], self.positions[stock_code]['size'], current_dt)
            #         return True
            #     else:
            #         logging.warning(f"[포트폴리오 청산] {stock_code}의 현재가 정보가 없어 매도 실행을 건너뜁니다.")
            #         return False
                
    def reset_daily_transactions(self):
        """일일 거래 초기화를 수행합니다. 현재는 빈 메서드로 두어 향후 확장 가능하도록 합니다."""
        # 일일 거래 관련 상태를 초기화하는 로직이 필요할 때 여기에 추가
        # 예: 일일 거래 횟수 제한, 일일 손실 한도 등
        pass
