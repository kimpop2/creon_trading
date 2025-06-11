import datetime
import logging

class Broker:
    def __init__(self, initial_cash, commission_rate=0.0003, slippage_rate=0.0):
        self.cash = initial_cash
        # positions 딕셔너리 구조 변경: highest_price와 entry_date는 이제 Broker가 관리
        self.positions = {}  # {stock_code: {'size': int, 'avg_price': float, 'entry_date': datetime.date, 'highest_price': float}}
        self.transaction_log = [] # (date, stock_code, type, price, quantity, commission, net_amount)
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        logging.info(f"브로커 초기화: 초기 현금 {self.cash:,.0f}원, 수수료율 {self.commission_rate*100:.2f}%")

        # 손절매 관련 파라미터
        self.stop_loss_ratio = None
        self.trailing_stop_ratio = None
        self.early_stop_loss = None
        self.portfolio_stop_loss = None
        self.max_losing_positions = None
        self.initial_portfolio_value = initial_cash # 포트폴리오 손절을 위한 초기값

    def set_stop_loss_params(self, params):
        """손절매 관련 파라미터를 설정합니다."""
        if params is None:
            return
        self.stop_loss_ratio = params.get('stop_loss_ratio')
        self.trailing_stop_ratio = params.get('trailing_stop_ratio')
        self.early_stop_loss = params.get('early_stop_loss')
        self.portfolio_stop_loss = params.get('portfolio_stop_loss')
        self.max_losing_positions = params.get('max_losing_positions')
        logging.info(f"브로커 손절매 파라미터 설정 완료: {params}")

    def execute_order(self, stock_code, order_type, price, quantity, current_dt):
        """매매 주문을 실행합니다."""
        if quantity <= 0:
            logging.warning(f"[{current_dt.isoformat()}] {stock_code}: {order_type} 수량 0. 주문 실행하지 않음.")
            return False

        effective_price = price * (1 + self.slippage_rate if order_type == 'buy' else 1 - self.slippage_rate)
        commission = effective_price * quantity * self.commission_rate

        if order_type == 'buy':
            total_cost = effective_price * quantity + commission
            if self.cash >= total_cost:
                # 기존 포지션이 있으면 평균 단가 계산
                if stock_code in self.positions and self.positions[stock_code]['size'] > 0:
                    current_size = self.positions[stock_code]['size']
                    current_avg_price = self.positions[stock_code]['avg_price']
                    new_size = current_size + quantity
                    new_avg_price = (current_avg_price * current_size + effective_price * quantity) / new_size
                    self.positions[stock_code]['size'] = new_size
                    self.positions[stock_code]['avg_price'] = new_avg_price
                    # 매수 시점에 최고가도 현재가로 초기화 (트레일링 스탑을 위해)
                    # 일봉 종가 손절에서는 highest_price 업데이트가 일봉 종가 시점에 이루어져야 함
                    # 여기서는 일단 매수 가격으로 초기화
                    self.positions[stock_code]['highest_price'] = effective_price
                else:
                    self.positions[stock_code] = {
                        'size': quantity,
                        'avg_price': effective_price,
                        'entry_date': current_dt.date(),
                        'highest_price': effective_price # 초기 최고가
                    }
                self.cash -= total_cost
                self.transaction_log.append((current_dt, stock_code, 'buy', price, quantity, commission, total_cost))
                logging.info(f"[{current_dt.isoformat()}] {stock_code}: {quantity}주 매수. 가격: {price:,.0f}원 (실제: {effective_price:,.0f}원), 수수료: {commission:,.0f}원. 남은 현금: {self.cash:,.0f}원")
                return True
            else:
                logging.warning(f"[{current_dt.isoformat()}] {stock_code}: 현금 부족으로 매수 불가. 필요: {total_cost:,.0f}원, 현재: {self.cash:,.0f}원")
                return False
        
        elif order_type == 'sell':
            if stock_code in self.positions and self.positions[stock_code]['size'] > 0:
                # 매도 수량이 현재 보유 수량보다 많으면, 보유 수량만큼만 매도
                actual_quantity = min(quantity, self.positions[stock_code]['size'])
                
                revenue = effective_price * actual_quantity - commission
                self.cash += revenue
                self.transaction_log.append((current_dt, stock_code, 'sell', price, actual_quantity, commission, revenue))

                self.positions[stock_code]['size'] -= actual_quantity
                if self.positions[stock_code]['size'] == 0:
                    del self.positions[stock_code] # 포지션 청산

                logging.info(f"[{current_dt.isoformat()}] {stock_code}: {actual_quantity}주 매도. 가격: {price:,.0f}원 (실제: {effective_price:,.0f}원), 수수료: {commission:,.0f}원. 남은 현금: {self.cash:,.0f}원")
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

    def update_daily_highest_price(self, stock_code, daily_close_price):
        """매일 일봉 종가를 기반으로 포지션의 최고가를 업데이트합니다."""
        if stock_code in self.positions:
            if daily_close_price is not None and daily_close_price > self.positions[stock_code]['highest_price']:
                self.positions[stock_code]['highest_price'] = daily_close_price

    def check_and_execute_stop_loss(self, stock_code, daily_close_price, current_dt):
        """
        개별 종목에 대한 손절 조건을 일봉 종가 기준으로 체크하고, 조건 충족 시 매도 주문을 실행합니다.
        """
        if stock_code not in self.positions or self.positions[stock_code]['size'] <= 0:
            return False # 보유하고 있지 않으면 손절 체크할 필요 없음
        if daily_close_price is None:
            logging.warning(f"[{current_dt.isoformat()}] {stock_code}: 일봉 종가 데이터가 없어 개별 손절 체크를 건너뜜.")
            return False

        pos_info = self.positions[stock_code]
        avg_price = pos_info['avg_price']
        highest_price = pos_info['highest_price'] # 이미 update_daily_highest_price로 업데이트됨

        loss_ratio = (daily_close_price - avg_price) / avg_price * 100
        
        # 1. 단순 손절 (stop_loss_ratio)
        if self.stop_loss_ratio is not None and loss_ratio <= self.stop_loss_ratio:
            logging.info(f"[개별 손절매 발생] {stock_code}: 일봉 종가 {daily_close_price:,.0f}원. 현재 손실률 {loss_ratio:.2f}%가 기준치 {self.stop_loss_ratio}%를 초과. {current_dt.isoformat()}")
            self.execute_order(stock_code, 'sell', daily_close_price, pos_info['size'], current_dt)
            return True
            
        # 2. 트레일링 스탑 (trailing_stop_ratio)
        if self.trailing_stop_ratio is not None and highest_price > 0: # highest_price가 초기화되지 않은 경우 방지
            trailing_loss_ratio = (daily_close_price - highest_price) / highest_price * 100
            if trailing_loss_ratio <= self.trailing_stop_ratio:
                logging.info(f"[트레일링 스탑 발생] {stock_code}: 일봉 종가 {daily_close_price:,.0f}원이 최고가 {highest_price:,.0f}원 대비 {trailing_loss_ratio:.2f}% 하락. {current_dt.isoformat()}")
                self.execute_order(stock_code, 'sell', daily_close_price, pos_info['size'], current_dt)
                return True
        
        # 3. 보유 기간 기반 손절 폭 조정 (early_stop_loss)
        if self.early_stop_loss is not None and pos_info['entry_date'] is not None:
            holding_days = (current_dt.date() - pos_info['entry_date']).days
            if holding_days <= 5 and loss_ratio <= self.early_stop_loss: # 매수 후 5일 이내
                logging.info(f"[조기 손절매 발생] {stock_code}: 매수 후 {holding_days}일 이내 일봉 손실률 {loss_ratio:.2f}%가 조기 손절 기준 {self.early_stop_loss}% 초과. {current_dt.isoformat()}")
                self.execute_order(stock_code, 'sell', daily_close_price, pos_info['size'], current_dt)
                return True
        
        return False

    def check_and_execute_portfolio_stop_loss(self, daily_close_prices, current_dt):
        """
        포트폴리오 전체 손절 조건을 일봉 종가 기준으로 체크하고, 조건 충족 시 모든 포지션을 청산합니다.
        """
        if not self.positions: # 보유 종목이 없으면 체크할 필요 없음
            return False

        # 1. 전체 손실폭 기준 (portfolio_stop_loss)
        if self.portfolio_stop_loss is not None:
            portfolio_value = self.get_portfolio_value(daily_close_prices) # 일봉 종가로 계산
            total_loss_ratio = (portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value * 100
            
            if total_loss_ratio <= self.portfolio_stop_loss:
                logging.info(f"[포트폴리오 손절매 발생] 전체 손실률 {total_loss_ratio:.2f}%가 기준치 {self.portfolio_stop_loss}% 초과. 모든 포지션 청산. {current_dt.isoformat()}")
                for stock_code, pos in list(self.positions.items()): 
                    if pos['size'] > 0:
                        price = daily_close_prices.get(stock_code, pos['avg_price']) # 현재가 없으면 평균단가로 매도 시도
                        logging.info(f"[포트폴리오 손절매 실행] {current_dt.isoformat()} - {stock_code} 매도. 가격: {price:,.0f}원")
                        self.execute_order(stock_code, 'sell', price, pos['size'], current_dt)
                return True
                
        # 2. 동시다발적 손실 기준 (max_losing_positions)
        if self.max_losing_positions is not None:
            losing_positions_count = 0
            # 브로커에 설정된 stop_loss_ratio를 사용하여 개별 종목의 손실 여부 판단
            if self.stop_loss_ratio is None: 
                logging.warning("max_losing_positions 사용을 위해서는 stop_loss_ratio가 설정되어야 합니다.")
                return False

            for stock_code, pos_info in self.positions.items():
                if stock_code in daily_close_prices and pos_info['size'] > 0:
                    loss_ratio = (daily_close_prices[stock_code] - pos_info['avg_price']) / pos_info['avg_price'] * 100
                    if loss_ratio <= self.stop_loss_ratio: # 개별 손절 기준과 동일하게 적용
                        losing_positions_count += 1
            
            if losing_positions_count >= self.max_losing_positions:
                logging.info(f"[포트폴리오 손절매 발생] 동시 손실 종목 수 {losing_positions_count}개가 최대 허용치 {self.max_losing_positions}개 초과. 모든 포지션 청산. {current_dt.isoformat()}")
                for stock_code, pos in list(self.positions.items()):
                    if pos['size'] > 0:
                        price = daily_close_prices.get(stock_code, pos['avg_price'])
                        logging.info(f"[포트폴리오 손절매 실행] {current_dt.isoformat()} - {stock_code} 매도. 가격: {price:,.0f}원")
                        self.execute_order(stock_code, 'sell', price, pos['size'], current_dt)
                return True
        
        return False