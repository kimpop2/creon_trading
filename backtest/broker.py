# broker.py

import datetime
import logging
import numpy as np
import pandas as pd # Removed if not strictly necessary for this class, but kept for safety

class Broker:
    def __init__(self, initial_cash, commission_rate):
        self.cash = initial_cash
        self.positions = {}  # {stock_code: {'size': int, 'avg_price': float, 'entry_date': datetime.date}}
        self.commission_rate = commission_rate
        self.trade_log = []  # 매매 기록

    def get_position_size(self, stock_code):
        """특정 종목의 현재 보유 수량을 반환합니다."""
        return self.positions.get(stock_code, {}).get('size', 0)

    def get_portfolio_value(self, current_prices):
        """현재 현금과 보유 포지션 평가액을 합산하여 포트폴리오 가치 반환합니다."""
        total_value = self.cash
        for stock_code, pos in self.positions.items():
            if pos['size'] > 0 and stock_code in current_prices:
                total_value += pos['size'] * current_prices[stock_code]
        return total_value

    def _calculate_commission(self, price, size):
        """거래 수수료를 계산합니다."""
        return price * size * self.commission_rate

    def execute_order(self, stock_code, order_type, price, size, current_time):
        """
        주문 실행을 시뮬레이션하고 현금/포지션을 업데이트합니다.
        order_type: 'buy' 또는 'sell'
        """
        commission = self._calculate_commission(price, size)
        
        log_msg = ""
        success = False

        if order_type == 'buy':
            total_cost = price * size + commission
            if self.cash >= total_cost:
                self.cash -= total_cost
                current_size = self.positions.get(stock_code, {'size': 0})['size']
                current_avg_price = self.positions.get(stock_code, {'avg_price': 0})['avg_price']

                new_size = current_size + size
                new_avg_price = ((current_avg_price * current_size) + (price * size)) / new_size if new_size > 0 else 0
                
                self.positions[stock_code] = {
                    'size': new_size, 
                    'avg_price': new_avg_price,
                    'entry_date': current_time.date()  # 매수 날짜 기록
                }
                
                log_msg = (f"매수 체결: {stock_code}, 수량: {size}주, 가격: {price:,.0f}원, "
                           f"수수료: {commission:,.0f}원, 현금잔고: {self.cash:,.0f}원, 보유수량: {new_size}주")
                success = True
            else:
                log_msg = f"매수 실패: {stock_code} - 현금 부족 (필요: {total_cost:,.0f}원, 보유: {self.cash:,.0f}원)"

        elif order_type == 'sell':
            current_size = self.positions.get(stock_code, {'size': 0})['size']
            if current_size >= size:
                current_avg_price = self.positions.get(stock_code, {'avg_price': 0})['avg_price']
                
                pnl = (price - current_avg_price) * size - commission
                self.cash += (price * size - commission)

                new_size = current_size - size
                if new_size == 0:
                    del self.positions[stock_code]
                    log_msg = (f"매도 체결: {stock_code}, 수량: {size}주, 가격: {price:,.0f}원, "
                               f"수수료: {commission:,.0f}원, 손익: {pnl:,.0f}원, 현금잔고: {self.cash:,.0f}원, 포지션: 청산완료")
                else:
                    self.positions[stock_code]['size'] = new_size
                    log_msg = (f"매도 체결: {stock_code}, 수량: {size}주, 가격: {price:,.0f}원, "
                               f"수수료: {commission:,.0f}원, 손익: {pnl:,.0f}원, 현금잔고: {self.cash:,.0f}원, 잔여수량: {new_size}주")
                success = True
            else:
                log_msg = f"매도 실패: {stock_code} - 보유수량 부족 (현재: {current_size}주, 요청: {size}주)"
        
        if success:
            logging.info(f"{current_time.isoformat()} - {log_msg}")
            self.trade_log.append({
                'datetime': current_time,
                'stock_code': stock_code,
                'order_type': order_type,
                'price': price,
                'size': size,
                'commission': commission,
                'cash_after_trade': self.cash,
                'position_size_after_trade': self.get_position_size(stock_code)
            })
        else:
            logging.warning(f"{current_time.isoformat()} - {log_msg}")
        return success