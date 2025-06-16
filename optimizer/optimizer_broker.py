"""
거래 실행을 시뮬레이션하는 OptimizerBroker 클래스
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

class OptimizerBroker:
    def __init__(self, initial_cash: float = 100000000, commission_rate: float = 0.0003):
        """
        브로커 초기화
        
        Args:
            initial_cash: 초기 자본금 (기본값: 1억원)
            commission_rate: 수수료율 (기본값: 0.03%)
        """
        self.logger = logging.getLogger(__name__)
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Dict] = {}  # {종목코드: {수량, 평균단가}}
        self.trades: List[Dict] = []  # 거래 내역
        self.commission_rate = commission_rate
        self.logger.info(f"옵티마이저 브로커 초기화: 초기 현금 {self.cash:,.0f}원, 수수료율 {self.commission_rate*100:.2f}%")
        
    def get_position_size(self, stock_code: str) -> int:
        """현재 포지션 크기 조회"""
        if stock_code in self.positions:
            return self.positions[stock_code]['quantity']
        return 0
    
    def get_position_value(self, stock_code: str, current_price: float) -> float:
        """현재 포지션 가치 계산"""
        if stock_code in self.positions:
            return self.positions[stock_code]['quantity'] * current_price
        return 0.0
    
    def get_portfolio_value(self, stock_prices: Dict[str, float]) -> float:
        """현재 포트폴리오 가치 계산"""
        total_value = self.cash
        for stock_code, price in stock_prices.items():
            total_value += self.get_position_value(stock_code, price)
        return total_value
    
    def execute_trade(
        self,
        stock_code: str,
        signal: int,
        price: float,
        timestamp: datetime,
        quantity: Optional[int] = None
    ) -> bool:
        """
        거래 실행
        
        Args:
            stock_code: 종목 코드
            signal: 매수(1) 또는 매도(-1) 신호
            price: 거래 가격
            timestamp: 거래 시점
            quantity: 거래 수량 (None이면 자동 계산)
            
        Returns:
            거래 성공 여부
        """
        try:
            if signal == 0:  # 관망
                return True
                
            if quantity is None:
                # 기본 거래 수량 계산 (현금의 10%)
                quantity = int((self.cash * 0.1) / price / 100) * 100
                
            if quantity <= 0:
                self.logger.warning(f"거래 수량이 0 이하입니다: {quantity}")
                return False
                
            # 매수
            if signal > 0:
                cost = price * quantity
                if cost > self.cash:
                    self.logger.warning(f"잔고 부족: 필요 {cost:,.0f}원, 보유 {self.cash:,.0f}원")
                    return False
                    
                self.cash -= cost
                if stock_code in self.positions:
                    # 평균단가 재계산
                    current_quantity = self.positions[stock_code]['quantity']
                    current_cost = self.positions[stock_code]['avg_price'] * current_quantity
                    new_quantity = current_quantity + quantity
                    new_avg_price = (current_cost + cost) / new_quantity
                    self.positions[stock_code] = {
                        'quantity': new_quantity,
                        'avg_price': new_avg_price
                    }
                else:
                    self.positions[stock_code] = {
                        'quantity': quantity,
                        'avg_price': price
                    }
                    
            # 매도
            elif signal < 0:
                if stock_code not in self.positions:
                    self.logger.warning(f"보유하지 않은 종목 매도 시도: {stock_code}")
                    return False
                    
                current_quantity = self.positions[stock_code]['quantity']
                if quantity > current_quantity:
                    self.logger.warning(f"매도 수량이 보유 수량을 초과합니다: {quantity} > {current_quantity}")
                    return False
                    
                revenue = price * quantity
                self.cash += revenue
                
                if quantity == current_quantity:
                    del self.positions[stock_code]
                else:
                    self.positions[stock_code]['quantity'] -= quantity
            
            # 거래 내역 기록
            self.trades.append({
                'timestamp': timestamp,
                'stock_code': stock_code,
                'signal': signal,
                'price': price,
                'quantity': quantity,
                'cash': self.cash
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"거래 실행 중 오류 발생: {str(e)}", exc_info=True)
            return False
    
    def get_trade_history(self) -> pd.DataFrame:
        """거래 내역 조회"""
        if not self.trades:
            return pd.DataFrame()
            
        return pd.DataFrame(self.trades)
    
    def reset(self):
        """브로커 상태 초기화"""
        self.cash = self.initial_cash
        self.positions.clear()
        self.trades.clear() 