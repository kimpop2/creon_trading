# trade/brokerage.py

import logging
import pandas as pd
from datetime import datetime, date
import sys
import os
from typing import Dict, Any, List, Optional
from trader.abstract_broker import AbstractBroker
from manager.data_manager import DataManager
from api.creon_api import CreonAPIClient

# sys.path에 프로젝트 루트 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 이제 BusinessManager 대신 DataManager를 직접 사용합니다.
from manager.data_manager import DataManager
# 실제 API 클라이언트 대신, 모의(Mock) 또는 실제 API 클라이언트 인터페이스를 사용합니다.
from api.creon_api import CreonAPIClient # 실제 API 사용

logger = logging.getLogger(__name__)

class Brokerage(AbstractBroker):
    """
    실전 자동매매를 위한 증권사 브로커 클래스.
    AbstractBroker를 상속받아 실제 증권사 API 연동을 구현합니다.
    """
    def __init__(self, api_client: CreonAPIClient, data_manager: DataManager, 
                 commission_rate: float = 0.0016, slippage_rate: float = 0.0004):
        self.api_client = api_client
        self.data_manager = data_manager
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.stop_loss_params = None
        
        # 실전에서는 초기 현금과 포지션을 API에서 조회하여 동기화
        self.cash = 0.0
        # 포지션 구조를 백테스터와 통일: {stock_code: {'size': int, 'avg_price': float, 'entry_date': datetime.date, 'highest_price': float}}
        self.positions: Dict[str, Dict[str, Any]] = {}  
        self.initial_portfolio_value = 0.0 # 포트폴리오 손절을 위한 초기값 (일일 초기화 필요)
        self.transaction_log = [] 
        
        self.sync_account_info() # 초기화 시 계좌 정보 동기화
        logger.info(f"Brokerage 초기화 완료: 초기 현금 {self.cash:,.0f}원")

    def sync_account_info(self) -> bool:
        """
        증권사 API로부터 최신 현금 잔고와 보유 종목 정보를 동기화합니다.
        """
        if not self.api_client.is_connected():
            logger.error("Creon API is not connected. Cannot sync account info.")
            return False

        # 현금 잔고 동기화 (CreonAPIClient에 get_current_cash 구현 필요)
        self.cash = self.api_client.get_current_cash()
        logger.info(f"계좌 현금 잔고 동기화 완료: {self.cash:,.0f}원")

        # 보유 종목 동기화
        # CreonAPIClient의 get_account_balance가 백테스터 포맷과 유사한 딕셔너리를 반환한다고 가정
        api_positions = self.api_client.get_account_balance() 
        db_positions = self.data_manager.get_current_positions()
        
        synced_positions = {}
        for code, details in api_positions.items():
            db_info = db_positions.get(code, {})
            synced_positions[code] = {
                'size': details['quantity'],
                'avg_price': details['purchase_price'],
                # DB에 저장된 진입일과 최고가가 있다면 사용, 없다면 API 정보 기반으로 설정
                'entry_date': db_info.get('entry_date', datetime.date.today()),
                'highest_price': db_info.get('highest_price', details['purchase_price'])
            }
        
        self.positions = synced_positions
        self.data_manager.save_current_positions(self.positions) # 동기화된 포지션 DB 저장
        logger.info(f"보유 종목 정보 동기화 완료: 총 {len(self.positions)}개 종목 보유 중.")
        return True

    def execute_order(self, stock_code: str, order_type: str, price: float, quantity: int, current_dt: datetime, order_kind: str = '01'):
        """
        주문을 실행하고 거래 로그를 기록합니다.
        order_kind: '01'(보통가), '03'(시장가)
        """
        if not self.api_client.is_connected():
            logger.error("Creon API is not connected. Order execution failed.")
            return

        logger.info(f"[주문 요청] {current_dt.isoformat()} - {stock_code} {order_type.upper()} {quantity}주 @ {price:,.0f}원, 유형: {order_kind}")
        
        # 실제 주문은 CreonAPIClient를 통해 전송
        order_result = self.api_client.send_order(order_type, stock_code, quantity, price, order_kind=order_kind)

        if order_result and order_result.get('success', False):
            # 주문 성공 시, 즉시 체결되었다고 가정하고 후처리 (실제로는 미체결/부분체결 처리 필요)
            actual_price = order_result.get('executed_price', price) # 체결가
            actual_quantity = order_result.get('executed_quantity', quantity) # 체결 수량
            commission = actual_price * actual_quantity * self.commission_rate
            tax = actual_price * actual_quantity * 0.0023 if order_type == 'sell' else 0.0
            
            log_entry = {
                "trade_datetime": current_dt,
                "stock_code": stock_code,
                "order_type": order_type,
                "quantity": actual_quantity,
                "price": actual_price,
                "commission": commission,
                "tax": tax,
                "reason": "strategy_signal" # 추후 주문 사유도 파라미터로 받을 수 있음
            }
            self.transaction_log.append(log_entry)
            self.data_manager.save_trade_log(log_entry)

            # 포지션 및 현금 업데이트
            self._update_position_and_cash(stock_code, order_type, actual_price, actual_quantity, commission, tax)
            self.data_manager.save_current_positions(self.positions)
            logger.info(f"거래 후 현금 잔고: {self.cash:,.0f}원")
        else:
            logger.error(f"주문 실패: {stock_code}, 사유: {order_result.get('message', 'N/A')}")
            
    def _update_position_and_cash(self, stock_code, order_type, price, quantity, commission, tax):
        """거래 체결 후 포지션과 현금을 업데이트하는 내부 메서드"""
        if order_type == 'buy':
            if stock_code in self.positions:
                # 평균 매수 단가 재계산
                current_total_value = self.positions[stock_code]['size'] * self.positions[stock_code]['avg_price']
                new_size = self.positions[stock_code]['size'] + quantity
                new_avg_price = (current_total_value + (price * quantity)) / new_size
                self.positions[stock_code]['size'] = new_size
                self.positions[stock_code]['avg_price'] = new_avg_price
                self.positions[stock_code]['highest_price'] = max(self.positions[stock_code]['highest_price'], price)
            else:
                self.positions[stock_code] = {
                    'size': quantity,
                    'avg_price': price,
                    'entry_date': datetime.date.today(),
                    'highest_price': price
                }
            self.cash -= (price * quantity) + commission
        elif order_type == 'sell':
            if stock_code in self.positions:
                self.cash += (price * quantity) - commission - tax
                self.positions[stock_code]['size'] -= quantity
                if self.positions[stock_code]['size'] <= 0:
                    del self.positions[stock_code]
            else:
                logger.warning(f"보유하지 않은 종목 {stock_code}에 대한 매도 체결 기록. 현금만 업데이트.")


    def get_position_size(self, stock_code: str) -> int:
        """현재 보유한 특정 종목의 수량을 반환합니다."""
        return self.positions.get(stock_code, {}).get('size', 0)

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """현재 포트폴리오의 총 가치를 계산하여 반환합니다."""
        total_value = self.cash
        for stock_code, position in self.positions.items():
            price = current_prices.get(stock_code, position['avg_price']) # 현재가 없으면 평단가로 계산
            total_value += position['size'] * price
        return total_value

    def set_stop_loss_params(self, stop_loss_params: Dict[str, Any]):
        """손절매 관련 파라미터를 설정합니다."""
        if stop_loss_params:
            self.stop_loss_params = stop_loss_params
            logger.info(f"Brokerage 손절매 파라미터 설정 완료: {stop_loss_params}")

    def check_and_execute_stop_loss(self, stock_code: str, current_price: float, current_dt: datetime) -> bool:
        """
        개별 종목에 대한 손절매 조건을 확인하고, 조건 충족 시 손절매 주문을 실행합니다.
        (백테스팅의 Broker와 동일한 로직)
        """
        if not self.stop_loss_params or stock_code not in self.positions:
            return False

        position = self.positions[stock_code]
        avg_price = position['avg_price']
        highest_price = position['highest_price']
        current_size = position['size']

        if current_size <= 0: return False

        # 1. 고정 손절매
        stop_loss_ratio = self.stop_loss_params.get('stop_loss_ratio')
        if stop_loss_ratio and (current_price / avg_price - 1) < -stop_loss_ratio:
            logger.warning(f"[개별 손절매] {stock_code}: 손실율 초과. 매수가: {avg_price}, 현재가: {current_price}")
            self.execute_order(stock_code, 'sell', current_price, current_size, current_dt, order_kind='03') # 시장가 매도
            return True

        # 2. 트레일링 스탑
        trailing_stop_ratio = self.stop_loss_params.get('trailing_stop_ratio')
        if trailing_stop_ratio:
            # 최고가 갱신
            new_highest = max(highest_price, current_price)
            if new_highest > highest_price:
                self.positions[stock_code]['highest_price'] = new_highest
                self.data_manager.save_current_positions(self.positions) # 최고가 갱신 시 DB 저장

            if (current_price / new_highest - 1) < -trailing_stop_ratio:
                logger.warning(f"[트레일링 스탑] {stock_code}: 최고가({new_highest}) 대비 하락율 초과. 현재가: {current_price}")
                self.execute_order(stock_code, 'sell', current_price, current_size, current_dt, order_kind='03') # 시장가 매도
                return True
        
        return False

    def check_and_execute_portfolio_stop_loss(self, current_prices: Dict[str, float], current_dt: datetime) -> bool:
        """
        포트폴리오 전체 손절매 조건을 확인하고, 조건 충족 시 전체 청산을 실행합니다.
        (백테스팅의 Broker와 유사한 로직)
        """
        if not self.stop_loss_params or not self.initial_portfolio_value > 0:
            return False
            
        max_drawdown_ratio = self.stop_loss_params.get('portfolio_max_drawdown_ratio')
        if not max_drawdown_ratio: return False

        current_portfolio_value = self.get_portfolio_value(current_prices)
        drawdown = (current_portfolio_value / self.initial_portfolio_value - 1)
        
        if drawdown < -max_drawdown_ratio:
            logger.warning(f'[포트폴리오 손절] 최대 손실 허용치({max_drawdown_ratio*100:.2f}%) 초과. 현재 손실: {drawdown*100:.2f}%')
            self._execute_portfolio_sellout(current_prices, current_dt)
            return True
            
        return False

    def _execute_portfolio_sellout(self, current_prices: Dict[str, float], current_dt: datetime):
        """포트폴리오 전체 청산을 실행합니다."""
        logger.info(f"포트폴리오 전체 청산 실행: {current_dt.isoformat()}")
        for stock_code, position in list(self.positions.items()):
            if position['size'] > 0:
                current_price = current_prices.get(stock_code)
                if current_price:
                    self.execute_order(stock_code, 'sell', current_price, position['size'], current_dt, order_kind='03') # 시장가 매도
                else:
                    logger.error(f"청산 실패: {stock_code}의 현재가를 알 수 없어 매도할 수 없습니다.")
        logger.info("포트폴리오 전체 청산 주문 완료.")