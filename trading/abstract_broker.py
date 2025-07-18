# trading/abstract_broker.py

import abc
from datetime import datetime
from typing import Dict, Any, Optional, List

class AbstractBroker(abc.ABC):
    """
    실제 매매 실행 또는 백테스팅 시뮬레이션을 위한 브로커 인터페이스를 정의하는 추상 기본 클래스입니다.
    모든 브로커 구현체는 이 클래스의 추상 메서드를 반드시 구현해야 합니다.
    """

    def __init__(self):
        self.stop_loss_params: Optional[Dict[str, Any]] = None
        self.initial_cash: float = 0.0
        self.commission_rate: float = 0.0

    @abc.abstractmethod
    def set_stop_loss_params(self, stop_loss_params: Optional[Dict[str, Any]]):
        """손절매 관련 파라미터를 설정합니다."""
        pass

    @abc.abstractmethod
    def execute_order(self,
                      stock_code: str,
                      order_type: str, # 'buy' 또는 'sell'
                      price: float,
                      quantity: int,
                      order_time: datetime,
                      order_id: Optional[str] = None
                     ) -> Optional[str]: # 성공 시 주문 ID(str), 실패 시 None
        """
        주문을 실행합니다. 실제 매매에서는 API를 호출하고, 백테스팅에서는 가상으로 처리합니다.
        """
        pass

    @abc.abstractmethod
    def get_current_cash_balance(self) -> float:
        """현재 현금 잔고를 조회합니다."""
        pass

    @abc.abstractmethod
    def get_current_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        현재 보유 종목 정보를 딕셔너리 형태로 조회합니다.
        반환 형식: {stock_code: {'size': int, 'avg_price': float, ...}}
        """
        pass

    @abc.abstractmethod
    def get_position_size(self, stock_code: str) -> int:
        """특정 종목의 보유 수량을 조회합니다."""
        pass

    @abc.abstractmethod
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """현재 포트폴리오의 총 가치를 계산하여 반환합니다."""
        pass

    @abc.abstractmethod
    def check_and_execute_stop_loss(self, current_prices: Dict[str, float], current_dt: datetime) -> bool:
        """
        설정된 손절매 조건을 모든 보유 포지션에 대해 확인하고 매도 주문을 실행합니다.
        하나라도 손절이 실행되면 True를 반환합니다.
        """
        pass

    @abc.abstractmethod
    def cleanup(self) -> None:
        """브로커가 사용한 리소스를 정리합니다."""
        pass