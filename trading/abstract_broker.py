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
        pass

    @abc.abstractmethod
    def execute_order(self,
                      stock_code: str,
                      order_type: str, # 'buy', 'sell'
                      price: float,
                      quantity: int,
                      order_time: datetime,
                      order_id: Optional[str] = None # 주문 정정/취소 시 사용될 수 있는 원주문번호
                     ) -> Optional[str]: # 체결 또는 접수된 주문번호 (또는 자체 생성 ID) 반환
        """
        주문을 실행합니다. 실제 매매에서는 증권사 API를 호출하고,
        백테스팅에서는 가상으로 주문을 처리합니다.
        """
        pass

    # @abc.abstractmethod
    # def cancel_order(self, order_id: str) -> bool:
    #     """
    #     진행 중인 주문을 취소합니다.
    #     """
    #     pass

    # @abc.abstractmethod
    # def amend_order(self,
    #                 order_id: str,
    #                 new_price: Optional[float] = None,
    #                 new_quantity: Optional[int] = None
    #                ) -> Optional[str]:
    #     """
    #     진행 중인 주문을 정정합니다.
    #     """
    #     pass

    # @abc.abstractmethod
    # def get_current_cash_balance(self) -> float:
    #     """
    #     현재 현금 잔고를 조회합니다.
    #     """
    #     pass

    # @abc.abstractmethod
    # def get_current_positions(self) -> Dict[str, Any]:
    #     """
    #     현재 보유 종목 정보를 조회합니다.
    #     반환 형식: {stock_code: {'quantity': int, 'average_buy_price': float, 'current_price': float, 'entry_date': date, ...}}
    #     """
    #     pass

    # @abc.abstractmethod
    # def get_unfilled_orders(self) -> List[Dict[str, Any]]:
    #     """
    #     현재 미체결 주문 내역을 조회합니다.
    #     반환 형식: [{'order_id': '...', 'stock_code': '...', 'order_type': 'buy/sell', 'order_price': ..., 'order_quantity': ..., 'unfilled_quantity': ..., ...}]
    #     """
    #     pass
    
    # @abc.abstractmethod
    # def update_portfolio_status(self, current_dt: datetime) -> None:
    #     """
    #     매일 또는 특정 시점에 포트폴리오 상태를 업데이트하고 DB에 저장합니다.
    #     """
    #     pass

    @abc.abstractmethod
    def check_and_execute_stop_loss(self, current_prices: Dict[str, float], current_dt: datetime) -> bool:
        """
        설정된 손절매 조건을 확인하고 해당되는 경우 매도 주문을 실행합니다.
        """
        pass
    
    @abc.abstractmethod
    def cleanup(self) -> None:
        """브로커 리소스 정리 """
        pass
    