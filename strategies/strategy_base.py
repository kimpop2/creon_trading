from abc import ABC, abstractmethod
import pandas as pd
import datetime
class Strategy(ABC):
    """
    모든 백테스팅 전략의 기본 인터페이스를 정의하는 추상 클래스.
    각 전략은 이 클래스를 상속받아 구체적인 로직을 구현해야 합니다.
    """
    def __init__(self, data_store: dict, strategy_params: dict, broker): # <--- 이 부분을 수정!
            """
            모든 전략의 기본 생성자입니다.
            Args:
                data_store (dict): 시장 데이터(일봉, 분봉 등)를 담고 있는 딕셔너리.
                strategy_params (dict): 전략별 매개변수 딕셔너리.
                broker: 거래 실행을 위한 Broker 인스턴스.
            """
            self.data_store = data_store
            self.strategy_params = strategy_params
            self.broker = broker
            # 공통적으로 필요한 다른 속성들도 여기서 초기화할 수 있습니다.
            self.positions = self.broker.positions # 현재 포지션에 접근
            self.last_rebalance_date = None # 일봉 전략에서 사용할 수 있음
            
    @abstractmethod
    def generate_signals(self, market_data: dict, current_date: datetime.date, **kwargs) -> dict:
        """
        주어진 시장 데이터(일봉 또는 분봉)를 기반으로 거래 신호를 생성합니다.
        이 메서드는 각 전략 클래스에서 반드시 구현되어야 합니다.

        Args:
            market_data (dict): 전략 분석에 필요한 시장 데이터.
                                일봉 전략의 경우 {stock_code: pd.DataFrame(daily_data)} 형태,
                                분봉 전략의 경우 {stock_code: pd.DataFrame(minute_data)} 형태가 될 수 있습니다.
            current_date (datetime.date): 현재 백테스팅 날짜 (일봉 전략용).
            **kwargs: 전략 실행에 필요한 추가적인 정보 (예: current_portfolio_positions 등).

        Returns:
            dict: { '종목코드': {'signal': 'BUY' | 'SELL' | 'HOLD', 'quantity': int, ...} } 형태의 거래 신호 딕셔너리.
                  예: {'005930': {'signal': 'BUY', 'quantity': 10}}
        """
        pass