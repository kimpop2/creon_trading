import abc # Abstract Base Class
import pandas as pd
import datetime

class BaseStrategy(abc.ABC):
    """모든 전략의 기반이 되는 추상 클래스."""
    def __init__(self, data_store, strategy_params, broker, position_info=None):
        self.data_store = data_store
        self.strategy_params = strategy_params
        self.broker = broker
        # position_info는 분봉 전략에서만 주로 사용될 수 있지만, BaseStrategy에서 모든 전략이 공통적으로
        # 참조할 수 있도록 초기화 시 받도록 유도합니다.
        self.position_info = position_info if position_info is not None else {}

    @abc.abstractmethod
    def run_daily_logic(self, current_date):
        """일봉 데이터를 기반으로 전략 로직을 실행하는 추상 메서드.
        (일봉 전략에서 주로 사용하며, 분봉 전략에서는 pass로 구현될 수 있습니다.)
        """
        pass

    @abc.abstractmethod
    def run_minute_logic(self, stock_code, current_minute_dt):
        """분봉 데이터를 기반으로 전략 로직을 실행하는 추상 메서드.
        (분봉 전략에서 주로 사용하며, 일봉 전략에서는 pass로 구현될 수 있습니다.)
        """
        pass

    @abc.abstractmethod
    def update_signals(self, signals):
        """다른 전략 (예: 일봉 전략)으로부터 시그널을 업데이트받는 추상 메서드.
        (분봉 전략에서 주로 사용하며, 일봉 전략에서는 pass로 구현될 수 있습니다.)
        """
        pass


class DailyStrategy(BaseStrategy):
    """일봉 전략을 위한 추상 클래스."""
    def __init__(self, data_store, strategy_params, broker, position_info=None):
        super().__init__(data_store, strategy_params, broker, position_info)

    @abc.abstractmethod
    def run_daily_logic(self, current_date):
        """일봉 데이터를 기반으로 전략 로직을 실행하고 매매 의도를 반환합니다.
        예: 듀얼 모멘텀의 매수/매도 종목 선정
        """
        pass
    
    # DailyStrategy는 분봉 로직을 직접 처리하지 않으므로, 추상 메서드를 구체적으로 구현합니다.
    def run_minute_logic(self, stock_code, current_minute_dt):
        """일봉 전략은 분봉 로직을 직접 수행하지 않으므로 이 메서드는 비워둡니다."""
        pass 

    # DailyStrategy는 주로 시그널을 생성하므로, 외부 시그널을 업데이트 받을 필요가 없을 수 있습니다.
    def update_signals(self, signals):
        """일봉 전략은 외부 시그널을 업데이트 받을 필요가 없을 수 있으므로 이 메서드는 비워둡니다."""
        pass


class MinuteStrategy(BaseStrategy):
    """분봉 전략을 위한 추상 클래스."""
    def __init__(self, data_store, strategy_params, broker, position_info=None):
        super().__init__(data_store, strategy_params, broker, position_info)
        self.signals = {} # 일봉 전략으로부터 받을 시그널을 저장할 속성

    # MinuteStrategy는 일봉 로직을 직접 처리하지 않으므로, 추상 메서드를 구체적으로 구현합니다.
    def run_daily_logic(self, current_date):
        """분봉 전략은 일봉 로직을 직접 수행하지 않으므로 이 메서드는 비워둡니다."""
        pass 

    @abc.abstractmethod
    def run_minute_logic(self, stock_code, current_minute_dt):
        """분봉 데이터를 기반으로 매매 의도에 따라 실제 매매 주문을 실행합니다.
        예: RSI 기반의 실제 매수/매도 주문 실행
        """
        pass

    @abc.abstractmethod
    def update_signals(self, signals):
        """일봉 전략으로부터 받은 매매 시그널을 업데이트합니다."""
        pass