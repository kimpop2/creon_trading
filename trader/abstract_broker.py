from abc import ABC, abstractmethod

class AbstractBroker(ABC):
    @abstractmethod
    def execute_order(self, stock_code, order_type, price, quantity, current_dt):
        pass

    @abstractmethod
    def get_position_size(self, stock_code):
        pass

    @abstractmethod
    def get_portfolio_value(self, current_prices):
        pass

    @abstractmethod
    def set_stop_loss_params(self, stop_loss_params):
        pass

    @abstractmethod
    def check_and_execute_stop_loss(self, stock_code, current_price, current_dt):
        pass

    @abstractmethod
    def check_and_execute_portfolio_stop_loss(self, current_prices, current_dt):
        pass 