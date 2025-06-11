import abc # Abstract Base Class
import pandas as pd
import datetime

class BaseStrategy(abc.ABC):
    """모든 전략의 기반이 되는 추상 클래스."""
    def __init__(self, data_store, strategy_params, broker, position_info=None):
        self.data_store = data_store
        self.strategy_params = strategy_params
        self.broker = broker
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

    def _get_bar_at_time(self, data_type, stock_code, target_dt):
        """주어진 시간(target_dt)에 해당하는 정확한 OHLCV 바를 반환합니다."""
        if data_type == 'daily':
            df = self.data_store['daily'].get(stock_code)
            if df is None or df.empty:
                return None
            try:
                target_dt_normalized = pd.Timestamp(target_dt).normalize()
                return df.loc[target_dt_normalized]
            except KeyError:
                return None
        elif data_type == 'minute':
            target_date = target_dt.date()
            if stock_code not in self.data_store['minute'] or target_date not in self.data_store['minute'][stock_code]:
                return None
            df = self.data_store['minute'][stock_code][target_date]
            if df is None or df.empty:
                return None
            try:
                return df.loc[target_dt]
            except KeyError:
                return None
        return None

    def _get_historical_data_up_to(self, data_type, stock_code, current_dt, lookback_period=None):
        """주어진 시간(current_dt)까지의 모든 과거 데이터를 반환합니다."""
        if data_type == 'daily':
            df = self.data_store['daily'].get(stock_code)
            if df is None or df.empty:
                return pd.DataFrame()
            current_dt_normalized = pd.Timestamp(current_dt).normalize()
            filtered_df = df.loc[df.index.normalize() <= current_dt_normalized]
            if lookback_period:
                return filtered_df.tail(lookback_period)
            return filtered_df
        
        elif data_type == 'minute':
            all_minute_dfs_for_stock = []
            if stock_code in self.data_store['minute']:
                for date_key in sorted(self.data_store['minute'][stock_code].keys()):
                    if date_key <= current_dt.date():
                        all_minute_dfs_for_stock.append(self.data_store['minute'][stock_code][date_key])
            
            if not all_minute_dfs_for_stock:
                return pd.DataFrame()
            
            combined_minute_df = pd.concat(all_minute_dfs_for_stock).sort_index()
            filtered_df = combined_minute_df.loc[combined_minute_df.index <= current_dt]
            if lookback_period:
                return filtered_df.tail(lookback_period)
            return filtered_df
        return pd.DataFrame()


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
    
    def run_minute_logic(self, stock_code, current_minute_dt):
        """일봉 전략은 분봉 로직을 직접 수행하지 않으므로 이 메서드는 비워둡니다."""
        pass 

    def update_signals(self, signals):
        """일봉 전략은 외부 시그널을 업데이트 받을 필요가 없을 수 있으므로 이 메서드는 비워둡니다."""
        pass


class MinuteStrategy(BaseStrategy):
    """분봉 전략을 위한 추상 클래스."""
    def __init__(self, data_store, strategy_params, broker, position_info=None):
        super().__init__(data_store, strategy_params, broker, position_info)
        self.signals = {} # 일봉 전략으로부터 받을 시그널을 저장할 속성

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