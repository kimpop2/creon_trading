import abc # Abstract Base Class
import pandas as pd
import datetime
import logging
# --- 로거 설정 (스크립트 최상단에서 설정하여 항상 보이도록 함) ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # 테스트 시 DEBUG로 설정하여 모든 로그 출력
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

    def _calculate_target_quantity(self, stock_code, current_price, num_stocks=None):
        """
        주어진 가격에서 동일비중 투자에 필요한 수량을 계산합니다.
        
        Args:
            stock_code (str): 종목 코드
            current_price (float): 현재 가격
            num_stocks (int, optional): 분배할 종목 수. None인 경우 strategy_params['num_top_stocks'] 사용
            
        Returns:
            int: 매수 가능한 주식 수량
        """
        # 분배할 종목 수 결정
        if num_stocks is None:
            num_stocks = self.strategy_params.get('num_top_stocks', 1)
        
        # 목표 투자 금액 계산
        target_amount = self.broker.cash / num_stocks
        
        # 현재 보유 현금 확인
        available_cash = self.broker.cash
        
        # 수수료를 고려한 실제 투자 가능 금액 계산
        commission_rate = self.broker.commission_rate
        max_buyable_amount = available_cash / (1 + commission_rate)
        
        # 목표 투자금액과 실제 투자 가능 금액 중 작은 값 선택
        actual_investment_amount = min(target_amount, max_buyable_amount)
        
        # 주식 수량 계산 (소수점 이하 버림)
        quantity = int(actual_investment_amount / current_price)
        
        if quantity > 0:
            # 실제 필요한 현금 (수수료 포함) 계산
            total_cost = current_price * quantity * (1 + commission_rate)
            if total_cost > available_cash:
                # 수량을 1주 줄여서 재계산 (최소 거래 단위 1주이므로)
                quantity -= 1
        
        if quantity > 0:
            logging.info(f"{stock_code} 종목 매수 수량 계산: {quantity}주 (목표금액: {target_amount:,.0f}원, 가용현금: {available_cash:,.0f}원, 현재가: {current_price:,.0f}원)")
        else:
            logging.warning(f"{stock_code} 종목 매수 불가: 현금 부족 (목표금액: {target_amount:,.0f}원, 가용현금: {available_cash:,.0f}원, 현재가: {current_price:,.0f}원)")
        
        return quantity

    def _log_rebalancing_summary(self, current_daily_date, buy_candidates, current_positions):
        """
        리밸런싱 계획을 요약하여 로깅합니다.
        
        Args:
            current_daily_date (datetime): 현재 날짜
            buy_candidates (set): 매수 대상 종목 코드 집합
            current_positions (set): 현재 보유 중인 종목 코드 집합
        """
        # 현재가 정보 수집
        current_prices_for_summary = {}
        for stock_code in self.data_store['daily']:
            daily_data = self._get_historical_data_up_to('daily', stock_code, current_daily_date, lookback_period=1)
            if not daily_data.empty:
                current_prices_for_summary[stock_code] = daily_data['close'].iloc[-1]

        # 포트폴리오 가치 계산
        portfolio_value = self.broker.get_portfolio_value(current_prices_for_summary)
        current_holdings = [(code, pos['size'] * current_prices_for_summary[code]) 
                          for code, pos in self.broker.positions.items() 
                          if code in current_prices_for_summary and pos['size'] > 0]
        total_holdings_value = sum(value for _, value in current_holdings)
        
        # 매수 계획 계산
        new_buys = [(code, self.signals[code]['target_quantity'] * current_prices_for_summary[code])
                   for code in buy_candidates 
                   if code not in current_positions and code in current_prices_for_summary]
        total_buy_amount = sum(amount for _, amount in new_buys)
        
        # 매도 계획 계산
        to_sell = [(code, pos['size'] * current_prices_for_summary[code])
                  for code, pos in self.broker.positions.items()
                  if code not in buy_candidates and code in current_prices_for_summary]
        total_sell_amount = sum(amount for _, amount in to_sell)

        # 리밸런싱 계획 로깅
        logging.info("\n=== 리밸런싱 계획 요약 ===")
        logging.info(f"현재 상태: 포트폴리오 가치 {portfolio_value:,.0f}원 = 보유종목 {len(current_holdings)}개 ({total_holdings_value:,.0f}원) + 현금 {self.broker.cash:,.0f}원")
        logging.info(f"매수 계획: {len(new_buys)}종목 (소요금액: {total_buy_amount:,.0f}원)")
        logging.info(f"매도 계획: {len(to_sell)}종목 (회수금액: {total_sell_amount:,.0f}원)")



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

