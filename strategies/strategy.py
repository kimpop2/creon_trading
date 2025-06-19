import abc # Abstract Base Class
import pandas as pd
import datetime
import logging
from util.strategies_util import *
# --- 로거 설정 (스크립트 최상단에서 설정하여 항상 보이도록 함) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 테스트 시 DEBUG로 설정하여 모든 로그 출력 - 제거
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

    def _calculate_safe_asset_momentum(self, current_daily_date):
        """안전자산의 모멘텀을 계산합니다."""
        safe_asset_df = self.data_store['daily'].get(self.strategy_params['safe_asset_code'])
        safe_asset_momentum = 0
        if safe_asset_df is not None and not safe_asset_df.empty:
            safe_asset_data = self._get_historical_data_up_to(
                'daily',
                self.strategy_params['safe_asset_code'],
                current_daily_date,
                lookback_period=self.strategy_params['momentum_period'] + 1
            )
            if len(safe_asset_data) >= self.strategy_params['momentum_period']:
                safe_asset_momentum = calculate_momentum(safe_asset_data, self.strategy_params['momentum_period']).iloc[-1]
        return safe_asset_momentum
    
    def _calculate_momentum_scores(self, current_daily_date):
        """모든 종목의 모멘텀 스코어를 계산합니다."""
        momentum_scores = {}
        for stock_code in self.data_store['daily']:
            if stock_code == self.strategy_params['safe_asset_code']:
                continue  # 안전자산은 모멘텀 계산에서 제외

            daily_df = self.data_store['daily'][stock_code]
            if daily_df.empty:
                continue

            historical_data = self._get_historical_data_up_to(
                'daily',
                stock_code,
                current_daily_date,
                lookback_period=self.strategy_params['momentum_period'] + 1
            )

            if len(historical_data) < self.strategy_params['momentum_period']:
                logging.debug(f'{stock_code} 종목의 모멘텀 계산을 위한 데이터가 부족합니다.')
                continue

            momentum_score = calculate_momentum(historical_data, self.strategy_params['momentum_period']).iloc[-1]
            momentum_scores[stock_code] = momentum_score

        return momentum_scores

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
        # 포트폴리오 가치 기준 종목당 투자금 계산
        # 현재가 정보 수집 (일봉 데이터 기준)
        current_prices_for_summary = {}
        for code in self.data_store['daily']:
            daily_data = self._get_historical_data_up_to('daily', code, pd.Timestamp.today(), lookback_period=1)
            if not daily_data.empty:
                current_prices_for_summary[code] = daily_data['close'].iloc[-1]
        portfolio_value = self.broker.get_portfolio_value(current_prices_for_summary)
        per_stock_investment = portfolio_value / num_stocks
        available_cash = self.broker.cash
        commission_rate = self.broker.commission_rate
        max_buyable_amount = available_cash / (1 + commission_rate)
        actual_investment_amount = min(per_stock_investment, max_buyable_amount)
        quantity = int(actual_investment_amount / current_price)
        if quantity > 0:
            total_cost = current_price * quantity * (1 + commission_rate)
            if total_cost > available_cash:
                quantity -= 1
        if quantity > 0:
            logging.info(f"{stock_code} 종목 매수 수량 계산: {quantity}주 (종목당 투자금: {per_stock_investment:,.0f}원, 가용현금: {available_cash:,.0f}원, 현재가: {current_price:,.0f}원)")
        else:
            logging.warning(f"{stock_code} 종목 매수 불가: 현금 부족 (종목당 투자금: {per_stock_investment:,.0f}원, 가용현금: {available_cash:,.0f}원, 현재가: {current_price:,.0f}원)")
        
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
        holding_codes = [code for code, _ in current_holdings]
        logging.info(f"현재 상태: 포트폴리오 가치 {portfolio_value:,.0f}원 = 보유종목 {len(current_holdings)}개 ({total_holdings_value:,.0f}원) + 현금 {self.broker.cash:,.0f}원")
        if holding_codes:
            logging.info(f"보유종목: {holding_codes}")
        buy_codes = [code for code, _ in new_buys]
        sell_codes = [code for code, _ in to_sell]
        logging.info(f"매수 계획: {len(new_buys)}종목 {buy_codes} (소요금액: {total_buy_amount:,.0f}원)")
        logging.info(f"매도 계획: {len(to_sell)}종목 {sell_codes} (회수금액: {total_sell_amount:,.0f}원)")
    def _initialize_signals_for_all_stocks(self): 
        """모든 종목에 대한 시그널을 초기화합니다.""" 
        # data_store에 있는 종목들 초기화
        for stock_code in self.data_store.get('daily', {}): 
            if stock_code not in self.signals: 
                self.signals[stock_code] = { 
                    'signal': None, 
                    'signal_date': None, 
                    'traded_today': False, 
                    'target_quantity': 0 
                } 

        # broker의 positions에 있는 종목들도 초기화
        for stock_code in self.broker.positions.keys():
            if stock_code not in self.signals:
                self.signals[stock_code] = {
                    'signal': None,
                    'signal_date': None,
                    'traded_today': False,
                    'target_quantity': 0
                }

    def _select_buy_candidates(self, momentum_scores, safe_asset_momentum):
        """모멘텀 스코어를 기반으로 매수 대상 종목을 선정합니다."""
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = set()
        
        for rank, (stock_code, score) in enumerate(sorted_stocks, 1):
            if rank <= self.strategy_params['num_top_stocks'] and score > safe_asset_momentum:
                buy_candidates.add(stock_code)

        return buy_candidates, sorted_stocks

    def _generate_signals(self, current_daily_date, buy_candidates, sorted_stocks):
        """매수/매도/홀딩 신호를 생성하고 업데이트합니다."""
        current_positions = set(self.broker.positions.keys())

        for stock_code, _ in sorted_stocks:
            # 종목이 signals에 초기화되지 않았다면 초기화
            if stock_code not in self.signals:
                self.signals[stock_code] = {
                    'signal': None,
                    'signal_date': None,
                    'traded_today': False,
                    'target_quantity': 0
                }
            
            # 기본 정보 업데이트
            self.signals[stock_code].update({
                'signal_date': current_daily_date,
                'traded_today': False
            })

            if stock_code in buy_candidates:
                self._handle_buy_candidate(stock_code, current_daily_date, current_positions)
            else:
                self._handle_sell_candidate(stock_code, current_positions)

        return current_positions

    def _handle_buy_candidate(self, stock_code, current_daily_date, current_positions):
        """매수 대상 종목에 대한 신호를 처리합니다."""
        # 종목이 signals에 초기화되지 않았다면 초기화
        if stock_code not in self.signals:
            self.signals[stock_code] = {
                'signal': None,
                'signal_date': None,
                'traded_today': False,
                'target_quantity': 0
            }
            
        current_price_daily = self._get_historical_data_up_to('daily', stock_code, current_daily_date, lookback_period=1)['close'].iloc[-1]
        target_quantity = self._calculate_target_quantity(stock_code, current_price_daily)

        if target_quantity > 0:
            if stock_code in current_positions:
                self.signals[stock_code]['signal'] = 'hold'
                logging.info(f'홀딩 신호 - {stock_code}: (기존 보유 종목)')
            else:
                self.signals[stock_code].update({
                    'signal': 'buy',
                    'target_quantity': target_quantity,
                    'target_price': current_price_daily  # 목표가격 추가
                })
                logging.info(f'매수 신호 - {stock_code}: 목표수량 {target_quantity}주, 목표가격 {current_price_daily:,.0f}원')

    def _handle_sell_candidate(self, stock_code, current_positions):
        """매도 대상 종목에 대한 신호를 처리합니다."""
        # 종목이 signals에 초기화되지 않았다면 초기화
        if stock_code not in self.signals:
            self.signals[stock_code] = {
                'signal': None,
                'signal_date': None,
                'traded_today': False,
                'target_quantity': 0
            }
        
        # 현재가를 목표가격으로 설정
        current_price_daily = self._get_historical_data_up_to('daily', stock_code, self.signals[stock_code].get('signal_date'), lookback_period=1)['close'].iloc[-1]
            
        self.signals[stock_code].update({
            'signal': 'sell',
            'target_price': current_price_daily  # 목표가격 추가
        })
        
        if stock_code in current_positions:
            logging.info(f'매도 신호 - {stock_code} (보유중): 목표가격 {current_price_daily:,.0f}원')
        else:
            logging.debug(f'매도 신호 - {stock_code} (미보유): 목표가격 {current_price_daily:,.0f}원')


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

    def reset_signal(self, stock_code):
        """매매 체결 후 신호 dict를 안전하게 초기화한다."""
        if stock_code in self.signals:
            self.signals[stock_code]['traded_today'] = True
            self.signals[stock_code]['target_quantity'] = 0
            self.signals[stock_code]['target_price'] = 0
            self.signals[stock_code]['signal'] = None

