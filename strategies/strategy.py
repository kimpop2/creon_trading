# strategies/strategy.py

import abc # Abstract Base Class
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Any
import logging
import sys
import os

from trading.broker import Broker
from util.strategies_util import *

# --- 로거 설정 ---
logger = logging.getLogger(__name__)

class BaseStrategy(abc.ABC):
    """모든 전략의 기반이 되는 추상 클래스."""
    def __init__(self):
        pass

    @abc.abstractmethod
    def run_daily_logic(self, current_date):
        """일봉 데이터를 기반으로 전략 로직을 실행하는 추상 메서드."""
        pass

    @abc.abstractmethod
    def run_minute_logic(self, current_minute_dt, stock_code):
        """분봉 데이터를 기반으로 전략 로직을 실행하는 추상 메서드."""
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
    """일봉 전략을 위한 추상 클래스 (예: 리밸런싱, 모멘텀 계산)."""
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__()
        self.broker = broker
        self.data_store = data_store
        self.strategy_params = strategy_params
        self.strategy_name = self.__class__.__name__ # 클래스 이름을 전략 이름으로 사용
        
        self.signals = {}
        self._initialize_signals_for_all_stocks()

    @abc.abstractmethod
    def run_daily_logic(self, current_date):
        """일봉 전략 로직을 실행하고 매매 신호를 생성합니다."""
        pass
    
    def run_minute_logic(self, current_minute_dt, stock_code):
        """일봉 전략은 분봉 로직을 직접 수행하지 않으므로 이 메서드는 비워둡니다."""
        pass 

    def _calculate_target_quantity(self, stock_code, current_price, num_stocks=None):
        """주어진 가격에서 동일비중 투자에 필요한 수량을 계산합니다."""
        if num_stocks is None:
            num_stocks = self.strategy_params.get('num_top_stocks', 1)

        current_prices_for_summary = {}
        for code in self.data_store['daily']:
            daily_data = self._get_historical_data_up_to('daily', code, pd.Timestamp.today(), lookback_period=1)
            if not daily_data.empty:
                current_prices_for_summary[code] = daily_data['close'].iloc[-1]
        
        portfolio_value = self.broker.get_portfolio_value(current_prices_for_summary)
        available_cash = self.broker.get_current_cash_balance()
        commission_rate = self.broker.commission_rate
        
        per_stock_investment = portfolio_value / num_stocks
        max_buyable_amount = available_cash / (1 + commission_rate)
        actual_investment_amount = min(per_stock_investment, max_buyable_amount)
        
        if current_price <= 0: return 0
        quantity = int(actual_investment_amount / current_price)
        
        if quantity > 0:
            total_cost = current_price * quantity * (1 + commission_rate)
            if total_cost > available_cash:
                quantity -= 1

        if quantity > 0:
            logger.info(f"{stock_code} 종목 매수 수량 계산: {quantity}주")
        else:
            logger.warning(f"{stock_code} 종목 매수 불가: 현금 부족")
        
        return quantity

    def _log_rebalancing_summary(self, current_daily_date, buy_candidates, current_positions, sell_candidates=None):
        if sell_candidates is None: sell_candidates = set()
        logging.info(f'[{current_daily_date}] === 리밸런싱 요약 ===')
        logging.info(f'매수 후보: {len(buy_candidates)}개 - {sorted(buy_candidates)}')
        logging.info(f'매도 후보: {len(sell_candidates)}개 - {sorted(sell_candidates)}')
        logging.info(f'현재 보유: {len(current_positions)}개 - {sorted(current_positions)}')
        
        buy_signals = sum(1 for signal in self.signals.values() if signal.get('signal_type') == 'buy')
        sell_signals = sum(1 for signal in self.signals.values() if signal.get('signal_type') == 'sell')
        hold_signals = sum(1 for signal in self.signals.values() if signal.get('signal_type') == 'hold')
        
        logging.info(f'생성된 신호 - 매수: {buy_signals}개, 매도: {sell_signals}개, 홀딩: {hold_signals}개')
        logging.info(f'=== 리밸런싱 요약 완료 ===')

    def _initialize_signals_for_all_stocks(self): 
        all_stocks = set(self.data_store.get('daily', {}).keys()) | set(self.broker.get_current_positions().keys())
        for stock_code in all_stocks: 
            if stock_code not in self.signals: 
                self.signals[stock_code] = {}

    def _reset_all_signals(self):
        self.signals = {}
        logging.debug("일봉 전략 신호 초기화 완료.")

    def _generate_signals(self, current_daily_date, buy_candidates, sorted_stocks, stock_target_prices, sell_candidates=None):
        """
        [최종 수정] 매수/매도/홀딩 신호 생성을 통합하고, None 값 에러를 처리합니다.
        """
        current_positions = set(self.broker.get_current_positions().keys())
        if sell_candidates is None: sell_candidates = set()

        # 처리 대상이 되는 모든 종목 (신규 매수후보, 매도후보, 현재 보유종목)
        stocks_to_process = buy_candidates | sell_candidates | current_positions
        
        new_signals = {}
        
        for stock_code in stocks_to_process:
            target_price = stock_target_prices.get(stock_code)
            
            # 기본 신호 정보 생성
            signal_info = {
                'signal_type': None, # 기본값은 None
                'signal_date': current_daily_date,
                'target_price': target_price,
                'stock_code': stock_code,
                'stock_name': stock_code, # self.broker.manager.api_client.get_stock_name(stock_code),
                'strategy_name': self.strategy_name,
                'is_executed': False
            }

            # 1. 매수 신호 결정
            if stock_code in buy_candidates and stock_code not in current_positions:
                if target_price is not None and target_price > 0:
                    target_quantity = self._calculate_target_quantity(stock_code, target_price)
                    if target_quantity > 0:
                        signal_info.update({
                            'signal_type': 'buy',
                            'target_quantity': target_quantity
                        })
                        logging.info(f'매수 신호 - {stock_code}: 목표수량 {target_quantity}주, 목표가격 {target_price:,.0f}원')
                else:
                    logging.warning(f"{stock_code} 매수 신호 생성 불가: 유효한 목표가격 없음")

            # 2. 매도 신호 결정
            elif stock_code in sell_candidates and stock_code in current_positions:
                signal_info.update({
                    'signal_type': 'sell',
                    'target_quantity': self.broker.get_position_size(stock_code)
                })
                price_log = f"{target_price:,.0f}원" if target_price is not None else "N/A"
                logging.info(f'매도 신호 - {stock_code} (포지션 보유): 목표가격 {price_log}')

            # 3. 홀드 신호 결정
            elif stock_code in current_positions:
                signal_info.update({
                    'signal_type': 'hold',
                    'target_quantity': self.broker.get_position_size(stock_code)
                })
                price_log = f"{target_price:,.0f}원" if target_price is not None else "N/A"
                logging.info(f'홀딩 신호 - {stock_code}: 목표수량 {signal_info["target_quantity"]}주, 목표가격 {price_log}')

            # 유효한 신호('buy', 'sell', 'hold')가 생성된 경우에만 최종 신호 목록에 추가
            if signal_info.get('signal_type'):
                new_signals[stock_code] = signal_info

        self.signals = new_signals
        return current_positions
    
    # _handle_*_candidate 메서드들은 _generate_signals 내부 로직으로 통합되었으므로 삭제하거나 비워둘 수 있습니다.
    # 여기서는 비워두겠습니다.
    def _handle_buy_candidate(self, stock_code, current_daily_date, current_price_daily):
        pass

    def _handle_hold_candidate(self, stock_code, current_daily_date, current_price_daily):
        """기존 포지션에 대한 홀딩 신호 생성을 처리합니다."""
        pass

    def _handle_sell_candidate(self, stock_code, current_positions, current_price_daily):
        pass

    def _select_buy_candidates(self, momentum_scores, safe_asset_momentum):
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = set()
        
        for rank, (stock_code, score) in enumerate(sorted_stocks, 1):
            if rank <= self.strategy_params['num_top_stocks'] and score > safe_asset_momentum:
                buy_candidates.add(stock_code)

        return buy_candidates, sorted_stocks

class MinuteStrategy(BaseStrategy):
    """분봉 전략을 위한 추상 클래스 (예: 매매 실행, 장내 신호)."""
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__()
        self.broker = broker
        self.data_store = data_store
        self.strategy_params = strategy_params
        self.strategy_name = self.__class__.__name__
        self.signals = {}

    def run_daily_logic(self, current_date):
        """분봉 전략은 일봉 로직을 직접 수행하지 않으므로 이 메서드는 비워둡니다."""
        pass 

    @abc.abstractmethod
    def run_minute_logic(self, current_minute_dt, stock_code):
        """분봉 데이터와 신호를 기반으로 실제 매매를 실행합니다."""
        pass

    def update_signals(self, signals: Dict[str, Any]):
        """DailyStrategy에서 신호를 업데이트합니다."""
        self.signals = signals
        logging.debug(f"[MinuteStrategy] '{self.strategy_name}' 신호 업데이트 완료. 총 {len(self.signals)}개 신호 수신.")

    def reset_signal(self, stock_code):
        """매매 체결 후 신호 딕셔너리를 안전하게 초기화합니다."""
        if stock_code in self.signals:
            self.signals[stock_code]['traded_today'] = True
            self.signals[stock_code]['is_executed'] = True
            logging.debug(f"{stock_code} 매매 후 신호 초기화 완료.")

    def execute_time_cut_buy(self, stock_code, current_dt, current_price, target_quantity, max_deviation_ratio):
        """타임컷 강제매수를 실행합니다."""
        if stock_code not in self.signals: return False
        target_price = self.signals[stock_code].get('target_price', current_price)
        if target_price is None or target_price <= 0: return False

        price_diff_ratio = abs(target_price - current_price) * 100 / target_price
        
        if price_diff_ratio <= max_deviation_ratio:
            logging.info(f'[타임컷 강제매수] {current_dt.isoformat()} - {stock_code}')
            self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
            self.reset_signal(stock_code)
            return True
        else:
            logging.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 괴리율: {price_diff_ratio:.2%}')
            return False

    def execute_time_cut_sell(self, stock_code, current_dt, current_price, current_position_size, max_deviation_ratio):
        """타임컷 강제매도를 실행합니다."""
        if stock_code not in self.signals: return False
        target_price = self.signals[stock_code].get('target_price', current_price)
        if target_price is None or target_price <= 0: return False

        price_diff_ratio = abs(target_price - current_price) * 100 / target_price 
        
        if price_diff_ratio <= max_deviation_ratio:
            logging.info(f'[타임컷 강제매도] {current_dt.isoformat()} - {stock_code}')
            self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
            self.reset_signal(stock_code)
            return True
        else:
            logging.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 괴리율: {price_diff_ratio:.2%}')
            return False