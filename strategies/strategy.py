# strategies/strategy.py

import abc # Abstract Base Class
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Any, Optional
import logging
import sys
import os

from trading.broker import Broker
from util.strategies_util import *
from config.settings import MIN_STOCK_CAPITAL
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
        target_date = target_dt.date()
        if data_type == 'daily':
            # 1. 해당 종목의 일봉 데이터프레임을 가져옵니다.
            daily_df = self.data_store['daily'].get(stock_code)
            
            if daily_df is None or daily_df.empty:
                return None
            
            # 2. 날짜를 기준으로 해당 일봉(하나의 행)을 정확히 찾습니다.
            try:
                # 날짜 인덱스로 해당 행(pd.Series)을 반환
                return daily_df.loc[pd.Timestamp(target_date)]
            except KeyError:
                # 해당 날짜에 데이터가 없는 경우
                return None
     
        elif data_type == 'minute':
            # 해당 날짜의 분봉 데이터 저장소 확인
            minute_data = self.data_store['minute'].get(stock_code, {}).get(target_date)
            
            if minute_data is None or minute_data.empty:
                return None
                
            try:
                # 정확한 시간의 분봉(Series) 반환
                return minute_data.loc[target_dt]
            except KeyError:
                # 해당 시간에 데이터가 없는 경우
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

    # 공통 로직을 담은 '템플릿 메서드'
    def run_daily_logic(self, current_date: date, strategy_capital: float):
        """
        모든 일봉 전략의 공통 실행 흐름(템플릿)을 정의합니다.
        """
        logging.info(f"{current_date} - --- {self.strategy_name} 일일 로직 실행 ---")
        self._reset_all_signals()

        universe = list(self.data_store['daily'].keys())
        if not universe:
            logger.warning("거래할 유니버스 종목이 없습니다.")
            return

        # 1. 각 전략의 고유 로직을 호출하여 매수/매도 후보를 받습니다.
        buy_candidates, sell_candidates, sorted_buy_stocks, stock_target_prices = self._calculate_strategy_signals(
            current_date, universe
        )

        # 2. 공통 신호 생성 로직을 호출합니다.
        final_positions = self._generate_signals(
            current_date, buy_candidates, sorted_buy_stocks, stock_target_prices, sell_candidates, strategy_capital
        )

        # 3. 공통 요약 로그를 호출합니다.
        self._log_rebalancing_summary(current_date, buy_candidates, final_positions, sell_candidates)
    
    def run_minute_logic(self, current_minute_dt, stock_code):
        """일봉 전략은 분봉 로직을 직접 수행하지 않으므로 이 메서드는 비워둡니다."""
        pass 
    
    # 각 전략이 반드시 구현해야 할 고유 로직 부분을 추상 메서드로 정의
    @abc.abstractmethod
    def _calculate_strategy_signals(self, current_date: date, universe: list) -> Tuple[set, set, list, dict]:
        """
        각 전략의 고유한 로직을 구현하여 다음을 반환해야 합니다:
        - buy_candidates (set): 매수 후보 종목 코드 집합
        - sell_candidates (set): 매도 후보 종목 코드 집합
        - sorted_buy_stocks (list): (종목코드, 점수) 튜플로 정렬된 리스트
        - stock_target_prices (dict): {종목코드: 목표가} 딕셔너리
        """
        pass
    
    # def _calculate_target_quantity(self, stock_code, current_price, current_date, num_stocks=None):
    #     if num_stocks is None:
    #         num_stocks = self.strategy_params.get('num_top_stocks', 1)

    #     # 1. 1주를 사는 데 필요한 총 비용 계산 (수수료 포함)
    #     per_share_cost = current_price * (1 + self.broker.commission_rate)
    #     if per_share_cost <= 0:
    #         return 0

    #     # 2. 이 종목에 투자할 수 있는 최대 금액 결정
    #     capital_base = self.broker.initial_cash
    #     per_stock_budget = capital_base / num_stocks if num_stocks > 0 else 0
    #     available_cash = self.broker.get_current_cash_balance()
    #     max_invest_amount = min(per_stock_budget, available_cash)

    #     # 3. 매수 가능한 수량 계산 (최대 투자 가능 금액 / 1주당 비용)
    #     # 만약 최대 투자 가능 금액이 1주 비용보다 작으면 quantity는 자동으로 0이 됨
    #     quantity = int(max_invest_amount / per_share_cost)

    #     # 4. 결과 로깅
    #     if quantity > 0:
    #         logging.info(f"✅ [{stock_code}] 매수 수량 계산 완료: {quantity}주")
    #     else:
    #         # INFO 레벨로 변경하여 항상 로그가 보이도록 하고, 원인을 명확히 표시
    #         logging.info(f"❌ [{stock_code}] 매수 불가: 할당/가용액({max_invest_amount:,.0f}원) < 1주 비용({per_share_cost:,.0f}원)")
        
    #     return quantity


    def _initialize_signals_for_all_stocks(self): 
        all_stocks = set(self.data_store.get('daily', {}).keys()) | set(self.broker.get_current_positions().keys())
        for stock_code in all_stocks: 
            if stock_code not in self.signals: 
                self.signals[stock_code] = {}

    def _reset_all_signals(self):
        self.signals = {}
        logging.debug("일봉 전략 신호 초기화 완료.")
    
    def _generate_signals(self, current_daily_date: date, buy_candidates: set, sorted_stocks: list, stock_target_prices: dict, sell_candidates: set, strategy_capital: float):
        """
        [최종 수정] '종목별 최소 투자금' 규칙에 따라, 할당된 예산 내에서 유연하게 매수 신호를 생성합니다.
        가장 점수가 높은 신호부터 순차적으로 예산을 소진합니다.
        """
        current_positions = set(self.broker.get_current_positions().keys())
        unfilled_order_codes = self.broker.get_unfilled_stock_codes()
        new_signals = {}
        if unfilled_order_codes:
            logger.info(f"[{self.strategy_name}] 신호 생성 전 미체결 주문 확인: {unfilled_order_codes}")
        # --- 1. [신규] 매수 신호 생성 루프 ---
        remaining_capital = strategy_capital  # 해당 전략에 할당된 가용 예산

        # 점수가 높은 순서대로 매수 후보(sorted_stocks)를 순회
        for stock_code, score in sorted_stocks:
            # 매수 후보 목록에 없거나, 이미 보유/미체결 상태면 건너뛰기
            if not (stock_code in buy_candidates and 
                    stock_code not in current_positions and 
                    stock_code not in unfilled_order_codes):
                continue
            
            # 1-1. 예산 확인: 남은 전략 예산이 '최소 투자금'보다 적으면 더 이상 매수 불가
            if remaining_capital < MIN_STOCK_CAPITAL:
                logging.info(f"[{self.strategy_name}] 남은 예산({remaining_capital:,.0f}원)이 부족하여 '{stock_code}' 추가 매수를 중단합니다.")
                break  # 매수 후보 처리를 종료

            target_price = stock_target_prices.get(stock_code)
            if not target_price or target_price <= 0:
                logging.warning(f"[{stock_code}] 유효한 목표가격이 없어 매수 신호를 생성할 수 없습니다.")
                continue

            # 1-2. 수량 계산: '최소 투자금'을 기준으로 매수 수량 결정
            budget_for_stock = MIN_STOCK_CAPITAL
            cost_per_share = target_price * (1 + self.broker.commission_rate)
            target_quantity = int(budget_for_stock / cost_per_share) if cost_per_share > 0 else 0

            if target_quantity > 0:
                new_signals[stock_code] = {
                    'signal_type': 'buy',
                    'signal_date': current_daily_date,
                    'target_price': target_price,
                    'stock_code': stock_code,
                    'strategy_name': self.strategy_name,
                    'is_executed': False,
                    'target_quantity': target_quantity
                }
                
                # 1-3. 예산 차감: 매수 신호를 생성했으므로, 전략 예산에서 사용한 만큼 차감
                actual_cost = target_quantity * cost_per_share
                remaining_capital -= actual_cost
                
                logging.info(f"매수 신호({self.strategy_name}) - {stock_code}: "
                             f"수량 {target_quantity}주 (사용한 예산: {actual_cost:,.0f}원 / 남은 예산: {remaining_capital:,.0f}원)")
        
        # --- 2. 매도 및 홀드 신호 생성 루프 ---
        stocks_to_process_for_sell_hold = sell_candidates | current_positions
        for stock_code in stocks_to_process_for_sell_hold:
            # 위에서 이미 'buy' 신호가 생성된 종목은 건너뛴다.
            if stock_code in new_signals:
                continue

            if stock_code in unfilled_order_codes:
                logger.info(f"[{self.strategy_name}] '{stock_code}'는 이미 미체결 주문이 있어 매도/홀드 신호 생성을 건너뜁니다.")
                continue

            signal_type = 'sell' if stock_code in sell_candidates else 'hold'
            
            signal_info = {
                'signal_type': signal_type,
                'signal_date': current_daily_date,
                'target_price': stock_target_prices.get(stock_code),
                'stock_code': stock_code,
                'stock_name': self.broker.manager.get_stock_name(stock_code),
                'strategy_name': self.strategy_name,
                'is_executed': False,
                'target_quantity': self.broker.get_position_size(stock_code)
            }
            new_signals[stock_code] = signal_info

            if signal_type == 'sell':
                logging.info(f"매도 신호({self.strategy_name}) - {stock_code}: "
                             f"수량 {signal_info['target_quantity']}주")
        
        self.signals = new_signals
        return current_positions

    
    def get_stock_name(self, stock_code: str) -> str:
        """브로커의 매니저를 통해 종목명을 조회하는 과정을 캡슐화합니다."""
        # 이 코드가 이제 최적화된 TradingManager의 메서드를 호출합니다.
        return self.broker.manager.get_stock_name(stock_code)
    
   
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
        """매매가 실행된 종목의 신호를 딕셔너리에서 제거하여 반복적인 주문을 방지합니다."""
        if stock_code in self.signals:
            del self.signals[stock_code]
            logging.info(f"[{stock_code}] 신호 처리 완료. signals에서 제거합니다.")

    def execute_time_cut_buy(self, stock_code, current_dt, current_price, target_quantity, max_deviation_ratio):
        """타임컷 강제매수를 실행합니다."""
        if stock_code not in self.signals: return False
        target_price = self.signals[stock_code].get('target_price', current_price)
        if target_price is None or target_price <= 0: return False

        price_diff_ratio = abs(target_price - current_price) / target_price
        
        if price_diff_ratio <= max_deviation_ratio / 100:
            logging.info(f'[타임컷 강제매수] {current_dt.isoformat()} - {stock_code}')
            # FIX: order_time 파라미터 이름으로 전달
            self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, order_time=current_dt)
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
        # 목표가와 현재가 간의 괴리율 계산
        price_diff_ratio = abs(target_price - current_price) / target_price 
        
        if price_diff_ratio <= max_deviation_ratio / 100:
            logging.info(f'[타임컷 강제매도] {current_dt.isoformat()} - {stock_code}')
            # FIX: order_time 파라미터 이름으로 전달
            self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, order_time=current_dt)
            self.reset_signal(stock_code)
            return True
        else:
            logging.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 괴리율: {price_diff_ratio:.2%}')
            return False
 
    def get_stock_name(self, stock_code: str) -> str:
        """브로커의 매니저를 통해 종목명을 조회하는 과정을 캡슐화합니다."""
        # 이 코드가 이제 최적화된 TradingManager의 메서드를 호출합니다.
        return self.broker.manager.get_stock_name(stock_code)

