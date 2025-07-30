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
        self.strategy_name = self.__class__.__name__
        
        self.signals = {}
        self._initialize_signals_for_all_stocks()

    @abc.abstractmethod
    def filter_universe(self, universe_codes: List[str], current_date: date) -> List[str]:
        """
        전체 유니버스 중 해당 전략의 로직을 적용할 대상 종목들만 선별합니다.
        """
        pass

    # 공통 로직을 담은 '템플릿 메서드'
    def run_daily_logic(self, current_date: date, strategy_capital: float):
        """
        모든 일봉 전략의 공통 실행 흐름(템플릿)을 정의합니다.
        """
        logging.info(f"{current_date} - --- {self.strategy_name} 일일 로직 실행 ---")
        self._reset_all_signals()

        # [수정] 1. 전체 유니버스를 가져옵니다.
        full_universe = list(self.data_store['daily'].keys())
        if not full_universe:
            logger.warning("거래할 유니버스 종목이 없습니다.")
            return

        # [수정] 2. 각 전략의 고유한 필터링 로직을 호출합니다.
        filtered_universe = self.filter_universe(full_universe, current_date)
        logging.info(f"[{self.strategy_name}] 유니버스 필터링 완료. 전체 {len(full_universe)}개 -> 대상 {len(filtered_universe)}개")

        # [수정] 3. 필터링된 유니버스를 기반으로 매수/매도 후보를 계산합니다.
        buy_candidates, sell_candidates, signal_attributes = self._calculate_strategy_signals(
            current_date, filtered_universe
        )

        # 4. 공통 신호 생성 로직을 호출합니다.
        final_positions = self._generate_signals(
            current_date, buy_candidates, sell_candidates, signal_attributes, strategy_capital
        )

        # 3. 공통 요약 로그를 호출합니다.
        self._log_rebalancing_summary(current_date, buy_candidates, final_positions, sell_candidates)
    
    def run_minute_logic(self, current_minute_dt, stock_code):
        """일봉 전략은 분봉 로직을 직접 수행하지 않으므로 이 메서드는 비워둡니다."""
        pass 
    
    # 각 전략이 반드시 구현해야 할 고유 로직 부분을 추상 메서드로 정의
    @abc.abstractmethod
    def _calculate_strategy_signals(self, current_date: date, universe: list) -> Tuple[set, set, dict]:
        """
        [최종 확정] 각 전략의 고유한 로직을 구현하여 다음을 반환해야 합니다:
        - buy_candidates (set): 매수 후보 종목 코드 집합
        - sell_candidates (set): 매도 후보 종목 코드 집합
        - signal_attributes (dict): {종목코드: {'score': 점수, ...}} 딕셔너리
        """
        pass
    

    def _initialize_signals_for_all_stocks(self): 
        all_stocks = set(self.data_store.get('daily', {}).keys()) | set(self.broker.get_current_positions().keys())
        for stock_code in all_stocks: 
            if stock_code not in self.signals: 
                self.signals[stock_code] = {}

    def _reset_all_signals(self):
        self.signals = {}
        logging.debug("일봉 전략 신호 초기화 완료.")
    
    def _generate_signals(self, current_daily_date: date, buy_candidates: set, sell_candidates: set, signal_attributes: dict, strategy_capital: float):
        """
        통합된 signal_attributes를 사용하여 최종 매매 신호를 생성합니다.
        - 점수 기반으로 매수 후보 정렬 및 상위 N개 선택
        - 종목별 최소 투자금 규칙에 따라 예산을 소진하며 매수 신호 생성
        - 매도/홀드 신호 생성
        """
        current_positions = set(self.broker.get_current_positions().keys())
        unfilled_order_codes = self.broker.get_unfilled_stock_codes()
        new_signals = {}
        
        # --- 1. 매수 신호 생성 ---
        remaining_capital = strategy_capital
        num_top_stocks = self.strategy_params.get('num_top_stocks', len(buy_candidates))

        # 점수가 높은 순서대로 매수 후보 정렬 및 상위 N개 선택
        sorted_buy_candidates = sorted(
            buy_candidates,
            key=lambda code: signal_attributes.get(code, {}).get('score', 0),
            reverse=True
        )[:num_top_stocks]

        for stock_code in sorted_buy_candidates:
            # (이중 체크) 이미 보유/미체결 상태면 건너뛰기
            if stock_code in current_positions or stock_code in unfilled_order_codes:
                continue

            if remaining_capital < MIN_STOCK_CAPITAL:
                logging.info(f"[{self.strategy_name}] 남은 예산({remaining_capital:,.0f}원)이 부족하여 '{stock_code}' 추가 매수를 중단합니다.")
                break

            attributes = signal_attributes.get(stock_code)
            if not attributes:
                logging.warning(f"[{stock_code}]에 대한 신호 속성이 없어 매수 신호를 생성할 수 없습니다.")
                continue

            target_price = attributes.get('target_price')
            execution_type = attributes.get('execution_type', 'touch')

            if not target_price or target_price <= 0:
                logging.warning(f"[{stock_code}] 유효한 목표가격이 없어 매수 신호를 생성할 수 없습니다.")
                continue

            cost_per_share = target_price * (1 + self.broker.commission_rate)
            target_quantity = int(MIN_STOCK_CAPITAL / cost_per_share) if cost_per_share > 0 else 0

            if target_quantity > 0:
                new_signals[stock_code] = {
                    'signal_type': 'buy',
                    'signal_date': current_daily_date,
                    'stock_code': stock_code,
                    'stock_name': self.get_stock_name(stock_code),
                    'strategy_name': self.strategy_name,
                    'score': attributes.get('score', 0),
                    'target_price': target_price,
                    'execution_type': execution_type,
                    'target_quantity': target_quantity,
                    'is_executed': False
                }
                
                actual_cost = target_quantity * cost_per_share
                remaining_capital -= actual_cost
                
                logging.info(f"매수 신호({self.strategy_name}) - {stock_code}: "
                             f"수량 {target_quantity}주 (사용한 예산: {actual_cost:,.0f}원 / 남은 예산: {remaining_capital:,.0f}원)")
        
        # --- 2. 매도 및 홀드 신호 생성 ---
        stocks_to_process_for_sell_hold = sell_candidates | current_positions
        for stock_code in stocks_to_process_for_sell_hold:
            if stock_code in new_signals or stock_code in unfilled_order_codes:
                continue

            signal_type = 'sell' if stock_code in sell_candidates else 'hold'
            
            # 매도/홀드 신호에 대한 속성도 signal_attributes에서 가져옴
            attributes = signal_attributes.get(stock_code, {})
            target_price = attributes.get('target_price')
            # 매도 신호의 기본 전술은 'market', 홀드는 'touch'로 설정
            default_tactic = 'market' if signal_type == 'sell' else 'touch'
            execution_type = attributes.get('execution_type', default_tactic)

            signal_info = {
                'signal_type': signal_type,
                'signal_date': current_daily_date,
                'stock_code': stock_code,
                'stock_name': self.get_stock_name(stock_code),
                'strategy_name': self.strategy_name,
                'score': attributes.get('score', 0),
                'target_price': target_price,
                'execution_type': execution_type,
                'target_quantity': self.broker.get_position_size(stock_code),
                'is_executed': False
            }
            new_signals[stock_code] = signal_info

            if signal_type == 'sell':
                logging.info(f"매도 신호({self.strategy_name}) - {stock_code}: "
                             f"수량 {signal_info['target_quantity']}주, 전술: {execution_type}")

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

