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


            # 해당 날짜의 분봉 데이터가 있는지 확인
            
            # if stock_code not in self.data_store['minute'] or \
            # target_date not in self.data_store['minute'][stock_code]:
            #     # 분봉 데이터가 없으면 실시간 일봉 생성 불가
            #     logging.warning(f"{target_date}의 분봉 데이터가 없어 실시간 일봉을 생성할 수 없습니다.")
            #     return None
                
            # df_minute_today = self.data_store['minute'][stock_code][target_date]
            # # target_dt 이전의 분봉만 필터링
            # intraday_df = df_minute_today[df_minute_today.index <= target_dt]
            # if intraday_df.empty:
            #     return None
            # # 분봉 데이터를 집계하여 실시간 일봉 생성
            # df_daily = intraday_df.agg(
            #     open=('open', 'first'),      # 당일 첫 분봉의 시가
            #     high=('high', 'max'),        # 지금까지의 최고가
            #     low=('low', 'min'),          # 지금까지의 최저가
            #     close=('close', 'last'),     # 현재 분봉의 종가
            #     volume=('volume', 'sum')     # 지금까지의 거래량 합계
            # )
            
            # # Series의 이름을 날짜로 설정하여 반환 형식을 맞춤
            # df_daily.name = pd.Timestamp(target_date).normalize()
            # return df_daily
        
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

    @abc.abstractmethod
    def run_daily_logic(self, current_date):
        """일봉 전략 로직을 실행하고 매매 신호를 생성합니다."""
        pass
    
    def run_minute_logic(self, current_minute_dt, stock_code):
        """일봉 전략은 분봉 로직을 직접 수행하지 않으므로 이 메서드는 비워둡니다."""
        pass 

    def _calculate_target_quantity(self, stock_code, current_price, current_date, num_stocks=None):
        if num_stocks is None:
            num_stocks = self.strategy_params.get('num_top_stocks', 1)

        # 1. 1주를 사는 데 필요한 총 비용 계산 (수수료 포함)
        per_share_cost = current_price * (1 + self.broker.commission_rate)
        if per_share_cost <= 0:
            return 0

        # 2. 이 종목에 투자할 수 있는 최대 금액 결정
        capital_base = self.broker.initial_cash
        per_stock_budget = capital_base / num_stocks if num_stocks > 0 else 0
        available_cash = self.broker.get_current_cash_balance()
        max_invest_amount = min(per_stock_budget, available_cash)

        # 3. 매수 가능한 수량 계산 (최대 투자 가능 금액 / 1주당 비용)
        # 만약 최대 투자 가능 금액이 1주 비용보다 작으면 quantity는 자동으로 0이 됨
        quantity = int(max_invest_amount / per_share_cost)

        # 4. 결과 로깅
        if quantity > 0:
            logging.info(f"✅ [{stock_code}] 매수 수량 계산 완료: {quantity}주")
        else:
            # INFO 레벨로 변경하여 항상 로그가 보이도록 하고, 원인을 명확히 표시
            logging.info(f"❌ [{stock_code}] 매수 불가: 할당/가용액({max_invest_amount:,.0f}원) < 1주 비용({per_share_cost:,.0f}원)")
        
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
                # [수정 시작] target_price가 유효하지 않은 경우, data_store의 최종 종가로 대체하는 로직
                    price_to_use = target_price
                    if price_to_use is None or price_to_use <= 0:
                        logging.warning(f"[{stock_code}] 유효한 목표가격이 없어 data_store의 최종 종가로 대체합니다.")
                        # data_store에서 해당 종목의 과거 데이터를 가져옴
                        historical_data = self._get_historical_data_up_to('daily', stock_code, current_daily_date)
                        
                        # 데이터가 있고 비어있지 않은지 확인
                        if historical_data is not None and not historical_data.empty:
                            # 마지막 행의 'close' 값을 사용
                            price_to_use = historical_data['close'].iloc[-1]
                        else:
                            logging.error(f"[{stock_code}] data_store에서도 가격을 찾을 수 없어 매수 신호를 생성할 수 없습니다.")
                            price_to_use = 0 # 유효하지 않은 값으로 설정하여 아래 로직을 건너뛰도록 함

                    # 이제 price_to_use 변수를 사용하여 수량 계산
                    if price_to_use > 0:
                        target_quantity = self._calculate_target_quantity(stock_code, price_to_use, current_daily_date, self.strategy_params['num_top_stocks'])
                        if target_quantity > 0:
                            signal_info.update({
                                'signal_type': 'buy',
                                'target_quantity': target_quantity
                            })
                            # 로그 출력 시에도 실제 사용할 가격(price_to_use)을 표시
                            logging.info(f"매수 신호 - {stock_code}: 목표수량 {target_quantity}주, 목표가격 {price_to_use:,.0f}원")
                    # [수정 끝]                
                else:
                    logging.warning(f"{stock_code} 매수 신호 생성 불가: 유효한 목표가격 없음")

            # 2. 매도 신호 결정
            elif stock_code in sell_candidates and stock_code in current_positions:
                signal_info.update({
                    'signal_type': 'sell',
                    'target_quantity': self.broker.get_position_size(stock_code)
                })
                price_log = f"{target_price:,.0f}원" if target_price is not None else "N/A"
                logging.info(f'매도 신호 - {stock_code} 매도수량 {signal_info["target_quantity"]}주: 목표가격 {price_log}')

            # 3. 홀드 신호 결정
            elif stock_code in current_positions:
                signal_info.update({
                    'signal_type': 'hold',
                    'target_quantity': self.broker.get_position_size(stock_code)
                })
                price_log = f"{target_price:,.0f}원" if target_price is not None else "N/A"
                logging.debug(f'홀딩 신호 - {stock_code}: 목표수량 {signal_info["target_quantity"]}주, 목표가격 {price_log}')

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
        



