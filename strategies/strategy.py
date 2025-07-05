import abc # Abstract Base Class
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging
import sys
import os

from util.strategies_util import *

# --- 로거 설정 (스크립트 최상단에서 설정하여 항상 보이도록 함) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 테스트 시 DEBUG로 설정하여 모든 로그 출력 - 제거
class BaseStrategy(abc.ABC):
    """모든 전략의 기반이 되는 추상 클래스."""
    def __init__(self, trade, strategy_params: Dict[str, Any]):
        self.broker = trade.broker
        self.data_store = trade.data_store
        self.strategy_params = strategy_params
        self.strategy_params = strategy_params

    @abc.abstractmethod
    def run_daily_logic(self, current_date):
        """일봉 데이터를 기반으로 전략 로직을 실행하는 추상 메서드.
        (일봉 전략에서 주로 사용하며, 분봉 전략에서는 pass로 구현될 수 있습니다.)
        """
        pass

    @abc.abstractmethod
    def run_minute_logic(self, current_minute_dt, stock_code):
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
    def __init__(self, trade, strategy_params: Dict[str, Any]):
        super().__init__(trade, strategy_params)

        self.signals = {}
        self._initialize_signals_for_all_stocks()   # self.signals 초기화

    @abc.abstractmethod
    def run_daily_logic(self, current_date):
        """일봉 데이터를 기반으로 전략 로직을 실행하고 매매 의도를 반환합니다.
        예: 듀얼 모멘텀의 매수/매도 종목 선정
        """
        pass
    
    def run_minute_logic(self, current_minute_dt, stock_code):
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

    def _log_rebalancing_summary(self, current_daily_date, buy_candidates, current_positions, sell_candidates=None):
        """리밸런싱 요약을 로깅합니다."""
        if sell_candidates is None:
            sell_candidates = set()
            
        logging.info(f'[{current_daily_date}] === 리밸런싱 요약 ===')
        logging.info(f'매수 후보: {len(buy_candidates)}개 - {sorted(buy_candidates)}')
        logging.info(f'매도 후보: {len(sell_candidates)}개 - {sorted(sell_candidates)}')
        logging.info(f'현재 보유: {len(current_positions)}개 - {sorted(current_positions)}')
        
        # 매수 신호 개수
        buy_signals = sum(1 for signal in self.signals.values() if signal.get('signal') == 'buy')
        # 매도 신호 개수
        sell_signals = sum(1 for signal in self.signals.values() if signal.get('signal') == 'sell')
        # 홀딩 신호 개수
        hold_signals = sum(1 for signal in self.signals.values() if signal.get('signal') == 'hold')
        
        logging.info(f'생성된 신호 - 매수: {buy_signals}개, 매도: {sell_signals}개, 홀딩: {hold_signals}개')
        logging.info(f'=== 리밸런싱 요약 완료 ===')

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

    def _reset_all_signals(self):
        """모든 신호를 완전히 초기화합니다. (다음날을 위해)"""
        self.signals = {}  # 모든 신호를 완전히 삭제
        logging.debug("일봉 전략의 모든 신호를 완전히 초기화했습니다.")

    def _generate_signals(self, current_daily_date, buy_candidates, sorted_stocks, sell_candidates=None):
        """매수/매도/홀딩 신호를 생성하고 업데이트합니다."""
        current_positions = set(self.broker.positions.keys())
        
        # sell_candidates가 None이면 빈 set으로 초기화
        if sell_candidates is None:
            sell_candidates = set()

        # 1. 매수 후보 종목들 처리
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
                self._handle_buy_candidate(stock_code, current_daily_date)
            else:
                # 매수 후보가 아니지만 보유 중인 종목은 홀딩
                if stock_code in current_positions:
                    self._handle_hold_candidate(stock_code, current_daily_date)

        # 2. 매도 후보 종목들 처리 (새로 추가)
        for stock_code in sell_candidates:
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
            
            # 매도 신호 생성
            self._handle_sell_candidate(stock_code, current_positions)

        # 3. 보유 중이지만 매수/매도 후보가 아닌 종목들 처리 (홀딩)
        for stock_code in current_positions:
            if stock_code not in buy_candidates and stock_code not in sell_candidates:
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
                
                self._handle_hold_candidate(stock_code, current_daily_date)

        return current_positions

    def _handle_buy_candidate(self, stock_code, current_daily_date):
        """매수 대상 종목에 대한 신호를 처리합니다."""
        # 종목이 signals에 초기화되지 않았다면 초기화
        if stock_code not in self.signals:
            self.signals[stock_code] = {
                'signal': None,
                'signal_date': None,
                'traded_today': False,
                'target_quantity': 0
            }
            
        # 종가를 현재가로 사용
        current_price_daily = self._get_historical_data_up_to('daily', stock_code, current_daily_date, lookback_period=1)['close'].iloc[-1]
        target_quantity = self._calculate_target_quantity(stock_code, current_price_daily)

        if target_quantity > 0:
            if stock_code in self.broker.positions:
                self.signals[stock_code]['signal'] = 'hold'
                logging.info(f'홀딩 신호 - {stock_code}: (기존 보유 종목)')
            else:
                self.signals[stock_code].update({
                    'signal': 'buy',
                    'target_quantity': target_quantity,
                    'target_price': current_price_daily  # 목표가격 추가 (전일 종가)
                })
                logging.info(f'매수 신호 - {stock_code}: 목표수량 {target_quantity}주, 목표가격 {current_price_daily:,.0f}원 (전일 종가)')

    def _handle_hold_candidate(self, stock_code, current_daily_date):
        """홀딩 대상 종목에 대한 신호를 처리합니다."""
        # 종목이 signals에 초기화되지 않았다면 초기화
        if stock_code not in self.signals:
            self.signals[stock_code] = {
                'signal': None,
                'signal_date': None,
                'traded_today': False,
                'target_quantity': 0
            }
        
        # 수정: signal_date가 None이면 current_daily_date 사용
        signal_date = self.signals[stock_code].get('signal_date')
        if signal_date is None:
            signal_date = current_daily_date
        
        # 수정: 전일 종가를 목표가격으로 설정 (장전 판단을 위해)
        current_price_daily = self._get_historical_data_up_to('daily', stock_code, signal_date, lookback_period=1)['close'].iloc[-1]
        
        # 홀딩 신호 설정
        self.signals[stock_code].update({
            'signal': 'hold',
            'signal_date': current_daily_date,
            'target_price': current_price_daily,
            'target_quantity': self.broker.positions.get(stock_code, {}).get('size', 0)
        })
        
        logging.info(f'홀딩 신호 - {stock_code}: 목표수량 {self.signals[stock_code]["target_quantity"]}주, 목표가격 {current_price_daily:,.0f}원 (전일 종가)')

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
        
        # 수정: signal_date가 None이면 current_daily_date 사용
        signal_date = self.signals[stock_code].get('signal_date')
        if signal_date is None:
            # signal_date가 None이면 현재 날짜를 사용 (임시 처리)
            # 실제로는 이 부분이 호출되면 안 되지만, 안전성을 위해 추가
            logging.warning(f"{stock_code}: signal_date가 None입니다. 매도 신호 생성을 건너뜁니다.")
            return
        
        # 수정: 전일 종가를 목표가격으로 설정 (장전 판단을 위해)
        current_price_daily = self._get_historical_data_up_to('daily', stock_code, signal_date, lookback_period=1)['close'].iloc[-1]
            
        self.signals[stock_code].update({
            'signal': 'sell',
            'target_price': current_price_daily  # 목표가격 추가 (전일 종가)
        })
        
        if stock_code in current_positions:
            logging.info(f'매도 신호 - {stock_code} (보유중): 목표가격 {current_price_daily:,.0f}원 (전일 종가)')
        else:
            logging.debug(f'매도 신호 - {stock_code} (미보유): 목표가격 {current_price_daily:,.0f}원 (전일 종가)')

    # 모멘텀 전략으로 보낼 것 -> 관심종목 필터링으로 용도변경경
    def _select_buy_candidates(self, momentum_scores, safe_asset_momentum):
        """모멘텀 스코어를 기반으로 매수 대상 종목을 선정합니다."""
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = set()
        
        for rank, (stock_code, score) in enumerate(sorted_stocks, 1):
            if rank <= self.strategy_params['num_top_stocks'] and score > safe_asset_momentum:
                buy_candidates.add(stock_code)

        return buy_candidates, sorted_stocks


class MinuteStrategy(BaseStrategy):
    """분봉 전략을 위한 추상 클래스."""
    def __init__(self, trade, strategy_params: Dict[str, Any]):
        super().__init__(trade, strategy_params)
        self.signals = {}
        #self.last_portfolio_check = None  # 마지막 포트폴리오 체크 시간

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

    def execute_time_cut_buy(self, stock_code, current_dt, current_price, target_quantity, max_price_diff_ratio=0.02):
        """
        타임컷 강제매수를 실행합니다.
        
        Args:
            stock_code (str): 종목 코드
            current_dt (datetime): 현재 시각
            current_price (float): 현재 가격
            target_quantity (int): 매수 수량
            max_price_diff_ratio (float): 최대 허용 괴리율 (기본값: 0.01 = 1%)
            
        Returns:
            bool: 매수 실행 여부
        """
        if stock_code not in self.signals:
            return False
            
        target_price = self.signals[stock_code].get('target_price', current_price)
        price_diff_ratio = abs(target_price - current_price) / target_price
        
        if price_diff_ratio <= max_price_diff_ratio:
            logging.info(f'[타임컷 강제매수] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:.2f}, 매수가: {current_price:.2f}, 괴리율: {price_diff_ratio:.2%}')
            self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
            self.reset_signal(stock_code)
            return True
        else:
            logging.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:.2f}, 매수가: {current_price:.2f}, 괴리율: {price_diff_ratio:.2%} ({max_price_diff_ratio:.1%} 초과)')
            return False

    def execute_time_cut_sell(self, stock_code, current_dt, current_price, current_position_size, max_price_diff_ratio=0.02):
        """
        타임컷 강제매도를 실행합니다.
        
        Args:
            stock_code (str): 종목 코드
            current_dt (datetime): 현재 시각
            current_price (float): 현재 가격
            current_position_size (int): 현재 보유 수량
            max_price_diff_ratio (float): 최대 허용 괴리율 (기본값: 0.01 = 1%)
            
        Returns:
            bool: 매도 실행 여부
        """
        if stock_code not in self.signals:
            return False
            
        target_price = self.signals[stock_code].get('target_price', current_price)
        price_diff_ratio = abs(target_price - current_price) / target_price
        
        if price_diff_ratio <= max_price_diff_ratio:
            logging.info(f'[타임컷 강제매도] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:.2f}, 매도가: {current_price:.2f}, 괴리율: {price_diff_ratio:.2%}')
            self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
            self.reset_signal(stock_code)
            return True
        else:
            logging.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:.2f}, 매도가: {current_price:.2f}, 괴리율: {price_diff_ratio:.2%} ({max_price_diff_ratio:.1%} 초과)')
            return False

    # def update_signals(self, new_signals):
    #     """일봉 전략으로부터 받은 신호를 업데이트합니다."""
    #     self.signals = new_signals
    #     logging.debug(f"[BreakoutMinute] 신호 업데이트 완료. 총 {len(self.signals)}개 신호 수신.")
    def update_signals(self, signals):
        """
        DailyStrategy에서 생성된 신호들을 업데이트합니다.
        """
        self.signals = {
            stock_code: {**info, 'traded_today': False}
            for stock_code, info in signals.items() #### info 에 set() 아이템 설정
        }