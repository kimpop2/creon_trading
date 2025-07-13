# trade/strategy.py
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
    def __init__(self):
        # self.broker = trade.broker
        # self.data_store = trade.data_store
        # self.strategy_params = strategy_params
        pass

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
                # 정렬된 날짜를 통해 시간순서를 보장
                for date_key in sorted(self.data_store['minute'][stock_code].keys()):
                    if date_key <= current_dt.date():
                        all_minute_dfs_for_stock.append(self.data_store['minute'][stock_code][date_key])
            
            if not all_minute_dfs_for_stock:
                return pd.DataFrame()
            
            combined_minute_df = pd.concat(all_minute_dfs_for_stock).sort_index()
            # 결합된 DataFrame을 현재 타임스탬프까지 필터링
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
        #self.manager = manager
        self.data_store = data_store
        self.strategy_params = strategy_params
        
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

        # 포트폴리오 가치 기준 종목당 투자금 계산 (일봉 데이터 기준)
        current_prices_for_summary = {}
        for code in self.data_store['daily']:
            # 오늘의 리밸런싱 계산을 위해 전일 종가를 참조로 사용
            daily_data = self._get_historical_data_up_to('daily', code, pd.Timestamp.today(), lookback_period=1)
            if not daily_data.empty:
                current_prices_for_summary[code] = daily_data['close'].iloc[-1]
        
        # 총 포트폴리오 가치와 현금 계산
        portfolio_value = self.broker.get_portfolio_value(current_prices_for_summary)
        available_cash = self.broker.initial_cash
        commission_rate = self.broker.commission_rate
        
        # 포트폴리오 가치를 기준으로 종목당 투자금 결정, 하지만 가용 현금에 제한됨
        per_stock_investment = portfolio_value / num_stocks
        
        # 현금과 수수료를 고려한 최대 매수 가능 금액 계산
        max_buyable_amount = available_cash / (1 + commission_rate)
        actual_investment_amount = min(per_stock_investment, max_buyable_amount)
        
        # 수량 계산
        quantity = int(actual_investment_amount / current_price)
        
        # 수량 조정 (수수료를 포함한 총 비용이 가용 현금을 초과하는 경우)
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
        
        # 신호 개수 계산
        buy_signals = sum(1 for signal in self.signals.values() if signal.get('signal') == 'buy')
        sell_signals = sum(1 for signal in self.signals.values() if signal.get('signal') == 'sell')
        hold_signals = sum(1 for signal in self.signals.values() if signal.get('signal') == 'hold')
        
        logging.info(f'생성된 신호 - 매수: {buy_signals}개, 매도: {sell_signals}개, 홀딩: {hold_signals}개')
        logging.info(f'=== 리밸런싱 요약 완료 ===')

    def _initialize_signals_for_all_stocks(self): 
        """data_store와 현재 포지션에 있는 모든 종목에 대한 신호를 초기화합니다.""" 
        all_stocks = set(self.data_store.get('daily', {}).keys()) | set(self.broker.positions.keys())

        for stock_code in all_stocks: 
            if stock_code not in self.signals: 
                self.signals[stock_code] = { 
                    'signal': None, 
                    'signal_date': None, 
                    'traded_today': False, 
                    'target_quantity': 0 
                } 

    def _reset_all_signals(self):
        """다음날을 위해 모든 신호를 완전히 초기화합니다."""
        self.signals = {}
        logging.debug("일봉 전략 신호 초기화 완료.")

    def _generate_signals(self, current_daily_date, buy_candidates, sorted_stocks, stock_target_prices, sell_candidates=None):
        """매수/매도/홀딩 신호를 생성하고 업데이트합니다."""
        current_positions = set(self.broker.positions.keys())
        
        if sell_candidates is None:
            sell_candidates = set()

        # 1. 잠재적 종목들(sorted_stocks)과 기존 포지션 처리
        stocks_to_process = set([stock[0] for stock in sorted_stocks]) | current_positions | sell_candidates
        
        for stock_code in stocks_to_process:
            if stock_code not in self.signals:
                self.signals[stock_code] = {
                    'signal': None,
                    'signal_date': None,
                    'traded_today': False,
                    'target_quantity': 0
                }
            
            # 현재 날짜에 대한 traded_today 플래그 리셋 및 신호 날짜 업데이트
            self.signals[stock_code].update({
                'signal_date': current_daily_date,
                'target_price': stock_target_prices[stock_code],
                'traded_today': False
            })

            # 매수 후보 확인
            if stock_code in buy_candidates:
                self._handle_buy_candidate(stock_code, current_daily_date)
            # 매도 후보 확인
            elif stock_code in sell_candidates:
                self._handle_sell_candidate(stock_code, current_positions)
            # 홀딩 확인 (현재 보유 중이지만 매수/매도 후보가 아닌 경우)
            elif stock_code in current_positions:
                self._handle_hold_candidate(stock_code, current_daily_date)

        # 참고: 종목이 매수/매도 후보가 아니고 현재 보유하지 않으면 신호는 None으로 유지됩니다 (또는 제거됨).

        return current_positions

    def _handle_buy_candidate(self, stock_code, current_daily_date):
        """종목에 대한 매수 신호 생성을 처리합니다."""
        # 현재 날짜까지의 일봉 데이터가 존재하는지 확인하여 종가를 얻기
        daily_data = self._get_historical_data_up_to('daily', stock_code, current_daily_date, lookback_period=1)
        
        if daily_data.empty:
            logging.warning(f"{stock_code}에 대한 매수 신호 생성 불가: {current_daily_date}까지의 일봉 데이터 없음")
            return

        #current_price_daily = daily_data['close'].iloc[-1]
        current_price_daily = self.signals[stock_code]['target_price']
        target_quantity = self._calculate_target_quantity(stock_code, current_price_daily)

        if target_quantity > 0:
            if stock_code in self.broker.positions:
                # 이미 보유 중이면 홀딩으로 처리 (포지션 유지)
                self.signals[stock_code]['signal'] = 'hold'
                logging.info(f'홀딩 신호 - {stock_code}: (이미 포지션 보유)')
            else:
                # 보유하지 않으면 매수 신호 생성
                self.signals[stock_code].update({
                    'signal': 'buy',
                    'target_quantity': target_quantity,
                #    'target_price': current_price_daily
                })
                logging.info(f'매수 신호 - {stock_code}: 목표수량 {target_quantity}주, 목표가격 {current_price_daily:,.0f}원 (전일 종가)')

    def _handle_hold_candidate(self, stock_code, current_daily_date):
        """기존 포지션에 대한 홀딩 신호 생성을 처리합니다."""
        # 현재 종가를 얻기 위해 일봉 데이터가 존재하는지 확인
        # daily_data = self._get_historical_data_up_to('daily', stock_code, current_daily_date, lookback_period=1)
        
        # if daily_data.empty:
        #     logging.warning(f"{stock_code}에 대한 홀딩 신호 생성 불가: {current_daily_date}까지의 일봉 데이터 없음")
        #     return
            
        #current_price_daily = daily_data['close'].iloc[-1]
        current_price_daily = self.signals[stock_code]['target_price']
        # 브로커 포지션의 현재 크기를 사용하여 홀딩 신호 설정
        self.signals[stock_code].update({
            'signal': 'hold',
            'signal_date': current_daily_date,
        #    'target_price': current_price_daily,
            'target_quantity': self.broker.positions.get(stock_code, {}).get('size', 0)
        })
        
        logging.info(f'홀딩 신호 - {stock_code}: 목표수량 {self.signals[stock_code]["target_quantity"]}주, 목표가격 {current_price_daily:,.0f}원 (전일 종가)')

    def _handle_sell_candidate(self, stock_code, current_positions):
        """종목에 대한 매도 신호 생성을 처리합니다."""
        # 일봉 데이터가 존재하는지 확인하여 현재 종가를 얻기
        # _generate_signals가 호출될 때 signal_date가 설정된다고 가정하지만, 일봉 데이터에서 가격을 얻습니다.
        # signal_date = self.signals[stock_code].get('signal_date')
        # if signal_date is None:
        #     logging.warning(f"{stock_code}: signal_date가 None입니다. 매도 신호 생성을 건너뜁니다.")
        #     return

        # daily_data = self._get_historical_data_up_to('daily', stock_code, signal_date, lookback_period=1)

        # if daily_data.empty:
        #     logging.warning(f"{stock_code}에 대한 매도 신호 생성 불가: 일봉 데이터 없음")
        #     return

        # current_price_daily = daily_data['close'].iloc[-1]
        current_price_daily = self.signals[stock_code]['target_price']    
        self.signals[stock_code].update({
            'signal': 'sell',
            'target_quantity': self.broker.positions.get(stock_code, {}).get('size', 0), # 보유 중이면 모두 매도
            #'target_price': current_price_daily
        })
        
        if stock_code in current_positions:
            logging.info(f'매도 신호 - {stock_code} (포지션 보유): 목표가격 {current_price_daily:,.0f}원 (전일 종가)')
        else:
            logging.debug(f'매도 신호 - {stock_code} (미보유): 목표가격 {current_price_daily:,.0f}원 (전일 종가)')

    def _select_buy_candidates(self, momentum_scores, safe_asset_momentum):
        """모멘텀 스코어를 기반으로 매수 후보 종목을 선정합니다."""
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
        #self.manager = manager
        self.data_store = data_store
        self.strategy_params = strategy_params
        self.signals = {}
        # self.last_portfolio_check = None # 이 구조에서는 필요하지 않음

    def run_daily_logic(self, current_date):
        """분봉 전략은 일봉 로직을 직접 수행하지 않으므로 이 메서드는 비워둡니다."""
        pass 

    @abc.abstractmethod
    def run_minute_logic(self, current_minute_dt, stock_code):
        """분봉 데이터와 신호를 기반으로 실제 매매를 실행합니다."""
        pass

    def update_signals(self, signals):
        """DailyStrategy에서 신호를 업데이트합니다."""
        # 하루 시작 시 신호가 업데이트될 때 'traded_today'를 False로 설정
        self.signals = {
            stock_code: {**info, 'traded_today': False}
            for stock_code, info in signals.items()
        }
        logging.debug(f"[MinuteStrategy] 신호 업데이트 완료. 총 {len(self.signals)}개 신호 수신.")

    def reset_signal(self, stock_code):
        """매매 체결 후 신호 딕셔너리를 안전하게 초기화합니다."""
        if stock_code in self.signals:
            self.signals[stock_code]['traded_today'] = True
            self.signals[stock_code]['target_quantity'] = 0
            self.signals[stock_code]['target_price'] = 0
            # 참고: 다음날의 리밸런싱 로직이 실행될 때까지 신호 타입(매수/매도/홀딩)은 유지하지만, 
            # 'traded_today'를 True로 표시합니다.
            logging.debug(f"{stock_code} 매매 후 신호 초기화 완료.")

    def execute_time_cut_buy(self, stock_code, current_dt, current_price, target_quantity, max_deviation_ratio):
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
        
        # target_price가 0이거나 유효하지 않은 경우 매매를 건너뜁니다.
        if target_price is None or target_price <= 0:
            logging.warning(f'[타임컷 매수] {current_dt.isoformat()} - {stock_code}: 목표 가격이 0이거나 유효하지 않아 매매를 건너뜁니다.')
            return False

        price_diff_ratio = abs(target_price - current_price) * 100 / target_price
        
        if price_diff_ratio <= max_deviation_ratio:
            logging.info(f'[타임컷 강제매수] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:.2f}, 매수가: {current_price:.2f}, 괴리율: {price_diff_ratio:.2%}')
            self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
            self.reset_signal(stock_code)
            return True
        else:
            logging.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:.2f}, 매수가: {current_price:.2f}, 괴리율: {price_diff_ratio:.2%} ({max_deviation_ratio:.1%} 초과)')
            return False

    def execute_time_cut_sell(self, stock_code, current_dt, current_price, current_position_size, max_deviation_ratio):
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
        
        # target_price가 0이거나 유효하지 않은 경우 매매를 건너뜁니다.
        if target_price is None or target_price <= 0:
            logging.warning(f'[타임컷 매도] {current_dt.isoformat()} - {stock_code}: 목표 가격이 0이거나 유효하지 않아 매매를 건너뜁니다.')
            return False

        price_diff_ratio = abs(target_price - current_price) * 100 / target_price 
        
        if price_diff_ratio <= max_deviation_ratio:
            logging.info(f'[타임컷 강제매도] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:.2f}, 매도가: {current_price:.2f}, 괴리율: {price_diff_ratio:.2%}')
            self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
            self.reset_signal(stock_code)
            return True
        else:
            logging.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:.2f}, 매도가: {current_price:.2f}, 괴리율: {price_diff_ratio:.2%} ({max_deviation_ratio:.1%} 초과)')
            return False

