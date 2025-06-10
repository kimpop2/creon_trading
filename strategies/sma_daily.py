# strategies.py

import datetime
import logging
import pandas as pd
import numpy as np

# from util.utils import calculate_momentum, get_next_weekday # 주석 처리 또는 제거 (이미 utils.py에서 임포트)
from util.utils import * # utils.py에 있는 모든 함수를 임포트한다고 가정
# DailyStrategy 추상 클래스 임포트
from strategies.strategy_base import DailyStrategy

class SMADaily(DailyStrategy):
    def __init__(self, data_store, strategy_params, broker):
        self.data_store = data_store
        self.strategy_params = strategy_params
        self.broker = broker
        self.momentum_signals = {} # {stock_code: {'signal': 'buy'/'sell'/'hold', 'signal_date': date, 'traded_today': False, 'target_quantity': int}}
        self.short_window = strategy_params.get('short_window', 20)  # 단기 이평선 기간, 기본값 20
        self.long_window = strategy_params.get('long_window', 60)    # 장기 이평선 기간, 기본값 60

        self._initialize_momentum_signals_for_all_stocks()

    def _initialize_momentum_signals_for_all_stocks(self):
        """백테스터에 추가된 모든 종목에 대해 모멘텀 시그널을 초기화합니다."""
        for stock_code in self.data_store['daily']:
            if stock_code not in self.momentum_signals:
                self.momentum_signals[stock_code] = {
                    'momentum_score': 0,
                    'rank': 0,
                    'signal': None,
                    'signal_date': None,
                    'traded_today': False,
                    'target_amount': 0,
                    'target_quantity': 0
                }

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

    def _calculate_target_quantity(self, stock_code, current_price):
        """주어진 가격에서 동일비중 투자에 필요한 수량을 계산합니다."""
        # 여기서는 매수 시그널이 발생했을 때만 계산하므로, 모든 가용 현금을 투자한다고 가정
        # 또는 num_stocks와 같은 파라미터가 strategy_params에 있다면 그에 따라 분배
        num_stocks_to_buy = self.strategy_params.get('num_stocks_to_buy', 1) # 한 번에 몇 종목 매수할지
        if num_stocks_to_buy == 0:
            logging.warning("num_stocks_to_buy가 0으로 설정되어 매수 수량을 계산할 수 없습니다.")
            return 0
            
        target_amount = self.broker.cash / num_stocks_to_buy
        available_cash = self.broker.cash
        commission_rate = self.broker.commission_rate
        max_buyable_amount = available_cash / (1 + commission_rate)
        
        actual_investment_amount = min(target_amount, max_buyable_amount)
        
        quantity = int(actual_investment_amount / current_price)
        
        if quantity > 0:
            total_cost = current_price * quantity * (1 + commission_rate)
            if total_cost > available_cash:
                quantity -= 1 # 1주 줄여서 재계산 (최소 거래 단위 1주이므로)
            
        if quantity > 0:
            logging.info(f"{stock_code} 종목 매수 수량 계산: {quantity}주 (목표금액: {target_amount:,.0f}원, 가용현금: {available_cash:,.0f}원, 현재가: {current_price:,.0f}원)")
        else:
            logging.warning(f"{stock_code} 종목 매수 불가: 현금 부족 (목표금액: {target_amount:,.0f}원, 가용현금: {available_cash:,.0f}원, 현재가: {current_price:,.0f}원)")
        
        return quantity
    
    def run_weekly_momentum_logic(self, current_daily_date):
        pass
    
    def run_daily_logic(self, current_daily_date):
        """
        일봉 SMA 전략 로직을 실행하고 신호를 생성합니다.
        골든 크로스(단기 이평선 > 장기 이평선) 매수, 데드 크로스(단기 이평선 < 장기 이평선) 매도.
        """
        logging.info(f'{current_daily_date.isoformat()} - --- 일봉 SMA 로직 실행 중 ---')

        # 모든 종목에 대해 시그널 초기화
        for stock_code in self.momentum_signals:
            self.momentum_signals[stock_code]['traded_today'] = False

        for stock_code in self.data_store['daily']:
            daily_df = self.data_store['daily'][stock_code]
            if daily_df.empty:
                logging.debug(f'{stock_code} 종목의 일봉 데이터가 비어있습니다.')
                continue

            # 필요한 데이터 기간 설정 (장기 이평선 기간 + 여유분)
            lookback_needed = self.long_window + 5 # 최소 필요한 데이터보다 조금 더 가져옴
            historical_data = self._get_historical_data_up_to(
                'daily', 
                stock_code, 
                current_daily_date, 
                lookback_period=lookback_needed
            )

            if len(historical_data) < self.long_window:
                logging.debug(f'{stock_code} 종목의 SMA 계산을 위한 데이터가 부족합니다 (필요: {self.long_window}, 현재: {len(historical_data)}).')
                continue

            # 이동평균선 계산
            # `ta` 라이브러리 등이 있다면 더 효율적이지만, 여기서는 직접 계산
            short_ma = historical_data['close'].rolling(window=self.short_window).mean()
            long_ma = historical_data['close'].rolling(window=self.long_window).mean()

            # 최신 이평선 값 가져오기
            current_short_ma = short_ma.iloc[-1]
            current_long_ma = long_ma.iloc[-1]

            # 직전 이평선 값 (크로스오버 확인용)
            previous_short_ma = short_ma.iloc[-2] if len(short_ma) >= 2 else None
            previous_long_ma = long_ma.iloc[-2] if len(long_ma) >= 2 else None

            current_signal = self.momentum_signals[stock_code]['signal']
            
            # 신호 생성
            if previous_short_ma is not None and previous_long_ma is not None:
                # 골든 크로스: 단기 이평선이 장기 이평선을 상향 돌파
                if current_short_ma > current_long_ma and previous_short_ma <= previous_long_ma:
                    if current_signal != 'buy' and self.broker.get_position_size(stock_code) == 0:
                        self.momentum_signals[stock_code]['signal'] = 'buy'
                        current_price_daily = historical_data['close'].iloc[-1]
                        target_quantity = self._calculate_target_quantity(stock_code, current_price_daily)
                        self.momentum_signals[stock_code]['target_quantity'] = target_quantity
                        logging.info(f'[{current_daily_date.isoformat()}] {stock_code}: 골든 크로스 발생 (단기MA: {current_short_ma:.2f}, 장기MA: {current_long_ma:.2f}) -> 매수 신호, 목표수량 {target_quantity}주')
                    elif current_signal == 'buy' or self.broker.get_position_size(stock_code) > 0:
                        self.momentum_signals[stock_code]['signal'] = 'hold'
                        logging.debug(f'[{current_daily_date.isoformat()}] {stock_code}: 골든 크로스 유지 (보유 중) -> 홀딩')

                # 데드 크로스: 단기 이평선이 장기 이평선을 하향 돌파
                elif current_short_ma < current_long_ma and previous_short_ma >= previous_long_ma:
                    if current_signal != 'sell' and self.broker.get_position_size(stock_code) > 0:
                        self.momentum_signals[stock_code]['signal'] = 'sell'
                        logging.info(f'[{current_daily_date.isoformat()}] {stock_code}: 데드 크로스 발생 (단기MA: {current_short_ma:.2f}, 장기MA: {current_long_ma:.2f}) -> 매도 신호')
                    elif current_signal == 'sell' or self.broker.get_position_size(stock_code) == 0:
                        self.momentum_signals[stock_code]['signal'] = 'hold' # 매도 신호가 발생했지만 보유중이지 않으면 홀딩
                        logging.debug(f'[{current_daily_date.isoformat()}] {stock_code}: 데드 크로스 유지 (미보유 중) -> 홀딩 (매도할 것이 없음)')
                else: # 크로스오버가 발생하지 않고 추세가 유지되는 경우
                    if current_signal in ['buy', 'hold'] and current_short_ma > current_long_ma:
                        self.momentum_signals[stock_code]['signal'] = 'hold'
                        logging.debug(f'[{current_daily_date.isoformat()}] {stock_code}: 상승 추세 유지 -> 홀딩')
                    elif current_signal in ['sell', 'hold'] and current_short_ma < current_long_ma:
                        self.momentum_signals[stock_code]['signal'] = 'hold'
                        logging.debug(f'[{current_daily_date.isoformat()}] {stock_code}: 하락 추세 유지 -> 홀딩')
                    else:
                        self.momentum_signals[stock_code]['signal'] = 'hold' # 기본적으로는 홀딩
                        logging.debug(f'[{current_daily_date.isoformat()}] {stock_code}: 명확한 신호 없음 -> 홀딩')
            else:
                self.momentum_signals[stock_code]['signal'] = 'hold' # 충분한 데이터가 없으면 홀딩
                logging.debug(f'[{current_daily_date.isoformat()}] {stock_code}: 이평선 계산을 위한 데이터 부족 또는 초기 단계 -> 홀딩')

        # 리밸런싱 계획 요약은 Backtester에서 처리하므로, 여기서는 별도 로깅 없이 시그널만 업데이트

    def run_minute_logic(self, stock_code, current_minute_dt):
        """일봉 전략은 분봉 로직을 직접 수행하지 않습니다."""
        pass

    def update_momentum_signals(self, signals):
        """
        일봉 전략은 외부에서 모멘텀 시그널을 업데이트 받지 않습니다.
        Backtester가 SMADaily의 signals를 직접 참조하여 MinuteStrategy로 전달하도록 설계되어야 합니다.
        """
        # 이 메서드는 실제로 DualMomentumDaily처럼 `momentum_signals`를 받아서 업데이트하지 않습니다.
        # 대신, 백테스터가 `self.momentum_signals`를 직접 참조하여 MinuteStrategy로 전달하도록 설계해야 합니다.
        # 따라서 이 메서드는 비워두거나 pass를 사용합니다.
        pass