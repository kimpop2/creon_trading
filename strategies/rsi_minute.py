import datetime
import logging
import pandas as pd
import numpy as np 

from strategies.strategy import MinuteStrategy 
from util.strategies_util import calculate_momentum, calculate_rsi 

logger = logging.getLogger(__name__)

class RSIMinute(MinuteStrategy): 
    def __init__(self, data_store, strategy_params, broker): 
        super().__init__(data_store, strategy_params, broker)
        self.signals = {}  # DailyStrategy에서 업데이트 받을 시그널 저장
        self.last_portfolio_check = None  # 마지막 포트폴리오 체크 시간
        self.rsi_cache = {}  # RSI 캐시 추가
        self.last_prices = {}  # 마지막 가격 캐시 추가
        self.last_rsi_values = {}  # 마지막 RSI 값 캐시 추가

    def update_signals(self, signals):
        """
        DailyStrategy에서 생성된 신호들을 업데이트합니다.
        """
        self.signals = {
            stock_code: {**info, 'traded_today': False}
            for stock_code, info in signals.items()
        }
        logging.debug(f"RSIMinute: {len(self.signals)}개의 시그널로 업데이트 완료. 첫 종목: {next(iter(self.signals), 'N/A')}")

    def _calculate_rsi(self, historical_data, stock_code):
        """
        RSI를 계산하고 캐시합니다.
        캐시된 데이터를 활용하여 효율적으로 계산합니다.
        """
        if len(historical_data) < self.strategy_params['minute_rsi_period'] + 1:
            return None

        # 캐시된 RSI 값이 있고, 새로운 데이터가 1개만 추가된 경우
        if stock_code in self.last_rsi_values and len(historical_data) == len(self.rsi_cache.get(stock_code, [])) + 1:
            last_rsi = self.last_rsi_values[stock_code]
            last_prices = self.rsi_cache[stock_code]
            new_price = historical_data['close'].iloc[-1]
            
            # RSI 업데이트 계산
            price_change = new_price - last_prices[-1]
            if price_change > 0:
                gain = price_change
                loss = 0
            else:
                gain = 0
                loss = -price_change
                
            # 이동평균 업데이트
            avg_gain = (last_rsi['avg_gain'] * (self.strategy_params['minute_rsi_period'] - 1) + gain) / self.strategy_params['minute_rsi_period']
            avg_loss = (last_rsi['avg_loss'] * (self.strategy_params['minute_rsi_period'] - 1) + loss) / self.strategy_params['minute_rsi_period']
            
            # RSI 계산
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # 캐시 업데이트
            self.last_rsi_values[stock_code] = {'avg_gain': avg_gain, 'avg_loss': avg_loss}
            self.rsi_cache[stock_code] = historical_data['close'].values
            self.last_prices[stock_code] = new_price
            
            return rsi
        
        # 전체 RSI 계산 (캐시가 없는 경우)
        delta = historical_data['close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gain = gains.rolling(window=self.strategy_params['minute_rsi_period'], min_periods=1).mean().iloc[-1]
        avg_loss = losses.rolling(window=self.strategy_params['minute_rsi_period'], min_periods=1).mean().iloc[-1]
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # 캐시 업데이트
        self.last_rsi_values[stock_code] = {'avg_gain': avg_gain, 'avg_loss': avg_loss}
        self.rsi_cache[stock_code] = historical_data['close'].values
        self.last_prices[stock_code] = historical_data['close'].iloc[-1]
        
        return rsi

    def _should_check_portfolio(self, current_dt):
        """포트폴리오 체크가 필요한 시점인지 확인합니다."""
        if self.last_portfolio_check is None:
            return True
        
        current_time = current_dt.time()
        # 시간 비교를 정수로 변환하여 효율적으로 비교
        current_minutes = current_time.hour * 60 + current_time.minute
        check_minutes = [9 * 60, 15 * 60 + 20]  # 9:00, 15:20
        
        if current_minutes in check_minutes and (self.last_portfolio_check.date() != current_dt.date() or 
                                               (self.last_portfolio_check.hour * 60 + self.last_portfolio_check.minute) not in check_minutes):
            self.last_portfolio_check = current_dt
            return True
            
        return False

    def run_minute_logic(self, stock_code, current_dt):
        """
        분봉 데이터를 기반으로 RSI 매수/매도 로직을 실행합니다.
        """
        if stock_code not in self.signals:
            return

        signal_info = self.signals[stock_code]
        momentum_signal = signal_info.get('signal')

        # 당일 이미 거래가 발생했으면 추가 거래 방지
        if signal_info.get('traded_today', False):
            return

        # 현재 시간의 분봉 데이터 가져오기 (한 번만 조회)
        minute_df = self._get_bar_at_time('minute', stock_code, current_dt)
        if minute_df.empty:
            return
        
        current_price = minute_df['close']
        current_minute_time = current_dt.time()
        current_minutes = current_minute_time.hour * 60 + current_minute_time.minute

        # RSI 계산 (캐시 활용)
        historical_minute_data = self._get_historical_data_up_to(
            'minute',
            stock_code,
            current_dt,
            lookback_period=self.strategy_params['minute_rsi_period'] + 1
        )
        
        current_rsi_value = self._calculate_rsi(historical_minute_data, stock_code)
        if current_rsi_value is None:
            return

        current_position_size = self.broker.get_position_size(stock_code)

        # 포트폴리오 체크 (필요한 시점에만)
        if self._should_check_portfolio(current_dt):
            current_prices_for_portfolio_check = {stock_code: current_price}
            for code in list(self.broker.positions.keys()):
                if code != stock_code:
                    # 캐시된 가격 사용
                    if code in self.last_prices:
                        current_prices_for_portfolio_check[code] = self.last_prices[code]
                    else:
                        price_data = self._get_bar_at_time('minute', code, current_dt)
                        if price_data is not None:
                            current_prices_for_portfolio_check[code] = price_data['close']
                            self.last_prices[code] = price_data['close']
                        else:
                            daily_price = self._get_bar_at_time('daily', code, current_dt.date())
                            if daily_price is not None:
                                current_prices_for_portfolio_check[code] = daily_price['close']
                                self.last_prices[code] = daily_price['close']

            if self.broker.check_and_execute_portfolio_stop_loss(current_prices_for_portfolio_check, current_dt):
                for code in list(self.signals.keys()):
                    if code in self.broker.positions and self.broker.positions[code]['size'] == 0:
                        self.signals[code]['traded_today'] = True
                return

        # 개별 종목 손절 체크
        if current_position_size > 0:
            if self.broker.check_and_execute_stop_loss(stock_code, current_price, current_dt):
                self.signals[stock_code]['traded_today'] = True
                return

        # 매수/매도 로직 실행
        if momentum_signal == 'buy' and current_position_size == 0:
            target_quantity = signal_info.get('target_quantity', 0)
            if target_quantity > 0:
                if current_rsi_value <= self.strategy_params['minute_rsi_oversold']:
                    logging.info(f'[RSI 매수] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
                    if self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt):
                        self.signals[stock_code]['traded_today'] = True
                elif current_minutes == 15 * 60 + 20:  # 15:20
                    logging.info(f'[강제 매수] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
                    if self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt):
                        self.signals[stock_code]['traded_today'] = True

        elif momentum_signal == 'sell' and current_position_size > 0:
            if current_rsi_value >= self.strategy_params['minute_rsi_overbought']:
                logging.info(f'[RSI 매도] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {current_position_size}주')
                if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt):
                    self.signals[stock_code]['traded_today'] = True
            elif current_minutes == 9 * 60 + 5:  # 9:05
                logging.info(f'[강제 매도] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {current_position_size}주')
                if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt):
                    self.signals[stock_code]['traded_today'] = True

