import datetime
import logging
import pandas as pd
import numpy as np

from strategies.strategy import MinuteStrategy

from util.utils import calculate_momentum, calculate_rsi

logger = logging.getLogger(__name__)

class RSIMinute(MinuteStrategy): # MinuteStrategy 상속 유지
    def __init__(self, data_store, strategy_params, broker):
        super().__init__(data_store, strategy_params, broker)
        self.signals = {} # This will be updated from DualMomentumDaily

        # 손절매 관련 파라미터 및 개별 종목 포지션 정보는 Broker로 이동했으므로 제거
        # self.stop_loss_ratio = strategy_params.get('stop_loss_ratio', -5.0) 
        # ... (나머지 손절 파라미터 제거)
        # self.position_info = {} 
        # self.initial_portfolio_value_at_start = self.broker.cash

    def update_signals(self, signals):
        """DualMomentumDaily에서 생성된 모멘텀 신호를 업데이트합니다."""
        self.signals = signals

    # 손절매 관련 메서드 제거
    # def _check_stop_loss(self, stock_code, current_price, current_dt): ...
    # def _check_portfolio_stop_loss(self, current_prices, current_dt): ...
    # def _update_position_info(self, stock_code, current_price): ...

    def _get_last_price(self, stock_code):
        """종목의 마지막 거래 가격을 반환합니다. (이 메서드는 그대로 유지)"""
        daily_df = self.data_store['daily'].get(stock_code)
        if daily_df is not None and not daily_df.empty:
            return daily_df['close'].iloc[-1]
        return None

    def run_minute_logic(self, stock_code, current_dt):
        """분봉 데이터를 기반으로 실제 매수/매도 주문을 실행합니다."""
        current_minute_date = current_dt.date()
        if stock_code not in self.data_store['minute'] or current_minute_date not in self.data_store['minute'][stock_code]:
            return

        current_minute_bar = self._get_bar_at_time('minute', stock_code, current_dt)
        if current_minute_bar is None:
            return

        current_minute_time = current_dt.time()
        current_price = current_minute_bar['close']

        # --- 손절 로직 (이제 Broker에서 일봉 기준으로 처리하므로 여기서는 제거) ---
        # if self._check_portfolio_stop_loss(current_prices_for_portfolio_check, current_dt): ...
        # if self.broker.get_position_size(stock_code) > 0:
        #     if self._check_stop_loss(stock_code, current_price, current_dt): ...

        # --- 기존 매수/매도 로직 ---
        momentum_signal_info = self.signals.get(stock_code)
        if momentum_signal_info is None:
            return

        momentum_signal = momentum_signal_info['signal']
        signal_date = momentum_signal_info['signal_date']
        target_quantity = momentum_signal_info.get('target_quantity', 0)

        if momentum_signal is None or current_minute_date < signal_date:
            return

        if self.signals[stock_code]['traded_today']:
            return

        current_position_size = self.broker.get_position_size(stock_code)

        required_rsi_data_len = self.strategy_params['minute_rsi_period'] + 1
        minute_historical_data = self._get_historical_data_up_to('minute', stock_code, current_dt, lookback_period=required_rsi_data_len)

        if len(minute_historical_data) < required_rsi_data_len:
            return

        current_rsi_value_series = calculate_rsi(minute_historical_data, self.strategy_params['minute_rsi_period'])
        current_rsi_value = current_rsi_value_series.iloc[-1]

        if pd.isna(current_rsi_value):
            return

        # --- 매수 로직 ---
        if momentum_signal == 'buy':
            if target_quantity <= 0:
                logging.warning(f"[{current_dt.isoformat()}] {stock_code}: 매수 시그널이나, DualMomentumDaily에서 계산된 목표 수량(target_quantity)이 0입니다. 매수 시도 건너뜜.")
                return

            if current_position_size <= 0:
                buy_executed = False
                if current_minute_time >= datetime.time(10, 0):
                    if current_rsi_value <= self.strategy_params['minute_rsi_oversold']:
                        logging.info(f'[RSI 매수] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
                        buy_executed = self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)

                elif current_minute_time == datetime.time(15, 20):
                    logging.info(f'[강제 매수] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
                    buy_executed = self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)

                if buy_executed:
                    self.signals[stock_code]['traded_today'] = True
                    # 매수 시 position_info에 최고가와 진입 날짜 저장 (Broker가 관리)
                    # self.position_info[stock_code] = {'highest_price': current_price, 'entry_date': current_dt.date()}

        # --- 매도 로직 ---
        elif momentum_signal == 'sell':
            if current_position_size > 0:
                sell_executed = False
                if current_rsi_value >= self.strategy_params['minute_rsi_overbought']:
                    logging.info(f'[RSI 매도] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원')
                    sell_executed = self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)

                elif current_minute_time == datetime.time(9, 5):
                    logging.info(f'[강제 매도] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원')
                    sell_executed = self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)

                if sell_executed:
                    self.signals[stock_code]['traded_today'] = True
                    # 매도 시 position_info에서 삭제 (Broker가 관리)
                    # if stock_code in self.position_info:
                    #     del self.position_info[stock_code]

    def run_daily_logic(self, current_daily_date):
        """RSIMinute는 일봉 로직을 직접 수행하지 않습니다."""
        pass