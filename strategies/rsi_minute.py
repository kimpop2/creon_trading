import datetime
import logging
import pandas as pd
import numpy as np 

from strategies.strategy import MinuteStrategy 

from util.utils import calculate_momentum, calculate_rsi 

logger = logging.getLogger(__name__)

class RSIMinute(MinuteStrategy): # MinuteStrategy 상속 유지
    def __init__(self, data_store, strategy_params, broker): 
        super().__init__(data_store, strategy_params, broker) # BaseStrategy의 __init__ 호출 시 position_info 인자 제거
        self.signals = {} # This will be updated from DualMomentumDaily
        # self.position_info는 이제 broker가 직접 관리하므로 삭제

    def update_signals(self, signals):
        """DualMomentumDaily에서 생성된 모멘텀 신호를 업데이트합니다."""
        self.signals = signals

    # 아래 손절 관련 메서드들은 Broker로 이동했으므로 삭제합니다.
    # def _check_stop_loss(self, stock_code, current_price, position_info, current_dt):
    # def _check_portfolio_stop_loss(self, current_prices):
    # def _update_position_info(self, stock_code, current_price):

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

        # --- 손절 로직 (매수/매도 신호와 관계없이 최우선으로 체크) ---
        # 포트폴리오 전체 손절 체크 (보유 종목이 하나라도 있을 때만)
        # 현재 가격 정보 수집 (포트폴리오 손절 체크를 위해 모든 보유 종목 가격 필요)
        # current_prices_for_portfolio_check = {stock_code: current_price}
        # for code in list(self.broker.positions.keys()): 
        #     if code != stock_code: # 현재 처리 중인 종목이 아닌 다른 보유 종목
        #         price_data = self._get_bar_at_time('minute', code, current_dt)
        #         if price_data is not None:
        #             current_prices_for_portfolio_check[code] = price_data['close']
        #         else: # 분봉 데이터가 없으면 일봉 마지막 가격이라도 사용 (정확도는 떨어지지만 없는 것보다 낫다)
        #             daily_price = self._get_bar_at_time('daily', code, current_dt.date())
        #             if daily_price is not None:
        #                 current_prices_for_portfolio_check[code] = daily_price['close']
                        
        # if self.broker.check_and_execute_portfolio_stop_loss(current_prices_for_portfolio_check, current_dt):
        #     # 포트폴리오 전체 손절이 발생하면 모든 포지션이 청산되므로 더 이상 다른 매매 로직 실행하지 않음
        #     # 모든 종목의 traded_today 플래그를 True로 설정 (이미 매도되었으므로)
        #     for code in list(self.signals.keys()):
        #         if code in self.broker.transaction_log[-1]: # 방금 매도된 종목이라면
        #              self.signals[code]['traded_today'] = True
        #     return

        # # 개별 종목 손절 체크 (종목을 보유하고 있는 경우에만)
        # if self.broker.get_position_size(stock_code) > 0:
        #     if self.broker.check_and_execute_stop_loss(stock_code, current_price, current_dt):
        #         # 개별 종목 손절이 발생했으므로 당일 해당 종목 추가 거래 방지
        #         self.signals[stock_code]['traded_today'] = True
        #         return

        # --- 기존 매수/매도 로직 (손절이 발생하지 않은 경우에만) ---
        momentum_signal_info = self.signals.get(stock_code)
        if momentum_signal_info is None: # 해당 종목에 대한 시그널 정보가 아직 없으면 (ex: 초기 단계)
            return

        momentum_signal = momentum_signal_info['signal']
        signal_date = momentum_signal_info['signal_date']
        target_quantity = momentum_signal_info.get('target_quantity', 0)

        # 시그널 유효성 검사: 시그널이 없거나, 시그널이 발생한 날짜 이전이면 건너뛴다.
        if momentum_signal is None or current_minute_date < signal_date:
            return
            
        # 당일 이미 거래가 발생했으면 추가 거래 방지 (손절에 의해 traded_today가 설정될 수 있음)
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
                return # 0주 매수 시도 방지

            if current_position_size <= 0: # 현재 보유하고 있지 않은 경우에만 매수 시도
                buy_executed = False
                # 오전 10시 이후 RSI 과매도 구간에서 매수 시도
                if current_minute_time >= datetime.time(10, 0):
                    if current_rsi_value <= self.strategy_params['minute_rsi_oversold']:
                        logging.info(f'[RSI 매수] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
                        buy_executed = self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
                        
                # 장 마감 직전 강제 매수
                elif current_minute_time == datetime.time(15, 20):
                    logging.info(f'[강제 매수] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
                    buy_executed = self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
                        
                if buy_executed:
                    self.signals[stock_code]['traded_today'] = True # 매수 완료 시 당일 추가 거래 방지
                    # self.position_info[stock_code] = {'highest_price': current_price, 'entry_date': current_dt.date()} # 이 부분은 이제 broker가 관리


        # --- 매도 로직 ---
        elif momentum_signal == 'sell':
            if current_position_size > 0: # 현재 보유하고 있는 경우에만 매도 시도
                sell_executed = False
                # RSI 과매수 구간에서 매도 시도
                if current_rsi_value >= self.strategy_params['minute_rsi_overbought']:
                    logging.info(f'[RSI 매도] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원')
                    sell_executed = self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
                    
                # 장 시작 직후 강제 매도 (리밸런싱 매도 종목)
                elif current_minute_time == datetime.time(9, 5):
                    logging.info(f'[강제 매도] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원')
                    sell_executed = self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
                    
                if sell_executed:
                    self.signals[stock_code]['traded_today'] = True # 매도 완료 시 당일 추가 거래 방지
                    # if stock_code in self.position_info: # 이 부분은 이제 broker가 관리
                    #     del self.position_info[stock_code]

    def run_daily_logic(self, current_daily_date):
        """RSIMinute는 일봉 로직을 직접 수행하지 않습니다."""
        pass