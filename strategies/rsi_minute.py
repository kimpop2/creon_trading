import datetime
import logging
import pandas as pd
import numpy as np 

from strategies.strategy import MinuteStrategy 
from util.utils import calculate_momentum, calculate_rsi 

logger = logging.getLogger(__name__)

class RSIMinute(MinuteStrategy): 
    def __init__(self, data_store, strategy_params, broker): 
        super().__init__(data_store, strategy_params, broker)
        self.signals = {} # DailyStrategy에서 업데이트 받을 시그널 저장

    def update_signals(self, signals):
        """
        DailyStrategy에서 생성된 신호들을 업데이트합니다.
        매일 새로운 신호가 들어오면 기존 신호를 덮어쓰고, traded_today를 False로 초기화합니다.
        """
        # 기존 self.signals를 완전히 새로운 signals로 대체합니다.
        # traded_today 플래그는 여기서 False로 초기화하여, 해당 날짜의 분봉 매매가 새로 시작됨을 알립니다.
        self.signals = {
            stock_code: {**info, 'traded_today': False} # info를 복사하고 traded_today를 초기화
            for stock_code, info in signals.items()
        }
        logging.debug(f"RSIMinute: {len(self.signals)}개의 시그널로 업데이트 완료. 첫 종목: {next(iter(self.signals), 'N/A')}")

    def run_minute_logic(self, stock_code, current_dt):
        """
        분봉 데이터를 기반으로 RSI 매수/매도 로직을 실행합니다.
        """
        if stock_code not in self.signals:
            logging.debug(f"{current_dt.isoformat()} - {stock_code}: 시그널 없음 (run_minute_logic), 분봉 로직 건너뜀.")
            return

        signal_info = self.signals[stock_code]
        momentum_signal = signal_info.get('signal') # 'buy', 'sell', 'hold'

        # 당일 이미 거래가 발생했으면 추가 거래 방지
        if signal_info.get('traded_today', False):
            logging.debug(f"{current_dt.isoformat()} - {stock_code}: 당일 이미 거래 완료 (traded_today=True), 분봉 로직 건너뜀.")
            return

        # 현재 시간의 분봉 데이터 가져오기
        minute_df = self._get_bar_at_time('minute', stock_code, current_dt) 
        if minute_df.empty:
            logging.debug(f"{current_dt.isoformat()} - {stock_code}: 해당 시간 분봉 데이터 없음, 건너뜜.")
            return
        
        current_price = minute_df['close']
        current_minute_time = current_dt.time() 

        # 최근 14분봉 데이터 (RSI 계산을 위해)
        historical_minute_data = self._get_historical_data_up_to(
            'minute',
            stock_code,
            current_dt,
            lookback_period=self.strategy_params['minute_rsi_period'] + 1 
        )

        if len(historical_minute_data) < self.strategy_params['minute_rsi_period'] + 1: # RSI를 위한 최소 데이터 수
            logging.debug(f"{current_dt.isoformat()} - {stock_code}: RSI 계산을 위한 분봉 데이터 부족 ({len(historical_minute_data)}/{self.strategy_params['minute_rsi_period'] + 1}).")
            return

        current_rsi_value = calculate_rsi(historical_minute_data, self.strategy_params['minute_rsi_period']).iloc[-1]
        current_position_size = self.broker.get_position_size(stock_code)

        buy_executed = False
        sell_executed = False
        # 사용하려면 매수 매도 로직 위에서 제일 먼저 실행되어야 한다. 코드변경으로 실행 안될지도 모른다.
        # --- 손절 로직 (매수/매도 신호와 관계없이 최우선으로 체크) ---
        # 포트폴리오 전체 손절 체크 (보유 종목이 하나라도 있을 때만)
        # 현재 가격 정보 수집 (포트폴리오 손절 체크를 위해 모든 보유 종목 가격 필요)
        current_prices_for_portfolio_check = {stock_code: current_price}
        for code in list(self.broker.positions.keys()): 
            if code != stock_code: # 현재 처리 중인 종목이 아닌 다른 보유 종목
                price_data = self._get_bar_at_time('minute', code, current_dt)
                if price_data is not None:
                    current_prices_for_portfolio_check[code] = price_data['close']
                else: # 분봉 데이터가 없으면 일봉 마지막 가격이라도 사용 (정확도는 떨어지지만 없는 것보다 낫다)
                    daily_price = self._get_bar_at_time('daily', code, current_dt.date())
                    if daily_price is not None:
                        current_prices_for_portfolio_check[code] = daily_price['close']
                        
        # if self.broker.check_and_execute_portfolio_stop_loss(current_prices_for_portfolio_check, current_dt):
        #     # 포트폴리오 전체 손절이 발생하면 모든 포지션이 청산되므로 더 이상 다른 매매 로직 실행하지 않음
        #     # 모든 종목의 traded_today 플래그를 True로 설정 (이미 매도되었으므로)
        #     for code in list(self.signals.keys()):
        #         if code in self.broker.transaction_log[-1]: # 방금 매도된 종목이라면
        #              self.signals[code]['traded_today'] = True
        #     return

        # 개별 종목 손절 체크 (종목을 보유하고 있는 경우에만)
        if self.broker.get_position_size(stock_code) > 0:
            if self.broker.check_and_execute_stop_loss(stock_code, current_price, current_dt):
                # 개별 종목 손절이 발생했으므로 당일 해당 종목 추가 거래 방지
                self.signals[stock_code]['traded_today'] = True
                return
        # --- 손절 로직 끝 ###

        # --- 매수 로직 ---
        if momentum_signal == 'buy':
            if current_position_size == 0: 
                target_quantity = signal_info.get('target_quantity', 0)
                if current_rsi_value <= self.strategy_params['minute_rsi_oversold']:
                    
                    if target_quantity > 0:
                        logging.info(f'[RSI 매수] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
                        buy_executed = self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
                # 강제매수
                elif current_minute_time == datetime.time(15, 20):
                    logging.info(f'[강제 매수] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
                    buy_executed = self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
                
                if buy_executed:
                    self.signals[stock_code]['traded_today'] = True 

        # --- 매도 로직 ---
        elif momentum_signal == 'sell':
            if current_position_size > 0: 
                sell_executed = False
                if current_rsi_value >= self.strategy_params['minute_rsi_overbought']:
                    logging.info(f'[RSI 매도] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {current_position_size}주')
                    sell_executed = self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
                
                # 장 시작 직후 강제 매도 (리밸런싱 매도 종목)
                elif current_minute_time == datetime.time(9, 5): # 9시 5분 강제 매도
                    logging.info(f'[강제 매도] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {current_position_size}주')
                    sell_executed = self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
                    
                if sell_executed:
                    self.signals[stock_code]['traded_today'] = True

