# strategies/rsi_minute.py (기존 코드를 바탕으로 주요 부분만)

import logging
import pandas as pd
import numpy as np

from strategies.strategy import MinuteStrategy 

class RSIMinute(MinuteStrategy):
    def __init__(self, data_store, strategy_params, broker):
        super().__init__(data_store, strategy_params, broker)
        self.data_store = data_store
        self.signals = {} # 일봉 전략으로부터 받은 시그널을 저장

        self.minute_rsi_period = self.strategy_params.get('minute_rsi_period', 14)
        self.minute_rsi_oversold = self.strategy_params.get('minute_rsi_oversold', 30)
        self.minute_rsi_overbought = self.strategy_params.get('minute_rsi_overbought', 70)
        
        # 듀얼 모멘텀 전략에서 사용할 ETF 코드를 여기서도 참조할 수 있도록
        # (만약 이 코드가 듀얼 모멘텀 전략과 연동된다면)
        self.safe_asset_code = self.strategy_params.get('safe_asset_code') 
        self.inverse_etf_code = self.strategy_params.get('inverse_etf_code')

        logging.info(f"RSIMinute 전략 초기화 완료. RSI 기간: {self.minute_rsi_period}, 과매도: {self.minute_rsi_oversold}, 과매수: {self.minute_rsi_overbought}")

    def update_signals(self, new_signals):
        """일봉 전략으로부터 최신 시그널을 업데이트합니다."""
        self.signals = new_signals
        logging.debug(f"분봉 전략 시그널 업데이트됨. 총 {len(self.signals)}개 종목.")

    def run_minute_logic(self, stock_code, current_minute_dt):
        """
        분봉 데이터를 기반으로 RSI 전략을 실행하고 실제 매매를 시도합니다.
        일봉 전략에서 전달받은 시그널에 따라 매매 여부를 결정합니다.
        """
        signal_info = self.signals.get(stock_code)
        if not signal_info or signal_info.get('signal') == 'hold':
            logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: 'hold' 시그널이거나 시그널 없음. 분봉 매매 로직 건너뜜.")
            return

        if signal_info.get('traded_today', False): # 이미 오늘 매매했으면 스킵
            logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: 이미 오늘 매매 완료. 분봉 매매 로직 건너뜜.")
            return

        # 분봉 데이터 로드 (Backtester에서 _get_minute_data_for_signal_dates를 통해 이미 로드됨)
        # self.data_store['minute'][stock_code]는 특정 날짜의 분봉 데이터 딕셔너리를 포함
        minute_data_for_stock = self.data_store['minute'].get(stock_code, {}).get(current_minute_dt.date()) # 그 날의 분봉 데이터
        
        if minute_data_for_stock is None or minute_data_for_stock.empty:
            logging.warning(f"[{current_minute_dt.isoformat()}] {stock_code}: 분봉 데이터 부족. 분봉 매매 로직 건너뜜.")
            return
        
        # 현재 분봉 데이터 (current_minute_dt까지의 데이터만 사용)
        current_minute_bar = minute_data_for_stock.loc[minute_data_for_stock.index <= current_minute_dt]
        if current_minute_bar.empty:
            logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: 현재 시간까지의 분봉 바 데이터 없음. 건너뜜.")
            return
        
        # RSI 계산을 위한 충분한 데이터 확인
        if len(current_minute_bar) < self.minute_rsi_period:
            logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: RSI 계산에 필요한 분봉 데이터 부족. ({len(current_minute_bar)} < {self.minute_rsi_period}).")
            return

        # RSI 계산 (utils.py 또는 RSI 계산 함수 직접 구현)
        # 이전에 제공된 calculate_rsi 함수가 있다고 가정
        try:
            rsi = self.calculate_rsi(current_minute_bar['close'], self.minute_rsi_period).iloc[-1]
        except Exception as e:
            logging.error(f"[{current_minute_dt.isoformat()}] {stock_code}: RSI 계산 오류: {e}. 매매 시도 불가.")
            return

        current_price = current_minute_bar['close'].iloc[-1]
        
        # --- 매수 로직 ---
        if signal_info['signal'] == 'buy':
            # 보유 중이거나, 매수 시그널인데 RSI 과매도 구간이면 매수
            if self.broker.get_position_size(stock_code) == 0 and rsi < self.minute_rsi_oversold:
                # 매수할 수량 결정 (예: 전체 현금의 일정 비율 또는 고정 금액)
                # 여기서는 'target_quantity'가 None으로 전달되었을 때,
                # 현재 현금을 기준으로 매수 가능한 최대 수량을 계산
                
                # 안전자산 또는 인버스 ETF의 경우 남은 현금 전량 매수 시도
                if stock_code == self.safe_asset_code or stock_code == self.inverse_etf_code:
                    quantity_to_buy = self.broker.cash // (current_price * (1 + self.broker.commission_rate))
                    if quantity_to_buy > 0:
                        self.broker.execute_order(stock_code, 'buy', quantity_to_buy, current_price, current_minute_dt)
                        signal_info['traded_today'] = True # 오늘 매매 완료 플래그
                        logging.info(f"[{current_minute_dt.isoformat()}] {stock_code} (ETF): RSI ({rsi:.2f}) 과매도({self.minute_rsi_oversold}). 전량 매수 시도. 수량: {quantity_to_buy}, 가격: {current_price:,.0f}")
                else: # 일반 주식
                    # 예를 들어, 남은 현금의 10%를 한 종목에 할당 (예시)
                    allocation_ratio = 0.1 
                    available_cash_for_trade = self.broker.cash * allocation_ratio
                    quantity_to_buy = int(available_cash_for_trade // (current_price * (1 + self.broker.commission_rate)))

                    if quantity_to_buy > 0:
                        self.broker.execute_order(stock_code, 'buy', quantity_to_buy, current_price, current_minute_dt)
                        signal_info['traded_today'] = True
                        logging.info(f"[{current_minute_dt.isoformat()}] {stock_code}: RSI ({rsi:.2f}) 과매도({self.minute_rsi_oversold}). 매수 시도. 수량: {quantity_to_buy}, 가격: {current_price:,.0f}")
            else:
                logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: 매수 시그널이나 RSI ({rsi:.2f}) 과매도 조건({self.minute_rsi_oversold}) 미충족 또는 이미 보유 중.")

        # --- 매도 로직 ---
        elif signal_info['signal'] == 'sell':
            # 보유 중이고, 매도 시그널이며 RSI 과매수 구간이면 매도
            if self.broker.get_position_size(stock_code) > 0 and rsi > self.minute_rsi_overbought:
                # 전량 매도
                quantity_to_sell = self.broker.get_position_size(stock_code)
                self.broker.execute_order(stock_code, 'sell', quantity_to_sell, current_price, current_minute_dt)
                signal_info['traded_today'] = True # 오늘 매매 완료 플래그
                logging.info(f"[{current_minute_dt.isoformat()}] {stock_code}: RSI ({rsi:.2f}) 과매수({self.minute_rsi_overbought}). 전량 매도 시도. 수량: {quantity_to_sell}, 가격: {current_price:,.0f}")
            else:
                logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: 매도 시그널이나 RSI ({rsi:.2f}) 과매수 조건({self.minute_rsi_overbought}) 미충족 또는 보유 포지션 없음.")

    # calculate_rsi 함수는 utils.py에 있거나 여기에 직접 구현해야 합니다.
    # 예시:
    def calculate_rsi(self, series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0) # 첫 부분의 NaN 값 처리