# strategy/reversion_strategy.py
from datetime import date, datetime
import pandas as pd
from strategies.strategy import Strategy
from util.strategies_util import *

class ReversionStrategy(Strategy):
    def __init__(self, broker, manager, data_store, strategy_params):
        super().__init__(broker, manager, data_store, strategy_params)
        self.rsi_period = self.strategy_params.get('rsi_period', 14)
        self.rsi_oversold = self.strategy_params.get('rsi_oversold', 30)
        self.rsi_overbought = self.strategy_params.get('rsi_overbought', 70)
        self.bollinger_period = self.strategy_params.get('bollinger_period', 20)
        self.bollinger_std_dev = self.strategy_params.get('bollinger_std_dev', 2)
        self.strategy_name = "ReversionStrategy"

    def run_strategy_logic(self, current_date: date) -> None:
        logger.info(f"[{current_date}] 평균 회귀 전략 (일봉) 로직 실행 시작")
        buy_candidates = set()
        sell_candidates = set()
        all_stocks = []

        for stock_code, daily_df in self.data_store['daily'].items():
            # df_up_to_current_date는 .copy()를 통해 명시적으로 복사본을 생성 (SettingWithCopyWarning 방지)
            df_up_to_current_date = self._get_historical_data_up_to('daily', stock_code, current_date).copy()
            if df_up_to_current_date.empty or len(df_up_to_current_date) < max(self.rsi_period, self.bollinger_period):
                logger.debug(f"{stock_code}: 데이터 부족으로 평균 회귀 전략 건너뜀.")
                continue

            # RSI 계산
            df_up_to_current_date['RSI'] = calculate_rsi(df_up_to_current_date['close'], self.rsi_period)

            # **수정된 부분:**
            # 1. calculate_bollinger_bands에 전체 DataFrame을 전달
            # 2. 반환된 튜플을 변수에 할당 후 DataFrame에 컬럼 추가
            bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(
                df_up_to_current_date,  # DataFrame을 전달
                self.bollinger_period,
                self.bollinger_std_dev
            )
            df_up_to_current_date['BB_Upper'] = bb_upper
            df_up_to_current_date['BB_Lower'] = bb_lower
            df_up_to_current_date['BB_Mid'] = bb_mid

            latest_data = df_up_to_current_date.iloc[-1]

            # 매수 조건: RSI 과매도 구간 진입
            if latest_data['RSI'] <= self.rsi_oversold:
                buy_candidates.add(stock_code)
                logger.debug(f"{stock_code}: 매수 신호 (RSI 과매도: {latest_data['RSI']:.2f})")
            
            # 추가: 볼린저 밴드 하단 터치 또는 이탈 후 재진입 조건 활성화
            if 'BB_Lower' in latest_data and latest_data['close'] <= latest_data['BB_Lower']: # 하단 밴드에 근접하거나 터치
                buy_candidates.add(stock_code)
                logger.debug(f"{stock_code}: 매수 신호 (볼린저 밴드 하단 터치)")

            # 매도 조건: RSI 과매수 구간 진입
            if latest_data['RSI'] >= self.rsi_overbought:
                if stock_code in self.broker.positions:
                    sell_candidates.add(stock_code)
                    logger.debug(f"{stock_code}: 매도 신호 (RSI 과매수: {latest_data['RSI']:.2f})")
            
            # 추가: 볼린저 밴드 상단 터치 또는 이탈 후 재진입 조건 활성화
            if 'BB_Upper' in latest_data and latest_data['close'] >= latest_data['BB_Upper']: # 상단 밴드에 근접하거나 터치
                if stock_code in self.broker.positions:
                    sell_candidates.add(stock_code)
                    logger.debug(f"{stock_code}: 매도 신호 (볼린저 밴드 상단 터치)")

            all_stocks.append((stock_code, 0)) # 순위는 의미 없지만 형식 맞춤

        self._reset_all_signals()
        self._generate_signals(current_date, buy_candidates, all_stocks, sell_candidates)
        self._log_rebalancing_summary(current_date, buy_candidates, set(self.broker.positions.keys()), sell_candidates)
        logger.info(f"[{current_date}] 평균 회귀 전략 (일봉) 로직 실행 완료")

    def run_trading_logic(self, current_minute_dt: datetime, stock_code: str) -> None:
        signal_info = self.signals.get(stock_code)
        if not signal_info or signal_info.get('traded_today'):
            return

        signal_type = signal_info.get('signal')
        target_quantity = signal_info.get('target_quantity', 0)
        target_price_daily = signal_info.get('target_price', 0) # 일봉 기준 목표가격

        current_bar = self._get_bar_at_time('minute', stock_code, current_minute_dt)
        if not current_bar:
            return
        current_price = current_bar['close']

        # 매수 로직 (반등 확인)
        if signal_type == 'buy' and target_quantity > 0:
            # 분봉 데이터를 활용하여 단기 반등 신호 확인 (예: 직전 3개 분봉 연속 상승)
            minute_history = self._get_historical_data_up_to('minute', stock_code, current_minute_dt, lookback_period=3)
            if len(minute_history) >= 3 and \
               minute_history['close'].iloc[-1] > minute_history['close'].iloc[-2] and \
               minute_history['close'].iloc[-2] > minute_history['close'].iloc[-3]:
                
                # 일봉 신호의 목표가(전일 종가) 대비 괴리율 확인 후 매수
                max_deviation_ratio = self.strategy_params.get('max_deviation_ratio', 0.01)
                if target_price_daily > 0 and abs(current_price - target_price_daily) / target_price_daily <= max_deviation_ratio:
                    if self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_minute_dt):
                        logger.info(f"[실시간 매매] {stock_code} 매수 완료 (평균 회귀): {target_quantity}주 @ {current_price:,.0f}원")
                        self.reset_signal(stock_code)
                else:
                    logger.debug(f"[실시간 매매] {stock_code} 매수 대기 (평균 회귀): 현재가 {current_price:,.0f}원, 목표가 {target_price_daily:,.0f}원, 괴리율 {(abs(current_price - target_price_daily) / target_price_daily):.2%}")

            # 장 마감 직전 타임컷 매수 (필요 시)
            if current_minute_dt.time() >= datetime.datetime.strptime("15:25", "%H:%M").time():
                if self.execute_time_cut_buy(stock_code, current_minute_dt, current_price, target_quantity, self.strategy_params.get('max_time_cut_deviation_ratio', 0.02)):
                    logger.info(f"[실시간 매매] {stock_code} 타임컷 매수 완료 (평균 회귀)")

        # 매도 로직 (하락 확인 또는 이탈 방지)
        elif signal_type == 'sell' and stock_code in self.broker.positions:
            current_position_size = self.broker.positions[stock_code]['size']
            if current_position_size <= 0:
                self.reset_signal(stock_code)
                return

            # 분봉 데이터를 활용하여 단기 하락 신호 확인 (예: 직전 3개 분봉 연속 하락)
            minute_history = self._get_historical_data_up_to('minute', stock_code, current_minute_dt, lookback_period=3)
            if len(minute_history) >= 3 and \
               minute_history['close'].iloc[-1] < minute_history['close'].iloc[-2] and \
               minute_history['close'].iloc[-2] < minute_history['close'].iloc[-3]:
                
                # 일봉 신호의 목표가 대비 괴리율 확인 후 매도
                max_deviation_ratio = self.strategy_params.get('max_deviation_ratio', 0.01)
                if target_price_daily > 0 and abs(current_price - target_price_daily) / target_price_daily <= max_deviation_ratio:
                    if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_minute_dt):
                        logger.info(f"[실시간 매매] {stock_code} 매도 완료 (평균 회귀): {current_position_size}주 @ {current_price:,.0f}원")
                        self.reset_signal(stock_code)
                else:
                    logger.debug(f"[실시간 매매] {stock_code} 매도 대기 (평균 회귀): 현재가 {current_price:,.0f}원, 목표가 {target_price_daily:,.0f}원, 괴리율 {(abs(current_price - target_price_daily) / target_price_daily):.2%}")

            # 장 마감 직전 타임컷 매도
            if current_minute_dt.time() >= datetime.datetime.strptime("15:25", "%H:%M").time():
                if self.execute_time_cut_sell(stock_code, current_minute_dt, current_price, current_position_size, self.strategy_params.get('max_time_cut_deviation_ratio', 0.02)):
                    logger.info(f"[실시간 매매] {stock_code} 타임컷 매도 완료 (평균 회귀)")