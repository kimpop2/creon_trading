# strategy/trend_strategy.py
import pandas as pd
from datetime import date, datetime
from strategies.strategy import Strategy
from util.strategies_util import * # calculate_sma, calculate_macd 등

class TrendStrategy(Strategy):
    def __init__(self, broker, manager, data_store, strategy_params):
        super().__init__(broker, manager, data_store, strategy_params)
        self.short_ma_period = self.strategy_params.get('short_ma_period', 20)
        self.long_ma_period = self.strategy_params.get('long_ma_period', 60)
        self.macd_fast = self.strategy_params.get('macd_fast', 12)
        self.macd_slow = self.strategy_params.get('macd_slow', 26)
        self.macd_signal = self.strategy_params.get('macd_signal', 9)
        self.strategy_name = "TrendStrategy"
        # self.strategy_params = strategy_params # super().__init__에서 이미 설정되므로 중복될 수 있습니다. 필요없으면 제거하세요.
        
    def run_strategy_logic(self, current_date: date) -> None:
        logger.info(f"[{current_date}] 추세 추종 전략 (일봉) 로직 실행 시작")
        buy_candidates = set()
        sell_candidates = set()
        # 모든 종목에 대한 스코어를 저장할 리스트 (stock_code, score)
        all_stocks_with_scores = [] 

        for stock_code, daily_df in self.data_store['daily'].items():
            # current_date까지의 데이터만 사용
            df_up_to_current_date = self._get_historical_data_up_to('daily', stock_code, current_date).copy()
            if df_up_to_current_date.empty or len(df_up_to_current_date) < max(self.long_ma_period, self.macd_slow + self.macd_signal):
                logger.debug(f"{stock_code}: 데이터 부족으로 추세 추종 전략 건너뜀.")
                continue

            # 이동평균선 계산
            df_up_to_current_date['SMA_Short'] = calculate_sma(df_up_to_current_date['close'], self.short_ma_period)
            df_up_to_current_date['SMA_Long'] = calculate_sma(df_up_to_current_date['close'], self.long_ma_period)

            # MACD 계산
            macd_data = calculate_macd(df_up_to_current_date['close'], self.macd_fast, self.macd_slow, self.macd_signal)
            df_up_to_current_date['MACD'] = macd_data['macd'] # 'MACD' 키 이름 수정 확인
            df_up_to_current_date['MACD_Signal'] = macd_data['macd_signal']

            # 최신 데이터 포인트
            latest_data = df_up_to_current_date.iloc[-1]
            prev_latest_data = df_up_to_current_date.iloc[-2] if len(df_up_to_current_date) >= 2 else None

            # 스코어 초기화
            current_score = 0.0 # 기본 스코어

            if prev_latest_data is not None:
                # 매수 조건: 골든 크로스 또는 MACD 골든 크로스
                sma_golden_cross = (latest_data['SMA_Short'] > latest_data['SMA_Long'] and \
                                    prev_latest_data['SMA_Short'] <= prev_latest_data['SMA_Long'])
                macd_golden_cross = (latest_data['MACD'] > latest_data['MACD_Signal'] and \
                                     prev_latest_data['MACD'] <= prev_latest_data['MACD_Signal'])

                if sma_golden_cross or macd_golden_cross:
                    buy_candidates.add(stock_code)
                    logger.debug(f"{stock_code}: 매수 신호 (추세 추종)")
                    
                    # 스코어 계산 로직 (예시: MACD 이격도와 이동평균선 이격도 활용)
                    # MACD 이격도 (양수일수록 강한 상승세)
                    current_score += (latest_data['MACD'] - latest_data['MACD_Signal']) 
                    
                    # 이동평균선 이격도 (단기선이 장기선보다 얼마나 위에 있는지)
                    if latest_data['SMA_Long'] > 0: # 0으로 나누는 것 방지
                         current_score += ((latest_data['SMA_Short'] - latest_data['SMA_Long']) / latest_data['SMA_Long']) * 100 
                    
                    # 추가적인 스코어 가산 (예: 최근 5일 상승률 등)
                    # if len(df_up_to_current_date) >= 5:
                    #     recent_return = (latest_data['close'] - df_up_to_current_date.iloc[-5]['close']) / df_up_to_current_date.iloc[-5]['close'] * 100
                    #     current_score += recent_return

                # 매도 조건: 데드 크로스 또는 MACD 데드 크로스
                if (latest_data['SMA_Short'] < latest_data['SMA_Long'] and \
                    prev_latest_data['SMA_Short'] >= prev_latest_data['SMA_Long']) or \
                   (latest_data['MACD'] < latest_data['MACD_Signal'] and \
                    prev_latest_data['MACD'] >= prev_latest_data['MACD_Signal']):
                    if stock_code in self.broker.positions: # 보유 종목만 매도 신호 생성
                        sell_candidates.add(stock_code)
                        logger.debug(f"{stock_code}: 매도 신호 (추세 추종)")
        
            # 모든 종목과 계산된 스코어를 all_stocks_with_scores 리스트에 추가
            all_stocks_with_scores.append((stock_code, current_score))

        self._reset_all_signals() # 이전 날짜의 신호 초기화

        # 스코어를 바탕으로 매수 종목 선정 로직이 포함된 _generate_signals 호출
        self._generate_signals(current_date, buy_candidates, all_stocks_with_scores, sell_candidates)
        self._log_rebalancing_summary(current_date, buy_candidates, set(self.broker.positions.keys()), sell_candidates)
        logger.info(f"[{current_date}] 추세 추종 전략 (일봉) 로직 실행 완료")

    def run_trading_logic(self, current_minute_dt: datetime, stock_code: str) -> None:
        signal_info = self.signals.get(stock_code)
        if not signal_info or signal_info.get('traded_today'):
            return

        signal_type = signal_info.get('signal')
        target_quantity = signal_info.get('target_quantity', 0)
        target_price_daily = signal_info.get('target_price', 0) # 일봉 기준 목표가격

        # 현재 분봉 데이터 가져오기
        current_bar = self._get_bar_at_time('minute', stock_code, current_minute_dt)
        if not current_bar:
            return

        current_price = current_bar['close'] # 현재 분봉 종가를 현재가로 사용

        # 매수 로직
        if signal_type == 'buy' and target_quantity > 0:
            # 장 시작 10분 후부터 3시 20분까지 (예시 시간, 실제는 시장 개장 시간에 맞춰 조절)
            if current_minute_dt.time() >= datetime.datetime.strptime("09:10", "%H:%M").time() and \
               current_minute_dt.time() <= datetime.datetime.strptime("15:20", "%H:%M").time():
                
                # 전일 종가(target_price_daily) 대비 현재가(current_price)가 과도하게 괴리되지 않았는지 확인
                max_deviation_ratio = self.strategy_params.get('max_deviation_ratio', 0.01) # 기본 1%
                if target_price_daily > 0 and abs(current_price - target_price_daily) / target_price_daily <= max_deviation_ratio:
                    if self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_minute_dt):
                        logger.info(f"[실시간 매매] {stock_code} 매수 완료 (추세 추종): {target_quantity}주 @ {current_price:,.0f}원")
                        self.reset_signal(stock_code) # 매수 완료 후 신호 초기화
                else:
                    logger.debug(f"[실시간 매매] {stock_code} 매수 대기 (추세 추종): 현재가 {current_price:,.0f}원, 목표가 {target_price_daily:,.0f}원, 괴리율 {(abs(current_price - target_price_daily) / target_price_daily):.2%}")
            
            # 장 마감 직전 타임컷 매수 (예: 15:25)
            if current_minute_dt.time() >= datetime.datetime.strptime("15:25", "%H:%M").time():
                if self.execute_time_cut_buy(stock_code, current_minute_dt, current_price, target_quantity, self.strategy_params.get('max_time_cut_deviation_ratio', 0.02)):
                    logger.info(f"[실시간 매매] {stock_code} 타임컷 매수 완료 (추세 추종)")

        # 매도 로직 (보유 종목만)
        elif signal_type == 'sell' and stock_code in self.broker.positions:
            current_position_size = self.broker.positions[stock_code]['size']
            if current_position_size <= 0:
                self.reset_signal(stock_code) # 포지션 없으면 신호 초기화
                return

            # 장 시작 10분 후부터 3시 20분까지
            if current_minute_dt.time() >= datetime.datetime.strptime("09:10", "%H:%M").time() and \
               current_minute_dt.time() <= datetime.datetime.strptime("15:20", "%H:%M").time():
                
                # 매도 시점 조건 (예: 현재가가 특정 손절 라인 또는 목표 수익률 도달)
                # 여기서는 단순화를 위해 매도 신호 발생 시 바로 매도 시도
                max_deviation_ratio = self.strategy_params.get('max_deviation_ratio', 0.01) # 기본 1%
                if target_price_daily > 0 and abs(current_price - target_price_daily) / target_price_daily <= max_deviation_ratio:
                    if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_minute_dt):
                        logger.info(f"[실시간 매매] {stock_code} 매도 완료 (추세 추종): {current_position_size}주 @ {current_price:,.0f}원")
                        self.reset_signal(stock_code)
                else:
                    logger.debug(f"[실시간 매매] {stock_code} 매도 대기 (추세 추종): 현재가 {current_price:,.0f}원, 목표가 {target_price_daily:,.0f}원, 괴리율 {(abs(current_price - target_price_daily) / target_price_daily):.2%}")

            # 장 마감 직전 타임컷 매도 (예: 15:25)
            if current_minute_dt.time() >= datetime.datetime.strptime("15:25", "%H:%M").time():
                if self.execute_time_cut_sell(stock_code, current_minute_dt, current_price, current_position_size, self.strategy_params.get('max_time_cut_deviation_ratio', 0.02)):
                    logger.info(f"[실시간 매매] {stock_code} 타임컷 매도 완료 (추세 추종)")