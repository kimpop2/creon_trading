# c:\project\cursor_ai\bt\strategies\rsi_minute_strategy.py

import datetime
import logging
import pandas as pd
import numpy as np 

# MinuteStrategy 추상 클래스 임포트 유지
from strategies.strategy_base import MinuteStrategy 

# Assuming utils.py is in the same directory
# 원본 코드에는 calculate_momentum이 있었으나, RSIMinute에서는 직접 사용하지 않으므로 그대로 유지
from util.utils import calculate_momentum, calculate_rsi 

logger = logging.getLogger(__name__)

class RSIMinute(MinuteStrategy): # MinuteStrategy 상속 유지
    def __init__(self, data_store, strategy_params, broker, position_info):
        # MinuteStrategy 상속에 따라 super().__init__ 호출 유지
        super().__init__(data_store, strategy_params, broker, position_info)

        # 원본 코드의 __init__은 data_store, strategy_params, broker, position_info만 받음
        # 현재 코드에서 추가된 self.rsi_period 등은 원본에 없었으므로 제거 (strategy_params에서 직접 접근)
        # self.rsi_period = strategy_params['minute_rsi_period'] # 제거
        # self.rsi_overbought = strategy_params['minute_rsi_overbought'] # 제거
        # self.rsi_oversold = strategy_params['minute_rsi_oversold'] # 제거
        # self.stop_loss_ratio = strategy_params['stop_loss_ratio'] / 100.0 # 제거
        # self.trailing_stop_ratio = strategy_params['trailing_stop_ratio'] / 100.0 # 제거
        # self.early_stop_loss = strategy_params['early_stop_loss'] / 100.0 # 제거
        # self.max_losing_positions = strategy_params['max_losing_positions'] # 제거
        
        # 원본 코드에 있었던 변수들만 초기화
        self.signals = {} # This will be updated from DualMomentumDaily
        # self.losing_positions_count = 0 # 원본 코드에 없는 변수이므로 제거
        # self.traded_today = {} # 원본 코드에 없는 변수이므로 제거 (signals 내 traded_today 사용)


    def update_signals(self, signals):
        """DualMomentumDaily에서 생성된 모멘텀 신호를 업데이트합니다."""
        self.signals = signals
        # logging.debug(f"RSIMinute: 모멘텀 시그널 업데이트 완료. 시그널 수: {len(self.signals)}") # 원본에 없던 로그 제거

    # 원본 코드의 헬퍼 메서드들을 그대로 복원
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

    def _check_stop_loss(self, stock_code, current_price, position_info, current_dt):
        """개별 종목 손절 조건 체크"""
        avg_price = position_info['avg_price']
        loss_ratio = (current_price - avg_price) / avg_price * 100
        
        # 1. 단순 손절
        if loss_ratio <= self.strategy_params['stop_loss_ratio']:
            logging.info(f"[손절매 발생] {stock_code}: 현재 손실률 {loss_ratio:.2f}%가 기준치 {self.strategy_params['stop_loss_ratio']}%를 초과")
            return True
            
        # 2. 트레일링 스탑
        # 원본 코드의 position_info는 Backtester의 self.position_info(외부 딕셔너리)를 참조
        # self.position_info[stock_code]에 highest_price가 있어야 함
        if stock_code in self.position_info: 
            highest_price = self.position_info[stock_code]['highest_price']
            trailing_loss_ratio = (current_price - highest_price) / highest_price * 100
            if trailing_loss_ratio <= self.strategy_params['trailing_stop_ratio']:
                logging.info(f"[트레일링 스탑] {stock_code}: 현재가 {current_price:,.0f}원이 최고가 {highest_price:,.0f}원 대비 {trailing_loss_ratio:.2f}% 하락")
                return True
        
        # 3. 보유 기간 기반 손절 폭 조정
        # position_info에는 'entry_date'가 있어야 함 (Backtester에서 설정)
        holding_days = (current_dt.date() - position_info['entry_date']).days
        if holding_days <= 5:   # 매수 후 5일 이내
            if loss_ratio <= self.strategy_params['early_stop_loss']:
                logging.info(f"[조기 손절매] {stock_code}: 매수 후 {holding_days}일 이내 손실률 {loss_ratio:.2f}%가 조기 손절 기준 {self.strategy_params['early_stop_loss']}% 초과")
                return True
        
        return False

    def _check_portfolio_stop_loss(self, current_prices):
        """포트폴리오 전체 손절 조건 체크"""
        # 1. 전체 손실폭 기준
        portfolio_value = self.broker.get_portfolio_value(current_prices)
        total_loss_ratio = (portfolio_value - self.strategy_params['initial_cash']) / self.strategy_params['initial_cash'] * 100
        
        if total_loss_ratio <= self.strategy_params['portfolio_stop_loss']:
            logging.info(f"[포트폴리오 손절매] 전체 손실률 {total_loss_ratio:.2f}%가 기준치 {self.strategy_params['portfolio_stop_loss']}% 초과")
            return True
            
        # 2. 동시다발적 손실 기준
        losing_positions = 0
        for stock_code, pos_info in self.broker.positions.items():
            if stock_code in current_prices:
                loss_ratio = (current_prices[stock_code] - pos_info['avg_price']) / pos_info['avg_price'] * 100
                if loss_ratio <= self.strategy_params['stop_loss_ratio']: # 개별 손절 기준과 동일하게 적용
                    losing_positions += 1
        
        if losing_positions >= self.strategy_params['max_losing_positions']:
            logging.info(f"[포트폴리오 손절매] 동시 손실 종목 수 {losing_positions}개가 최대 허용치 {self.strategy_params['max_losing_positions']}개 초과")
            return True
        
        return False

    def _update_position_info(self, stock_code, current_price):
        """포지션 정보 업데이트 (고점 등)"""
        # self.position_info는 백테스터로부터 주입받은 외부 딕셔너리
        if stock_code not in self.position_info:
            self.position_info[stock_code] = {
                'highest_price': current_price
            }
        else:
            if current_price > self.position_info[stock_code]['highest_price']:
                self.position_info[stock_code]['highest_price'] = current_price

    def _get_last_price(self, stock_code):
        """종목의 마지막 거래 가격을 반환합니다."""
        daily_df = self.data_store['daily'].get(stock_code)
        if daily_df is not None and not daily_df.empty:
            return daily_df['close'].iloc[-1]
        return None

    def run_minute_logic(self, stock_code, current_dt):
        """분봉 데이터를 기반으로 실제 매수/매도 주문을 실행합니다."""
        current_minute_date = current_dt.date()
        if stock_code not in self.data_store['minute'] or current_minute_date not in self.data_store['minute'][stock_code]:
            # logging.debug(f"[{current_dt.isoformat()}] {stock_code}: 해당 날짜의 분봉 데이터 없음. 건너뜜.")
            return

        current_minute_bar = self._get_bar_at_time('minute', stock_code, current_dt)
        if current_minute_bar is None:
            # logging.debug(f"[{current_dt.isoformat()}] {stock_code}: 해당 시간의 분봉 바 데이터 없음. 건너뜜.")
            return

        current_minute_time = current_dt.time()
        current_price = current_minute_bar['close']

        # 포지션 정보 업데이트 (트레일링 스탑을 위해)
        if stock_code in self.broker.positions and self.broker.positions[stock_code]['size'] > 0:
            self._update_position_info(stock_code, current_price)

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


        # --- 손절 로직 (매수/매도 신호와 관계없이 최우선으로 체크) ---
        position_info = self.broker.positions.get(stock_code)
        if position_info and position_info['size'] > 0: # 현재 종목을 보유하고 있는 경우
            # 개별 종목 손절 체크
            # 원본 코드의 position_info에는 'entry_date'가 있어야 함. (Backtester에서 채워줘야 함)
            if self._check_stop_loss(stock_code, current_price, position_info, current_dt):
                logging.info(f'[손절매 실행] {current_dt.isoformat()} - {stock_code} 매도. 가격: {current_price:,.0f}원 (개별 손절)')
                self.broker.execute_order(stock_code, 'sell', current_price, position_info['size'], current_dt)
                if stock_code in self.position_info:
                    del self.position_info[stock_code]
                # 원본 코드에 'traded_today' 플래그를 signals 딕셔너리 내에 설정하는 로직 유지
                self.signals[stock_code]['traded_today'] = True # 손절했으므로 당일 추가 거래 방지
                return
        
        # 포트폴리오 전체 손절 체크 (보유 종목이 하나라도 있을 때만)
        if self.broker.positions and self._check_portfolio_stop_loss(current_prices_for_portfolio_check):
            logging.info(f'[포트폴리오 손절매 실행] {current_dt.isoformat()} - 전체 포트폴리오 손절 조건 충족. 모든 포지션 청산.')
            for code, pos in list(self.broker.positions.items()): # 리스트로 복사하여 순회 중 딕셔너리 변경 허용
                if pos['size'] > 0:
                    price = current_prices_for_portfolio_check.get(code, self._get_last_price(code))
                    if price is None: # 혹시 모를 경우를 대비
                        price = pos['avg_price'] # 평균 단가로라도 매도 시도
                        logging.warning(f"{current_dt.isoformat()} - {code} 현재가 정보 없음. 평균단가로 매도 시도: {price:,.0f}원")
                    
                    logging.info(f'[포트폴리오 손절매 실행] {current_dt.isoformat()} - {code} 매도. 가격: {price:,.0f}원')
                    self.broker.execute_order(code, 'sell', price, pos['size'], current_dt)
                    if code in self.position_info:
                        del self.position_info[code]
                    self.signals[code]['traded_today'] = True # 청산했으므로 당일 추가 거래 방지
            return # 포트폴리오 전체 손절이 발생하면 더 이상 다른 매매 로직 실행하지 않음


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
            
        # 당일 이미 거래가 발생했으면 추가 거래 방지
        # 원본 코드의 이 부분은 signals 딕셔너리 내에 traded_today 플래그를 가정
        if self.signals[stock_code]['traded_today']:
            return
            
        current_position_size = self.broker.get_position_size(stock_code)
        
        required_rsi_data_len = self.strategy_params['minute_rsi_period'] + 1
        minute_historical_data = self._get_historical_data_up_to('minute', stock_code, current_dt, lookback_period=required_rsi_data_len)
        
        if len(minute_historical_data) < required_rsi_data_len:
            # logging.debug(f"[{current_dt.isoformat()}] {stock_code}: RSI 계산을 위한 분봉 데이터 부족. ({len(minute_historical_data)}/{required_rsi_data_len})")
            return

        current_rsi_value_series = calculate_rsi(minute_historical_data, self.strategy_params['minute_rsi_period'])
        current_rsi_value = current_rsi_value_series.iloc[-1]
        
        if pd.isna(current_rsi_value):
            # logging.debug(f"[{current_dt.isoformat()}] {stock_code}: RSI 값 계산 불가 (NaN).")
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
                    # 매수 후 포지션 최고가 초기화 및 entry_date 기록
                    # self.position_info는 Backtester가 관리하는 외부 딕셔너리이므로 여기에 반영
                    self.position_info[stock_code] = {'highest_price': current_price, 'entry_date': current_dt.date()} 


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
                    if stock_code in self.position_info:
                        del self.position_info[stock_code] # 포지션 청산 시 최고가 정보도 삭제

    # MinuteStrategy 상속에 따라 run_daily_logic 메서드 유지
    def run_daily_logic(self, current_daily_date):
        """RSIMinute는 일봉 로직을 직접 수행하지 않습니다."""
        pass