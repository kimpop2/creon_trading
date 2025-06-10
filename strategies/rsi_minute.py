import datetime
import logging
import pandas as pd
import numpy as np

from util.utils import calculate_rsi # RSI 계산 함수 임포트
from strategies.strategy_base import MinuteStrategy # MinuteStrategy 추상 클래스 임포트

class RSIMinute(MinuteStrategy): # MinuteStrategy 상속
    def __init__(self, data_store, strategy_params, broker, position_info):
        super().__init__(data_store, strategy_params, broker, position_info)

        self.rsi_period = strategy_params['minute_rsi_period']
        self.rsi_overbought = strategy_params['minute_rsi_overbought']
        self.rsi_oversold = strategy_params['minute_rsi_oversold']
        self.stop_loss_ratio = strategy_params['stop_loss_ratio'] / 100.0
        self.trailing_stop_ratio = strategy_params['trailing_stop_ratio'] / 100.0
        self.early_stop_loss = strategy_params['early_stop_loss'] / 100.0
        self.max_losing_positions = strategy_params['max_losing_positions']
        # initial_cash는 strategy_params에서 바로 접근 가능하며,
        # buy_amount_per_stock 계산에 사용되므로 self.initial_cash로 별도 저장할 필요 없음 (원본과 동일)

        # 원본 코드의 매수 금액 계산 로직 복원
        # self.initial_cash = initial_cash  # Backtester로부터 받은 초기 현금 (이제 strategy_params에 포함)
        # num_top_stocks는 DailyStrategy의 파라미터이지만, MinuteStrategy의 매수 금액 계산에 사용됨
        # (원본 코드의 의도를 따름)
        self.buy_amount_per_stock = strategy_params['initial_cash'] / strategy_params['num_top_stocks']
        
        self.momentum_signals = {}
        self.losing_positions_count = 0

        # 매매 시그널 처리 여부 플래그 (이전 코드에 있던 traded_today를 여기로 옮김)
        self.traded_today = {} # {stock_code: bool}


    def update_momentum_signals(self, momentum_signals: dict):
        """
        일봉 전략으로부터 생성된 모멘텀 시그널을 업데이트합니다.
        """
        # 새롭게 매수/매도 시그널이 발생한 종목만 traded_today를 False로 초기화
        for stock_code, info in momentum_signals.items():
            # 백테스트 날짜와 시그널 날짜가 같은 경우 (실제 구동 시에는 오늘 날짜)
            if info['signal'] in ['buy', 'sell'] and info['signal_date'] == info['signal_date']: # 조건이 항상 참이므로 백테스트에서는 효과없음.
                # 실제 구동에서는 datetime.date.today()와 info['signal_date']를 비교
                # 백테스트에서는 backtester에서 current_daily_date와 signal_date를 비교하여 분봉 데이터를 로드하므로,
                # 이 플래그는 run_minute_logic 내부에서 관리될 것.
                pass # 이 부분은 백테스터 로직이 책임지므로 여기서는 별도 초기화 불필요.
        
        self.momentum_signals = momentum_signals
        # logging.debug(f"RSIMinute: 모멘텀 시그널 업데이트 완료. 시그널 수: {len(self.momentum_signals)}")


    def run_minute_logic(self, stock_code: str, current_minute_dt: datetime.datetime):
        """
        분봉 데이터 및 시장 상황을 기반으로 매매 시그널을 생성합니다.
        """
        # 해당 종목의 최신 분봉 데이터 가져오기
        current_daily_date = current_minute_dt.date()
        if stock_code not in self.data_store['minute'] or \
           current_daily_date not in self.data_store['minute'][stock_code]:
            logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: 해당 날짜의 분봉 데이터가 없습니다. 분봉 로직을 건너뜀.")
            return

        minute_df = self.data_store['minute'][stock_code][current_daily_date]
        
        # 현재 시간까지의 데이터만 필터링
        past_minute_data = minute_df.loc[minute_df.index <= current_minute_dt]
        
        if past_minute_data.empty:
            logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: 현재 시간까지의 분봉 데이터가 없습니다. 분봉 로직을 건너뜀.")
            return

        current_price = past_minute_data['close'].iloc[-1]
        
        # 포지션 관리 정보 업데이트 (현재 가격 업데이트)
        if stock_code in self.position_info:
            self.position_info[stock_code]['highest_price'] = max(self.position_info[stock_code].get('highest_price', current_price), current_price)

        # 1. 매매 시그널 확인 (듀얼 모멘텀 전략으로부터의 시그널)
        signal_info = self.momentum_signals.get(stock_code)
        if not signal_info:
            # logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: 듀얼 모멘텀 시그널 없음.")
            pass # 시그널이 없어도 손절/트레일링 스탑은 계속 검사해야 함

        # 2. RSI 계산
        if len(past_minute_data) >= self.rsi_period:
            current_rsi = calculate_rsi(past_minute_data, self.rsi_period).iloc[-1]
        else:
            current_rsi = 50 # 데이터 부족 시 중립 값
            # logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: RSI 계산을 위한 충분한 분봉 데이터 부족. (현재: {len(past_minute_data)}행, 필요: {self.rsi_period}행)")

        # 3. 매매 로직 (매도 > 매수 우선순위)
        # 3-1. 손절매 (일반 손절, 트레일링 스탑, 초기 손절)
        if self.broker.get_position_size(stock_code) > 0: # 포지션이 있을 경우만 검사
            entry_price = self.broker.positions[stock_code]['avg_price']
            entry_date = self.position_info[stock_code].get('entry_date')
            highest_price = self.position_info[stock_code].get('highest_price', entry_price)

            loss_ratio = ((current_price - entry_price) / entry_price) * 100
            trailing_loss_ratio = ((current_price - highest_price) / highest_price) * 100 if highest_price > 0 else 0

            # 초기 손절 (매수 후 5거래일 이내)
            if entry_date and (current_minute_dt.date() - entry_date).days <= 5: # 5거래일 이내
                if loss_ratio <= self.early_stop_loss:
                    if self.broker.execute_order(stock_code, 'sell', current_price, self.broker.get_position_size(stock_code), current_minute_dt):
                        logging.warning(f"[{current_minute_dt.isoformat()}] {stock_code}: 초기 손절 ({loss_ratio:.2f}%) 실행. 매수가: {entry_price:,.0f}, 현재가: {current_price:,.0f}")
                        del self.position_info[stock_code]
                        if stock_code in self.momentum_signals: self.momentum_signals[stock_code]['traded_today'] = True # 백테스터에서 관리하므로 실제로는 불필요
                        return

            # 일반 손절
            if loss_ratio <= self.stop_loss_ratio:
                if self.broker.execute_order(stock_code, 'sell', current_price, self.broker.get_position_size(stock_code), current_minute_dt):
                    logging.warning(f"[{current_minute_dt.isoformat()}] {stock_code}: 일반 손절 ({loss_ratio:.2f}%) 실행. 매수가: {entry_price:,.0f}, 현재가: {current_price:,.0f}")
                    del self.position_info[stock_code]
                    if stock_code in self.momentum_signals: self.momentum_signals[stock_code]['traded_today'] = True # 백테스터에서 관리하므로 실제로는 불필요
                    return
            
            # 트레일링 스탑
            if trailing_loss_ratio <= self.trailing_stop_ratio and highest_price > entry_price: # 수익 상태에서 고점 대비 하락 시
                if self.broker.execute_order(stock_code, 'sell', current_price, self.broker.get_position_size(stock_code), current_minute_dt):
                    logging.warning(f"[{current_minute_dt.isoformat()}] {stock_code}: 트레일링 스탑 ({trailing_loss_ratio:.2f}%) 실행. 최고가: {highest_price:,.0f}, 현재가: {current_price:,.0f}")
                    del self.position_info[stock_code]
                    if stock_code in self.momentum_signals: self.momentum_signals[stock_code]['traded_today'] = True # 백테스터에서 관리하므로 실제로는 불필요
                    return

            # RSI 과매수 매도 시그널 (듀얼 모멘텀 매도 시그널과 결합)
            if current_rsi >= self.rsi_overbought:
                if signal_info and signal_info['signal'] == 'sell':
                    if self.broker.execute_order(stock_code, 'sell', current_price, self.broker.get_position_size(stock_code), current_minute_dt):
                        logging.info(f"[{current_minute_dt.isoformat()}] {stock_code}: RSI 과매수 ({current_rsi:.2f}) & 듀얼 모멘텀 매도 시그널로 매도 완료. 가격: {current_price:,.0f}원")
                        del self.position_info[stock_code]
                        if stock_code in self.momentum_signals: self.momentum_signals[stock_code]['traded_today'] = True # 백테스터에서 관리하므로 실제로는 불필요
                        return # 매도 완료 시 다음 로직 건너뜀

        # 3-2. 매수 (듀얼 모멘텀 매수 시그널 & RSI 과매도)
        # 당일 매매를 안 했고 (traded_today), 듀얼 모멘텀 매수 시그널이 있으며, 현재 포지션이 없고, RSI가 과매도 상태일 때
        # ( traded_today 플래그는 Backtester에서 관리하는 momentum_signals의 'traded_today'를 사용)
        if stock_code in self.momentum_signals and \
           self.momentum_signals[stock_code]['signal'] == 'buy' and \
           not self.momentum_signals[stock_code]['traded_today']: # 당일 이미 매매한 경우 제외
            
            if self.broker.get_position_size(stock_code) == 0: # 현재 포지션이 없어야 매수
                if current_rsi < self.rsi_oversold: # RSI가 과매도 구간 진입
                    buy_price = current_price # 현재가로 매수
                    
                    # 원본 코드의 buy_size 계산 로직 복원
                    buy_size = int(self.buy_amount_per_stock // buy_price)
                    
                    if buy_size > 0:
                        # 매수 주문 실행
                        if self.broker.execute_order(stock_code, 'buy', buy_price, buy_size, current_minute_dt):
                            self.momentum_signals[stock_code]['traded_today'] = True # 당일 매매 완료 플래그
                            # 포지션 정보 저장 (최고가, 매수일 등)
                            self.position_info[stock_code] = {
                                'highest_price': current_price,
                                'entry_date': current_minute_dt.date() # 매수 일자 기록
                            }
                            logging.info(f"[{current_minute_dt.isoformat()}] {stock_code}: RSI 과매도 ({current_rsi:.2f})로 매수 완료. 가격: {current_price:,.0f}원, 수량: {buy_size:,.0f}주")
                        else:
                            logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: 주문 실행 실패 (아마도 현금 부족).")
                    else:
                        logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: 매수 가능 수량 부족. (현금: {self.broker.cash:,.0f}원, 현재가: {current_price:,.0f}원)")
            else:
                logging.debug(f"[{current_minute_dt.isoformat()}] {stock_code}: 이미 보유 중이므로 매수 시도 안함.")

    def run_daily_logic(self, current_daily_date: datetime.date):
        """RSIMinute는 일봉 로직을 직접 수행하지 않습니다."""
        pass