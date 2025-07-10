# strategy/breakout_strategy.py
from datetime import date, datetime, timedelta
import pandas as pd
from strategies.strategy import Strategy
from util.strategies_util import *

class BreakoutStrategy(Strategy):
    def __init__(self, broker, manager, data_store, strategy_params):
        super().__init__(broker, manager, data_store, strategy_params)
        self.range_ratio = self.strategy_params.get('range_ratio', 0.5) # 전일 변동폭 대비 비율
        self.lookback_days = self.strategy_params.get('lookback_days', 1) # 전일 데이터 사용
        self.strategy_name = "BreakoutStrategy"

    def run_strategy_logic(self, current_date: date) -> None:
        logger.info(f"[{current_date}] 변동성 돌파 전략 (일봉) 로직 실행 시작")
        # 변동성 돌파 전략은 주로 분봉에서 실시간으로 진입/청산 판단을 하므로,
        # 일봉에서는 매수/매도 후보군을 선정하거나, 단순히 신호를 초기화하고 넘어갈 수 있습니다.
        # 여기서는 다음 날 분봉 로직을 위해 매수/매도 신호를 생성하지 않고, 단순히 관심 종목을 유지합니다.
        
        # 보유 종목 중 매도 신호가 필요한 경우(예: N일 보유 후 자동 매도 등) 처리
        sell_candidates = set()
        for stock_code in self.broker.positions.keys():
            # 예시: 5일 보유 후 매도 (간단한 매도 조건)
            # 이 부분은 전략의 특성에 맞게 구현해야 합니다.
            position_date = self.broker.positions[stock_code]['buy_date'].date()
            if (current_date - position_date).days >= 5:
                sell_candidates.add(stock_code)
            pass

        # 모든 종목을 일단 후보군에 포함시켜 분봉 로직에서 검토하도록 합니다.
        all_stocks = [(stock_code, 0) for stock_code in self.data_store['daily'].keys()]
        
        self._reset_all_signals() # 이전 날짜의 신호 초기화
        # 여기서는 명시적인 buy_candidates를 만들지 않고, run_trading_logic에서 바로 진입 판단
        self._generate_signals(current_date, set(), all_stocks, sell_candidates)
        
        # 현재 신호 상태를 로깅 (주로 홀딩 또는 매도 신호만 있을 것임)
        # self._log_rebalancing_summary(current_date, set(), set(self.broker.positions.keys()), sell_candidates)
        logger.info(f"[{current_date}] 변동성 돌파 전략 (일봉) 로직 실행 완료 (분봉 로직에 집중)")


    def run_trading_logic(self, current_minute_dt: datetime, stock_code: str) -> None:
        signal_info = self.signals.get(stock_code)
        # 이미 오늘 매매했거나 신호가 없으면 스킵
        if not signal_info or signal_info.get('traded_today'):
            return

        # 매수/매도 신호는 분봉에서 실시간으로 생성
        # 변동성 돌파 전략은 당일 시가를 기준으로 하므로, 당일 첫 분봉 데이터 이후부터 검토 시작
        if current_minute_dt.time() < datetime.strptime("09:00", "%H:%M").time():
            return # 9시 이전에는 매매 로직 실행 안 함

        # 전일 일봉 데이터 가져오기 (변동폭 계산용)
        # current_minute_dt의 날짜에서 하루 전으로 설정
        prev_day = current_minute_dt.date() - timedelta(days=1)
        # 주말/공휴일 건너뛰고 가장 가까운 거래일의 데이터 가져와야 함 (TradingManager에 해당 기능 필요)
        # 현재는 단순하게 전일 날짜를 사용한다고 가정
        daily_data_prev_day = self._get_historical_data_up_to('daily', stock_code, prev_day, lookback_period=1)

        if daily_data_prev_day.empty:
            logger.debug(f"[{current_minute_dt}] {stock_code}: 전일 일봉 데이터 부족으로 변동성 돌파 전략 건너뜀.")
            return

        prev_high = daily_data_prev_day['high'].iloc[-1]
        prev_low = daily_data_prev_day['low'].iloc[-1]
        prev_close = daily_data_prev_day['close'].iloc[-1]

        if prev_high == 0 or prev_low == 0:
            logger.warning(f"[{current_minute_dt}] {stock_code}: 전일 고가 또는 저가 데이터가 유효하지 않습니다.")
            return

        range_daily = prev_high - prev_low
        breakout_price_buy = current_bar['open'] + range_daily * self.range_ratio # 당일 시가 + (전일 변동폭 * 비율)
        breakout_price_sell = current_bar['open'] - range_daily * self.range_ratio # 당일 시가 - (전일 변동폭 * 비율) (공매도 또는 하락 돌파 매도)

        # 현재 분봉 데이터 가져오기
        # 필요한 분봉 데이터를 lookback_period만큼 가져옵니다.
        # 변동성 돌파는 주로 'open'을 기준으로 'high' 또는 'close'가 돌파하는 시점을 봅니다.
        historical_minute_data = self._get_historical_data_up_to('minute', stock_code, current_minute_dt, lookback_period=1)
        if historical_minute_data.empty:
            return

        current_bar = historical_minute_data.iloc[-1]
        current_price = current_bar['close']
        current_open = current_bar['open']
        current_high = current_bar['high']

        # 매수 로직
        if stock_code not in self.broker.positions: # 미보유 종목만 매수 시도
            if current_high >= breakout_price_buy: # 당일 고가가 돌파 가격을 넘었을 때
                target_quantity = self._calculate_target_quantity(stock_code, current_price)
                if target_quantity > 0:
                    if self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_minute_dt):
                        logger.info(f"[실시간 매매] {stock_code} 매수 완료 (변동성 돌파): {target_quantity}주 @ {current_price:,.0f}원")
                        # 매수 성공 시, 해당 종목의 신호를 'traded_today'로 표시하여 오늘 더 이상 매매하지 않도록 합니다.
                        if stock_code in self.signals:
                            self.signals[stock_code]['traded_today'] = True
                        # 또는 특정 조건 충족 시 reset_signal 호출
                        # self.reset_signal(stock_code)
        
        # 매도 로직 (보유 종목만)
        if stock_code in self.broker.positions:
            current_position_size = self.broker.positions[stock_code]['size']
            buy_price = self.broker.positions[stock_code]['avg_buy_price'] # 평균 매수 단가

            # 손절매 로직 (예: 매수 가격 대비 -3% 하락 시)
            stop_loss_ratio = self.strategy_params.get('stop_loss_ratio', 0.03)
            if current_price < buy_price * (1 - stop_loss_ratio):
                if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_minute_dt):
                    logger.warning(f"[실시간 매매] {stock_code} 손절매 완료 (변동성 돌파): {current_position_size}주 @ {current_price:,.0f}원 (매수가 {buy_price:,.0f}원)")
                    self.reset_signal(stock_code)

            # 목표 수익률 달성 시 매도 (예: 매수 가격 대비 +5% 상승 시)
            take_profit_ratio = self.strategy_params.get('take_profit_ratio', 0.05)
            if current_price > buy_price * (1 + take_profit_ratio):
                if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_minute_dt):
                    logger.info(f"[실시간 매매] {stock_code} 익절 완료 (변동성 돌파): {current_position_size}주 @ {current_price:,.0f}원 (매수가 {buy_price:,.0f}원)")
                    self.reset_signal(stock_code)
            
            # 하락 돌파 매도 (옵션: 추세 전환 또는 빠른 익절/손절)
            # if current_price < breakout_price_sell:
            #     if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_minute_dt):
            #         logger.info(f"[실시간 매매] {stock_code} 하락 돌파 매도 완료 (변동성 돌파): {current_position_size}주 @ {current_price:,.0f}원")
            #         self.reset_signal(stock_code)

        # 장 마감 직전 타임컷 매도 (보유 포지션 정리)
        if current_minute_dt.time() >= datetime.strptime("15:25", "%H:%M").time():
            if stock_code in self.broker.positions:
                current_position_size = self.broker.positions[stock_code]['size']
                if current_position_size > 0:
                    if self.execute_time_cut_sell(stock_code, current_minute_dt, current_price, current_position_size, self.strategy_params.get('max_time_cut_deviation_ratio', 0.02)):
                        logger.info(f"[실시간 매매] {stock_code} 타임컷 매도 완료 (변동성 돌파)")