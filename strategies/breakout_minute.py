import datetime
import logging
import pandas as pd
import numpy as np 
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from strategies.strategy import MinuteStrategy 
from util.strategies_util import *

logger = logging.getLogger(__name__)

class BreakoutMinute(MinuteStrategy): 
    """
    일봉 돌파 전략에서 발생한 매수/매도 신호를 받아 분봉에서 실제 거래를 실행하는 전략입니다.
    분봉에서도 추가적인 돌파 조건 또는 시간 기반의 강제 매매 조건을 활용합니다.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self.strategy_name = "BreakoutMinute"
        
        # 전략 파라미터 검증
        self._validate_strategy_params()
        
        # 돌파 분봉 누적 계산을 위한 캐시 추가
        self.last_portfolio_check = None  # 마지막 포트폴리오 체크 시간 (필요시 사용)
        self.last_prices = {}  # 마지막 가격 캐시 추가
        
    # 필수: 파라미터 검증
    def _validate_strategy_params(self):
        """전략 파라미터의 유효성을 검증합니다."""
        required_params = ['minute_breakout_period', 'minute_volume_multiplier']
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"돌파 분봉 전략에 필요한 파라미터 '{param}'이 설정되지 않았습니다.")
        
        if self.strategy_params['minute_breakout_period'] <= 0:
            raise ValueError("분봉 돌파 기간은 0보다 커야 합니다.")
        
        if self.strategy_params['minute_volume_multiplier'] <= 0:
            raise ValueError("분봉 거래량 배수는 0보다 커야 합니다.")

        logging.info(f"돌파 분봉 전략 파라미터 검증 완료: 분봉 돌파 기간={self.strategy_params['minute_breakout_period']}, "
                     f"분봉 거래량 배수={self.strategy_params['minute_volume_multiplier']}")

    # 필수: 부모 클래스에 올릴지 판다 (DailyStrategy에서 전달받은 신호 업데이트)
    def update_signals(self, signals):
        """
        DailyStrategy에서 생성된 신호들을 업데이트합니다.
        """
        # DailyStrategy에서 전달받은 신호에 'traded_today' 플래그 추가
        # 이 플래그는 해당 종목이 당일 매수/매도되었는지 추적하여 중복 거래를 방지합니다.
        self.signals = {
            stock_code: {**info, 'traded_today': False}
            for stock_code, info in signals.items()
        }
        logging.debug(f"BreakoutMinute updated signals: {self.signals.keys()}")


    # 지표 계산 (분봉 최고가 및 거래량 이동평균)
    def _calculate_minute_breakout_high(self, historical_data):
        """
        주어진 분봉 데이터에서 'minute_breakout_period' 동안의 최고가를 계산합니다.
        """
        if len(historical_data) < self.strategy_params['minute_breakout_period'] + 1:
            return None
        
        # 현재 캔들을 제외한 이전 'minute_breakout_period' 동안의 고가
        period_high_data = historical_data['high'].iloc[-(self.strategy_params['minute_breakout_period'] + 1):-1]
        if not period_high_data.empty:
            return period_high_data.max()
        return -1 # 유효하지 않은 값


    def _calculate_minute_volume_ma(self, historical_data):
        """
        주어진 분봉 데이터에서 거래량 이동평균을 계산합니다.
        """
        if len(historical_data) < self.strategy_params['minute_breakout_period']: # volume_ma_period를 공유하거나 별도 정의
            return None
        
        # 여기서는 breakout_period와 동일한 기간을 사용한다고 가정하거나, 별도 파라미터 추가
        return historical_data['volume'].iloc[-self.strategy_params['minute_breakout_period']:].mean()


    # 필수: 반드시 구현
    def run_minute_logic(self, current_dt, stock_code):
        """
        분봉 데이터를 기반으로 돌파 매수/매도 로직을 실행합니다.
        DailyStrategy에서 받은 신호가 있는 종목에 대해서만 동작합니다.
        """
        # DailyStrategy에서 신호가 없거나, 이미 오늘 거래된 종목이면 건너뜀
        if stock_code not in self.signals or self.signals[stock_code].get('traded_today', False):
            return
        
        signal_info = self.signals[stock_code]
        order_signal = signal_info.get('signal') # 'buy' 또는 'sell'

        # 현재 시간의 분봉 데이터 가져오기 (마지막 완성된 봉)
        # _get_bar_at_time은 특정 시점의 완성된 분봉을 가져온다고 가정
        minute_df = self._get_bar_at_time('minute', stock_code, current_dt)
        if minute_df is None or minute_df.empty:
            return
        
        current_price = minute_df['close']
        current_volume = minute_df['volume']
        current_minute_time = current_dt.time()
        
        # 필요한 과거 분봉 데이터 가져오기 (매 분마다 업데이트되는 데이터)
        # 'minute_breakout_period' + 1 보다 더 많은 데이터가 필요할 수 있음 (volume MA 기간 포함)
        lookback_needed = self.strategy_params['minute_breakout_period'] + 1 
        historical_minute_data = self._get_historical_data_up_to(
            'minute',
            stock_code,
            current_dt,
            lookback_period=lookback_needed
        )

        if historical_minute_data.empty or len(historical_minute_data) < lookback_needed:
            logging.debug(f"[{stock_code}] {current_dt.isoformat()} - 충분한 분봉 데이터({len(historical_minute_data)}/{lookback_needed}) 부족.")
            return

        # 손절 처리 (Broker에서 담당)
        current_position_size = self.broker.get_position_size(stock_code)
        if self.broker.stop_loss_params is not None and current_position_size > 0:
            if self.broker.check_and_execute_stop_loss(stock_code, current_price, current_dt):
                self.reset_signal(stock_code) # 손절 발생 시 해당 종목 신호 초기화
                self.signals[stock_code]['traded_today'] = True # 당일 추가 거래 방지
                return # 손절 했으면 더 이상 진행 안 함

        # 매수/매도 공통 로직
        market_open_time = datetime.time(9, 0)
        market_close_time = datetime.time(15, 30) # 장 마감 3시 30분
        
        # 장 초반/후반 시간 제한 (예: 장 시작 5분 이내, 장 마감 10분 전부터는 거래 안 함)
        # 불확실성을 줄이기 위해 매매 가능 시간대를 제한합니다.
        if current_minute_time < (datetime.datetime.combine(current_dt.date(), market_open_time) + datetime.timedelta(minutes=5)).time() or \
           current_minute_time > (datetime.datetime.combine(current_dt.date(), market_close_time) - datetime.timedelta(minutes=10)).time():
            # logging.debug(f"[{stock_code}] {current_dt.isoformat()} - 거래 제한 시간.")
            return

        target_range = 0.3 # 타임컷         
        # --- 매수 로직 ---
        if order_signal == 'buy' and current_position_size == 0:
            target_quantity = signal_info.get('target_quantity', 0)
            if target_quantity <= 0:
                return

            # 분봉 최고가 및 거래량 이동평균 계산
            highest_high_in_minute_period = self._calculate_minute_breakout_high(historical_minute_data)
            minute_volume_ma = self._calculate_minute_volume_ma(historical_minute_data)

            if highest_high_in_minute_period is None or minute_volume_ma is None:
                logging.debug(f"[{stock_code}] {current_dt.isoformat()} - 분봉 지표 계산 불가.")
                return

            # 매수 조건: 분봉에서도 신고가 돌파 + 거래량 조건 충족
            # 현재 종가(current_price)가 이전 minute_breakout_period 동안의 최고가를 돌파
            # 그리고 현재 거래량(current_volume)이 분봉 거래량 MA의 'minute_volume_multiplier' 배 이상
            if current_price > highest_high_in_minute_period and \
               current_volume > minute_volume_ma * self.strategy_params['minute_volume_multiplier']:
                
                logging.info(f'[분봉 돌파 매수] {current_dt.isoformat()} - {stock_code} 가격: {current_price:,.0f}원, '
                             f'분봉신고가({self.strategy_params["minute_breakout_period"]}봉): {highest_high_in_minute_period:,.0f}원, '
                             f'거래량: {current_volume}, 거래량MA: {minute_volume_ma:,.0f}, 수량: {target_quantity}주')
                
                self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
                self.signals[stock_code]['traded_today'] = True # 오늘 매수 완료 플래그 설정
                # self.reset_signal(stock_code) # 매수 후 신호 초기화 (필요시)
            # --- 타임컷 강제 매수 ---
            elif current_minute_time >= datetime.time(15, 18) and current_minute_time < datetime.time(15, 20): # 오후 3:05 ~ 3:15 사이
                # 이 부분은 DailyStrategy에서 전달받은 'target_price'나 'score' 등을 활용하여
                # 여전히 매수 매력도가 높은 경우에만 실행하는 것이 좋습니다.
                # 현재는 단순히 시간이 되면 매수 시도하는 로직이므로,
                # DailyStrategy의 매수 신호가 여전히 유효한지 확인하는 로직 추가 필요
                
                today_data = minute_df[minute_df.index == pd.Timestamp(current_dt)]
                if today_data.empty:
                    return
                
                # 가격 제한 범위
                today_high = today_data['high'].max() 
                today_low = today_data['low'].min()
                today_move = today_data['high'].max() - today_data['low'].min()
                high = today_high - today_move * target_range
                low = today_low + today_move * target_range
                # 목표가
                target_price = self.signals[stock_code].get('target_price', current_price)
                # 가격 제한 범위 내 목표가와 현재가 있어 매수가능
                if low <= target_price <= high and low <= current_price:
                    logging.info(f'[타임컷 강제매도] {current_dt.isoformat()} - {stock_code} 목표가: {target_price}, 매도가: {current_price}, 오늘 저가~고가: {low}~{high}')
                    self.broker.execute_order(stock_code, 'sell', target_price, current_position_size, current_dt)
                    self.reset_signal(stock_code)
                else:
                    logging.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 목표가: {target_price}, 매도가: {current_price}, 오늘 저가~고가: {low}~{high}')
                
                logging.info(f'[분봉 타임컷 강제 매수 시도] {current_dt.isoformat()} - {stock_code} 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
                self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
                self.signals[stock_code]['traded_today'] = True # 오늘 매수 완료 플래그 설정
                # self.reset_signal(stock_code) # 매수 후 신호 초기화 (필요시)
        
        # --- 매도 로직 ---
        elif order_signal == 'sell' and current_position_size > 0:
            
            # 분봉 최고가 및 거래량 이동평균 (매도 조건에도 필요하다면 계산)
            # highest_high_in_minute_period = self._calculate_minute_breakout_high(historical_minute_data)
            # minute_volume_ma = self._calculate_minute_volume_ma(historical_minute_data)

            # 매도 조건: RSI 전략에서처럼 과매수 조건이나, 일봉에서 발생한 매도 신호를 분봉에서 즉시 이행.
            # 여기서는 일봉 신호가 'sell' 이면 분봉에서 특정 조건 없이 또는 추가적인 분봉 조건으로 매도.
            # 예: 가격이 특정 지지선을 하향 돌파하거나, 특정 이평선을 하향 이탈할 때
            
            # 가장 간단한 매도 조건: 일봉 신호가 'sell'이면 최대한 빨리 매도
            # (추가적인 분봉 매도 조건을 여기에 넣을 수 있습니다.)
            
            # --- 타임컷 강제 매도 ---
            # 장 마감 임박 시 보유 종목 강제 매도
            if current_minute_time >= datetime.time(15, 18) and current_minute_time < datetime.time(15, 20): # 장 마감 10분 전부터
                today_data = minute_df[minute_df.index == pd.Timestamp(current_dt)]
                if today_data.empty:
                    return
                
                # 가격 제한 범위
                today_high = today_data['high'].max() 
                today_low = today_data['low'].min()
                today_move = today_data['high'].max() - today_data['low'].min()
                high = today_high - today_move * target_range
                low = today_low + today_move * target_range
                # 목표가
                target_price = self.signals[stock_code].get('target_price', current_price)
                # 가격 제한 범위 내 목표가와 현재가 있어 매수가능
                if low <= target_price <= high and low <= current_price:
                    logging.info(f'[타임컷 강제매도] {current_dt.isoformat()} - {stock_code} 목표가: {target_price}, 매도가: {current_price}, 오늘 저가~고가: {low}~{high}')
                    self.broker.execute_order(stock_code, 'sell', target_price, current_position_size, current_dt)
                    self.reset_signal(stock_code)
                else:
                    logging.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 목표가: {target_price}, 매도가: {current_price}, 오늘 저가~고가: {low}~{high}')
                
