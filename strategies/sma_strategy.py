# strategies/sma_daily.py
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from strategies.strategy import Strategy
from util.strategies_util import *

logger = logging.getLogger(__name__)

class SMAStrategy(Strategy):
    """
    SMA(Simple Moving Average) 기반 일봉 전략입니다.
    골든 크로스/데드 크로스와 거래량 조건을 활용하여 매매 신호를 생성합니다.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        # DailyStrategy 에서 trade의 broker, data_store 연결, signal 초기화 진행
        super().__init__(broker, data_store, strategy_params)
        self.broker = broker
        self.data_store = data_store
        self.strategy_name = "SMAStrategy"
        # SMA 누적 계산을 위한 캐시 추가
        self.sma_cache = {}  # SMA 캐시
        self.volume_cache = {}  # 거래량 MA 캐시
        self.last_prices = {}  # 마지막 가격 캐시
        self.last_volumes = {}  # 마지막 거래량 캐시
        self.strategy_params = strategy_params
        self._validate_strategy_params() # 전략 파라미터 검증

        
    def _validate_strategy_params(self):
        """전략 파라미터의 유효성을 검증합니다."""
        required_params = ['short_sma_period', 'long_sma_period', 'volume_ma_period', 
                           'minute_rsi_period', 'minute_rsi_oversold', 'minute_rsi_overbought', 
                           'num_top_stocks']
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"SMA 전략에 필요한 파라미터 '{param}'이 설정되지 않았습니다.")
        
        # SMA 기간 검증
        if self.strategy_params['short_sma_period'] >= self.strategy_params['long_sma_period']:
            raise ValueError("단기 SMA 기간은 장기 SMA 기간보다 짧아야 합니다.")
            
        logging.info(f"SMA 전략 파라미터 검증 완료: "
                   f"단기 SMA={self.strategy_params['short_sma_period']}, "
                   f"장기 SMA={self.strategy_params['long_sma_period']}, "
                   f"거래량 MA={self.strategy_params['volume_ma_period']}, "
                   f"RSI 분봉 기간={self.strategy_params['minute_rsi_period']}, "
                   f"과매수 점수={self.strategy_params['minute_rsi_oversold']}, "
                   f"과매도 점수={self.strategy_params['minute_rsi_overbought']}, "
                   f"선택종목 수={self.strategy_params['num_top_stocks']}")
        
    def _calculate_rsi(self, historical_data, stock_code):
        """
        RSI를 계산하고 캐시합니다.
        캐시된 데이터를 활용하여 효율적으로 계산합니다.
        """
        if len(historical_data) < self.strategy_params['minute_rsi_period'] + 1:
            logger.debug(f"[{stock_code}] RSI 계산을 위한 데이터 부족: {len(historical_data)}개, 최소 {self.strategy_params['minute_rsi_period'] + 1}개 필요.")
            return None

        # 캐시된 RSI 값이 있고, 새로운 데이터가 1개만 추가된 경우
        if stock_code in self.last_rsi_values and len(historical_data) == len(self.rsi_cache.get(stock_code, [])) + 1:
            last_rsi = self.last_rsi_values[stock_code]
            last_prices = self.rsi_cache[stock_code]
            new_price = historical_data['close'].iloc[-1]
            
            # RSI 업데이트 계산
            price_change = new_price - last_prices[-1]
            if price_change > 0:
                gain = price_change
                loss = 0
            else:
                gain = 0
                loss = -price_change
                
            # 이동평균 업데이트
            avg_gain = (last_rsi['avg_gain'] * (self.strategy_params['minute_rsi_period'] - 1) + gain) / self.strategy_params['minute_rsi_period']
            avg_loss = (last_rsi['avg_loss'] * (self.strategy_params['minute_rsi_period'] - 1) + loss) / self.strategy_params['minute_rsi_period']
            
            # RSI 계산
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # 캐시 업데이트
            self.last_rsi_values[stock_code] = {'avg_gain': avg_gain, 'avg_loss': avg_loss}
            self.rsi_cache[stock_code] = historical_data['close'].values
            self.last_prices[stock_code] = new_price

            logger.debug(f"[{stock_code}] RSI 업데이트 계산: {rsi:.2f}")
            return rsi
        
        # 전체 RSI 계산 (캐시가 없는 경우)
        delta = historical_data['close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gain = gains.rolling(window=self.strategy_params['minute_rsi_period'], min_periods=1).mean().iloc[-1]
        avg_loss = losses.rolling(window=self.strategy_params['minute_rsi_period'], min_periods=1).mean().iloc[-1]
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # 캐시 업데이트
        self.last_rsi_values[stock_code] = {'avg_gain': avg_gain, 'avg_loss': avg_loss}
        self.rsi_cache[stock_code] = historical_data['close'].values
        self.last_prices[stock_code] = historical_data['close'].iloc[-1]
        
        return rsi
    
    # 필수 : 반드시 구현
    def run_strategy_logic(self, current_date: datetime) -> None:
        """ 
        분봉 데이터를 기반으로 매매 신호 발생 시키는 전략 로직 
        """
        # 1. SMA 신호 점수 계산 (전일 데이터까지만 사용)
        buy_scores = {}  # 매수 점수
        sell_scores = {}  # 매도 점수 (새로 추가)
        all_stock_codes = list(self.data_store['daily'].keys())
        
        for stock_code in all_stock_codes:
            historical_data = self._get_historical_data_up_to('daily', stock_code, current_date, lookback_period=max(self.strategy_params['long_sma_period'], self.strategy_params['volume_ma_period']) + 1)
            if historical_data.empty or len(historical_data) < max(self.strategy_params['long_sma_period'], self.strategy_params['volume_ma_period']) + 1:
                continue
                
            short_sma = calculate_sma_incremental(historical_data, self.strategy_params['short_sma_period'], self.sma_cache)[0]
            long_sma = calculate_sma_incremental(historical_data, self.strategy_params['long_sma_period'], self.sma_cache)[0]
            prev_short_sma = calculate_sma_incremental(historical_data.iloc[:-1], self.strategy_params['short_sma_period'], self.sma_cache)[0]
            prev_long_sma = calculate_sma_incremental(historical_data.iloc[:-1], self.strategy_params['long_sma_period'], self.sma_cache)[0]
            current_volume = historical_data['volume'].iloc[-1]
            volume_ma = calculate_volume_ma_incremental(historical_data, self.strategy_params['volume_ma_period'], self.volume_cache)[0]
            
            # 골든크로스 + 거래량 조건 완화 (1.0배 이상)
            if short_sma > long_sma and prev_short_sma <= prev_long_sma and current_volume > volume_ma * 1.0:
                score = (short_sma - long_sma) / long_sma * 100
                buy_scores[stock_code] = score
            # 추가 매수 조건(강한 상승 완화)
            elif short_sma > long_sma and current_volume > volume_ma * 1.2:
                score = (short_sma - long_sma) / long_sma * 50
                buy_scores[stock_code] = score
            
            # 데드크로스 + 거래량 조건이 모두 충족될 때만 매도 신호
            if short_sma < long_sma and prev_short_sma >= prev_long_sma and current_volume > volume_ma * 1.0:
                score = (long_sma - short_sma) / long_sma * 100
                sell_scores[stock_code] = score
            # 강한 하락(추가 매도)은 제외(신호 완화)

        # 2. 매수 후보 종목 선정
        sorted_buy_stocks = sorted(buy_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = set()
        for rank, (stock_code, _) in enumerate(sorted_buy_stocks, 1):
            if rank <= self.strategy_params['num_top_stocks']:
                buy_candidates.add(stock_code)
                
        # 매도 후보 종목 선정 (보유 중인데 매수 후보에 없는 종목은 일정 기간(3일) 이상 홀딩 후에만 매도 후보로 추가)
        current_positions = set(self.broker.positions.keys())
        sell_candidates = set()
        min_holding_days = self.strategy_params.get('min_holding_days', 3)
        for stock_code in current_positions:
            # 데드크로스+거래량 조건이 충족된 경우
            if stock_code in sell_scores:
                sell_candidates.add(stock_code)
                logging.info(f"데드크로스+거래량 매도 후보 추가: {stock_code}")
            # 매수 후보에서 빠진 종목은 일정 기간 홀딩 후 매도 후보
            elif stock_code not in buy_candidates:
                position_info = self.broker.positions.get(stock_code, {})
                entry_date = position_info.get('entry_date', current_date)
                holding_days = (current_date - entry_date).days if entry_date else 0
                if holding_days >= min_holding_days:
                    sell_candidates.add(stock_code)
                    logging.info(f"매수 후보 제외+홀딩기간 경과로 매도 후보 추가: {stock_code}")

        logging.info(f"매도 후보 최종 선정: {sorted(sell_candidates)}")
        
        # 3. 신호 생성 및 업데이트 (부모 메서드) - 전일 데이터 기준
        # 수정: buy_candidates와 sell_candidates를 모두 전달
        final_positions = self._generate_signals(current_date, buy_candidates, sorted_buy_stocks, sell_candidates)
        
        # 4. 리밸런싱 요약 로그
        self._log_rebalancing_summary(current_date, buy_candidates, final_positions, sell_candidates)

    # 필수 : 반드시 구현
    def run_trading_logic(self, current_dt: datetime, stock_code: str) -> None:
        """
        매매 신호 발생 여부를 확인하고, 분봉데이터를 기반으로 필요시 매매 주문을 실행합니다.
        """
        if stock_code not in self.signals:
            logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 시그널 없음. 매매 건너뜀.")
            return
        
        signal_info = self.signals[stock_code]
        order_signal = signal_info.get('signal')
        
        # 당일 이미 거래가 발생했으면 추가 거래 방지
        if signal_info.get('traded_today', False):
            logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 이미 오늘 거래 완료 (traded_today=True). 매매 건너뜀.")
            return

        logger.debug(f"[{current_dt.isoformat()}] {stock_code}: 시그널 정보: {signal_info}")
 
        # 현재 시간의 분봉 데이터 가져오기 (한 번만 조회)
        minute_df = self._get_bar_at_time('minute', stock_code, current_dt)
        if minute_df is None or minute_df.empty:
            return
        
        current_price = minute_df['close']
        current_minute_time = current_dt.time()
        current_minutes = current_minute_time.hour * 60 + current_minute_time.minute

        # RSI 계산 (캐시 활용)
        historical_minute_data = self._get_historical_data_up_to(
            'minute',
            stock_code,
            current_dt,
            lookback_period=self.strategy_params['minute_rsi_period'] + 1
        )
        
        # 1. 손절 체크 ->broker 손절 처리
        current_position_size = self.broker.get_position_size(stock_code)
        if self.broker.stop_loss_params is not None and current_position_size > 0:
            if self.broker.check_and_execute_stop_loss(stock_code, current_price, current_dt):
                self.reset_signal(stock_code)
 

        # ##########################
        # 2. 종목 매도
        # --------------------------
        if order_signal == 'sell' and current_position_size > 0:
            # 매도 전략 로직 시작 : (자유롭게 작성) ==============================
            #
            # ----------------------------------------------------------------
            # 2-1. RSI 지표 계산 : 매수와 중복되지만 코드를 보기 쉽게 중복 함
            logging.info(f'[매도] {current_dt.isoformat()} - {stock_code}, 매도가: {current_price:,.0f}원, 수량: {current_position_size}주')
            self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
            self.reset_signal(stock_code)
            # current_rsi_value = self._calculate_rsi(historical_minute_data, stock_code)
            # if current_rsi_value is None:
            #     return        

            # if current_rsi_value >= self.strategy_params['minute_rsi_overbought']:
            #     logging.info(f'[매도] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 매도가: {current_price:,.0f}원, 수량: {current_position_size}주')
            #     self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
            #     self.reset_signal(stock_code)
            # 끝 매도 전략 로직 =================================================
            
            # 2-2. 강제매도 (타임컷) 전략에 강제매도가 필요하다면 다음 형식으로 추가
            if current_minutes >= (15 * 60 + 5) and current_minutes >= (15 * 60 + 20): # 오후 3:05 이후 강제매도 가능 (시간 범위 확장)
                today_data = minute_df[minute_df.index == pd.Timestamp(current_dt)]
                if today_data.empty:
                    return
                
                # 타임컷 강제매도 실행 (괴리율 조건 완화)
                self.execute_time_cut_sell(stock_code, current_dt, current_price, current_position_size, max_price_diff_ratio=0.02)
            # 끝 강제매도도 -------------------------------------------------------    

        # ##########################
        # 3. 종목 매수
        # --------------------------
        if order_signal == 'buy' and current_position_size == 0:
            target_quantity = signal_info.get('target_quantity', 0)
            if target_quantity <= 0:
                return

            # 매수 전략 로직 시작 : (자유롭게 작성) ==============================
            # RSI 매수 로직 설명 :
            #
            # ----------------------------------------------------------------
            # 3-1. RSI 지표 계산
            logging.info(f'[매수] {current_dt.isoformat()} - {stock_code} 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
            self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
            self.reset_signal(stock_code)
            # current_rsi_value = self._calculate_rsi(historical_minute_data, stock_code)
            # if current_rsi_value is None:
            #     return        
    
            # if current_rsi_value <= self.strategy_params['minute_rsi_oversold']:
            #     logging.info(f'[RSI 매수] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
            #     self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
            #     self.reset_signal(stock_code)
            
            # # 끝 매수 전략 로직 =================================================
            
            # # 3-2. 강제매수 (타임컷) 수정금지 -----------------------------------------
            # if current_minutes >= (15 * 60 + 5) and current_minutes >= (15 * 60 + 20): # 오후 3:05 이후 강제매수 가능 (시간 범위 확장)
            #     today_data = minute_df[minute_df.index == pd.Timestamp(current_dt)]
            #     if today_data.empty:
            #         return
            #     # 타임컷 강제매수 실행 (괴리율 조건 완화)
            #     self.execute_time_cut_buy(stock_code, current_dt, current_price, target_quantity, max_price_diff_ratio=0.02)
            # # 끝 강제매수 -------------------------------------------------------

