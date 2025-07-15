# strategies/breakout_daily.py

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from util.strategies_util import calculate_volume_ma_incremental
from strategies.strategy import DailyStrategy

class BreakoutDaily(DailyStrategy):
    """
    돌파 매매 기반 일봉 전략입니다.
    최근 N봉 신고가 돌파와 거래량 조건을 활용하여 매매 신호를 생성합니다.
    """
    
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        # DailyStrategy에서 broker, data_store 연결, signal 초기화 진행
        super().__init__(broker, data_store, strategy_params)
        self.strategy_name = "BreakoutDaily"
        
        # 전략 파라미터 검증
        self._validate_parameters()
        
        # 돌파 매매 누적 계산을 위한 캐시 추가
        self.high_price_cache = {}  # 최고가 캐시
        self.volume_cache = {}  # 거래량 MA 캐시
        self.last_prices = {}  # 마지막 가격 캐시 (부모 클래스에서 관리될 수 있음)
        self.last_volumes = {}  # 마지막 거래량 캐시 (부모 클래스에서 관리될 수 있음)
        
    def _validate_parameters(self):
        """전략 파라미터의 유효성을 검증합니다."""
        required_params = ['breakout_period', 'volume_ma_period', 'num_top_stocks', 'volume_multiplier']
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"돌파 매매 전략에 필요한 파라미터 '{param}'이 설정되지 않았습니다.")
        
        # 돌파 기간 검증
        if self.strategy_params['breakout_period'] <= 0:
            raise ValueError("돌파 기간은 0보다 커야 합니다.")
            
        if self.strategy_params['volume_multiplier'] <= 0:
            raise ValueError("거래량 배수는 0보다 커야 합니다.")
            
        logging.info(f"돌파 매매 전략 파라미터 검증 완료: "
                     f"돌파기간={self.strategy_params['breakout_period']}, "
                     f"거래량MA={self.strategy_params['volume_ma_period']}, "
                     f"거래량배수={self.strategy_params['volume_multiplier']}, "
                     f"선택종목수={self.strategy_params['num_top_stocks']}")

    def run_daily_logic(self, current_date):
        """
        돌파 매매 로직을 실행합니다: 돌파 신호 점수 계산 → 상위 N개 종목 선정 → _generate_signals → _log_rebalancing_summary 호출
        전일 데이터까지만 사용하여 장전 판단이 가능하도록 합니다.
        """
        logging.info(f"{current_date} - --- 일간 돌파 매매 로직 실행 중 (전일 데이터 기준) ---")
        prev_trading_day = current_date # 현재 날짜를 prev_trading_day로 사용하여 전일 데이터를 가져옴

        # 1. 돌파 신호 점수 계산 (전일 데이터까지만 사용)
        buy_scores = {}  # 매수 점수
        sell_scores = {} # 매도 점수
        all_stock_codes = list(self.data_store['daily'].keys())
        
        # 필요한 최대 과거 데이터 기간
        lookback_needed = max(self.strategy_params['breakout_period'], self.strategy_params['volume_ma_period']) + 1
        
        for stock_code in all_stock_codes:
            historical_data = self._get_historical_data_up_to('daily', stock_code, prev_trading_day, lookback_period=lookback_needed)
            
            # 충분한 데이터가 없으면 건너김
            if historical_data.empty or len(historical_data) < lookback_needed:
                continue
            
            current_close = historical_data['close'].iloc[-1]
            current_volume = historical_data['volume'].iloc[-1]
            
            # 이전 breakout_period 기간 동안의 최고가 (현재 종가 제외)
            # historical_data.iloc[:-1]는 현재 날짜를 제외한 이전 데이터
            # .rolling(window=self.strategy_params['breakout_period']) 는 해당 기간의 최고가를 구함
            # .max().iloc[-1]는 그 최고가 중 가장 마지막 값을 가져옴
            
            # 이전 'breakout_period' 기간 동안의 고가 (현재 캔들 제외)
            period_high_data = historical_data['high'].iloc[-(self.strategy_params['breakout_period'] + 1):-1]
            if not period_high_data.empty:
                highest_high_in_period = period_high_data.max()
            else:
                highest_high_in_period = -1 # 유효하지 않은 값으로 초기화

            volume_ma = calculate_volume_ma_incremental(historical_data, self.strategy_params['volume_ma_period'], self.volume_cache)[0]
            
            # 매수 조건: 종가가 최근 N봉 신고가 돌파 + 거래량 조건
            # 현재 종가(current_close)가 이전 'breakout_period' 동안의 최고가(highest_high_in_period)를 돌파
            if current_close > highest_high_in_period and current_volume > volume_ma * self.strategy_params['volume_multiplier']:
                score = (current_close - highest_high_in_period) / highest_high_in_period * 100
                buy_scores[stock_code] = score
                logging.debug(f"[{stock_code}] 돌파 매수 신호 발생: 종가={current_close:.2f}, 최고가={highest_high_in_period:.2f}, 거래량={current_volume}, 거래량MA={volume_ma:.2f}")

            # 매도 조건: 보유 중인 종목이 돌파 조건을 만족하지 못하거나, 특정 손절/이익실현 조건을 추가할 수 있음
            # 예시: 현재 보유 종목이 최고가 대비 X% 하락하거나, 거래량이 급감하는 경우 등.
            # 여기서는 매수 후보에서 빠진 경우와 특정 손실율 초과 시 매도 신호를 발생시킵니다.
            # (SMA 전략의 매도 로직과 유사하게 가져감)
            
        # 2. 매수 후보 종목 선정
        sorted_buy_stocks = sorted(buy_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = set()
        for rank, (stock_code, _) in enumerate(sorted_buy_stocks, 1):
            if rank <= self.strategy_params['num_top_stocks']:
                buy_candidates.add(stock_code)
        
        # 매도 후보 종목 선정 (보유 중인데 매수 후보에 없는 종목은 일정 기간(3일) 이상 홀딩 후에만 매도 후보로 추가)
        current_positions = set(self.broker.positions.keys())
        sell_candidates = set()
        min_holding_days = self.strategy_params.get('min_holding_days', 3) # 최소 보유 기간
        
        for stock_code in current_positions:
            # 매수 후보에서 빠진 종목은 일정 기간 홀딩 후 매도 후보
            if stock_code not in buy_candidates:
                position_info = self.broker.positions.get(stock_code, {})
                entry_date = position_info.get('entry_date', prev_trading_day)
                holding_days = (prev_trading_day - entry_date).days if entry_date else 0
                if holding_days >= min_holding_days:
                    sell_candidates.add(stock_code)
                    logging.info(f"매수 후보 제외+홀딩기간 경과로 매도 후보 추가: {stock_code}")
            
            # (선택 사항) 손절 또는 이익실현 조건 추가 (예시: -5% 손실 발생 시 무조건 매도)
            # if stock_code in self.broker.positions:
            #     current_price = self.data_store['daily'][stock_code]['close'].loc[prev_trading_day] if prev_trading_day in self.data_store['daily'][stock_code]['close'].index else self.broker.positions[stock_code]['current_price'] # 실제 가격 업데이트 필요
            #     purchase_price = self.broker.positions[stock_code]['purchase_price']
            #     if current_price < purchase_price * 0.95: # 5% 손실 시
            #         sell_candidates.add(stock_code)
            #         logging.info(f"손절 조건 충족으로 매도 후보 추가: {stock_code}")


        logging.info(f"매도 후보 최종 선정: {sorted(sell_candidates)}")
        
        # 3. 신호 생성 및 업데이트 (부모 메서드) - 전일 데이터 기준
        final_positions = self._generate_signals(prev_trading_day, buy_candidates, sorted_buy_stocks, sell_candidates)
        
        # 4. 리밸런싱 요약 로그
        self._log_rebalancing_summary(prev_trading_day, buy_candidates, final_positions, sell_candidates)