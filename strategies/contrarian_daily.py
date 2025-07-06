# strategies/contrarian_daily.py

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from trade.trader import Trader
from strategies.strategy import DailyStrategy
from util.strategies_util import *

logger = logging.getLogger(__name__)

class ContrarianDaily(DailyStrategy):
    """
    역추세(Contrarian) 기반 일봉 전략입니다.
    과매수/과매도 상황에서 반대 매매를 통해 수익을 추구합니다.
    RSI, 볼린저 밴드, 스토캐스틱 등을 활용하여 역추세 신호를 생성합니다.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        # DailyStrategy 에서 trade의 broker, data_store 연결, signal 초기화 진행
        super().__init__(broker, data_store, strategy_params)
        self._validate_strategy_params() # 전략 파라미터 검증

        # 역추세 지표 계산을 위한 캐시 추가
        self.rsi_cache = {}  # RSI 캐시
        self.bb_cache = {}   # 볼린저 밴드 캐시
        self.stoch_cache = {}  # 스토캐스틱 캐시
        self.last_prices = {}  # 마지막 가격 캐시

        self.strategy_name = "ContrarianDaily"
        
    def _validate_strategy_params(self):
        """전략 파라미터의 유효성을 검증합니다."""
        required_params = ['rsi_period', 'rsi_oversold', 'rsi_overbought', 'bb_period', 'bb_std', 'stoch_period', 'num_top_stocks']
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"역추세 전략에 필요한 파라미터 '{param}'이 설정되지 않았습니다.")
        
        # RSI 과매수/과매도 검증
        if self.strategy_params['rsi_overbought'] <= self.strategy_params['rsi_oversold']:
            raise ValueError("RSI 과매수 점수는 과매도 점수보다 커야 합니다.")
            
        logging.info(f"역추세 전략 파라미터 검증 완료: "
                   f"RSI기간={self.strategy_params['rsi_period']}, "
                   f"RSI과매도={self.strategy_params['rsi_oversold']}, "
                   f"RSI과매수={self.strategy_params['rsi_overbought']}, "
                   f"볼린저밴드기간={self.strategy_params['bb_period']}, "
                   f"볼린저밴드표준편차={self.strategy_params['bb_std']}, "
                   f"스토캐스틱기간={self.strategy_params['stoch_period']}, "
                   f"선택종목수={self.strategy_params['num_top_stocks']}")

    def _calculate_contrarian_score(self, historical_data, stock_code):
        """
        역추세 점수를 계산합니다.
        RSI, 볼린저 밴드, 스토캐스틱을 종합하여 점수를 산출합니다.
        """
        if len(historical_data) < max(self.strategy_params['rsi_period'], 
                                    self.strategy_params['bb_period'], 
                                    self.strategy_params['stoch_period']) + 1:
            return None, None
        
        # 1. RSI 계산
        rsi = calculate_rsi(historical_data['close'], self.strategy_params['rsi_period'])
        current_rsi = rsi.iloc[-1]
        
        # 2. 볼린저 밴드 계산
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            historical_data, 
            self.strategy_params['bb_period'], 
            self.strategy_params['bb_std']
        )
        current_price = historical_data['close'].iloc[-1]
        bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        # 3. 스토캐스틱 계산
        stoch_k, stoch_d = calculate_stochastic(
            historical_data, 
            self.strategy_params['stoch_period']
        )
        current_stoch_k = stoch_k.iloc[-1]
        current_stoch_d = stoch_d.iloc[-1]
        
        # 4. 역추세 점수 계산
        buy_score = 0
        sell_score = 0
        
        # 매수 점수 (과매도 상황에서 반등 기대)
        if current_rsi <= self.strategy_params['rsi_oversold']:
            buy_score += (self.strategy_params['rsi_oversold'] - current_rsi) * 2  # RSI 과매도 정도
        
        if bb_position <= 0.2:  # 볼린저 밴드 하단 20% 이하
            buy_score += (0.2 - bb_position) * 100
        
        if current_stoch_k <= 20 and current_stoch_d <= 20:  # 스토캐스틱 과매도
            buy_score += (20 - min(current_stoch_k, current_stoch_d)) * 2
        
        # 매도 점수 (과매수 상황에서 하락 기대)
        if current_rsi >= self.strategy_params['rsi_overbought']:
            sell_score += (current_rsi - self.strategy_params['rsi_overbought']) * 2  # RSI 과매수 정도
        
        if bb_position >= 0.8:  # 볼린저 밴드 상단 20% 이상
            sell_score += (bb_position - 0.8) * 100
        
        if current_stoch_k >= 80 and current_stoch_d >= 80:  # 스토캐스틱 과매수
            sell_score += (min(current_stoch_k, current_stoch_d) - 80) * 2
        
        return buy_score, sell_score

    def run_daily_logic(self, current_date: datetime.date):
        """
        역추세 매매 로직을 실행합니다.
        과매수/과매도 상황에서 반대 매매 신호를 생성합니다.
        """
        logging.info(f"{current_date} - --- 일간 역추세 로직 실행 중 (전일 데이터 기준) ---")
        prev_trading_day = current_date

        # 1. 역추세 신호 점수 계산 (전일 데이터까지만 사용)
        buy_scores = {}  # 매수 점수 (과매도에서 반등 기대)
        sell_scores = {}  # 매도 점수 (과매수에서 하락 기대)
        all_stock_codes = list(self.data_store['daily'].keys())
        
        for stock_code in all_stock_codes:
            historical_data = self._get_historical_data_up_to(
                'daily', 
                stock_code, 
                prev_trading_day, 
                lookback_period=max(self.strategy_params['rsi_period'], 
                                  self.strategy_params['bb_period'], 
                                  self.strategy_params['stoch_period']) + 1
            )
            
            if historical_data.empty or len(historical_data) < max(self.strategy_params['rsi_period'], 
                                                                 self.strategy_params['bb_period'], 
                                                                 self.strategy_params['stoch_period']) + 1:
                continue
            
            buy_score, sell_score = self._calculate_contrarian_score(historical_data, stock_code)
            
            if buy_score is not None and buy_score > 0:
                buy_scores[stock_code] = buy_score
                
            if sell_score is not None and sell_score > 0:
                sell_scores[stock_code] = sell_score

        # 2. 매수 후보 종목 선정 (과매도 상황에서 반등 기대)
        sorted_buy_stocks = sorted(buy_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = set()
        for rank, (stock_code, score) in enumerate(sorted_buy_stocks, 1):
            if rank <= self.strategy_params['num_top_stocks'] and score >= 10:  # 최소 점수 조건
                buy_candidates.add(stock_code)
                logging.info(f"역추세 매수 후보 {rank}위: {stock_code} (점수: {score:.1f})")
                
        # 3. 매도 후보 종목 선정 (과매수 상황에서 하락 기대)
        sorted_sell_stocks = sorted(sell_scores.items(), key=lambda x: x[1], reverse=True)
        sell_candidates = set()
        current_positions = set(self.broker.positions.keys())
        
        # 보유 중인 종목 중 과매수 상황인 종목들
        for stock_code in current_positions:
            if stock_code in sell_scores and sell_scores[stock_code] >= 10:
                sell_candidates.add(stock_code)
                logging.info(f"역추세 매도 후보: {stock_code} (점수: {sell_scores[stock_code]:.1f})")
        
        # 보유 중이지만 매수 후보에서 빠진 종목들 (일정 기간 후 매도 고려)
        min_holding_days = self.strategy_params.get('min_holding_days', 5)  # 역추세는 더 오래 홀딩
        for stock_code in current_positions:
            if stock_code not in buy_candidates and stock_code not in sell_candidates:
                position_info = self.broker.positions.get(stock_code, {})
                entry_date = position_info.get('entry_date', prev_trading_day)
                holding_days = (prev_trading_day - entry_date).days if entry_date else 0
                if holding_days >= min_holding_days:
                    sell_candidates.add(stock_code)
                    logging.info(f"홀딩기간 경과로 매도 후보 추가: {stock_code} (홀딩일수: {holding_days}일)")

        logging.info(f"역추세 매도 후보 최종 선정: {sorted(sell_candidates)}")
        
        # 4. 신호 생성 및 업데이트 (부모 메서드) - 전일 데이터 기준
        final_positions = self._generate_signals(prev_trading_day, buy_candidates, sorted_buy_stocks, sell_candidates)
        
        # 5. 리밸런싱 요약 로그
        self._log_rebalancing_summary(prev_trading_day, buy_candidates, final_positions, sell_candidates) 