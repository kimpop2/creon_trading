# strategies/pairs_trading_daily.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import date
import logging
from .strategy import DailyStrategy

logger = logging.getLogger(__name__)

class PairsTradingDaily(DailyStrategy):
    """
    지정된 종목 쌍(Pair)의 가격 비율(스프레드)이 평균에서 벗어났을 때 진입하여
    평균으로 회귀할 때 청산하는 통계적 차익거래 전략.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self._validate_parameters()

    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = ['pairs_list', 'lookback_period', 'entry_std_dev', 'exit_std_dev']
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        logger.info(f"페어 트레이딩 전략 파라미터 검증 완료.")

    def filter_universe(self, universe_codes: List[str], current_date: date) -> List[str]:
        """
        이 전략의 유니버스는 파라미터로 지정된 'pairs_list'에 한정됩니다.
        리스트에 포함된 모든 종목을 필터링 대상으로 반환합니다.
        """
        pairs = self.strategy_params.get('pairs_list', [])
        # 중첩 리스트를 단일 리스트로 펼치고 중복 제거
        filtered_codes = list(set([code for pair in pairs for code in pair]))
        return filtered_codes

    def _calculate_strategy_signals(self, current_date: date, universe: list) -> Tuple[set, set, dict]:
        """
        각 페어의 스프레드와 Z-score를 계산하여 매매 신호를 생성합니다.
        """
        pairs_list = self.strategy_params['pairs_list']
        lookback = self.strategy_params['lookback_period']
        entry_z = self.strategy_params['entry_std_dev']
        exit_z = self.strategy_params['exit_std_dev']

        buy_candidates = set()
        sell_candidates = set()
        signal_attributes = {}
        
        current_positions = self.broker.get_current_positions()

        for stock_a_code, stock_b_code in pairs_list:
            # 두 종목의 데이터 조회
            df_a = self._get_historical_data_up_to('daily', stock_a_code, current_date, lookback_period=lookback + 5)
            df_b = self._get_historical_data_up_to('daily', stock_b_code, current_date, lookback_period=lookback + 5)

            if len(df_a) < lookback or len(df_b) < lookback:
                continue

            # 1. 스프레드(가격 비율) 계산
            spread = df_a['close'] / df_b['close']
            
            # 2. 스프레드의 이동평균, 표준편차, Z-score 계산
            spread_ma = spread.rolling(window=lookback).mean()
            spread_std = spread.rolling(window=lookback).std()
            z_score = (spread - spread_ma) / spread_std
            
            latest_z = z_score.iloc[-1]
            
            if latest_z > entry_z and stock_b_code not in current_positions: # B가 저평가
                buy_candidates.add(stock_b_code)
                # [수정] target_price 추가 (B종목 종가)
                signal_attributes[stock_b_code] = {
                    'score': latest_z,
                    'target_price': df_b.iloc[-1]['close']
                }
            elif latest_z < -entry_z and stock_a_code not in current_positions: # A가 저평가
                buy_candidates.add(stock_a_code)
                # [수정] target_price 추가 (A종목 종가)
                signal_attributes[stock_a_code] = {
                    'score': -latest_z,
                    'target_price': df_a.iloc[-1]['close']
                }
            elif abs(latest_z) < exit_z:
                if stock_a_code in current_positions: sell_candidates.add(stock_a_code)
                if stock_b_code in current_positions: sell_candidates.add(stock_b_code)

        return buy_candidates, sell_candidates, signal_attributes