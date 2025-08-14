# strategies/pullback_daily.py

import pandas as pd
from datetime import date
from typing import List, Tuple, Set, Dict, Any
from .strategy import DailyStrategy
from util.indicators import calculate_sma
import logging

logger = logging.getLogger(__name__)

class PullbackDaily(DailyStrategy):
    """
    [전략 설명]
    안정적인 상승 추세를 보이는 종목이 일시적인 조정을 보일 때(눌림목),
    기술적 지지 구간에서 매수하여 반등을 노리는 단기 매매 전략입니다.
    """
    def __init__(self, broker, data_store): 
        # DailyStrategy에서 broker, data_store 연결, signal 초기화 진행
        super().__init__(broker, data_store)
        self._validate_strategy_params()

    def _validate_strategy_params(self):
        """전략 파라미터의 유효성을 검증합니다."""
        pass
    
    def filter_universe(self, universe_codes: List[str], current_date: date) -> List[str]:
        min_trading_value = self.strategy_params.get('min_trading_value', 1_000_000_000)
        
        filtered_universe = []
        for code in universe_codes:
            price_df = self._get_historical_data_up_to('daily', code, current_date, lookback_period=21)
            if len(price_df) < 21:
                continue
            
            avg_trading_value = (price_df['close'] * price_df['volume']).rolling(window=20).mean().iloc[-1]
            if avg_trading_value < min_trading_value:
                continue

            filtered_universe.append(code)
        
        logger.info(f"[{self.strategy_name}] 유니버스 필터링: {len(universe_codes)}개 -> {len(filtered_universe)}개")
        return filtered_universe

    def _calculate_strategy_signals(self, current_date: date, universe: list) -> Tuple[Set[str], Set[str], Dict[str, Any]]:
        # 1. 설정 파일에서 파라미터 불러오기
        trend_sma_period = self.strategy_params['trend_sma_period']
        pullback_sma_period = self.strategy_params['pullback_sma_period']

        buy_candidates = set()
        sell_candidates = set()
        signal_attributes = {}
        current_positions = set(self.broker.get_current_positions().keys())

        lookback_period = max(trend_sma_period, pullback_sma_period) + 2

        for code in universe:
            price_df = self._get_historical_data_up_to('daily', code, current_date, lookback_period=lookback_period)
            if len(price_df) < lookback_period:
                continue

            # 2. 지표 계산 시 파라미터 변수 사용
            pullback_sma = calculate_sma(price_df['close'], pullback_sma_period)
            trend_sma = calculate_sma(price_df['close'], trend_sma_period)
            
            # --- 매수 신호 ---
            if code not in current_positions:
                is_pullback_candidate = (
                    trend_sma.iloc[-1] >= trend_sma.iloc[-2] and
                    price_df.iloc[-1]['close'] < pullback_sma.iloc[-1] and
                    price_df.iloc[-1]['low'] > price_df.iloc[-2]['low']
                )

                if is_pullback_candidate:
                    buy_candidates.add(code)
                    score = (trend_sma.iloc[-1] / price_df.iloc[-1]['close'] - 1) * 100
                    signal_attributes[code] = {
                        'score': score,
                        'target_price': price_df.iloc[-1]['close']
                    }

            # --- 매도 신호 ---
            elif code in current_positions:
                if price_df.iloc[-1]['close'] < trend_sma.iloc[-1]:
                    sell_candidates.add(code)
                    signal_attributes[code] = {'score': 100, 'target_price': None}

        return buy_candidates, sell_candidates, signal_attributes