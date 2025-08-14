# strategies/closing_bet_daily.py

import pandas as pd
from datetime import date
from typing import List, Tuple, Set, Dict, Any
from .strategy import DailyStrategy
from util.indicators import calculate_sma
import logging

logger = logging.getLogger(__name__)

class ClosingBetDaily(DailyStrategy):
    """
    [전략 설명]
    당일 강한 상승세를 보인 종목을 장 마감 시점에 매수하여,
    다음 날 아침의 갭(Gap) 상승 수익을 노리는 초단기 전략입니다.
    """
    def __init__(self, broker, data_store): 
        # DailyStrategy에서 broker, data_store 연결, signal 초기화 진행
        super().__init__(broker, data_store)
        self._validate_strategy_params()

    def _validate_strategy_params(self):
        """전략 파라미터의 유효성을 검증합니다."""
        pass

    def filter_universe(self, universe_codes: List[str], current_date: date) -> List[str]:
        min_trading_value = self.strategy_params.get('min_trading_value', 2_000_000_000)
        
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
        close_high_ratio = self.strategy_params['close_high_ratio']

        buy_candidates = set()
        sell_candidates = set()
        signal_attributes = {}
        current_positions = set(self.broker.get_current_positions().keys())

        lookback_period = trend_sma_period + 2

        for code in universe:
            if code in current_positions:
                continue
            
            price_df = self._get_historical_data_up_to('daily', code, current_date, lookback_period=lookback_period)
            if len(price_df) < lookback_period:
                continue

            # 2. 지표 계산 및 조건 확인 시 파라미터 변수 사용
            trend_sma = calculate_sma(price_df['close'], trend_sma_period)
            
            today = price_df.iloc[-1]
            high_low_range = today['high'] - today['low']
            if high_low_range == 0:
                continue

            is_closing_bet_candidate = (
                trend_sma.iloc[-1] > trend_sma.iloc[-2] and
                today['close'] > today['open'] and
                ((today['close'] - today['low']) / high_low_range) > close_high_ratio
            )

            if is_closing_bet_candidate:
                buy_candidates.add(code)
                score = (today['close'] / today['open'] - 1) * 100
                signal_attributes[code] = {
                    'score': score,
                    'target_price': today['close']
                }
        
        return buy_candidates, sell_candidates, signal_attributes