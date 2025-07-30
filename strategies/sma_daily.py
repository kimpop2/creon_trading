# strategies/sma_daily.py (최종 수정안)
import pandas as pd
from datetime import date
from typing import Dict, List, Tuple, Any
from .strategy import DailyStrategy
from util.strategies_util import calculate_sma
import logging

logger = logging.getLogger(__name__)

class SMADaily(DailyStrategy):
    """
    SMA(Simple Moving Average) 기반 일봉 전략 (단순화 버전).
    골든크로스/데드크로스 발생 시 매매 대상을 결정하는 역할만 수행합니다.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self._validate_parameters()

    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = ['short_sma_period', 'long_sma_period', 'min_trading_value']
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"SMA 전략에 필요한 파라미터 '{param}'이 설정되지 않았습니다.")
        logger.info("SMA 전략 파라미터 검증 완료.")

    def filter_universe(self, universe_codes: List[str], current_date: date) -> List[str]:
        """최소 거래대금 조건을 만족하는 종목만 선별합니다."""
        min_trading_value = self.strategy_params['min_trading_value']
        lookback_start_date = current_date - pd.DateOffset(days=30)

        avg_values = self.broker.manager.fetch_average_trading_values(
            universe_codes, lookback_start_date, current_date
        )
        
        if not avg_values:
            return []

        return [code for code, avg_value in avg_values.items() if avg_value >= min_trading_value]

    def _calculate_strategy_signals(self, current_date: date, universe: list) -> Tuple[set, set, dict]:
        """
        골든크로스/데드크로스 조건을 만족하는 매수/매도 후보를 찾습니다.
        """
        short_period = self.strategy_params['short_sma_period']
        long_period = self.strategy_params['long_sma_period']
        
        buy_candidates = set()
        sell_candidates = set()
        signal_attributes = {}

        current_positions = set(self.broker.get_current_positions().keys())

        for code in universe:
            price_df = self._get_historical_data_up_to('daily', code, current_date, lookback_period=long_period + 2)
            if price_df is None or len(price_df) < long_period + 2:
                continue

            # 이동평균 계산
            short_sma = calculate_sma(price_df['close'], short_period)
            long_sma = calculate_sma(price_df['close'], long_period)

            # [단순화] 골든크로스 조건
            if short_sma.iloc[-1] > long_sma.iloc[-1] and short_sma.iloc[-2] <= long_sma.iloc[-2]:
                if code not in current_positions:
                    buy_candidates.add(code)
                    score = (short_sma.iloc[-1] / long_sma.iloc[-1] - 1) * 100
                    # [수정] target_price 추가 (신호 발생일 종가)
                    signal_attributes[code] = {
                        'score': score,
                        'target_price': price_df.iloc[-1]['close']
                    }

            # [단순화] 데드크로스 조건
            elif short_sma.iloc[-1] < long_sma.iloc[-1] and short_sma.iloc[-2] >= long_sma.iloc[-2]:
                if code in current_positions:
                    sell_candidates.add(code)

        return buy_candidates, sell_candidates, signal_attributes