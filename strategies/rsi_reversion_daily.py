# strategies/rsi_reversion_daily.py

import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import date
import logging
from .strategy import DailyStrategy

logger = logging.getLogger(__name__)

class RsiReversionDaily(DailyStrategy):
    """
    단기 RSI 지표를 이용한 평균 회귀(과매도 매수) 전략.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self._validate_parameters()

    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = ['rsi_period', 'buy_threshold', 'sell_threshold', 'min_trading_value']
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        logger.info(f"RSI 평균 회귀 전략 파라미터 검증 완료.")

    def filter_universe(self, universe_codes: List[str], current_date: date) -> List[str]:
        """
        최소 거래대금 조건을 만족하는 종목만 선별합니다.
        """
        min_trading_value = self.strategy_params['min_trading_value']
        
        # 20일간의 평균 거래대금을 계산하기 위한 조회 시작일
        lookback_start_date = current_date - pd.DateOffset(days=30) # 주말 포함 여유있게

        # [수정] DB에서 모든 유니버스 종목의 평균 거래대금 정보를 한 번에 가져옵니다.
        avg_values = self.broker.manager.fetch_average_trading_values(
            universe_codes, lookback_start_date, current_date
        )
        
        if not avg_values:
            return []

        # [수정] 딕셔너리를 기반으로 빠르게 필터링합니다.
        filtered_codes = [
            code for code, avg_value in avg_values.items() 
            if avg_value >= min_trading_value
        ]
        
        return filtered_codes

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """RSI 지표를 계산합니다."""
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_strategy_signals(self, current_date: date, universe: list) -> Tuple[set, set, dict]:
        """
        RSI 지표를 기반으로 과매도 종목을 매수 후보로, 기준선 이상 회복 시 매도 후보로 결정합니다.
        """
        rsi_period = self.strategy_params['rsi_period']
        buy_threshold = self.strategy_params['buy_threshold']
        sell_threshold = self.strategy_params['sell_threshold']

        buy_candidates = set()
        sell_candidates = set()
        signal_attributes = {}
        
        current_positions = set(self.broker.get_current_positions().keys())

        for code in universe:
            price_df = self._get_historical_data_up_to('daily', code, current_date, lookback_period=rsi_period + 5)
            if price_df is None or len(price_df) < rsi_period:
                continue

            rsi_series = self._calculate_rsi(price_df['close'], rsi_period)
            if rsi_series.empty:
                continue
            
            latest_rsi = rsi_series.iloc[-1]

            # 매수 조건: 과매도 상태이며, 현재 보유하고 있지 않음
            if latest_rsi < buy_threshold and code not in current_positions:
                buy_candidates.add(code)
                signal_attributes[code] = {
                    'score': 100 - latest_rsi,
                    'target_price': price_df.iloc[-1]['close']
                }
            elif latest_rsi > sell_threshold and code in current_positions:
                sell_candidates.add(code)

        return buy_candidates, sell_candidates, signal_attributes