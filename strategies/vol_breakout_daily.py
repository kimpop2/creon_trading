# strategies/vol_breakout_daily.py

import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import date
import logging
from .strategy import DailyStrategy

logger = logging.getLogger(__name__)

class VolBreakoutDaily(DailyStrategy):
    """
    래리 윌리엄스의 변동성 돌파 전략.
    (전일 변동폭 * k) 만큼 시가에서 상승 시 추격 매수합니다.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self._validate_parameters()

    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = ['k_value', 'min_trading_value']
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        logger.info(f"변동성 돌파 전략 파라미터 검증 완료.")

    def filter_universe(self, universe_codes: List[str], current_date: date) -> List[str]:
        """
        최소 거래대금 조건을 만족하는 종목만 선별합니다.
        (RsiReversionDaily와 동일한 최적화 로직 적용)
        """
        return universe_codes
    
        min_trading_value = self.strategy_params['min_trading_value']
        lookback_start_date = current_date - pd.DateOffset(days=30)

        avg_values = self.broker.manager.fetch_average_trading_values(
            universe_codes, lookback_start_date, current_date
        )
        
        if not avg_values:
            return []

        return [
            code for code, avg_value in avg_values.items() 
            if avg_value >= min_trading_value
        ]

    def _calculate_strategy_signals(self, current_date: date, universe: list) -> Tuple[set, set, dict]:
        """
        돌파 가격을 계산하고, 이를 신호의 목표 가격으로 설정합니다.
        """
        k = self.strategy_params['k_value']
        buy_candidates = set()
        signal_attributes = {}

        for code in universe:
            # 어제와 오늘의 가격 데이터가 필요
            price_df = self._get_historical_data_up_to('daily', code, current_date, lookback_period=2)
            if price_df is None or len(price_df) < 2:
                continue
            
            yesterday_bar = price_df.iloc[-2]
            today_bar = price_df.iloc[-1]

            # 1. 전일 변동폭(range) 계산
            price_range = yesterday_bar['high'] - yesterday_bar['low']
            
            # 2. 돌파 목표가 계산
            breakout_price = today_bar['open'] + (price_range * k)
            
            # 3. 매수 조건: 당일 고가가 돌파 목표가보다 높거나 같으면 신호 발생
            # (실제 매매는 분봉 전략에서 목표가 도달 시 실행)
            if today_bar['high'] >= breakout_price:
                buy_candidates.add(code)
                score = (today_bar['high'] / breakout_price - 1) * 100 if breakout_price > 0 else 0
                # [수정] target_price 추가 (계산된 돌파 가격)
                signal_attributes[code] = {
                    'score': score,
                    'target_price': breakout_price
                }

        # 이 전략은 당일 매수 후 당일 청산을 기본으로 하므로, 별도의 매도 후보는 생성하지 않습니다.
        # (청산 로직은 분봉 전략이나 다음 날 로직에서 처리 가능)
        sell_candidates = set()
        
        return buy_candidates, sell_candidates, signal_attributes