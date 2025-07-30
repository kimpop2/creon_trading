# strategies/inverse_daily.py

import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import date
import logging
from .strategy import DailyStrategy

logger = logging.getLogger(__name__)

class InverseDaily(DailyStrategy):
    """
    시장 지수의 장기 추세를 판단하여 하락 추세일 때 인버스 ETF에 진입하는 전략.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self._validate_parameters()

    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = ['market_index_code', 'inverse_etf_code', 'ma_period']
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        logger.info(f"인버스 전략 파라미터 검증 완료.")

    def filter_universe(self, universe_codes: List[str], current_date: date) -> List[str]:
        """
        이 전략의 유니버스는 파라미터로 지정된 '인버스 ETF'와 '시장 지수'에 한정됩니다.
        """
        market_index = self.strategy_params['market_index_code']
        inverse_etf = self.strategy_params['inverse_etf_code']
        
        # 실제 신호 계산에 필요한 종목 코드들만 필터링하여 반환
        return [code for code in [market_index, inverse_etf] if code in universe_codes]

    def _calculate_strategy_signals(self, current_date: date, universe: list) -> Tuple[set, set, dict]:
        """
        시장 지수의 이동평균을 기준으로 하락장을 판단하고 매매 신호를 생성합니다.
        """
        market_index_code = self.strategy_params['market_index_code']
        inverse_etf_code = self.strategy_params['inverse_etf_code']
        ma_period = self.strategy_params['ma_period']

        buy_candidates = set()
        sell_candidates = set()
        signal_attributes = {}

        # 1. 시장 지수 데이터 조회
        market_df = self._get_historical_data_up_to('daily', market_index_code, current_date, lookback_period=ma_period + 5)
        if market_df is None or len(market_df) < ma_period:
            logger.warning(f"[{self.strategy_name}] 시장 지수({market_index_code}) 데이터가 부족하여 신호를 생성할 수 없습니다.")
            return buy_candidates, sell_candidates, signal_attributes

        # 2. 이동평균 계산
        market_df['ma'] = market_df['close'].rolling(window=ma_period).mean()
        
        latest_market_data = market_df.iloc[-1]
        previous_market_data = market_df.iloc[-2]

        # 3. 하락장 진입/이탈 조건 판단
        is_bear_market = (latest_market_data['close'] < latest_market_data['ma']) and \
                         (latest_market_data['ma'] < previous_market_data['ma']) # MA도 하락 추세

        current_positions = set(self.broker.get_current_positions().keys())

        # 4. 신호 생성
        if is_bear_market:
            if inverse_etf_code not in current_positions:
                etf_df = self._get_historical_data_up_to('daily', inverse_etf_code, current_date, lookback_period=1)
                if not etf_df.empty:
                    buy_candidates.add(inverse_etf_code)
                    # [수정] target_price 추가 (ETF 종가)
                    signal_attributes[inverse_etf_code] = {
                        'score': 100,
                        'target_price': etf_df.iloc[-1]['close']
                    }
        else:
            if inverse_etf_code in current_positions:
                sell_candidates.add(inverse_etf_code)

        return buy_candidates, sell_candidates, signal_attributes