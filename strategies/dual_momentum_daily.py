# strategies/dual_momentum_daily.py

import logging
import pandas as pd
import numpy as np 
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Any

from util.indicators import * # utils.py에 있는 모든 함수를 임포트한다고 가정 
from strategies.strategy import DailyStrategy 

logger = logging.getLogger(__name__)

class DualMomentumDaily(DailyStrategy): # DailyStrategy 상속 
    def __init__(self, broker, data_store): 
        # DailyStrategy에서 broker, data_store 연결, signal 초기화 진행
        super().__init__(broker, data_store)
        self._validate_strategy_params()

    def _validate_strategy_params(self):
        """전략 파라미터의 유효성을 검증합니다."""
        pass

    def filter_universe(self, universe_codes: List[str], current_date: date) -> List[str]:
        return universe_codes
        # [수정] 최소 거래대금 필터 추가
        min_trading_value = self.strategy_params.get('min_trading_value', 1000000000)
        lookback_start_date = current_date - timedelta(days=30)
        avg_values = self.broker.manager.fetch_average_trading_values(
            universe_codes, lookback_start_date, current_date
        )
        return [code for code, avg_value in avg_values.items() if avg_value >= min_trading_value] if avg_values else []
    
    def _calculate_strategy_signals(self, current_date: date, universe: list) -> Tuple[set, set, dict]:
        """
        [수정됨] 신호 속성을 통합된 딕셔너리(signal_attributes)로 반환합니다.
        """
        signal_attributes = {} # [신규] 통합 속성 딕셔너리
        
        # prev_trading_day = self.broker.manager.get_previous_trading_day(current_date)
        # if prev_trading_day is None:
        #     return set(), set(), {}
        
        momentum_scores = self._calculate_momentum_scores(current_date)
        safe_asset_momentum = self._calculate_safe_asset_momentum(current_date)
        current_positions = set(self.broker.get_current_positions().keys())
        inverse_asset_code = self.strategy_params.get('inverse_asset_code')
        
        if safe_asset_momentum < 0:
            # 하락장: 모든 위험자산 매도, 인버스 ETF 매수
            buy_candidates = {inverse_asset_code} if inverse_asset_code else set()
            sell_candidates = {code for code in current_positions if code != inverse_asset_code}
        else:
            # 상승장: 상대 모멘텀 로직 수행
            buy_candidates, _ = self._select_buy_candidates(momentum_scores, 0)
            sell_candidates = {code for code in current_positions if code not in buy_candidates}
            if inverse_asset_code and inverse_asset_code in current_positions:
                sell_candidates.add(inverse_asset_code)

        # signal_attributes 채우기 (score 위주)        
        for code in buy_candidates:
            price_df = self._get_historical_data_up_to('daily', code, current_date, lookback_period=1)
            if not price_df.empty:
                # [수정] target_price 추가 (신호 발생일 종가)
                signal_attributes[code] = {
                    'score': momentum_scores.get(code, 999),
                    'target_price': price_df.iloc[-1]['close']
                }
        
        return buy_candidates, sell_candidates, signal_attributes
    
    def _calculate_momentum_scores(self, current_daily_date):
        """모든 종목의 모멘텀 스코어를 계산합니다."""
        momentum_scores = {}
        for stock_code in self.data_store['daily']:
            if stock_code == self.strategy_params['safe_asset_code']:
                continue  # 안전자산은 모멘텀 계산에서 제외

            daily_df = self.data_store['daily'][stock_code]
            if daily_df.empty:
                continue

            historical_data = self._get_historical_data_up_to(
                'daily',
                stock_code,
                current_daily_date,
                lookback_period=self.strategy_params['momentum_period'] + 1
            )

            if len(historical_data) < self.strategy_params['momentum_period']:
                logging.debug(f'{stock_code} 종목의 모멘텀 계산을 위한 데이터가 부족합니다.')
                continue

            momentum_score = calculate_momentum(historical_data, self.strategy_params['momentum_period']).iloc[-1]
            momentum_scores[stock_code] = momentum_score

        return momentum_scores
        
    def _calculate_safe_asset_momentum(self, current_daily_date):
        """안전자산의 모멘텀을 계산합니다."""
        safe_asset_df = self.data_store['daily'].get(self.strategy_params['safe_asset_code'])
        safe_asset_momentum = 0
        if safe_asset_df is not None and not safe_asset_df.empty:
            safe_asset_data = self._get_historical_data_up_to(
                'daily',
                self.strategy_params['safe_asset_code'],
                current_daily_date,
                lookback_period=self.strategy_params['momentum_period'] + 1
            )
            if len(safe_asset_data) >= self.strategy_params['momentum_period']:
                safe_asset_momentum = calculate_momentum(safe_asset_data, self.strategy_params['momentum_period']).iloc[-1]
        return safe_asset_momentum

    def _select_buy_candidates(self, momentum_scores, safe_asset_momentum):
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = set()
        
        for rank, (stock_code, score) in enumerate(sorted_stocks, 1):
            if rank <= self.strategy_params['max_position_count'] and score > safe_asset_momentum:
                buy_candidates.add(stock_code)

        return buy_candidates, sorted_stocks
