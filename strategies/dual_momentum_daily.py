# strategies/dual_momentum_daily.py

import datetime
import logging
import pandas as pd
import numpy as np 
from datetime import datetime, time
from typing import Dict, List, Tuple, Any

from util.strategies_util import * # utils.py에 있는 모든 함수를 임포트한다고 가정 
from strategies.strategy import DailyStrategy 

logger = logging.getLogger(__name__)

class DualMomentumDaily(DailyStrategy): # DailyStrategy 상속 
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]): 
        # DailyStrategy에서 broker, data_store 연결, signal 초기화 진행
        super().__init__(broker, data_store, strategy_params)
        self.strategy_name = "DualMomentumDaily"
        
        # 파라미터 검증
        self._validate_parameters()

    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = [
            'momentum_period', 'rebalance_weekday', 'safe_asset_code'
        ]
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        
        logger.info(f"듀얼 모멘텀 전략 파라미터 검증 완료: "
                   f"모멘텀기간={self.strategy_params['momentum_period']}일, "
                   f"리밸런싱요일={self.strategy_params['rebalance_weekday']}, "
                   f"선택종목수={self.strategy_params['num_top_stocks']}개")

    def _calculate_strategy_signals(self, current_date: datetime.date, universe: list) -> Tuple[set, set, dict]:
        """
        [수정됨] 신호 속성을 통합된 딕셔너리(signal_attributes)로 반환합니다.
        """
        signal_attributes = {} # [신규] 통합 속성 딕셔너리
        
        prev_trading_day = self.broker.manager.get_previous_trading_day(current_date)
        if prev_trading_day is None:
            return set(), set(), {}
        
        momentum_scores = self._calculate_momentum_scores(prev_trading_day)
        safe_asset_momentum = self._calculate_safe_asset_momentum(prev_trading_day)
        current_positions = set(self.broker.get_current_positions().keys())
        inverse_asset_code = self.strategy_params.get('inverse_asset_code')
        
        if safe_asset_momentum < 0:
            # 하락장: 모든 위험자산 매도, 인버스 ETF 매수
            buy_candidates = {inverse_asset_code} if inverse_asset_code else set()
            sell_candidates = {code for code in current_positions if code != inverse_asset_code}
            
            # [신규] 신호 속성 정의
            if inverse_asset_code:
                signal_attributes[inverse_asset_code] = {'score': 999, 'execution_type': 'market'}
            for code in sell_candidates:
                signal_attributes[code] = {'execution_type': 'market'}
        else:
            # 상승장: 상대 모멘텀 로직 수행
            buy_candidates, _ = self._select_buy_candidates(momentum_scores, 0)
            sell_candidates = {code for code in current_positions if code not in buy_candidates}
            if inverse_asset_code and inverse_asset_code in current_positions:
                sell_candidates.add(inverse_asset_code)
                
            # [신규] 신호 속성 정의
            for code in buy_candidates:
                hist_data = self._get_historical_data_up_to('daily', code, prev_trading_day)
                if not hist_data.empty:
                    target_price = hist_data['close'].iloc[-1]
                    signal_attributes[code] = {
                        'score': momentum_scores.get(code, 0),
                        'target_price': target_price,
                        'execution_type': 'pullback' # 전일 종가 이하는 'pullback' 전술
                    }
            for code in sell_candidates:
                signal_attributes[code] = {'execution_type': 'market'}

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
            if rank <= self.strategy_params['num_top_stocks'] and score > safe_asset_momentum:
                buy_candidates.add(stock_code)

        return buy_candidates, sorted_stocks
