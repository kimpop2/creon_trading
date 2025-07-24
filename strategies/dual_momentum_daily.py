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

    def _calculate_strategy_signals(self, current_date: datetime.date, universe: list) -> Tuple[set, set, list, dict]:
        # --- 기존 run_daily_logic의 내용이 이 안으로 들어옵니다 ---
        
        # 듀얼 모멘텀 전략은 목표가를 사용하지 않으므로 빈 딕셔너리로 초기화
        stock_target_prices = {}

        prev_trading_day = self.broker.manager.get_previous_trading_day(current_date) # 단순화된 예시
        if prev_trading_day is None:
            logger.warning(f"[{self.strategy_name}] {current_date}의 이전 영업일을 찾을 수 없어 전략 실행을 건너뜁니다.")
            # 빈 값을 반환하여 오류 없이 정상 종료
            return set(), set(), [], {}
        
        # 1. 모멘텀 스코어 계산
        momentum_scores = self._calculate_momentum_scores(prev_trading_day)
        # 2. 안전자산 모멘텀 계산
        safe_asset_momentum = self._calculate_safe_asset_momentum(prev_trading_day)
        current_positions = set(self.broker.get_current_positions().keys())
        inverse_asset_code = self.strategy_params.get('inverse_asset_code')
        
        if safe_asset_momentum < 0:
            # 하락장: 모든 위험자산 매도, 인버스 ETF 매수
            logger.info(f"[{self.strategy_name}] 절대 모멘텀 하락 감지. 인버스 ETF({inverse_asset_code})로 전환합니다.")
            
            # 1. 매수 후보는 오직 인버스 ETF 하나입니다.
            buy_candidates = {inverse_asset_code}
            sorted_buy_stocks = [(inverse_asset_code, 999)] # 점수는 임의로 높게 부여
            
            # 2. 매도 후보는 현재 보유 중인 모든 종목(인버스 ETF 제외)입니다.
            sell_candidates = {code for code in current_positions if code != inverse_asset_code}
        else:
            # 상승장: 기존 상대 모멘텀 로직 수행
            buy_candidates, sorted_buy_stocks = self._select_buy_candidates(momentum_scores, 0) # [수정] 이제 비교 기준은 0
            
            # 매도 후보는 현재 보유 종목 중 새로운 매수 후보에 포함되지 않은 모든 것
            sell_candidates = {code for code in current_positions if code not in buy_candidates}
            
            # 인버스 ETF를 보유하고 있다면 무조건 매도
            if inverse_asset_code and inverse_asset_code in current_positions:
                sell_candidates.add(inverse_asset_code)        
        
        if buy_candidates:
            logger.info(f"[{self.strategy_name}] 매수 후보 {buy_candidates}의 목표가(전일 종가)를 설정합니다.")
            for code in buy_candidates:
                # current_date 기준의 데이터는 당일 데이터이므로, prev_trading_day를 사용해야 함
                historical_data = self._get_historical_data_up_to('daily', code, prev_trading_day)
                if historical_data is not None and not historical_data.empty:
                    # 데이터의 마지막 행(전일)의 종가를 목표가로 설정
                    target_price = historical_data['close'].iloc[-1]
                    stock_target_prices[code] = target_price
                    logger.debug(f"  - {code}: 목표가 {target_price:,.0f}원 설정")

        # 5. 최종적으로 계산된 값들을 튜플 형태로 반환
        return buy_candidates, sell_candidates, sorted_buy_stocks, stock_target_prices
    
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
