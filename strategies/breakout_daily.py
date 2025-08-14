# strategies/breakout_daily.py
import pandas as pd
from datetime import date, timedelta
from typing import Dict, List, Tuple, Any
from .strategy import DailyStrategy
# 'calculate_sma'는 사용하지 않으므로 삭제하거나 주석 처리합니다.
# from util.strategies_util import calculate_sma 
import logging

logger = logging.getLogger(__name__)

class BreakoutDaily(DailyStrategy):
    """
    박스권 돌파 및 거래량 급증 기반 일봉 전략.
    
    1. 일정 기간(box_period) 동안의 박스권을 정의합니다.
    2. 박스권 상단을 강하게 돌파하며 거래량이 급증하는 종목을 매수 후보로 선정합니다.
    3. 박스권 하단을 이탈하는 보유 종목을 매도 대상으로 선정합니다.
    """
    def __init__(self, broker, data_store): 
        # DailyStrategy에서 broker, data_store 연결, signal 초기화 진행
        super().__init__(broker, data_store)
        self._validate_strategy_params()

    def _validate_strategy_params(self):
        """전략 파라미터의 유효성을 검증합니다."""
        pass

    def filter_universe(self, universe_codes: List[str], current_date: date) -> List[str]:
        """최소 거래대금 조건을 만족하는 종목만 선별합니다."""
        min_trading_value = self.strategy_params['min_trading_value']
        # 거래대금 계산을 위해 최소 30일의 데이터를 확인합니다.
        lookback_start_date = current_date - timedelta(days=30)

        avg_values = self.broker.manager.fetch_average_trading_values(
            universe_codes, lookback_start_date, current_date
        )
        
        if not avg_values:
            return []

        # 최소 거래대금 기준을 통과한 종목 코드 리스트를 반환합니다.
        return [code for code, avg_value in avg_values.items() if avg_value >= min_trading_value]

    def _calculate_strategy_signals(self, current_date: date, universe: list) -> Tuple[set, set, dict]:
        """
        박스권 돌파(매수) 또는 박스권 하단 이탈(매도) 조건을 만족하는 후보를 찾습니다.
        """
        box_period = self.strategy_params['box_period']
        volume_multiplier = self.strategy_params['volume_multiplier']
        
        buy_candidates = set()
        sell_candidates = set()
        signal_attributes = {}

        current_positions = set(self.broker.get_current_positions().keys())

        for code in universe:
            # 박스권 기간과 거래량 평균 계산을 위해 충분한 데이터를 조회합니다. (최소 box_period + 1일)
            price_df = self._get_historical_data_up_to('daily', code, current_date, lookback_period=box_period + 2)
            
            if price_df is None or len(price_df) < box_period + 1:
                continue

            # --- 매수 신호 계산 (박스권 돌파) ---
            
            # 1. 박스권 정의: 오늘을 제외한 과거 'box_period'일 동안의 데이터
            lookback_df = price_df.iloc[-(box_period + 1):-1]
            
            # 2. 박스권 상단(저항선)과 평균 거래량 계산
            box_high = lookback_df['high'].max()
            avg_volume = lookback_df['volume'].mean()

            # 3. 오늘 데이터
            today = price_df.iloc[-1]

            # 4. 돌파 조건 확인
            # 조건 1: 오늘 종가가 박스권 상단보다 높음
            # 조건 2: 오늘 거래량이 박스권 평균 거래량보다 n배 이상 많음
            # 조건 3: 현재 보유하고 있지 않은 종목
            if today['close'] > box_high and today['volume'] > (avg_volume * volume_multiplier):
                if code not in current_positions:
                    buy_candidates.add(code)
                    
                    # 신호 강도(score)와 속성 정보 추가
                    score = (today['close'] / box_high - 1) * 100  # 돌파 강도를 점수화
                    signal_attributes[code] = {
                        'score': score,
                        'target_price': today['close'],  # 신호 발생일 종가
                        'stop_loss_price': box_high      # 손절매 기준 가격 (박스권 상단)
                    }

            # --- 매도 신호 계산 (박스권 하단 이탈) ---
            elif code in current_positions:
                # 1. 박스권 하단(지지선) 계산
                box_low = lookback_df['low'].min()

                # 2. 하단 이탈 조건 확인: 오늘 종가가 박스권 하단보다 낮아짐
                if today['close'] < box_low:
                    sell_candidates.add(code)
                    # 매도 신호는 즉시 실행이 중요하므로 score를 높게 설정
                    signal_attributes[code] = {
                        'score': 100, 
                        'target_price': 0
                    }
                    
        return buy_candidates, sell_candidates, signal_attributes