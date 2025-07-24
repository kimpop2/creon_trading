# strategies/triple_screen_daily.py (Refactored)

import logging
from datetime import date
from typing import Dict, List, Tuple, Any
from strategies.strategy import DailyStrategy
from util.strategies_util import calculate_sma, calculate_rsi

logger = logging.getLogger(__name__)

class TripleScreenDaily(DailyStrategy):
    """
    [리팩토링됨] 알렉산더 엘더의 삼중창 시스템
    - 1단계: 장기 추세 확인 (이동평균)
    - 2단계: 중기 모멘텀 확인 (RSI)
    - 3단계: 단기 진입점 설정 (전일 고가 돌파)
    """
    
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self.strategy_name = "TripleScreenDaily"
        self._validate_parameters()
        
    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = [
            'trend_ma_period', 'momentum_rsi_period', 'momentum_rsi_oversold', 
            'momentum_rsi_overbought', 'num_top_stocks', 'min_trend_strength'
        ]
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        logger.info(f"삼중창 시스템 파라미터 검증 완료.")

    def _calculate_strategy_signals(self, current_date: date, universe: list) -> Tuple[set, set, list, dict]:
        """
        삼중창 시스템의 고유 로직을 구현하여 매수/매도 후보를 선정합니다.
        """
        buy_scores = {}
        stock_target_prices = {}
        
        # 파라미터 로드
        trend_ma_period = self.strategy_params['trend_ma_period']
        rsi_period = self.strategy_params['momentum_rsi_period']
        rsi_oversold = self.strategy_params['momentum_rsi_oversold']
        min_trend_strength = self.strategy_params['min_trend_strength']
        num_top_stocks = self.strategy_params['num_top_stocks']
        
        for stock_code in universe:
            if stock_code.startswith('U'): continue

            historical_data = self._get_historical_data_up_to(
                'daily', stock_code, current_date, lookback_period=trend_ma_period + 5
            )
            if len(historical_data) < trend_ma_period + 1:
                continue

            current_price = historical_data['close'].iloc[-1]

            # --- 1단계: 장기 추세 확인 (이동평균) ---
            trend_ma = calculate_sma(historical_data['close'], trend_ma_period).iloc[-1]
            trend_strength = (current_price - trend_ma) / trend_ma

            # 추세가 상승이고 최소 강도를 만족하는가?
            if trend_strength < min_trend_strength:
                continue # 1단계 탈락

            # --- 2단계: 중기 모멘텀 확인 (RSI) ---
            rsi = calculate_rsi(historical_data['close'], rsi_period).iloc[-1]
            
            # 상승 추세에서는 주가가 조정을 받을 때(RSI 과매도 근처)가 매수 기회
            if rsi > rsi_oversold:
                continue # 2단계 탈락

            # --- 3단계: 단기 진입점 설정 및 신호 생성 ---
            # 모든 스크린을 통과한 종목
            # 진입 목표가는 '전일 고가'를 돌파하는 시점으로 설정
            target_price = historical_data['high'].iloc[-2]
            stock_target_prices[stock_code] = target_price
            
            # 점수는 RSI를 사용 (RSI가 낮을수록 더 좋은 조정으로 간주 -> 점수 높게)
            buy_scores[stock_code] = 100 - rsi
        
        # 최종 매수/매도 후보 선정
        sorted_buy_stocks = sorted(buy_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = {code for i, (code, score) in enumerate(sorted_buy_stocks) if i < num_top_stocks}
        
        current_positions = set(self.broker.get_current_positions().keys())
        sell_candidates = set()
        for code in current_positions:
            # 매도 조건: 장기 추세(1단계)가 꺾였을 때
            hist_data = self._get_historical_data_up_to('daily', code, current_date, lookback_period=trend_ma_period + 1)
            if len(hist_data) < trend_ma_period + 1:
                continue
                
            current_price = hist_data['close'].iloc[-1]
            trend_ma = calculate_sma(hist_data['close'], trend_ma_period).iloc[-1]
            
            if current_price < trend_ma:
                sell_candidates.add(code)
                
        return buy_candidates, sell_candidates, sorted_buy_stocks, stock_target_prices