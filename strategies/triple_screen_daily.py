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

    def _calculate_strategy_signals(self, current_date: date, universe: list) -> Tuple[set, set, dict]:
        """
        [수정됨] 신호 속성을 통합된 딕셔너리(signal_attributes)로 반환합니다.
        """
        buy_scores = {}
        # [수정] 여러 데이터 구조를 signal_attributes 하나로 통합
        signal_attributes = {}

        # 파라미터 로드
        trend_ma_period = self.strategy_params['trend_ma_period']
        rsi_period = self.strategy_params['momentum_rsi_period']
        rsi_oversold = self.strategy_params['momentum_rsi_oversold']
        min_trend_strength = self.strategy_params['min_trend_strength']

        owned_codes = set(self.broker.get_current_positions().keys())
        unfilled_codes = self.broker.get_unfilled_stock_codes()
        universe_to_analyze = [code for code in universe if code not in owned_codes and code not in unfilled_codes and not code.startswith('U')]
        
        for stock_code in universe_to_analyze:
            if stock_code.startswith('U'): continue

            historical_data = self._get_historical_data_up_to(
                'daily', stock_code, current_date, lookback_period=trend_ma_period + 5
            )
            if len(historical_data) < trend_ma_period + 1: continue

            current_price = historical_data['close'].iloc[-1]
            # --- 1단계: 장기 추세 확인 (이동평균) ---
            trend_ma = calculate_sma(historical_data['close'], trend_ma_period).iloc[-1]
            trend_strength = (current_price - trend_ma) / trend_ma
            # 추세가 상승이고 최소 강도를 만족하는가?
            if trend_strength < min_trend_strength: continue
            # --- 2단계: 중기 모멘텀 확인 (RSI) ---
            rsi = calculate_rsi(historical_data['close'], rsi_period).iloc[-1]
            # 상승 추세에서는 주가가 조정을 받을 때(RSI 과매도 근처)가 매수 기회
            if rsi > rsi_oversold: continue

            # --- 3단계: 단기 진입점 설정 및 신호 생성 ---
            target_price = historical_data['high'].iloc[-2]
            score = 100 - rsi
            buy_scores[stock_code] = score
            
            # [신규] 통합된 속성 딕셔너리에 정보 저장
            signal_attributes[stock_code] = {
                'score': score,
                'target_price': target_price,
                'execution_type': 'breakout' # 전일 고가 돌파는 'breakout' 전술
            }
        
        # 최종 매수 후보 선정 (점수 기반)
        buy_candidates = {code for code, score in buy_scores.items()}

        # 매도 후보 선정
        sell_candidates = set()
        for code in owned_codes:
            if code in unfilled_codes: continue # 미체결 매도 주문이 있으면 제외
            hist_data = self._get_historical_data_up_to('daily', code, current_date, lookback_period=trend_ma_period + 1)
            if len(hist_data) < trend_ma_period + 1: continue
                
            current_price = hist_data['close'].iloc[-1]
            trend_ma = calculate_sma(hist_data['close'], trend_ma_period).iloc[-1]
            
            if current_price < trend_ma:
                sell_candidates.add(code)
                # [신규] 매도 신호에 대한 속성도 추가
                signal_attributes[code] = {
                    'score': 0, # 매도 점수는 별도 관리 안하므로 0
                    'target_price': None, # 즉시 매도이므로 목표가 없음
                    'execution_type': 'market' # 추세 이탈은 'market' 전술로 즉시 매도
                }
                
        return buy_candidates, sell_candidates, signal_attributes