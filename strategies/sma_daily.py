# strategies/sma_daily.py

import logging
from datetime import datetime, time
from typing import Dict, List, Tuple, Any
from strategies.strategy import DailyStrategy
from util.strategies_util import calculate_sma_incremental, calculate_volume_ma_incremental, calculate_sma

logger = logging.getLogger(__name__)

class SMADaily(DailyStrategy):
    """
    [수정됨] SMA(Simple Moving Average) 기반 일봉 전략입니다.
    매도 후보 선정 및 리밸런싱 로직을 개선하여 안정성을 높였습니다.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self._validate_strategy_params()
        self.sma_cache = {}
        self.volume_cache = {}
        self.strategy_name = "SMADaily"
        
    def _validate_strategy_params(self):
        required_params = ['short_sma_period', 'long_sma_period', 'volume_ma_period', # SMA 파라미터
                           'num_top_stocks', 'max_deviation_ratio',
                           'market_index_code', 'market_sma_period',
                           'range_coefficient', 'volume_lookback_days'] 

        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"SMA 전략에 필요한 파라미터 '{param}'이 설정되지 않았습니다.")
        
        # SMA 기간 검증
        if self.strategy_params['short_sma_period'] >= self.strategy_params['long_sma_period']:
            raise ValueError("단기 SMA 기간은 장기 SMA 기간보다 짧아야 합니다.")

        # 최대 진입 가격 괴리율 검증 (0% 이상)
        if not (0 <= self.strategy_params['max_deviation_ratio']):
            raise ValueError("max_deviation_ratio는 0% 이상이어야 합니다.")
        
        # 시장 추세 SMA 기간 검증
        if not (self.strategy_params['market_sma_period'] > 0):
            raise ValueError("market_sma_period는 0보다 커야 합니다.")
        
        logging.info(f"SMA 전략 파라미터 검증 완료: "
                   f"단기 SMA={self.strategy_params['short_sma_period']}, "
                   f"장기 SMA={self.strategy_params['long_sma_period']}, "
                   f"거래량 MA={self.strategy_params['volume_ma_period']}, "
                   f"선택종목 수={self.strategy_params['num_top_stocks']}, "
                   f"진입 가격 괴리율= {self.strategy_params['max_deviation_ratio']}%, "
                   f"시장 지수 코드= {self.strategy_params['market_index_code']}, "
                   f"시장 추세 SMA 기간= {self.strategy_params['market_sma_period']}, "
                   f"변동성 계수(K)= {self.strategy_params['range_coefficient']}, "
                   f"거래량 비교 일수= {self.strategy_params['volume_lookback_days']} ")
        
    # [신규] 동시간대 누적 거래량 계산을 위한 헬퍼 메서드
    def _get_cumulative_volume_at_time(self, stock_code: str, target_date: datetime.date, target_time: datetime.time) -> int:
        """특정 날짜, 특정 시간까지의 누적 거래량을 data_store에서 계산합니다."""
        daily_minute_data = self.data_store.get('minute', {}).get(stock_code, {}).get(target_date)
        
        if daily_minute_data is None or daily_minute_data.empty:
            return 0
        
        # 해당 시간까지의 데이터만 필터링하여 거래량 합산
        volume_sum = daily_minute_data.between_time(time(9, 0), target_time)['volume'].sum()
        return int(volume_sum)
 
    def _calculate_strategy_signals(self, current_date: datetime.date, universe: list) -> Tuple[set, set, list, dict]:
        """
        [수정됨] 신호 속성을 통합된 딕셔너리(signal_attributes)로 반환합니다.
        """
        # --- 1. 유니버스 필터링 ---
        owned_codes = set(self.broker.get_current_positions().keys())
        unfilled_codes = self.broker.get_unfilled_stock_codes()
        stocks_to_exclude = owned_codes | unfilled_codes
        universe_to_analyze = [code for code in universe if code not in stocks_to_exclude and not code.startswith('U')]

        # --- 2. 신호 속성 계산 ---
        buy_scores = {}
        sell_scores = {}
        # [수정] 여러 딕셔너리를 하나로 통합
        signal_attributes = {}
        
        # 파라미터 가져오기
        short_sma_period = self.strategy_params['short_sma_period']
        long_sma_period = self.strategy_params['long_sma_period']
        volume_lookback_days = self.strategy_params['volume_lookback_days']
        range_coefficient = self.strategy_params.get('range_coefficient', 0.5)
        
        for stock_code in universe_to_analyze:

            if stock_code.startswith('U'): continue

            historical_data = self._get_historical_data_up_to('daily', stock_code, current_date, lookback_period=long_sma_period + volume_lookback_days + 2)
            if len(historical_data) < long_sma_period + 1:
                continue

            current_timestamp = historical_data.index[-1]
            current_time = current_timestamp.time()
            today_cumulative_volume = historical_data['volume'].iloc[-1]

            # --- [신규] 모든 계산에 필요한 공통 데이터 준비 ---
            yesterday_high = historical_data['high'].iloc[-2]
            yesterday_low = historical_data['low'].iloc[-2]
            today_open = historical_data['open'].iloc[-1]
            price_range = yesterday_high - yesterday_low
            
            short_sma = calculate_sma(historical_data['close'], short_sma_period).iloc[-1]
            long_sma = calculate_sma(historical_data['close'], long_sma_period).iloc[-1]
            prev_short_sma = calculate_sma(historical_data['close'].iloc[:-1], short_sma_period).iloc[-1]
            prev_long_sma = calculate_sma(historical_data['close'].iloc[:-1], long_sma_period).iloc[-1]

            historical_volumes = [self._get_cumulative_volume_at_time(stock_code, historical_data.index[-1-i].date(), current_time) 
                                  for i in range(1, volume_lookback_days + 1) if len(historical_data.index) > 1 + i]
            historical_volumes = [v for v in historical_volumes if v > 0]
            expected_cumulative_volume = sum(historical_volumes) / len(historical_volumes) if historical_volumes else 0

            # --- 3. 매수/매도 조건 판단 및 signal_attributes 채우기 ---
            is_golden_cross = (short_sma > long_sma and prev_short_sma <= prev_long_sma)
            is_volume_strong = (today_cumulative_volume > expected_cumulative_volume)

            if is_golden_cross and is_volume_strong:
                score = (short_sma - long_sma) / long_sma * 100
                buy_scores[stock_code] = score
                signal_attributes[stock_code] = {
                    'score': score,
                    'target_price': today_open + (price_range * range_coefficient),
                    'execution_type': 'breakout'
                }
                logger.info(f"[{stock_code}] 매수 신호 발생. 현재 거래량({today_cumulative_volume:,.0f}) > 최근{len(historical_volumes)}일 평균({expected_cumulative_volume:,.0f})")

            is_dead_cross = (short_sma < long_sma and prev_short_sma >= prev_long_sma)
            if is_dead_cross and is_volume_strong:
                score = (long_sma - short_sma) / long_sma * 100
                sell_scores[stock_code] = score
                signal_attributes[stock_code] = {
                    'score': -score, # 매도 점수는 음수로 표현
                    'target_price': today_open - (price_range * range_coefficient),
                    'execution_type': 'breakdown' # 하향 돌파
                }
                        
        # --- [수정] 시장 장세 필터링 ---
        if self.strategy_params.get('market_sma_period'): # [수정] .get()으로 안전하게 접근
            market_index_code = self.strategy_params['market_index_code']
            market_sma_period = self.strategy_params['market_sma_period']

            market_data = self._get_historical_data_up_to('daily', market_index_code, current_date, lookback_period=market_sma_period + 1)
            market_sma = calculate_sma(market_data['close'], period=market_sma_period).iloc[-1]
            current_market_price = market_data['close'].iloc[-1]
            # [디버깅 로그 추가] 실제 계산 값을 확인합니다.
            logger.info(f"[디버깅] 시장({market_index_code}) 실시간 가격: {current_market_price:,.2f} | 계산된 {market_sma_period}일 SMA: {market_sma:,.2f}")

            if current_market_price < market_sma:
                logger.info(f"시장 약세로 모든 신규 매수 신호를 제거합니다.")
                buy_scores = {} # 매수 점수 딕셔너리를 비움
            else:
                logger.info(f"[{current_date}] 시장({market_index_code})이 강세장({market_sma_period}일 SMA 상회)으로 판단됩니다.")
        
        # --- 4. 최종 후보 선정 ---
        buy_candidates = {
            code for code, score in buy_scores.items()
            if score > 0 # 점수가 0보다 큰 것만 최종 후보로 인정
        }
        
        # num_top_stocks 필터링은 _generate_signals에서 처리
        sell_candidates = set(sell_scores.keys())
        
        # --- 5. 최종 결과 반환 ---
        return buy_candidates, sell_candidates, signal_attributes
