# strategies/sma_daily.py
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from strategies.strategy import DailyStrategy
from util.strategies_util import *

logger = logging.getLogger(__name__)

class SMADaily(DailyStrategy):
    """
    SMA(Simple Moving Average) 기반 일봉 전략입니다.
    골든 크로스/데드 크로스와 거래량 조건을 활용하여 매매 신호를 생성합니다.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        # DailyStrategy 에서 trade의 broker, data_store 연결, signal 초기화 진행
        super().__init__(broker, data_store, strategy_params)
        #self.strategy_params = strategy_params
        self._validate_strategy_params() # 전략 파라미터 검증

        # SMA 누적 계산을 위한 캐시 추가
        self.sma_cache = {}  # SMA 캐시
        self.volume_cache = {}  # 거래량 MA 캐시
        self.last_prices = {}  # 마지막 가격 캐시
        self.last_volumes = {}  # 마지막 거래량 캐시

        self.strategy_name = "SMADaily"
        
    def _validate_strategy_params(self):
        """전략 파라미터의 유효성을 검증합니다."""
        required_params = ['short_sma_period', 'long_sma_period', 'volume_ma_period', 'num_top_stocks']
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"SMA 전략에 필요한 파라미터 '{param}'이 설정되지 않았습니다.")
        
        # SMA 기간 검증
        if self.strategy_params['short_sma_period'] >= self.strategy_params['long_sma_period']:
            raise ValueError("단기 SMA 기간은 장기 SMA 기간보다 짧아야 합니다.")
            
        logging.info(f"SMA 전략 파라미터 검증 완료: "
                   f"단기SMA={self.strategy_params['short_sma_period']}, "
                   f"장기SMA={self.strategy_params['long_sma_period']}, "
                   f"거래량MA={self.strategy_params['volume_ma_period']}, "
                   f"선택종목수={self.strategy_params['num_top_stocks']}")

    def _calculate_momentum_and_target_prices(self, universe: List[str], current_date: datetime.date) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        모멘텀 점수와 목표 가격을 계산합니다.
        """
        buy_scores = {}
        sell_scores = {}
        stock_target_prices = {}
        short_sma_period = self.strategy_params['short_sma_period']
        long_sma_period = self.strategy_params['long_sma_period']
        volume_ma_period = self.strategy_params['volume_ma_period']
        
        for stock_code in universe:
            historical_data = self._get_historical_data_up_to('daily', stock_code, current_date, lookback_period=max(long_sma_period, volume_ma_period) + 2)
            if historical_data.empty or len(historical_data) < max(long_sma_period, volume_ma_period) + 1:
                continue

            short_sma = calculate_sma_incremental(historical_data, short_sma_period, self.sma_cache)[0]
            long_sma = calculate_sma_incremental(historical_data, long_sma_period, self.sma_cache)[0]
            prev_short_sma = calculate_sma_incremental(historical_data.iloc[:-1], short_sma_period, self.sma_cache)[0]
            prev_long_sma = calculate_sma_incremental(historical_data.iloc[:-1], long_sma_period, self.sma_cache)[0]
            current_volume = historical_data['volume'].iloc[-1]
            volume_ma = calculate_volume_ma_incremental(historical_data, volume_ma_period, self.volume_cache)[0]
            
            sum_short_prev = prev_short_sma * short_sma_period
            sum_long_prev = prev_long_sma * long_sma_period
            close_oldest_short = historical_data['close'].iloc[-(short_sma_period + 1)]
            close_oldest_long = historical_data['close'].iloc[-(long_sma_period + 1)]
            A = (sum_short_prev - close_oldest_short) / short_sma_period
            B = (sum_long_prev - close_oldest_long) / long_sma_period
            
            target_price = None
            if long_sma_period != short_sma_period:
                target_price = (B - A) * (short_sma_period * long_sma_period) / (long_sma_period - short_sma_period)
            
            stock_target_prices[stock_code] = target_price if target_price is not None else historical_data['close'].iloc[-1]

            if short_sma > long_sma and prev_short_sma <= prev_long_sma and current_volume > volume_ma * 1.0:
                buy_scores[stock_code] = (short_sma - long_sma) / long_sma * 100
            elif short_sma > long_sma and current_volume > volume_ma * 1.2:
                buy_scores[stock_code] = (short_sma - long_sma) / long_sma * 50
            
            if short_sma < long_sma and prev_short_sma >= prev_long_sma and current_volume > volume_ma * 1.0:
                sell_scores[stock_code] = (long_sma - short_sma) / long_sma * 100
                
        return buy_scores, sell_scores, stock_target_prices

    def run_daily_logic(self, current_date: datetime.date):
        logging.info(f"{current_date} - --- SMADaily 일일 로직 실행 ---")
        
        universe = list(self.data_store['daily'].keys())
        if not universe:
            logger.warning("거래할 유니버스 종목이 없습니다.")
            return

        buy_scores, sell_scores, stock_target_prices = self._calculate_momentum_and_target_prices(universe, current_date)
        
        sorted_buy_stocks = sorted(buy_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = {
            stock_code for rank, (stock_code, score) in enumerate(sorted_buy_stocks, 1)
            if rank <= self.strategy_params['num_top_stocks'] and score > 5
        }
        
        current_positions = set(self.broker.get_current_positions().keys())
        sell_candidates = set()
        min_holding_days = self.strategy_params.get('min_holding_days', 3)

        for stock_code in current_positions:
            if stock_code in sell_scores:
                sell_candidates.add(stock_code)
                logging.info(f"데드크로스 매도 후보 추가: {stock_code}")
            elif stock_code not in buy_candidates:
                position_info = self.broker.get_current_positions().get(stock_code, {})
                entry_date = position_info.get('entry_date')
                holding_days = (current_date - entry_date).days if entry_date else 0
                if holding_days >= min_holding_days:
                    sell_candidates.add(stock_code)
                    logging.info(f"매수 후보 제외 및 홀딩 기간 경과로 매도 후보 추가: {stock_code}")

        logging.info(f"매도 후보 최종 선정: {sorted(sell_candidates)}")
        
        # 💡 [수정] 부모 클래스의 강화된 _generate_signals 호출
        final_positions = self._generate_signals(current_date, buy_candidates, sorted_buy_stocks, stock_target_prices, sell_candidates)
        
        self._log_rebalancing_summary(current_date, buy_candidates, final_positions, sell_candidates)