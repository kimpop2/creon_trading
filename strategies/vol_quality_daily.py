# strategies/vol_quality_daily.py

import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import date
import logging
from .strategy import DailyStrategy

logger = logging.getLogger(__name__)

class VolQualityDaily(DailyStrategy):
    """
    저변동성/퀄리티 팩터에 기반하여 유니버스를 필터링하고,
    선별된 종목을 동일 비중으로 매수하는 전략.
    """
    def __init__(self, broker, data_store, strategy_params: Dict[str, Any]):
        super().__init__(broker, data_store, strategy_params)
        self._validate_parameters()

    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = ['vol_quantile','roe_quantile', 'min_trading_value']
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        logger.info(f"저변동성/퀄리티 전략 파라미터 검증 완료.")

    def filter_universe(self, universe_codes: List[str], current_date: date) -> List[str]:
        """
        전략의 핵심 로직. 유니버스에서 저변동성/고퀄리티 종목만 선별합니다.
        """
        try:
            # 1. 파라미터 가져오기
            vol_quantile = self.strategy_params.get('vol_quantile', 0.2) # 변동성 하위 20%
            roe_quantile = self.strategy_params.get('roe_quantile', 0.8) # ROE 상위 20% (높은 값)
            min_trading_value = self.strategy_params.get('min_trading_value', 1_000_000_000) # 최소 거래대금 10억

            # [수정] 2. DB에서 모든 유니버스 종목의 팩터 및 재무 정보를 한 번에 가져옵니다.
            all_factors_df = self.broker.manager.fetch_latest_factors_for_universe(universe_codes, current_date)
            stock_info_df = self.broker.manager.fetch_stock_info(universe_codes)

            if all_factors_df.empty or stock_info_df.empty:
                logger.warning(f"[{self.strategy_name}] {current_date} 기준 조회된 팩터 또는 재무 정보가 없습니다.")
                return []

            # 'roe' 정보를 팩터 데이터프레임에 병합합니다.
            all_factors_df = pd.merge(all_factors_df, stock_info_df[['stock_code', 'roe']], on='stock_code', how='left')

            # 3. 조건에 따라 필터링
            # 최소 거래대금 필터
            filtered_df = all_factors_df[all_factors_df['trading_value'] >= min_trading_value]
            
            # 퀄리티(ROE) 필터
            roe_threshold = filtered_df['roe'].quantile(roe_quantile)
            filtered_df = filtered_df[filtered_df['roe'] >= roe_threshold]

            # 저변동성 필터
            vol_threshold = filtered_df['historical_volatility_20d'].quantile(vol_quantile)
            filtered_df = filtered_df[filtered_df['historical_volatility_20d'] <= vol_threshold]

            final_codes = filtered_df['stock_code'].tolist()
            logger.info(f"[{self.strategy_name}] 최종 필터링된 종목: {final_codes}")
            
            return final_codes

        except Exception as e:
            logger.error(f"[{self.strategy_name}] 유니버스 필터링 중 오류 발생: {e}", exc_info=True)
            return []

    def _calculate_strategy_signals(self, current_date: date, universe: list) -> Tuple[set, set, dict]:
        buy_candidates = set(universe)
        current_positions = set(self.broker.get_current_positions().keys())
        sell_candidates = current_positions - buy_candidates
        
        signal_attributes = {}
        for code in buy_candidates:
            price_df = self._get_historical_data_up_to('daily', code, current_date, lookback_period=1)
            if not price_df.empty:
                # [수정] target_price 추가 (신호 발생일 종가)
                signal_attributes[code] = {
                    'score': 100,
                    'target_price': price_df.iloc[-1]['close']
                }
        
        return buy_candidates, sell_candidates, signal_attributes