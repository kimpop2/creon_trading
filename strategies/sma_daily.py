# strategies/sma_daily.py

import logging
from datetime import datetime
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
                           'market_index_code', 'market_sma_period'] # 시장 장세 필터 파라미터 추가

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
                   f"시장 추세 SMA 기간= {self.strategy_params['market_sma_period']} ")
        
    def _calculate_momentum_and_target_prices(self, universe: List[str], current_date: datetime.date) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        # ... (이전과 동일하게, 목표가는 당일 종가를 사용) ...
        buy_scores = {}
        sell_scores = {}
        stock_target_prices = {}
        short_sma_period = self.strategy_params['short_sma_period']
        long_sma_period = self.strategy_params['long_sma_period']
        volume_ma_period = self.strategy_params['volume_ma_period']
        
        for stock_code in universe:
            if stock_code == self.strategy_params['safe_asset_code']: continue
            
            historical_data = self._get_historical_data_up_to('daily', stock_code, current_date, lookback_period=max(long_sma_period, volume_ma_period) + 2)
            if len(historical_data) < max(long_sma_period, volume_ma_period) + 2:
                continue
            
            # current_date 인자값은 백테스트에서는 전일, 자동매매에서는 당일 임
            # 그러므로 아래 .iloc[-1] 마지막 데이터는 백테스트에서는 전일, 자동매매에서는 오늘의 가격에 해당함
            short_sma = calculate_sma_incremental(historical_data, short_sma_period, self.sma_cache)[0]
            long_sma = calculate_sma_incremental(historical_data, long_sma_period, self.sma_cache)[0]
            prev_short_sma = calculate_sma_incremental(historical_data.iloc[:-1], short_sma_period, self.sma_cache)[0]
            prev_long_sma = calculate_sma_incremental(historical_data.iloc[:-1], long_sma_period, self.sma_cache)[0]
            current_volume = historical_data['volume'].iloc[-1]
            volume_ma = calculate_volume_ma_incremental(historical_data, volume_ma_period, self.volume_cache)[0]

            target_price = historical_data['close'].iloc[-1]
            stock_target_prices[stock_code] = target_price

            # 골든크로스 + 거래량 조건 완화 (1.0배 이상)
            if short_sma > long_sma and prev_short_sma <= prev_long_sma and current_volume > volume_ma * 1.0:
                score = (short_sma - long_sma) / long_sma * 100
                buy_scores[stock_code] = score
            # 추가 매수 조건(강한 상승 완화)
            elif short_sma > long_sma and current_volume > volume_ma * 1.2:
                score = (short_sma - long_sma) / long_sma * 50
                buy_scores[stock_code] = score
            
            # 데드크로스 + 거래량 조건이 모두 충족될 때만 매도 신호
            if short_sma < long_sma and prev_short_sma >= prev_long_sma and current_volume > volume_ma * 1.0:
                score = (long_sma - short_sma) / long_sma * 100
                sell_scores[stock_code] = score
            # 강한 하락(추가 매도)은 제외(신호 완화)

        # 시장 장세 필터링
        if self.strategy_params['market_sma_period']:
            market_index_code = self.strategy_params['market_index_code']
            market_sma_period = self.strategy_params['market_sma_period']

            # 시장 지수 데이터 로드, 약세장에서는 매수 후보를 모두 제거하는 로직
            market_data = self._get_historical_data_up_to('daily', market_index_code, current_date, lookback_period=market_sma_period + 1)
            
            if not market_data.empty and len(market_data) >= market_sma_period:
                market_sma = calculate_sma(market_data['close'], period=market_sma_period).iloc[-1]
                current_market_price = market_data['close'].iloc[-1]

                if current_market_price < market_sma:
                    logger.info(f"[{current_date}] 시장({market_index_code})이 약세장({market_sma_period}일 SMA 하회)으로 판단되어, 신규 매수 신호를 제한합니다.")
                    buy_candidates = set() # 약세장에서는 매수 후보를 모두 제거
                else:
                    logger.info(f"[{current_date}] 시장({market_index_code})이 강세장({market_sma_period}일 SMA 상회)으로 판단됩니다.")
            else:
                logger.warning(f"[{current_date}] 시장 지수({market_index_code}) 데이터 부족 또는 기간 부족으로 시장 장세 판단을 건너뜁니다.")

        return buy_scores, sell_scores, stock_target_prices

    def run_daily_logic(self, current_date: datetime.date):
        """
        [수정됨] 매도 후보 선정 로직을 개선하여, 보유 종목 중에서만 매도 후보를 찾고
        리밸런싱(교체 매매)이 가능하도록 수정했습니다.
        """
        logging.info(f"{current_date} - --- SMADaily 일일 로직 실행 ---")
        
        universe = list(self.data_store['daily'].keys())
        if not universe:
            logger.warning("거래할 유니버스 종목이 없습니다.")
            return
        
        # 1. SMA 신호 점수 계산, current_date 인자값은 백테스트에서는 전일, 자동매매에서는 당일 임
        buy_scores, sell_scores, stock_target_prices = self._calculate_momentum_and_target_prices(universe, current_date)
        
        # 2. 매수 후보 종목 선정
        sorted_buy_stocks = sorted(buy_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = {
            stock_code for rank, (stock_code, score) in enumerate(sorted_buy_stocks, 1)
            if rank <= self.strategy_params['num_top_stocks']
        }
        
        # 3. 매도 후보 종목 선정 (보유 중인데 매수 후보에 없는 종목은 일정 기간(3일) 이상 홀딩 후에만 매도 후보로 추가)
        current_positions = set(self.broker.positions.keys())
        sell_candidates = set()
        
        min_holding_days = self.strategy_params.get('min_holding_days', 3)
        for stock_code in current_positions:
            # 데드크로스+거래량 조건이 충족된 경우
            if stock_code in sell_scores:
                sell_candidates.add(stock_code)
                logging.info(f"데드크로스+거래량 매도 후보 추가: {stock_code}")
            
            # 매수 후보에서 빠진 종목은 일정 기간 홀딩 후 매도 후보
            elif stock_code not in buy_candidates:
                position_info = self.broker.positions.get(stock_code, {})
                entry_date = position_info.get('entry_date', current_date)
                holding_days = (current_date - entry_date).days if entry_date else 0
                if holding_days >= min_holding_days:
                    sell_candidates.add(stock_code)
                    logging.info(f"매수 후보 제외+홀딩기간 경과로 매도 후보 추가: {stock_code}")

        logging.info(f"매도 후보 최종 선정: {sorted(sell_candidates)}")        
        current_positions = set(self.broker.get_current_positions().keys())

        logging.info(f"매도 후보 최종 선정: {sorted(sell_candidates)}")
        
        final_positions = self._generate_signals(current_date, buy_candidates, sorted_buy_stocks, stock_target_prices, sell_candidates)
        
        self._log_rebalancing_summary(current_date, buy_candidates, final_positions, sell_candidates)
