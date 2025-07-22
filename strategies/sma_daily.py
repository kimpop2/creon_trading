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
        
    def _calculate_momentum_and_target_prices(self, universe: List[str], current_date: datetime.date) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        buy_scores = {}
        sell_scores = {}
        stock_target_prices = {}
        short_sma_period = self.strategy_params['short_sma_period']
        long_sma_period = self.strategy_params['long_sma_period']
        volume_lookback_days = self.strategy_params['volume_lookback_days']
        range_coefficient = self.strategy_params.get('range_coefficient', 0.5) # K값 파라미터 추가

        for stock_code in universe:
            if stock_code.startswith('U'): continue

            historical_data = self._get_historical_data_up_to('daily', stock_code, current_date, lookback_period=long_sma_period + volume_lookback_days + 2)
            if len(historical_data) < long_sma_period + 1:
                continue

            # --- [핵심 수정] ---
            # `datetime.now()` 대신, 데이터의 마지막 시간을 기준으로 현재 시각을 추론합니다.
            # 이렇게 하면 백테스트와 실시간 거래 모두에서 올바르게 동작합니다.
            current_timestamp = historical_data.index[-1]
            current_time = current_timestamp.time()
            # --- 수정 끝 ---

            today_cumulative_volume = historical_data['volume'].iloc[-1]

            historical_volumes = []
            for i in range(1, volume_lookback_days + 1):
                # 데이터가 충분한지 확인
                if len(historical_data.index) > 1 + i:
                    past_date = historical_data.index[-1-i].date()
                    past_volume = self._get_cumulative_volume_at_time(stock_code, past_date, current_time)
                    if past_volume > 0:
                        historical_volumes.append(past_volume)

            yesterday_high = historical_data['high'].iloc[-2]
            yesterday_low = historical_data['low'].iloc[-2]
            today_open = historical_data['open'].iloc[-1]
            
            # 변동성 돌파 전략의 매수 목표가 계산
            price_range = yesterday_high - yesterday_low
            target_price = today_open + (price_range * range_coefficient)
            stock_target_prices[stock_code] = target_price
            
            expected_cumulative_volume = sum(historical_volumes) / len(historical_volumes) if historical_volumes else 0
            short_sma = calculate_sma(historical_data['close'], short_sma_period).iloc[-1]
            long_sma = calculate_sma(historical_data['close'], long_sma_period).iloc[-1]
            prev_short_sma = calculate_sma(historical_data['close'].iloc[:-1], short_sma_period).iloc[-1]
            prev_long_sma = calculate_sma(historical_data['close'].iloc[:-1], long_sma_period).iloc[-1]

            if short_sma > long_sma and prev_short_sma <= prev_long_sma and today_cumulative_volume > expected_cumulative_volume :
                score = (short_sma - long_sma) / long_sma * 100
                buy_scores[stock_code] = score
                logger.info(f"[{stock_code}] 매수 신호 발생. 현재 거래량({today_cumulative_volume:,.0f}) > 최근{len(historical_volumes)}일 평균({expected_cumulative_volume:,.0f})")
            else:
                reason = []
                if short_sma <= long_sma: reason.append(f"단기SMA({short_sma:,.0f}) <= 장기SMA({long_sma:,.0f})")
                if prev_short_sma > prev_long_sma: reason.append("전일 이미 골든크로스 상태")
                if today_cumulative_volume <= expected_cumulative_volume : reason.append(f"거래량({today_cumulative_volume:,.0f}) <= 기대치({expected_cumulative_volume:,.0f})")
                logger.debug(f"[{stock_code}] 매수 신호 미발생. 이유: {', '.join(reason)}")

            if short_sma < long_sma and prev_short_sma >= prev_long_sma and today_cumulative_volume > expected_cumulative_volume :
                score = (long_sma - short_sma) / long_sma * 100
                sell_scores[stock_code] = score

        # --- [수정] 시장 장세 필터링 ---
        if self.strategy_params.get('market_sma_period'): # [수정] .get()으로 안전하게 접근
            market_index_code = self.strategy_params['market_index_code']
            market_sma_period = self.strategy_params['market_sma_period']

            market_data = self._get_historical_data_up_to('daily', market_index_code, current_date, lookback_period=market_sma_period + 1)
            
            if not market_data.empty and len(market_data) >= market_sma_period:
                market_sma = calculate_sma(market_data['close'], period=market_sma_period).iloc[-1]
                current_market_price = market_data['close'].iloc[-1]
                # [디버깅 로그 추가] 실제 계산 값을 확인합니다.
                logger.info(f"[디버깅] 시장({market_index_code}) 실시간 가격: {current_market_price:,.2f} | 계산된 {market_sma_period}일 SMA: {market_sma:,.2f}")

                if current_market_price < market_sma:
                    logger.info(f"[{current_date}] 시장({market_index_code})이 약세장({market_sma_period}일 SMA 하회)으로 판단되어, 신규 매수 신호를 제한합니다.")
                    ########### buy_scores.clear() 
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
        # [핵심 수정] 로직 실행 전, 이전 신호를 모두 초기화합니다.
        self._reset_all_signals()

        universe = list(self.data_store['daily'].keys())
        if not universe:
            logger.warning("거래할 유니버스 종목이 없습니다.")
            return
        # 1. 현재 보유 중인 종목과 미체결 주문이 있는 종목을 합쳐 '제외 목록'을 만듭니다.
        owned_codes = set(self.broker.get_current_positions().keys())
        unfilled_codes = self.broker.get_unfilled_stock_codes()
        stocks_to_exclude = owned_codes | unfilled_codes
        
        # 2. 전체 유니버스에서 제외 목록에 있는 종목을 뺀 '순수 분석 대상'을 만듭니다.
        universe_to_analyze = [code for code in universe if code not in stocks_to_exclude]
        
        if stocks_to_exclude:
            logger.info(f"분석 제외 종목 (보유/미체결): {list(stocks_to_exclude)}")
        
        # 1. SMA 신호 점수 계산, current_date 인자값은 백테스트에서는 전일, 자동매매에서는 당일 임
        buy_scores, sell_scores, stock_target_prices = self._calculate_momentum_and_target_prices(universe_to_analyze, current_date)
        
        # 2. 매수 후보 종목 선정
        sorted_buy_stocks = sorted(buy_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = {
            stock_code for rank, (stock_code, score) in enumerate(sorted_buy_stocks, 1)
            if rank <= self.strategy_params['num_top_stocks']
        }
        
        # 3. 매도 후보 종목 선정 (보유 중인데 매수 후보에 없는 종목은 일정 기간(3일) 이상 홀딩 후에만 매도 후보로 추가)
        # [수정] current_positions 대신 owned_codes 사용
        sell_candidates = set()
        
        min_holding_days = self.strategy_params.get('min_holding_days', 3)
        
        # --- [핵심 수정] ---
        # 매도 후보는 '현재 보유 중'이고 '미체결 매도 주문이 없는' 종목 중에서만 찾습니다.
        sellable_positions = owned_codes - unfilled_codes

        for stock_code in sellable_positions:
            # 데드크로스 조건 충족 시
            if stock_code in sell_scores:
                sell_candidates.add(stock_code)
                logging.info(f"데드크로스 매도 후보 추가: {stock_code}")
            
            # 리밸런싱 조건 충족 시
            elif stock_code not in buy_candidates:
                position_info = self.broker.positions.get(stock_code, {})
                entry_date = position_info.get('entry_date') # [수정] .get()으로 안전하게 접근
                holding_days = (current_date - entry_date).days if entry_date else 0
                if holding_days >= min_holding_days:
                    sell_candidates.add(stock_code)
                    logging.info(f"리밸런싱 매도 후보 추가: {stock_code}")

        logging.info(f"매도 후보 최종 선정: {sorted(list(sell_candidates))}") # [수정] list로 변환        
        # --- [핵심 수정] 리밸런싱 매도 대상의 목표가(target_price)를 제거하는 로직 추가 ---
        for stock_code in list(sell_candidates): # 복사본으로 순회
            # 매도 후보이지만, 데드크로스 점수(sell_scores)는 없는 경우 = 리밸런싱 매도
            if stock_code not in sell_scores:
                if stock_code in stock_target_prices:
                    del stock_target_prices[stock_code] # 목표가 딕셔너리에서 해당 종목 제거
                    logging.info(f"리밸런싱 매도 대상 {stock_code}의 target_price를 제거합니다.")
        # --- 수정 끝 ---
        final_positions = self._generate_signals(current_date, buy_candidates, sorted_buy_stocks, stock_target_prices, sell_candidates)
        
        self._log_rebalancing_summary(current_date, buy_candidates, final_positions, sell_candidates)