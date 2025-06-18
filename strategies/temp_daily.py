# strategies/temp_daily.py

import logging
import pandas as pd
from util.strategies_util import calculate_momentum
from strategies.strategy import DailyStrategy 

class TempletDaily(DailyStrategy):
    """
    새로운 일봉 전략 개발을 위한 템플릿 클래스입니다.
    DualMomentumDaily와 동일한 구조로 DailyStrategy의 run_daily_logic을 구현합니다.
    """
    def __init__(self, data_store, strategy_params, broker): 
        super().__init__(data_store, strategy_params, broker) # BaseStrategy의 __init__ 호출
        self.signals = {} 
        self._initialize_signals_for_all_stocks() 

    def run_daily_logic(self, current_daily_date):
        """일간 듀얼 모멘텀 로직을 실행하고 신호를 생성합니다."""
        # if current_daily_date.weekday() != self.strategy_params['rebalance_weekday']:
        #     return

        logging.info(f'{current_daily_date.isoformat()} - --- 일간 모멘텀 로직 실행 중 ---')

        # signals 초기화 보장
        self._initialize_signals_for_all_stocks()

        # 1. 모멘텀 스코어 계산
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
        
        if not momentum_scores:
            logging.warning('계산된 모멘텀 스코어가 없습니다.')
            return

        # 2. 안전자산 모멘텀 계산
        safe_asset_momentum = self._calculate_safe_asset_momentum(current_daily_date)

        # 3. 매수 대상 종목 선정
        buy_candidates, sorted_stocks = self._select_buy_candidates(momentum_scores, safe_asset_momentum)
        buy_candidates = set()

        for rank, (stock_code, _) in enumerate(sorted_stocks, 1):
            if rank <= self.strategy_params['num_top_stocks']:
                buy_candidates.add(stock_code)

        if not buy_candidates:
            return
        
        # 4. 신호 생성 및 업데이트 (부모 클래스의 _generate_signals 사용)
        current_positions = self._generate_signals(current_daily_date, buy_candidates, sorted_stocks)

        # 5. 리밸런싱 계획 요약 로깅
        self._log_rebalancing_summary(current_daily_date, buy_candidates, current_positions)

        #self.last_rebalance_date = current_daily_date

