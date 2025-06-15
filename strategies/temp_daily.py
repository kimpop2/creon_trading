# strategies/temp_daily.py

import logging
import pandas as pd
from util.utils import calculate_momentum
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

    def _initialize_signals_for_all_stocks(self): 
        """모든 종목에 대한 시그널을 초기화합니다.""" 
        for stock_code in self.data_store['daily']: 
            if stock_code not in self.signals: 
                self.signals[stock_code] = { 
                    'signal': None, 
                    'signal_date': None, 
                    'traded_today': False, 
                    'target_quantity': 0 
                } 

    def _generate_signals(self, current_daily_date, buy_candidates, sorted_stocks):
        """매수/매도/홀딩 신호를 생성하고 업데이트합니다."""
        current_positions = set(self.broker.positions.keys())

        for stock_code, _ in sorted_stocks:
            # 기본 정보 업데이트
            self.signals[stock_code].update({
                'signal_date': current_daily_date,
                'traded_today': False
            })

            if stock_code in buy_candidates:
                self._handle_buy_candidate(stock_code, current_daily_date, current_positions)
            else:
                self._handle_sell_candidate(stock_code, current_positions)

        return current_positions

    def _handle_buy_candidate(self, stock_code, current_daily_date, current_positions):
        """매수 대상 종목에 대한 신호를 처리합니다."""
        current_price_daily = self._get_historical_data_up_to('daily', stock_code, current_daily_date, lookback_period=1)['close'].iloc[-1]
        target_quantity = super()._calculate_target_quantity(stock_code, current_price_daily)

        if target_quantity > 0:
            if stock_code in current_positions:
                self.signals[stock_code]['signal'] = 'hold'
                logging.info(f'홀딩 신호 - {stock_code}: (기존 보유 종목)')
            else:
                self.signals[stock_code].update({
                    'signal': 'buy',
                    'target_quantity': target_quantity
                })
                logging.info(f'매수 신호 - {stock_code}: 목표수량 {target_quantity}주')

    def _handle_sell_candidate(self, stock_code, current_positions):
        """매도 대상 종목에 대한 신호를 처리합니다."""
        self.signals[stock_code]['signal'] = 'sell'
        if stock_code in current_positions:
            logging.info(f'매도 신호 - {stock_code} (보유중): ')
        else:
            logging.debug(f'매도 신호 - {stock_code} (미보유): ')

    def run_daily_logic(self, current_daily_date): 
        """주간 듀얼 모멘텀 로직을 실행하고 신호를 생성합니다.""" 
        if current_daily_date.weekday() != self.strategy_params['rebalance_weekday']: 
            return 

        logging.info(f'{current_daily_date.isoformat()} - --- 주간 모멘텀 로직 실행 중 ---') 

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

        # 2. 매수 대상 종목 선정
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = set()
        
        for rank, (stock_code, _) in enumerate(sorted_stocks, 1):
            if rank <= self.strategy_params['num_top_stocks']:
                buy_candidates.add(stock_code)

        if not buy_candidates:
            return

        # 3. 신호 생성 및 업데이트
        current_positions = self._generate_signals(current_daily_date, buy_candidates, sorted_stocks)

        # 4. 리밸런싱 계획 요약 로깅
        self._log_rebalancing_summary(current_daily_date, buy_candidates, current_positions)
