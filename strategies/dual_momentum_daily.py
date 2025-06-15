import datetime
import logging
import pandas as pd
import numpy as np 

from util.utils import * # utils.py에 있는 모든 함수를 임포트한다고 가정 
from strategies.strategy import DailyStrategy 

class DualMomentumDaily(DailyStrategy): # DailyStrategy 상속 
    def __init__(self, data_store, strategy_params, broker): 
        super().__init__(data_store, strategy_params, broker) # BaseStrategy의 __init__ 호출
        self.signals = {} # {stock_code: {'momentum_score', 'rank', 'signal', 'signal_date', 'traded_today', 'target_amount', 'target_quantity'}} 
        self.last_rebalance_date = None 

        self._initialize_signals_for_all_stocks() 

    def _initialize_signals_for_all_stocks(self): 
        """백테스터에 추가된 모든 종목에 대해 모멘텀 시그널을 초기화합니다.""" 
        for stock_code in self.data_store['daily']: 
            if stock_code not in self.signals: 
                self.signals[stock_code] = { 
                    'momentum_score': 0, 
                    'rank': 0, 
                    'signal': None, 
                    'signal_date': None, 
                    'traded_today': False, 
                    'target_amount': 0, 
                    'target_quantity': 0 
                } 

    # _get_historical_data_up_to 메서드 삭제 (BaseStrategy로 이동)
    # _get_bar_at_time 메서드는 DualMomentumDaily에서 사용하지 않으므로 변경 없음
    # _calculate_target_quantity 메서드 삭제 (DailyStrategy로 이동)

    def run_daily_logic(self, current_daily_date): 
        """ 
        DailyStrategy의 추상 메서드를 구현합니다. 
        기존의 run_weekly_momentum_logic을 호출하여 일봉 전략 로직을 실행합니다. 
        """ 
        """주간 듀얼 모멘텀 로직을 실행하고 신호를 생성합니다.""" 
        if current_daily_date.weekday() != self.strategy_params['rebalance_weekday']: 
            return 

        logging.info(f'{current_daily_date.isoformat()} - --- 주간 모멘텀 로직 실행 중 ---') 

        # 모든 종목의 모멘텀 스코어 계산 
        momentum_scores = {} 
        for stock_code in self.data_store['daily']: 
            if stock_code == self.strategy_params['safe_asset_code']: 
                continue # 안전자산은 모멘텀 계산에서 제외 

            daily_df = self.data_store['daily'][stock_code] 
            if daily_df.empty: 
                continue 

            historical_data = self._get_historical_data_up_to( # BaseStrategy의 메서드 사용
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
            logging.warning('계산된 모멘텀 스코어가 없습니다. 리밸런싱을 건너뜁니다.') 
            return 

        # 모멘텀 스코어로 정렬하여 순위 매기기 
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True) 
        
        # 절대 모멘텀: 안전자산의 모멘텀 계산 
        safe_asset_df = self.data_store['daily'].get(self.strategy_params['safe_asset_code']) 
        safe_asset_momentum = 0 
        if safe_asset_df is not None and not safe_asset_df.empty: 
            safe_asset_data = self._get_historical_data_up_to( # BaseStrategy의 메서드 사용
                'daily', 
                self.strategy_params['safe_asset_code'], 
                current_daily_date, 
                lookback_period=self.strategy_params['momentum_period'] + 1 
            ) 
            if len(safe_asset_data) >= self.strategy_params['momentum_period']: 
                safe_asset_momentum = calculate_momentum(safe_asset_data, self.strategy_params['momentum_period']).iloc[-1] 

        # 현재 보유 종목 확인 
        current_positions = set(self.broker.positions.keys()) 
        
        # 매수 대상 종목 선정 
        buy_candidates = set() 
        for rank, (stock_code, score) in enumerate(sorted_stocks, 1): 
            if rank <= self.strategy_params['num_top_stocks'] and score > safe_asset_momentum: 
                buy_candidates.add(stock_code) 

        # 신호 생성 및 업데이트 
        for rank, (stock_code, score) in enumerate(sorted_stocks, 1): 
            # 먼저 모든 종목에 대해 기본 정보 업데이트 
            self.signals[stock_code].update({ 
                'momentum_score': score, 
                'rank': rank, 
                'signal_date': current_daily_date, 
                'traded_today': False # 매주 리밸런싱 시 신호 생성일에는 traded_today 초기화 
            }) 

            if stock_code in buy_candidates: 
                # --- 여기에서 target_quantity가 0인지 체크하는 로직 추가 --- 
                # 새로 매수할 종목 
                current_price_daily = self._get_historical_data_up_to('daily', stock_code, current_daily_date, lookback_period=1)['close'].iloc[-1] # BaseStrategy의 메서드 사용
                # 부모 클래스의 _calculate_target_quantity 메서드 사용
                target_quantity = super()._calculate_target_quantity(stock_code, current_price_daily)
                if target_quantity > 0: # 매수 수량이 1주 이상일 경우에만 'buy' 신호 생성 
                    if stock_code in current_positions: 
                        # 이미 보유 중인 종목은 홀딩 
                        self.signals[stock_code]['signal'] = 'hold' 
                        logging.info(f'홀딩 신호 - {stock_code}: 순위 {rank}위, 모멘텀 {score:.2f} (기존 보유 종목)') 
                    else: 
                        self.signals[stock_code].update({ 
                            'signal': 'buy', 
                            'target_amount': self.broker.cash / self.strategy_params['num_top_stocks'], 
                            'target_quantity': target_quantity 
                        }) 
                        logging.info(f'매수 신호 - {stock_code}: 순위 {rank}위, 모멘텀 {score:.2f}, 목표수량 {target_quantity}주') 
            else: 
                self.signals[stock_code]['signal'] = 'sell' 
                if stock_code in current_positions: 
                    logging.info(f'매도 신호 - {stock_code} (보유중): 순위 {rank}위, 모멘텀 {score:.2f}') 
                else: 
                    logging.debug(f'매도 신호 - {stock_code} (미보유): 순위 {rank}위, 모멘텀 {score:.2f}') 

        # 리밸런싱 계획 요약 로깅
        self._log_rebalancing_summary(current_daily_date, buy_candidates, current_positions)

        self.last_rebalance_date = current_daily_date 