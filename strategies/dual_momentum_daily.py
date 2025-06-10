# strategies.py

import datetime
import logging
import pandas as pd
import numpy as np

# from util.utils import calculate_momentum, get_next_weekday # 주석 처리 또는 제거 (이미 utils.py에서 임포트)
from util.utils import * # utils.py에 있는 모든 함수를 임포트한다고 가정
# DailyStrategy 추상 클래스 임포트
from strategies.strategy_base import DailyStrategy

class DualMomentumDaily(DailyStrategy): # DailyStrategy 상속
    def __init__(self, data_store, strategy_params, broker):
        self.data_store = data_store # Reference to Backtester's data_store
        self.strategy_params = strategy_params # Reference to Backtester's strategy_params
        self.broker = broker # Reference to Backtester's broker
        self.momentum_signals = {} # {stock_code: {'momentum_score', 'rank', 'signal', 'signal_date', 'traded_today', 'target_amount', 'target_quantity'}}
        self.last_rebalance_date = None

        self._initialize_momentum_signals_for_all_stocks()

    def _initialize_momentum_signals_for_all_stocks(self):
        """백테스터에 추가된 모든 종목에 대해 모멘텀 시그널을 초기화합니다."""
        for stock_code in self.data_store['daily']:
            if stock_code not in self.momentum_signals:
                self.momentum_signals[stock_code] = {
                    'momentum_score': 0,
                    'rank': 0,
                    'signal': None,
                    'signal_date': None,
                    'traded_today': False,
                    'target_amount': 0,
                    'target_quantity': 0
                }

    def _get_historical_data_up_to(self, data_type, stock_code, current_dt, lookback_period=None):
        """주어진 시간(current_dt)까지의 모든 과거 데이터를 반환합니다."""
        if data_type == 'daily':
            df = self.data_store['daily'].get(stock_code)
            if df is None or df.empty:
                return pd.DataFrame()
            current_dt_normalized = pd.Timestamp(current_dt).normalize()
            filtered_df = df.loc[df.index.normalize() <= current_dt_normalized]
            if lookback_period:
                return filtered_df.tail(lookback_period)
            return filtered_df
        
        elif data_type == 'minute':
            all_minute_dfs_for_stock = []
            if stock_code in self.data_store['minute']:
                for date_key in sorted(self.data_store['minute'][stock_code].keys()):
                    if date_key <= current_dt.date():
                        all_minute_dfs_for_stock.append(self.data_store['minute'][stock_code][date_key])
            
            if not all_minute_dfs_for_stock:
                return pd.DataFrame()
            
            combined_minute_df = pd.concat(all_minute_dfs_for_stock).sort_index()
            filtered_df = combined_minute_df.loc[combined_minute_df.index <= current_dt]
            if lookback_period:
                return filtered_df.tail(lookback_period)
            return filtered_df
        return pd.DataFrame()

    # 원본 파일의 _calculate_target_quantity 메서드 복원
    def _calculate_target_quantity(self, stock_code, current_price):
        """주어진 가격에서 동일비중 투자에 필요한 수량을 계산합니다. 원본 로직 그대로."""
        target_amount = self.broker.cash / self.strategy_params['num_top_stocks']
        
        # 현재 보유 현금 확인
        available_cash = self.broker.cash
        
        # 수수료를 고려한 실제 투자 가능 금액 계산
        commission_rate = self.broker.commission_rate
        max_buyable_amount = available_cash / (1 + commission_rate)
        
        # 목표 투자금액과 실제 투자 가능 금액 중 작은 값 선택
        actual_investment_amount = min(target_amount, max_buyable_amount)
        
        # 주식 수량 계산 (소수점 이하 버림)
        quantity = int(actual_investment_amount / current_price)
        
        if quantity > 0:
            # 실제 필요한 현금 (수수료 포함) 계산
            total_cost = current_price * quantity * (1 + commission_rate)
            if total_cost > available_cash:
                # 수량을 1주 줄여서 재계산 (최소 거래 단위 1주이므로)
                quantity -= 1
            
        if quantity > 0:
            logging.info(f"{stock_code} 종목 매수 수량 계산: {quantity}주 (목표금액: {target_amount:,.0f}원, 가용현금: {available_cash:,.0f}원, 현재가: {current_price:,.0f}원)")
        else:
            logging.warning(f"{stock_code} 종목 매수 불가: 현금 부족 (목표금액: {target_amount:,.0f}원, 가용현금: {available_cash:,.0f}원, 현재가: {current_price:,.0f}원)")
        
        return quantity


    def run_weekly_momentum_logic(self, current_daily_date):
        """주간 듀얼 모멘텀 로직을 실행하고 신호를 생성합니다."""
        if current_daily_date.weekday() != self.strategy_params['rebalance_weekday']:
            return

        logging.info(f'{current_daily_date.isoformat()} - --- 주간 모멘텀 로직 실행 중 ---')

        # 모든 종목의 모멘텀 스코어 계산
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
            logging.warning('계산된 모멘텀 스코어가 없습니다. 리밸런싱을 건너뜁니다.')
            return

        # 모멘텀 스코어로 정렬하여 순위 매기기
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 절대 모멘텀: 안전자산의 모멘텀 계산
        safe_asset_df = self.data_store['daily'].get(self.strategy_params['safe_asset_code'])
        safe_asset_momentum = 0
        if safe_asset_df is not None and not safe_asset_df.empty:
            safe_asset_data = self._get_historical_data_up_to(
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
            self.momentum_signals[stock_code].update({
                'momentum_score': score,
                'rank': rank,
                'signal_date': current_daily_date,
                'traded_today': False # 매주 리밸런싱 시 신호 생성일에는 traded_today 초기화
            })

            if stock_code in buy_candidates:
                if stock_code in current_positions:
                    # 이미 보유 중인 종목은 홀딩
                    self.momentum_signals[stock_code]['signal'] = 'hold'
                    logging.info(f'홀딩 신호 - {stock_code}: 순위 {rank}위, 모멘텀 {score:.2f} (기존 보유 종목)')
                else:
                    # 새로 매수할 종목
                    current_price_daily = self._get_historical_data_up_to('daily', stock_code, current_daily_date, lookback_period=1)['close'].iloc[-1]
                    target_quantity = self._calculate_target_quantity(stock_code, current_price_daily)
                    
                    self.momentum_signals[stock_code].update({
                        'signal': 'buy',
                        'target_amount': self.broker.cash / self.strategy_params['num_top_stocks'],
                        'target_quantity': target_quantity
                    })
                    logging.info(f'매수 신호 - {stock_code}: 순위 {rank}위, 모멘텀 {score:.2f}, 목표수량 {target_quantity}주')
            else:
                self.momentum_signals[stock_code]['signal'] = 'sell'
                if stock_code in current_positions:
                    logging.info(f'매도 신호 - {stock_code} (보유중): 순위 {rank}위, 모멘텀 {score:.2f}')
                else:
                    logging.debug(f'매도 신호 - {stock_code} (미보유): 순위 {rank}위, 모멘텀 {score:.2f}')

        # 리밸런싱 계획 요약
        current_prices_for_summary = {}
        for stock_code in self.data_store['daily']:
            daily_data = self._get_historical_data_up_to('daily', stock_code, current_daily_date, lookback_period=1)
            if not daily_data.empty:
                current_prices_for_summary[stock_code] = daily_data['close'].iloc[-1]

        portfolio_value = self.broker.get_portfolio_value(current_prices_for_summary)
        current_holdings = [(code, pos['size'] * current_prices_for_summary[code]) for code, pos in self.broker.positions.items() if code in current_prices_for_summary and pos['size'] > 0]
        total_holdings_value = sum(value for _, value in current_holdings)
        
        # 매수 계획 계산
        new_buys = [(code, self.momentum_signals[code]['target_quantity'] * current_prices_for_summary[code]) 
                    for code in buy_candidates if code not in current_positions and code in current_prices_for_summary]
        total_buy_amount = sum(amount for _, amount in new_buys)
        
        # 매도 계획 계산
        to_sell = [(code, pos['size'] * current_prices_for_summary[code]) 
                   for code, pos in self.broker.positions.items() 
                   if code not in buy_candidates and code in current_prices_for_summary]
        total_sell_amount = sum(amount for _, amount in to_sell)

        logging.info("\n=== 리밸런싱 계획 요약 ===")
        logging.info(f"현재 상태: 포트폴리오 가치 {portfolio_value:,.0f}원 = 보유종목 {len(current_holdings)}개 ({total_holdings_value:,.0f}원) + 현금 {self.broker.cash:,.0f}원")
        logging.info(f"매수 계획: {len(new_buys)}종목 (소요금액: {total_buy_amount:,.0f}원)")
        logging.info(f"매도 계획: {len(to_sell)}종목 (회수금액: {total_sell_amount:,.0f}원)")

        self.last_rebalance_date = current_daily_date

    # DailyStrategy의 추상 메서드인 run_daily_logic을 구현합니다.
    # 기존 run_weekly_momentum_logic의 이름을 그대로 사용하므로, 래퍼 역할을 합니다.
    def run_daily_logic(self, current_daily_date):
        """
        DailyStrategy의 추상 메서드를 구현합니다.
        기존의 run_weekly_momentum_logic을 호출하여 일봉 전략 로직을 실행합니다.
        """
        self.run_weekly_momentum_logic(current_daily_date)

    # DailyStrategy의 추상 메서드이지만 DailyStrategy에서는 필요 없는 메서드 구현
    def run_minute_logic(self, stock_code, current_minute_dt):
        """DailyStrategy는 분봉 로직을 직접 수행하지 않습니다."""
        pass

    # DailyStrategy의 추상 메서드이지만 DailyStrategy에서는 필요 없는 메서드 구현
    def update_momentum_signals(self, momentum_signals):
        """
        DailyStrategy는 외부에서 모멘텀 시그널을 업데이트 받지 않습니다.
        Backtester가 DualMomentumDaily의 momentum_signals를 직접 참조하여 MinuteStrategy로 전달하도록 설계되어야 합니다.
        """
        pass