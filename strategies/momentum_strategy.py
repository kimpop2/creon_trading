# strategy/momentum_strategy.py
from datetime import date, datetime, timedelta
import pandas as pd
from strategies.strategy import Strategy
from util.strategies_util import *

class MomentumStrategy(Strategy):
    def __init__(self, broker, manager, data_store, strategy_params):
        super().__init__(broker, manager, data_store, strategy_params)
        self.momentum_lookback_period = self.strategy_params.get('momentum_lookback_period', 120) # 120일 (약 6개월)
        self.num_top_stocks = self.strategy_params.get('num_top_stocks', 5)
        self.safe_asset_code = self.strategy_params.get('safe_asset_code', None) # 무위험 자산 코드 (예: 132030 (KOFR금리ETF))
        self.strategy_name = "MomentumStrategy"

    def run_strategy_logic(self, current_date: date) -> None:
        logger.info(f"[{current_date}] 모멘텀 전략 (일봉) 로직 실행 시작")
        momentum_scores = {}
        
        # 1. 절대 모멘텀 계산 (무위험 자산 대비)
        safe_asset_momentum = -float('inf') # 무위험 자산 모멘텀 초기화
        if self.safe_asset_code:
            safe_asset_df = self._get_historical_data_up_to('daily', self.safe_asset_code, current_date, lookback_period=self.momentum_lookback_period)
            if not safe_asset_df.empty and len(safe_asset_df) >= self.momentum_lookback_period:
                safe_asset_start_price = safe_asset_df['close'].iloc[0]
                safe_asset_end_price = safe_asset_df['close'].iloc[-1]
                if safe_asset_start_price > 0:
                    safe_asset_momentum = (safe_asset_end_price - safe_asset_start_price) / safe_asset_start_price
                    logger.debug(f"무위험 자산 ({self.safe_asset_code}) 모멘텀: {safe_asset_momentum:.4f}")
                else:
                    logger.warning(f"무위험 자산 ({self.safe_asset_code}) 시작 가격이 0 이하입니다. 모멘텀 계산 불가.")

        # 2. 개별 종목 모멘텀 계산 및 필터링 (상대 모멘텀)
        for stock_code, daily_df in self.data_store['daily'].items():
            if stock_code == self.safe_asset_code: # 안전자산은 포트폴리오 종목에서 제외
                continue

            df_up_to_current_date = self._get_historical_data_up_to('daily', stock_code, current_date, lookback_period=self.momentum_lookback_period)
            if df_up_to_current_date.empty or len(df_up_to_current_date) < self.momentum_lookback_period:
                logger.debug(f"{stock_code}: 데이터 부족으로 모멘텀 계산 건너뜀.")
                continue

            start_price = df_up_to_current_date['close'].iloc[0]
            end_price = df_up_to_current_date['close'].iloc[-1]

            if start_price > 0:
                momentum = (end_price - start_price) / start_price
                if momentum > safe_asset_momentum: # 절대 모멘텀 기준 충족
                    momentum_scores[stock_code] = momentum
            else:
                logger.warning(f"{stock_code}: 시작 가격이 0 이하입니다. 모멘텀 계산 불가.")

        # 3. 매수 후보 선정
        buy_candidates, sorted_stocks = self._select_buy_candidates(momentum_scores, safe_asset_momentum)
        
        # 4. 매도 후보 선정 (보유 종목 중 모멘텀이 약하거나 순위 밖으로 밀린 경우)
        sell_candidates = set()
        current_positions = set(self.broker.positions.keys())
        
        # 현재 보유 종목이 선정된 매수 후보군에 없으면 매도 후보로 간주
        for stock_code in current_positions:
            if stock_code not in buy_candidates:
                sell_candidates.add(stock_code)
                logger.debug(f"{stock_code}: 매도 신호 (모멘텀 약화 또는 순위 이탈)")

        # self.signals 초기화 및 신호 생성
        self._reset_all_signals()
        self._generate_signals(current_date, buy_candidates, sorted_stocks, sell_candidates)
        self._log_rebalancing_summary(current_date, buy_candidates, current_positions, sell_candidates)
        logger.info(f"[{current_date}] 모멘텀 전략 (일봉) 로직 실행 완료")


    def _select_buy_candidates(self, momentum_scores, safe_asset_momentum):
        """
        모멘텀 스코어를 기반으로 매수 대상 종목을 선정합니다.
        BaseStrategy의 함수를 오버라이드하여 모멘텀 전략에 맞게 수정합니다.
        """
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = set()
        
        for rank, (stock_code, score) in enumerate(sorted_stocks, 1):
            # 상위 num_top_stocks 이내이고, 절대 모멘텀 기준(safe_asset_momentum)을 충족하는 경우
            if rank <= self.num_top_stocks and score > safe_asset_momentum:
                buy_candidates.add(stock_code)
                logger.debug(f"선정된 매수 후보: {stock_code} (모멘텀: {score:.4f}, 순위: {rank})")

        return buy_candidates, sorted_stocks


    def run_trading_logic(self, current_minute_dt: datetime, stock_code: str) -> None:
        signal_info = self.signals.get(stock_code)
        if not signal_info or signal_info.get('traded_today'):
            return

        signal_type = signal_info.get('signal')
        target_quantity = signal_info.get('target_quantity', 0)
        target_price_daily = signal_info.get('target_price', 0) # 일봉 기준 목표가격

        current_bar = self._get_bar_at_time('minute', stock_code, current_minute_dt)
        if not current_bar:
            return
        current_price = current_bar['close']

        # 매수 로직
        if signal_type == 'buy' and target_quantity > 0:
            # 모멘텀 전략은 장 초반에 빠르게 진입하는 경우가 많음 (09:00 ~ 09:15 사이)
            if current_minute_dt.time() >= datetime.strptime("09:00", "%H:%M").time() and \
               current_minute_dt.time() <= datetime.strptime("09:15", "%H:%M").time():
                
                # 전일 종가(target_price_daily) 대비 현재가(current_price)가 과도하게 괴리되지 않았는지 확인
                max_deviation_ratio = self.strategy_params.get('max_deviation_ratio', 0.01) # 기본 1%
                if target_price_daily > 0 and abs(current_price - target_price_daily) / target_price_daily <= max_deviation_ratio:
                    if self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_minute_dt):
                        logger.info(f"[실시간 매매] {stock_code} 매수 완료 (모멘텀): {target_quantity}주 @ {current_price:,.0f}원")
                        self.reset_signal(stock_code)
                else:
                    logger.debug(f"[실시간 매매] {stock_code} 매수 대기 (모멘텀): 현재가 {current_price:,.0f}원, 목표가 {target_price_daily:,.0f}원, 괴리율 {(abs(current_price - target_price_daily) / target_price_daily):.2%}")
            
            # 장 마감 직전 타임컷 매수 (선택 사항, 모멘텀은 일반적으로 장 초반 진입)
            # if current_minute_dt.time() >= datetime.strptime("15:25", "%H:%M").time():
            #     if self.execute_time_cut_buy(stock_code, current_minute_dt, current_price, target_quantity, self.strategy_params.get('max_time_cut_deviation_ratio', 0.02)):
            #         logger.info(f"[실시간 매매] {stock_code} 타임컷 매수 완료 (모멘텀)")

        # 매도 로직
        elif signal_type == 'sell' and stock_code in self.broker.positions:
            current_position_size = self.broker.positions[stock_code]['size']
            if current_position_size <= 0:
                self.reset_signal(stock_code)
                return
            
            # 장 시작 10분 후부터 3시 20분까지
            if current_minute_dt.time() >= datetime.strptime("09:10", "%H:%M").time() and \
               current_minute_dt.time() <= datetime.strptime("15:20", "%H:%M").time():

                # 일봉 신호의 목표가 대비 괴리율 확인 후 매도
                max_deviation_ratio = self.strategy_params.get('max_deviation_ratio', 0.01) # 기본 1%
                if target_price_daily > 0 and abs(current_price - target_price_daily) / target_price_daily <= max_deviation_ratio:
                    if self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_minute_dt):
                        logger.info(f"[실시간 매매] {stock_code} 매도 완료 (모멘텀): {current_position_size}주 @ {current_price:,.0f}원")
                        self.reset_signal(stock_code)
                else:
                    logger.debug(f"[실시간 매매] {stock_code} 매도 대기 (모멘텀): 현재가 {current_price:,.0f}원, 목표가 {target_price_daily:,.0f}원, 괴리율 {(abs(current_price - target_price_daily) / target_price_daily):.2%}")

            # 장 마감 직전 타임컷 매도
            if current_minute_dt.time() >= datetime.strptime("15:25", "%H:%M").time():
                if self.execute_time_cut_sell(stock_code, current_minute_dt, current_price, current_position_size, self.strategy_params.get('max_time_cut_deviation_ratio', 0.02)):
                    logger.info(f"[실시간 매매] {stock_code} 타임컷 매도 완료 (모멘텀)")