# strategies/sma_daily.py

import logging
import pandas as pd
import numpy as np
from util.strategies_util import calculate_sma, calculate_sma_incremental, calculate_volume_ma_incremental
from strategies.strategy import DailyStrategy

class SMADaily(DailyStrategy):
    """
    SMA(Simple Moving Average) 기반 일봉 전략입니다.
    골든 크로스/데드 크로스와 거래량 조건을 활용하여 매매 신호를 생성합니다.
    """
    
    def __init__(self, data_store, strategy_params, broker):
        super().__init__(data_store, strategy_params, broker)
        self.signals = {}
        self._initialize_signals_for_all_stocks() # This method likely populates self.signals initially
        
        # SMA 누적 계산을 위한 캐시 추가
        self.sma_cache = {}  # SMA 캐시
        self.volume_cache = {}  # 거래량 MA 캐시
        self.last_prices = {}  # 마지막 가격 캐시
        self.last_volumes = {}  # 마지막 거래량 캐시
        self.strategy_name = "SMADaily"
        
        # SMA 전략 파라미터 검증
        self._validate_parameters()
        
    def _validate_parameters(self):
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

    def run_daily_logic(self, current_date):
        """
        듀얼 모멘텀 스타일로 리팩터링: SMA 신호 점수 계산 → 상위 N개 종목 선정 → _generate_signals → _log_rebalancing_summary 호출
        수정: 전일 데이터까지만 사용하여 장전 판단이 가능하도록 함
        수정: 매도 신호도 생성하도록 데드크로스 조건 추가
        """
        logging.info(f"{current_date} - --- 일간 SMA 로직 실행 중 (전일 데이터 기준) ---")

        # 0. 전 영업일 계산 (일봉 인덱스에서 today 바로 전 날짜)
        prev_trading_day = None
        for stock_code in self.data_store['daily']:
            df = self.data_store['daily'][stock_code]
            if not df.empty and current_date in df.index.date:
                idx = list(df.index.date).index(current_date)
                if idx > 0:
                    prev_trading_day = df.index.date[idx-1]
                    break
        self.prev_trading_day = prev_trading_day

        # 수정: 전일 데이터가 없으면 실행하지 않음
        if prev_trading_day is None:
            logging.warning(f"{current_date}: 전일 데이터를 찾을 수 없어 SMA 전략을 건너뜁니다.")
            return

        # 1. SMA 신호 점수 계산 (전일 데이터까지만 사용)
        buy_scores = {}  # 매수 점수
        sell_scores = {}  # 매도 점수 (새로 추가)
        all_stock_codes = list(self.data_store['daily'].keys())
        
        for stock_code in all_stock_codes:
            if stock_code == self.strategy_params.get('safe_asset_code'):
                continue
            historical_data = self._get_historical_data_up_to('daily', stock_code, prev_trading_day, lookback_period=max(self.strategy_params['long_sma_period'], self.strategy_params['volume_ma_period']) + 1)
            if historical_data.empty or len(historical_data) < max(self.strategy_params['long_sma_period'], self.strategy_params['volume_ma_period']) + 1:
                continue
                
            short_sma = calculate_sma_incremental(historical_data, self.strategy_params['short_sma_period'], self.sma_cache)[0]
            long_sma = calculate_sma_incremental(historical_data, self.strategy_params['long_sma_period'], self.sma_cache)[0]
            prev_short_sma = calculate_sma_incremental(historical_data.iloc[:-1], self.strategy_params['short_sma_period'], self.sma_cache)[0]
            prev_long_sma = calculate_sma_incremental(historical_data.iloc[:-1], self.strategy_params['long_sma_period'], self.sma_cache)[0]
            current_volume = historical_data['volume'].iloc[-1]
            volume_ma = calculate_volume_ma_incremental(historical_data, self.strategy_params['volume_ma_period'], self.volume_cache)[0]
            
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

        # 2. 매수 후보 종목 선정
        sorted_buy_stocks = sorted(buy_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = set()
        for rank, (stock_code, _) in enumerate(sorted_buy_stocks, 1):
            if rank <= self.strategy_params['num_top_stocks']:
                buy_candidates.add(stock_code)
                
        # 매도 후보 종목 선정 (보유 중인데 매수 후보에 없는 종목은 일정 기간(3일) 이상 홀딩 후에만 매도 후보로 추가)
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
                entry_date = position_info.get('entry_date', prev_trading_day)
                holding_days = (prev_trading_day - entry_date).days if entry_date else 0
                if holding_days >= min_holding_days:
                    sell_candidates.add(stock_code)
                    logging.info(f"매수 후보 제외+홀딩기간 경과로 매도 후보 추가: {stock_code}")

        logging.info(f"매도 후보 최종 선정: {sorted(sell_candidates)}")
        
        # 3. 신호 생성 및 업데이트 (부모 메서드) - 전일 데이터 기준
        # 수정: buy_candidates와 sell_candidates를 모두 전달
        final_positions = self._generate_signals(prev_trading_day, buy_candidates, sorted_buy_stocks, sell_candidates)
        
        # 4. 리밸런싱 요약 로그
        self._log_rebalancing_summary(prev_trading_day, buy_candidates, final_positions, sell_candidates)