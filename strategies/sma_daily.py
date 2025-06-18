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
        self._initialize_signals_for_all_stocks()
        
        # SMA 누적 계산을 위한 캐시 추가
        self.sma_cache = {}  # SMA 캐시
        self.volume_cache = {}  # 거래량 MA 캐시
        self.last_prices = {}  # 마지막 가격 캐시
        self.last_volumes = {}  # 마지막 거래량 캐시
        
        # SMA 전략 파라미터 검증
        self._validate_strategy_params()
        
    def _validate_strategy_params(self):
        """전략 파라미터의 유효성을 검증합니다."""
        required_params = ['short_sma_period', 'long_sma_period', 'volume_ma_period', 'num_top_stocks']
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"SMA 전략에 필요한 파라미터 '{param}'이 설정되지 않았습니다.")
        
        # SMA 기간 검증
        if self.strategy_params['short_sma_period'] >= self.strategy_params['long_sma_period']:
            raise ValueError("단기 SMA 기간은 장기 SMA 기간보다 짧아야 합니다.")
            
        logging.info(f"SMA 전략 파라미터 검증 완료: 단기SMA={self.strategy_params['short_sma_period']}, "
                    f"장기SMA={self.strategy_params['long_sma_period']}, "
                    f"거래량MA={self.strategy_params['volume_ma_period']}, "
                    f"선택종목수={self.strategy_params['num_top_stocks']}")

    def run_daily_logic(self, current_daily_date):
        """일간 SMA 로직을 실행하고 신호를 생성합니다."""
        logging.info(f'{current_daily_date.isoformat()} - --- 일간 SMA 로직 실행 중 ---')

        # signals 초기화 보장
        self._initialize_signals_for_all_stocks()

        # 1. 모든 종목의 SMA 신호 계산
        sma_signals = {}
        for stock_code in self.data_store['daily']:
            if stock_code == self.strategy_params.get('safe_asset_code'):
                continue  # 안전자산은 제외

            daily_df = self.data_store['daily'][stock_code]
            if daily_df.empty:
                continue

            # 충분한 데이터가 있는지 확인
            required_periods = max(self.strategy_params['short_sma_period'], 
                                 self.strategy_params['long_sma_period'],
                                 self.strategy_params['volume_ma_period'])
            
            historical_data = self._get_historical_data_up_to(
                'daily',
                stock_code,
                current_daily_date,
                lookback_period=required_periods + 1
            )

            if len(historical_data) < required_periods:
                logging.debug(f'{stock_code} 종목의 SMA 계산을 위한 데이터가 부족합니다.')
                continue

            # SMA 신호 계산 (누적 계산 방식 사용)
            signal_info = self._calculate_sma_signal_incremental(stock_code, historical_data, current_daily_date)
            if signal_info:
                sma_signals[stock_code] = signal_info

        if not sma_signals:
            logging.warning('계산된 SMA 신호가 없습니다.')
            return

        # 2. 매수/매도 신호 생성
        buy_candidates = []
        sell_candidates = []
        
        for stock_code, signal_info in sma_signals.items():
            if signal_info['signal'] == 'buy':
                buy_candidates.append((stock_code, signal_info['score']))
            elif signal_info['signal'] == 'sell':
                sell_candidates.append((stock_code, signal_info['score']))

        # 3. 매수 후보 종목 선정 (점수 기준 상위 N개)
        buy_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_buy_candidates = set()
        
        for i, (stock_code, score) in enumerate(buy_candidates):
            if i < self.strategy_params['num_top_stocks']:
                selected_buy_candidates.add(stock_code)
                logging.info(f'매수 후보 {i+1}: {stock_code} (점수: {score:.2f})')

        # 4. 신호 생성 및 업데이트 (부모 클래스의 _generate_signals 사용)
        # 부모 클래스의 _generate_signals는 모든 종목에 대해 신호를 생성하므로,
        # SMA 전략에 맞게 수정된 로직을 사용합니다.
        current_positions = self._generate_sma_signals(current_daily_date, selected_buy_candidates, sma_signals)

        # 5. 리밸런싱 계획 요약 로깅
        self._log_rebalancing_summary(current_daily_date, selected_buy_candidates, current_positions)

    def _calculate_sma_signal_incremental(self, stock_code, historical_data, current_date):
        """
        특정 종목의 SMA 신호를 누적 계산 방식으로 계산합니다.
        
        Returns:
            dict: {'signal': 'buy'/'sell'/'hold', 'score': float, 'short_sma': float, 'long_sma': float}
        """
        try:
            # 누적 계산 방식으로 SMA 계산 (util 함수 사용)
            short_sma, self.sma_cache[stock_code] = calculate_sma_incremental(
                historical_data, 
                self.strategy_params['short_sma_period'],
                self.sma_cache.get(stock_code)
            )
            
            long_sma, self.sma_cache[stock_code] = calculate_sma_incremental(
                historical_data, 
                self.strategy_params['long_sma_period'],
                self.sma_cache.get(stock_code)
            )
            
            if short_sma is None or long_sma is None:
                return None
            
            # 누적 계산 방식으로 거래량 이동평균 계산 (util 함수 사용)
            volume_ma, self.volume_cache[stock_code] = calculate_volume_ma_incremental(
                historical_data, 
                self.strategy_params['volume_ma_period'],
                self.volume_cache.get(stock_code)
            )
            
            if volume_ma is None:
                return None
            
            current_price = historical_data['close'].iloc[-1]
            current_volume = historical_data['volume'].iloc[-1]
            
            # 이전 날짜의 SMA (골든 크로스/데드 크로스 확인용)
            if len(historical_data) > 1:
                prev_historical_data = historical_data.iloc[:-1]
                prev_short_sma, _ = calculate_sma_incremental(
                    prev_historical_data, 
                    self.strategy_params['short_sma_period'],
                    self.sma_cache.get(stock_code)
                )
                prev_long_sma, _ = calculate_sma_incremental(
                    prev_historical_data, 
                    self.strategy_params['long_sma_period'],
                    self.sma_cache.get(stock_code)
                )
                
                if prev_short_sma is None or prev_long_sma is None:
                    prev_short_sma = short_sma
                    prev_long_sma = long_sma
            else:
                prev_short_sma = short_sma
                prev_long_sma = long_sma
            
            # 매수/매도 신호 판단
            signal = 'hold'
            score = 0.0
            
            # 골든 크로스 (매수 신호)
            if (short_sma > long_sma and prev_short_sma <= prev_long_sma and 
                current_volume > volume_ma * 1.2):  # 거래량 20% 이상 증가
                signal = 'buy'
                score = (short_sma - long_sma) / long_sma * 100  # SMA 차이 비율
                
            # 데드 크로스 (매도 신호)
            elif (short_sma < long_sma and prev_short_sma >= prev_long_sma):
                signal = 'sell'
                score = (long_sma - short_sma) / long_sma * 100  # SMA 차이 비율
                
            # 추가 매수 조건: 강한 상승 추세
            elif (signal == 'hold' and short_sma > long_sma and 
                  current_price > short_sma and 
                  current_volume > volume_ma * 1.5):  # 거래량 50% 이상 증가
                signal = 'buy'
                score = (current_price - short_sma) / short_sma * 50  # 가격 대비 SMA 차이
                
            return {
                'signal': signal,
                'score': score,
                'short_sma': short_sma,
                'long_sma': long_sma,
                'current_price': current_price,
                'volume_ratio': current_volume / volume_ma
            }
            
        except Exception as e:
            logging.error(f'{stock_code} SMA 신호 계산 중 오류: {str(e)}')
            return None

    def _generate_sma_signals(self, current_daily_date, buy_candidates, sma_signals):
        """SMA 전략에 맞는 매수/매도/홀딩 신호를 생성하고 업데이트합니다."""
        current_positions = set(self.broker.positions.keys())
        
        # 신호를 생성할 종목들: 매수/매도 신호가 있는 종목 + 현재 보유 중인 종목
        stocks_to_process = set()
        
        # 매수/매도 신호가 있는 종목들 추가
        stocks_to_process.update(buy_candidates)
        stocks_to_process.update([
            stock_code for stock_code, signal_info in sma_signals.items() 
            if signal_info['signal'] == 'sell'
        ])
        
        # 현재 보유 중인 종목들 추가 (홀딩 신호 생성용)
        stocks_to_process.update(current_positions)

        # 필요한 종목들에 대해서만 신호 처리
        for stock_code in stocks_to_process:
            if stock_code == self.strategy_params.get('safe_asset_code'):
                continue
                
            # 종목이 signals에 초기화되지 않았다면 초기화
            if stock_code not in self.signals:
                self.signals[stock_code] = {
                    'signal': None,
                    'signal_date': None,
                    'traded_today': False,
                    'target_quantity': 0
                }
            
            # 기본 정보 업데이트
            self.signals[stock_code].update({
                'signal_date': current_daily_date,
                'traded_today': False
            })

            if stock_code in buy_candidates:
                self._handle_buy_candidate(stock_code, current_daily_date, current_positions)
            elif stock_code in sma_signals and sma_signals[stock_code]['signal'] == 'sell':
                self._handle_sell_candidate(stock_code, current_positions)
            elif stock_code in current_positions:
                # 현재 보유 중인 종목이지만 매수/매도 신호가 없는 경우 홀딩
                self.signals[stock_code]['signal'] = 'hold'
                logging.debug(f'홀딩 신호 - {stock_code}: (기존 보유 종목, 추가 신호 없음)')

        return current_positions 