"""
파일명: sma_daily_strategy.py
설명: 이동평균선 기반 일봉 매매 전략
작성일: 2024-03-19
수정일: 2024-03-20 (디버그 로깅 추가)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from .base import DailyStrategy

class SMADailyStrategy(DailyStrategy):
    """이동평균선 기반 일봉 매매 전략 클래스"""
    
    def __init__(self, name: str = "sma_daily", params: Dict = None):
        """
        Args:
            name: 전략 이름
            params: 전략 파라미터
        """
        default_params = {
            'short_window': 5,     # 단기 이동평균선 기간
            'long_window': 20,     # 장기 이동평균선 기간
            'lookback_window': 10, # 교차 확인 기간
            'trend_threshold': 0.001,  # 추세 확인을 위한 최소 변화율
            'stop_loss': -0.05,    # 손절매 기준
            'trailing_stop': -0.03, # 트레일링 스탑 기준
            'take_profit': 0.1     # 익절 기준
        }
        if params:
            default_params.update(params)
        super().__init__(name, default_params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """지표 계산
        
        Args:
            data: OHLCV 데이터프레임
            
        Returns:
            pd.DataFrame: 지표가 추가된 데이터프레임
        """
        # 데이터프레임 복사본 생성
        result = data.copy()
        
        # 이동평균선
        result.loc[:, 'SMA_short'] = result['close'].rolling(window=self.params['short_window']).mean()
        result.loc[:, 'SMA_long'] = result['close'].rolling(window=self.params['long_window']).mean()
        
        # 추세 확인을 위한 변화율
        result.loc[:, 'price_change'] = result['close'].pct_change()
        result.loc[:, 'ma_trend'] = result['SMA_long'].pct_change(self.params['lookback_window'])
        
        return result

    def generate_signals(self, data: pd.DataFrame, additional_data: Optional[Dict] = None) -> Dict:
        """매매 신호 생성
        
        Args:
            data: OHLCV 데이터프레임
            additional_data: 추가 데이터 (옵션)
            
        Returns:
            Dict: 생성된 매매 신호 딕셔너리
        """
        if len(data) < self.params['long_window']:
            logging.info(f"데이터 부족: 필요한 길이 {self.params['long_window']}, 현재 길이 {len(data)}")
            return {'signal': 'hold', 'reason': 'not_enough_data'}
            
        # 지표 계산
        data = self.calculate_indicators(data)
        
        # 현재 값
        current_price = data['close'].iloc[-1]
        price_change = data['price_change'].iloc[-1]
        ma_trend = data['ma_trend'].iloc[-1]
        
        # 이동평균선 골든/데드 크로스 확인
        sma_short_prev = data['SMA_short'].iloc[-2]
        sma_long_prev = data['SMA_long'].iloc[-2]
        sma_short_curr = data['SMA_short'].iloc[-1]
        sma_long_curr = data['SMA_long'].iloc[-1]
        
        # 현재 상태 로깅
        logging.info(f"\n=== 일봉 전략 상태 ===")
        logging.info(f"현재가: {current_price:,.0f}")
        logging.info(f"단기 이평선: {sma_short_curr:,.2f} (이전: {sma_short_prev:,.2f})")
        logging.info(f"장기 이평선: {sma_long_curr:,.2f} (이전: {sma_long_prev:,.2f})")
        logging.info(f"MA 추세: {ma_trend*100:.2f}%")
        
        # 매매 신호 생성
        signal = 'hold'
        reason = 'no_signal'
        
        # 골든 크로스 (매수 신호)
        golden_cross = sma_short_prev <= sma_long_prev and sma_short_curr > sma_long_curr
        trend_condition = ma_trend > self.params['trend_threshold']
        
        if golden_cross:
            logging.info("골든 크로스 발생")
            if not trend_condition:
                logging.info(f"추세 조건 미충족 (현재: {ma_trend*100:.2f}%)")
            else:
                signal = 'buy'
                reason = 'golden_cross'
                logging.info("매수 신호 생성")
            
        # 데드 크로스 (매도 신호)
        dead_cross = sma_short_prev >= sma_long_prev and sma_short_curr < sma_long_curr
        if dead_cross:
            logging.info("데드 크로스 발생")
            if not (ma_trend < -self.params['trend_threshold']):
                logging.info(f"추세 조건 미충족 (현재: {ma_trend*100:.2f}%)")
            else:
                signal = 'sell'
                reason = 'dead_cross'
                logging.info("매도 신호 생성")
            
        # 손절매/익절 확인
        if additional_data and 'position_info' in additional_data:
            position = additional_data['position_info']
            if position and position['size'] > 0:
                entry_price = position['avg_price']
                profit_ratio = (current_price - entry_price) / entry_price
                
                logging.info(f"포지션 상태 - 진입가: {entry_price:,.0f}, 수익률: {profit_ratio*100:.2f}%")
                
                # 익절
                if profit_ratio >= self.params['take_profit']:
                    signal = 'sell'
                    reason = 'take_profit'
                    logging.info("익절 신호 생성")
                # 손절매
                elif profit_ratio <= self.params['stop_loss']:
                    signal = 'sell'
                    reason = 'stop_loss'
                    logging.info("손절매 신호 생성")
                # 트레일링 스탑
                elif ('highest_price' in position and
                      (current_price - position['highest_price']) / position['highest_price'] <= self.params['trailing_stop']):
                    signal = 'sell'
                    reason = 'trailing_stop'
                    logging.info("트레일링 스탑 신호 생성")
        
        return {
            'signal': signal,
            'reason': reason,
            'indicators': {
                'sma_short': sma_short_curr,
                'sma_long': sma_long_curr,
                'ma_trend': ma_trend
            }
        } 