"""
볼린저 밴드 + RSI 결합 일봉 전략
변동성과 모멘텀을 동시에 활용하는 전략
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from strategies.strategy import DailyStrategy
from trade.backtest import Backtest

logger = logging.getLogger(__name__)

class BollingerRSIDaily(DailyStrategy):
    """
    볼린저 밴드 + RSI 결합 전략
    - 볼린저 밴드: 변동성 기반 매매 시점 포착
    - RSI: 과매수/과매도 구간 판단
    - 거래량: 신호의 신뢰도 검증
    """
    
    def __init__(self, trade:Backtest, strategy_params: Dict[str, Any]):
        super().__init__(trade, strategy_params)
        self.broker = trade.broker
        self.data_store = trade.data_store
        # 전략 파라미터 검증
        self.strategy_params = None
        self._validate_parameters()
        # self.signals 초기화
        self.signals = {}
        self._initialize_signals_for_all_stocks()    
        
    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = [
            'bb_period', 'bb_std', 'rsi_period', 'rsi_oversold', 'rsi_overbought',
            'volume_ma_period', 'num_top_stocks', 'safe_asset_code'
        ]
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        
        logger.info(f"볼린저+RSI 전략 파라미터 검증 완료: "
                   f"BB기간={self.strategy_params['bb_period']}일, "
                   f"BB표준편차={self.strategy_params['bb_std']}, "
                   f"RSI기간={self.strategy_params['rsi_period']}일")
    
    def run_daily_logic(self, current_date: datetime.date):
        """
        볼린저+RSI 전략 실행
        수정: 전일 데이터까지만 사용하여 장전 판단이 가능하도록 함
        """
        try:
            logger.info(f'[{current_date.isoformat()}] --- 볼린저+RSI 전략 실행 중 (전일 데이터 기준) ---')
            
            # 수정: 전일 영업일 계산
            prev_trading_day = None
            for stock_code in self.data_store['daily']:
                df = self.data_store['daily'][stock_code]
                if not df.empty and current_date in df.index.date:
                    idx = list(df.index.date).index(current_date)
                    if idx > 0:
                        prev_trading_day = df.index.date[idx-1]
                        break

            # 수정: 전일 데이터가 없으면 실행하지 않음
            if prev_trading_day is None:
                logger.warning(f'{current_date}: 전일 데이터를 찾을 수 없어 볼린저+RSI 전략을 건너뜁니다.')
                return
            
            # 1. 모든 종목 신호 계산 (전일 데이터 기준)
            signals = {}
            for stock_code in self.data_store['daily']:
                if stock_code == self.strategy_params['safe_asset_code']:
                    continue
                # 수정: prev_trading_day 사용
                signal = self._calculate_stock_signal(stock_code, prev_trading_day)
                if signal and signal['signal'] == 'buy':
                    signals[stock_code] = signal
            # 2. 상위 신호만 추출 (점수 기준 정렬)
            sorted_signals = sorted(signals.items(), key=lambda x: x[1]['score'], reverse=True)
            buy_candidates = set()
            for i, (stock_code, signal) in enumerate(sorted_signals):
                if i < self.strategy_params['num_top_stocks']:
                    buy_candidates.add(stock_code)
            sorted_stocks = [(code, signals[code]['score']) for code in buy_candidates]
            if not buy_candidates:
                logger.warning('매수 후보 종목이 없습니다.')
                return
            # 3. 신호 생성 및 업데이트 (부모 클래스 메서드 사용) - 전일 데이터 기준
            current_positions = self._generate_signals(prev_trading_day, buy_candidates, sorted_stocks)
            # 4. 리밸런싱 계획 요약 로깅 (전일 데이터 기준)
            self._log_rebalancing_summary(prev_trading_day, buy_candidates, current_positions)
        except Exception as e:
            logger.error(f"볼린저+RSI 전략 실행 중 오류: {str(e)}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
    
    def _calculate_stock_signal(self, stock_code: str, current_date: datetime.date) -> Dict[str, Any]:
        """개별 종목 신호 계산"""
        df = self.data_store['daily'][stock_code]
        if df.empty or len(df) < max(self.strategy_params['bb_period'], self.strategy_params['rsi_period']):
            return None
        
        # 최근 데이터 추출
        recent_df = df[df.index.date <= current_date].tail(100)
        if len(recent_df) < 50:
            return None
        
        # 기술적 지표 계산
        bb_data = self._calculate_bollinger_bands(recent_df)
        rsi_data = self._calculate_rsi(recent_df)
        volume_data = self._calculate_volume_indicators(recent_df)
        
        if bb_data is None or rsi_data is None or volume_data is None:
            return None
        
        # 신호 점수 계산
        signal_score = self._calculate_signal_score(bb_data, rsi_data, volume_data)
        
        if signal_score > 0.5:  # 매수 신호
            return {
                'signal': 'buy',
                'score': signal_score,
                'price': recent_df.iloc[-1]['close'],
                'signal_date': current_date,
                'strategy': self.strategy_name,
                'indicators': {
                    'bb_position': bb_data['position'],
                    'rsi': rsi_data['current'],
                    'volume_ratio': volume_data['ratio']
                }
            }
        elif signal_score < -0.5:  # 매도 신호
            return {
                'signal': 'sell',
                'score': signal_score,
                'price': recent_df.iloc[-1]['close'],
                'signal_date': current_date,
                'strategy': self.strategy_name,
                'indicators': {
                    'bb_position': bb_data['position'],
                    'rsi': rsi_data['current'],
                    'volume_ratio': volume_data['ratio']
                }
            }
        
        return None
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, float]:
        """볼린저 밴드 계산"""
        try:
            bb_period = self.strategy_params['bb_period']
            bb_std = self.strategy_params['bb_std']
            
            # 이동평균 계산
            ma = df['close'].rolling(window=bb_period).mean()
            
            # 표준편차 계산
            std = df['close'].rolling(window=bb_period).std()
            
            # 볼린저 밴드 계산
            upper_band = ma + (std * bb_std)
            lower_band = ma - (std * bb_std)
            
            # 현재 위치 계산 (0~1, 0=하단밴드, 1=상단밴드)
            current_price = df.iloc[-1]['close']
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            if current_upper == current_lower:
                position = 0.5
            else:
                position = (current_price - current_lower) / (current_upper - current_lower)
                position = max(0, min(1, position))  # 0~1 범위로 제한
            
            return {
                'position': position,
                'upper': current_upper,
                'lower': current_lower,
                'middle': ma.iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"볼린저 밴드 계산 오류: {str(e)}")
            return None
    
    def _calculate_rsi(self, df: pd.DataFrame) -> Dict[str, float]:
        """RSI 계산"""
        try:
            rsi_period = self.strategy_params['rsi_period']
            
            # 가격 변화 계산
            delta = df['close'].diff()
            
            # 상승/하락 분리
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # 평균 계산
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            
            # RSI 계산
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            return {
                'current': current_rsi,
                'oversold': self.strategy_params['rsi_oversold'],
                'overbought': self.strategy_params['rsi_overbought']
            }
            
        except Exception as e:
            logger.error(f"RSI 계산 오류: {str(e)}")
            return None
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """거래량 지표 계산"""
        try:
            volume_ma_period = self.strategy_params['volume_ma_period']
            
            # 거래량 이동평균
            volume_ma = df['volume'].rolling(window=volume_ma_period).mean()
            
            # 현재 거래량 비율
            current_volume = df.iloc[-1]['volume']
            avg_volume = volume_ma.iloc[-1]
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
            else:
                volume_ratio = 1.0
            
            return {
                'ratio': volume_ratio,
                'current': current_volume,
                'average': avg_volume
            }
            
        except Exception as e:
            logger.error(f"거래량 지표 계산 오류: {str(e)}")
            return None
    
    def _calculate_signal_score(self, bb_data: Dict, rsi_data: Dict, volume_data: Dict) -> float:
        """신호 점수 계산"""
        score = 0.0
        
        # 볼린저 밴드 점수 (-1 ~ 1)
        bb_position = bb_data['position']
        if bb_position < 0.2:  # 하단 밴드 근처 (매수 신호)
            bb_score = (0.2 - bb_position) * 5  # 0 ~ 1
        elif bb_position > 0.8:  # 상단 밴드 근처 (매도 신호)
            bb_score = (bb_position - 0.8) * -5  # -1 ~ 0
        else:
            bb_score = 0
        
        # RSI 점수 (-1 ~ 1)
        current_rsi = rsi_data['current']
        oversold = rsi_data['oversold']
        overbought = rsi_data['overbought']
        
        if current_rsi < oversold:  # 과매도 (매수 신호)
            rsi_score = (oversold - current_rsi) / oversold
        elif current_rsi > overbought:  # 과매수 (매도 신호)
            rsi_score = (current_rsi - overbought) / (100 - overbought) * -1
        else:
            rsi_score = 0
        
        # 거래량 가중치 (0.5 ~ 1.5)
        volume_weight = min(volume_data['ratio'] / 2, 1.5)
        volume_weight = max(volume_weight, 0.5)
        
        # 최종 점수 계산
        score = (bb_score * 0.6 + rsi_score * 0.4) * volume_weight
        
        return score
    
    def _get_current_price(self, stock_code: str, current_date: datetime.date) -> float:
        """현재가 조회"""
        if stock_code not in self.data_store['daily']:
            return 0.0
        
        df = self.data_store['daily'][stock_code]
        if df.empty:
            return 0.0
        
        # 해당 날짜의 종가 반환
        date_data = df[df.index.date == current_date]
        if not date_data.empty:
            return date_data.iloc[-1]['close']
        
        # 해당 날짜가 없으면 최근 종가 반환
        recent_data = df[df.index.date <= current_date]
        if not recent_data.empty:
            return recent_data.iloc[-1]['close']
        
        return 0.0 