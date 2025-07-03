"""
알렉산더 엘더의 삼중창 시스템 일봉 전략
3단계 필터링을 통한 신뢰도 높은 매매 신호 생성
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from strategies.strategy import DailyStrategy
from trade.backtest import Backtest
from util.strategies_util import (
    calculate_rsi_incremental, calculate_macd_incremental, 
    initialize_indicator_caches, calculate_volume_ma_incremental_simple
)

logger = logging.getLogger(__name__)

# 전략 상수 정의
class TripleScreenConstants:
    """삼중창 시스템 전략 상수"""
    
    # 1단계 추세 분석 상수
    TREND_LOOKBACK_MULTIPLIER = 3  # 추세 분석용 데이터 기간 배수
    TREND_STRENGTH_MULTIPLIER = 0.8  # 최소 추세 강도 완화 배수
    
    # 2단계 모멘텀 분석 상수
    MOMENTUM_LOOKBACK_MULTIPLIER = 3  # 모멘텀 분석용 데이터 기간 배수
    RSI_OVERSOLD_MULTIPLIER = 0.9  # RSI 과매도 완화 배수
    RSI_OVERBOUGHT_MULTIPLIER = 1.1  # RSI 과매수 완화 배수
    BOLLINGER_UPPER_THRESHOLD = 0.9  # 볼린저 밴드 상단 임계값
    MOMENTUM_SCORE_THRESHOLD = 0.15  # 모멘텀 점수 임계값
    
    # 3단계 진입점 분석 상수
    PATTERN_ANALYSIS_PERIOD = 20  # 패턴 분석용 데이터 기간
    PATTERN_SCORE_THRESHOLD = -0.1  # 패턴 점수 임계값
    VOLUME_RATIO_THRESHOLD = 0.3  # 거래량 비율 임계값
    SUPPORT_DISTANCE_THRESHOLD = 0.05  # 지지선 근처 거리 임계값 (5%)
    
    # 기술적 지표 상수
    STOCHASTIC_PERIOD = 14  # 스토캐스틱 계산 기간
    STOCHASTIC_SIGNAL_PERIOD = 3  # 스토캐스틱 신호선 기간
    BOLLINGER_PERIOD = 20  # 볼린저 밴드 계산 기간
    BOLLINGER_STD_MULTIPLIER = 2  # 볼린저 밴드 표준편차 배수
    
    # MACD 다이버전스 분석 상수
    DIVERGENCE_ANALYSIS_PERIOD = 20  # 다이버전스 분석용 데이터 기간
    DIVERGENCE_WINDOW = 5  # 다이버전스 고점/저점 찾기 윈도우
    DIVERGENCE_COMPARISON_PERIOD = 10  # 다이버전스 비교 기간
    
    # 모멘텀 점수 가중치
    RSI_SCORE_WEIGHT = 0.4  # RSI 점수 가중치
    STOCHASTIC_SCORE_WEIGHT = 0.3  # 스토캐스틱 점수 가중치
    BOLLINGER_SCORE_WEIGHT = 0.3  # 볼린저 밴드 점수 가중치
    MACD_SCORE_WEIGHT = 0.2  # MACD 점수 가중치
    
    # 패턴 분석 상수
    PATTERN_ANALYSIS_DAYS = 5  # 패턴 분석용 일수
    DOUBLE_BOTTOM_TOLERANCE = 0.02  # 더블 바텀 패턴 허용 오차 (2%)
    PATTERN_TREND_SCORE = 0.3  # 상승 패턴 점수
    DOUBLE_BOTTOM_SCORE = 0.4  # 더블 바텀 패턴 점수
    ASCENDING_TRIANGLE_SCORE = 0.3  # 상승 삼각형 패턴 점수
    
    # 지지/저항 분석 상수
    SUPPORT_ANALYSIS_PERIOD = 10  # 지지선 분석용 데이터 기간
    SUPPORT_LEVELS_COUNT = 3  # 분석할 지지선 개수
    
    # 매도 조건 상수
    SELL_ANALYSIS_PERIOD = 50  # 매도 분석용 데이터 기간
    MIN_DATA_FOR_SELL = 20  # 매도 분석 최소 데이터 수
    RSI_OVERBOUGHT_THRESHOLD = 80  # RSI 과매수 임계값
    BOLLINGER_UPPER_SELL_THRESHOLD = 0.95  # 볼린저 밴드 상단 매도 임계값
    MIN_HOLDING_DAYS = 5  # 최소 보유 기간
    PROFIT_TAKE_THRESHOLD = 10  # 익절 수익률 임계값 (%)
    STOP_LOSS_THRESHOLD = -5  # 손절 손실률 임계값 (%)
    
    # 종합 점수 가중치
    TREND_STRENGTH_WEIGHT = 0.4  # 추세 강도 가중치
    MOMENTUM_SCORE_WEIGHT = 0.4  # 모멘텀 점수 가중치
    PATTERN_SCORE_WEIGHT = 0.2  # 패턴 점수 가중치
    
    # 기본값
    DEFAULT_STOCHASTIC_K = 50  # 스토캐스틱 K 기본값
    DEFAULT_STOCHASTIC_D = 50  # 스토캐스틱 D 기본값
    DEFAULT_BOLLINGER_POSITION = 0.5  # 볼린저 밴드 위치 기본값
    DEFAULT_RSI = 50  # RSI 기본값

class TripleScreenDaily(DailyStrategy):
    """
    알렉산더 엘더의 삼중창 시스템
    - 1단계: 장기 추세 확인 (주봉 기준)
    - 2단계: 중기 모멘텀 확인 (일봉 기준)
    - 3단계: 단기 진입점 확인 (일봉 패턴)
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
        self.strategy_name = "TripleScreenDaily"
        
        # 증분 계산을 위한 캐시 초기화
        self.macd_cache = {}  # {stock_code: {'macd': float, 'signal': float, 'histogram': float, 'last_date': date}}
        self.rsi_cache = {}   # {stock_code: {'rsi': float, 'avg_gain': float, 'avg_loss': float, 'last_date': date}}
        self.volume_cache = {} # {stock_code: {'ratio': float, 'current': float, 'average': float, 'last_date': date}}
        
    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = [
            'trend_ma_period', 'momentum_rsi_period', 'momentum_rsi_oversold', 
            'momentum_rsi_overbought', 'volume_ma_period', 'num_top_stocks', 
            'safe_asset_code', 'min_trend_strength'
        ]
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        
        logger.info(f"삼중창 시스템 파라미터 검증 완료: "
                   f"추세기간={self.strategy_params['trend_ma_period']}일, "
                   f"RSI기간={self.strategy_params['momentum_rsi_period']}일")
    
    def _apply_triple_screen(self, stock_code: str, current_date: datetime.date) -> Dict[str, Any]:
        """삼중창 필터링 적용"""
        result = {
            'passed': False,
            'screen1': {'passed': False, 'trend': 0, 'strength': 0},
            'screen2': {'passed': False, 'rsi': 0, 'momentum': 0},
            'screen3': {'passed': False, 'pattern': '', 'volume': 0}
        }
        
        try:
            # 1단계: 장기 추세 확인
            screen1_result = self._screen1_trend_analysis(stock_code, current_date)
            result['screen1'] = screen1_result
            
            if not screen1_result['passed']:
                logger.debug(f"[{current_date}] {stock_code}: 1단계(추세) 실패 - "
                           f"방향: {screen1_result['trend']}, 강도: {screen1_result['strength']:.3f}, "
                           f"MACD시그널: {screen1_result.get('macd', {}).get('signal', 0):.3f}")
                return result
            
            # 2단계: 중기 모멘텀 확인
            screen2_result = self._screen2_momentum_analysis(stock_code, current_date)
            result['screen2'] = screen2_result
            
            if not screen2_result['passed']:
                logger.debug(f"[{current_date}] {stock_code}: 2단계(모멘텀) 실패 - "
                           f"RSI: {screen2_result['rsi']:.1f}, 모멘텀점수: {screen2_result['momentum']:.3f}")
                return result
            
            # 3단계: 단기 진입점 확인
            screen3_result = self._screen3_entry_analysis(stock_code, current_date)
            result['screen3'] = screen3_result
            
            if not screen3_result['passed']:
                logger.debug(f"[{current_date}] {stock_code}: 3단계(진입점) 실패 - "
                           f"패턴: {screen3_result['pattern']}, 점수: {screen3_result['pattern_score']:.3f}, "
                           f"거래량비율: {screen3_result['volume']:.2f}")
                return result
            
            # 모든 단계 통과 시 최종 통과
            result['passed'] = screen3_result['passed']
            
            if result['passed']:
                logger.info(f"[{current_date}] {stock_code}: 삼중창 통과! "
                           f"추세강도: {screen1_result['strength']:.3f}, "
                           f"RSI: {screen2_result['rsi']:.1f}, "
                           f"패턴: {screen3_result['pattern']}")
            
        except Exception as e:
            logger.error(f"삼중창 필터링 중 오류 ({stock_code}): {str(e)}")
        
        return result
    
    def _screen1_trend_analysis(self, stock_code: str, current_date: datetime.date) -> Dict[str, Any]:
        """1단계: 장기 추세 분석 (MACD 활용)"""
        result = {'passed': False, 'trend': 0, 'strength': 0}
        
        # 다른 전략들과 동일한 방식으로 데이터 가져오기
        lookback_period = self.strategy_params['trend_ma_period'] * TripleScreenConstants.TREND_LOOKBACK_MULTIPLIER
        recent_df = self._get_historical_data_up_to(
            'daily', stock_code, current_date, lookback_period=lookback_period
        )
        
        if recent_df.empty or len(recent_df) < self.strategy_params['trend_ma_period']:
            return result
        
        # 이동평균 계산
        ma_period = self.strategy_params['trend_ma_period']
        ma = recent_df['close'].rolling(window=ma_period).mean()
        
        if ma.iloc[-1] is None or pd.isna(ma.iloc[-1]):
            return result
        
        current_price = recent_df.iloc[-1]['close']
        current_ma = ma.iloc[-1]
        
        # 추세 방향 확인
        trend_direction = 1 if current_price > current_ma else -1
        
        # 추세 강도 계산 (가격과 이동평균의 거리)
        trend_strength = abs(current_price - current_ma) / current_ma
        
        # MACD 계산 및 분석
        macd_data = calculate_macd_incremental(
            recent_df, stock_code, 
            self.macd_cache.get(stock_code)
        )
        
        # MACD 기반 추세 확인
        macd_trend = 0
        if macd_data['macd'] > macd_data['signal']:
            macd_trend = 1  # MACD가 신호선 위에 있음 (상승추세)
        elif macd_data['macd'] < macd_data['signal']:
            macd_trend = -1  # MACD가 신호선 아래에 있음 (하락추세)
        
        # MACD 히스토그램 확인 (모멘텀 강도)
        macd_momentum = macd_data['histogram']
        
        # 조건: 이동평균과 MACD 모두 상승추세이고, 최소 강도 만족
        min_strength = self.strategy_params['min_trend_strength']
        
        # 이동평균 조건
        ma_condition = (trend_direction > 0 and trend_strength >= min_strength * TripleScreenConstants.TREND_STRENGTH_MULTIPLIER)
        
        # MACD 조건
        macd_condition = (macd_trend > 0 and macd_momentum > 0)  # MACD 상승추세 + 히스토그램 양수
        
        # 두 조건 모두 만족해야 통과
        passed = ma_condition and macd_condition
        
        result.update({
            'passed': passed,
            'trend': trend_direction,
            'strength': trend_strength,
            'macd': macd_data,
            'macd_trend': macd_trend,
            'macd_momentum': macd_momentum
        })
        
        return result
    
    def _screen2_momentum_analysis(self, stock_code: str, current_date: datetime.date) -> Dict[str, Any]:
        """2단계: 중기 모멘텀 분석 (MACD 다이버전스 활용)"""
        result = {'passed': False, 'rsi': 0, 'momentum': 0}
        
        # 다른 전략들과 동일한 방식으로 데이터 가져오기
        lookback_period = self.strategy_params['momentum_rsi_period'] * TripleScreenConstants.MOMENTUM_LOOKBACK_MULTIPLIER
        recent_df = self._get_historical_data_up_to(
            'daily', stock_code, current_date, lookback_period=lookback_period
        )
        
        if recent_df.empty or len(recent_df) < self.strategy_params['momentum_rsi_period']:
            return result
        
        # RSI 계산
        rsi_data = calculate_rsi_incremental(
            recent_df, self.strategy_params['momentum_rsi_period'], 
            stock_code, self.rsi_cache.get(stock_code)
        )
        current_rsi = rsi_data['current']
        
        # 스토캐스틱 계산
        stoch_data = self._calculate_stochastic(recent_df)
        
        # 볼린저 밴드 계산
        bb_data = self._calculate_bollinger_bands(recent_df)
        
        # MACD 계산 및 다이버전스 분석
        macd_data = calculate_macd_incremental(
            recent_df, stock_code, 
            self.macd_cache.get(stock_code)
        )
        macd_divergence = self._check_macd_divergence(recent_df, macd_data)
        
        # 모멘텀 점수 계산 (MACD 포함)
        momentum_score = self._calculate_momentum_score(rsi_data, stoch_data, bb_data, macd_data)
        
        # 조건: RSI가 과매도에서 반등 중이고, 스토캐스틱이 상승 중 (소폭 완화)
        oversold = self.strategy_params['momentum_rsi_oversold']
        overbought = self.strategy_params['momentum_rsi_overbought']
        
        rsi_condition = (oversold * TripleScreenConstants.RSI_OVERSOLD_MULTIPLIER < 
                        current_rsi < overbought * TripleScreenConstants.RSI_OVERBOUGHT_MULTIPLIER)
        stoch_condition = stoch_data['k'] > stoch_data['d']
        bb_condition = bb_data['position'] < TripleScreenConstants.BOLLINGER_UPPER_THRESHOLD
        
        # MACD 조건 추가
        macd_condition = (macd_data['histogram'] > 0 and  # 히스토그램 양수
                         macd_data['macd'] > macd_data['signal'])  # MACD가 신호선 위
        
        # 다이버전스가 있으면 추가 점수
        divergence_bonus = macd_divergence['bullish']  # 상승 다이버전스
        
        passed = (rsi_condition and stoch_condition and bb_condition and 
                 macd_condition and momentum_score > TripleScreenConstants.MOMENTUM_SCORE_THRESHOLD)
        
        result.update({
            'passed': passed,
            'rsi': current_rsi,
            'momentum': momentum_score,
            'stochastic': stoch_data,
            'bollinger': bb_data,
            'macd': macd_data,
            'macd_divergence': macd_divergence,
            'divergence_bonus': divergence_bonus
        })
        
        return result
    
    def _screen3_entry_analysis(self, stock_code: str, current_date: datetime.date) -> Dict[str, Any]:
        """3단계: 단기 진입점 분석"""
        result = {'passed': False, 'pattern': '', 'volume': 0}
        
        # 다른 전략들과 동일한 방식으로 데이터 가져오기
        lookback_period = TripleScreenConstants.PATTERN_ANALYSIS_PERIOD
        recent_df = self._get_historical_data_up_to(
            'daily', stock_code, current_date, lookback_period=lookback_period
        )
        
        if recent_df.empty or len(recent_df) < TripleScreenConstants.PATTERN_ANALYSIS_PERIOD:
            return result
        
        # 가격 패턴 분석
        pattern = self._analyze_price_pattern(recent_df)
        
        # 거래량 분석
        volume_data = calculate_volume_ma_incremental_simple(
            recent_df, self.strategy_params['volume_ma_period'],
            stock_code, self.volume_cache.get(stock_code)
        )
        
        # 지지/저항 분석
        support_resistance = self._analyze_support_resistance(recent_df)
        
        # 조건: 긍정적 패턴, 거래량 증가, 지지선 근처
        pattern_condition = pattern['score'] > TripleScreenConstants.PATTERN_SCORE_THRESHOLD
        volume_condition = volume_data['ratio'] > TripleScreenConstants.VOLUME_RATIO_THRESHOLD
        support_condition = support_resistance['near_support']
        
        passed = pattern_condition and volume_condition and support_condition
        
        result.update({
            'passed': passed,
            'pattern': pattern['name'],
            'volume': volume_data['ratio'],
            'pattern_score': pattern['score'],
            'support_resistance': support_resistance
        })
        
        return result
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> Dict[str, float]:
        """스토캐스틱 계산"""
        try:
            period = TripleScreenConstants.STOCHASTIC_PERIOD
            
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            
            k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window=TripleScreenConstants.STOCHASTIC_SIGNAL_PERIOD).mean()
            
            return {
                'k': k_percent.iloc[-1],
                'd': d_percent.iloc[-1]
            }
        except:
            return {
                'k': TripleScreenConstants.DEFAULT_STOCHASTIC_K, 
                'd': TripleScreenConstants.DEFAULT_STOCHASTIC_D
            }
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, float]:
        """볼린저 밴드 계산"""
        try:
            period = TripleScreenConstants.BOLLINGER_PERIOD
            std = TripleScreenConstants.BOLLINGER_STD_MULTIPLIER
            
            ma = df['close'].rolling(window=period).mean()
            std_dev = df['close'].rolling(window=period).std()
            
            upper = ma + (std_dev * std)
            lower = ma - (std_dev * std)
            
            current_price = df.iloc[-1]['close']
            position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            
            return {
                'position': position,
                'upper': upper.iloc[-1],
                'lower': lower.iloc[-1],
                'middle': ma.iloc[-1]
            }
        except:
            return {
                'position': TripleScreenConstants.DEFAULT_BOLLINGER_POSITION, 
                'upper': 0, 'lower': 0, 'middle': 0
            }
    
    def _check_macd_divergence(self, df: pd.DataFrame, macd_data: Dict) -> Dict[str, bool]:
        """MACD 다이버전스 확인"""
        try:
            # 최근 데이터로 다이버전스 확인
            recent_data = df.tail(TripleScreenConstants.DIVERGENCE_ANALYSIS_PERIOD)
            
            # 가격의 고점과 저점 찾기
            price_highs = recent_data['high'].rolling(window=TripleScreenConstants.DIVERGENCE_WINDOW, center=True).max()
            price_lows = recent_data['low'].rolling(window=TripleScreenConstants.DIVERGENCE_WINDOW, center=True).min()
            
            # MACD 히스토그램의 고점과 저점 찾기
            macd_histogram = recent_data['close'].ewm(span=12).mean() - recent_data['close'].ewm(span=26).mean()
            macd_histogram = macd_histogram - macd_histogram.ewm(span=9).mean()
            
            macd_highs = macd_histogram.rolling(window=TripleScreenConstants.DIVERGENCE_WINDOW, center=True).max()
            macd_lows = macd_histogram.rolling(window=TripleScreenConstants.DIVERGENCE_WINDOW, center=True).min()
            
            # 상승 다이버전스: 가격은 저점을 높이는데 MACD는 저점을 낮춤
            bullish_divergence = False
            if (price_lows.iloc[-1] > price_lows.iloc[-TripleScreenConstants.DIVERGENCE_COMPARISON_PERIOD] and 
                macd_lows.iloc[-1] < macd_lows.iloc[-TripleScreenConstants.DIVERGENCE_COMPARISON_PERIOD]):
                bullish_divergence = True
            
            # 하락 다이버전스: 가격은 고점을 낮추는데 MACD는 고점을 높임
            bearish_divergence = False
            if (price_highs.iloc[-1] < price_highs.iloc[-TripleScreenConstants.DIVERGENCE_COMPARISON_PERIOD] and 
                macd_highs.iloc[-1] > macd_highs.iloc[-TripleScreenConstants.DIVERGENCE_COMPARISON_PERIOD]):
                bearish_divergence = True
            
            return {
                'bullish': bullish_divergence,
                'bearish': bearish_divergence
            }
        except:
            return {'bullish': False, 'bearish': False}
    
    def _calculate_momentum_score(self, rsi_data: Dict, stoch_data: Dict, bb_data: Dict, macd_data: Dict) -> float:
        """모멘텀 점수 계산"""
        score = 0.0
        
        # RSI 점수 (30-70 구간에서 높은 점수)
        rsi = rsi_data['current']
        if 30 <= rsi <= 70:
            score += TripleScreenConstants.RSI_SCORE_WEIGHT
        elif 25 <= rsi <= 75:
            score += TripleScreenConstants.RSI_SCORE_WEIGHT * 0.5
        
        # 스토캐스틱 점수 (K > D일 때)
        if stoch_data['k'] > stoch_data['d']:
            score += TripleScreenConstants.STOCHASTIC_SCORE_WEIGHT
        
        # 볼린저 밴드 점수 (중간 밴드 근처)
        bb_pos = bb_data['position']
        if 0.3 <= bb_pos <= 0.7:
            score += TripleScreenConstants.BOLLINGER_SCORE_WEIGHT
        
        # MACD 점수 (히스토그램 양수)
        if macd_data['histogram'] > 0:
            score += TripleScreenConstants.MACD_SCORE_WEIGHT
        
        return score
    
    def _analyze_price_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """가격 패턴 분석"""
        try:
            # 최근 일간의 패턴 분석
            recent_data = df.tail(TripleScreenConstants.PATTERN_ANALYSIS_DAYS)
            
            # 상승 패턴 확인
            price_trend = recent_data['close'].iloc[-1] > recent_data['close'].iloc[0]
            
            # 고점/저점 패턴
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # 더블 바텀 패턴 (간단한 버전)
            double_bottom = (lows[1] < lows[0] and lows[3] < lows[2] and 
                           abs(lows[1] - lows[3]) / lows[1] < TripleScreenConstants.DOUBLE_BOTTOM_TOLERANCE)
            
            # 상승 삼각형 패턴 (간단한 버전)
            ascending_triangle = (highs[1] > highs[0] and highs[3] > highs[2] and
                                lows[1] > lows[0] and lows[3] > lows[2])
            
            pattern_score = 0.0
            pattern_name = "None"
            
            if price_trend:
                pattern_score += TripleScreenConstants.PATTERN_TREND_SCORE
                pattern_name = "Uptrend"
            
            if double_bottom:
                pattern_score += TripleScreenConstants.DOUBLE_BOTTOM_SCORE
                pattern_name = "Double Bottom"
            
            if ascending_triangle:
                pattern_score += TripleScreenConstants.ASCENDING_TRIANGLE_SCORE
                pattern_name = "Ascending Triangle"
            
            return {
                'name': pattern_name,
                'score': pattern_score,
                'trend': price_trend
            }
            
        except:
            return {'name': 'Unknown', 'score': 0.0, 'trend': False}
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """지지/저항 분석"""
        try:
            current_price = df.iloc[-1]['close']
            
            # 최근 저점들을 지지선으로 간주
            recent_lows = df['low'].tail(TripleScreenConstants.SUPPORT_ANALYSIS_PERIOD).sort_values()
            support_levels = recent_lows.head(TripleScreenConstants.SUPPORT_LEVELS_COUNT).values
            
            # 현재가가 지지선 근처에 있는지 확인
            near_support = False
            for support in support_levels:
                if abs(current_price - support) / current_price < TripleScreenConstants.SUPPORT_DISTANCE_THRESHOLD:
                    near_support = True
                    break
            
            return {
                'near_support': near_support,
                'support_levels': support_levels.tolist(),
                'current_price': current_price
            }
            
        except:
            return {'near_support': False, 'support_levels': [], 'current_price': 0}
    
    def _should_sell_stock(self, stock_code: str, current_date: datetime.date) -> bool:
        """매도 조건을 확인합니다."""
        try:
            # 데이터 유효성 확인
            daily_df = self.data_store['daily'][stock_code]
            if daily_df.empty:
                return False
            
            historical_data = self._get_historical_data_up_to(
                'daily', stock_code, current_date, lookback_period=TripleScreenConstants.SELL_ANALYSIS_PERIOD
            )
            
            if len(historical_data) < TripleScreenConstants.MIN_DATA_FOR_SELL:
                return False
            
            # 1. RSI 과매수 조건
            rsi_data = calculate_rsi_incremental(
                historical_data, self.strategy_params['momentum_rsi_period'],
                stock_code, self.rsi_cache.get(stock_code)
            )
            if rsi_data['current'] > TripleScreenConstants.RSI_OVERBOUGHT_THRESHOLD:
                logger.debug(f'{stock_code}: RSI 과매수 ({rsi_data["current"]:.1f}) - 매도 조건')
                return True
            
            # 2. 추세 반전 조건 (이동평균 하향 돌파)
            trend_ma_period = self.strategy_params['trend_ma_period']
            ma = historical_data['close'].rolling(window=trend_ma_period).mean()
            current_price = historical_data['close'].iloc[-1]
            
            if len(ma) >= 2 and current_price < ma.iloc[-1] and historical_data['close'].iloc[-2] >= ma.iloc[-2]:
                logger.debug(f'{stock_code}: 이동평균 하향 돌파 - 매도 조건')
                return True
            
            # 3. 볼린저 밴드 상단 돌파 후 하락
            bb_data = self._calculate_bollinger_bands(historical_data)
            if bb_data['position'] > TripleScreenConstants.BOLLINGER_UPPER_SELL_THRESHOLD:
                logger.debug(f'{stock_code}: 볼린저 밴드 상단 근처 ({bb_data["position"]:.2f}) - 매도 조건')
                return True
            
            # 4. 보유 기간 조건 (최소 보유 기간 이상 보유 시 수익률 확인)
            position_info = self.broker.positions.get(stock_code)
            if position_info and position_info.get('entry_date'):
                holding_days = (current_date - position_info['entry_date']).days
                if holding_days >= TripleScreenConstants.MIN_HOLDING_DAYS:
                    # 수익률 계산
                    avg_price = position_info['avg_price']
                    profit_ratio = (current_price - avg_price) / avg_price * 100
                    
                    # 익절 조건
                    if profit_ratio >= TripleScreenConstants.PROFIT_TAKE_THRESHOLD:
                        logger.debug(f'{stock_code}: 보유 {holding_days}일, 수익률 {profit_ratio:.1f}% - 매도 조건')
                        return True
                    
                    # 손절 조건
                    if profit_ratio <= TripleScreenConstants.STOP_LOSS_THRESHOLD:
                        logger.debug(f'{stock_code}: 보유 {holding_days}일, 손실률 {profit_ratio:.1f}% - 매도 조건')
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f'{stock_code} 매도 조건 확인 중 오류: {str(e)}')
            return False
    
    def run_daily_logic(self, current_date: datetime.date):
        """
        일봉 데이터를 기반으로 삼중창 시스템 전략 로직을 실행합니다.
        수정: 전일 데이터까지만 사용하여 장전 판단이 가능하도록 함
        
        Args:
            current_date: 현재 날짜
        """
        try:
            logger.info(f'[{current_date.isoformat()}] --- 일간 삼중창 시스템 로직 실행 중 (전일 데이터 기준) ---')
            
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
                logger.warning(f'{current_date}: 전일 데이터를 찾을 수 없어 삼중창 전략을 건너뜁니다.')
                return
            
            # signals 초기화
            self._initialize_signals_for_all_stocks()
            
            # 1. 모든 종목의 삼중창 점수 계산 (전일 데이터 기준)
            triple_screen_scores = {}
            for stock_code in self.data_store['daily']:
                if stock_code == self.strategy_params['safe_asset_code']:
                    continue
                
                # 데이터 유효성 확인 (다른 전략들과 동일한 방식)
                daily_df = self.data_store['daily'][stock_code]
                if daily_df.empty:
                    continue
                
                # 충분한 데이터가 있는지 확인
                required_periods = max(
                    self.strategy_params['trend_ma_period'],
                    self.strategy_params['momentum_rsi_period'],
                    self.strategy_params['volume_ma_period']
                )
                
                # 수정: prev_trading_day까지만 사용하여 장전 판단
                historical_data = self._get_historical_data_up_to(
                    'daily', stock_code, prev_trading_day, lookback_period=required_periods + 1
                )
                
                if len(historical_data) < required_periods:
                    logger.debug(f'{stock_code} 종목의 삼중창 계산을 위한 데이터가 부족합니다.')
                    continue
                
                # 캐시 초기화 (증분 계산을 위해)
                self._initialize_caches(stock_code, historical_data)
                
                # 삼중창 필터링 통과 여부 확인 (전일 데이터 기준)
                screen_result = self._apply_triple_screen(stock_code, prev_trading_day)
                
                if screen_result['passed']:
                    # 종합 점수 계산
                    total_score = (
                        screen_result['screen1']['strength'] * TripleScreenConstants.TREND_STRENGTH_WEIGHT +
                        screen_result['screen2']['momentum'] * TripleScreenConstants.MOMENTUM_SCORE_WEIGHT +
                        screen_result['screen3']['pattern_score'] * TripleScreenConstants.PATTERN_SCORE_WEIGHT
                    )
                    triple_screen_scores[stock_code] = total_score
                    logger.info(f"[{prev_trading_day}] {stock_code}: 삼중창 통과 (점수: {total_score:.3f})")
            
            if not triple_screen_scores:
                logger.warning('계산된 삼중창 점수가 없습니다.')
                return
            
            # 2. 매수 후보 종목 선정 (점수 기준 상위 N개)
            sorted_stocks = sorted(triple_screen_scores.items(), key=lambda x: x[1], reverse=True)
            buy_candidates = set()
            for i, (stock_code, score) in enumerate(sorted_stocks):
                if i < self.strategy_params['num_top_stocks']:
                    buy_candidates.add(stock_code)
                    logger.info(f'매수 후보 {i+1}: {stock_code} (점수: {score:.3f})')
            if not buy_candidates:
                logger.warning('매수 후보 종목이 없습니다.')
                return
            
            # 3. 매도 후보 종목 선정 (매수 후보에서 제외된 보유 종목 + 명시적 매도 조건)
            sell_candidates = set()
            current_positions = set(self.broker.positions.keys())
            for stock_code in current_positions:
                if stock_code == self.strategy_params['safe_asset_code']:
                    continue
                # 매수 후보에서 빠진 종목은 무조건 매도 후보
                if stock_code not in buy_candidates:
                    sell_candidates.add(stock_code)
                    logger.info(f'매수 후보 제외로 인한 매도 후보 추가: {stock_code}')
                # 명시적 매도 조건도 추가
                elif self._should_sell_stock(stock_code, prev_trading_day):
                    sell_candidates.add(stock_code)
                    logger.info(f'매도 후보: {stock_code} (매도 조건 만족)')
            
            # 4. 신호 생성 및 업데이트 (부모 클래스 메서드 사용) - 전일 데이터 기준
            final_positions = self._generate_signals(prev_trading_day, buy_candidates, sorted_stocks, sell_candidates)
            
            # 5. 리밸런싱 계획 요약 로깅 (전일 데이터 기준)
            self._log_rebalancing_summary(prev_trading_day, buy_candidates, final_positions, sell_candidates)
            
            logger.info(f"[{prev_trading_day}] 삼중창 시스템 일봉 로직 실행 완료: {len(buy_candidates)}개 신호")
            
        except Exception as e:
            logger.error(f"삼중창 시스템 일봉 로직 실행 중 오류: {str(e)}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")

    def _initialize_caches(self, stock_code: str, df: pd.DataFrame):
        """캐시 초기화 (새로운 종목이나 데이터 추가 시)"""
        try:
            # strategies_util의 함수 사용
            cache_data = initialize_indicator_caches(df, stock_code, 'all')
            
            # MACD 캐시 업데이트
            if 'macd' in cache_data:
                self.macd_cache[stock_code] = cache_data['macd']
            
            # RSI 캐시 업데이트
            if 'rsi' in cache_data:
                self.rsi_cache[stock_code] = cache_data['rsi']
                
            # 거래량 캐시 업데이트
            if 'volume' in cache_data:
                self.volume_cache[stock_code] = cache_data['volume']
                
        except Exception as e:
            logger.error(f"캐시 초기화 중 오류 ({stock_code}): {str(e)}") 