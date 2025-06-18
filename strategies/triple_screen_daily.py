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

logger = logging.getLogger(__name__)

class TripleScreenDaily(DailyStrategy):
    """
    알렉산더 엘더의 삼중창 시스템
    - 1단계: 장기 추세 확인 (주봉 기준)
    - 2단계: 중기 모멘텀 확인 (일봉 기준)
    - 3단계: 단기 진입점 확인 (일봉 패턴)
    """
    
    def __init__(self, data_store: Dict, strategy_params: Dict[str, Any], broker):
        super().__init__(data_store, strategy_params, broker)
        self.strategy_name = "TripleScreenDaily"
        
        # 파라미터 검증
        self._validate_parameters()
        
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
    
    def calculate_signals(self, current_date: datetime.date) -> Dict[str, Dict[str, Any]]:
        """
        삼중창 시스템 신호 계산
        """
        signals = {}
        
        try:
            # 모든 종목에 대해 삼중창 필터링 적용
            for stock_code in self.data_store['daily']:
                if stock_code == self.strategy_params['safe_asset_code']:
                    continue
                
                # 삼중창 필터링 통과 여부 확인
                screen_result = self._apply_triple_screen(stock_code, current_date)
                
                if screen_result['passed']:
                    # 매매 신호 생성
                    signal = self._generate_signal(stock_code, current_date, screen_result)
                    if signal:
                        signals[stock_code] = signal
            
            # 상위 신호만 선택
            top_signals = self._select_top_signals(signals)
            
            logger.info(f"[{current_date}] 삼중창 시스템 신호 생성 완료: {len(top_signals)}개 종목")
            
        except Exception as e:
            logger.error(f"삼중창 시스템 신호 계산 중 오류: {str(e)}")
        
        return top_signals
    
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
                return result
            
            # 2단계: 중기 모멘텀 확인
            screen2_result = self._screen2_momentum_analysis(stock_code, current_date)
            result['screen2'] = screen2_result
            
            if not screen2_result['passed']:
                return result
            
            # 3단계: 단기 진입점 확인
            screen3_result = self._screen3_entry_analysis(stock_code, current_date)
            result['screen3'] = screen3_result
            
            # 모든 단계 통과 시 최종 통과
            result['passed'] = screen3_result['passed']
            
        except Exception as e:
            logger.error(f"삼중창 필터링 중 오류 ({stock_code}): {str(e)}")
        
        return result
    
    def _screen1_trend_analysis(self, stock_code: str, current_date: datetime.date) -> Dict[str, Any]:
        """1단계: 장기 추세 분석"""
        result = {'passed': False, 'trend': 0, 'strength': 0}
        
        df = self.data_store['daily'][stock_code]
        if df.empty or len(df) < self.strategy_params['trend_ma_period'] * 2:
            return result
        
        # 최근 데이터 추출 (주봉 기준으로 충분한 기간)
        recent_df = df[df.index.date <= current_date].tail(self.strategy_params['trend_ma_period'] * 3)
        
        if len(recent_df) < self.strategy_params['trend_ma_period']:
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
        
        # MACD 계산 (추세 확인용)
        macd_data = self._calculate_macd(recent_df)
        
        # 조건: 상승추세이고 최소 강도 만족
        min_strength = self.strategy_params['min_trend_strength']
        passed = (trend_direction > 0 and 
                 trend_strength >= min_strength and 
                 macd_data['signal'] > 0)
        
        result.update({
            'passed': passed,
            'trend': trend_direction,
            'strength': trend_strength,
            'macd': macd_data
        })
        
        return result
    
    def _screen2_momentum_analysis(self, stock_code: str, current_date: datetime.date) -> Dict[str, Any]:
        """2단계: 중기 모멘텀 분석"""
        result = {'passed': False, 'rsi': 0, 'momentum': 0}
        
        df = self.data_store['daily'][stock_code]
        if df.empty or len(df) < self.strategy_params['momentum_rsi_period'] * 2:
            return result
        
        # 최근 데이터 추출
        recent_df = df[df.index.date <= current_date].tail(self.strategy_params['momentum_rsi_period'] * 3)
        
        if len(recent_df) < self.strategy_params['momentum_rsi_period']:
            return result
        
        # RSI 계산
        rsi_data = self._calculate_rsi(recent_df)
        current_rsi = rsi_data['current']
        
        # 스토캐스틱 계산
        stoch_data = self._calculate_stochastic(recent_df)
        
        # 볼린저 밴드 계산
        bb_data = self._calculate_bollinger_bands(recent_df)
        
        # 모멘텀 점수 계산
        momentum_score = self._calculate_momentum_score(rsi_data, stoch_data, bb_data)
        
        # 조건: RSI가 과매도에서 반등 중이고, 스토캐스틱이 상승 중
        oversold = self.strategy_params['momentum_rsi_oversold']
        overbought = self.strategy_params['momentum_rsi_overbought']
        
        rsi_condition = oversold < current_rsi < overbought
        stoch_condition = stoch_data['k'] > stoch_data['d'] and stoch_data['k'] < 80
        bb_condition = bb_data['position'] < 0.8  # 상단 밴드 아래
        
        passed = rsi_condition and stoch_condition and bb_condition and momentum_score > 0.3
        
        result.update({
            'passed': passed,
            'rsi': current_rsi,
            'momentum': momentum_score,
            'stochastic': stoch_data,
            'bollinger': bb_data
        })
        
        return result
    
    def _screen3_entry_analysis(self, stock_code: str, current_date: datetime.date) -> Dict[str, Any]:
        """3단계: 단기 진입점 분석"""
        result = {'passed': False, 'pattern': '', 'volume': 0}
        
        df = self.data_store['daily'][stock_code]
        if df.empty or len(df) < 20:
            return result
        
        # 최근 데이터 추출 (단기 패턴 분석용)
        recent_df = df[df.index.date <= current_date].tail(20)
        
        # 가격 패턴 분석
        pattern = self._analyze_price_pattern(recent_df)
        
        # 거래량 분석
        volume_data = self._analyze_volume_pattern(recent_df)
        
        # 지지/저항 분석
        support_resistance = self._analyze_support_resistance(recent_df)
        
        # 조건: 긍정적 패턴, 거래량 증가, 지지선 근처
        pattern_condition = pattern['score'] > 0.5
        volume_condition = volume_data['ratio'] > 1.2
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
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, float]:
        """MACD 계산"""
        try:
            # MACD 계산 (12, 26, 9)
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line.iloc[-1],
                'signal': signal_line.iloc[-1],
                'histogram': histogram.iloc[-1]
            }
        except:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    def _calculate_rsi(self, df: pd.DataFrame) -> Dict[str, float]:
        """RSI 계산"""
        try:
            rsi_period = self.strategy_params['momentum_rsi_period']
            
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return {
                'current': rsi.iloc[-1],
                'oversold': self.strategy_params['momentum_rsi_oversold'],
                'overbought': self.strategy_params['momentum_rsi_overbought']
            }
        except:
            return {'current': 50, 'oversold': 30, 'overbought': 70}
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> Dict[str, float]:
        """스토캐스틱 계산"""
        try:
            period = 14
            
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            
            k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window=3).mean()
            
            return {
                'k': k_percent.iloc[-1],
                'd': d_percent.iloc[-1]
            }
        except:
            return {'k': 50, 'd': 50}
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, float]:
        """볼린저 밴드 계산"""
        try:
            period = 20
            std = 2
            
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
            return {'position': 0.5, 'upper': 0, 'lower': 0, 'middle': 0}
    
    def _calculate_momentum_score(self, rsi_data: Dict, stoch_data: Dict, bb_data: Dict) -> float:
        """모멘텀 점수 계산"""
        score = 0.0
        
        # RSI 점수 (30-70 구간에서 높은 점수)
        rsi = rsi_data['current']
        if 30 <= rsi <= 70:
            score += 0.4
        elif 25 <= rsi <= 75:
            score += 0.2
        
        # 스토캐스틱 점수 (K > D일 때)
        if stoch_data['k'] > stoch_data['d']:
            score += 0.3
        
        # 볼린저 밴드 점수 (중간 밴드 근처)
        bb_pos = bb_data['position']
        if 0.3 <= bb_pos <= 0.7:
            score += 0.3
        
        return score
    
    def _analyze_price_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """가격 패턴 분석"""
        try:
            # 최근 5일간의 패턴 분석
            recent_5 = df.tail(5)
            
            # 상승 패턴 확인
            price_trend = recent_5['close'].iloc[-1] > recent_5['close'].iloc[0]
            
            # 고점/저점 패턴
            highs = recent_5['high'].values
            lows = recent_5['low'].values
            
            # 더블 바텀 패턴 (간단한 버전)
            double_bottom = (lows[1] < lows[0] and lows[3] < lows[2] and 
                           abs(lows[1] - lows[3]) / lows[1] < 0.02)
            
            # 상승 삼각형 패턴 (간단한 버전)
            ascending_triangle = (highs[1] > highs[0] and highs[3] > highs[2] and
                                lows[1] > lows[0] and lows[3] > lows[2])
            
            pattern_score = 0.0
            pattern_name = "None"
            
            if price_trend:
                pattern_score += 0.3
                pattern_name = "Uptrend"
            
            if double_bottom:
                pattern_score += 0.4
                pattern_name = "Double Bottom"
            
            if ascending_triangle:
                pattern_score += 0.3
                pattern_name = "Ascending Triangle"
            
            return {
                'name': pattern_name,
                'score': pattern_score,
                'trend': price_trend
            }
            
        except:
            return {'name': 'Unknown', 'score': 0.0, 'trend': False}
    
    def _analyze_volume_pattern(self, df: pd.DataFrame) -> Dict[str, float]:
        """거래량 패턴 분석"""
        try:
            volume_ma_period = self.strategy_params['volume_ma_period']
            
            volume_ma = df['volume'].rolling(window=volume_ma_period).mean()
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
            
        except:
            return {'ratio': 1.0, 'current': 0, 'average': 0}
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """지지/저항 분석"""
        try:
            current_price = df.iloc[-1]['close']
            
            # 최근 저점들을 지지선으로 간주
            recent_lows = df['low'].tail(10).sort_values()
            support_levels = recent_lows.head(3).values
            
            # 현재가가 지지선 근처에 있는지 확인
            near_support = False
            for support in support_levels:
                if abs(current_price - support) / current_price < 0.03:  # 3% 이내
                    near_support = True
                    break
            
            return {
                'near_support': near_support,
                'support_levels': support_levels.tolist(),
                'current_price': current_price
            }
            
        except:
            return {'near_support': False, 'support_levels': [], 'current_price': 0}
    
    def _generate_signal(self, stock_code: str, current_date: datetime.date, 
                        screen_result: Dict[str, Any]) -> Dict[str, Any]:
        """매매 신호 생성"""
        current_price = self._get_current_price(stock_code, current_date)
        
        if current_price <= 0:
            return None
        
        # 종합 점수 계산
        total_score = (
            screen_result['screen1']['strength'] * 0.4 +
            screen_result['screen2']['momentum'] * 0.4 +
            screen_result['screen3']['pattern_score'] * 0.2
        )
        
        # 매수 수량 계산
        target_amount = self.broker.cash * 0.15
        target_quantity = int(target_amount / current_price)
        
        if target_quantity > 0:
            return {
                'signal': 'buy',
                'quantity': target_quantity,
                'price': current_price,
                'signal_date': current_date,
                'strategy': self.strategy_name,
                'score': total_score,
                'screen_results': screen_result
            }
        
        return None
    
    def _select_top_signals(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """상위 신호 선택"""
        if not signals:
            return {}
        
        # 점수 기준 정렬
        sorted_signals = sorted(signals.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # 상위 N개 선택
        num_top_stocks = self.strategy_params['num_top_stocks']
        top_signals = dict(sorted_signals[:num_top_stocks])
        
        # 선택된 신호들의 정보 로깅
        for i, (stock_code, signal) in enumerate(sorted_signals[:num_top_stocks]):
            stock_name = self.broker.api_client.get_stock_name(stock_code)
            screen_results = signal['screen_results']
            logger.info(f"삼중창 통과 {i+1}: {stock_code} ({stock_name}) "
                       f"(점수: {signal['score']:.2f}, "
                       f"추세강도: {screen_results['screen1']['strength']:.2f}, "
                       f"모멘텀: {screen_results['screen2']['momentum']:.2f}, "
                       f"패턴: {screen_results['screen3']['pattern']})")
        
        return top_signals
    
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