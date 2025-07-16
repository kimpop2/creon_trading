# utils.py 또는 indicators.py

import datetime
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_momentum(data, period):
    """주어진 데이터프레임의 'close' 가격에 대한 모멘텀 스코어를 계산합니다."""
    # Ensure 'close' column exists
    if 'close' not in data.columns:
        raise ValueError("데이터프레임에 'close' 컬럼이 없습니다.")
    return (data['close'].pct_change(period).fillna(0) * 100)

def calculate_rsi(data: pd.Series, period: int) -> pd.Series:
    """
    주어진 Series 데이터에 대해 상대강도지수(RSI)를 계산합니다.
    Args:
        data (pd.Series): 종가(close) 데이터 Series.
        period (int): RSI 계산 기간.
    Returns:
        pd.Series: 계산된 RSI 값 Series.
    """
    # 'data' 인자가 이미 종가(close) Series 자체라고 가정합니다.
    # 따라서 'data.columns'에 접근할 필요가 없습니다.
    if not isinstance(data, pd.Series):
        raise TypeError("calculate_rsi 함수는 Pandas Series 객체를 인자로 받아야 합니다.")

    # 가격 변화 계산
    delta = data.diff()

    # 상승분(gain)과 하락분(loss) 분리
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # 지수 이동 평균(EMA) 계산
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # RSI 계산
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# def calculate_rsi(data, period):
#     """주어진 데이터프레임의 'close' 가격에 대한 RSI를 계산합니다."""
#     # Ensure 'close' column exists
#     if 'close' not in data.columns:
#         raise ValueError("데이터프레임에 'close' 컬럼이 없습니다.")
#     delta = data['close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    
#     rs = gain / loss.replace(0, np.nan)
#     rsi = 100 - (100 / (1 + rs))
    
#     return rsi.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 100)

def calculate_rsi_incremental(df, rsi_period, stock_code=None, cache=None):
    """
    RSI 계산 (증분 방식)
    
    Args:
        df: 가격 데이터 (DataFrame with 'close' column)
        rsi_period: RSI 계산 기간
        stock_code: 종목 코드 (캐시 키용)
        cache: 이전 계산 결과 캐시 (dict)
    
    Returns:
        dict: RSI 계산 결과
    """
    try:
        if cache is None:
            # 캐시가 없으면 전체 계산
            return calculate_rsi_full(df, rsi_period)
        
        last_date = cache.get('last_date')
        
        # 마지막 계산 이후 새로운 데이터가 있는지 확인
        if last_date is None or df.index[-1] <= last_date:
            # 새로운 데이터가 없으면 캐시된 값 반환
            return {
                'current': cache['rsi'],
                'oversold': 30,  # 기본값
                'overbought': 70  # 기본값
            }
        
        # 새로운 데이터만 증분 계산
        new_data = df[df.index > last_date]
        if len(new_data) == 0:
            return {
                'current': cache['rsi'],
                'oversold': 30,
                'overbought': 70
            }
        
        # 증분 RSI 계산
        avg_gain = cache.get('avg_gain', 0)
        avg_loss = cache.get('avg_loss', 0)
        
        for idx, row in new_data.iterrows():
            # 가격 변화 계산
            if 'prev_price' in cache:
                change = row['close'] - cache['prev_price']
                gain = max(change, 0)
                loss = max(-change, 0)
                
                # 지수 이동평균 업데이트
                avg_gain = (avg_gain * (rsi_period - 1) + gain) / rsi_period
                avg_loss = (avg_loss * (rsi_period - 1) + loss) / rsi_period
            
            cache['prev_price'] = row['close']
            cache['last_date'] = idx
        
        # RSI 계산
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # 캐시 업데이트
        cache['rsi'] = rsi
        cache['avg_gain'] = avg_gain
        cache['avg_loss'] = avg_loss
        
        return {
            'current': rsi,
            'oversold': 30,
            'overbought': 70
        }
        
    except Exception as e:
        logger.error(f"RSI 증분 계산 중 오류: {str(e)}")
        return {'current': 50, 'oversold': 30, 'overbought': 70}

def calculate_rsi_full(df, rsi_period):
    """전체 RSI 계산 (초기화용)"""
    try:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'current': rsi.iloc[-1],
            'oversold': 30,
            'overbought': 70
        }
    except:
        return {'current': 50, 'oversold': 30, 'overbought': 70}

def calculate_macd_incremental(df, stock_code=None, cache=None):
    """
    MACD 계산 (증분 방식)
    
    Args:
        df: 가격 데이터 (DataFrame with 'close' column)
        stock_code: 종목 코드 (캐시 키용)
        cache: 이전 계산 결과 캐시 (dict)
    
    Returns:
        dict: MACD 계산 결과
    """
    try:
        if cache is None:
            # 캐시가 없으면 전체 계산
            return calculate_macd_full(df)
        
        last_date = cache.get('last_date')
        
        # 마지막 계산 이후 새로운 데이터가 있는지 확인
        if last_date is None or df.index[-1] <= last_date:
            # 새로운 데이터가 없으면 캐시된 값 반환
            return {
                'macd': cache['macd'],
                'signal': cache['signal'],
                'histogram': cache['histogram']
            }
        
        # 새로운 데이터만 증분 계산
        new_data = df[df.index > last_date]
        if len(new_data) == 0:
            return {
                'macd': cache['macd'],
                'signal': cache['signal'],
                'histogram': cache['histogram']
            }
        
        # 증분 MACD 계산
        for idx, row in new_data.iterrows():
            price = row['close']
            
            # EMA 증분 업데이트
            alpha1 = 2.0 / (12 + 1)  # 12일 EMA
            alpha2 = 2.0 / (26 + 1)  # 26일 EMA
            alpha3 = 2.0 / (9 + 1)   # 9일 EMA
            
            # MACD 라인 업데이트
            cache['ema12'] = alpha1 * price + (1 - alpha1) * cache.get('ema12', price)
            cache['ema26'] = alpha2 * price + (1 - alpha2) * cache.get('ema26', price)
            cache['macd'] = cache['ema12'] - cache['ema26']
            
            # 신호선 업데이트
            cache['signal'] = alpha3 * cache['macd'] + (1 - alpha3) * cache.get('signal', cache['macd'])
            
            # 히스토그램 업데이트
            cache['histogram'] = cache['macd'] - cache['signal']
            
            cache['last_date'] = idx
        
        return {
            'macd': cache['macd'],
            'signal': cache['signal'],
            'histogram': cache['histogram']
        }
        
    except Exception as e:
        logger.error(f"MACD 증분 계산 중 오류: {str(e)}")
        return {'macd': 0, 'signal': 0, 'histogram': 0}

def calculate_macd_full(df):
    """전체 MACD 계산 (초기화용)"""
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

def initialize_indicator_caches(df, stock_code, cache_type='all'):
    """
    지표 캐시 초기화 (새로운 종목이나 데이터 추가 시)
    
    Args:
        df: 가격 데이터
        stock_code: 종목 코드
        cache_type: 초기화할 캐시 타입 ('rsi', 'macd', 'all')
    
    Returns:
        dict: 초기화된 캐시
    """
    cache = {}
    
    try:
        if cache_type in ['rsi', 'all']:
            # RSI 캐시 초기화
            rsi_period = 14  # 기본값
            rsi_data = calculate_rsi_full(df, rsi_period)
            
            # 초기 평균값 계산
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi_period).mean().iloc[-1]
            avg_loss = loss.rolling(window=rsi_period).mean().iloc[-1]
            
            cache['rsi'] = {
                'rsi': rsi_data['current'],
                'avg_gain': avg_gain if not pd.isna(avg_gain) else 0,
                'avg_loss': avg_loss if not pd.isna(avg_loss) else 0,
                'prev_price': df['close'].iloc[-1] if len(df) > 0 else 0,
                'last_date': df.index[-1] if len(df) > 0 else None
            }
        
        if cache_type in ['macd', 'all']:
            # MACD 캐시 초기화
            macd_data = calculate_macd_full(df)
            cache['macd'] = {
                'macd': macd_data['macd'],
                'signal': macd_data['signal'],
                'histogram': macd_data['histogram'],
                'ema12': df['close'].ewm(span=12).mean().iloc[-1],
                'ema26': df['close'].ewm(span=26).mean().iloc[-1],
                'last_date': df.index[-1] if len(df) > 0 else None
            }
        
        if cache_type in ['volume', 'all']:
            # 거래량 캐시 초기화
            volume_period = 20  # 기본값
            volume_ma = df['volume'].rolling(window=volume_period, min_periods=1).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1] if len(df) > 0 else 0
            
            cache['volume'] = {
                'ratio': current_volume / volume_ma if volume_ma > 0 else 1.0,
                'current': current_volume,
                'average': volume_ma,
                'last_date': df.index[-1] if len(df) > 0 else None
            }
            
    except Exception as e:
        logger.error(f"캐시 초기화 중 오류 ({stock_code}): {str(e)}")
    
    return cache

def get_next_weekday(date, target_weekday):
    """주어진 날짜로부터 다음 target_weekday(0=월요일, 6=일요일)를 찾습니다."""
    days_ahead = target_weekday - date.weekday()
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return date + datetime.timedelta(days=days_ahead)

# def calculate_sma(data, period):
#     """주어진 데이터프레임의 'close' 가격에 대한 단순 이동평균(SMA)을 계산합니다."""
#     if 'close' not in data.columns:
#         raise ValueError("데이터프레임에 'close' 컬럼이 없습니다.")
#     return data['close'].rolling(window=period, min_periods=1).mean() # min_periods=1로 설정하여 초기에도 계산되도록 함

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """
    주어진 Series 데이터에 대해 단순 이동 평균(SMA)을 계산합니다.
    Args:
        data (pd.Series): 종가(close) 데이터 Series.
        period (int): SMA 계산 기간.
    Returns:
        pd.Series: 계산된 SMA 값 Series.
    """
    # 'data' 인자가 이미 종가(close) Series 자체라고 가정합니다.
    # 따라서 'data.columns'에 접근할 필요가 없습니다.
    if not isinstance(data, pd.Series):
        raise TypeError("calculate_sma 함수는 Pandas Series 객체를 인자로 받아야 합니다.")
    
    # Series의 이름이 'close'가 아닐 수도 있으므로, 직접 접근하지 않습니다.
    # 전달받은 Series에 대해 바로 rolling().mean()을 적용합니다.
    sma = data.rolling(window=period).mean()
    return sma


def calculate_sma_incremental(historical_data, period, cache=None):
    """
    SMA를 누적 계산 방식으로 계산합니다.
    캐시된 데이터를 활용하여 효율적으로 계산합니다.
    
    Args:
        historical_data: 가격 데이터 (DataFrame with 'close' column)
        period: 이동평균 기간
        cache: 이전 계산 결과 캐시 (dict)
    
    Returns:
        tuple: (sma_value, updated_cache)
    """
    if len(historical_data) < period:
        return None, cache
    
    # 캐시된 SMA 값이 있고, 새로운 데이터가 1개만 추가된 경우
    if (cache is not None and 
        'sma' in cache and 
        'prices' in cache and
        len(historical_data) == len(cache['prices']) + 1):
        
        old_sma = cache['sma']
        new_price = historical_data['close'].iloc[-1]
        
        # 누적 계산: 새로운 SMA = (기존 SMA * (period-1) + 새로운 가격) / period
        new_sma = (old_sma * (period - 1) + new_price) / period
        
        # 캐시 업데이트
        updated_cache = {
            'sma': new_sma,
            'prices': historical_data['close'].values
        }
        
        return new_sma, updated_cache
    
    # 전체 SMA 계산 (캐시가 없는 경우)
    sma_value = historical_data['close'].rolling(window=period, min_periods=1).mean().iloc[-1]
    
    # 캐시 초기화
    updated_cache = {
        'sma': sma_value,
        'prices': historical_data['close'].values
    }
    
    return sma_value, updated_cache

def calculate_volume_ma_incremental(historical_data, period, cache=None):
    """
    거래량 이동평균을 누적 계산 방식으로 계산합니다.
    
    Args:
        historical_data: 거래량 데이터 (DataFrame with 'volume' column)
        period: 이동평균 기간
        cache: 이전 계산 결과 캐시 (dict)
    
    Returns:
        tuple: (volume_ma_value, updated_cache)
    """
    if len(historical_data) < period:
        return None, cache
    
    # 캐시된 거래량 MA 값이 있고, 새로운 데이터가 1개만 추가된 경우
    if (cache is not None and 
        'volume_ma' in cache and 
        'volumes' in cache and
        len(historical_data) == len(cache['volumes']) + 1):
        
        old_volume_ma = cache['volume_ma']
        new_volume = historical_data['volume'].iloc[-1]
        
        # 누적 계산
        new_volume_ma = (old_volume_ma * (period - 1) + new_volume) / period
        
        # 캐시 업데이트
        updated_cache = {
            'volume_ma': new_volume_ma,
            'volumes': historical_data['volume'].values
        }
        
        return new_volume_ma, updated_cache
    
    # 전체 거래량 MA 계산 (캐시가 없는 경우)
    volume_ma_value = historical_data['volume'].rolling(window=period, min_periods=1).mean().iloc[-1]
    
    # 캐시 초기화
    updated_cache = {
        'volume_ma': volume_ma_value,
        'volumes': historical_data['volume'].values
    }
    
    return volume_ma_value, updated_cache

def calculate_volume_ma_incremental_simple(df, period, stock_code=None, cache=None):
    """
    거래량 이동평균 계산 (증분 방식, 단순화된 버전)
    
    Args:
        df: 거래량 데이터 (DataFrame with 'volume' column)
        period: 이동평균 기간
        stock_code: 종목 코드 (캐시 키용)
        cache: 이전 계산 결과 캐시 (dict)
    
    Returns:
        dict: 거래량 분석 결과
    """
    try:
        if cache is None:
            # 캐시가 없으면 전체 계산
            volume_ma = df['volume'].rolling(window=period, min_periods=1).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            return {
                'ratio': current_volume / volume_ma if volume_ma > 0 else 1.0,
                'current': current_volume,
                'average': volume_ma
            }
        
        last_date = cache.get('last_date')
        
        # 마지막 계산 이후 새로운 데이터가 있는지 확인
        if last_date is None or df.index[-1] <= last_date:
            # 새로운 데이터가 없으면 캐시된 값 반환
            return {
                'ratio': cache.get('ratio', 1.0),
                'current': cache.get('current', 0),
                'average': cache.get('average', 0)
            }
        
        # 새로운 데이터만 증분 계산
        new_data = df[df.index > last_date]
        if len(new_data) == 0:
            return {
                'ratio': cache.get('ratio', 1.0),
                'current': cache.get('current', 0),
                'average': cache.get('average', 0)
            }
        
        # 증분 거래량 MA 계산
        volume_ma = cache.get('average', 0)
        for idx, row in new_data.iterrows():
            new_volume = row['volume']
            # 지수 이동평균 업데이트
            alpha = 2.0 / (period + 1)
            volume_ma = alpha * new_volume + (1 - alpha) * volume_ma
            cache['last_date'] = idx
        
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
        
        # 캐시 업데이트
        cache['ratio'] = volume_ratio
        cache['current'] = current_volume
        cache['average'] = volume_ma
        
        return {
            'ratio': volume_ratio,
            'current': current_volume,
            'average': volume_ma
        }
        
    except Exception as e:
        logger.error(f"거래량 MA 증분 계산 중 오류: {str(e)}")
        return {'ratio': 1.0, 'current': 0, 'average': 0}

def calculate_performance_metrics(portfolio_value_series: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    [수정됨] 포트폴리오 가치 시계열 데이터를 기반으로 주요 성능 지표를 계산합니다.
    짧은 기간에 대한 연환산 오류를 방지하는 로직이 포함됩니다.
    """
    if portfolio_value_series.empty or len(portfolio_value_series) < 2:
        return {'total_return': 0, 'annual_return': 0, 'annual_volatility': 0, 'sharpe_ratio': 0, 'mdd': 0}

    # 총 수익률
    total_return = (portfolio_value_series.iloc[-1] / portfolio_value_series.iloc[0]) - 1

    # 최대 낙폭 (MDD)
    rolling_max = portfolio_value_series.expanding(min_periods=1).max()
    daily_drawdown = (portfolio_value_series - rolling_max) / rolling_max
    mdd = daily_drawdown.min()

    # 연간 수익률 (CAGR)
    # [수정] 투자 기간이 1일 미만일 경우 발생하는 오류 방지
    days = (portfolio_value_series.index[-1] - portfolio_value_series.index[0]).days
    if days < 1:  # 투자 기간이 하루도 안 되면 연환산은 의미가 없으므로 총 수익률을 그대로 사용
        annual_return = total_return
    else:
        annual_return = ((1 + total_return) ** (365.0 / days)) - 1
        
    # 연간 변동성
    daily_returns = portfolio_value_series.pct_change().dropna()
    # [수정] 거래가 거의 없어 변동성이 0인 경우 예외 처리
    if daily_returns.std() == 0 or np.isnan(daily_returns.std()):
        annual_volatility = 0
    else:
        annual_volatility = daily_returns.std() * np.sqrt(252) # 1년 거래일 기준

    # 샤프 지수
    # [수정] 변동성이 0일 때 나누기 오류 방지
    if annual_volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'mdd': mdd
    }

def calculate_ema(series, period):
    """지수이동평균(EMA)을 계산합니다."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(series, short_period=12, long_period=26, signal_period=9):
    """
    MACD (Moving Average Convergence Divergence)를 계산합니다.
    MACD 라인, 시그널 라인, 히스토그램을 포함하는 DataFrame을 반환합니다.
    """
    ema_short = calculate_ema(series, period=short_period)
    ema_long = calculate_ema(series, period=long_period)
    
    macd_line = ema_short - ema_long
    signal_line = calculate_ema(macd_line, period=signal_period)
    histogram = macd_line - signal_line
    
    macd_df = pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram
    })
    
    return macd_df

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """
    볼린저 밴드를 계산합니다.
    
    Args:
        data (pd.DataFrame): OHLCV 데이터프레임
        period (int): 이동평균 기간 (기본값: 20)
        std_dev (int): 표준편차 배수 (기본값: 2)
    
    Returns:
        tuple: (upper_band, middle_band, lower_band) - 각각 pd.Series
    """
    if 'close' not in data.columns:
        raise ValueError("데이터프레임에 'close' 컬럼이 없습니다.")
    
    # 중간선 (이동평균)
    middle_band = data['close'].rolling(window=period, min_periods=1).mean()
    
    # 표준편차
    rolling_std = data['close'].rolling(window=period, min_periods=1).std()
    
    # 상단선과 하단선
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    return upper_band, middle_band, lower_band

def calculate_stochastic(data, period=14, k_period=3, d_period=3):
    """
    스토캐스틱을 계산합니다.
    
    Args:
        data (pd.DataFrame): OHLCV 데이터프레임
        period (int): %K 계산 기간 (기본값: 14)
        k_period (int): %K 스무딩 기간 (기본값: 3)
        d_period (int): %D 스무딩 기간 (기본값: 3)
    
    Returns:
        tuple: (%K, %D) - 각각 pd.Series
    """
    if not all(col in data.columns for col in ['high', 'low', 'close']):
        raise ValueError("데이터프레임에 'high', 'low', 'close' 컬럼이 필요합니다.")
    
    # 기간 내 최고가와 최저가
    highest_high = data['high'].rolling(window=period, min_periods=1).max()
    lowest_low = data['low'].rolling(window=period, min_periods=1).min()
    
    # %K 계산
    k_raw = ((data['close'] - lowest_low) / (highest_high - lowest_low)) * 100
    
    # %K 스무딩
    k_smoothed = k_raw.rolling(window=k_period, min_periods=1).mean()
    
    # %D 계산 (%K의 이동평균)
    d_smoothed = k_smoothed.rolling(window=d_period, min_periods=1).mean()
    
    return k_smoothed, d_smoothed