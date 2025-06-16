# utils.py 또는 indicators.py

import datetime
import pandas as pd
import numpy as np

def calculate_momentum(data, period):
    """주어진 데이터프레임의 'close' 가격에 대한 모멘텀 스코어를 계산합니다."""
    # Ensure 'close' column exists
    if 'close' not in data.columns:
        raise ValueError("데이터프레임에 'close' 컬럼이 없습니다.")
    return (data['close'].pct_change(period).fillna(0) * 100)

def calculate_rsi(data, period):
    """주어진 데이터프레임의 'close' 가격에 대한 RSI를 계산합니다."""
    # Ensure 'close' column exists
    if 'close' not in data.columns:
        raise ValueError("데이터프레임에 'close' 컬럼이 없습니다.")
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 100)

def get_next_weekday(date, target_weekday):
    """주어진 날짜로부터 다음 target_weekday(0=월요일, 6=일요일)를 찾습니다."""
    days_ahead = target_weekday - date.weekday()
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return date + datetime.timedelta(days=days_ahead)
def calculate_sma(data, period):
    """주어진 데이터프레임의 'close' 가격에 대한 단순 이동평균(SMA)을 계산합니다."""
    if 'close' not in data.columns:
        raise ValueError("데이터프레임에 'close' 컬럼이 없습니다.")
    return data['close'].rolling(window=period, min_periods=1).mean() # min_periods=1로 설정하여 초기에도 계산되도록 함

def calculate_performance_metrics(portfolio_values, risk_free_rate=0.03):
    """
    포트폴리오 성과 지표를 계산합니다.
    
    Args:
        portfolio_values: 일별 포트폴리오 가치 시계열 데이터 (Pandas Series)
        risk_free_rate: 무위험 수익률 (연율화된 값, 예: 0.03 = 3%)
    """
    if portfolio_values.empty or len(portfolio_values) < 2:
        return {
            'total_return': 0, 'annual_return': 0, 'annual_volatility': 0,
            'sharpe_ratio': 0, 'mdd': 0, 'win_rate': 0, 'profit_factor': 0
        }

    # 일별 수익률 계산
    daily_returns = portfolio_values.pct_change().dropna()
    
    # 누적 수익률 계산
    cumulative_returns = (1 + daily_returns).cumprod()
    
    # MDD (Maximum Drawdown) 계산
    cumulative_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / cumulative_max - 1
    mdd = drawdowns.min()
    
    # 연간 수익률 계산 (연율화)
    total_days = len(daily_returns)
    total_years = total_days / 252  # 거래일 기준
    total_return = cumulative_returns.iloc[-1] - 1 if not cumulative_returns.empty else 0
    annual_return = (1 + total_return) ** (1 / total_years) - 1 if total_years > 0 else 0
    
    # 연간 변동성 계산 (연율화)
    annual_volatility = daily_returns.std() * np.sqrt(252) if not daily_returns.empty else 0
    
    # 샤프 지수 계산
    excess_returns = annual_return - risk_free_rate
    sharpe_ratio = excess_returns / annual_volatility if annual_volatility != 0 else 0
    
    # 승률 계산
    win_rate = len(daily_returns[daily_returns > 0]) / len(daily_returns) if not daily_returns.empty else 0
    
    # 평균 수익거래 대비 손실거래 비율 (Profit Factor)
    positive_returns_mean = daily_returns[daily_returns > 0].mean()
    negative_returns_mean_abs = abs(daily_returns[daily_returns < 0].mean())
    profit_factor = positive_returns_mean / negative_returns_mean_abs if negative_returns_mean_abs != 0 else float('inf')
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'mdd': mdd,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }