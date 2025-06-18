"""
전략별 파라미터 설정
"""

# 기존 SMA 일봉 전략 파라미터
SMA_DAILY_PARAMS = {
    'short_sma_period': 2,
    'long_sma_period': 10,
    'volume_ma_period': 3,
    'num_top_stocks': 4,
    'safe_asset_code': 'A439870'
}

# 기존 RSI 분봉 전략 파라미터
RSI_MINUTE_PARAMS = {
    'minute_rsi_period': 45,
    'minute_rsi_oversold': 30,
    'minute_rsi_overbought': 70
}

# 새로운 듀얼 모멘텀 전략 파라미터
DUAL_MOMENTUM_PARAMS = {
    'momentum_period': 20,  # 모멘텀 계산 기간 (일)
    'rebalance_weekday': 0,  # 리밸런싱 요일 (0=월요일)
    'num_top_stocks': 5,  # 선택할 상위 종목 수
    'safe_asset_code': 'A439870'  # 안전자산 코드
}

# 새로운 볼린저 밴드 + RSI 전략 파라미터
BOLLINGER_RSI_PARAMS = {
    'bb_period': 20,  # 볼린저 밴드 기간
    'bb_std': 2.0,  # 볼린저 밴드 표준편차
    'rsi_period': 14,  # RSI 계산 기간
    'rsi_oversold': 30,  # RSI 과매도 기준
    'rsi_overbought': 70,  # RSI 과매수 기준
    'volume_ma_period': 20,  # 거래량 이동평균 기간
    'num_top_stocks': 4,  # 선택할 상위 종목 수
    'safe_asset_code': 'A439870'  # 안전자산 코드
}

# 새로운 섹터 로테이션 전략 파라미터
SECTOR_ROTATION_PARAMS = {
    'momentum_period': 30,  # 모멘텀 계산 기간 (일)
    'rebalance_weekday': 0,  # 리밸런싱 요일 (0=월요일)
    'num_top_sectors': 3,  # 선택할 상위 섹터 수
    'stocks_per_sector': 2,  # 섹터당 선택할 종목 수
    'safe_asset_code': 'A439870'  # 안전자산 코드
}

# 알렉산더 엘더 삼중창 시스템 전략 파라미터
TRIPLE_SCREEN_PARAMS = {
    'trend_ma_period': 50,  # 1단계: 장기 추세 이동평균 기간
    'momentum_rsi_period': 14,  # 2단계: RSI 계산 기간
    'momentum_rsi_oversold': 30,  # RSI 과매도 기준
    'momentum_rsi_overbought': 70,  # RSI 과매수 기준
    'volume_ma_period': 20,  # 거래량 이동평균 기간
    'num_top_stocks': 4,  # 선택할 상위 종목 수
    'safe_asset_code': 'A439870',  # 안전자산 코드
    'min_trend_strength': 0.02  # 최소 추세 강도 (2%)
}

# 손절매 파라미터 (모든 전략 공통)
STOP_LOSS_PARAMS = {
    'stop_loss_ratio': -5.0,  # 개별 종목 손절매 비율 (%)
    'trailing_stop_ratio': -3.0,  # 트레일링 스탑 비율 (%)
    'portfolio_stop_loss': -5.0,  # 포트폴리오 전체 손절매 비율 (%)
    'early_stop_loss': -5.0,  # 조기 손절매 비율 (%)
    'max_losing_positions': 3  # 최대 손실 포지션 수
}

# 전략별 파라미터 매핑
STRATEGY_PARAMS_MAPPING = {
    'sma_daily': SMA_DAILY_PARAMS,
    'rsi_minute': RSI_MINUTE_PARAMS,
    'dual_momentum_daily': DUAL_MOMENTUM_PARAMS,
    'bollinger_rsi_daily': BOLLINGER_RSI_PARAMS,
    'sector_rotation_daily': SECTOR_ROTATION_PARAMS,
    'triple_screen_daily': TRIPLE_SCREEN_PARAMS
}

def get_strategy_params(strategy_name: str) -> dict:
    """전략명에 따른 파라미터 반환"""
    return STRATEGY_PARAMS_MAPPING.get(strategy_name, {})

def get_all_strategy_params() -> dict:
    """모든 전략 파라미터 반환"""
    return {
        'strategies': STRATEGY_PARAMS_MAPPING,
        'stop_loss': STOP_LOSS_PARAMS
    } 