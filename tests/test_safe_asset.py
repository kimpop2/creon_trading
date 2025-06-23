#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.creon_api import CreonAPIClient
from datetime import datetime, timedelta
import pandas as pd

def test_safe_asset_data():
    """안전자산 종목 A439870의 데이터를 테스트합니다."""
    
    print("=== 안전자산 데이터 테스트 ===")
    
    # 1. API 클라이언트 초기화
    api = CreonAPIClient()
    print(f"API 연결 상태: {api.connected}")
    
    if not api.connected:
        print("API가 연결되지 않았습니다. 크레온 플러스가 실행 중인지 확인하세요.")
        return
    
    # 2. 안전자산 종목 정보 확인
    stock_name = api.get_stock_name('A439870')
    print(f"종목명: {stock_name}")
    
    # 3. 일봉 데이터 조회
    start_date = '20250201'
    end_date = '20250401'
    
    print(f"\n일봉 데이터 조회: {start_date} ~ {end_date}")
    daily_df = api.get_daily_ohlcv('A439870', start_date, end_date)
    
    if daily_df.empty:
        print("일봉 데이터가 없습니다.")
        return
    
    print(f"데이터 수: {len(daily_df)}행")
    print("\n처음 5행:")
    print(daily_df.head())
    
    print("\n마지막 5행:")
    print(daily_df.tail())
    
    print(f"\n데이터 범위: {daily_df.index.min()} ~ {daily_df.index.max()}")
    print(f"컬럼: {list(daily_df.columns)}")
    
    # 4. 데이터 검증
    print("\n=== 데이터 검증 ===")
    print(f"시가 범위: {daily_df['open'].min():.2f} ~ {daily_df['open'].max():.2f}")
    print(f"종가 범위: {daily_df['close'].min():.2f} ~ {daily_df['close'].max():.2f}")
    print(f"거래량 범위: {daily_df['volume'].min():,} ~ {daily_df['volume'].max():,}")
    
    # 5. 최근 데이터 확인
    recent_date = daily_df.index.max()
    recent_data = daily_df.loc[recent_date]
    print(f"\n최근 거래일 ({recent_date.strftime('%Y-%m-%d')}) 데이터:")
    print(f"시가: {recent_data['open']:.2f}")
    print(f"고가: {recent_data['high']:.2f}")
    print(f"저가: {recent_data['low']:.2f}")
    print(f"종가: {recent_data['close']:.2f}")
    print(f"거래량: {recent_data['volume']:,}")

if __name__ == "__main__":
    test_safe_asset_data() 