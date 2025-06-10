"""
파일명: test_dual_momentum_strategy.py
설명: 듀얼 모멘텀 전략 단위 테스트 코드
작성일: 2024-03-19
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import time

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.dual_momentum_daily_strategy import DualMomentumDailyStrategy
from api.creon_api import CreonAPIClient

class TestDualMomentumStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화 - Creon API 연결 및 테스트 데이터 로드"""
        cls.creon_api = CreonAPIClient()
        if not cls.creon_api.connected:
            raise unittest.SkipTest("Creon API에 연결할 수 없습니다.")
        
        # 테스트용 종목 리스트 (주요 섹터별 대표 종목)
        cls.test_stocks = {
            'IT': ['삼성전자', 'SK하이닉스'],
            '2차전지': ['LG에너지솔루션'],
            '바이오': ['삼성바이오로직스'],
            '금융': ['KB금융'],
            '안전자산': ['KODEX 국고채30년 액티브']
        }
        
        # 테스트 기간 설정 (현재 날짜 기준 2개월)
        cls.end_date = datetime.now().strftime('%Y%m%d')
        cls.start_date = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
        
        # 종목별 데이터 저장
        cls.test_data = {}
        
        print("\n=== 테스트 데이터 로드 중 ===")
        print(f"테스트 기간: {cls.start_date} ~ {cls.end_date} (약 40 거래일)")
        
        for sector, stocks in cls.test_stocks.items():
            for stock_name in stocks:
                code = cls.creon_api.get_stock_code(stock_name)
                if code:
                    data = cls.creon_api.get_daily_ohlcv(code, cls.start_date, cls.end_date)
                    if not data.empty:
                        cls.test_data[code] = data
                        print(f"{stock_name} ({code}) 데이터 로드 완료: {len(data)}행")
                    time.sleep(0.3)  # API 호출 제한 방지
                else:
                    print(f"경고: {stock_name} 종목 코드를 찾을 수 없습니다.")
        
        if not cls.test_data:
            raise unittest.SkipTest("테스트 데이터를 가져올 수 없습니다.")
        
        print(f"\n총 {len(cls.test_data)}개 종목 데이터 로드 완료")
        print("======================\n")

    def setUp(self):
        """각 테스트 케이스 실행 전 초기화"""
        self.strategy = DualMomentumDailyStrategy(params={
            'momentum_period': 20,      # 모멘텀 계산 기간 (20 거래일)
            'volume_period': 20,        # 거래량 계산 기간
            'rsi_period': 14,          # RSI 계산 기간
            'rsi_buy_threshold': 30,    # RSI 매수 기준
            'rsi_sell_threshold': 70,   # RSI 매도 기준
            'stop_loss': -0.05,        # 손절 기준
            'trailing_stop': -0.03,     # 트레일링 스탑 기준
            'take_profit': 0.1         # 익절 기준
        })

    def test_calculate_indicators(self):
        """지표 계산 메서드 테스트"""
        print("\n=== 지표 계산 테스트 ===")
        for code, data in self.test_data.items():
            result = self.strategy.calculate_indicators(data.copy())
            
            # 필수 지표들이 계산되었는지 확인
            required_columns = ['price_momentum', 'volume_momentum', 'MA5', 'MA20', 'MA60']
            for col in required_columns:
                self.assertIn(col, result.columns, f"지표 {col}이(가) 계산되지 않았습니다.")
            
            # 모멘텀 스코어가 계산되었는지 확인
            self.assertFalse(result['price_momentum'].isna().all(), "가격 모멘텀이 계산되지 않았습니다.")
            self.assertFalse(result['volume_momentum'].isna().all(), "거래량 모멘텀이 계산되지 않았습니다.")
            
            print(f"\n종목코드 {code} 최근 지표:")
            print(result[['close', 'price_momentum', 'volume_momentum', 'MA20']].tail(1))
        
        print("\n지표 계산 테스트 완료")

    def test_rank_stocks(self):
        """종목 순위 결정 메서드 테스트"""
        print("\n=== 종목 순위 테스트 ===")
        
        # 모든 종목의 모멘텀 스코어 계산
        momentum_scores = {}
        for code, data in self.test_data.items():
            if code == list(self.test_data.keys())[-1]:  # 마지막 종목은 안전자산
                continue
            
            result = self.strategy.calculate_indicators(data.copy())
            if not result.empty:
                momentum_scores[code] = result['price_momentum'].iloc[-1]
        
        # 순위 정렬
        ranked_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\n종목별 모멘텀 순위:")
        for rank, (code, score) in enumerate(ranked_stocks, 1):
            print(f"{rank}위: 종목 {code}, 모멘텀 스코어: {score:.2f}")
        
        # 상위 종목들의 모멘텀이 양수인지 확인
        if ranked_stocks:
            self.assertGreater(ranked_stocks[0][1], 0, "최상위 종목의 모멘텀이 양수여야 합니다")
        
        print("\n종목 순위 테스트 완료")

    def test_generate_signals(self):
        """매매 신호 생성 메서드 테스트"""
        print("\n=== 매매 신호 생성 테스트 ===")
        
        for code, data in self.test_data.items():
            result = self.strategy.calculate_indicators(data.copy())
            signals = self.strategy.generate_signals(result)
            
            # 신호 형식 검증
            self.assertIsInstance(signals, dict, "신호는 딕셔너리 형태여야 합니다")
            self.assertIn('signal', signals, "신호에 'signal' 키가 없습니다")
            self.assertIn('reason', signals, "신호에 'reason' 키가 없습니다")
            
            print(f"\n종목코드 {code} 신호:")
            print(f"신호: {signals['signal']}")
            print(f"이유: {signals['reason']}")
            print(f"현재가: {data['close'].iloc[-1]:,.0f}")
            print(f"모멘텀 스코어: {result['price_momentum'].iloc[-1]:.2f}")
        
        print("\n매매 신호 생성 테스트 완료")

    def test_risk_management(self):
        """리스크 관리 로직 테스트"""
        print("\n=== 리스크 관리 테스트 ===")
        
        for code, data in self.test_data.items():
            if code == list(self.test_data.keys())[-1]:  # 안전자산 제외
                continue
                
            current_price = data['close'].iloc[-1]
            entry_price = current_price * 0.95  # 5% 낮은 가격으로 매수 가정
            
            # 포지션 설정
            self.strategy.position_info[code] = {
                'entry_price': entry_price,
                'highest_price': current_price
            }
            
            # 손절 조건 테스트
            stop_loss_price = entry_price * (1 + self.strategy.params['stop_loss'])
            trailing_stop_price = current_price * (1 + self.strategy.params['trailing_stop'])
            
            print(f"\n종목코드 {code} 리스크 관리:")
            print(f"매수가: {entry_price:,.0f}")
            print(f"현재가: {current_price:,.0f}")
            print(f"손절가: {stop_loss_price:,.0f}")
            print(f"트레일링 스탑: {trailing_stop_price:,.0f}")
            print(f"수익률: {((current_price/entry_price)-1)*100:.2f}%")
            
            # 손절 신호 확인
            if current_price <= stop_loss_price:
                print("→ 손절 조건 발생")
            elif current_price <= trailing_stop_price:
                print("→ 트레일링 스탑 조건 발생")
            else:
                print("→ 홀딩 유지")
        
        print("\n리스크 관리 테스트 완료")

if __name__ == '__main__':
    unittest.main() 