"""
파일명: test_dual_momentum_rsi2.py
설명: 듀얼 모멘텀 일봉 전략과 RSI 분봉 전략의 통합 테스트 (개선된 버전)
작성일: 2024-03-20
"""

import sys
import os
import datetime
import logging
import pandas as pd
from typing import Dict, List

# 상위 디렉토리 추가 (상대 임포트를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.creon_api import CreonAPIClient
from backtest.backtester import Backtester
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.rsi_minute import RSIMinute

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class DualMomentumRSIBacktest:
    """듀얼 모멘텀과 RSI 전략을 결합한 백테스트 클래스"""
    
    def __init__(self, creon_api_client: CreonAPIClient):
        """
        Args:
            creon_api_client: 크레온 API 클라이언트 인스턴스
        """
        self.creon_api = creon_api_client
        
        # 전략 파라미터 설정
        self.daily_params = {
            'momentum_period': 20,           # 모멘텀 계산 기간 (거래일)
            'rebalance_weekday': 4,         # 리밸런싱 요일 (0: 월요일, 4: 금요일)
            'num_top_stocks': 7,            # 상위 7종목 선택
            'safe_asset_code': 'A439870',   # 안전자산 코드 (국고채 ETF)
        }
        
        # RSI 전략 파라미터 업데이트
        self.minute_params = {
            'rsi_period': 45,          # 분봉 RSI 기간 (60분 → 45분)
            'rsi_upper': 65,           # RSI 과매수 기준 (70 → 65)
            'rsi_lower': 35,           # RSI 과매도 기준 (30 → 35)
            'morning_entry_hour': 10,   # 매수 시작 시간 (시)
            'morning_entry_minute': 0,  # 매수 시작 시간 (분)
            'morning_exit_hour': 9,     # 매도 시작 시간 (시)
            'morning_exit_minute': 5,   # 매도 시작 시간 (분)
            'force_entry_hour': 15,     # 강제 매수 시간 (시)
            'force_entry_minute': 20,   # 강제 매수 시간 (분)
            'force_exit_hour': 15,      # 강제 매도 시간 (시)
            'force_exit_minute': 25     # 강제 매도 시간 (분)
        }
        
        # 전략 인스턴스 생성
        self.daily_strategy = DualMomentumDaily(params=self.daily_params)
        self.minute_strategy = RSIMinute(params=self.minute_params)
        
        # 백테스터 인스턴스 생성 (전략을 생성자에서 받음)
        self.backtester = Backtester(
            creon_api_client,
            initial_cash=10_000_000,
            daily_strategy=self.daily_strategy,
            minute_strategy=self.minute_strategy
        )
        
    def prepare_backtest_data(self, stock_codes: List[str], start_date: str, end_date: str) -> None:
        """백테스트에 필요한 데이터를 준비합니다.
        
        Args:
            stock_codes: 대상 종목 코드 리스트
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
        """
        for code in stock_codes:
            # 일봉 데이터 로드
            logging.info(f"{code} 종목의 일봉 데이터를 가져오는 중...")
            daily_df = self.creon_api.get_daily_ohlcv(code, start_date, end_date)
            if daily_df.empty:
                logging.warning(f"{code} 종목의 일봉 데이터를 가져올 수 없습니다.")
                continue
                
            self.backtester.add_daily_data(code, daily_df)
            logging.info(f"{code} 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
    
    def run_backtest(self, backtest_start_dt: datetime.datetime, backtest_end_dt: datetime.datetime) -> None:
        """백테스트를 실행합니다.
        
        Args:
            backtest_start_dt: 백테스트 시작 시간
            backtest_end_dt: 백테스트 종료 시간
        """
        # 백테스트 실행
        portfolio_values, metrics = self.backtester.run(backtest_start_dt, backtest_end_dt)
        
        # 결과는 Backtester 클래스에서 이미 출력되므로 여기서는 생략
        return portfolio_values, metrics
    
    def _print_backtest_results(self, portfolio_values: pd.Series, metrics: Dict) -> None:
        """백테스트 결과를 출력합니다."""
        # Backtester 클래스에서 이미 결과를 출력하므로 이 메서드는 사용하지 않음
        pass

def main():
    """메인 실행 함수"""
    logging.info("백테스트 시작")
    
    # CreonAPIClient 인스턴스 생성
    try:
        creon_api = CreonAPIClient()
        logging.debug(f"CreonAPI 연결 상태: {creon_api.connected}")
        if not creon_api.connected:
            logging.error("Creon API에 연결할 수 없습니다.")
            return
    except Exception as e:
        logging.error(f"CreonAPI 초기화 중 오류 발생: {str(e)}")
        return

    # 테스트 대상 종목 (예시)
    test_stocks = {
        'IT': ['삼성전자', 'SK하이닉스', 'NAVER'],
        '2차전지': ['LG에너지솔루션', '삼성SDI', 'SK이노베이션'],
        '바이오': ['삼성바이오로직스', '셀트리온'],
        '금융': ['KB금융', '신한지주', '하나금융지주']
    }

    # 종목 코드 변환
    stock_codes = []
    for sector, stocks in test_stocks.items():
        for stock_name in stocks:
            code = creon_api.get_stock_code(stock_name)
            if code:
                stock_codes.append(code)
                logging.info(f"'{stock_name}' (코드: {code}) 종목이 백테스트에 포함됩니다.")
            else:
                logging.warning(f"'{stock_name}' 종목의 코드를 찾을 수 없습니다.")

    if not stock_codes:
        logging.error("테스트할 종목이 없습니다.")
        return

    # 백테스트 기간 설정 (2025년으로 변경)
    start_date = '20250301'  # 데이터 준비 시작일
    backtest_start = datetime.datetime(2025, 4, 1, 9, 0, 0)  # 백테스트 시작 시간
    backtest_end = datetime.datetime(2025, 6, 4, 15, 30, 0)  # 백테스트 종료 시간

    # 백테스트 실행
    backtest = DualMomentumRSIBacktest(creon_api)
    backtest.prepare_backtest_data(stock_codes, start_date, backtest_end.strftime('%Y%m%d'))
    backtest.run_backtest(backtest_start, backtest_end)

if __name__ == '__main__':
    main() 