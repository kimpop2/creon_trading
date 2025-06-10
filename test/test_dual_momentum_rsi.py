# test/test_dual_momentum_rsi.py

"""
파일명: test_dual_momentum_rsi.py
설명: 듀얼 모멘텀 일봉 전략과 RSI 분봉 전략의 통합 백테스트 실행 진입 파일
작성일: 2024-03-20 (모듈화 버전)
"""

import sys
import os
import datetime
import logging
import pandas as pd
import time # API 호출 지연을 위한 time 모듈 추가
from typing import Dict, List

# 상위 디렉토리 추가 (상대 임포트를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 모듈 임포트
from api.creon_api import CreonAPIClient # CreonAPIClient 임포트 경로 변경
from backtest.backtester import Backtester
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.rsi_minute import RSIMinute

# --- 로깅 설정 (원본 소스 유지) ---
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
            'momentum_period': 20,          # 모멘텀 계산 기간 (거래일)
            'rebalance_weekday': 4,         # 리밸런싱 요일 (0: 월요일, 4: 금요일)
            'num_top_stocks': 7,            # 상위 7종목 선택
            'safe_asset_code': 'A439870',   # 안전자산 코드 (국고채 ETF)
            'stop_loss_percent': -0.05,     # 개별 종목 단순 손절 비율
            'trailing_stop_percent': -0.03, # 개별 종목 트레일링 스탑 비율 (현재 Backtester에서 미구현, placeholder)
            'initial_holding_stop_loss_days': 5, # 초기 보유 기간 (며칠 이내)
            'initial_holding_stop_loss_percent': -0.03, # 초기 보유 기간 손절 비율
            'portfolio_stop_loss_percent': -0.10, # 전체 포트폴리오 손절 비율
            'max_simultaneous_losses': 3    # 동시에 손실 중인 종목 최대 개수
        }
        
        # RSI 전략 파라미터 업데이트
        self.minute_params = {
            'rsi_period': 45,            # 분봉 RSI 기간 (60분 → 45분)
            'rsi_upper': 65,             # RSI 과매수 기준 (70 → 65)
            'rsi_lower': 35,             # RSI 과매도 기준 (30 → 35)
            'morning_entry_hour': 10,    # 매수 시작 시간 (시)
            'morning_entry_minute': 0,   # 매수 시작 시간 (분)
            'morning_exit_hour': 9,      # 매도 시작 시간 (시)
            'morning_exit_minute': 5,    # 매도 시작 시간 (분)
            'force_entry_hour': 15,      # 강제 매수 시간 (시)
            'force_entry_minute': 20,    # 강제 매수 시간 (분)
            'force_exit_hour': 15,       # 강제 매도 시간 (시)
            'force_exit_minute': 25      # 강제 매도 시간 (분)
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
        
    def prepare_backtest_data(self, stock_codes: List[str], data_fetch_start_date: str, data_fetch_end_date: str) -> None:
        """
        백테스트에 필요한 데이터를 준비합니다.
        Args:
            stock_codes: 대상 종목 코드 리스트
            data_fetch_start_date: 데이터 조회 시작일 (YYYYMMDD) - 모멘텀 계산 기간 고려
            data_fetch_end_date: 데이터 조회 종료일 (YYYYMMDD) - 백테스트 종료일까지
        """
        # 안전자산 데이터 로드 (모멘텀 계산에 필요하므로 가장 먼저 로드)
        safe_asset_code = self.daily_params['safe_asset_code']
        logging.info(f"안전자산 ({safe_asset_code})의 일봉 데이터를 Creon API에서 가져오는 중...")
        # get_daily_ohlcv는 from_date, to_date를 모두 받으므로 수정
        safe_asset_df = self.creon_api.get_daily_ohlcv(safe_asset_code, data_fetch_start_date, data_fetch_end_date)
        time.sleep(self.creon_api.request_interval) # API 호출 제한 방지를 위한 대기
        
        if not safe_asset_df.empty:
            self.backtester.add_daily_data(safe_asset_code, safe_asset_df)
            logging.info(f"안전자산 데이터 로드 완료. 데이터 수: {len(safe_asset_df)}행")
        else:
            logging.error("안전자산 데이터를 로드할 수 없습니다. 절대 모멘텀 비교에 영향을 미칠 수 있습니다.")
            sys.exit(1) # 안전자산 데이터 없으면 백테스트 진행 불가

        # 나머지 대상 종목 데이터 로드
        for code in stock_codes:
            # 일봉 데이터 로드
            logging.info(f"{code} 종목의 일봉 데이터를 가져오는 중... (기간: {data_fetch_start_date} ~ {data_fetch_end_date})")
            # get_daily_ohlcv는 from_date, to_date를 모두 받으므로 수정
            daily_df = self.creon_api.get_daily_ohlcv(code, data_fetch_start_date, data_fetch_end_date)
            time.sleep(self.creon_api.request_interval) # API 호출 제한 방지를 위한 대기
            
            if daily_df.empty:
                logging.warning(f"{code} 종목의 일봉 데이터를 가져올 수 없습니다. 해당 종목을 건너뜁니다.")
                continue
                
            self.backtester.add_daily_data(code, daily_df)
            logging.info(f"{code} 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
    
    def run_backtest(self, backtest_start_dt: datetime.datetime, backtest_end_dt: datetime.datetime):
        """
        백테스트를 실행합니다.
        Args:
            backtest_start_dt: 백테스트 시작 시간
            backtest_end_dt: 백테스트 종료 시간
        """
        # 백테스터의 run 메서드 호출
        portfolio_values, metrics = self.backtester.run(backtest_start_dt, backtest_end_dt)
        
        # 결과는 Backtester 클래스에서 이미 출력되므로 여기서는 추가 출력 없음
        return portfolio_values, metrics

def main():
    """메인 실행 함수"""
    logging.info("듀얼 모멘텀 RSI 통합 백테스트 스크립트를 실행합니다.")
    
    # CreonAPIClient 인스턴스 생성
    try:
        creon_api = CreonAPIClient()
        logging.debug(f"CreonAPI 연결 상태: {creon_api.connected}")
        if not creon_api.connected:
            logging.error("Creon API에 연결할 수 없습니다. 프로그램을 종료합니다.")
            sys.exit(1)
    except ConnectionError as e: # CreonAPIClient에서 ConnectionError를 발생시키므로 이에 맞춰 예외 처리
        logging.error(f"CreonAPI 초기화 중 연결 오류 발생: {str(e)}. 프로그램을 종료합니다.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"CreonAPI 초기화 중 예기치 않은 오류 발생: {str(e)}. 프로그램을 종료합니다.")
        sys.exit(1)

    # 테스트 대상 종목 (예시: 기존 스크립트의 섹터별 종목 리스트를 활용)
    # 실제 백테스트에서는 더 많은 종목을 추가하거나, 특정 기준에 따라 선정할 수 있습니다.
    sector_stocks = {
        '반도체': [('삼성전자', 'IT'), ('SK하이닉스', 'IT'), ('DB하이텍', 'IT')],
        '2차전지': [('LG에너지솔루션', '2차전지'), ('삼성SDI', '2차전지')],
        '바이오': [('삼성바이오로직스', '바이오'), ('셀트리온', '바이오')],
        '플랫폼/인터넷': [('NAVER', 'IT'), ('카카오', 'IT')],
        '자동차': [('현대차', '자동차'), ('기아', '자동차')],
        '철강/화학': [('POSCO홀딩스', '철강'), ('롯데케미칼', '화학')],
        '금융': [('KB금융', '금융'), ('신한지주', '금융')],
        # ... 필요에 따라 더 많은 섹터/종목 추가
    }

    # 모든 종목명을 하나의 리스트로 변환
    stock_names_to_fetch = []
    for sector, stocks_in_sector in sector_stocks.items():
        for stock_name, _ in stocks_in_sector:
            stock_names_to_fetch.append(stock_name)

    stock_codes_for_backtest = []
    for name in stock_names_to_fetch:
        code = creon_api.get_stock_code(name)
        if code:
            stock_codes_for_backtest.append(code)
            logging.info(f"'{name}' (코드: {code}) 종목이 백테스트 대상에 추가됩니다.")
        else:
            logging.warning(f"'{name}' 종목의 코드를 찾을 수 없습니다. 해당 종목을 건너뜁니다.")
    
    if not stock_codes_for_backtest:
        logging.error("백테스트를 위한 유효한 종목 코드가 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    # 백테스트 기간 설정
    # 모멘텀 계산에 필요한 기간(momentum_period)을 고려하여 data_fetch_start_date를 설정합니다.
    # 예: momentum_period가 20일이면, 백테스트 시작일로부터 최소 20거래일 이전 데이터가 필요합니다.
    # 넉넉하게 backtest_start_date보다 90일 전부터 데이터를 가져옵니다.
    backtest_start_date = datetime.datetime(2025, 4, 1, 9, 0, 0)
    backtest_end_date = datetime.datetime(2025, 6, 4, 15, 30, 0)

    # 데이터 가져올 시작 날짜 (넉넉하게 백테스트 시작일보다 2-3개월 이전)
    # CreonAPIClient의 _get_price_data 메서드가 YYYYMMDD 문자열을 받으므로 이에 맞게 포맷팅
    data_fetch_start = (backtest_start_date - datetime.timedelta(days=90)).strftime('%Y%m%d')
    data_fetch_end = backtest_end_date.strftime('%Y%m%d')

    # 백테스트 인스턴스 생성 및 실행
    backtest = DualMomentumRSIBacktest(creon_api)
    
    logging.info(f"데이터 로드 기간: {data_fetch_start} ~ {data_fetch_end}")
    backtest.prepare_backtest_data(stock_codes_for_backtest, data_fetch_start, data_fetch_end)
    
    logging.info(f"백테스트 실행 기간: {backtest_start_date.strftime('%Y-%m-%d')} ~ {backtest_end_date.strftime('%Y-%m-%d')}")
    portfolio_values, metrics = backtest.run_backtest(backtest_start_date, backtest_end_date)

    logging.info("백테스트 완료.")

if __name__ == '__main__':
    main()