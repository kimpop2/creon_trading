import unittest
from unittest.mock import MagicMock
from datetime import date, datetime, time
import pandas as pd
import sys
import os

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 테스트 대상 모듈 임포트
from trading.hmm_backtest import HMMBacktest
from strategies.sma_daily import SMADaily # 아무 DailyStrategy나 임포트
from strategies.pass_minute import PassMinute
from config.settings import INITIAL_CASH, STOP_LOSS_PARAMS

class TestBacktestPassMinute(unittest.TestCase):

    def setUp(self):
        """백테스트 환경 설정"""
        self.mock_manager = MagicMock()
        self.start_date = date(2025, 8, 21)
        self.end_date = date(2025, 8, 21)
        
        # HMMBacktest 인스턴스 생성 (DB 저장 안 함)
        self.backtest = HMMBacktest(
            manager=self.mock_manager,
            initial_cash=INITIAL_CASH,
            start_date=self.start_date,
            end_date=self.end_date,
            save_to_db=False
        )
        
        # PassMinute 전략 설정
        minute_strategy = PassMinute(self.backtest.broker, self.backtest.data_store)
        self.backtest.set_strategies(daily_strategies=[], minute_strategy=minute_strategy, stop_loss_params=STOP_LOSS_PARAMS)
        # [수정] daily_strategies를 빈 리스트가 아닌, 실제 전략 객체로 설정
        daily_strategy = SMADaily(self.backtest.broker, self.backtest.data_store)
        self.backtest.set_strategies(
            daily_strategies=[daily_strategy],  # <-- 이 부분을 수정
            minute_strategy=minute_strategy, 
            stop_loss_params=STOP_LOSS_PARAMS
        )
        # 테스트용 데이터 저장소 설정
        self.stock_code = 'A005930'
        self.market_open_time = self.backtest.market_open_time
        self.market_close_time = self.backtest.market_close_time

        # Mock DB가 영업일을 반환하도록 설정
        self.mock_manager.fetch_market_calendar.return_value = pd.DataFrame({
            'date': [pd.Timestamp(self.start_date)],
            'is_holiday': [0]
        })

    def test_01_stop_loss_at_market_open(self):
        """[Red] 시가가 손절 라인 아래일 때, 장 시작 시점에 손절매가 실행되어야 함"""
        print("\n--- 테스트 1: PassMinute 시가 손절매 ---")
        
        # --- Arrange (준비) ---
        # 1. 전일에 10,000원에 매수하여 보유 중인 상황을 가정
        entry_price = 10000
        stop_loss_price = entry_price * (1 + STOP_LOSS_PARAMS['stop_loss_pct'] / 100) # 9500원
        self.backtest.broker.execute_order(self.stock_code, 'buy', entry_price, 10, datetime(2025, 8, 20))
        
        # 2. 당일 시가가 손절 가격(9500원)보다 낮게 시작하도록 데이터 설정
        open_price = 9000
        self.backtest.data_store['daily'][self.stock_code] = pd.DataFrame({
            'open': [open_price], 'high': [9200], 'low': [8900], 'close': [9100]
        }, index=[pd.Timestamp(self.start_date)])
        
        # --- Act (실행) ---
        # 백테스트 실행 (run() 대신 reset_and_rerun() 사용이 더 적합할 수 있으나, 현재 구조상 run() 테스트)
        self.backtest.run()
        
        # --- Assert (검증) ---
        sell_log = [log for log in self.backtest.broker.transaction_log if log['trade_type'] == 'SELL']
        self.assertEqual(len(sell_log), 1, "매도 거래가 실행되지 않았습니다.")
        
        # 시가(open_price)로 손절매가 실행되었는지 확인
        self.assertEqual(sell_log[0]['trade_price'], open_price, "손절매 가격이 시가와 다릅니다.")
        
        # 장 시작 시간에 거래가 기록되었는지 확인
        expected_time = datetime.combine(self.start_date, self.market_open_time)
        self.assertEqual(sell_log[0]['trade_datetime'], expected_time, "손절매 시간이 장 시작 시간과 다릅니다.")
        print(f"거래 로그 확인: {sell_log[0]['trade_datetime']}에 {sell_log[0]['trade_price']} 가격으로 매도 완료.")

    def test_02_stop_loss_at_market_close(self):
        """[Red] 종가가 손절 라인 아래일 때, 장 마감 시점에 손절매가 실행되어야 함"""
        print("\n--- 테스트 2: PassMinute 종가 손절매 ---")

        # --- Arrange (준비) ---
        entry_price = 10000
        stop_loss_price = entry_price * (1 + STOP_LOSS_PARAMS['stop_loss_pct'] / 100) # 9500원
        self.backtest.broker.execute_order(self.stock_code, 'buy', entry_price, 10, datetime(2025, 8, 20))

        # 시가는 손절가보다 높지만, 종가는 낮은 상황 설정
        close_price = 9000
        self.backtest.data_store['daily'][self.stock_code] = pd.DataFrame({
            'open': [9800], 'high': [9900], 'low': [8900], 'close': [close_price]
        }, index=[pd.Timestamp(self.start_date)])
        
        # --- Act (실행) ---
        self.backtest.run()

        # --- Assert (검증) ---
        sell_log = [log for log in self.backtest.broker.transaction_log if log['trade_type'] == 'SELL']
        self.assertEqual(len(sell_log), 1, "매도 거래가 실행되지 않았습니다.")

        # 종가(close_price)로 손절매가 실행되었는지 확인
        self.assertEqual(sell_log[0]['trade_price'], close_price, "손절매 가격이 종가와 다릅니다.")

        # 장 마감 시간에 거래가 기록되었는지 확인
        expected_time = datetime.combine(self.start_date, self.market_close_time)
        self.assertEqual(sell_log[0]['trade_datetime'], expected_time, "손절매 시간이 장 마감 시간과 다릅니다.")
        print(f"거래 로그 확인: {sell_log[0]['trade_datetime']}에 {sell_log[0]['trade_price']} 가격으로 매도 완료.")

if __name__ == '__main__':
    unittest.main()