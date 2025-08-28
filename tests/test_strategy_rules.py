import unittest
from unittest.mock import MagicMock
from datetime import date, timedelta
import sys
import os

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from strategies.strategy import DailyStrategy
from strategies.sma_daily import SMADaily # 구체적인 테스트를 위해 SMADaily 사용

class TestDailyStrategyRules(unittest.TestCase):
    
    def setUp(self):
        """각 테스트 전에 필요한 객체들을 설정합니다."""
        self.mock_broker = MagicMock()
        # [추가] TypeError 방지를 위해 commission_rate 속성 설정
        self.mock_broker.commission_rate = 0.0015
        
        self.mock_data_store = {'daily': {}}
        
        self.strategy = SMADaily(self.mock_broker, self.mock_data_store)
        self.current_date = date(2025, 8, 21)

    def test_01_min_holding_period_blocks_sell_signal(self):
        """
        [Red] 최소 보유기간(min_position_period)이 지나지 않은 종목은
        매도 후보가 되어도 최종적으로 'hold' 신호를 생성해야 합니다.
        """
        print("--- 테스트 1: 최소 보유기간 매도 방어 로직 ---")
        
        # --- Arrange (준비) ---
        # 1. 전략 파라미터 설정: 최소 보유기간 3일
        self.strategy.strategy_params['min_position_period'] = 3
        
        # 2. Broker 모의 설정: 'A00001' 종목을 1일 전에 매수하여 보유 중
        entry_date_1_day_ago = self.current_date - timedelta(days=1)
        self.mock_broker.get_current_positions.return_value = {
            'A00001': {'entry_date': entry_date_1_day_ago}
        }
        
        # 3. 신호 계산 결과 모의 설정: 'A00001'이 매도 후보로 선정됨
        buy_candidates = set()
        sell_candidates = {'A00001'}
        signal_attributes = {}
        
        # --- Act (실행) ---
        # 최종 신호 생성 로직 호출
        self.strategy._generate_signals(
            self.current_date, buy_candidates, sell_candidates, signal_attributes, 1_000_000
        )
        
        # --- Assert (검증) ---
        # 보유기간이 1일밖에 되지 않았으므로, 최종 신호는 'sell'이 아닌 'hold'여야 함
        final_signal_type = self.strategy.signals.get('A00001', {}).get('signal_type')
        print(f"A00001 최종 신호: {final_signal_type}")
        self.assertEqual(final_signal_type, 'hold', "최소 보유기간 미만인 종목이 매도 처리되었습니다.")

    def test_02_max_position_count_limits_buy_signals(self):
        """
        [Red] 최대 보유 종목 수(max_position_count)에 도달하면
        새로운 매수 후보가 있어도 추가 매수 신호를 생성하지 않아야 합니다.
        """
        print("--- 테스트 2: 최대 보유 종목 수 매수 제한 로직 ---")
        
        # --- Arrange (준비) ---
        # 1. 전략 파라미터 설정: 최대 보유 종목 2개
        self.strategy.strategy_params['max_position_count'] = 2
        
        # 2. Broker 모의 설정: 이미 'A00001' 1종목 보유 중
        self.mock_broker.get_current_positions.return_value = {'A00001': {}}
        
        # 3. 신호 계산 결과 모의 설정: 'A00002', 'A00003' 2개의 신규 매수 후보 발생 (스코어 순)
        buy_candidates = {'A00002', 'A00003'}
        sell_candidates = set()
        signal_attributes = {
            'A00002': {'score': 100, 'target_price': 10000}, # 더 높은 점수
            'A00003': {'score': 90, 'target_price': 20000}
        }
        
        # --- Act (실행) ---
        self.strategy._generate_signals(
            self.current_date, buy_candidates, sell_candidates, signal_attributes, 5_000_000
        )
        
        # --- Assert (검증) ---
        # 이미 1개 보유 + 신규 1개 매수 = 최대 2개. 스코어가 더 높은 'A00002'만 매수되어야 함.
        buy_signals = [s for s in self.strategy.signals.values() if s.get('signal_type') == 'buy']
        print(f"생성된 신규 매수 신호 개수: {len(buy_signals)}")
        if buy_signals:
            print(f"매수 신호 종목: {buy_signals[0]['stock_code']}")

        self.assertEqual(len(buy_signals), 1, "최대 보유 종목 수를 초과하여 매수 신호가 생성되었습니다.")
        self.assertEqual(buy_signals[0]['stock_code'], 'A00002', "스코어가 가장 높은 종목이 매수되지 않았습니다.")

if __name__ == '__main__':
    unittest.main()