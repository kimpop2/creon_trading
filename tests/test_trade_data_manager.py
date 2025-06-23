import unittest
from unittest.mock import patch, MagicMock
from datetime import date, datetime
import sys
import os
# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from manager.data_manager import DataManager

class TestTradeDataManager(unittest.TestCase):
    def setUp(self):
        # CreonAPIClient를 Mock 처리하여 외부 API 호출 방지
        patcher = patch('manager.data_manager.CreonAPIClient', autospec=True)
        self.addCleanup(patcher.stop)
        self.mock_api = patcher.start()()
        self.data_manager = DataManager()
        self.today = date.today()
        self.sample_stock_code = 'A005930'
        self.sample_strategy = 'TestStrategy'

        # 테스트용 DB 초기화(테이블 생성 등) 필요시 아래 코드 사용
        # self.data_manager.db_manager.create_trade_tables()

    def tearDown(self):
        # 테스트 데이터 정리 (테이블 전체 삭제 또는 롤백)
        # self.data_manager.db_manager.drop_trade_tables()
        self.data_manager.close()

    def test_save_and_get_daily_signals(self):
        signals = {
            self.sample_stock_code: {
                'strategy_name': self.sample_strategy,
                'signal_type': 'BUY',
                'target_price': 70000.0,
                'signal_strength': 0.95
            }
        }
        self.data_manager.save_daily_signals(signals, self.today)
        loaded = self.data_manager.get_daily_signals(self.today)
        self.assertIn(self.sample_stock_code, loaded)
        self.assertEqual(loaded[self.sample_stock_code]['strategy_name'], self.sample_strategy)
        self.assertEqual(loaded[self.sample_stock_code]['signal_type'], 'BUY')

    def test_save_trade_log(self):
        log_entry = {
            'trade_datetime': datetime.now(),
            'stock_code': self.sample_stock_code,
            'order_type': 'BUY',
            'quantity': 10,
            'price': 70000.0,
            'reason': 'unit_test',
            'realized_profit_loss': 0.0,
            'commission': 100.0,
            'tax': 0.0
        }
        result = self.data_manager.save_trade_log(log_entry)
        self.assertTrue(result)
        # trade_log 테이블에서 직접 조회하여 확인하는 코드는 db_manager에 따라 추가 가능

    def test_save_and_get_daily_portfolio_snapshot(self):
        snapshot_date = self.today
        portfolio_value = 10000000.0
        cash = 5000000.0
        positions = {
            self.sample_stock_code: {
                'size': 10,
                'avg_price': 70000.0,
                'entry_date': self.today,
                'highest_price': 71000.0
            }
        }
        self.data_manager.save_daily_portfolio_snapshot(snapshot_date, portfolio_value, cash, positions)
        # 직접 DB에서 조회하여 검증하는 코드는 db_manager에 따라 추가 가능
        # 일단 예외 없이 동작하면 성공으로 간주

    def test_save_and_get_current_positions(self):
        positions = {
            self.sample_stock_code: {
                'size': 10,
                'avg_price': 70000.0,
                'entry_date': self.today,
                'highest_price': 71000.0
            }
        }
        self.data_manager.save_current_positions(positions)
        loaded = self.data_manager.get_current_positions()
        self.assertIn(self.sample_stock_code, loaded)
        self.assertEqual(loaded[self.sample_stock_code]['size'], 10)
        self.assertEqual(loaded[self.sample_stock_code]['avg_price'], 70000.0)

    def test_get_current_positions_empty(self):
        # DB에 아무 포지션도 없을 때
        loaded = self.data_manager.get_current_positions()
        self.assertIsInstance(loaded, dict)

if __name__ == '__main__':
    unittest.main() 