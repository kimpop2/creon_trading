import unittest
from datetime import date, datetime, timedelta
from manager.db_manager import DBManager
import pandas as pd

class TestTradeDBManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DBManager()
        cls.db.create_trade_tables()

    @classmethod
    def tearDownClass(cls):
        #cls.db.drop_trade_tables()
        pass

    def test_daily_signals_insert_and_fetch(self):
        # insert
        signals = {
            'A000001': {
                'signal_type': 'BUY',
                'signal_price': 10000,
                'volume_ratio': 0.5,
                'strategy_name': 'test_strategy',
                'params_json': '{}'
            }
        }
        signal_date = date.today()
        self.db.save_daily_signals(signals, signal_date)
        # fetch (직접 쿼리)
        df = self.db.execute_sql(f"SELECT * FROM daily_signals WHERE signal_date = '{signal_date.isoformat()}'")
        self.assertIsNotNone(df)
        rows = df.fetchall()
        self.assertTrue(any(row['stock_code'] == 'A000001' for row in rows))

    def test_daily_portfolio_snapshot_insert_and_fetch(self):
        snapshot_date = date.today()
        portfolio_value = 1000000
        cash = 500000
        positions = {'A000001': {'size': 10, 'avg_price': 10000}}
        self.db.save_daily_portfolio_snapshot(snapshot_date, portfolio_value, cash, positions)
        # fetch
        result = self.db.execute_sql(f"SELECT * FROM daily_portfolio_snapshot WHERE snapshot_date = '{snapshot_date.isoformat()}'")
        self.assertIsNotNone(result)
        rows = result.fetchall()
        self.assertTrue(any(row['snapshot_date'] == snapshot_date for row in rows))

    def test_current_positions_insert_and_fetch(self):
        positions = {
            'A000001': {
                'size': 10,
                'avg_price': 10000,
                'entry_date': date.today(),
                'highest_price': 12000
            }
        }
        self.db.save_current_positions(positions)
        # fetch
        result = self.db.execute_sql("SELECT * FROM current_positions WHERE stock_code = 'A000001'")
        self.assertIsNotNone(result)
        rows = result.fetchall()
        self.assertTrue(any(row['stock_code'] == 'A000001' for row in rows))

    def test_trade_log_insert(self):
        log_entry = {
            'order_time': datetime.now(),
            'stock_code': 'A000001',
            'order_type': 'BUY',
            'order_price': 10000,
            'order_quantity': 10,
            'executed_price': 10000,
            'executed_quantity': 10,
            'commission': 10,
            'tax': 5,
            'net_amount': 99985,
            'order_status': 'FILLED',
            'creon_order_id': 'TEST123',
            'message': '테스트'
        }
        # trade_log 테이블에 insert
        df = pd.DataFrame([log_entry])
        self.db.insert_df_to_db('trade_log', df, option='append', is_index=False)
        # fetch
        result = self.db.execute_sql("SELECT * FROM trade_log WHERE stock_code = 'A000001'")
        self.assertIsNotNone(result)
        rows = result.fetchall()
        self.assertTrue(any(row['stock_code'] == 'A000001' for row in rows))

if __name__ == '__main__':
    unittest.main() 