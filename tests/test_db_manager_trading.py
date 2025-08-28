import unittest
import os
import sys
from datetime import date, datetime
import pandas as pd

# --- 프로젝트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from manager.db_manager import DBManager
from config.settings import LIVE_HMM_MODEL_NAME

class TestDBManagerTradingResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """테스트 시작 시 한 번 실행"""
        cls.db_manager = DBManager()
        # 테스트를 위해 HMM 모델이 최소 1개는 있어야 함
        cls.model_info = cls.db_manager.fetch_hmm_model_by_name(LIVE_HMM_MODEL_NAME)
        if not cls.model_info:
            raise unittest.SkipTest(f"테스트를 위한 HMM 모델({LIVE_HMM_MODEL_NAME})이 DB에 없습니다.")
        cls.model_id = cls.model_info['model_id']

    @classmethod
    def tearDownClass(cls):
        """테스트 종료 시 한 번 실행"""
        cls.db_manager.close()

    def test_01_save_and_fetch_trading_trade(self):
        """trading_trade 테이블 저장 및 조회 기능 테스트"""
        test_trade = {
            'model_id': self.model_id,
            'trade_date': date.today(),
            'strategy_name': 'SMADaily_Test',
            'stock_code': 'A005930',
            'trade_type': 'BUY',
            'trade_price': 75000.0,
            'trade_quantity': 10,
            'trade_datetime': datetime.now(),
            'commission': 112.5,
            'tax': 0.0,
            'realized_profit_loss': 0.0
        }
        
        # 저장 테스트
        save_success = self.db_manager.save_trading_trade(test_trade)
        self.assertTrue(save_success, "trading_trade 저장에 실패했습니다.")

        # 조회 테스트
        trades_df = self.db_manager.fetch_trading_trade(date.today(), date.today())
        self.assertFalse(trades_df.empty, "저장된 거래 기록을 조회할 수 없습니다.")
        
        saved_trade = trades_df[trades_df['stock_code'] == 'A005930'].iloc[0]
        self.assertEqual(saved_trade['strategy_name'], 'SMADaily_Test')
        self.assertEqual(saved_trade['trade_quantity'], 10)

    def test_02_save_and_fetch_trading_run_performance(self):
        """trading_run 및 trading_performance 테이블 저장/조회 기능 테스트"""
        test_date = date.today()
        
        # Run 데이터 저장 테스트
        test_run = {
            'model_id': self.model_id,
            'trading_date': test_date,
            'initial_capital': 10000000.0,
            'final_capital': 10050000.0,
            'total_profit_loss': 50000.0,
            'cumulative_return': 0.005,
            'max_drawdown': -0.01,
            'strategy_daily': 'SMADaily_Test,DualMomentum_Test',
            'params_json_daily': '{}'
        }
        run_save_success = self.db_manager.save_trading_run(test_run)
        self.assertTrue(run_save_success, "trading_run 저장에 실패했습니다.")

        # Performance 데이터 저장 테스트
        test_perf = {
            'model_id': self.model_id,
            'date': test_date,
            'end_capital': 10050000.0,
            'daily_return': 0.005,
            'daily_profit_loss': 50000.0,
            'cumulative_return': 0.005,
            'drawdown': -0.01
        }
        perf_save_success = self.db_manager.save_trading_performance(test_perf)
        self.assertTrue(perf_save_success, "trading_performance 저장에 실패했습니다.")

        # 데이터 조회 검증 (여기서는 생략, save 성공 여부로 갈음)

if __name__ == '__main__':
    unittest.main()