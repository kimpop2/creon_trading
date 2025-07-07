"""
DBManager 클래스 단위 테스트 (자동매매 관련)
"""

import unittest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta, time
import sys
import os

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from manager.db_manager import DBManager

class TestTradingDBManager(unittest.TestCase):
    def setUp(self):
        """테스트 설정: DBManager 인스턴스 생성 및 테스트 데이터 준비"""
        self.db_manager = DBManager()
        
        # 테스트 데이터 초기화 (겹치지 않도록 날짜를 조정하거나 명확히 함)
        self.test_trading_log_data_1 = {
            'order_id': 'ORD001', 'original_order_id': None, 'stock_code': 'A005930', 'stock_name': '삼성전자',
            'trading_date': date(2025, 1, 10), 'trading_time': time(9, 30, 0), 'order_type': 'buy',
            'order_price': 70000.0, 'order_quantity': 10, 'filled_price': 70000.0, 'filled_quantity': 10,
            'unfilled_quantity': 0, 'order_status': '체결', 'commission': 500.0, 'tax': 0.0,
            'net_amount': 700000.0 - 500.0
        }
        self.test_trading_log_data_2 = {
            'order_id': 'ORD002', 'original_order_id': 'ORD001', 'stock_code': 'A005930', 'stock_name': '삼성전자',
            'trading_date': date(2025, 1, 15), 'trading_time': time(10, 0, 0), 'order_type': 'sell',
            'order_price': 72000.0, 'order_quantity': 10, 'filled_price': 72000.0, 'filled_quantity': 10,
            'unfilled_quantity': 0, 'order_status': '체결', 'commission': 520.0, 'tax': 100.0,
            'net_amount': 720000.0 - 520.0 - 100.0
        }
        self.test_trading_log_data_3 = {
            'order_id': 'ORD003', 'original_order_id': None, 'stock_code': 'A000660', 'stock_name': 'SK하이닉스',
            'trading_date': date(2025, 1, 10), 'trading_time': time(11, 0, 0), 'order_type': 'buy',
            'order_price': 150000.0, 'order_quantity': 5, 'filled_price': 150000.0, 'filled_quantity': 5,
            'unfilled_quantity': 0, 'order_status': '체결', 'commission': 600.0, 'tax': 0.0,
            'net_amount': 750000.0 - 600.0
        }
        
        self.test_portfolio_data_1 = {
            'record_date': date(2025, 1, 10), 'total_capital': 10000000.0, 'cash_balance': 9000000.0,
            'total_asset_value': 1000000.0, 'daily_profit_loss': 50000.0, 'daily_return_rate': 0.005,
            'cumulative_profit_loss': 50000.0, 'cumulative_return_rate': 0.005, 'max_drawdown': 0.0
        }
        self.test_portfolio_data_2 = {
            'record_date': date(2025, 1, 11), 'total_capital': 10020000.0, 'cash_balance': 9020000.0,
            'total_asset_value': 1000000.0, 'daily_profit_loss': 20000.0, 'daily_return_rate': 0.002,
            'cumulative_profit_loss': 70000.0, 'cumulative_return_rate': 0.007, 'max_drawdown': 0.0
        }

        self.test_position_data_1 = {
            'stock_code': 'A005930', 'stock_name': '삼성전자', 'quantity': 10, 'sell_avail_qty': 9, 'average_buy_price': 70000.0,
            'current_price': 70500.0, 'eval_profit_loss': 5000.0, 'eval_return_rate': 0.0071,
            'entry_date': date(2025, 1, 10)
        }
        self.test_position_data_2 = {
            'stock_code': 'A000660', 'stock_name': 'SK하이닉스', 'quantity': 5, 'sell_avail_qty': 4, 'average_buy_price': 150000.0,
            'current_price': 151000.0, 'eval_profit_loss': 5000.0, 'eval_return_rate': 0.0066,
            'entry_date': date(2025, 1, 10)
        }
        # 업데이트 테스트를 위한 데이터
        self.test_position_data_1_updated = {
            'stock_code': 'A005930', 'stock_name': '삼성전자', 'quantity': 15, 'sell_avail_qty': 15, 'average_buy_price': 70200.0,
            'current_price': 71000.0, 'eval_profit_loss': 12000.0, 'eval_return_rate': 0.0114,
            'entry_date': date(2025, 1, 10) # entry_date는 보통 변하지 않음
        }
        
        self.test_signal_data_1 = {
            'signal_date': date(2025, 1, 10), 'stock_code': 'A005930', 'stock_name': '삼성전자',
            'signal_type': 'BUY', 'strategy_name': 'RSI_Strategy', 'target_price': 70000.0,
            'target_quantity': 10, 'is_executed': False, 'executed_order_id': None
        }
        self.test_signal_data_2 = {
            'signal_date': date(2025, 1, 10), 'stock_code': 'A000660', 'stock_name': 'SK하이닉스',
            'signal_type': 'BUY', 'strategy_name': 'MACD_Strategy', 'target_price': 150000.0,
            'target_quantity': 5, 'is_executed': False, 'executed_order_id': None
        }
        self.test_signal_data_3 = {
            'signal_date': date(2025, 1, 11), 'stock_code': 'A005930', 'stock_name': '삼성전자',
            'signal_type': 'SELL', 'strategy_name': 'RSI_Strategy', 'target_price': 72000.0,
            'target_quantity': 10, 'is_executed': False, 'executed_order_id': None
        }

        # 테스트 전 자동매매 관련 테이블들을 삭제하고 다시 생성하여 클린 상태를 보장
        self.db_manager.drop_trading_tables()
        self.db_manager.create_trading_tables()


    def tearDown(self):
        """테스트 정리: DB 연결 종료 및 테이블 정리 (선택 사항)"""
        # 테스트 후 모든 자동매매 관련 테이블을 삭제하여 다음 테스트에 영향이 없도록 함
        #self.db_manager.drop_trading_tables()
        self.db_manager.close()

    # --- 테이블 생성/삭제 테스트 ---
    def test_01_create_and_drop_trading_tables(self):
        """자동매매 관련 테이블 생성 및 삭제 테스트"""
        # setUp에서 이미 drop과 create를 수행하므로, 여기서는 존재 여부만 확인
        self.assertTrue(self.db_manager.check_table_exist('trading_log'))
        self.assertTrue(self.db_manager.check_table_exist('daily_portfolio'))
        self.assertTrue(self.db_manager.check_table_exist('current_positions'))
        self.assertTrue(self.db_manager.check_table_exist('daily_signals'))

        # 추가로 drop 시도 후 존재하지 않음을 확인
        result = self.db_manager.drop_trading_tables()
        self.assertTrue(result)
        self.assertFalse(self.db_manager.check_table_exist('trading_log'))
        self.assertFalse(self.db_manager.check_table_exist('daily_portfolio'))
        self.assertFalse(self.db_manager.check_table_exist('current_positions'))
        self.assertFalse(self.db_manager.check_table_exist('daily_signals'))
        
        # 다시 생성하여 다음 테스트에 영향 없도록 복구
        self.db_manager.create_trading_tables()


    # --- trading_log 테스트 ---
    def test_02_save_and_fetch_trading_log(self):
        """매매 로그 저장 및 조회 테스트"""
        # 로그 저장
        self.assertTrue(self.db_manager.save_trading_log(self.test_trading_log_data_1))
        self.assertTrue(self.db_manager.save_trading_log(self.test_trading_log_data_2))
        self.assertTrue(self.db_manager.save_trading_log(self.test_trading_log_data_3))

        # 특정 기간 전체 로그 조회
        fetched_logs_all = self.db_manager.fetch_trading_logs(date(2025, 1, 1), date(2025, 1, 31))
        self.assertIsInstance(fetched_logs_all, pd.DataFrame)
        self.assertEqual(len(fetched_logs_all), 3)
        self.assertEqual(fetched_logs_all.index.name, 'trading_datetime')
        
        # 특정 종목 로그 조회
        fetched_logs_samsung = self.db_manager.fetch_trading_logs(date(2025, 1, 1), date(2025, 1, 31), stock_code='A005930')
        self.assertIsInstance(fetched_logs_samsung, pd.DataFrame)
        self.assertEqual(len(fetched_logs_samsung), 2)
        self.assertTrue(all(fetched_logs_samsung['stock_code'] == 'A005930'))

        # 없는 기간/종목 조회
        fetched_logs_empty = self.db_manager.fetch_trading_logs(date(2026, 1, 1), date(2026, 1, 2))
        self.assertTrue(fetched_logs_empty.empty)
        
        fetched_logs_nonexistent_stock = self.db_manager.fetch_trading_logs(date(2025, 1, 1), date(2025, 1, 31), stock_code='A999999')
        self.assertTrue(fetched_logs_nonexistent_stock.empty)


    # --- daily_portfolio 테스트 ---
    def test_03_save_and_fetch_daily_portfolio(self):
        """일별 포트폴리오 저장 및 조회 테스트"""
        # 첫 번째 데이터 저장
        self.assertTrue(self.db_manager.save_daily_portfolio(self.test_portfolio_data_1))
        
        # 중복 날짜 업데이트 테스트
        updated_data = self.test_portfolio_data_1.copy()
        updated_data['total_capital'] = 10100000.0
        self.assertTrue(self.db_manager.save_daily_portfolio(updated_data))

        # 두 번째 데이터 저장
        self.assertTrue(self.db_manager.save_daily_portfolio(self.test_portfolio_data_2))

        # 전체 기간 조회
        fetched_portfolio_all = self.db_manager.fetch_daily_portfolio(date(2025, 1, 1), date(2025, 1, 31))
        self.assertIsInstance(fetched_portfolio_all, pd.DataFrame)
        self.assertEqual(len(fetched_portfolio_all), 2)
        # pd.Timestamp로 인덱스 접근
        self.assertEqual(fetched_portfolio_all.loc[pd.Timestamp(date(2025, 1, 10))]['total_capital'], 10100000.0)
        self.assertIn('record_date', fetched_portfolio_all.index.name)

        # 특정 날짜만 조회
        fetched_portfolio_single = self.db_manager.fetch_daily_portfolio(date(2025, 1, 10), date(2025, 1, 10))
        self.assertEqual(len(fetched_portfolio_single), 1)
        self.assertEqual(fetched_portfolio_single.iloc[0]['total_capital'], 10100000.0)

        # 최신 포트폴리오 조회
        latest_portfolio = self.db_manager.fetch_latest_daily_portfolio()
        self.assertIsNotNone(latest_portfolio)
        self.assertEqual(latest_portfolio['record_date'], date(2025, 1, 11))
        self.assertEqual(latest_portfolio['total_capital'], self.test_portfolio_data_2['total_capital'])
        

    # --- current_positions 테스트 ---
    def test_04_save_fetch_delete_current_position(self):
        """현재 보유 종목 저장, 조회, 삭제 테스트"""
        # 종목 저장
        self.assertTrue(self.db_manager.save_current_position(self.test_position_data_1))
        self.assertTrue(self.db_manager.save_current_position(self.test_position_data_2))
        
        # 전체 조회
        fetched_positions_all = self.db_manager.fetch_current_positions()
        self.assertIsInstance(fetched_positions_all, list)
        self.assertEqual(len(fetched_positions_all), 2)
        # 딕셔너리 리스트이므로 특정 값 확인
        samsung_pos = next((p for p in fetched_positions_all if p['stock_code'] == 'A005930'), None)
        self.assertIsNotNone(samsung_pos)
        self.assertEqual(samsung_pos['quantity'], 10)

        # 종목 업데이트 (quantity, avg_buy_price 변경)
        self.assertTrue(self.db_manager.save_current_position(self.test_position_data_1_updated))
        fetched_positions_after_update = self.db_manager.fetch_current_positions()
        samsung_pos_updated = next((p for p in fetched_positions_after_update if p['stock_code'] == 'A005930'), None)
        self.assertIsNotNone(samsung_pos_updated)
        self.assertEqual(samsung_pos_updated['quantity'], 15)
        self.assertEqual(samsung_pos_updated['average_buy_price'], 70200.0)

        # 종목 삭제
        self.assertTrue(self.db_manager.delete_current_position('A005930'))
        fetched_positions_after_delete = self.db_manager.fetch_current_positions()
        self.assertEqual(len(fetched_positions_after_delete), 1)
        self.assertEqual(fetched_positions_after_delete[0]['stock_code'], 'A000660')

        # 없는 종목 삭제 시도
        self.assertFalse(self.db_manager.delete_current_position('A999999')) # 이미 삭제되었거나 없는 종목
        self.assertEqual(len(self.db_manager.fetch_current_positions()), 1) # 여전히 1개여야 함


    # --- daily_signals 테스트 ---
    def test_05_save_fetch_update_clear_daily_signals(self):
        """일일 매매 신호 저장, 조회, 업데이트, 삭제 테스트"""
        # 신호 저장
        self.assertTrue(self.db_manager.save_daily_signal(self.test_signal_data_1))
        self.assertTrue(self.db_manager.save_daily_signal(self.test_signal_data_2))
        self.assertTrue(self.db_manager.save_daily_signal(self.test_signal_data_3))

        # 특정 날짜 신호 조회 (미실행 포함)
        fetched_signals_date10_all = self.db_manager.fetch_daily_signals(date(2025, 1, 10))
        self.assertIsInstance(fetched_signals_date10_all, list)
        self.assertEqual(len(fetched_signals_date10_all), 2)
        # 정렬 순서에 따라 A000660, A005930 순서일 것으로 예상
        self.assertEqual(fetched_signals_date10_all[0]['stock_code'], 'A000660') 
        self.assertEqual(fetched_signals_date10_all[1]['stock_code'], 'A005930')

        # 특정 날짜 실행된 신호 조회 (없음)
        fetched_signals_date10_executed = self.db_manager.fetch_daily_signals(date(2025, 1, 10), is_executed=True)
        self.assertEqual(len(fetched_signals_date10_executed), 0)

        # 신호 업데이트 (실행 상태 변경)
        # 신호 ID를 찾기 위해 먼저 조회
        signal_to_update = next((s for s in fetched_signals_date10_all if s['stock_code'] == 'A005930'), None)
        self.assertIsNotNone(signal_to_update)
        signal_id = signal_to_update['signal_id']
        test_order_id = "EXEC001"
        self.assertTrue(self.db_manager.update_daily_signal_status(signal_id, True, test_order_id))

        # 업데이트된 신호 확인
        fetched_signals_after_update = self.db_manager.fetch_daily_signals(date(2025, 1, 10), is_executed=True)
        self.assertEqual(len(fetched_signals_after_update), 1)
        self.assertEqual(fetched_signals_after_update[0]['stock_code'], 'A005930')
        self.assertTrue(fetched_signals_after_update[0]['is_executed'])
        self.assertEqual(fetched_signals_after_update[0]['executed_order_id'], test_order_id)
        
        # 특정 날짜의 모든 신호 삭제
        self.assertTrue(self.db_manager.clear_daily_signals(date(2025, 1, 10)))
        fetched_signals_after_clear = self.db_manager.fetch_daily_signals(date(2025, 1, 10))
        self.assertEqual(len(fetched_signals_after_clear), 0)
        
        # 다른 날짜의 신호는 남아있음을 확인
        fetched_signals_date11 = self.db_manager.fetch_daily_signals(date(2025, 1, 11))
        self.assertEqual(len(fetched_signals_date11), 1)
        self.assertEqual(fetched_signals_date11[0]['stock_code'], 'A005930')

        # 존재하지 않는 신호 ID 업데이트 시도
        self.assertFalse(self.db_manager.update_daily_signal_status(99999, True, "NONEXISTENT_ORDER"))


if __name__ == '__main__':
    unittest.main()