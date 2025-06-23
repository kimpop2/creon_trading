import unittest
from datetime import date, datetime
import pandas as pd
import sys
import os

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from manager.business_manager import BusinessManager
from manager.data_manager import DataManager

class TestTradeBusinessManager(unittest.TestCase):
    business_manager: BusinessManager
    db_manager: 'DBManager'

    @classmethod
    def setUpClass(cls):
        """테스트 시작 전, DB 연결 및 테이블 생성"""
        # 실제 CreonAPIClient 인스턴스를 사용하여 테스트
        try:
            # BusinessManager가 DataManager에 의존하므로 함께 초기화
            cls.data_manager = DataManager()
            cls.business_manager = BusinessManager(cls.data_manager)
            cls.db_manager = cls.business_manager.db_manager

            # 테스트 전 기존 테이블 삭제 및 재생성
            cls.db_manager.drop_trade_tables()
            cls.db_manager.create_trade_tables()
            print("TestTradeBusinessManager: DB 테이블 설정 완료")
        except Exception as e:
            print(f"TestTradeBusinessManager.setUpClass에서 에러 발생: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        """테스트 종료 후, 테이블 삭제 및 DB 연결 종료"""
        if hasattr(cls, 'db_manager') and cls.db_manager:
            cls.db_manager.drop_trade_tables()
            cls.db_manager.close()
            print("TestTradeBusinessManager: DB 테이블 정리 및 연결 종료 완료")

    def test_save_and_load_daily_signals(self):
        """daily_signals 저장 및 로드 기능 테스트"""
        signal_date = date.today()
        signals_to_save = {
            'A005930': {'signal_type': 'BUY', 'signal_price': 70000, 'strategy_name': 'TestStrat', 'volume_ratio': 1.5},
            'A000660': {'signal_type': 'SELL', 'signal_price': 100000, 'strategy_name': 'TestStrat', 'volume_ratio': 0.8}
        }

        # BusinessManager에 해당 메서드가 있다는 가정 하에 테스트
        # AttributeError 발생 시, BusinessManager에 메서드 구현 필요
        try:
            self.business_manager.save_daily_signals(signals_to_save, signal_date)
            loaded_signals = self.business_manager.load_daily_signals_for_today(signal_date)
        except AttributeError as e:
            self.fail(f"BusinessManager에 필요한 메서드가 없습니다: {e}")

        self.assertIn('A005930', loaded_signals)
        self.assertEqual(loaded_signals['A005930']['signal_type'], 'BUY')
        self.assertEqual(len(loaded_signals), 2)

    def test_save_and_load_current_positions(self):
        """current_positions 저장 및 로드 기능 테스트"""
        positions_to_save = {
            'A005930': {'size': 10, 'avg_price': 70000, 'entry_date': date.today(), 'highest_price': 71000}
        }

        try:
            self.business_manager.save_current_positions(positions_to_save)
            loaded_positions = self.business_manager.load_current_positions()
        except AttributeError as e:
            self.fail(f"BusinessManager에 필요한 메서드가 없습니다: {e}")

        self.assertIn('A005930', loaded_positions)
        self.assertEqual(loaded_positions['A005930']['size'], 10)
        self.assertEqual(loaded_positions['A005930']['avg_price'], 70000)

    def test_save_trade_log(self):
        """trade_log 저장 기능 테스트"""
        log_entry = {
            'order_time': datetime.now(),
            'stock_code': 'A000660',
            'order_type': 'BUY',
            'order_price': 100000,
            'order_quantity': 5,
            'executed_price': 100000,
            'executed_quantity': 5,
            'commission': 160,
            'tax': 0,
            'net_amount': 500160,
            'order_status': 'FILLED',
            'creon_order_id': 'ORD12345',
            'message': '테스트 주문'
        }
        
        try:
            self.business_manager.save_trade_log(log_entry)
        except AttributeError as e:
            self.fail(f"BusinessManager에 필요한 메서드가 없습니다: {e}")

        # DB에서 직접 확인하여 검증
        cursor = self.db_manager.execute_sql("SELECT * FROM trade_log WHERE stock_code = 'A000660'")
        self.assertIsNotNone(cursor, "trade_log 조회 실패")
        logs = cursor.fetchall()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]['order_type'], 'BUY')

    def test_save_daily_portfolio_snapshot(self):
        """daily_portfolio_snapshot 저장 기능 테스트"""
        snapshot_date = date.today()
        
        try:
            self.business_manager.save_daily_portfolio_snapshot(
                snapshot_date=snapshot_date,
                portfolio_value=10000000,
                cash=5000000,
                positions={'A005930': {'size': 50, 'avg_price': 70000}}
            )
        except AttributeError as e:
            self.fail(f"BusinessManager에 필요한 메서드가 없습니다: {e}")

        # DB에서 직접 확인하여 검증
        cursor = self.db_manager.execute_sql(f"SELECT * FROM daily_portfolio_snapshot WHERE snapshot_date = '{snapshot_date.isoformat()}'")
        self.assertIsNotNone(cursor, "daily_portfolio_snapshot 조회 실패")
        snapshots = cursor.fetchall()
        self.assertEqual(len(snapshots), 1)
        # DB 스키마에 따라 total_asset_value를 확인
        self.assertEqual(snapshots[0]['total_asset_value'], 10000000)

if __name__ == '__main__':
    unittest.main() 