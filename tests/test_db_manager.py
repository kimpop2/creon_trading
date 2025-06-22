"""
DBManager 클래스 단위 테스트
"""

import unittest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import sys
import os

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from manager.db_manager import DBManager

class TestDBManager(unittest.TestCase):
    def setUp(self):
        """테스트 설정"""
        self.db_manager = DBManager()
        
        # 테스트용 데이터 준비
        self.test_stock_info_data = [
            {
                'stock_code': 'A005930', 'stock_name': '삼성전자', 'market_type': 'KOSPI', 'sector': '반도체 및 반도체 장비',
                'per': 15.20, 'pbr': 1.60, 'eps': 5000.00, 'roe': 10.50, 'debt_ratio': 25.30,
                'sales': 280000000, 'operating_profit': 4000000, 'net_profit': 3000000,
                'recent_financial_date': date(2024, 3, 31)
            },
            {
                'stock_code': 'A000660', 'stock_name': 'SK하이닉스', 'market_type': 'KOSPI', 'sector': '반도체 및 반도체 장비',
                'per': 20.10, 'pbr': 2.50, 'eps': 3500.00, 'roe': 12.00, 'debt_ratio': 35.00,
                'sales': 35000000, 'operating_profit': 500000, 'net_profit': 400000,
                'recent_financial_date': date(2024, 3, 31)
            },
            {
                'stock_code': 'A035420', 'stock_name': 'NAVER', 'market_type': 'KOSPI', 'sector': '소프트웨어',
                'per': 30.50, 'pbr': 3.00, 'eps': 2000.00, 'roe': 8.00, 'debt_ratio': 15.00,
                'sales': 900000, 'operating_profit': 150000, 'net_profit': 100000,
                'recent_financial_date': date(2024, 3, 31)
            }
        ]
        
        self.test_daily_price_data = [
            {
                'stock_code': 'A005930', 'date': date(2024, 6, 10), 'open': 78000.0, 'high': 78500.0,
                'low': 77500.0, 'close': 78200.0, 'volume': 10000000, 'trading_value': 782000000000, 'change_rate': 0.25
            },
            {
                'stock_code': 'A005930', 'date': date(2024, 6, 11), 'open': 78300.0, 'high': 79000.0,
                'low': 78000.0, 'close': 78800.0, 'volume': 12000000, 'trading_value': 945600000000, 'change_rate': 0.77
            },
            {
                'stock_code': 'A000660', 'date': date(2024, 6, 10), 'open': 180000.0, 'high': 181000.0,
                'low': 179000.0, 'close': 180500.0, 'volume': 5000000, 'trading_value': 902500000000, 'change_rate': 0.50
            }
        ]
        
        self.test_minute_price_data = [
            {
                'stock_code': 'A005930', 'datetime': datetime(2024, 6, 10, 9, 0, 0), 'open': 78200.0, 
                'high': 78250.0, 'low': 78150.0, 'close': 78220.0, 'volume': 100000,
                'trading_value': 7822000000, 'change_rate': 0.01
            },
            {
                'stock_code': 'A005930', 'datetime': datetime(2024, 6, 10, 9, 1, 0), 'open': 78220.0, 
                'high': 78300.0, 'low': 78200.0, 'close': 78280.0, 'volume': 80000,
                'trading_value': 6262400000, 'change_rate': 0.07
            },
            {
                'stock_code': 'A000660', 'datetime': datetime(2024, 6, 10, 9, 0, 0), 'open': 180500.0, 
                'high': 180600.0, 'low': 180400.0, 'close': 180550.0, 'volume': 30000,
                'trading_value': 5416500000, 'change_rate': 0.03
            }
        ]
        
        self.test_calendar_data = [
            {'date': date(2025, 6, 1), 'is_holiday': False, 'description': '영업일'},
            {'date': date(2025, 6, 2), 'is_holiday': False, 'description': '영업일'},
            {'date': date(2025, 6, 3), 'is_holiday': False, 'description': '영업일'},
            {'date': date(2025, 6, 4), 'is_holiday': False, 'description': '영업일'},
            {'date': date(2025, 6, 5), 'is_holiday': True, 'description': '가상공휴일'},
            {'date': date(2025, 6, 6), 'is_holiday': False, 'description': '영업일'},
            {'date': date(2025, 6, 7), 'is_holiday': False, 'description': '영업일'},
            {'date': date(2025, 6, 8), 'is_holiday': False, 'description': '영업일'},
            {'date': date(2025, 6, 9), 'is_holiday': False, 'description': '영업일'},
            {'date': date(2025, 6, 10), 'is_holiday': True, 'description': '가상공휴일'},
            {'date': date(2025, 6, 11), 'is_holiday': False, 'description': '영업일'},
            {'date': date(2025, 6, 12), 'is_holiday': False, 'description': '영업일'},
            {'date': date(2025, 6, 13), 'is_holiday': True, 'description': '주말(토)'}
        ]
        
        self.test_run_info = {
            'start_date': date(2023, 1, 1),
            'end_date': date(2023, 12, 31),
            'initial_capital': 10000000.00,
            'final_capital': 12000000.00,
            'total_profit_loss': 2000000.00,
            'cumulative_return': 0.20,
            'max_drawdown': 0.05,
            'strategy_daily': 'DualMomentumDaily',
            'strategy_minute': 'RSIMinute',
            'params_json_daily': {'lookback_period': 12, 'num_top_stocks': 5},
            'params_json_minute': {'rsi_period': 14, 'buy_signal': 30, 'sell_signal': 70}
        }
        
        self.test_trade_data = [
            {
                'run_id': 1, 'stock_code': 'A005930', 'trade_type': 'BUY', 'trade_price': 70000.0,
                'trade_quantity': 10, 'trade_amount': 700000.0, 'trade_datetime': datetime(2023, 1, 5, 9, 30, 0),
                'commission': 500.0, 'tax': 0.0, 'realized_profit_loss': 0.0, 'entry_trade_id': None
            },
            {
                'run_id': 1, 'stock_code': 'A000660', 'trade_type': 'BUY', 'trade_price': 100000.0,
                'trade_quantity': 5, 'trade_amount': 500000.0, 'trade_datetime': datetime(2023, 1, 5, 10, 0, 0),
                'commission': 300.0, 'tax': 0.0, 'realized_profit_loss': 0.0, 'entry_trade_id': None
            }
        ]
        
        self.test_performance_data = [
            {
                'run_id': 1, 'date': date(2023, 1, 1), 'end_capital': 10000000.0,
                'daily_return': 0.0, 'daily_profit_loss': 0.0, 'cumulative_return': 0.0, 'drawdown': 0.0
            },
            {
                'run_id': 1, 'date': date(2023, 1, 2), 'end_capital': 10050000.0,
                'daily_return': 0.005, 'daily_profit_loss': 50000.0, 'cumulative_return': 0.005, 'drawdown': 0.0
            },
            {
                'run_id': 1, 'date': date(2023, 1, 3), 'end_capital': 9900000.0,
                'daily_return': -0.0149, 'daily_profit_loss': -150000.0, 'cumulative_return': -0.01, 'drawdown': 0.015
            }
        ]

    def tearDown(self):
        """테스트 정리"""
        if hasattr(self, 'db_manager'):
            self.db_manager.close()

    def test_01_create_and_drop_stock_tables(self):
        """종목 관련 테이블 생성 및 삭제 테스트"""
        # 테이블 삭제
        result = self.db_manager.drop_stock_tables()
        self.assertTrue(result)
        
        # 테이블 생성
        result = self.db_manager.create_stock_tables()
        self.assertTrue(result)

    def test_02_save_and_fetch_stock_info(self):
        """종목 정보 저장 및 조회 테스트"""
        # 종목 정보 저장
        result = self.db_manager.save_stock_info(self.test_stock_info_data)
        self.assertTrue(result)
        
        # 전체 종목 정보 조회
        fetched_data = self.db_manager.fetch_stock_info()
        self.assertIsInstance(fetched_data, pd.DataFrame)
        self.assertGreater(len(fetched_data), 0)
        
        # 특정 종목 정보 조회
        fetched_specific = self.db_manager.fetch_stock_info(stock_codes=['A005930'])
        self.assertIsInstance(fetched_specific, pd.DataFrame)
        self.assertEqual(len(fetched_specific), 1)
        self.assertEqual(fetched_specific.iloc[0]['stock_code'], 'A005930')

    def test_03_get_all_stock_codes(self):
        """모든 종목 코드 조회 테스트"""
        codes = self.db_manager.get_all_stock_codes()
        self.assertIsInstance(codes, list)
        self.assertGreater(len(codes), 0)
        self.assertIn('A005930', codes)

    def test_04_fetch_stock_codes_by_criteria(self):
        """조건부 종목 코드 조회 테스트"""
        # EPS 3000 이상 종목 조회
        filtered_codes = self.db_manager.fetch_stock_codes_by_criteria(eps_min=3000)
        self.assertIsInstance(filtered_codes, list)
        
        # PBR 2.0 이하, ROE 10.0 이상 종목 조회
        filtered_codes_complex = self.db_manager.fetch_stock_codes_by_criteria(pbr_max=2.0, roe_min=10.0)
        self.assertIsInstance(filtered_codes_complex, list)

    def test_05_save_and_fetch_daily_price(self):
        """일봉 데이터 저장 및 조회 테스트"""
        # 일봉 데이터 저장
        result = self.db_manager.save_daily_price(self.test_daily_price_data)
        self.assertTrue(result)
        
        # 일봉 데이터 조회 (전체 기간)
        fetched_data = self.db_manager.fetch_daily_price(stock_code='A005930')
        self.assertIsInstance(fetched_data, pd.DataFrame)
        self.assertGreater(len(fetched_data), 0)
        
        # 일봉 데이터 조회 (특정 기간)
        fetched_period = self.db_manager.fetch_daily_price(
            stock_code='A005930', 
            start_date=date(2024, 6, 10), 
            end_date=date(2024, 6, 10)
        )
        self.assertIsInstance(fetched_period, pd.DataFrame)

    def test_06_get_latest_daily_price_date(self):
        """최신 일봉 날짜 조회 테스트"""
        latest_date = self.db_manager.get_latest_daily_price_date('A005930')
        self.assertIsNotNone(latest_date)

    def test_07_save_and_fetch_minute_price(self):
        """분봉 데이터 저장 및 조회 테스트"""
        # 분봉 데이터 저장
        result = self.db_manager.save_minute_price(self.test_minute_price_data)
        self.assertTrue(result)
        
        # 분봉 데이터 조회 (전체 기간)
        fetched_data = self.db_manager.fetch_minute_price(stock_code='A005930')
        self.assertIsInstance(fetched_data, pd.DataFrame)
        self.assertGreater(len(fetched_data), 0)
        
        # 분봉 데이터 조회 (특정 기간)
        fetched_period = self.db_manager.fetch_minute_price(
            stock_code='A005930', 
            start_date=date(2024, 6, 10), 
            end_date=date(2024, 6, 10)
        )
        self.assertIsInstance(fetched_period, pd.DataFrame)

    def test_08_save_and_fetch_market_calendar(self):
        """시장 캘린더 저장 및 조회 테스트"""
        # 시장 캘린더 저장
        result = self.db_manager.save_market_calendar(pd.DataFrame(self.test_calendar_data))
        self.assertTrue(result)
        
        # 시장 캘린더 조회
        fetched_data = self.db_manager.fetch_market_calendar(date(2025, 6, 1), date(2025, 6, 13))
        self.assertIsInstance(fetched_data, pd.DataFrame)
        self.assertGreater(len(fetched_data), 0)

    def test_09_get_all_trading_days(self):
        """영업일 조회 테스트"""
        trading_days = self.db_manager.get_all_trading_days(date(2025, 6, 1), date(2025, 6, 13))
        self.assertIsInstance(trading_days, list)
        self.assertGreater(len(trading_days), 0)

    def test_10_create_and_drop_backtest_tables(self):
        """백테스트 관련 테이블 생성 및 삭제 테스트"""
        # 테이블 삭제
        result = self.db_manager.drop_backtest_tables()
        self.assertTrue(result)
        
        # 테이블 생성
        result = self.db_manager.create_backtest_tables()
        self.assertTrue(result)

    def test_11_save_and_fetch_backtest_run(self):
        """백테스트 실행 정보 저장 및 조회 테스트"""
        # 백테스트 실행 정보 저장
        run_id = self.db_manager.save_backtest_run(self.test_run_info)
        self.assertIsNotNone(run_id)
        self.assertIsInstance(run_id, int)
        
        # 백테스트 실행 정보 조회 (특정 run_id)
        fetched_data = self.db_manager.fetch_backtest_run(run_id=run_id)
        self.assertIsInstance(fetched_data, pd.DataFrame)
        self.assertGreater(len(fetched_data), 0)
        
        # 백테스트 실행 정보 조회 (기간)
        fetched_period = self.db_manager.fetch_backtest_run(
            start_date=date(2023, 1, 1), 
            end_date=date(2023, 12, 31)
        )
        self.assertIsInstance(fetched_period, pd.DataFrame)

    def test_12_save_and_fetch_backtest_trade(self):
        """백테스트 거래 내역 저장 및 조회 테스트"""
        # 거래 내역 저장
        result = self.db_manager.save_backtest_trade(self.test_trade_data)
        self.assertTrue(result)
        
        # 거래 내역 조회 (전체)
        fetched_data = self.db_manager.fetch_backtest_trade(run_id=1)
        self.assertIsInstance(fetched_data, pd.DataFrame)
        self.assertGreater(len(fetched_data), 0)
        
        # 거래 내역 조회 (특정 종목)
        fetched_stock = self.db_manager.fetch_backtest_trade(run_id=1, stock_code='A005930')
        self.assertIsInstance(fetched_stock, pd.DataFrame)

    def test_13_save_and_fetch_backtest_performance(self):
        """백테스트 성능 지표 저장 및 조회 테스트"""
        # 성능 지표 저장
        result = self.db_manager.save_backtest_performance(self.test_performance_data)
        self.assertTrue(result)
        
        # 성능 지표 조회 (전체)
        fetched_data = self.db_manager.fetch_backtest_performance(run_id=1)
        self.assertIsInstance(fetched_data, pd.DataFrame)
        self.assertGreater(len(fetched_data), 0)
        
        # 성능 지표 조회 (특정 기간)
        fetched_period = self.db_manager.fetch_backtest_performance(
            run_id=1, 
            start_date=date(2023, 1, 1), 
            end_date=date(2023, 1, 2)
        )
        self.assertIsInstance(fetched_period, pd.DataFrame)

    def test_14_check_table_exist(self):
        """테이블 존재 여부 확인 테스트"""
        # 존재하는 테이블 확인
        result = self.db_manager.check_table_exist('stock_info')
        self.assertIsInstance(result, bool)
        
        # 존재하지 않는 테이블 확인
        result = self.db_manager.check_table_exist('non_existent_table')
        self.assertIsInstance(result, bool)

    def test_15_insert_df_to_db(self):
        """DataFrame을 DB에 삽입하는 테스트"""
        # 테스트용 DataFrame 생성
        test_df = pd.DataFrame({
            'test_col1': [1, 2, 3],
            'test_col2': ['a', 'b', 'c']
        })
        
        # DataFrame 삽입
        result = self.db_manager.insert_df_to_db('test_table', test_df, option='replace')
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main() 



# --- 테스트 코드 ---
    
    # db_manager = DBManager()

    # # --- 1. 종목 및 주가 관련 테이블 테스트 ---
    # logger.info("--- 종목 및 주가 관련 테이블 테스트 시작 ---")
    
    # # 1.1 테이블 삭제 및 재생성 (stock_info, daily_price, minute_price, market_calendar)
    # logger.info("기존 stock_tables 삭제 중...")
    # if not db_manager.drop_stock_tables(): # <-- 반환값 확인
    #     logger.error("stock_tables 삭제 실패! 존재하지 않을 수 있음")
        
    # logger.info("stock_tables 생성 중...")
    # if not db_manager.create_stock_tables(): # <-- 반환값 확인
    #     logger.error("stock_tables 생성 실패! 테스트를 중단합니다. SQL 파일 경로 또는 SQL 문법을 확인하세요.")
    #     sys.exit(1) # 테스트 중단
    # logger.info("stock_tables 생성 완료.")

    # # 1.2 save_stock_info 테스트
    # logger.info("save_stock_info 테스트 시작...")
    # test_stock_info_data = [
    #     {
    #         'stock_code': 'A005930', 'stock_name': '삼성전자', 'market_type': 'KOSPI', 'sector': '반도체 및 반도체 장비',
    #         'per': 15.20, 'pbr': 1.60, 'eps': 5000.00, 'roe': 10.50, 'debt_ratio': 25.30,
    #         'sales': 280000000, 'operating_profit': 4000000, 'net_profit': 3000000,
    #         'recent_financial_date': date(2024, 3, 31)
    #     },
    #     {
    #         'stock_code': 'A000660', 'stock_name': 'SK하이닉스', 'market_type': 'KOSPI', 'sector': '반도체 및 반도체 장비',
    #         'per': 20.10, 'pbr': 2.50, 'eps': 3500.00, 'roe': 12.00, 'debt_ratio': 35.00,
    #         'sales': 35000000, 'operating_profit': 500000, 'net_profit': 400000,
    #         'recent_financial_date': date(2024, 3, 31)
    #     },
    #     {
    #         'stock_code': 'A035420', 'stock_name': 'NAVER', 'market_type': 'KOSPI', 'sector': '소프트웨어',
    #         'per': 30.50, 'pbr': 3.00, 'eps': 2000.00, 'roe': 8.00, 'debt_ratio': 15.00,
    #         'sales': 900000, 'operating_profit': 150000, 'net_profit': 100000,
    #         'recent_financial_date': date(2024, 3, 31)
    #     }
    # ]
    # db_manager.save_stock_info(test_stock_info_data)
    # logger.info("save_stock_info 테스트 완료.")

    # # 1.3 fetch_stock_info 테스트
    # logger.info("fetch_stock_info 테스트 시작 (전체 조회)...")
    # fetched_stock_info = db_manager.fetch_stock_info()
    # logger.info(f"조회된 stock_info 데이터:\n{fetched_stock_info}")

    # logger.info("fetch_stock_info 테스트 시작 (특정 종목 조회)...")
    # fetched_specific_stock_info = db_manager.fetch_stock_info(stock_codes=['A005930'])
    # logger.info(f"조회된 특정 종목 stock_info 데이터 (A005930):\n{fetched_specific_stock_info}")
    # logger.info("fetch_stock_info 테스트 완료.")

    # # 1.4 get_all_stock_codes 테스트
    # logger.info("get_all_stock_codes 테스트 시작...")
    # all_codes = db_manager.get_all_stock_codes()
    # logger.info(f"모든 종목 코드:\n{all_codes}")
    # logger.info("get_all_stock_codes 테스트 완료.")

    # # 1.5 fetch_stock_codes_by_criteria 테스트
    # logger.info("fetch_stock_codes_by_criteria 테스트 시작 (EPS 3000 이상)...")
    # filtered_codes = db_manager.fetch_stock_codes_by_criteria(eps_min=3000)
    # logger.info(f"EPS 3000 이상 종목 코드:\n{filtered_codes}")

    # logger.info("fetch_stock_codes_by_criteria 테스트 시작 (PBR 2.0 이하, ROE 10.0 이상)...")
    # filtered_codes_complex = db_manager.fetch_stock_codes_by_criteria(pbr_max=2.0, roe_min=10.0)
    # logger.info(f"PBR 2.0 이하, ROE 10.0 이상 종목 코드:\n{filtered_codes_complex}")
    # logger.info("fetch_stock_codes_by_criteria 테스트 완료.")

    # # 1.6 save_daily_price 테스트
    # logger.info("save_daily_price 테스트 시작...")
    # test_daily_price_data = [
    #     {
    #         'stock_code': 'A005930', 'date': date(2024, 6, 10), 'open': 78000.0, 'high': 78500.0,
    #         'low': 77500.0, 'close': 78200.0, 'volume': 10000000, 'trading_value': 782000000000, 'change_rate': 0.25
    #     },
    #     {
    #         'stock_code': 'A005930', 'date': date(2024, 6, 11), 'open': 78300.0, 'high': 79000.0,
    #         'low': 78000.0, 'close': 78800.0, 'volume': 12000000, 'trading_value': 945600000000, 'change_rate': 0.77
    #     },
    #     {
    #         'stock_code': 'A000660', 'date': date(2024, 6, 10), 'open': 180000.0, 'high': 181000.0,
    #         'low': 179000.0, 'close': 180500.0, 'volume': 5000000, 'trading_value': 902500000000, 'change_rate': 0.50
    #     }
    # ]
    # db_manager.save_daily_price(test_daily_price_data)
    # logger.info("save_daily_price 테스트 완료.")

    # # # 1.7 fetch_daily_price 테스트
    # logger.info("fetch_daily_price 테스트 시작 (A005930, 전체 기간)...")
    # fetched_price_data = db_manager.fetch_daily_price(stock_code='A005930')
    # logger.info(f"조회된 daily_price 데이터 (A005930):\n{fetched_price_data}")

    # logger.info("fetch_daily_price 테스트 시작 (A005930, 특정 기간)...")
    # fetched_price_data_period = db_manager.fetch_daily_price(
    #     stock_code='A005930', 
    #     start_date=date(2024, 6, 10), 
    #     end_date=date(2024, 6, 10)
    # )
    # logger.info(f"조회된 daily_price 데이터 (A005930, 2024-06-10~2024-06-10):\n{fetched_price_data_period}")
    # logger.info("fetch_daily_price 테스트 완료.")

    # # # 1.8 get_latest_daily_price_date 테스트
    # logger.info("get_latest_daily_price_date 테스트 시작 (A005930)...")
    # latest_date_5930 = db_manager.get_latest_daily_price_date('A005930')
    # logger.info(f"A005930 최신 일봉 날짜: {latest_date_5930}")

    # logger.info("get_latest_daily_price_date 테스트 시작 (A000660)...")
    # latest_date_660 = db_manager.get_latest_daily_price_date('A000660')
    # logger.info(f"A000660 최신 일봉 날짜: {latest_date_660}")
    # logger.info("get_latest_daily_price_date 테스트 완료.")

    # # # 1.9 save_minute_price 테스트 (새로 추가)
    # logger.info("save_minute_price 테스트 시작...")
    # test_minute_price_data = [
    #     {
    #         'stock_code': 'A005930', 'datetime': datetime(2024, 6, 10, 9, 0, 0), 'open': 78200.0, 
    #         'high': 78250.0, 'low': 78150.0, 'close': 78220.0, 'volume': 100000,
    #         'trading_value': 7822000000, 'change_rate': 0.01
    #     },
    #     {
    #         'stock_code': 'A005930', 'datetime': datetime(2024, 6, 10, 9, 1, 0), 'open': 78220.0, 
    #         'high': 78300.0, 'low': 78200.0, 'close': 78280.0, 'volume': 80000,
    #         'trading_value': 6262400000, 'change_rate': 0.07
    #     },
    #     {
    #         'stock_code': 'A000660', 'datetime': datetime(2024, 6, 10, 9, 0, 0), 'open': 180500.0, 
    #         'high': 180600.0, 'low': 180400.0, 'close': 180550.0, 'volume': 30000,
    #         'trading_value': 5416500000, 'change_rate': 0.03
    #     }
    # ]
    # db_manager.save_minute_price(test_minute_price_data)
    # logger.info("save_minute_price 테스트 완료.")

    # # # 1.10 fetch_minute_price 테스트 (새로 추가)
    # logger.info("fetch_minute_price 테스트 시작 (A005930, 전체 기간)...")
    # fetched_minute_data = db_manager.fetch_minute_price(stock_code='A005930')
    # logger.info(f"조회된 minute_price 데이터 (A005930):\n{fetched_minute_data}")

    # logger.info("fetch_minute_price 테스트 시작 (A005930, 특정 기간)...")
    # fetched_minute_data_period = db_manager.fetch_minute_price(
    #     stock_code='A005930', 
    #     start_date=date(2024, 6, 10), 
    #     end_date=date(2024, 6, 10)
    # )
    # logger.info(f"조회된 minute_price 데이터 (A005930, 2024-06-10 전체):\n{fetched_minute_data_period}")

    # logger.info("fetch_minute_price 테스트 시작 (A005930, 특정 시각 범위)...")
    # fetched_minute_data_time_range = db_manager.fetch_minute_price(
    #     stock_code='A005930', 
    #     start_date=datetime(2024, 6, 10, 9, 0, 0), 
    #     end_date=datetime(2024, 6, 10, 9, 0, 0)
    # )
    # logger.info(f"조회된 minute_price 데이터 (A005930, 2024-06-10 09:00:00):\n{fetched_minute_data_time_range}")
    # logger.info("fetch_minute_price 테스트 완료.")

    # # # 1.11 save_market_calendar 테스트 (새로 추가)
    # logger.info("save_market_calendar 테스트 시작...")
    # # 2024년 6월 10일(월)은 영업일, 6월 11일(화)도 영업일, 6월 12일(수)은 휴일(테스트용), 6월 13일(목)은 영업일
    # test_calendar_data = [
    #     {'date': date(2025, 6, 1), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 2), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 3), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 4), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 5), 'is_holiday': True, 'description': '가상공휴일'},
    #     {'date': date(2025, 6, 6), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 7), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 8), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 9), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 10), 'is_holiday': True, 'description': '가상공휴일'},
    #     {'date': date(2025, 6, 11), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 12), 'is_holiday': False, 'description': '영업일'},
    #     {'date': date(2025, 6, 13), 'is_holiday': True, 'description': '주말(토)'}
    # ]
    # db_manager.save_market_calendar(pd.DataFrame(test_calendar_data))
    # logger.info("save_market_calendar 테스트 완료.")

    # # # 1.12 fetch_market_calendar 테스트 (새로 추가)
    # logger.info("fetch_market_calendar 테스트 시작 (2024-06-10 ~ 2024-06-15)...")
    # fetched_calendar_data = db_manager.fetch_market_calendar(date(2024, 6, 10), date(2024, 6, 15))
    # logger.info(f"조회된 market_calendar 데이터:\n{fetched_calendar_data}")
    # logger.info("fetch_market_calendar 테스트 완료.")

    # # 1.13 get_all_trading_days 테스트 (새로 추가)
    # logger.info("get_all_trading_days 테스트 시작 (2024-06-10 ~ 2024-06-15)...")
    # trading_days = db_manager.get_all_trading_days(date(2024, 6, 10), date(2025, 6, 15))
    # logger.info(f"조회된 영업일 데이터 (list):\n{trading_days}")
    # logger.info("get_all_trading_days 테스트 완료.")


    # logger.info("--- 종목 및 주가 관련 테이블 테스트 완료 ---")

    # # --- 2. 백테스팅 관련 테이블 테스트 ---
    # logger.info("--- 백테스팅 관련 테이블 테스트 시작 ---")

    # # 2.1 테이블 삭제 및 재생성 (backtest_run, backtest_trade, backtest_performance)
    # logger.info("기존 backtest_tables 삭제 중...")
    # if not db_manager.drop_backtest_tables(): # <-- 반환값 확인
    #     logger.error("backtest_tables 삭제 실패! 테스트를 중단합니다. SQL 파일 경로 또는 권한을 확인하세요.")
    #     sys.exit(1) # 테스트 중단
    # logger.info("backtest_tables 삭제 완료.")

    # logger.info("backtest_tables 생성 중...")
    # if not db_manager.create_backtest_tables(): # <-- 반환값 확인
    #     logger.error("backtest_tables 생성 실패! 테스트를 중단합니다. SQL 파일 경로 또는 SQL 문법을 확인하세요.")
    #     sys.exit(1) # 테스트 중단
    # logger.info("backtest_tables 생성 완료.")

    # # 2.2 save_backtest_run 테스트
    # logger.info("save_backtest_run 테스트 시작...")
    # test_run_info = {
    #     'start_date': date(2023, 1, 1),
    #     'end_date': date(2023, 12, 31),
    #     'initial_capital': 10000000.00,
    #     'final_capital': 12000000.00,
    #     'total_profit_loss': 2000000.00,
    #     'cumulative_return': 0.20,
    #     'max_drawdown': 0.05,
    #     'strategy_daily': 'DualMomentumDaily',
    #     'strategy_minute': 'RSIMinute',
    #     'params_json_daily': {'lookback_period': 12, 'num_top_stocks': 5},
    #     'params_json_minute': {'rsi_period': 14, 'buy_signal': 30, 'sell_signal': 70}
    # }
    # # 새로운 run_id 생성 및 저장
    # run_id = db_manager.save_backtest_run(test_run_info)
    # if run_id:
    #     logger.info(f"새로운 backtest_run 저장 완료. run_id: {run_id}")
    # else:
    #     logger.error("backtest_run 저장 실패!")
    #     run_id = 999999 # 테스트를 위해 임의의 ID 할당 (실패 시에도 계속 진행 위함)

    # # 기존 run_id로 업데이트 테스트
    # logger.info(f"기존 backtest_run 업데이트 테스트 (run_id: {run_id})...")
    # update_run_info = test_run_info.copy()
    # update_run_info['run_id'] = run_id
    # update_run_info['final_capital'] = 12500000.00
    # update_run_info['total_profit_loss'] = 2500000.00
    # update_run_info['cumulative_return'] = 0.25
    # db_manager.save_backtest_run(update_run_info)
    # logger.info("save_backtest_run 테스트 완료.")

    # # 2.3 fetch_backtest_run 테스트
    # logger.info(f"fetch_backtest_run 테스트 시작 (run_id: {run_id})...")
    # fetched_run_data = db_manager.fetch_backtest_run(run_id=run_id)
    # logger.info(f"조회된 backtest_run 데이터:\n{fetched_run_data}")

    # logger.info("fetch_backtest_run 테스트 시작 (기간 조회)...")
    # fetched_run_data_period = db_manager.fetch_backtest_run(
    #     start_date=date(2023, 1, 1), 
    #     end_date=date(2023, 12, 31)
    # )
    # logger.info(f"조회된 backtest_run 데이터 (2023년):\n{fetched_run_data_period}")
    # logger.info("fetch_backtest_run 테스트 완료.")

    # # 2.4 save_backtest_trade 테스트
    # logger.info("save_backtest_trade 테스트 시작...")
    # test_trade_data = [
    #     {
    #         'run_id': run_id, 'stock_code': 'A005930', 'trade_type': 'BUY', 'trade_price': 70000.0,
    #         'trade_quantity': 10, 'trade_amount': 700000.0, 'trade_datetime': datetime(2023, 1, 5, 9, 30, 0),
    #         'commission': 500.0, 'tax': 0.0, 'realized_profit_loss': 0.0, 'entry_trade_id': None
    #     },
    #     {
    #         'run_id': run_id, 'stock_code': 'A000660', 'trade_type': 'BUY', 'trade_price': 100000.0,
    #         'trade_quantity': 5, 'trade_amount': 500000.0, 'trade_datetime': datetime(2023, 1, 5, 10, 0, 0),
    #         'commission': 300.0, 'tax': 0.0, 'realized_profit_loss': 0.0, 'entry_trade_id': None
    #     }
    # ]
    # db_manager.save_backtest_trade(test_trade_data)

    # # 매도 거래 추가 (entry_trade_id를 위해 첫 번째 매수 거래의 trade_id가 필요)
    # # 실제 시스템에서는 매수 거래 저장 후 trade_id를 받아와 사용해야 하지만, 여기서는 임의로 가정
    # # 또는 fetch_backtest_trade로 매수 거래를 조회하여 trade_id를 얻어야 합니다.
    # fetched_trades_for_entry_id = db_manager.fetch_backtest_trade(run_id=run_id, stock_code='A005930', start_datetime=datetime(2023,1,5,9,29,0), end_datetime=datetime(2023,1,5,9,31,0))
    # entry_trade_id = fetched_trades_for_entry_id['trade_id'].iloc[0] if not fetched_trades_for_entry_id.empty else None

    # test_trade_sell_data = [
    #     {
    #         'run_id': run_id, 'stock_code': 'A005930', 'trade_type': 'SELL', 'trade_price': 75000.0,
    #         'trade_quantity': 10, 'trade_amount': 750000.0, 'trade_datetime': datetime(2023, 1, 10, 15, 0, 0),
    #         'commission': 500.0, 'tax': 75.0, 'realized_profit_loss': 49925.0, # (75000-70000)*10 - 500 - 75
    #         'entry_trade_id': entry_trade_id 
    #     }
    # ]
    # db_manager.save_backtest_trade(test_trade_sell_data)
    # logger.info("save_backtest_trade 테스트 완료.")

    # # 2.5 fetch_backtest_trade 테스트
    # logger.info(f"fetch_backtest_trade 테스트 시작 (run_id: {run_id})...")
    # fetched_trade_data = db_manager.fetch_backtest_trade(run_id=run_id)
    # logger.info(f"조회된 backtest_trade 데이터:\n{fetched_trade_data}")

    # logger.info(f"fetch_backtest_trade 테스트 시작 (run_id: {run_id}, 특정 종목 A005930)...")
    # fetched_trade_data_stock = db_manager.fetch_backtest_trade(run_id=run_id, stock_code='A005930')
    # logger.info(f"조회된 backtest_trade 데이터 (A005930):\n{fetched_trade_data_stock}")
    # logger.info("fetch_backtest_trade 테스트 완료.")

    # # 2.6 save_backtest_performance 테스트
    # logger.info("save_backtest_performance 테스트 시작...")
    # test_performance_data = [
    #     {
    #         'run_id': run_id, 'date': date(2023, 1, 1), 'end_capital': 10000000.0,
    #         'daily_return': 0.0, 'daily_profit_loss': 0.0, 'cumulative_return': 0.0, 'drawdown': 0.0
    #     },
    #     {
    #         'run_id': run_id, 'date': date(2023, 1, 2), 'end_capital': 10050000.0,
    #         'daily_return': 0.005, 'daily_profit_loss': 50000.0, 'cumulative_return': 0.005, 'drawdown': 0.0
    #     },
    #     {
    #         'run_id': run_id, 'date': date(2023, 1, 3), 'end_capital': 9900000.0,
    #         'daily_return': -0.0149, 'daily_profit_loss': -150000.0, 'cumulative_return': -0.01, 'drawdown': 0.015
    #     }
    # ]
    # db_manager.save_backtest_performance(test_performance_data)
    # logger.info("save_backtest_performance 테스트 완료.")

    # # 2.7 fetch_backtest_performance 테스트
    # logger.info(f"fetch_backtest_performance 테스트 시작 (run_id: {run_id})...")
    # fetched_performance_data = db_manager.fetch_backtest_performance(run_id=run_id)
    # logger.info(f"조회된 backtest_performance 데이터:\n{fetched_performance_data}")

    # logger.info(f"fetch_backtest_performance 테스트 시작 (run_id: {run_id}, 특정 기간)...")
    # fetched_performance_data_period = db_manager.fetch_backtest_performance(
    #     run_id=run_id, 
    #     start_date=date(2023, 1, 1), 
    #     end_date=date(2023, 1, 2)
    # )
    # logger.info(f"조회된 backtest_performance 데이터 (2023-01-01~2023-01-02):\n{fetched_performance_data_period}")
    # logger.info("fetch_backtest_performance 테스트 완료.")

    # logger.info("--- 백테스팅 관련 테이블 테스트 완료 ---")

    # # --- 클린업 ---
    # logger.info("--- 모든 테이블 삭제 시작 ---")
    # logger.info("backtest_tables 삭제 중...")
    # #db_manager.drop_backtest_tables()
    # logger.info("stock_tables 삭제 중...")
    # #db_manager.drop_stock_tables()
    # logger.info("--- 모든 테이블 삭제 완료 ---")

    # db_manager.close()
    # logger.info("DB 연결 종료.")
    # logger.info("모든 테스트 완료.")