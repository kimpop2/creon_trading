# PYTHON 블럭
import unittest
import logging
from datetime import date, timedelta
import os
import sys

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.setup_manager import SetupManager
from config.settings import UNIVERSE_CONFIGS # 실제 경로에 맞게 수정 필요

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestSetupManagerUniverseIntegration(unittest.TestCase):
    """
    SetupManager의 동적 유니버스 생성 기능을 검증하는 통합 테스트 클래스.
    [주의] 실제 DB에 데이터를 쓰고 지우므로, 개발/테스트용 DB 환경에서 실행해야 합니다.
    """

    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시, API 및 DB 연결"""
        try:
            cls.api_client = CreonAPIClient()
            cls.db_manager = DBManager()
            cls.setup_manager = SetupManager(cls.api_client, cls.db_manager)
            cls.assertTrue(cls.api_client.is_connected(), "Creon API 연결 실패")
            cls.assertIsNotNone(cls.db_manager.get_db_connection(), "DB 연결 실패")
            logging.info("통합 테스트 환경 설정 완료: Creon API 및 DB 연결 성공")
        except Exception as e:
            logging.critical(f"테스트 환경 설정 중 심각한 오류 발생: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시, 연결 해제"""
        if hasattr(cls, 'db_manager'):
            cls.db_manager.close()
        logging.info("통합 테스트 환경 정리 완료: DB 연결 해제")

    def setUp(self):
        """각 테스트 케이스 실행 전, 테스트 데이터 삽입"""
        self.test_date = date.today() - timedelta(days=365)
        self.test_stocks = {
            # 동적 필터링 통과 대상
            'A005930': {'name': '삼성전자'},
            # 동적 필터링 탈락 대상
            'A035720': {'name': '카카오'},           # min_market_cap (1000억) 미달
            'A373220': {'name': 'LG에너지솔루션'},   # min_avg_trading_value (10억) 미달
            'A000020': {'name': '동화약품'},         # min_price (1000원) 미달
            'A123456': {'name': '가상종목_NULL'},    # market_cap이 NULL이라 자동 제외
            # 기본 필터링 탈락 대상
            'A005935': {'name': '삼성전자우'},
            'A382300': {'name': '교보11호스팩'},
        }
        self._insert_test_data()

    def tearDown(self):
        """각 테스트 케이스 실행 후, 테스트 데이터 삭제"""
        self._delete_test_data()

    def _insert_test_data(self):
        """테스트에 필요한 임시 데이터를 DB에 삽입합니다."""
        logging.info("테스트 데이터 삽입 시작")
        stock_info_data = [(code, data['name']) for code, data in self.test_stocks.items()]
        self.db_manager.execute_sql(
            "INSERT INTO stock_info (stock_code, stock_name) VALUES (%s, %s) ON DUPLICATE KEY UPDATE stock_name=VALUES(stock_name)",
            stock_info_data
        )
        
        # [수정] save_daily_price가 요구하는 모든 키를 포함하여 데이터 생성
        price_data = [
            {'stock_code': 'A005930', 'date': self.test_date, 'open': 70000, 'high': 71000, 'low': 69000, 'close': 70000, 'volume': 1000000, 'trading_value': 700 * 10**8},
            {'stock_code': 'A035720', 'date': self.test_date, 'open': 50000, 'high': 51000, 'low': 49000, 'close': 50000, 'volume': 500000, 'trading_value': 250 * 10**8},
            {'stock_code': 'A373220', 'date': self.test_date, 'open': 400000, 'high': 410000, 'low': 390000, 'close': 400000, 'volume': 200000, 'trading_value': 800 * 10**8},
            {'stock_code': 'A000020', 'date': self.test_date, 'open': 900, 'high': 910, 'low': 890, 'close': 900, 'volume': 10000, 'trading_value': 0.09 * 10**8}, # min_price 탈락
            {'stock_code': 'A123456', 'date': self.test_date, 'open': 10000, 'high': 11000, 'low': 9000, 'close': 10000, 'volume': 50000, 'trading_value': 5 * 10**8},
        ]
        self.db_manager.save_daily_price(price_data)

        # [수정] save_daily_factors에 맞게 데이터를 생성하고 호출 코드 주석 해제
        #       - avg_trading_value 대신 trading_value를 사용하므로, daily_factors의 trading_value를 avg_trading_value처럼 사용
        factor_data = [
            {'date': self.test_date, 'stock_code': 'A005930', 'market_cap': 4000 * 10**8, 'trading_value': 50 * 10**8}, # 통과
            {'date': self.test_date, 'stock_code': 'A035720', 'market_cap': 500 * 10**8,  'trading_value': 20 * 10**8},  # min_market_cap 탈락
            {'date': self.test_date, 'stock_code': 'A373220', 'market_cap': 8000 * 10**8, 'trading_value': 5 * 10**8},   # min_avg_trading_value 탈락
            {'date': self.test_date, 'stock_code': 'A000020', 'market_cap': 300 * 10**8,  'trading_value': 1 * 10**8},
            {'date': self.test_date, 'stock_code': 'A123456', 'market_cap': None,         'trading_value': 30 * 10**8},   # market_cap이 NULL
        ]
        # save_daily_factors는 딕셔너리 리스트를 받으므로 그대로 전달
        self.db_manager.save_daily_factors(factor_data)

        logging.info(f"{len(self.test_stocks)}개 종목 테스트 데이터 삽입 완료")

    def _delete_test_data(self):
        """테스트에 사용된 임시 데이터를 DB에서 삭제합니다."""
        logging.info("테스트 데이터 삭제 시작")
        codes_tuple = tuple(self.test_stocks.keys())
        self.db_manager.execute_sql(f"DELETE FROM daily_price WHERE stock_code IN {codes_tuple}")
        self.db_manager.execute_sql(f"DELETE FROM daily_factors WHERE stock_code IN {codes_tuple}")
        self.db_manager.execute_sql(f"DELETE FROM stock_info WHERE stock_code IN {codes_tuple}")
        logging.info("테스트 데이터 삭제 완료")

    # def test_01_select_universe_initial_filtering(self):
    #     """
    #     [Test] 1단계: 기본적인 필터링(우선주, 스팩 제외) 기능 검증
    #     """
    #     # GIVEN: setUp에서 테스트 데이터가 준비됨
        
    #     # WHEN: 1차 필터링 로직 호출
    #     filtered_codes = self.setup_manager.select_universe_initial_filter()

    #     # THEN: 검증 로직 수정
    #     # 1. 통과해야 할 정상 종목들이 결과에 모두 포함되어 있는지 확인
    #     expected_pass_codes = {'A005930', 'A035720', 'A373220'}
    #     self.assertTrue(expected_pass_codes.issubset(filtered_codes),
    #                     f"통과해야 할 종목 {expected_pass_codes - filtered_codes}이(가) 결과에 없습니다.")

    #     # 2. 필터링되어야 할 종목들이 결과에서 모두 제외되었는지 확인
    #     expected_fail_codes = {'A005935', 'A382300'}
    #     for code in expected_fail_codes:
    #         self.assertNotIn(code, filtered_codes,
    #                          f"제외되어야 할 종목 {code}이(가) 결과에 포함되어 있습니다.")
        
    #     logging.info("기본 필터링 테스트 통과!")

    # def test_02_select_universe_dynamic_filtering(self):
    #     """
    #     [Test] 2단계: 동적 필터링(min_price, min_market_cap 등) 기능 검증
    #     """
    #     # GIVEN: setUp에서 각 필터링 조건에 맞는 통과/실패 데이터가 준비됨
    #     # UNIVERSE_CONFIGS의 'MOMENTUM_THEME_LEADER_V1' 사용
    #     # (min_market_cap: 1000억, min_avg_trading_value: 10억, min_price: 1000원)

    #     # WHEN: select_universe 메서드 호출
    #     filtered_codes = self.setup_manager.select_universe('MOMENTUM_THEME_LEADER_V1', self.test_date)

    #     # THEN: 모든 필터링 조건을 통과한 'A005930'만 남아야 함
    #     expected_codes = {'A005930'}
        
    #     self.assertEqual(filtered_codes, expected_codes, "동적 필터링이 올바르게 동작하지 않았습니다.")
    #     logging.info("동적 필터링 테스트 통과!")

    # 아래 test_03, test_04를 이전에 제안드렸던 최종 버전으로 교체합니다.
    # def test_03_update_daily_factors_scores(self):
    #     """
    #     [Test] 3단계 (신규): 배치 메서드가 원본 팩터로 0~9 등급 점수를 올바르게 계산하는지 검증
    #     """
    #     scoring_test_date = date.today()
 
    #     # WHEN: 새로운 배치 메서드 호출
    #     self.setup_manager.update_daily_factors_scores(scoring_test_date)

    #     # THEN: DB에서 다시 조회하여 price_trend_score가 0~9 사이의 등급으로 매겨졌는지 확인
    #     updated_factors_df = self.db_manager.fetch_all_factors_for_scoring(scoring_test_date)
        
    #     self.assertFalse(updated_factors_df.empty)
    #     self.assertIn('price_trend_score', updated_factors_df.columns)
        
    #     score_low = updated_factors_df[updated_factors_df['stock_code'] == 'A100090']['price_trend_score'].iloc[0]
    #     score_high = updated_factors_df[updated_factors_df['stock_code'] == 'A042670']['price_trend_score'].iloc[0]

    #     self.assertLess(score_high, score_low, "qcut 등급 점수가 올바르게 매겨지지 않았습니다.")
    #     self.assertTrue(0 <= score_low <= 9)
    #     self.assertTrue(0 <= score_high <= 9)
        
    #     logging.info("개별 점수 계산 배치 메서드 테스트 통과!")


    def test_04_universe_set_operations(self):
        """
        [Test] 4단계 (신규): 두 개의 다른 유니버스 설정을 실행하고,
        결과가 정상적으로 set 타입으로 반환되어 집합 연산이 가능한지 검증
        """
        # GIVEN: 최근 팩터 데이터가 DB에 있다고 가정하고, 테스트 날짜를 설정
        # (실제 운영 시에는 date.today() 또는 직전 영업일을 사용하는 것이 더 좋습니다)
        test_date = date.today()
        
        # WHEN: settings.py에 정의된 실제 유니버스 설정 두 개를 사용하여 각각 종목을 선정
        momentum_universe = self.setup_manager.select_universe('MOMENTUM_THEME_LEADER_V1', test_date)
        stable_universe = self.setup_manager.select_universe('INSTITUTIONAL_BUY_STABLE_V1', test_date)
        # THEN: 반환된 결과가 set 타입이며, 집합 연산이 오류 없이 수행되는지 검증
        
        # 1. 반환 타입 검증
        self.assertIsInstance(momentum_universe, set, "MOMENTUM_THEME_LEADER_V1의 결과가 set이 아닙니다.")
        self.assertIsInstance(stable_universe, set, "INSTITUTIONAL_BUY_STABLE_V1의 결과가 set이 아닙니다.")
        
        # 2. 유니버스가 비어있지 않은지 확인 (DB에 데이터가 충분하다면 통과)
        self.assertTrue(momentum_universe, "MOMENTUM_THEME_LEADER_V1 유니버스가 비어있습니다.")
        self.assertTrue(stable_universe, "INSTITUTIONAL_BUY_STABLE_V1 유니버스가 비어있습니다.")
        
        # 3. 집합 연산 수행 및 결과 로깅
        try:
            # 교집합: 두 유니버스에 공통으로 속한 종목
            intersection_set = momentum_universe & stable_universe
            
            # 합집합: 두 유니버스를 합친 전체 종목 (중복 제외)
            union_set = momentum_universe | stable_universe
            
            # 차집합: 모멘텀 유니버스에만 속한 종목
            momentum_only_set = momentum_universe - stable_universe

            logging.info("집합 연산 테스트 통과!")
            logging.info(f"  - 모멘텀 유니버스: {len(momentum_universe)}개")
            logging.info(f"  - 안정형 유니버스: {len(stable_universe)}개")
            logging.info(f"  - 교집합: {len(intersection_set)}개")
            logging.info(f"  - 합집합: {len(union_set)}개")
            
            # 연산 결과도 set 타입인지 확인
            self.assertIsInstance(intersection_set, set)
            
        except Exception as e:
            self.fail(f"집합 연산 중 예외가 발생했습니다: {e}")

if __name__ == '__main__':
    unittest.main()