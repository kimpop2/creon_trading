import unittest
import sys
import os
import logging
from datetime import datetime, date, timedelta
import time

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.trader_manager import TraderManager
from trade.brokerage import Brokerage

class TestTraderStopLoss(unittest.TestCase):
    """손절매 로직 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 시 한 번만 실행"""
        cls.api = CreonAPIClient()
        cls.trader_manager = TraderManager()
        cls.brokerage = Brokerage(cls.api, cls.trader_manager)
        cls.test_stock_code = 'A005930'  # 삼성전자
        logging.info("테스트 환경 설정 완료")

    def setUp(self):
        """각 테스트 메서드 실행 전에 실행"""
        self.assertTrue(self.api.connected, "크레온 API가 연결되어 있지 않습니다.")
        
        # 현재가 조회
        self.current_price = self.api.get_current_price(self.test_stock_code)
        self.assertIsNotNone(self.current_price, "현재가 조회 실패")
        
        # 손절매 파라미터 설정
        self.stop_loss_params = {
            'stop_loss_pct': -3.0,  # 3% 손실 시 손절
            'trailing_stop_pct': -2.0,  # 최고가 대비 2% 하락 시 손절
            'portfolio_max_drawdown_ratio': -5.0  # 포트폴리오 5% 손실 시 전량 매도
        }
        self.brokerage.set_stop_loss_params(self.stop_loss_params)

    def test_1_individual_stop_loss(self):
        """개별 종목 손절매 테스트"""
        # 1. 시장가 매수로 포지션 생성
        quantity = 1
        order_id = self.api.send_order(
            stock_code=self.test_stock_code,
            order_type='buy',
            price=0,
            quantity=quantity,
            order_kind='03'  # 시장가
        )
        
        self.assertIsNotNone(order_id, "매수 주문 실패")
        time.sleep(1)  # 체결 대기
        
        # 2. 손절매 조건 체크 (현재가 -3% 가정)
        simulated_price = self.current_price * 0.97  # 3% 하락 가정
        now = datetime.now()
        
        executed = self.brokerage.check_and_execute_stop_loss(
            self.test_stock_code,
            simulated_price,
            now
        )
        
        self.assertTrue(executed, "손절매가 실행되지 않았습니다")
        logging.info(f"개별 종목 손절매 실행 결과: {executed}")

    def test_2_trailing_stop(self):
        """트레일링 스탑 테스트"""
        # 1. 시장가 매수로 포지션 생성
        quantity = 1
        order_id = self.api.send_order(
            stock_code=self.test_stock_code,
            order_type='buy',
            price=0,
            quantity=quantity,
            order_kind='03'  # 시장가
        )
        
        self.assertIsNotNone(order_id, "매수 주문 실패")
        time.sleep(1)  # 체결 대기
        
        # 2. 가격 상승 시뮬레이션 (+5%)
        simulated_high_price = self.current_price * 1.05
        now = datetime.now()
        
        # 최고가 갱신
        self.brokerage.check_and_execute_stop_loss(
            self.test_stock_code,
            simulated_high_price,
            now
        )
        
        # 3. 최고가 대비 하락 시뮬레이션 (-2%)
        simulated_current_price = simulated_high_price * 0.98
        executed = self.brokerage.check_and_execute_stop_loss(
            self.test_stock_code,
            simulated_current_price,
            now
        )
        
        self.assertTrue(executed, "트레일링 스탑이 실행되지 않았습니다")
        logging.info(f"트레일링 스탑 실행 결과: {executed}")

    def test_3_portfolio_stop_loss(self):
        """포트폴리오 손절매 테스트"""
        # 1. 초기 포트폴리오 가치 설정
        initial_prices = {self.test_stock_code: self.current_price}
        self.brokerage.initial_portfolio_value = self.brokerage.get_portfolio_value(initial_prices)
        
        # 2. 포트폴리오 가치 하락 시뮬레이션 (-5%)
        simulated_prices = {self.test_stock_code: self.current_price * 0.95}
        now = datetime.now()
        
        executed = self.brokerage.check_and_execute_portfolio_stop_loss(
            simulated_prices,
            now
        )
        
        self.assertTrue(executed, "포트폴리오 손절매가 실행되지 않았습니다")
        logging.info(f"포트폴리오 손절매 실행 결과: {executed}")

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 시 한 번만 실행"""
        if hasattr(cls, 'trader_manager'):
            cls.trader_manager.close()
        logging.info("테스트 환경 정리 완료")

if __name__ == '__main__':
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    unittest.main() 