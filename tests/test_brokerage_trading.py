# tests/test_brokerage.py

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
from datetime import datetime

# --- 프로젝트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.brokerage import Brokerage

class TestBrokerageTradeRecording(unittest.TestCase):

    @patch('manager.trading_manager.TradingManager')
    @patch('api.creon_api.CreonAPIClient')
    def test_trade_conclusion_saves_with_strategy_name(self, mock_api_client, mock_manager):
        """체결 콜백 수신 시, 올바른 strategy_name으로 DB 저장 메서드가 호출되는지 검증"""
        
        # ▼▼▼ [수정] Mock 객체가 현실적인 데이터를 반환하도록 설정 ▼▼▼
        # 1. sync_account_status가 호출하는 API 메서드들의 반환 값을 미리 정의합니다.
        mock_api_client.get_account_balance.return_value = {'cash_balance': 10000000.0}
        mock_api_client.get_portfolio_positions.return_value = [] # 보유 종목 없음
        mock_api_client.get_unfilled_orders.return_value = []   # 미체결 주문 없음
        # ▲▲▲ 수정 완료 ▲▲▲

        # 1. Arrange (준비)
        # 이제 Brokerage를 생성하면 __init__ 내부의 sync_account_status가 Mock 데이터를 사용해 정상적으로 실행됩니다.
        brokerage = Brokerage(api_client=mock_api_client, manager=mock_manager, notifier=MagicMock(), initial_cash=10000000)
        
        test_order_id = "12345"
        test_strategy_name = "SMADaily_Test"
        
        # execute_order가 호출되면 _active_orders에 전략 이름을 저장했다고 가정
        brokerage._active_orders[test_order_id] = {
            'stock_code': 'A005930',
            'stock_name': '삼성전자',
            'order_type': 'buy',
            'strategy_name': test_strategy_name,
            'order_quantity': 10,
            'filled_quantity': 0
        }
        
        # 체결 데이터 시뮬레이션
        conclusion_data = {
            'order_status': '체결',
            'order_id': test_order_id,
            'stock_code': 'A005930',
            'price': 75000,
            'quantity': 10 # 전량 체결
        }

        # 2. Act (실행)
        brokerage.handle_order_conclusion(conclusion_data)
        
        # 3. Assert (검증)
        mock_db_manager = mock_manager.db_manager
        mock_db_manager.save_trading_trade.assert_called_once()
        
        call_args, call_kwargs = mock_db_manager.save_trading_trade.call_args
        saved_data = call_args[0]
        
        self.assertEqual(saved_data['strategy_name'], test_strategy_name)
        self.assertEqual(saved_data['trade_quantity'], 10)

if __name__ == '__main__':
    unittest.main()