import time
from datetime import datetime
import pythoncom

# 프로젝트 경로 추가
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 실제 객체들 임포트
from api.creon_api2 import CreonAPIClient, OrderType
from trading.brokerage import Brokerage
from manager.db_manager import DBManager
from manager.trading_manager import TradingManager
from util.notifier import Notifier

# --- 테스트 설정 ---
# ❗️ 모의 투자 계좌에서 사용할 테스트 종목
TEST_STOCK_CODE = 'A090710'  # 휴림로봇
TEST_ORDER_QUANTITY = 1

def run_real_trading_test():
    """실제 객체를 사용한 매매 시나리오 테스트"""
    api = None
    try:
        # 1. 실제 객체 초기화
        print("--- 1. 실제 객체 초기화 시작 ---")
        pythoncom.CoInitialize()
        api = CreonAPIClient()
        db = DBManager()
        manager = TradingManager(api, db)
        notifier = Notifier()
        broker = Brokerage(api, manager, notifier)
        print("--- 객체 초기화 완료 ---")

        # 2. 테스트 시작 전 모든 미체결 주문 취소 (안전장치)
        print("\n--- 2. 테스트 시작 전 미체결 주문 정리 ---")
        unfilled_orders = api.get_unfilled_orders()
        for order in unfilled_orders:
            if order.get('stock_code') == TEST_STOCK_CODE:
                print(f"미체결 주문 취소 시도: {order}")
                broker.cancel_order(order['order_id'], TEST_STOCK_CODE)
                time.sleep(1) # 취소 요청 간 간격
        
        # 3. 테스트 시나리오: 시장가 매수 -> 잔고 확인 -> 시장가 매도 -> 잔고 확인
        print("\n--- 3. 매수-매도 시나리오 시작 ---")
        
        # 매수 전 잔고 확인
        initial_positions = broker.get_current_positions()
        print(f"매수 전 보유 수량: {initial_positions.get(TEST_STOCK_CODE, {}).get('quantity', 0)}주")

        # [STEP 3-1] 시장가 매수 주문 실행
        print("\n[STEP 3-1] 시장가 매수 주문 실행")
        buy_order_id = broker.execute_order(
            stock_code=TEST_STOCK_CODE,
            order_type='buy',
            price=0,
            quantity=TEST_ORDER_QUANTITY,
            order_time=datetime.now()  # 💡 [수정] 현재 시간을 order_time으로 전달
        )
        if not buy_order_id:
            raise Exception("시장가 매수 주문 실패")

        print(f"매수 주문 접수 완료. 주문번호: {buy_order_id}. 5초 후 체결 및 잔고 확인...")
        time.sleep(5) # 체결 대기

        # 매수 후 잔고 확인
        broker.sync_account_status() # 계좌 상태 강제 동기화
        positions_after_buy = broker.get_current_positions()
        qty_after_buy = positions_after_buy.get(TEST_STOCK_CODE, {}).get('quantity', 0)
        print(f"매수 후 보유 수량: {qty_after_buy}주")
        if qty_after_buy < TEST_ORDER_QUANTITY:
            raise Exception("매수 후 수량이 일치하지 않음")

        # [STEP 3-2] 시장가 매도 주문 실행
        print("\n[STEP 3-2] 시장가 매도 주문 실행")
        sell_order_id = broker.execute_order(
            stock_code=TEST_STOCK_CODE,
            order_type='sell',
            price=0,
            quantity=qty_after_buy,
            order_time=datetime.now()  # 💡 [수정] 현재 시간을 order_time으로 전달
        )
        if not sell_order_id:
            raise Exception("시장가 매도 주문 실패")
            
        print(f"매도 주문 접수 완료. 주문번호: {sell_order_id}. 5초 후 체결 및 잔고 확인...")
        time.sleep(5) # 체결 대기
        
        # 매도 후 잔고 확인
        broker.sync_account_status()
        positions_after_sell = broker.get_current_positions()
        qty_after_sell = positions_after_sell.get(TEST_STOCK_CODE, {}).get('quantity', 0)
        print(f"매도 후 보유 수량: {qty_after_sell}주")
        if qty_after_sell != 0:
            raise Exception("매도 후 잔고가 0이 아님")
            
        print("\n✅ 모든 시나리오 테스트 성공!")

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
    finally:
        # 4. 리소스 정리
        if api:
            print("\n--- 4. 리소스 정리 ---")
            api.cleanup()
        pythoncom.CoUninitialize()
        print("테스트 종료.")


if __name__ == '__main__':
    run_real_trading_test()