# trading/brokerage.py

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, date, time, timedelta

from trading.abstract_broker import AbstractBroker
from api.creon_api import CreonAPIClient, OrderType
from manager.db_manager import DBManager # DBManager 직접 사용
from manager.trading_manager import TradingManager # TradingManager에 필요한 데이터 저장 로직이 있으므로 주입
from util.notifier import Notifier # 알림 기능 (텔레그램 등)

# --- 로거 설정 ---
logger = logging.getLogger(__name__)

class Brokerage(AbstractBroker):
    """
    [수정됨] 실제 증권사 API를 통해 매매를 실행하며, 통일된 AbstractBroker 인터페이스를 구현합니다.
    """
    def __init__(self, 
                 api_client: CreonAPIClient, 
                 manager: TradingManager, 
                 notifier: Notifier):
        super().__init__()
        self.api_client = api_client
        self.manager = manager
        self.notifier = notifier
        self._current_cash_balance: float = 0.0
        self._current_positions: Dict[str, Any] = {}
        self.api_client.set_conclusion_callback(self.handle_order_conclusion)
        self.sync_account_status()

    def set_stop_loss_params(self, stop_loss_params: Optional[Dict[str, Any]]):
        self.stop_loss_params = stop_loss_params
        logging.info(f"자동매매 브로커 손절매 파라미터 설정 완료: {stop_loss_params}")

    def execute_order(self, stock_code: str, order_type: str, price: float, quantity: int, order_time: datetime, order_id: Optional[str] = None) -> bool:
        """ [수정] 주문 성공 여부를 bool 값으로 반환하도록 인터페이스와 일치시킵니다. """
        order_type_enum = OrderType.BUY if order_type.lower() == 'buy' else OrderType.SELL
        order_unit = "03" if price == 0 else "01"
        price_to_send = 0 if order_unit == "03" else int(price)
        
        result = self.api_client.send_order(
            stock_code=stock_code, 
            order_type=order_type_enum, 
            quantity=quantity, 
            price=price_to_send, 
            order_unit=order_unit
        )
        
        if result and result['status'] == 'success':
            logger.info(f"주문 요청 성공: {stock_code}, 주문번호: {result['order_num']}")
            return True
        else:
            stock_name = self.manager.get_stock_name(stock_code)
            self.notifier.send_message(f"❌ 주문 실패: {stock_name}({stock_code})")
            logger.error(f"주문 실패: {stock_code}, 메시지: {result.get('message', 'N/A')}")
            return False
    
    def get_current_cash_balance(self) -> float:
        self.sync_account_status()
        return self._current_cash_balance

    def get_current_positions(self) -> Dict[str, Dict[str, Any]]:
        self.sync_account_status()
        return self._current_positions

    def get_position_size(self, stock_code: str) -> int:
        position = self.get_current_positions().get(stock_code)
        return position.get('quantity', 0) if position else 0

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        cash = self.get_current_cash_balance()
        positions = self.get_current_positions()
        holdings_value = sum(pos.get('quantity', 0) * current_prices.get(code, pos.get('avg_price', 0)) for code, pos in positions.items())
        return cash + holdings_value


# [수정] broker.py와 동일한 구조로 변경
    def check_and_execute_stop_loss(self, current_prices: Dict[str, float], current_dt: datetime) -> bool:
        if not self.stop_loss_params:
            return False
            
        executed_any = False
        # 1. 개별 종목 손절/익절 로직
        # get_current_positions()를 호출하여 최신 포지션을 가져옴
        for stock_code in list(self.get_current_positions().keys()):
            if self._check_individual_stock_conditions(stock_code, current_prices, current_dt):
                executed_any = True

        # 2. 포트폴리오 레벨 손절 로직
        if self._check_portfolio_conditions(current_prices, current_dt):
            executed_any = True
            
        return executed_any

    # [신규] broker.py의 로직을 기반으로 작성
    def _check_individual_stock_conditions(self, stock_code: str, current_prices: Dict[str, float], current_dt: datetime) -> bool:
        pos_info = self.get_current_positions().get(stock_code)
        current_price = current_prices.get(stock_code)
        if not pos_info or not current_price or pos_info.get('quantity', 0) <= 0:
            return False

        # 최고가 업데이트 (실시간 환경에서는 DB/캐시와 연동 필요)
        highest_price = pos_info.get('highest_price', 0)
        if current_price > highest_price:
            pos_info['highest_price'] = current_price
            # manager를 통해 DB에 최고가 갱신
            self.manager.save_current_position(pos_info)

        avg_price = pos_info['avg_price'] 
        profit_pct = (current_price - avg_price) * 100 / avg_price if avg_price > 0 else 0

        # 익절
        if profit_pct >= self.stop_loss_params.get('take_profit_ratio', float('inf')):
            logging.info(f"[익절] {stock_code}")
            self.notifier.send_message(f"💰 익절: {stock_code} ({profit_pct:.2%})")
            return self.execute_order(stock_code, 'sell', current_price, pos_info['quantity'], current_dt)
        # 보유기간 기반 손절
        entry_date = pos_info.get('entry_date', current_dt.date())
        holding_days = (current_dt.date() - entry_date).days
        if holding_days <= 3 and profit_pct <= self.stop_loss_params.get('early_stop_loss', -float('inf')):
             logging.info(f"[조기손절] {stock_code}")
             self.notifier.send_message(f"📉 초기 손절: {stock_code} ({profit_pct:.2%})")
             return self.execute_order(stock_code, 'sell', current_price, pos_info['quantity'], current_dt)
        # 일반 손절
        if profit_pct <= self.stop_loss_params.get('stop_loss_ratio', -float('inf')):
            logging.info(f"[손절] {stock_code}")
            self.notifier.send_message(f"🚨 손절: {stock_code} ({profit_pct:.2%})")
            return self.execute_order(stock_code, 'sell', current_price, pos_info['quantity'], current_dt)
        # 트레일링 스탑
        if highest_price > 0:
            trailing_stop_pct = (current_price - highest_price) * 100 / highest_price
            if trailing_stop_pct <= self.stop_loss_params.get('trailing_stop_ratio', -float('inf')):
                logging.info(f"[트레일링 스탑] {stock_code}")
                self.notifier.send_message(f"🛑 트레일링 스탑: {stock_code}")
                return self.execute_order(stock_code, 'sell', current_price, pos_info['quantity'], current_dt)
        
        return False

    # [신규] broker.py의 로직을 기반으로 작성
    def _check_portfolio_conditions(self, current_prices: Dict[str, float], current_dt: datetime) -> bool:
        positions = self.get_current_positions()
        
        # 포트폴리오 전체 손실률 기준
        total_cost = sum(p['quantity'] * p['avg_price'] for p in positions.values())
        if total_cost == 0: return False
        
        total_current_value = sum(p['quantity'] * current_prices.get(code, p['avg_price']) for code, p in positions.items())
        total_profit_pct = (total_current_value - total_cost) * 100 / total_cost

        if total_profit_pct <= self.stop_loss_params.get('portfolio_stop_loss', -float('inf')):
            logging.info(f"[포트폴리오 손절] 전체 손실률 {total_profit_pct:.2%}가 기준치 도달")
            self.notifier.send_message(f"🔥 포트폴리오 손절! 전체 손실률: {total_profit_pct:.2%}")
            self._execute_portfolio_sellout(current_prices, current_dt)
            return True

        # 동시다발적 손실 기준
        stop_loss_pct = self.stop_loss_params.get('stop_loss_ratio', -float('inf'))
        losing_positions_count = 0
        for code, pos in positions.items():
            price = current_prices.get(code)
            if price and ((price - pos['avg_price']) / pos['avg_price']) * 100 <= stop_loss_pct:
                losing_positions_count += 1
        
        if losing_positions_count >= self.stop_loss_params.get('max_losing_positions', float('inf')):
            logging.info(f"[포트폴리오 손절] 손실 종목 수 {losing_positions_count}개가 기준치 도달")
            self.notifier.send_message(f"🔥 포트폴리오 손절! 손실 종목 수: {losing_positions_count}개")
            self._execute_portfolio_sellout(current_prices, current_dt)
            return True

        return False
    
    # [신규] broker.py의 로직을 기반으로 작성
    def _execute_portfolio_sellout(self, current_prices: Dict[str, float], current_dt: datetime):
        """포트폴리오 전체 청산을 실행합니다."""
        logging.info("포트폴리오 전체 청산을 시작합니다.")
        for stock_code in list(self.get_current_positions().keys()):
            pos_info = self.get_current_positions()[stock_code]
            price = current_prices.get(stock_code, pos_info['avg_price'])
            self.execute_order(stock_code, 'sell', price, pos_info['quantity'], current_dt)

    def cleanup(self) -> None:
        """리소스 정리 및 Creon API 연결 해제"""
        logger.info("Brokerage cleanup initiated.")
        # CreonAPIClient의 cleanup은 Trading 클래스에서 최종적으로 호출
        logger.info("Brokerage cleanup completed.")

    def sync_account_status(self):
        """
        Creon API로부터 최신 계좌 잔고, 보유 종목, 미체결 주문 정보를 가져와
        내부 캐시 변수(_current_cash_balance, _current_positions, _unfilled_orders)를 업데이트합니다.
        """
        logger.info("계좌 상태 동기화 시작...")

        # 1. 현금 잔고 업데이트
        balance_info = self.api_client.get_account_balance()
        if balance_info:
            self._current_cash_balance = balance_info.get('cash_balance', 0.0)
            logger.info(f"현금 잔고 업데이트: {self._current_cash_balance:,.0f}원")
        else:
            logger.warning("현금 잔고 조회에 실패했습니다. 현금 잔고를 0으로 설정합니다.")
            self._current_cash_balance = 0.0

        # 2. 보유 종목 업데이트
        # get_portfolio_positions()는 quantity, avg_price, stock_name 등을 포함한 딕셔너리 리스트를 반환합니다.
        positions_list = self.api_client.get_portfolio_positions()
        self._current_positions = {pos['stock_code']: pos for pos in positions_list}
        logger.info(f"보유 종목 업데이트: 총 {len(self._current_positions)}건")

        # 3. 미체결 주문 업데이트
        self._unfilled_orders = self.api_client.get_unfilled_orders()
        logger.info(f"미체결 주문 업데이트: 총 {len(self._unfilled_orders)}건")

        logger.info("계좌 상태 동기화 완료.")


    # --- 실시간 체결/잔고 업데이트 콜백 핸들러 (Creon API 연동) ---
    def handle_order_conclusion(self, conclusion_data: Dict[str, Any]):
        """
        Creon API에서 실시간 체결/주문 응답이 왔을 때 호출되는 콜백 함수.
        trading_log 테이블을 업데이트하고, 보유 종목 및 현금 잔고를 동기화합니다.
        CreonAPIClient의 `set_conclusion_callback`에 등록됩니다.
        """
        logger.info(f"체결/주문응답 수신: {conclusion_data}")
        order_id = conclusion_data.get('order_num')
        original_order_id = conclusion_data.get('order_num')
        stock_code = conclusion_data.get('code')
        # order_type 영문 변환 (콜백 데이터)
        order_type = conclusion_data.get('buy_sell').lower() # '매수' -> 'buy', '매도' -> 'sell'
        if order_type == '매수':
            order_type_for_log = 'buy'
        elif order_type == '매도':
            order_type_for_log = 'sell'
        else:
            order_type_for_log = order_type
        order_status = conclusion_data.get('flag') # 예: '접수', '체결', '부분체결', '거부', '확인', '정정', '취소'
        filled_quantity = conclusion_data.get('quantity', 0)
        filled_price = conclusion_data.get('price', 0)
        unfilled_quantity = 0 # TODO: 정확한 미체결 수량 계산 로직 필요
        stock_name = self.manager.get_stock_name(stock_code)
        trade_date = datetime.now().date()
        trade_time = datetime.now().time()
        commission = 0
        tax = 0
        net_amount = 0
        if order_status in ['체결', '부분체결'] and filled_quantity > 0:
            transaction_amount = filled_price * filled_quantity
            commission = transaction_amount * self.commission_rate
            if order_type == 'sell':
                tax = transaction_amount * self.tax_rate_sell
            if order_type == 'buy':
                net_amount = -(transaction_amount + commission)
            else:
                net_amount = transaction_amount - commission - tax
            self.notifier.send_message(f"🔔 {order_status}: {stock_name}({stock_code}) {order_type.upper()} {filled_quantity}주 @ {filled_price:,.0f}원")
            logger.info(f"거래 체결: {stock_code}, 수량: {filled_quantity}, 가격: {filled_price}, 순매매액: {net_amount}")
        log_data = {
            'order_id': order_id,
            'original_order_id': original_order_id,
            'stock_code': stock_code,
            'stock_name': stock_name,
            'trading_date': trade_date,
            'trading_time': trade_time,
            'order_type': order_type_for_log,
            'order_price': filled_price,
            'order_quantity': filled_quantity,
            'filled_price': filled_price,
            'filled_quantity': filled_quantity,
            'unfilled_quantity': unfilled_quantity,
            'order_status': order_status,
            'commission': commission,
            'tax': tax,
            'net_amount': net_amount,
            'credit_type': '현금'
        }
        self.manager.save_trading_log(log_data)
        self.sync_account_status()


    def cancel_order(self, order_id: str, stock_code: str, quantity: int = 0) -> bool:
        """
        진행 중인 주문을 취소합니다. Creon API를 통해 취소 요청을 보냅니다.
        :param order_id: 취소할 주문의 주문번호
        :param stock_code: 종목코드
        :param quantity: 취소할 수량 (0이면 잔량 취소)
        """
        result = self.api_client.send_order(
            stock_code=stock_code,
            order_type=OrderType.CANCEL,
            quantity=quantity,
            org_order_num=order_id
        )
        if result and result['status'] == 'success':
            logger.info(f"주문 취소 요청 성공: 주문번호 {order_id}")
            self.notifier.send_message(f"⚠️ 주문 취소 요청: 주문ID {order_id}")
            return True
        else:
            logger.error(f"주문 취소 요청 실패: 주문번호 {order_id}")
            self.notifier.send_message(f"❗ 주문 취소 요청 실패: 주문ID {order_id}")
            return False

    def amend_order(self,
                    order_id: str,
                    stock_code: str, # 종목코드 추가
                    new_price: Optional[float] = None,
                    new_quantity: Optional[int] = None
                   ) -> Optional[str]:
        """
        진행 중인 주문을 정정합니다. Creon API를 통해 정정 요청을 보냅니다.
        """
        result = self.api_client.send_order(
            stock_code=stock_code,
            order_type=OrderType.MODIFY,
            quantity=new_quantity or 0,
            price=int(new_price) if new_price else 0,
            org_order_num=order_id
        )
        if result and result['status'] == 'success':
            amended_order_id = result['order_num']
            logger.info(f"주문 정정 요청 성공: 원주문 {order_id} -> 정정주문 {amended_order_id}")
            self.notifier.send_message(f"🔄 주문 정정 요청: 원주문ID {order_id} -> 새 주문ID {amended_order_id}")
            return amended_order_id
        else:
            logger.error(f"주문 정정 요청 실패: 주문번호 {order_id}")
            self.notifier.send_message(f"❗ 주문 정정 요청 실패: 주문ID {order_id}")
            return None


    def get_unfilled_orders(self) -> List[Dict[str, Any]]:
        """
        실시간 미체결 주문 내역을 조회하고 반환합니다.
        내부 캐시를 반환하며, 이 캐시는 sync_account_status()를 통해 주기적으로 업데이트됩니다.
        """
        return self._unfilled_orders





