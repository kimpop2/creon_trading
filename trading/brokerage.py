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
    실제 증권사 API를 통해 매매를 실행하며, 통일된 AbstractBroker 인터페이스를 구현합니다.
    """
    def __init__(self, 
                 api_client: CreonAPIClient, 
                 manager: TradingManager, 
                 notifier: Notifier,
                 initial_cash: float):
        super().__init__()
        self.api_client = api_client
        self.manager = manager
        self.notifier = notifier
        self.initial_cash=initial_cash
        self.commission_rate = 0.0015
        self.tax_rate_sell = 0.002
        self._current_cash_balance: float = 0.0
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.transaction_log: list = []
        self._active_orders: Dict[str, Dict[str, Any]] = {}
        # 
        self.api_client.set_conclusion_callback(self.handle_order_conclusion)
        # 현금잔고 _current_cash_balance, 보유종목 positions 증권사 정보로 업데이트
        self.sync_account_status() 

    def set_stop_loss_params(self, stop_loss_params: Optional[Dict[str, Any]]):
        self.stop_loss_params = stop_loss_params
        logging.info(f"자동매매 브로커 손절매 파라미터 설정 완료: {stop_loss_params}")

    def execute_order(self, stock_code: str, order_type: str, price: float, quantity: int, order_time: datetime, order_id: Optional[str] = None) -> Optional[str]: # 성공 시 주문 ID(str), 실패 시 None
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
            order_id = result['order_id']
            logger.info(f"주문 요청 성공: {stock_code}, 주문번호: {result['order_id']}")
            
            # 주문 성공 시, _active_orders에 주문 정보 등록
            self._active_orders[order_id] = {
                'stock_code': stock_code,
                'stock_name': self.manager.get_stock_name(stock_code),
                'order_type': order_type.lower(),
                'order_status': '접수',  # '접수' 상태
                'order_price': price,
                'order_quantity': quantity,
                'filled_quantity': 0,
                'unfilled_quantity': quantity,
                'order_time': order_time,
                'original_order_id': None
            }
            return order_id
            
        else:
            stock_name = self.manager.get_stock_name(stock_code)
            self.notifier.send_message(f"❌ 주문 실패: {stock_name}({stock_code})")
            logger.error(f"주문 실패: {stock_code}, 메시지: {result.get('message', 'N/A')}")
            return None
    
    def get_current_cash_balance(self) -> float:
        #self.sync_account_status() 불필요한 호출 제거, 내부상태 반환
        return self._current_cash_balance

    def get_current_positions(self) -> Dict[str, Dict[str, Any]]:
        #self.sync_account_status() 불필요한 호출 제거, 내부상태 반환
        return self.positions

    def get_position_size(self, stock_code: str) -> int:
        return self.positions.get(stock_code, {}).get('quantity', 0)

    def get_portfolio_value(self, current_prices: Dict[str, Any]) -> float:
        "주식 가치 계산 로직"
        cash = self._current_cash_balance

        holdings_value = 0
        for code, pos in self.positions.items():
            price_data = current_prices.get(code)
            # 실시간 가격이 있으면 'close' 값을 사용하고, 없으면 기존 평균가를 사용
            price_to_use = price_data['close'] if price_data and 'close' in price_data else pos.get('avg_price', 0)
            holdings_value += pos.get('quantity', 0) * price_to_use
        
    
    def get_unfilled_stock_codes(self) -> set:
        """미체결 상태인 주문들의 종목 코드 집합을 반환합니다."""
        return {order['stock_code'] for order in self._active_orders.values()}
        
    def handle_order_conclusion(self, conclusion_data: Dict[str, Any]):
        """
        [수정 필수] '체결' 상태일 때만 잔고 및 포지션을 업데이트하도록 수정한 최종 버전입니다.
        """
        logger.info(f"체결/주문응답 수신: {conclusion_data}")
        order_status_str = conclusion_data.get('order_status')
        order_id = conclusion_data.get('order_id')
        stock_code = conclusion_data.get('stock_code')
        
        active_order = self._active_orders.get(order_id)
        if not active_order:
            logger.warning(f"활성 주문 목록에 없는 주문 응답 수신: {order_id}")
            return

        # [수정] 상태 문자열만 먼저 업데이트
        active_order['order_status'] = order_status_str.lower()
        
        # '체결' 또는 '부분체결' 이벤트일 때만 실제 잔고 및 포지션을 변경합니다.
        if order_status_str in ['체결', '부분체결']:
            filled_quantity = conclusion_data.get('quantity', 0)
            if filled_quantity > 0:
                active_order['filled_quantity'] += filled_quantity
                active_order['unfilled_quantity'] = active_order['order_quantity'] - active_order['filled_quantity']
                
                logger.info(f"주문({order_id}) 상태 업데이트: {active_order['order_status']}, 누적 체결수량: {active_order['filled_quantity']}")

                filled_price = conclusion_data.get('price', 0)
                order_type = active_order['order_type']

                # --- (이하 현금 및 포지션 업데이트 로직은 동일) ---
                transaction_amount = filled_price * filled_quantity
                commission = transaction_amount * self.commission_rate
                tax = transaction_amount * self.tax_rate_sell if order_type == 'sell' else 0
                net_amount = (transaction_amount - commission - tax) if order_type == 'sell' else -(transaction_amount + commission)
                
                self._current_cash_balance += net_amount
                logger.info(f"[{order_type.upper()}] 현금 잔고 업데이트: {net_amount:,.0f}원 -> 현재 잔고: {self._current_cash_balance:,.0f}원")

                if order_type == 'buy':
                    if stock_code in self.positions:
                        pos = self.positions[stock_code]
                        total_cost = (pos['avg_price'] * pos['quantity']) + (filled_price * filled_quantity)
                        pos['quantity'] += filled_quantity
                        pos['avg_price'] = total_cost / pos['quantity']
                    else:
                        self.positions[stock_code] = {
                            'stock_code': stock_code, 'stock_name': active_order.get('stock_name'),
                            'quantity': filled_quantity, 'avg_price': filled_price,
                            'entry_date': datetime.now().date(), 'highest_price': filled_price
                        }
                elif order_type == 'sell':
                    if stock_code in self.positions:
                        pos = self.positions[stock_code]
                        pos['quantity'] -= filled_quantity
                        if pos['quantity'] <= 0:
                            del self.positions[stock_code]
        
        # '접수', '확인'은 최종 상태가 아니므로, ['체결', '취소', '거부']일 때만 최종 완료로 간주합니다.
        if active_order['unfilled_quantity'] <= 0 or active_order['order_status'] in ['체결', '취소', '거부']:
            logger.info(f"주문({order_id}) 최종 완료({active_order['order_status']}). 활성 주문 목록에서 제거합니다.")
            del self._active_orders[order_id]

    # --- 실시간 체결/잔고 업데이트 콜백 핸들러 (Creon API 연동) ---
    # def handle_order_conclusion(self, conclusion_data: Dict[str, Any]):
    #     """
    #     Creon API에서 실시간 체결/주문 응답이 왔을 때 호출되는 콜백 함수.
    #     trading_log 테이블을 업데이트하고, 보유 종목 및 현금 잔고를 동기화합니다.
    #     CreonAPIClient의 `set_conclusion_callback`에 등록됩니다.
    #     """
    #     logger.info(f"체결/주문응답 수신: {conclusion_data}")
    #     order_status = conclusion_data.get('order_status') # 예: '접수', '체결', '부분체결', '거부', '확인', '정정', '취소'
    #     order_id = conclusion_data.get('order_id')
    #     origin_order_id = conclusion_data.get('origin_order_id')
    #     stock_code = conclusion_data.get('stock_code')
        
    #     # 1. 활성 주문 목록(_active_orders)에서 해당 주문 정보 업데이트
    #     if order_id in self._active_orders:
    #         order_info = self._active_orders[order_id]
            
    #         filled_quantity = conclusion_data.get('quantity', 0)
    #         order_info['order_status'] = order_status.lower()
    #         order_info['filled_quantity'] += filled_quantity
    #         order_info['unfilled_quantity'] = order_info['order_quantity'] - order_info['filled_quantity']

    #         logger.info(f"주문({order_id}) 상태 업데이트: {order_info['order_status']}, 체결수량: {filled_quantity}")
    #     else:
    #         logger.warning(f"활성 주문 목록에 없는 주문 응답 수신: {order_id}")
    #         # 필요시 여기서 DB 조회 후 비정상 주문 처리 로직 추가 가능
    #         return

    #      # 2. 체결 이벤트인 경우, DB에 로그 저장 및 알림
    #     if order_status in ['체결', '부분체결'] and filled_quantity > 0:
    #         filled_price = conclusion_data.get('price', 0)
    #         order_type_for_log = order_info['order_type']

    #         transaction_amount = filled_price * filled_quantity
    #         commission = transaction_amount * self.commission_rate
    #         tax = transaction_amount * self.tax_rate_sell if order_type_for_log == 'sell' else 0
    #         net_amount = (transaction_amount - commission - tax) if order_type_for_log == 'sell' else -(transaction_amount + commission)
    #         trade_date = datetime.now().date()
    #         trade_time = datetime.now().time()
    #         stock_name_for_log = order_info.get('stock_name')
    #         original_order_id_for_log = order_info.get('original_order_id')
            
    #         log_data = {
    #             # --- 주문 식별 정보 ---
    #             'order_id': order_id,
    #             'original_order_id': origin_order_id,
    #             # --- 체결 기본 정보 ---
    #             'stock_code': stock_code,
    #             'stock_name': stock_name_for_log,
    #             'trade_type': order_type_for_log, # 'buy' 또는 'sell'
    #             'trading_datetime': datetime.combine(trade_date, trade_time), # 날짜와 시간을 합침
    #             # --- 체결 결과 정보 (핵심) ---
    #             'filled_price': filled_price,
    #             'filled_quantity': filled_quantity,
    #             # --- 비용 및 정산 정보 ---
    #             'commission': commission,
    #             'tax': tax,
    #             'net_amount': net_amount, # 순매매금액 : 수수료+세금 포함 매매에 소요된 총비용용
    #             'credit_type': '현금'
    #         }
    #         self.manager.save_trading_log(log_data)
            
            
    #         self.notifier.send_message(f"🔔 {order_status}: {stock_name_for_log}({stock_code}) {order_type_for_log.upper()} {filled_quantity}주 @ {filled_price:,.0f}원")
       
    #     # 3. 주문이 완전히 종료되었는지 확인하고 처리
    #     is_complete = order_info['unfilled_quantity'] <= 0 or order_status in ['취소', '거부', '체결']
    #     if is_complete:
    #         logger.info(f"주문({order_id}) 최종 완료. 활성 주문 목록에서 제거합니다.")
    #         del self._active_orders[order_id]
    #         # 주문이 완전히 종료되면, 계좌 상태를 최종 동기화하여 정확성을 보장
    #         self.sync_account_status()


    # [신규] DB에 저장하지 않는 상태(highest_price)를 복원하는 메서드
    def _restore_positions_state(self, data_store: Dict[str, Any]):
        """
        프로그램 시작 시, 메모리상의 포지션 정보에 DB에 없는 상태값(예: 최고가)을
        과거 데이터를 기반으로 계산하여 복원합니다.
        """
        logger.info("보유 포지션의 상태(최고가) 복원을 시작합니다.")
        today = datetime.now().date()

        for code, pos_info in self.positions.items():
            entry_date = pos_info.get('entry_date')
            if not entry_date:
                pos_info['highest_price'] = pos_info.get('avg_price', 0)
                continue

            # 1. 매수일 ~ 어제까지의 일봉 데이터에서 최고가 찾기
            daily_df = data_store['daily'].get(code)
            historical_high = 0
            if daily_df is not None:
                historical_df = daily_df[(daily_df.index.date >= entry_date) & (daily_df.index.date < today)]
                if not historical_df.empty:
                    historical_high = historical_df['high'].max()

            # 2. 오늘의 분봉 데이터에서 최고가 찾기
            today_high = 0
            if code in data_store['minute'] and today in data_store['minute'][code]:
                today_minute_df = data_store['minute'][code][today]
                if not today_minute_df.empty:
                    today_high = today_minute_df['high'].max()
            
            # 3. 두 기간의 최고가와 평균 매수가 중 가장 높은 값을 최종 highest_price로 설정
            restored_highest_price = max(historical_high, today_high, pos_info.get('avg_price', 0))
            pos_info['highest_price'] = restored_highest_price
            
            logger.debug(f"[{code}] 최고가 복원 완료: {restored_highest_price:,.0f}원")

    def sync_account_status(self):
        """
        [수정] API에서 계좌 정보를 가져와 내부 상태(현금, 포지션)를 업데이트합니다.
        프로그램 시작 시, 미체결 내역을 _active_orders로 복원합니다.
        """
        logger.info("계좌 상태 동기화 시작...")

        # 1. 현금 잔고 업데이트
        balance_info = self.api_client.get_account_balance()
        self._current_cash_balance = balance_info.get('cash_balance', 0.0) if balance_info else 0.0
        logger.info(f"현금 잔고 업데이트: {self._current_cash_balance:,.0f}원")

        # 2. 보유 종목 업데이트
        positions_list = self.api_client.get_portfolio_positions()
        self.positions = {pos['stock_code']: pos for pos in positions_list}
        logger.info(f"보유 종목 업데이트: 총 {len(self.positions)}건")

        # 3. 미체결 주문을 _active_orders로 복원 (주로 프로그램 시작 시)
        unfilled_orders_from_api = self.api_client.get_unfilled_orders()
        # 현재 _active_orders에 없는 미체결 주문만 추가 (중복 방지)
        for order in unfilled_orders_from_api:
            order_id = order.get('order_id')
            if order_id not in self._active_orders:
                self._active_orders[order_id] = {
                    'stock_code': order.get('stock_code'),
                    'order_type': 'buy' if order.get('buy_sell') == '매수' else 'sell',
                    'order_status': 'submitted', # API 미체결은 '접수' 상태로 간주
                    'order_price': order.get('price'),
                    'order_quantity': order.get('quantity'),
                    'filled_quantity': order.get('filled_quantity', 0),
                    'unfilled_quantity': order.get('unfilled_quantity'),
                    'order_time': order.get('time'),
                    'original_order_id': None
                }
                logger.info(f"미체결 주문 복원: 주문번호 {order_id}")
        
        logger.info("계좌 상태 동기화 완료.")

    def get_unfilled_orders(self) -> List[Dict[str, Any]]:
        """
        실시간으로 관리되는 활성 주문(주문 후 ~ 체결 직전) 목록을 반환합니다.
        """
        return list(self._active_orders.values())

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
            origin_order_id=order_id
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
            org_order_id=order_id
        )
        if result and result['status'] == 'success':
            amended_order_id = result['order_id']
            logger.info(f"주문 정정 요청 성공: 원주문 {order_id} -> 정정주문 {amended_order_id}")
            self.notifier.send_message(f"🔄 주문 정정 요청: 원주문ID {order_id} -> 새 주문ID {amended_order_id}")
            return amended_order_id
        else:
            logger.error(f"주문 정정 요청 실패: 주문번호 {order_id}")
            self.notifier.send_message(f"❗ 주문 정정 요청 실패: 주문ID {order_id}")
            return None


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
        
        # [핵심 수정] 딕셔너리 형태의 가격 데이터에서 'close' 값을 추출합니다.
        price_data = current_prices.get(stock_code)
        if not pos_info or not price_data or pos_info.get('quantity', 0) <= 0:
            return False
        
        current_price = price_data.get('close') # 'close' 키로 현재가를 가져옵니다.
        if current_price is None:
            logger.warning(f"[{stock_code}] 가격 데이터에 'close' 값이 없습니다.")
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
        
        total_cost = sum(p['quantity'] * p['avg_price'] for p in positions.values())
        if total_cost == 0: return False
        
        # [핵심 수정] 포트폴리오 가치 계산 로직 수정
        total_current_value = 0
        for code, p in positions.items():
            price_data = current_prices.get(code)
            # 실시간 가격이 있으면 'close' 값을 사용하고, 없으면 기존 평균가를 사용
            price_to_use = price_data['close'] if price_data and 'close' in price_data else p['avg_price']
            total_current_value += p['quantity'] * price_to_use
        
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
            price_data = current_prices.get(code)
            if price_data and 'close' in price_data:
                current_price = price_data['close'] # 실제 현재가(float)를 가져옴
                # avg_price가 0인 경우 ZeroDivisionError 방지
                if pos['avg_price'] > 0 and ((current_price - pos['avg_price']) / pos['avg_price']) * 100 <= stop_loss_pct:
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







