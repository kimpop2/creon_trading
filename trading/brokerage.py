# trading/brokerage.py

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, date, time, timedelta

from trading.abstract_broker import AbstractBroker
from api.creon_api import CreonAPIClient, OrderType
from manager.db_manager import DBManager # DBManager 직접 사용
from manager.trading_manager import TradingManager # TradingManager에 필요한 데이터 저장 로직이 있으므로 주입
from util.notifier import Notifier # 알림 기능 (텔레그램 등)
from config.settings import LIVE_HMM_MODEL_NAME
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

    def execute_order(self, stock_code: str, order_type: str, price: float, quantity: int,
                      order_time: datetime, order_id: Optional[str] = None,
                      strategy_name: str = "Unknown") -> Optional[str]:
        """ 주문을 실행하고, 성공 시 주문 ID를, 실패 시 None을 반환합니다. """
        order_type_enum = OrderType.BUY if order_type.lower() == 'buy' else OrderType.SELL
        # 시장가 주문을 위해 가격이 0이면 주문 단위를 "03"으로 설정
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
                'order_id': order_id,  # _active_orders[order_id] 를 구분하지만, 내부에도 있어야 함
                'stock_code': stock_code,
                'stock_name': self.manager.get_stock_name(stock_code),
                'strategy_name': strategy_name, # <-- 전략 이름 저장
                'order_type': order_type.lower(),
                'order_status': '접수',
                'order_price': price,
                'order_quantity': quantity,
                'filled_quantity': 0,
                'unfilled_quantity': quantity,
                'order_time': order_time,
                'original_order_id': None
            }
            return order_id
            
        else:
            # --- [핵심 수정] 주문 실패 시, 에러 메시지를 확인하여 상태 자동 교정 ---
            error_message = result.get('message', '')
            if '13036' in error_message or '주문가능수량' in error_message:
                logger.critical(f"상태 불일치 감지! [{stock_code}]의 실제 잔고가 0입니다. 내부 포지션을 강제 동기화(제거)합니다.")
                # 내부 포지션 목록에서 해당 '유령 포지션'을 즉시 제거
                if stock_code in self.positions:
                    del self.positions[stock_code]
            # --- 수정 끝 ---

            stock_name = self.manager.get_stock_name(stock_code)
            self.notifier.send_message(f"❌ 주문 실패: {stock_name}({stock_code})")
            logger.error(f"주문 실패: {stock_code}, 메시지: {error_message}")
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

        holdings_value = 0
        for code, pos in self.positions.items():
            price_data = current_prices.get(code)
            # 실시간 가격이 있으면 'close' 값을 사용하고, 없으면 기존 평균가를 사용
            price_to_use = price_data['close'] if price_data and 'close' in price_data else pos.get('avg_price', 0)
            holdings_value += pos.get('quantity', 0) * price_to_use
        
        return holdings_value   
    
    def get_unfilled_stock_codes(self) -> set:
        """미체결 상태인 주문들의 종목 코드 집합을 반환합니다."""
        return {order['stock_code'] for order in self._active_orders.values()}
        
    def handle_order_conclusion(self, conclusion_data: Dict[str, Any]):
        """
        [최종 수정] 체결 시 (1)메모리 상태를 먼저 업데이트하고 (2)그 결과를 DB에 동기화합니다.
        """
        logger.info(f"체결/주문응답 수신: {conclusion_data}")
        order_status_str = conclusion_data.get('order_status')
        order_id = conclusion_data.get('order_id')
        stock_code = conclusion_data.get('stock_code')
        
        active_order = self._active_orders.get(order_id)
        if not active_order:
            logger.warning(f"활성 주문 목록에 없는 주문 응답 수신: {order_id}")
            return

        active_order['order_status'] = order_status_str.lower()
        
        if order_status_str in ['체결', '부분체결']:
            filled_quantity = conclusion_data.get('quantity', 0)
            filled_price = conclusion_data.get('price', 0)
            order_type = active_order['order_type']

            if filled_quantity > 0:
                # --- (공통) 현금 잔고 및 누적 체결 수량 업데이트 ---
                active_order['filled_quantity'] += filled_quantity
                active_order['unfilled_quantity'] -= filled_quantity
                
                transaction_amount = filled_price * filled_quantity
                commission = transaction_amount * self.commission_rate
                tax = transaction_amount * self.tax_rate_sell if order_type == 'sell' else 0
                net_amount = (transaction_amount - commission - tax) if order_type == 'sell' else -(transaction_amount + commission)
                self._current_cash_balance += net_amount
                
                # --- ▼▼▼ [핵심] 1. 메모리(self.positions) 상태 업데이트 로직 추가 ▼▼▼ ---
                realized_profit_loss = 0
                if order_type == 'buy':
                    if stock_code in self.positions: # 기존 보유 종목 추가 매수 (물타기)
                        pos = self.positions[stock_code]
                        total_cost = (pos['avg_price'] * pos['quantity']) + (filled_price * filled_quantity)
                        pos['quantity'] += filled_quantity
                        pos['avg_price'] = total_cost / pos['quantity']
                    else: # 신규 종목 매수
                        self.positions[stock_code] = {
                            'stock_code': stock_code,
                            'stock_name': active_order.get('stock_name'),
                            'quantity': filled_quantity,
                            'avg_price': filled_price,
                            'entry_date': datetime.now().date(),
                            'strategy_name': active_order.get('strategy_name'),
                            'highest_price': filled_price
                        }
                elif order_type == 'sell':
                    if stock_code in self.positions:
                        pos = self.positions[stock_code]
                        realized_profit_loss = (filled_price - pos['avg_price']) * filled_quantity
                        pos['quantity'] -= filled_quantity
                        if pos['quantity'] <= 0:
                            del self.positions[stock_code]
                # --- ▲▲▲ 메모리 업데이트 완료 ▲▲▲ ---

                # --- ▼▼▼ 2. DB 상태 동기화 (기존 로직 보강) ▼▼▼ ---
                if order_type == 'buy' and stock_code in self.positions:
                    self.manager.save_current_position(self.positions[stock_code])
                elif order_type == 'sell':
                    if stock_code in self.positions: # 부분 매도
                        self.manager.save_current_position(self.positions[stock_code])
                    else: # 전량 매도
                        self.manager.db_manager.delete_current_position(stock_code)
                # --- ▲▲▲ DB 동기화 완료 ▲▲▲ ---

                # --- 3. 거래 이력(trading_trade) 저장 ---
                model_info = self.manager.db_manager.fetch_hmm_model_by_name(LIVE_HMM_MODEL_NAME)
                model_id = model_info['model_id'] if model_info else 1 # Fallback model_id
                
                trade_data_to_save = {
                    'model_id': model_id, 'trade_date': datetime.now().date(),
                    'strategy_name': active_order.get('strategy_name'), 'stock_code': stock_code,
                    'trade_type': order_type.upper(), 'trade_price': filled_price,
                    'trade_quantity': filled_quantity, 'trade_datetime': datetime.now(),
                    'commission': commission, 'tax': tax, 'realized_profit_loss': realized_profit_loss
                }
                self.manager.save_trading_trade(trade_data_to_save)

        # 주문 최종 완료 시 활성 주문 목록에서 제거
        if active_order.get('unfilled_quantity', 1) <= 0 or order_status_str in ['체결', '취소', '거부']:
            logger.info(f"주문({order_id}) 최종 완료({order_status_str}). 활성 주문 목록에서 제거합니다.")
            if order_id in self._active_orders:
                del self._active_orders[order_id]


    def sync_account_status(self):
        """
        [전면 수정] API와 DB 정보를 조합하여 '완전한' 포지션 상태를 재구성하고,
        이를 메모리와 DB 양쪽에 모두 동기화하는 마스터 메서드입니다.
        """
        logger.info("계좌 상태 마스터 동기화 시작 (API + DB)...")

        # 1. API와 DB의 현재 종목 코드 리스트를 각각 가져옵니다.
        positions_from_api = {pos['stock_code']: pos for pos in self.api_client.get_portfolio_positions()}
        api_stock_codes = set(positions_from_api.keys())
        
        positions_from_db = self.manager.db_manager.fetch_current_positions()
        db_stock_codes = {pos['stock_code'] for pos in positions_from_db}

        # 2. DB에는 있지만 API에는 없는 종목(유령 포지션)을 찾습니다.
        codes_to_delete = db_stock_codes - api_stock_codes
        if codes_to_delete:
            logger.info(f"API 잔고에 없는 {len(codes_to_delete)}개의 유령 포지션을 DB에서 삭제합니다: {codes_to_delete}")
            for code in codes_to_delete:
                self.manager.db_manager.delete_current_position(code)

        # 3. API 잔고를 기준으로, DB 정보를 병합하여 최종 포지션을 재구성합니다.
        #    (positions_from_db 대신 이미 조회한 positions_from_db_state 사용)
        positions_from_db_state = {pos['stock_code']: pos for pos in positions_from_db}
        synced_positions = {}
        for stock_code, api_pos in positions_from_api.items():
            last_buy_info = self.manager.db_manager.fetch_last_buy_trade_for_stock(stock_code)
            db_state = positions_from_db_state.get(stock_code, {})

            api_pos['strategy_name'] = last_buy_info.get('strategy_name', 'MANUAL') if last_buy_info else 'MANUAL'
            api_pos['entry_date'] = last_buy_info.get('trade_date') if last_buy_info else date.today()
            api_pos['highest_price'] = db_state.get('highest_price', api_pos.get('avg_price', 0))
            
            eval_pl = api_pos.get('eval_profit_loss', 0)
            avg_price = api_pos.get('avg_price', 0)
            quantity = api_pos.get('quantity', 0)
            
            if avg_price > 0 and quantity > 0:
                cost_basis = avg_price * quantity
                api_pos['eval_return_rate'] = (eval_pl / cost_basis) * 100
            else:
                api_pos['eval_return_rate'] = 0.0

            synced_positions[stock_code] = api_pos

        # 4. 메모리(self.positions)와 DB(current_positions 테이블)에 최종 상태를 업데이트합니다.
        self.positions = synced_positions
        if self.positions:
            logger.info(f"총 {len(self.positions)}건의 보유 종목 최종 상태 업데이트 및 DB 저장 시작...")
            for pos_data in self.positions.values():
                self.manager.save_current_position(pos_data) # save_current_position은 UPSERT 기능


        # 4. 현금 잔고 동기화
        balance_info = self.api_client.get_account_balance()
        self._current_cash_balance = balance_info.get('cash_balance', 0.0) if balance_info else 0.0
        #logger.info(f"현금 잔고 업데이트: {self._current_cash_balance:,.0f}원")

        # 5. 미체결 주문 동기화
        unfilled_orders_from_api = self.api_client.get_unfilled_orders()
        api_order_ids = {order.get('order_id') for order in unfilled_orders_from_api}

        # 현재 시스템 내부에 기록된 미체결 주문 목록
        internal_order_ids = set(self._active_orders.keys())
        # 6. 내부에 있지만 API에는 없는 주문 제거 (사용자가 취소한 경우)
        orders_to_remove = internal_order_ids - api_order_ids
        for order_id in orders_to_remove:
            stock_code = self._active_orders[order_id].get('stock_code', 'N/A')
            del self._active_orders[order_id]
            logger.warning(f"유령 주문 제거: API에 존재하지 않는 미체결 주문({order_id}, {stock_code})을 내부 목록에서 삭제합니다.")

        # 7 API에는 있지만 내부에 없는 주문
        orders_to_add = api_order_ids - internal_order_ids
        # API 응답을 기반으로 주문 정보를 구성 (기존 로직과 유사)
        api_orders_map = {order.get('order_id'): order for order in unfilled_orders_from_api}
        for order_id in orders_to_add:
            order_info = api_orders_map[order_id]
            self._active_orders[order_id] = {
                'order_id': order_id,  # _active_orders[order_id] 를 구분하지만, 내부에도 있어야 함
                'stock_code': order_info.get('stock_code'),
                'order_type': 'buy' if order_info.get('buy_sell') == '매수' else 'sell',
                'order_status': 'submitted', # API 미체결은 '접수' 상태로 간주
                'order_price': order_info.get('price'),
                'order_quantity': order_info.get('quantity'),
                'filled_quantity': order_info.get('filled_quantity', 0),
                'unfilled_quantity': order_info.get('unfilled_quantity'),
                'order_time': order_info.get('time'),
                'original_order_id': None
            }
            logger.info(f"미체결 주문 복원: 주문번호 {order_id}")
            
        logger.info(f"미체결 주문 동기화 완료. 현재 활성 주문: {len(self._active_orders)}건")
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
            origin_order_id=order_id
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
        if profit_pct >= self.stop_loss_params.get('take_profit_pct', float('inf')):
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
        if profit_pct <= self.stop_loss_params.get('stop_loss_pct', -float('inf')):
            logging.info(f"[손절] {stock_code}")
            self.notifier.send_message(f"🚨 손절: {stock_code} ({profit_pct:.2%})")
            return self.execute_order(stock_code, 'sell', current_price, pos_info['quantity'], current_dt)
        # 트레일링 스탑
        if highest_price > 0:
            trailing_stop_pct = (current_price - highest_price) * 100 / highest_price
            if trailing_stop_pct <= self.stop_loss_params.get('trailing_stop_pct', -float('inf')):
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
        stop_loss_pct = self.stop_loss_params.get('stop_loss_pct', -float('inf'))
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







