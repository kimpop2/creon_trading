# trading/brokerage.py

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, date, time, timedelta

from trading.abstract_broker import AbstractBroker
from api.creon_api2 import CreonAPIClient, OrderType
from manager.db_manager import DBManager # DBManager 직접 사용
from manager.trading_manager import TradingManager # TradingManager에 필요한 데이터 저장 로직이 있으므로 주입
from util.notifier import Notifier # 알림 기능 (텔레그램 등)

# --- 로거 설정 ---
logger = logging.getLogger(__name__)

class Brokerage(AbstractBroker):
    """
    실제 증권사 API (Creon)를 통해 매매를 실행하는 브로커 구현체입니다.
    AbstractBroker를 상속받아 실제 주문, 잔고 조회, 포지션 관리를 수행합니다.
    """
    def __init__(self, 
                 api_client: CreonAPIClient, 
                 manager: TradingManager, 
                 notifier: Notifier, 
                 initial_cash:float=10_000_000):
        
        super().__init__()
        self.api_client = api_client
        self.manager = manager
        self.notifier = notifier # 알림 객체 주입
        self.initial_cash=initial_cash
        self.commission_rate = 0.00165 # 매도시에만 부과
        self.stop_loss_params = None
        self.initial_portfolio_value = initial_cash # 포트폴리오 손절을 위한 초기값
        # 현재 포지션 및 잔고는 API를 통해 실시간으로 조회하고 DB에 동기화.
        # 내부적으로 캐시할 수도 있으나, 항상 최신 정보는 API에서 가져오는 것을 우선.
        self.positions = {}  # {stock_code: {'size': int, 'avg_price': float, 'entry_date': datetime.date, 'highest_price': float}}
        #self.transaction_log = [] # (date, stock_code, type, price, quantity, commission, net_amount)

        self._current_cash_balance: float = 0.0
        self._current_positions: Dict[str, Any] = {} # {stock_code: {...}}
        self._unfilled_orders: List[Dict[str, Any]] = []

        # --- CreonAPIClient에 콜백 함수 등록 ---
        self.api_client.set_conclusion_callback(self.handle_order_conclusion)
        # 주문 요청 응답 콜백도 등록 (필요시)
        # self.api_client.set_order_reply_callback(self.handle_order_reply)

        logging.info(f"브로커 초기화: 초기 현금 {self.initial_cash:,.0f}원, 수수료율 {self.commission_rate*100:.2f}%")
        self.sync_account_status() # 초기 계좌 상태 동기화
    
    def set_stop_loss_params(self, stop_loss_params):
        """손절매 관련 파라미터를 설정합니다."""
        if stop_loss_params is None:
            return
        self.stop_loss_params = stop_loss_params
        logging.info(f"브로커 손절매 파라미터 설정 완료: {stop_loss_params}")

    def get_position_size(self, stock_code: str) -> int:
        """
        특정 종목의 현재 보유 수량을 반환합니다.
        실시간으로 API를 통해 조회하는 self._current_positions를 참조합니다.
        """
        position = self._current_positions.get(stock_code)
        if position:
            return position.get('quantity', 0)
        return 0
            
    def execute_order(self,
                      stock_code: str,
                      order_type: str, # 'buy', 'sell'
                      price: float, # 지정가 또는 시장가 구분 필요 (Creon API)
                      quantity: int,
                      order_time: datetime,
                      order_id: Optional[str] = None # 주문 정정/취소 시 사용될 수 있는 원주문번호
                     ) -> Optional[str]:
        """
        실제 주문을 Creon API를 통해 실행하고, 주문 로그를 DB에 저장합니다.
        """
        stock_name = self.manager.get_stock_name(stock_code)
        # OrderType Enum 변환
        order_type_enum = OrderType.BUY if order_type.lower() == 'buy' else OrderType.SELL
        # 시장가/지정가 구분
        order_unit = "03" if price == 0 or price is None else "01"
        price_to_send = 0 if order_unit == "03" else int(price)
        result = self.api_client.send_order(
            stock_code=stock_code,
            order_type=order_type_enum,
            quantity=quantity,
            price=price_to_send,
            order_unit=order_unit
        )
        if result and result['status'] == 'success':
            order_num = result['order_num']
            # order_type 영문 변환
            order_type_for_log = 'buy' if order_type.lower() in ['buy', '매수'] else 'sell'
            log_data = {
                'order_id': order_num,
                'original_order_id': order_num,
                'stock_code': stock_code,
                'stock_name': stock_name,
                'trading_date': order_time.date(),
                'trading_time': order_time.time(),
                'order_type': order_type_for_log,
                'order_price': price,
                'order_quantity': quantity,
                'filled_price': 0,
                'filled_quantity': 0,
                'unfilled_quantity': quantity,
                'order_status': '접수',
                'commission': 0,
                'tax': 0,
                'net_amount': 0,
                'credit_type': '현금'
            }
            self.manager.save_trading_log(log_data)
            self.notifier.send_message(f"✅ 주문 접수: {stock_name}({stock_code}) {order_type.upper()} {quantity}주 (가격: {price:,.0f}원, 주문ID: {order_num})")
            logger.info(f"주문 성공: {stock_code}, 주문번호: {order_num}")
            return order_num
        else:
            self.notifier.send_message(f"❌ 주문 실패: {stock_name}({stock_code}) {order_type.upper()} {quantity}주")
            logger.error(f"주문 실패: {stock_code}")
            return None



    def check_and_execute_stop_loss(self, current_prices: Dict[str, float], current_dt: datetime) -> bool:
        """
        설정된 손절매/익절매 조건을 확인하고 해당되는 경우 매도 주문을 실행합니다.
        (backtest.Broker의 로직을 Real Brokerage에 맞게 수정)
        """
        # stop_loss_params는 TradingStrategy에서 설정되므로, TradingStrategy에서 전달받거나
        # TradingManager를 통해 관리하는 것이 더 적합할 수 있습니다.
        # 여기서는 임시로 매번 최신 포지션 정보를 가져와서 계산합니다.
        positions = self.get_current_positions()
        executed_any_stop_loss = False

        for stock_code, pos_info in positions.items():
            quantity = pos_info.get('quantity', 0)
            avg_price = pos_info.get('average_buy_price', 0)
            entry_date = pos_info.get('entry_date', date.today())
            current_price = current_prices.get(stock_code)

            if quantity <= 0 or current_price is None or current_price == 0:
                continue

            # TODO: 손절매 파라미터는 TradingStrategy (또는 Trading 클래스)에서 관리하고,
            # 이곳으로 전달되어야 합니다. 현재는 하드코딩된 예시.
            # 실제 파라미터는 `strategy_params`나 전역 설정에서 로드될 것입니다.
            stop_loss_params = {
                'take_profit_ratio': 0.20,       # 20% 익절
                'early_stop_loss': -0.05,        # 매수 후 초기 손실 제한: -5% (예: 매수 후 3일 이내)
                'stop_loss_ratio': -0.10,        # 매수가 기준 손절율: -10%
                'trailing_stop_ratio': -0.07,    # 최고가 기준 트레일링 손절률: -7%
            }

            profit_loss_ratio = (current_price - avg_price) / avg_price if avg_price != 0 else 0

            # 익절 조건 (take_profit_ratio)
            if profit_loss_ratio >= stop_loss_params['take_profit_ratio']:
                logger.info(f"[익절] {stock_code} - {profit_loss_ratio:.2%} 수익, 매도.")
                self.execute_order(stock_code, 'sell', current_price, quantity, current_dt)
                executed_any_stop_loss = True
                self.notifier.send_message(f"💰 익절: {stock_code} {quantity}주 ({profit_loss_ratio:.2%})")
                continue

            # 초기 손실 제한 (early_stop_loss) - 예: 매수 후 3일 이내
            holding_days = (current_dt.date() - entry_date).days
            if holding_days <= 3 and profit_loss_ratio <= stop_loss_params['early_stop_loss']:
                logger.info(f"[초기 손절] {stock_code} - {profit_loss_ratio:.2%} 손실, 매도.")
                self.execute_order(stock_code, 'sell', current_price, quantity, current_dt)
                executed_any_stop_loss = True
                self.notifier.send_message(f"📉 초기 손절: {stock_code} {quantity}주 ({profit_loss_ratio:.2%})")
                continue

            # 일반 손절 조건 (stop_loss_ratio)
            if profit_loss_ratio <= stop_loss_params['stop_loss_ratio']:
                logger.info(f"[손절] {stock_code} - {profit_loss_ratio:.2%} 손실, 매도.")
                self.execute_order(stock_code, 'sell', current_price, quantity, current_dt)
                executed_any_stop_loss = True
                self.notifier.send_message(f"🚨 손절: {stock_code} {quantity}주 ({profit_loss_ratio:.2%})")
                continue

            # 트레일링 스탑 (trailing_stop_ratio)
            # 최고가 정보는 `current_positions`에 `highest_price`로 저장되어야 함.
            avg_price = pos_info.get('avg_price', 0) # 💡 'average_buy_price' -> 'avg_price'
            highest_price = pos_info.get('highest_price', avg_price)
            if current_price > highest_price: # 현재가가 최고가보다 높으면 갱신
                logger.debug(f"[{stock_code}] 최고가 갱신: {highest_price} -> {current_price}")
                pos_info['highest_price'] = current_price
                
                # 💡 [수정] DB 저장을 위해 올바른 키 이름('avg_price')으로 값을 가져옴
                data_to_save = {
                    'stock_code': stock_code,
                    'stock_name': pos_info.get('stock_name'),
                    'quantity': pos_info.get('quantity'),
                    'sell_avail_qty': pos_info.get('sell_avail_qty'),
                    'average_buy_price': pos_info.get('avg_price'), # 'average_buy_price' -> 'avg_price'
                    'eval_profit_loss': pos_info.get('eval_profit_loss'),
                    'eval_return_rate': pos_info.get('eval_return_rate'),
                    'entry_date': pos_info.get('entry_date'),
                    'highest_price': current_price
                }
                self.manager.save_current_position(data_to_save)
                
            elif current_price < highest_price * (1 + stop_loss_params['trailing_stop_ratio']): # 트레일링 손절 조건
                logger.info(f"[트레일링 스탑] {stock_code} - 최고가 대비 하락, 매도.")
                self.execute_order(stock_code, 'sell', current_price, quantity, current_dt)
                executed_any_stop_loss = True
                self.notifier.send_message(f"🛑 트레일링 스탑: {stock_code} {quantity}주")
                continue


        # TODO: 포트폴리오 전체 손절매 로직 추가
        # (check_and_execute_stop_loss 내부에서 _check_portfolio_stop_loss_conditions 호출)
        # 이 부분은 Backtest Broker의 로직을 참고하여 구현.
        # 이 함수는 Brokerage 내부에서 전체 포트폴리오를 평가하여 청산 결정을 내릴 수 있습니다.
        # 예시:
        # if self._check_portfolio_stop_loss_conditions(current_prices, current_dt):
        #     self._execute_portfolio_sellout(current_prices, current_dt)
        #     executed_any_stop_loss = True

        return executed_any_stop_loss
    
    def execute_time_cut_sell(self,
                              stock_code: str,
                              current_price: float,
                              current_position_size: int,
                              current_dt: datetime,
                              max_price_diff_ratio: float = 0.01
                              ) -> bool:
        """
        타임컷 강제 매도 로직을 실행합니다.
        이 함수는 `RSIMinute` 등의 분봉 전략에서 사용될 수 있습니다.
        """
        # 현재 보유하고 있는 해당 종목에 대한 매수 신호가 있는지 확인 (아직 체결되지 않은 신호)
        if stock_code not in self.signals or self.signals[stock_code]['signal_type'] != 'BUY':
            logger.debug(f"[타임컷] {stock_code}: 매수 신호가 없거나 다른 신호임. 타임컷 건너뜀.")
            return False

        # 해당 매수 신호의 목표가
        target_price = self.signals[stock_code].get('target_price')

        if target_price is None or target_price <= 0:
            logger.warning(f"[타임컷] {stock_code}: 유효한 목표 가격이 없습니다. 타임컷 매도 건너뜀.")
            return False

        # 목표가와 현재가 간의 괴리율 계산
        price_diff_ratio = abs(target_price - current_price) / target_price

        if price_diff_ratio <= max_price_diff_ratio:
            logger.info(f'[타임컷 강제매도] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:,.0f}, 현재가: {current_price:,.0f}, 괴리율: {price_diff_ratio:.2%}. 매도 실행.')
            # broker 대신 brokerage 사용
            self.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
            self.reset_signal(stock_code) # 신호 처리 완료
            return True
        else:
            logger.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:,.0f}, 현재가: {current_price:,.0f}, 괴리율: {price_diff_ratio:.2%} ({max_price_diff_ratio:.1%} 초과).')
            return False



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

    def get_current_cash_balance(self) -> float:
        """
        실시간 계좌 현금 잔고를 조회하고 반환합니다.
        내부 캐시를 반환하며, 이 캐시는 sync_account_status()를 통해 주기적으로 업데이트됩니다.
        """
        return self._current_cash_balance

    def get_current_positions(self) -> Dict[str, Any]:
        """
        실시간 보유 종목 정보를 조회하고 반환합니다.
        내부 캐시를 반환하며, 이 캐시는 sync_account_status()를 통해 주기적으로 업데이트됩니다.
        """
        return self._current_positions

    def get_unfilled_orders(self) -> List[Dict[str, Any]]:
        """
        실시간 미체결 주문 내역을 조회하고 반환합니다.
        내부 캐시를 반환하며, 이 캐시는 sync_account_status()를 통해 주기적으로 업데이트됩니다.
        """
        return self._unfilled_orders

    def update_portfolio_status(self, current_dt: datetime, current_prices: Dict[str, float]):
        """
        현재 시점의 포트폴리오 상태를 계산하고 DB에 저장합니다.
        이 함수는 주로 장 마감 후 또는 일일 결산 시 호출됩니다.

        Args:
            current_dt (datetime): 현재 시각
            current_prices (Dict[str, float]): 보유 종목들의 현재가 딕셔너리
        """
        logger.info(f"{current_dt.date()} 포트폴리오 상태 업데이트 시작...")

        # 1. 최신 계좌 정보로 동기화
        self.sync_account_status()
        
        cash_balance = self.get_current_cash_balance()
        current_positions = self.get_current_positions()

        # 2. 보유 주식의 총 평가액 계산
        total_asset_value = 0.0
        for stock_code, pos_info in current_positions.items():
            quantity = pos_info.get('quantity', 0)
            current_price = current_prices.get(stock_code, pos_info.get('avg_price', 0)) # 현재가가 없으면 매입가로 계산
            total_asset_value += quantity * current_price

        # 3. 총자산(현금 + 주식 평가액) 계산
        total_capital = cash_balance + total_asset_value

        # 4. DB에서 직전 거래일의 포트폴리오 정보를 가져옴
        latest_portfolio = self.manager.load_latest_daily_portfolio()
        prev_day_capital = latest_portfolio.get('total_capital', self.initial_cash)
        prev_cumulative_profit_loss = latest_portfolio.get('cumulative_profit_loss', 0)

        # 5. 일일 손익 및 누적 손익 계산
        daily_profit_loss = total_capital - prev_day_capital
        daily_return_rate = (daily_profit_loss / prev_day_capital) * 100 if prev_day_capital != 0 else 0
        
        cumulative_profit_loss = (total_capital - self.initial_cash)
        cumulative_return_rate = (cumulative_profit_loss / self.initial_cash) * 100 if self.initial_cash != 0 else 0

        # 6. 데이터베이스에 저장할 데이터 준비
        portfolio_data = {
            'record_date': current_dt.date(),
            'total_capital': total_capital,
            'cash_balance': cash_balance,
            'total_asset_value': total_asset_value,
            'daily_profit_loss': daily_profit_loss,
            'daily_return_rate': daily_return_rate,
            'cumulative_profit_loss': cumulative_profit_loss,
            'cumulative_return_rate': cumulative_return_rate,
            # 'max_drawdown' 등 다른 지표는 trading_manager에서 계산하거나 여기서 추가할 수 있습니다.
        }

        # 7. TradingManager를 통해 DB에 저장
        success = self.manager.save_daily_portfolio(portfolio_data)
        
        if success:
            logger.info(f"일일 포트폴리오 상태 DB 저장 성공: {current_dt.date()}")
            # 8. 알림 발송
            self.notifier.send_message(
                f"📊 일일 포트폴리오 결산 ({current_dt.date()})\n"
                f"총 자산: {total_capital:,.0f}원\n"
                f" - 현금: {cash_balance:,.0f}원\n"
                f" - 주식: {total_asset_value:,.0f}원\n"
                f"일일 손익: {daily_profit_loss:,.0f}원 ({daily_return_rate:+.2f}%)\n"
                f"누적 손익: {cumulative_profit_loss:,.0f}원 ({cumulative_return_rate:+.2f}%)"
            )
        else:
            logger.error(f"일일 포트폴리오 상태 DB 저장 실패: {current_dt.date()}")

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

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        현재 포트폴리오의 총 가치 (현금 + 주식 평가액)를 계산하여 반환합니다.
        
        Args:
            current_prices (Dict[str, float]): 종목 코드를 키, 현재가를 값으로 하는 딕셔너리.
                                               전략단에서 계산한 최신 가격 정보를 받습니다.
        
        Returns:
            float: 포트폴리오 총 평가액
        """
        # 1. 최신 계좌 정보 동기화 (정확한 계산을 위해)
        self.sync_account_status()
        
        # 2. 현재 현금 잔고 조회
        cash_balance = self.get_current_cash_balance()
        
        # 3. 보유 주식의 총 평가액 계산
        positions = self.get_current_positions()
        total_asset_value = 0.0
        for stock_code, pos_info in positions.items():
            quantity = pos_info.get('quantity', 0)
            # 전략에서 제공한 최신 가격 사용, 없으면 매입가로 대체 (안전장치)
            current_price = current_prices.get(stock_code, pos_info.get('average_buy_price', 0))
            total_asset_value += quantity * current_price
            
        # 4. 총 포트폴리오 가치(현금 + 주식 평가액) 반환
        return cash_balance + total_asset_value
    # def handle_order_reply(self, reply_data: Dict[str, Any]):
    #     """
    #     주문 요청 응답 (td0314) 이벤트를 처리하는 콜백 함수.
    #     주문 접수, 거부 등의 응답을 처리하고 trading_log를 업데이트합니다.
    #     """
    #     logger.info(f"주문 응답 수신: {reply_data}")
    #     order_id = reply_data.get('order_id')
    #     original_order_id = reply_data.get('original_order_id')
    #     stock_code = reply_data.get('stock_code')
    #     status = reply_data.get('status')
    #     message = reply_data.get('message')

    #     # trading_log에서 해당 order_id를 찾아 상태 업데이트
    #     # 또는 새로운 로그 엔트리 추가 (예: 거부된 주문)
    #     # 이 부분은 save_trading_log의 ON DUPLICATE KEY UPDATE를 활용할 수 있음
    #     log_data = {
    #         'order_id': order_id,
    #         'original_order_id': original_order_id,
    #         'stock_code': stock_code,
    #         'stock_name': self.manager.get_stock_name(stock_code),
    #         'trade_date': datetime.now().date(),
    #         'trade_time': datetime.now().time(),
    #         'order_type': 'unknown', # 주문 유형은 응답 데이터에서 더 정확히 파싱해야 함
    #         'order_price': 0,
    #         'order_quantity': reply_data.get('quantity', 0),
    #         'filled_price': 0,
    #         'filled_quantity': 0,
    #         'unfilled_quantity': reply_data.get('quantity', 0),
    #         'order_status': message, # 응답 메시지를 상태로 사용하거나, Creon 상태 코드 매핑
    #         'commission': 0,
    #         'tax': 0,
    #         'net_amount': 0,
    #         'credit_type': '현금'
    #     }
    #     self.manager.save_trading_log(log_data)
    #     self.sync_account_status() # 계좌 상태 동기화

