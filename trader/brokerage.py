# trade/brokerage.py

import datetime
import logging
from typing import Dict, Any
from trader.abstract_broker import AbstractBroker
from manager.business_manager import BusinessManager # BusinessManager 추가

logger = logging.getLogger(__name__)

class Brokerage(AbstractBroker):
    """
    실전 자동매매를 위한 증권사 브로커 클래스.
    AbstractBroker를 상속받아 실제 증권사 API 연동을 구현합니다.
    """
    def __init__(self, business_manager: BusinessManager, 
                 commission_rate: float = 0.0016, slippage_rate: float = 0.0004):
        self.api_client = business_manager.api_client
        self.business_manager = business_manager
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.stop_loss_params = None
        
        # 실전에서는 초기 현금과 포지션을 API에서 조회하여 동기화
        self.cash = 0.0
        self.positions = {}  # {stock_code: {'size': int, 'avg_price': float, 'entry_date': datetime.date, 'highest_price': float}}
        self.initial_portfolio_value = 0.0 # 포트폴리오 손절을 위한 초기값 (일일 초기화 필요)
        self.transaction_log = [] # (date, stock_code, type, price, quantity, commission, net_amount, order_id, trade_type)
        
        self.sync_account_info() # 초기화 시 계좌 정보 동기화
        logger.info(f"Brokerage 초기화 완료: 초기 현금 {self.cash:,.0f}원, 수수료율 {self.commission_rate*100:.2f}%")

    def sync_account_info(self):
        """
        증권사 API로부터 최신 현금 잔고와 보유 종목 정보를 동기화합니다.
        매매 시작 전, 그리고 주기적으로 호출하여 최신 정보를 반영합니다.
        """
        if not self.api_client.is_connected():
            logger.error("Creon API is not connected. Cannot sync account info.")
            return False

        # 현금 잔고 동기화
        balance = self.api_client.get_account_balance()
        if balance:
            self.cash = float(balance.get("cash", 0.0))
            logger.info(f"계좌 현금 잔고 동기화 완료: {self.cash:,.0f}원")
        else:
            logger.error("계좌 현금 잔고 동기화 실패.")
            return False

        # 보유 종목 동기화
        live_positions_list = self.api_client.get_portfolio_positions()
        new_positions = {}
        for pos in live_positions_list:
            stock_code = pos['stock_code']
            # 실전에서는 진입일, 최고가 등의 정보는 DB에서 로드하거나 별도 관리 필요.
            # 여기서는 API에서 제공하는 정보만으로 업데이트.
            # 백테스팅과 필드 통일을 위해 avg_price, size, entry_date, highest_price 키 사용
            # entry_date, highest_price는 DB나 BusinessManager에서 로드해야 함
            loaded_pos = self.business_manager.load_current_positions().get(stock_code, {})
            new_positions[stock_code] = {
                'size': int(pos['size']),
                'avg_price': float(pos['avg_price']),
                'entry_date': loaded_pos.get('entry_date', datetime.date.today()), # 임시 기본값, DB에서 로드 권장
                'highest_price': loaded_pos.get('highest_price', float(pos['avg_price'])) # 임시 기본값, DB에서 로드 권장
            }
        self.positions = new_positions
        self.business_manager.save_current_positions(self.positions) # 동기화된 포지션 DB 저장
        logger.info(f"보유 종목 정보 동기화 완료: 총 {len(self.positions)}개 종목 보유 중.")
        return True

    def execute_order(self, stock_code: str, order_type: str, price: float, quantity: int, current_dt: datetime, order_kind: str = '01'):
        """
        주문을 실행하고 거래 로그를 기록합니다.
        실전에서는 실제 증권사 API를 통해 주문을 전송합니다.
        price는 보통가 주문 시 사용, 시장가 주문 시에는 무시될 수 있습니다.
        """
        if not self.api_client.is_connected():
            logger.error("Creon API is not connected. Order execution failed.")
            return

        # 슬리피지 적용 (실전에서는 호가창 분석 후 실제 체결가 결정 로직 추가 가능)
        actual_price = price
        if order_type == 'buy':
            actual_price = price * (1 + self.slippage_rate)
        elif order_type == 'sell':
            actual_price = price * (1 - self.slippage_rate)
        
        # 수수료 계산
        commission = actual_price * quantity * self.commission_rate
        net_amount = (actual_price * quantity) + commission if order_type == 'buy' else (actual_price * quantity) - commission

        logger.info(f"[주문 요청] {current_dt.isoformat()} - {stock_code} {order_type.upper()} {quantity}주 @ {actual_price:,.0f}원 (원가: {price:,.0f}원)")
        
        order_id = self.api_client.send_order(stock_code, order_type, actual_price, quantity, order_kind)

        if order_id:
            logger.info(f"주문 성공: {stock_code}, 주문번호: {order_id}. 체결 대기 중...")
            
            # TODO: 실제 체결 여부 및 체결가 확인 로직 필요 (실시간 수신 또는 주기적 조회)
            # 지금은 주문 성공 시 바로 로그에 기록하지만, 실제 자동매매는 비동기 처리 또는 콜백으로 체결 확인
            
            # 현재는 주문 요청 성공 시 즉시 로그에 기록 (최소한의 구현)
            log_entry = {
                "date": current_dt.date(),
                "datetime": current_dt,
                "stock_code": stock_code,
                "type": order_type,
                "price": float(actual_price), # 체결가로 기록
                "quantity": int(quantity),
                "commission": float(commission),
                "net_amount": float(net_amount),
                "order_id": order_id,
                "trade_type": order_kind # '01' 보통, '03' 시장가 등
            }
            self.transaction_log.append(log_entry)
            self.business_manager.save_trade_log(log_entry) # DB에 로그 저장

            # 포지션 업데이트 (가정된 체결) - 실제 체결가/수량으로 업데이트 필요
            if order_type == 'buy':
                if stock_code in self.positions:
                    current_total_value = self.positions[stock_code]['size'] * self.positions[stock_code]['avg_price']
                    new_size = self.positions[stock_code]['size'] + quantity
                    new_avg_price = (current_total_value + (actual_price * quantity)) / new_size
                    self.positions[stock_code]['size'] = new_size
                    self.positions[stock_code]['avg_price'] = new_avg_price
                    self.positions[stock_code]['highest_price'] = max(self.positions[stock_code]['highest_price'], actual_price) # 매수 시 최고가 갱신
                else:
                    self.positions[stock_code] = {
                        'size': quantity,
                        'avg_price': actual_price,
                        'entry_date': current_dt.date(),
                        'highest_price': actual_price # 매수 시 최고가
                    }
                self.cash -= net_amount
            elif order_type == 'sell':
                if stock_code in self.positions:
                    self.cash += net_amount
                    self.positions[stock_code]['size'] -= quantity
                    if self.positions[stock_code]['size'] <= 0:
                        del self.positions[stock_code] # 전량 매도 시 포지션 삭제
                else:
                    logger.warning(f"보유하지 않은 종목 {stock_code}에 대한 매도 요청입니다.")
            
            self.business_manager.save_current_positions(self.positions) # DB에 현재 포지션 저장
            logger.info(f"거래 후 현금 잔고: {self.cash:,.0f}원")

        else:
            logger.error(f"주문 실패: {stock_code} {order_type} {quantity}주 @ {actual_price:,.0f}원")

    def get_position_size(self, stock_code: str) -> int:
        """현재 보유한 특정 종목의 수량을 반환합니다."""
        # 실전에서는 실시간으로 API에서 조회하는 것이 정확하지만, 빈번한 호출을 피하기 위해
        # 동기화된 self.positions를 활용하고 주기적으로 sync_account_info를 호출하는 방식.
        return self.positions.get(stock_code, {}).get('size', 0)

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """현재 포트폴리오의 총 가치를 계산하여 반환합니다."""
        total_value = self.cash
        for stock_code, position in self.positions.items():
            if stock_code in current_prices:
                total_value += position['size'] * current_prices[stock_code]
            else:
                logger.warning(f"현재 가격을 알 수 없는 종목 {stock_code}이 포트폴리오에 포함되어 있습니다. 계산에서 제외.")
        return total_value

    def set_stop_loss_params(self, stop_loss_params: Dict[str, Any]):
        """손절매 관련 파라미터를 설정합니다."""
        if stop_loss_params is None:
            return
        self.stop_loss_params = stop_loss_params
        logger.info(f"Brokerage 손절매 파라미터 설정 완료: {stop_loss_params}")

    def check_and_execute_stop_loss(self, stock_code: str, current_price: float, current_dt: datetime) -> bool:
        """
        개별 종목에 대한 손절매 조건을 확인하고, 조건 충족 시 손절매 주문을 실행합니다.
        (백테스팅의 Broker와 동일한 로직을 가지도록 구현)
        """
        if not self.stop_loss_params or stock_code not in self.positions:
            return False

        position = self.positions[stock_code]
        avg_price = position['avg_price']
        highest_price = position['highest_price']
        current_size = position['size']

        if current_size <= 0:
            return False

        loss_ratio = self._calculate_loss_ratio(current_price, avg_price)
        trailing_stop_loss_ratio = self.stop_loss_params.get('trailing_stop_loss_ratio')

        # 1. 고정 손절매
        if 'stop_loss_ratio' in self.stop_loss_params and loss_ratio <= self.stop_loss_params['stop_loss_ratio']:
            logger.warning(f"[개별 손절매] {current_dt.isoformat()} - {stock_code}: 손실율 {loss_ratio*100:.2f}% (기준: {self.stop_loss_params['stop_loss_ratio']*100:.2f}%)")
            self.execute_order(stock_code, 'sell', current_price, current_size, current_dt, order_kind='03') # 시장가 매도
            return True

        # 2. 트레일링 스탑
        if trailing_stop_loss_ratio and highest_price > 0:
            # 최고가 대비 현재가 하락률
            trailing_loss_from_peak = (highest_price - current_price) / highest_price
            if trailing_loss_from_peak >= trailing_stop_loss_ratio:
                logger.warning(f"[트레일링 스탑] {current_dt.isoformat()} - {stock_code}: 최고가({highest_price:,.0f}원) 대비 {trailing_loss_from_peak*100:.2f}% 하락 (기준: {trailing_stop_loss_ratio*100:.2f}%)")
                self.execute_order(stock_code, 'sell', current_price, current_size, current_dt, order_kind='03') # 시장가 매도
                return True
            else:
                # 현재가가 최고가보다 높으면 최고가 갱신
                self.positions[stock_code]['highest_price'] = max(highest_price, current_price)
        return False

    def _calculate_loss_ratio(self, current_price: float, avg_price: float) -> float:
        """손실율을 계산합니다 (양수: 이득, 음수: 손실)."""
        if avg_price == 0:
            return 0.0 # 매입 단가가 0일 경우 (예: 무상증자 등으로 취득)
        return (current_price - avg_price) / avg_price

    def check_and_execute_portfolio_stop_loss(self, current_prices: Dict[str, float], current_dt: datetime) -> bool:
        """
        포트폴리오 전체 손절매 조건을 확인하고, 조건 충족 시 전체 청산을 실행합니다.
        (백테스팅의 Broker와 동일한 로직을 가지도록 구현)
        """
        if not self.stop_loss_params or not self.stop_loss_params.get('max_losing_positions'):
            return False

        losing_positions_count = 0
        for stock_code, position in self.positions.items():
            if position['size'] > 0 and stock_code in current_prices:
                loss_ratio = self._calculate_loss_ratio(current_prices[stock_code], position['avg_price'])
                if loss_ratio <= self.stop_loss_params['stop_loss_ratio']: # 포트폴리오 손절 시 개별 손절율 기준 사용
                    losing_positions_count += 1

        if losing_positions_count >= self.stop_loss_params['max_losing_positions']:
            logger.warning(f'[포트폴리오 손절] {current_dt.isoformat()} - 손실 종목 수: {losing_positions_count}개 (기준: {self.stop_loss_params["max_losing_positions"]}개)')
            self._execute_portfolio_sellout(current_prices, current_dt)
            return True
        return False

    def _execute_portfolio_sellout(self, current_prices: Dict[str, float], current_dt: datetime):
        """포트폴리오 전체 청산을 실행합니다."""
        logger.info(f"포트폴리오 전체 청산 실행: {current_dt.isoformat()}")
        stocks_to_sell = list(self.positions.keys()) # 현재 보유 종목 리스트 복사
        for stock_code in stocks_to_sell:
            if stock_code in self.positions and self.positions[stock_code]['size'] > 0:
                quantity_to_sell = self.positions[stock_code]['size']
                current_price = current_prices.get(stock_code)
                if current_price:
                    self.execute_order(stock_code, 'sell', current_price, quantity_to_sell, current_dt, order_kind='03') # 시장가 매도
                else:
                    logger.error(f"청산 실패: {stock_code}의 현재가를 알 수 없어 매도할 수 없습니다.")
        logger.info("포트폴리오 전체 청산 주문 완료.")

    def reset_daily_transactions(self):
        """일일 거래 초기화를 수행합니다. (실전에서는 필요에 따라 구현)"""
        # 실전에서는 transaction_log를 매일 초기화할 필요는 없고, DB에 계속 쌓는 것이 일반적.
        # 여기서는 백테스팅과의 일관성을 위해 메서드만 유지.
        self.transaction_log = []
        logger.debug("일일 거래 로그 초기화 완료.")