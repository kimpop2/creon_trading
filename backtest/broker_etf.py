# backtest/broker_etf.py

import logging
import pandas as pd
from datetime import datetime, date # datetime.date 임포트 추가

class Broker:
    def __init__(self, initial_cash, commission_rate=0.0003, slippage_rate=0.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {} # {stock_code: {'size': quantity, 'avg_price': avg_price, 'entry_date': datetime.date}}
        self.transaction_log = [] # 매매 기록
        self.commission_rate = commission_rate
        
        self.stop_loss_params = None # 손절매 파라미터 초기화 (없으면 AttributeError 발생)
        logging.info(f"브로커 초기화: 초기 현금 = {self.initial_cash:,.0f}원")

    def set_stop_loss_params(self, params):
        """손절매 파라미터를 설정합니다."""
        self.stop_loss_params = params
        if params:
            logging.info(f"브로커 손절매 파라미터 설정됨: {params}")
        else:
            logging.info("브로커 손절매 기능 비활성화.")

    def execute_order(self, stock_code, order_type, quantity, price, current_dt):
        """실제 주문을 실행하고 현금 및 포지션을 업데이트합니다."""
        if quantity <= 0:
            return

        commission = price * quantity * self.commission_rate
        
        # current_dt가 datetime.datetime 객체일 수도 있으므로, .date()를 사용하여 날짜만 추출
        entry_date_for_log = current_dt.date() if isinstance(current_dt, datetime) else current_dt

        if order_type == 'buy':
            total_cost = (price * quantity) + commission
            if self.cash >= total_cost:
                self.cash -= total_cost
                if stock_code not in self.positions:
                    self.positions[stock_code] = {
                        'size': 0, 
                        'avg_price': 0, 
                        'entry_date': entry_date_for_log # <--- 수정: entry_date_for_log 사용
                    }
                
                # 가중 평균 가격 계산
                current_size = self.positions[stock_code]['size']
                current_value = current_size * self.positions[stock_code]['avg_price']
                
                new_size = current_size + quantity
                new_avg_price = (current_value + (price * quantity)) / new_size
                
                self.positions[stock_code]['size'] = new_size
                self.positions[stock_code]['avg_price'] = new_avg_price
                
                self.transaction_log.append({
                    'date': current_dt, 'stock_code': stock_code, 'type': 'buy',
                    'quantity': quantity, 'price': price, 'commission': commission,
                    'cash_after': self.cash
                })
                logging.info(f"[{current_dt.isoformat()}] 매수: {stock_code}, 수량: {quantity}, 가격: {price:,.0f}, 수수료: {commission:,.0f}, 남은 현금: {self.cash:,.0f}")
            else:
                logging.warning(f"[{current_dt.isoformat()}] 현금 부족으로 매수 불가: {stock_code}, 수량: {quantity}, 현재 현금: {self.cash:,.0f}, 필요 현금: {total_cost:,.0f}")
        
        elif order_type == 'sell':
            if stock_code in self.positions and self.positions[stock_code]['size'] >= quantity:
                sales_proceeds = (price * quantity) - commission
                tax = price * quantity * self.tax_rate # 매도 시 세금 부과
                sales_proceeds -= tax

                self.cash += sales_proceeds
                self.positions[stock_code]['size'] -= quantity
                
                if self.positions[stock_code]['size'] == 0:
                    del self.positions[stock_code] # 전량 매도 시 포지션 제거
                
                self.transaction_log.append({
                    'date': current_dt, 'stock_code': stock_code, 'type': 'sell',
                    'quantity': quantity, 'price': price, 'commission': commission, 'tax': tax,
                    'cash_after': self.cash
                })
                logging.info(f"[{current_dt.isoformat()}] 매도: {stock_code}, 수량: {quantity}, 가격: {price:,.0f}, 수수료: {commission:,.0f}, 세금: {tax:,.0f}, 남은 현금: {self.cash:,.0f}")
            else:
                logging.warning(f"[{current_dt.isoformat()}] 보유 수량 부족으로 매도 불가: {stock_code}, 요청 수량: {quantity}, 보유 수량: {self.positions.get(stock_code, {}).get('size', 0)}")

    def place_orders_from_signals(self, signals, current_date):
        """
        생성된 매수/매도 시그널에 따라 주문을 실행합니다.
        Broker 내부의 data_store를 사용하여 가격 정보를 가져옵니다.
        """
        logging.info(f"[{current_date.isoformat()}] 시그널에 따른 주문 실행 시작.")
        
        # 매도 시그널 먼저 처리하여 현금 확보
        sell_orders = []
        for stock_code, signal_info in signals.items():
            if signal_info['signal'] == 'sell':
                sell_orders.append((stock_code, signal_info.get('target_quantity', 0)))
        
        for stock_code, quantity_to_sell in sell_orders:
            try:
                current_price_data = self._get_historical_data_up_to('daily', stock_code, current_date, lookback_period=1)
                current_price = current_price_data['close'].iloc[-1]
                self.execute_order(stock_code, 'sell', quantity_to_sell, current_price, current_date)
            except (ValueError, IndexError, AttributeError, KeyError) as e: 
                logging.error(f"[{current_date.isoformat()}] {stock_code} 매도 가격 정보 없음 또는 데이터 부족: {e}. 매도 불가.")
                continue 

        # 매수 시그널 처리 (현금 확보 후)
        buy_orders = []
        for stock_code, signal_info in signals.items():
            if signal_info['signal'] == 'buy':
                buy_orders.append((stock_code, signal_info.get('target_quantity', 0)))
        
        for stock_code, quantity_to_buy in buy_orders:
            try:
                current_price_data = self._get_historical_data_up_to('daily', stock_code, current_date, lookback_period=1)
                current_price = current_price_data['close'].iloc[-1]
                self.execute_order(stock_code, 'buy', quantity_to_buy, current_price, current_date)
            except (ValueError, IndexError, AttributeError, KeyError) as e:
                logging.error(f"[{current_date.isoformat()}] {stock_code} 매수 가격 정보 없음 또는 데이터 부족: {e}. 매수 불가.")
                continue 
        
        logging.info(f"[{current_date.isoformat()}] 시그널에 따른 주문 실행 완료. 현재 현금: {self.cash:,.0f}원")

    def _get_historical_data_up_to(self, data_type, stock_code, current_date, lookback_period=1):
        """
        지정된 날짜까지의 과거 데이터를 가져옵니다.
        Broker가 초기화 시 받은 data_store를 사용합니다.
        """
        if not hasattr(self, 'data_store') or self.data_store is None:
            raise AttributeError("Broker에 data_store가 설정되지 않았습니다. 가격 데이터를 가져올 수 없습니다.")

        data = self.data_store[data_type].get(stock_code)
        if data is None or data.empty:
            raise ValueError(f"데이터 스토어에 {stock_code}의 {data_type} 데이터가 없습니다.")

        end_date = pd.Timestamp(current_date).normalize()
        historical_data = data.loc[data.index.normalize() <= end_date] # 인덱스를 normalize하여 비교
        
        if len(historical_data) < lookback_period:
            raise ValueError(f"데이터 부족: {stock_code}에 대해 {lookback_period}일 데이터가 필요하지만 {len(historical_data)}일만 있습니다 (종료 날짜: {current_date}).")
        
        return historical_data.iloc[-lookback_period:]

    def get_portfolio_value(self, current_prices):
        """현재 보유한 모든 종목의 가치를 포함한 총 포트폴리오 가치를 계산합니다."""
        total_stock_value = 0
        for stock_code, position in self.positions.items():
            if stock_code in current_prices:
                total_stock_value += position['size'] * current_prices[stock_code]
            else:
                logging.warning(f"경고: {stock_code}의 현재 가격을 찾을 수 없습니다. 포트폴리오 가치 계산에서 제외됩니다.")
        return self.cash + total_stock_value

    def get_total_stock_value(self, current_date):
        """
        현재 보유한 모든 주식 포지션의 총 가치를 계산합니다.
        이 메서드는 Broker 내부의 data_store를 사용하여 가격을 조회합니다.
        """
        total_value = 0
        for stock_code, position_info in self.positions.items():
            if position_info['size'] > 0:
                try:
                    current_price_data = self._get_historical_data_up_to('daily', stock_code, current_date, lookback_period=1)
                    current_price = current_price_data['close'].iloc[-1]
                    total_value += position_info['size'] * current_price
                except (ValueError, IndexError, AttributeError, KeyError) as e:
                    logging.warning(f"[{current_date.isoformat()}] {stock_code}의 현재 가격을 가져올 수 없어 총 주식 가치 계산에서 제외: {e}")
        return total_value
    
    def get_positions(self):
        """현재 보유 포지션을 반환합니다."""
        return self.positions

    def get_cash(self):
        """현재 현금 잔고를 반환합니다."""
        return self.cash

    def get_position_size(self, stock_code):
        """특정 종목의 보유 수량을 반환합니다."""
        return self.positions.get(stock_code, {}).get('size', 0)

    def close_all_positions(self, current_date):
        """현재 보유한 모든 종목을 매도하여 현금화합니다."""
        logging.info(f"[{current_date.isoformat()}] 모든 포지션 청산 시도 중...")
        
        positions_to_close = list(self.positions.keys()) 
        
        if not positions_to_close:
            logging.info(f"[{current_date.isoformat()}] 청산할 포지션이 없습니다.")
            return

        signals_to_place = {}
        for stock_code in positions_to_close:
            if self.positions[stock_code]['size'] > 0:
                signals_to_place[stock_code] = {
                    'signal': 'sell',
                    'target_quantity': self.positions[stock_code]['size']
                }
        
        if signals_to_place:
            self.place_orders_from_signals(signals_to_place, current_date)
            logging.info(f"[{current_date.isoformat()}] 모든 포지션 청산 완료.")
        else:
            logging.info(f"[{current_date.isoformat()}] 청산할 포지션이 없거나 이미 청산되었습니다.")

    def check_stop_loss(self, current_date, current_prices):
        """
        현재 포지션에 대해 손절매 조건을 확인하고 필요한 경우 매도 주문을 실행합니다.
        
        Args:
            current_date (datetime.datetime): 현재 백테스트 날짜.
            current_prices (dict): 현재 시점의 각 종목별 종가 {stock_code: price}.
        """
        if not self.stop_loss_params:
            return

        positions_to_close_by_stop_loss = {}
        for stock_code, position_info in list(self.positions.items()):
            if position_info['size'] > 0:
                entry_price = position_info['avg_price']
                current_price = current_prices.get(stock_code)

                if current_price is None:
                    logging.warning(f"[{current_date.isoformat()}] 손절매 확인: {stock_code}의 현재 가격을 찾을 수 없습니다. 건너뜁니다.")
                    continue

                profit_loss_ratio = ((current_price - entry_price) / entry_price) * 100

                # 기본 손절매
                if self.stop_loss_params.get('stop_loss_ratio') is not None and \
                   profit_loss_ratio <= self.stop_loss_params['stop_loss_ratio']:
                    logging.warning(f"[{current_date.isoformat()}] 손절매 발생: {stock_code}, 수익률: {profit_loss_ratio:.2f}%, 기준: {self.stop_loss_params['stop_loss_ratio']:.2f}%")
                    positions_to_close_by_stop_loss[stock_code] = position_info['size']
                
                # 초기 손절매
                if self.stop_loss_params.get('early_stop_loss') is not None and \
                   self.stop_loss_params.get('early_stop_loss_days') is not None:
                    days_since_entry = (current_date.date() - position_info['entry_date']).days
                    if days_since_entry <= self.stop_loss_params['early_stop_loss_days'] and \
                       profit_loss_ratio <= self.stop_loss_params['early_stop_loss']:
                        logging.warning(f"[{current_date.isoformat()}] 초기 손절매 발생: {stock_code}, 경과일: {days_since_entry}일, 수익률: {profit_loss_ratio:.2f}%")
                        positions_to_close_by_stop_loss[stock_code] = position_info['size']

        if positions_to_close_by_stop_loss:
            logging.info(f"[{current_date.isoformat()}] 손절매 실행: {list(positions_to_close_by_stop_loss.keys())}")
            signals_to_place = {}
            for stock_code, quantity in positions_to_close_by_stop_loss.items():
                signals_to_place[stock_code] = {'signal': 'sell', 'target_quantity': quantity}
            self.place_orders_from_signals(signals_to_place, current_date)