import datetime
import logging
import pandas as pd
import sys
import os
import win32com.client
import numpy as np
import time
from api.creon_api import CreonAPIClient

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- 지표 계산 함수 (Pandas 기반) ---
def calculate_momentum(data, period):
    """주어진 데이터프레임의 'close' 가격에 대한 모멘텀 스코어를 계산합니다."""
    return (data['close'].pct_change(period).fillna(0) * 100)

def calculate_rsi(data, period):
    """주어진 데이터프레임의 'close' 가격에 대한 RSI를 계산합니다."""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 100)

def get_next_weekday(date, target_weekday):
    """주어진 날짜로부터 다음 target_weekday(0=월요일, 6=일요일)를 찾습니다."""
    days_ahead = target_weekday - date.weekday()
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return date + datetime.timedelta(days=days_ahead)

# --- 포트폴리오 및 브로커 시뮬레이션 클래스 ---
class Broker:
    def __init__(self, initial_cash, commission_rate):
        self.cash = initial_cash
        self.positions = {}  # {stock_code: {'size': int, 'avg_price': float, 'entry_date': datetime.date}}
        self.commission_rate = commission_rate
        self.trade_log = []  # 매매 기록

    def get_position_size(self, stock_code):
        """특정 종목의 현재 보유 수량을 반환합니다."""
        return self.positions.get(stock_code, {}).get('size', 0)

    def get_portfolio_value(self, current_prices):
        """현재 현금과 보유 포지션 평가액을 합산하여 포트폴리오 가치 반환합니다."""
        total_value = self.cash
        for stock_code, pos in self.positions.items():
            if pos['size'] > 0 and stock_code in current_prices:
                total_value += pos['size'] * current_prices[stock_code]
        return total_value

    def _calculate_commission(self, price, size):
        """거래 수수료를 계산합니다."""
        return price * size * self.commission_rate

    def execute_order(self, stock_code, order_type, price, size, current_time):
        """
        주문 실행을 시뮬레이션하고 현금/포지션을 업데이트합니다.
        order_type: 'buy' 또는 'sell'
        """
        commission = self._calculate_commission(price, size)
        
        log_msg = ""
        success = False

        if order_type == 'buy':
            total_cost = price * size + commission
            if self.cash >= total_cost:
                self.cash -= total_cost
                current_size = self.positions.get(stock_code, {'size': 0})['size']
                current_avg_price = self.positions.get(stock_code, {'avg_price': 0})['avg_price']

                new_size = current_size + size
                new_avg_price = ((current_avg_price * current_size) + (price * size)) / new_size if new_size > 0 else 0
                
                self.positions[stock_code] = {
                    'size': new_size, 
                    'avg_price': new_avg_price,
                    'entry_date': current_time.date()  # 매수 날짜 기록
                }
                
                log_msg = (f"매수 체결: {stock_code}, 수량: {size}주, 가격: {price:,.0f}원, "
                           f"수수료: {commission:,.0f}원, 현금잔고: {self.cash:,.0f}원, 보유수량: {new_size}주")
                success = True
            else:
                log_msg = f"매수 실패: {stock_code} - 현금 부족 (필요: {total_cost:,.0f}원, 보유: {self.cash:,.0f}원)"

        elif order_type == 'sell':
            current_size = self.positions.get(stock_code, {'size': 0})['size']
            if current_size >= size:
                current_avg_price = self.positions.get(stock_code, {'avg_price': 0})['avg_price']
                
                pnl = (price - current_avg_price) * size - commission
                self.cash += (price * size - commission)

                new_size = current_size - size
                if new_size == 0:
                    del self.positions[stock_code]
                    log_msg = (f"매도 체결: {stock_code}, 수량: {size}주, 가격: {price:,.0f}원, "
                               f"수수료: {commission:,.0f}원, 손익: {pnl:,.0f}원, 현금잔고: {self.cash:,.0f}원, 포지션: 청산완료")
                else:
                    self.positions[stock_code]['size'] = new_size
                    log_msg = (f"매도 체결: {stock_code}, 수량: {size}주, 가격: {price:,.0f}원, "
                               f"수수료: {commission:,.0f}원, 손익: {pnl:,.0f}원, 현금잔고: {self.cash:,.0f}원, 잔여수량: {new_size}주")
                success = True
            else:
                log_msg = f"매도 실패: {stock_code} - 보유수량 부족 (현재: {current_size}주, 요청: {size}주)"
        
        if success:
            logging.info(f"{current_time.isoformat()} - {log_msg}")
            self.trade_log.append({
                'datetime': current_time,
                'stock_code': stock_code,
                'order_type': order_type,
                'price': price,
                'size': size,
                'commission': commission,
                'cash_after_trade': self.cash,
                'position_size_after_trade': self.get_position_size(stock_code)
            })
        else:
            logging.warning(f"{current_time.isoformat()} - {log_msg}")
        return success 

# --- 백테스팅 엔진 클래스 ---
class Backtester:
    def __init__(self, creon_api_client, initial_cash=10_000_000, commission_rate=0.00015):
        self.broker = Broker(initial_cash, commission_rate)
        self.initial_cash = initial_cash
        self.creon_api_client = creon_api_client
        self.data_store = {
            'daily': {},   # {stock_code: DataFrame}
            'minute': {}   # {stock_code: {date: DataFrame}}
        }
        self.strategy_params = {
            'momentum_period': 20,          # 모멘텀 계산 기간 (거래일)
            'rebalance_weekday': 4,         # 리밸런싱 요일 (0: 월요일, 4: 금요일)
            'num_top_stocks': 7,            # 상위 7종목 선택
            'safe_asset_code': 'A439870',   # 안전자산 코드 (국고채 ETF)
            'minute_rsi_period': 45,        # 분봉 RSI 기간 (60분 → 45분)
            'minute_rsi_overbought': 65,    # RSI 과매수 기준 (70 → 65)
            'minute_rsi_oversold': 35,      # RSI 과매도 기준 (30 → 35)
            'stop_loss_ratio': -5.0,        # 기본 손절 비율
            'trailing_stop_ratio': -3.0,    # 트레일링 스탑 비율
            'portfolio_stop_loss': -10.0,   # 포트폴리오 전체 손절 비율
            'early_stop_loss': -3.0,        # 초기 손절 비율 (5일 이내)
            'max_losing_positions': 3        # 동시 손실 허용 종목 수
        }
        
        # 동일비중 투자금액 설정
        self.strategy_params['equal_weight_amount'] = initial_cash / self.strategy_params['num_top_stocks']
        
        # 듀얼 모멘텀 전략 상태 관리
        self.momentum_signals = {}
        self.last_rebalance_date = None
        self._last_daily_processed_date = None
        
        # 포지션 정보 (손절 관련)
        self.position_info = {}  # {stock_code: {'highest_price': float}}

    def add_daily_data(self, stock_code, daily_df):
        """백테스터에 종목별 일봉 데이터를 추가합니다."""
        self.data_store['daily'][stock_code] = daily_df
        self._initialize_momentum_signal(stock_code)

    def _initialize_momentum_signal(self, stock_code):
        """종목별 모멘텀 시그널 초기화"""
        if stock_code not in self.momentum_signals:
            self.momentum_signals[stock_code] = {
                'momentum_score': 0,
                'rank': 0,
                'signal': None,
                'signal_date': None,
                # 'signal_valid_until': None,  # 시그널 유효기간 만료일
                'traded_today': False,
                'target_amount': 0,
                'target_quantity': 0
            }

    def get_next_business_day(self, date):
        """일봉 데이터를 기반으로 다음 거래일을 찾습니다."""
        next_day = date + datetime.timedelta(days=1)
        max_attempts = 10
        attempts = 0
        
        while attempts < max_attempts:
            has_data = False
            for stock_code in self.data_store['daily']:
                daily_df = self.data_store['daily'][stock_code]
                if not daily_df.empty:
                    next_day_normalized = pd.Timestamp(next_day).normalize()
                    if next_day_normalized in daily_df.index:
                        has_data = True
                        break
            
            if has_data:
                return next_day
            
            next_day += datetime.timedelta(days=1)
            attempts += 1
        
        logging.warning(f"{date.strftime('%Y-%m-%d')} 이후 {max_attempts}일 이내에 거래일을 찾을 수 없습니다.")
        return None

    def get_previous_business_days(self, current_date, n_days=2):
        """현재 날짜로부터 이전 n개의 거래일을 찾습니다."""
        previous_days = []
        check_date = current_date - datetime.timedelta(days=1)
        days_found = 0
        
        while days_found < n_days and check_date >= pd.Timestamp(daily_data_fetch_start).date():
            has_data = False
            for stock_code in self.data_store['daily']:
                daily_df = self.data_store['daily'][stock_code]
                if not daily_df.empty:
                    check_date_normalized = pd.Timestamp(check_date).normalize()
                    if check_date_normalized in daily_df.index:
                        has_data = True
                        break
            
            if has_data:
                previous_days.append(check_date)
                days_found += 1
            
            check_date -= datetime.timedelta(days=1)
        
        return sorted(previous_days)  # 날짜 순으로 정렬하여 반환

    def _get_minute_data_for_signal_dates(self, stock_code, signal_date):
        """매수/매도 시그널이 발생한 날짜와 다음 거래일의 분봉 데이터를 조회합니다."""
        # 다음 거래일 찾기
        next_trading_day = self.get_next_business_day(signal_date)
        if next_trading_day is None:
            logging.warning(f"{signal_date} 이후의 다음 거래일을 찾을 수 없습니다 - {stock_code}")
            return pd.DataFrame()
            
        # 시그널 발생일과 다음 거래일의 분봉 데이터 로드
        dates_to_load = [signal_date, next_trading_day]
        
        dfs_to_concat = []
        for date in dates_to_load:
            date_str = date.strftime('%Y%m%d')
            
            # 해당 날짜의 분봉 데이터가 이미 있는지 확인
            if stock_code in self.data_store['minute'] and date in self.data_store['minute'][stock_code]:
                dfs_to_concat.append(self.data_store['minute'][stock_code][date])
                continue
            
            # 해당 날짜가 거래일인지 확인
            daily_df = self.data_store['daily'][stock_code]
            if not daily_df.empty and pd.Timestamp(date).normalize() in daily_df.index:
                minute_df_day = self.creon_api_client.get_minute_ohlcv(stock_code, date_str, date_str, interval=1)
                time.sleep(0.3)  # API 호출 제한 방지를 위한 대기
                
                if not minute_df_day.empty:
                    if stock_code not in self.data_store['minute']:
                        self.data_store['minute'][stock_code] = {}
                    self.data_store['minute'][stock_code][date] = minute_df_day
                    dfs_to_concat.append(minute_df_day)
                    logging.info(f"{stock_code} 종목의 {date_str} 분봉 데이터 로드 완료. 데이터 수: {len(minute_df_day)}행")
                else:
                    logging.warning(f"{stock_code} 종목의 {date_str} 분봉 데이터가 없습니다 (거래일임에도 불구하고).")
        
        if dfs_to_concat:
            full_df = pd.concat(dfs_to_concat).sort_index()
            return full_df
        return pd.DataFrame()

    def _get_bar_at_time(self, data_type, stock_code, target_dt):
        """주어진 시간(target_dt)에 해당하는 정확한 OHLCV 바를 반환합니다."""
        if data_type == 'daily':
            df = self.data_store['daily'].get(stock_code)
            if df is None or df.empty:
                return None
            try:
                target_dt_normalized = pd.Timestamp(target_dt).normalize()
                return df.loc[target_dt_normalized]
            except KeyError:
                return None
        elif data_type == 'minute':
            target_date = target_dt.date()
            if stock_code not in self.data_store['minute'] or target_date not in self.data_store['minute'][stock_code]:
                return None
            df = self.data_store['minute'][stock_code][target_date]
            if df is None or df.empty:
                return None
            try:
                return df.loc[target_dt]
            except KeyError:
                return None
        return None

    def _get_historical_data_up_to(self, data_type, stock_code, current_dt, lookback_period=None):
        """주어진 시간(current_dt)까지의 모든 과거 데이터를 반환합니다."""
        if data_type == 'daily':
            df = self.data_store['daily'].get(stock_code)
            if df is None or df.empty:
                return pd.DataFrame()
            current_dt_normalized = pd.Timestamp(current_dt).normalize()
            filtered_df = df.loc[df.index.normalize() <= current_dt_normalized]
            if lookback_period:
                return filtered_df.tail(lookback_period)
            return filtered_df
        
        elif data_type == 'minute':
            all_minute_dfs_for_stock = []
            if stock_code in self.data_store['minute']:
                for date_key in sorted(self.data_store['minute'][stock_code].keys()):
                    if date_key <= current_dt.date():
                        all_minute_dfs_for_stock.append(self.data_store['minute'][stock_code][date_key])
            
            if not all_minute_dfs_for_stock:
                return pd.DataFrame()
            
            combined_minute_df = pd.concat(all_minute_dfs_for_stock).sort_index()
            filtered_df = combined_minute_df.loc[combined_minute_df.index <= current_dt]
            if lookback_period:
                return filtered_df.tail(lookback_period)
            return filtered_df
        return pd.DataFrame()

    def _calculate_target_quantity(self, stock_code, current_price):
        """주어진 가격에서 동일비중 투자에 필요한 수량을 계산합니다."""
        target_amount = self.strategy_params['equal_weight_amount']
        
        # 현재 보유 현금 확인
        available_cash = self.broker.cash
        
        # 수수료를 고려한 실제 투자 가능 금액 계산
        commission_rate = self.broker.commission_rate
        max_buyable_amount = available_cash / (1 + commission_rate)
        
        # 목표 투자금액과 실제 투자 가능 금액 중 작은 값 선택
        actual_investment_amount = min(target_amount, max_buyable_amount)
        
        # 주식 수량 계산 (소수점 이하 버림)
        quantity = int(actual_investment_amount / current_price)
        
        if quantity > 0:
            # 실제 필요한 현금 (수수료 포함) 계산
            total_cost = current_price * quantity * (1 + commission_rate)
            if total_cost > available_cash:
                # 수량을 1주 줄여서 재계산
                quantity -= 1
            
        if quantity > 0:
            logging.info(f"{stock_code} 종목 매수 수량 계산: {quantity}주 (목표금액: {target_amount:,.0f}원, 가용현금: {available_cash:,.0f}원, 현재가: {current_price:,.0f}원)")
        else:
            logging.warning(f"{stock_code} 종목 매수 불가: 현금 부족 (목표금액: {target_amount:,.0f}원, 가용현금: {available_cash:,.0f}원, 현재가: {current_price:,.0f}원)")
        
        return quantity

    def get_next_n_business_days(self, start_date, n_days):
        """주어진 날짜로부터 n개의 거래일 이후 날짜를 찾습니다."""
        next_date = start_date
        days_found = 0
        
        while days_found < n_days:
            next_date = self.get_next_business_day(next_date)
            if next_date is None:
                return None
            days_found += 1
            
        return next_date

    def _run_weekly_momentum_logic(self, current_daily_date):
        """주간 듀얼 모멘텀 로직을 실행합니다."""
        if current_daily_date.weekday() != self.strategy_params['rebalance_weekday']:
            return

        logging.info(f'{current_daily_date.isoformat()} - --- 주간 모멘텀 로직 실행 중 ---')

        # 현재 포트폴리오 가치 계산
        current_prices = {}
        for stock_code in self.data_store['daily']:
            daily_data = self._get_bar_at_time('daily', stock_code, pd.Timestamp(current_daily_date))
            if daily_data is not None:
                current_prices[stock_code] = daily_data['close']

        # 모든 종목의 모멘텀 스코어 계산
        momentum_scores = {}
        for stock_code in self.data_store['daily']:
            if stock_code == self.strategy_params['safe_asset_code']:
                continue  # 안전자산은 모멘텀 계산에서 제외

            # 종목별 시그널 초기화
            self._initialize_momentum_signal(stock_code)

            daily_df = self.data_store['daily'][stock_code]
            if daily_df.empty:
                continue

            historical_data = self._get_historical_data_up_to(
                'daily', 
                stock_code, 
                current_daily_date, 
                lookback_period=self.strategy_params['momentum_period'] + 1
            )

            if len(historical_data) < self.strategy_params['momentum_period']:
                logging.debug(f'{stock_code} 종목의 모멘텀 계산을 위한 데이터가 부족합니다.')
                continue

            momentum_score = calculate_momentum(historical_data, self.strategy_params['momentum_period']).iloc[-1]
            momentum_scores[stock_code] = momentum_score

        if not momentum_scores:
            logging.warning('계산된 모멘텀 스코어가 없습니다. 리밸런싱을 건너뜁니다.')
            return

        # 모멘텀 스코어로 정렬하여 순위 매기기
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 절대 모멘텀: 안전자산의 모멘텀 계산
        safe_asset_df = self.data_store['daily'].get(self.strategy_params['safe_asset_code'])
        safe_asset_momentum = 0
        if safe_asset_df is not None and not safe_asset_df.empty:
            safe_asset_data = self._get_historical_data_up_to(
                'daily',
                self.strategy_params['safe_asset_code'],
                current_daily_date,
                lookback_period=self.strategy_params['momentum_period'] + 1
            )
            if len(safe_asset_data) >= self.strategy_params['momentum_period']:
                safe_asset_momentum = calculate_momentum(safe_asset_data, self.strategy_params['momentum_period']).iloc[-1]

        # 현재 보유 종목 확인
        current_positions = set(self.broker.positions.keys())
        
        # 매수 대상 종목 선정
        buy_candidates = set()
        for rank, (stock_code, score) in enumerate(sorted_stocks, 1):
            if rank <= self.strategy_params['num_top_stocks'] and score > safe_asset_momentum:
                buy_candidates.add(stock_code)

        # 신호 생성
        for rank, (stock_code, score) in enumerate(sorted_stocks, 1):
            self.momentum_signals[stock_code].update({
                'momentum_score': score,
                'rank': rank,
                'signal_date': current_daily_date,
                'traded_today': False
            })

            if stock_code in buy_candidates:
                if stock_code in current_positions:
                    # 이미 보유 중인 종목은 홀딩
                    self.momentum_signals[stock_code]['signal'] = 'hold'
                    logging.info(f'홀딩 신호 - {stock_code}: 순위 {rank}위, 모멘텀 {score:.2f} (기존 보유 종목)')
                else:
                    # 새로 매수할 종목
                    current_price = self.data_store['daily'][stock_code].loc[pd.Timestamp(current_daily_date).normalize()]['close']
                    target_quantity = self._calculate_target_quantity(stock_code, current_price)
                    
                    self.momentum_signals[stock_code].update({
                        'signal': 'buy',
                        'target_amount': self.strategy_params['equal_weight_amount'],
                        'target_quantity': target_quantity
                    })
                    logging.info(f'매수 신호 - {stock_code}: 순위 {rank}위, 모멘텀 {score:.2f}, 목표수량 {target_quantity}주')
            else:
                self.momentum_signals[stock_code]['signal'] = 'sell'
                if stock_code in current_positions:
                    logging.info(f'매도 신호 - {stock_code} (보유중): 순위 {rank}위, 모멘텀 {score:.2f}')
                else:
                    logging.debug(f'매도 신호 - {stock_code} (미보유): 순위 {rank}위, 모멘텀 {score:.2f}')

        # 리밸런싱 계획 요약
        portfolio_value = self.broker.get_portfolio_value(current_prices)
        current_holdings = [(code, pos['size'] * current_prices[code]) for code, pos in self.broker.positions.items()]
        total_holdings_value = sum(value for _, value in current_holdings)
        
        # 매수 계획 계산
        new_buys = [(code, self.momentum_signals[code]['target_quantity'] * current_prices[code]) 
                    for code in buy_candidates if code not in current_positions]
        total_buy_amount = sum(amount for _, amount in new_buys)
        
        # 매도 계획 계산
        to_sell = [(code, pos['size'] * current_prices[code]) 
                   for code, pos in self.broker.positions.items() 
                   if code not in buy_candidates]
        total_sell_amount = sum(amount for _, amount in to_sell)

        # 리밸런싱 계획 출력
        logging.info("\n=== 리밸런싱 계획 요약 ===")
        logging.info(f"현재 상태: 포트폴리오 가치 {portfolio_value:,.0f}원 = 보유종목 {len(current_holdings)}개 ({total_holdings_value:,.0f}원) + 현금 {self.broker.cash:,.0f}원")
        logging.info(f"매수 계획: {len(new_buys)}종목 (소요금액: {total_buy_amount:,.0f}원)")
        logging.info(f"매도 계획: {len(to_sell)}종목 (회수금액: {total_sell_amount:,.0f}원)")

        self.last_rebalance_date = current_daily_date

    def _check_stop_loss(self, stock_code, current_price, position_info, current_dt):
        """개별 종목 손절 조건 체크"""
        avg_price = position_info['avg_price']
        loss_ratio = (current_price - avg_price) / avg_price * 100
        
        # 1. 단순 손절
        if loss_ratio <= self.strategy_params['stop_loss_ratio']:
            logging.info(f"[손절매 발생] {stock_code}: 현재 손실률 {loss_ratio:.2f}%가 기준치 {self.strategy_params['stop_loss_ratio']}%를 초과")
            return True
            
        # 2. 트레일링 스탑
        if stock_code in self.position_info:
            highest_price = self.position_info[stock_code]['highest_price']
            trailing_loss_ratio = (current_price - highest_price) / highest_price * 100
            if trailing_loss_ratio <= self.strategy_params['trailing_stop_ratio']:
                logging.info(f"[트레일링 스탑] {stock_code}: 현재가 {current_price:,.0f}원이 최고가 {highest_price:,.0f}원 대비 {trailing_loss_ratio:.2f}% 하락")
                return True
        
        # 3. 보유 기간 기반 손절 폭 조정
        holding_days = (current_dt.date() - position_info['entry_date']).days
        if holding_days <= 5:  # 매수 후 5일 이내
            if loss_ratio <= self.strategy_params['early_stop_loss']:
                logging.info(f"[조기 손절매] {stock_code}: 매수 후 {holding_days}일 이내 손실률 {loss_ratio:.2f}%가 조기 손절 기준 {self.strategy_params['early_stop_loss']}% 초과")
                return True
        
        return False

    def _check_portfolio_stop_loss(self, current_prices):
        """포트폴리오 전체 손절 조건 체크"""
        # 1. 전체 손실폭 기준
        portfolio_value = self.broker.get_portfolio_value(current_prices)
        total_loss_ratio = (portfolio_value - self.initial_cash) / self.initial_cash * 100
        
        if total_loss_ratio <= self.strategy_params['portfolio_stop_loss']:
            logging.info(f"[포트폴리오 손절매] 전체 손실률 {total_loss_ratio:.2f}%가 기준치 {self.strategy_params['portfolio_stop_loss']}% 초과")
            return True
        
        # 2. 동시다발적 손실 기준
        losing_positions = 0
        for stock_code, pos_info in self.broker.positions.items():
            if stock_code in current_prices:
                loss_ratio = (current_prices[stock_code] - pos_info['avg_price']) / pos_info['avg_price'] * 100
                if loss_ratio <= self.strategy_params['stop_loss_ratio']:
                    losing_positions += 1
        
        if losing_positions >= self.strategy_params['max_losing_positions']:
            logging.info(f"[포트폴리오 손절매] 동시 손실 종목 수 {losing_positions}개가 최대 허용치 {self.strategy_params['max_losing_positions']}개 초과")
            return True
        
        return False

    def _update_position_info(self, stock_code, current_price):
        """포지션 정보 업데이트 (고점 등)"""
        if stock_code not in self.position_info:
            self.position_info[stock_code] = {
                'highest_price': current_price
            }
        else:
            if current_price > self.position_info[stock_code]['highest_price']:
                self.position_info[stock_code]['highest_price'] = current_price

    def _get_last_price(self, stock_code):
        """종목의 마지막 거래 가격을 반환합니다."""
        daily_df = self.data_store['daily'].get(stock_code)
        if daily_df is not None and not daily_df.empty:
            return daily_df['close'].iloc[-1]
        return None

    def _run_minute_logic(self, stock_code, current_dt):
        """분봉 데이터를 기반으로 실제 매수/매도 주문을 실행합니다."""
        current_minute_date = current_dt.date()
        if stock_code not in self.data_store['minute'] or current_minute_date not in self.data_store['minute'][stock_code]:
            return

        current_minute_bar = self._get_bar_at_time('minute', stock_code, current_dt)
        if current_minute_bar is None:
            return

        current_minute_time = current_dt.time()
        current_price = current_minute_bar['close']

        # 포지션 정보 업데이트
        self._update_position_info(stock_code, current_price)

        # 현재 가격 정보 수집
        current_prices = {stock_code: current_price}
        for code in list(self.broker.positions.keys()):  # positions.keys()를 리스트로 변환하여 순회
            if code != stock_code:
                price_data = self._get_bar_at_time('minute', code, current_dt)
                if price_data is not None:
                    current_prices[code] = price_data['close']

        # 보유 중인 종목의 손절 체크
        position_info = self.broker.positions.get(stock_code)
        if position_info and position_info['size'] > 0:
            # 개별 종목 손절 체크
            if self._check_stop_loss(stock_code, current_price, position_info, current_dt):
                logging.info(f'[손절매 실행] {current_dt.isoformat()} - {stock_code} 매도. 가격: {current_price:,.0f}원')
                self.broker.execute_order(stock_code, 'sell', current_price, position_info['size'], current_dt)
                if stock_code in self.position_info:
                    del self.position_info[stock_code]
                return
                
            # 포트폴리오 전체 손절 체크
            if self._check_portfolio_stop_loss(current_prices):
                for code, pos in list(self.broker.positions.items()):
                    if pos['size'] > 0:
                        price = current_prices.get(code, self._get_last_price(code))
                        logging.info(f'[포트폴리오 손절매 실행] {current_dt.isoformat()} - {code} 매도. 가격: {price:,.0f}원')
                        self.broker.execute_order(code, 'sell', price, pos['size'], current_dt)
                        if code in self.position_info:
                            del self.position_info[code]
                return

        # 기존 매수/매도 로직 실행
        momentum_signal_info = self.momentum_signals[stock_code]
        momentum_signal = momentum_signal_info['signal']
        signal_date = momentum_signal_info['signal_date']
        target_quantity = momentum_signal_info.get('target_quantity', 0)

        # 시그널 유효성 검사
        if momentum_signal is None or current_minute_date <= signal_date:
            return
            
        if self.momentum_signals[stock_code]['traded_today']:
            return
            
        current_position_size = self.broker.get_position_size(stock_code)
        
        required_rsi_data_len = self.strategy_params['minute_rsi_period'] + 1
        minute_historical_data = self._get_historical_data_up_to('minute', stock_code, current_dt, lookback_period=required_rsi_data_len)
        
        if len(minute_historical_data) < required_rsi_data_len:
            return

        current_rsi_value_series = calculate_rsi(minute_historical_data, self.strategy_params['minute_rsi_period'])
        current_rsi_value = current_rsi_value_series.iloc[-1]
        
        if pd.isna(current_rsi_value):
            return

        # --- 매수 로직 ---
        if momentum_signal == 'buy':
            if current_position_size <= 0:
                buy_executed = False
                if current_minute_time >= datetime.time(10, 0):
                    if current_rsi_value <= self.strategy_params['minute_rsi_oversold']:
                        logging.info(f'[RSI 매수] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
                        buy_executed = self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
                        
                    elif current_minute_time == datetime.time(15, 20):
                        logging.info(f'[강제 매수] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원, 수량: {target_quantity}주')
                        buy_executed = self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
                        
                    if buy_executed:
                        self.momentum_signals[stock_code]['traded_today'] = True

        # --- 매도 로직 ---
        elif momentum_signal == 'sell':
            if current_position_size > 0:
                sell_executed = False
                if current_rsi_value >= self.strategy_params['minute_rsi_overbought']:
                    logging.info(f'[RSI 매도] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원')
                    sell_executed = self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
                    
                elif current_minute_time == datetime.time(9, 5):
                    logging.info(f'[강제 매도] {current_dt.isoformat()} - {stock_code} RSI: {current_rsi_value:.2f}, 가격: {current_price:,.0f}원')
                    sell_executed = self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
                    
                if sell_executed:
                    self.momentum_signals[stock_code]['traded_today'] = True
                    if stock_code in self.position_info:
                        del self.position_info[stock_code]

    def calculate_performance_metrics(self, portfolio_values, risk_free_rate=0.03):
        """
        포트폴리오 성과 지표를 계산합니다.
        
        Args:
            portfolio_values: 일별 포트폴리오 가치 시계열 데이터
            risk_free_rate: 무위험 수익률 (연율화된 값, 예: 0.03 = 3%)
        """
        # 일별 수익률 계산
        daily_returns = portfolio_values.pct_change().dropna()
        
        # 누적 수익률 계산
        cumulative_returns = (1 + daily_returns).cumprod()
        
        # MDD (Maximum Drawdown) 계산
        cumulative_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / cumulative_max - 1
        mdd = drawdowns.min()
        
        # 연간 수익률 계산 (연율화)
        total_days = len(daily_returns)
        total_years = total_days / 252  # 거래일 기준
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (1 / total_years) - 1
        
        # 연간 변동성 계산 (연율화)
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        # 샤프 지수 계산
        excess_returns = annual_return - risk_free_rate
        sharpe_ratio = excess_returns / annual_volatility if annual_volatility != 0 else 0
        
        # 승률 계산
        win_rate = len(daily_returns[daily_returns > 0]) / len(daily_returns)
        
        # 평균 수익거래 대비 손실거래 비율 (Profit Factor)
        positive_returns = daily_returns[daily_returns > 0].mean()
        negative_returns = abs(daily_returns[daily_returns < 0].mean())
        profit_factor = positive_returns / negative_returns if negative_returns != 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'mdd': mdd,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

    def run(self, backtest_start_dt, backtest_end_dt):
        """백테스트를 실행합니다."""
        logging.info("=== 듀얼 모멘텀 주간 백테스트 시작 ===")
        
        # 포트폴리오 가치 기록을 위한 리스트
        portfolio_values = []
        dates = []
        
        # 일봉 데이터를 날짜 순으로 정렬하여 이터레이션
        all_daily_dates = pd.DatetimeIndex([])
        for stock_code, daily_df in self.data_store['daily'].items():
            if not daily_df.empty:
                all_daily_dates = all_daily_dates.union(pd.DatetimeIndex(daily_df.index).normalize())

        daily_dates_to_process = all_daily_dates[
            (all_daily_dates >= pd.Timestamp(backtest_start_dt).normalize()) & \
            (all_daily_dates <= pd.Timestamp(backtest_end_dt).normalize())
        ].sort_values()

        if daily_dates_to_process.empty:
            logging.error("No daily data available within the specified backtest period. Exiting.")
            return

        # 백테스트 진행 루프 (매일 단위로 진행)
        for current_daily_date_full in daily_dates_to_process:
            current_daily_date = current_daily_date_full.date()
            logging.info(f"\n--- 처리 중인 날짜: {current_daily_date.isoformat()} ---")

            # 매일 시작 시 'traded_today' 플래그 초기화
            for stock_code in self.momentum_signals:
                self.momentum_signals[stock_code]['traded_today'] = False

            # 주간 모멘텀 로직 실행 (지정된 요일에만)
            self._run_weekly_momentum_logic(current_daily_date)

            # 매수/매도 시그널이 있는 종목들에 대해서만 분봉 데이터 처리
            for stock_code, signal_info in self.momentum_signals.items():
                if signal_info['signal'] in ['buy', 'sell'] and not signal_info['traded_today']:
                    signal_date = signal_info['signal_date']
                    
                    # 매도 시그널인 경우 현재 보유 중인 종목만 처리
                    if signal_info['signal'] == 'sell' and self.broker.get_position_size(stock_code) <= 0:
                        continue
                    
                    # 다음 거래일 찾기
                    next_trading_day = self.get_next_business_day(signal_date)
                    if next_trading_day is None:
                        logging.warning(f"Could not find next trading day after {signal_date} for {stock_code}. Skipping.")
                        continue

                    if next_trading_day == current_daily_date:
                        # 분봉 데이터 로드 및 매매 로직 실행
                        minute_data = self._get_minute_data_for_signal_dates(stock_code, signal_date)
                        if not minute_data.empty:
                            for minute_dt in minute_data.index:
                                if minute_dt < backtest_start_dt:
                                    continue
                                if minute_dt > backtest_end_dt:
                                    break
                                self._run_minute_logic(stock_code, minute_dt)

            # 현재 가격 정보 수집 및 포트폴리오 가치 계산
            current_prices = {}
            for stock_code in self.data_store['daily']:
                daily_data = self._get_bar_at_time('daily', stock_code, current_daily_date_full)
                if daily_data is not None:
                    current_prices[stock_code] = daily_data['close']
            
            # 포트폴리오 가치 계산 및 기록
            portfolio_value = self.broker.get_portfolio_value(current_prices)
            portfolio_values.append(portfolio_value)
            dates.append(current_daily_date_full)

        # 포트폴리오 가치 시계열 데이터 생성
        portfolio_value_series = pd.Series(portfolio_values, index=dates)
        
        # 성과 지표 계산
        metrics = self.calculate_performance_metrics(portfolio_value_series)
        
        # 최종 결과 출력
        logging.info("\n=== 백테스트 결과 ===")
        logging.info(f"시작일: {backtest_start_dt.date().isoformat()}")
        logging.info(f"종료일: {backtest_end_dt.date().isoformat()}")
        logging.info(f"초기자금: {self.initial_cash:,.0f}원")
        logging.info(f"최종 포트폴리오 가치: {portfolio_values[-1]:,.0f}원")
        logging.info(f"총 수익률: {metrics['total_return']*100:.2f}%")
        logging.info(f"연간 수익률: {metrics['annual_return']*100:.2f}%")
        logging.info(f"연간 변동성: {metrics['annual_volatility']*100:.2f}%")
        logging.info(f"샤프 지수: {metrics['sharpe_ratio']:.2f}")
        logging.info(f"최대 낙폭 (MDD): {metrics['mdd']*100:.2f}%")
        logging.info(f"승률: {metrics['win_rate']*100:.2f}%")
        logging.info(f"수익비 (Profit Factor): {metrics['profit_factor']:.2f}")
        
        # 최종 포지션 정보 출력
        logging.info("\n--- 최종 포지션 현황 ---")
        if self.broker.positions:
            for stock_code, pos_info in self.broker.positions.items():
                logging.info(f"{stock_code}: 보유수량 {pos_info['size']}주, 평균단가 {pos_info['avg_price']:,.0f}원")
        else:
            logging.info("보유 중인 종목 없음")
        
        return portfolio_value_series, metrics


# --- 메인 실행 로직 ---
if __name__ == '__main__':
    logging.info("듀얼 모멘텀 주간 백테스트 스크립트를 실행합니다.")

    # CreonAPIClient 인스턴스 생성
    creon_api_client = CreonAPIClient()
    if not creon_api_client.connected:
        logging.error("Creon API에 연결할 수 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    # 백테스트 기간 설정
    daily_data_fetch_start = '20250301'  # 3개월 백테스트를 위해 시작일 조정 (2025년 1월 1일부터)
    backtest_start_date = datetime.datetime(2025, 4, 1, 9, 0, 0)  # 실제 백테스트 시작 시간
    backtest_end_date = datetime.datetime(2025, 6, 4, 15, 30, 0)  # 실제 백테스트 종료 시간

    # 섹터별 대표 종목 리스트
    sector_stocks = {
        '반도체': [
            ('삼성전자', 'IT'), ('SK하이닉스', 'IT'), ('DB하이텍', 'IT'),
            ('네패스아크', 'IT'), ('와이아이케이', 'IT')
        ],
        '2차전지': [
            ('LG에너지솔루션', '2차전지'), ('삼성SDI', '2차전지'), ('SK이노베이션', '2차전지'),
            ('에코프로비엠', '2차전지'), ('포스코퓨처엠', '2차전지'), ('LG화학', '2차전지'),
            ('일진머티리얼즈', '2차전지'), ('엘앤에프', '2차전지')
        ],
        '바이오': [
            ('삼성바이오로직스', '바이오'), ('셀트리온', '바이오'), ('SK바이오사이언스', '바이오'),
            ('유한양행', '바이오'), ('한미약품', '바이오')
        ],
        '플랫폼/인터넷': [
            ('NAVER', 'IT'), ('카카오', 'IT'), ('크래프톤', 'IT'),
            ('엔씨소프트', 'IT'), ('넷마블', 'IT')
        ],
        '자동차': [
            ('현대차', '자동차'), ('기아', '자동차'), ('현대모비스', '자동차'),
            ('만도', '자동차'), ('한온시스템', '자동차')
        ],
        '철강/화학': [
            ('POSCO홀딩스', '철강'), ('고려아연', '철강'), ('롯데케미칼', '화학'),
            ('금호석유', '화학'), ('효성첨단소재', '화학')
        ],
        '금융': [
            ('KB금융', '금융'), ('신한지주', '금융'), ('하나금융지주', '금융'),
            ('우리금융지주', '금융'), ('메리츠금융지주', '금융')
        ],
        '통신': [
            ('SK텔레콤', '통신'), ('KT', '통신'), ('LG유플러스', '통신'),
            ('SK스퀘어', '통신')
        ],
        '유통/소비재': [
            ('CJ제일제당', '소비재'), ('오리온', '소비재'), ('롯데쇼핑', '유통'),
            ('이마트', '유통'), ('BGF리테일', '유통')
        ],
        '건설/기계': [
            ('현대건설', '건설'), ('대우건설', '건설'), ('GS건설', '건설'),
            ('두산에너빌리티', '기계'), ('두산밥캣', '기계')
        ],
        '조선/항공': [
            ('한국조선해양', '조선'), ('삼성중공업', '조선'), ('대한항공', '항공'),
            ('현대미포조선', '조선')
        ],
        '에너지': [
            ('한국전력', '에너지'), ('한국가스공사', '에너지'), ('두산퓨얼셀', '에너지'),
            ('에스디바이오센서', '에너지')
        ],
        '반도체장비': [
            ('원익IPS', 'IT'), ('피에스케이', 'IT'), ('주성엔지니어링', 'IT'),
            ('테스', 'IT'), ('에이피티씨', 'IT')
        ],
        '디스플레이': [
            ('LG디스플레이', 'IT'), ('덕산네오룩스', 'IT'), ('동운아나텍', 'IT'),
            ('매크로젠', 'IT')
        ],
        '방산': [
            ('한화에어로스페이스', '방산'), ('LIG넥스원', '방산'), ('한화시스템', '방산'),
            ('현대로템', '방산')
        ]
    }

    # 모든 종목을 하나의 리스트로 변환
    stock_names = []
    for sector, stocks in sector_stocks.items():
        for stock_name, _ in stocks:
            stock_names.append(stock_name)

    # 전략 파라미터 수정을 위한 Backtester 클래스 인스턴스 생성 전 설정
    strategy_params = {
        'momentum_period': 20,
        'rebalance_weekday': 4,  # 금요일
        'num_top_stocks': 5,    # 상위 10종목 선택으로 변경
        'safe_asset_code': 'A439870'  # KODEX 국고채30년 액티브
    }

    stock_codes_for_backtest = []
    backtester_instance = None

    # 종목 코드 확인 및 일봉 데이터 로딩
    for name in stock_names:
        code = creon_api_client.get_stock_code(name)
        if code:
            logging.info(f"'{name}' (코드: {code}) 종목이 백테스트에 포함됩니다.")
            stock_codes_for_backtest.append(code)
            
            # 각 종목에 대해 일봉 데이터만 로드
            logging.info(f"{name} ({code}) 종목의 일봉 데이터를 Creon API에서 가져오는 중... (기간: {daily_data_fetch_start} ~ {backtest_end_date.strftime('%Y%m%d')})")
            daily_df = creon_api_client.get_daily_ohlcv(code, daily_data_fetch_start, backtest_end_date.strftime('%Y%m%d'))
            time.sleep(0.3)  # API 호출 제한 방지를 위한 대기
            
            if daily_df.empty:
                logging.warning(f"{name} ({code}) 종목의 일봉 데이터를 가져올 수 없습니다. 해당 종목을 건너뜁니다.")
                continue
            logging.info(f"{name} ({code}) 종목의 일봉 데이터 로드 완료. 데이터 수: {len(daily_df)}행")
            
            if backtester_instance is None:
                backtester_instance = Backtester(creon_api_client, initial_cash=10_000_000)
                # 전략 파라미터 설정
                backtester_instance.strategy_params.update(strategy_params)
            
            backtester_instance.add_daily_data(code, daily_df)
        else:
            logging.warning(f"'{name}' 종목의 코드를 찾을 수 없습니다. 해당 종목을 건너뜁니다.")

    # 안전자산 데이터 로드
    safe_asset_code = backtester_instance.strategy_params['safe_asset_code']
    logging.info(f"안전자산 ({safe_asset_code})의 일봉 데이터를 Creon API에서 가져오는 중...")
    safe_asset_df = creon_api_client.get_daily_ohlcv(safe_asset_code, daily_data_fetch_start, backtest_end_date.strftime('%Y%m%d'))
    time.sleep(0.3)  # API 호출 제한 방지를 위한 대기
    
    if not safe_asset_df.empty:
        backtester_instance.add_daily_data(safe_asset_code, safe_asset_df)
        logging.info(f"안전자산 데이터 로드 완료. 데이터 수: {len(safe_asset_df)}행")
    else:
        logging.error("안전자산 데이터를 로드할 수 없습니다. 절대 모멘텀 비교에 영향을 미칠 수 있습니다.")

    if not stock_codes_for_backtest:
        logging.error("백테스트를 위한 유효한 종목 코드가 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)
        
    if backtester_instance is None:
        logging.error("백테스터 인스턴스가 생성되지 않았습니다 (어떤 종목도 일봉 데이터가 로드되지 않음). 프로그램을 종료합니다.")
        sys.exit(1)

    # 백테스트 실행
    portfolio_values, metrics = backtester_instance.run(backtest_start_date, backtest_end_date) 