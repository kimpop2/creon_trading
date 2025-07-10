# strategy/trading_strategy.py

import abc # Abstract Base Class
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Any, Optional
import logging
import sys
import os

# 프로젝트 루트 경로를 sys.path에 추가 (manager 디렉토리에서 실행 가능하도록)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.abstract_broker import AbstractBroker
from trading.broker import Broker
# from trading.brokerage import Brokerage # 현재 사용되지 않으므로 주석 처리
from manager.backtest_manager import BacktestManager
# from manager.trading_manager import TradingManager # 현재 사용되지 않으므로 주석 처리
from util.strategies_util import calculate_sma, calculate_rsi, calculate_ema, calculate_macd

# --- 로거 설정 ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 테스트 시 DEBUG로 설정하여 모든 로그 출력 - 제거

class BaseStrategy(abc.ABC):
    """
    모든 자동매매 전략의 기반이 되는 추상 클래스입니다.
    실시간 환경에서 Brokerage와 TradingManager와 연동됩니다.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def run_strategy_logic(self, current_date: date) -> None:
        """
        일봉 데이터를 기반으로 전략 로직을 실행하고 매매 신호를 생성합니다.
        생성된 신호는 self.signals에 저장되고, TradingManager를 통해 DB에 영속화됩니다.
        """
        pass

    @abc.abstractmethod
    def run_trading_logic(self, current_minute_dt: datetime, stock_code: str) -> None:
        """
        분봉 데이터를 기반으로 전략 로직을 실행하고 실시간 매매 주문을 처리합니다.
        (주로 매수 신호가 발생한 종목에 대한 매수/매도 시점 포착)
        """
        pass

class Strategy(BaseStrategy):
    """통합 전략을 위한 추상 클래스."""
    def __init__(self, broker, manager, data_store, strategy_params: Dict[str, Any]):
        super().__init__()
        self.broker = broker
        self.manager = manager # BacktestManager 인스턴스
        self.data_store = data_store
        self.strategy_params = strategy_params
        # 매매 신호 관리: {stock_code: {'signal_type': 'BUY'/'SELL', 'target_price': float, 'target_quantity': int, 'strategy_name': str, 'is_executed': bool, 'signal_id': Optional[int]}}
        self.signals = {}
        self._initialize_signals_for_all_stocks()   # self.signals 초기화

        # 자동매매
        self.subscribed_realtime_stocks: List[str] = [] # 실시간 구독 중인 종목 리스트

    def _initialize_signals_for_all_stocks(self): 
        """모든 종목에 대한 시그널을 초기화합니다.""" 
        # data_store에 있는 종목들 초기화
        for stock_code in self.data_store.get('daily', {}): 
            if stock_code not in self.signals: 
                self.signals[stock_code] = { 
                    'signal': None, 
                    'signal_date': None, 
                    'traded_today': False, 
                    'target_quantity': 0 
                } 

        # broker의 positions에 있는 종목들도 초기화
        for stock_code in self.broker.positions.keys():
            if stock_code not in self.signals:
                self.signals[stock_code] = {
                    'signal': None,
                    'signal_date': None,
                    'traded_today': False,
                    'target_quantity': 0
                }


    def _reset_all_signals(self):
        """모든 신호를 완전히 초기화합니다. (다음날을 위해)"""
        self.signals = {}  # 모든 신호를 완전히 삭제
        logging.debug("일봉 전략의 모든 신호를 완전히 초기화했습니다.")

    ### 전략 파일이 백테스트와 자동매매에 공통으로 사용되어야 하므로,
    ### self.broker 에 따라 다음 처럼 다르게 처리할 수 있다.
    def reset_signal(self, stock_code: str) -> None:
        """
        전략 파일이 
        (예: 주문이 완전히 체결되거나 취소되어 더 이상 유효하지 않은 경우)
        """        
        # Brokerage 대신 Broker 인스턴스를 직접 참조
        if isinstance(self.broker, Broker): # 'Brokerage' 대신 'Broker' 사용
            """매매 체결 후 신호 dict를 안전하게 초기화한다."""
            if stock_code in self.signals:
                self.signals[stock_code]['traded_today'] = True
                self.signals[stock_code]['target_quantity'] = 0
                self.signals[stock_code]['target_price'] = 0
                self.signals[stock_code]['signal'] = None
        # elif isinstance(self.broker, Brokerage): # Brokerage 로직은 현재 사용되지 않으므로 주석 처리
        #     """
        #     특정 종목의 신호를 초기화(삭제)하고 DB에서도 해당 신호를 제거합니다.
        #     (예: 주문이 완전히 체결되거나 취소되어 더 이상 유효하지 않은 경우)
        #     """
        #     if stock_code in self.signals:
        #         signal_id = self.signals[stock_code].get('signal_id')
        #         if signal_id:
        #             success = self.manager.update_daily_signal_status(signal_id, is_executed=True) # 실행 완료로 표시
        #             if success:
        #                 del self.signals[stock_code]
        #                 logger.info(f"신호 초기화 (실행 완료): {stock_code}")
        #             else:
        #                 logger.error(f"신호 상태 DB 업데이트 실패: {stock_code}")
        #         else: # signal_id가 없는 경우 (예: 임시 신호)
        #             del self.signals[stock_code]
        #             logger.info(f"신호 초기화 (메모리만): {stock_code}")
        #     else:
        #         logger.warning(f"초기화할 신호가 없습니다: {stock_code}")


    def update_signals(self, signals):
        """
        전략에서 생성된 신호들을 업데이트합니다.
        """
        self.signals = {
            stock_code: {**info, 'traded_today': False}
            for stock_code, info in signals.items() #### info 에 set() 아이템 설정
        }


    def _generate_signals(self, current_daily_date, buy_candidates, sorted_stocks, sell_candidates=None):
        """매수/매도/홀딩 신호를 생성하고 업데이트합니다."""
        current_positions = set(self.broker.positions.keys())
        
        # sell_candidates가 None이면 빈 set으로 초기화
        if sell_candidates is None:
            sell_candidates = set()

        # 1. 매수 후보 종목들 처리
        for stock_code, _ in sorted_stocks:
            # 종목이 signals에 초기화되지 않았다면 초기화
            if stock_code not in self.signals:
                self.signals[stock_code] = {
                    'signal': None,
                    'signal_date': None,
                    'traded_today': False,
                    'target_quantity': 0
                }
            
            # 기본 정보 업데이트
            self.signals[stock_code].update({
                'signal_date': current_daily_date,
                'traded_today': False
            })

            if stock_code in buy_candidates:
                self._handle_buy_candidate(stock_code, current_daily_date)
            else:
                # 매수 후보가 아니지만 보유 중인 종목은 홀딩
                if stock_code in current_positions:
                    self._handle_hold_candidate(stock_code, current_daily_date)

        # 2. 매도 후보 종목들 처리 (새로 추가)
        for stock_code in sell_candidates:
            # 종목이 signals에 초기화되지 않았다면 초기화
            if stock_code not in self.signals:
                self.signals[stock_code] = {
                    'signal': None,
                    'signal_date': None,
                    'traded_today': False,
                    'target_quantity': 0
                }
            
            # 기본 정보 업데이트
            self.signals[stock_code].update({
                'signal_date': current_daily_date,
                'traded_today': False
            })
            
            # 매도 신호 생성
            self._handle_sell_candidate(stock_code, current_positions)

        # 3. 보유 중이지만 매수/매도 후보가 아닌 종목들 처리 (홀딩)
        for stock_code in current_positions:
            if stock_code not in buy_candidates and stock_code not in sell_candidates:
                # 종목이 signals에 초기화되지 않았다면 초기화
                if stock_code not in self.signals:
                    self.signals[stock_code] = {
                        'signal': None,
                        'signal_date': None,
                        'traded_today': False,
                        'target_quantity': 0
                    }
                
                # 기본 정보 업데이트
                self.signals[stock_code].update({
                    'signal_date': current_daily_date,
                    'traded_today': False
                })
                
                self._handle_hold_candidate(stock_code, current_daily_date)

        return current_positions

    def _handle_buy_candidate(self, stock_code, current_daily_date):
        """매수 대상 종목에 대한 신호를 처리합니다."""
        # 종목이 signals에 초기화되지 않았다면 초기화
        if stock_code not in self.signals:
            self.signals[stock_code] = {
                'signal': None,
                'signal_date': None,
                'traded_today': False,
                'target_quantity': 0
            }
            
        # 종가를 현재가로 사용
        # current_price_daily를 가져올 때 유효성 검사를 강화
        daily_data = self._get_historical_data_up_to('daily', stock_code, current_daily_date, lookback_period=1)
        if daily_data.empty or 'close' not in daily_data.columns or daily_data['close'].empty:
            logging.warning(f"[{current_daily_date}] {stock_code}: 일봉 종가 데이터를 가져올 수 없어 매수 신호를 생성할 수 없습니다.")
            # 유효한 가격이 없으므로 target_price를 설정하지 않거나 0으로 설정하여 이후 로직에서 처리
            current_price_daily = 0.0 # 또는 None
        else:
            current_price_daily = daily_data['close'].iloc[-1]

        target_quantity = self._calculate_target_quantity(stock_code, current_price_daily)

        if target_quantity > 0:
            if stock_code in self.broker.positions:
                self.signals[stock_code]['signal'] = 'hold'
                logging.info(f'홀딩 신호 - {stock_code}: (기존 보유 종목)')
            else:
                self.signals[stock_code].update({
                    'signal': 'buy',
                    'target_quantity': target_quantity,
                    'target_price': current_price_daily  # 목표가격 추가 (전일 종가)
                })
                logging.info(f'매수 신호 - {stock_code}: 목표수량 {target_quantity}주, 목표가격 {current_price_daily:,.0f}원 (전일 종가)')


    def _handle_hold_candidate(self, stock_code, current_daily_date):
        """홀딩 대상 종목에 대한 신호를 처리합니다."""
        # 종목이 signals에 초기화되지 않았다면 초기화
        if stock_code not in self.signals:
            self.signals[stock_code] = {
                'signal': None,
                'signal_date': None,
                'traded_today': False,
                'target_quantity': 0
            }
        
        # 수정: signal_date가 None이면 current_daily_date 사용
        signal_date = self.signals[stock_code].get('signal_date')
        if signal_date is None:
            signal_date = current_daily_date
        
        # 수정: 전일 종가를 목표가격으로 설정 (장전 판단을 위해)
        daily_data = self._get_historical_data_up_to('daily', stock_code, signal_date, lookback_period=1)
        if daily_data.empty or 'close' not in daily_data.columns or daily_data['close'].empty:
            logging.warning(f"[{current_daily_date}] {stock_code}: 일봉 종가 데이터를 가져올 수 없어 홀딩 신호의 목표 가격을 설정할 수 없습니다.")
            current_price_daily = 0.0 # 또는 적절한 기본값
        else:
            current_price_daily = daily_data['close'].iloc[-1]
        
        # 홀딩 신호 설정
        self.signals[stock_code].update({
            'signal': 'hold',
            'signal_date': current_daily_date,
            'target_price': current_price_daily,
            'target_quantity': self.broker.positions.get(stock_code, {}).get('size', 0)
        })
        
        logging.info(f'홀딩 신호 - {stock_code}: 목표수량 {self.signals[stock_code]["target_quantity"]}주, 목표가격 {current_price_daily:,.0f}원 (전일 종가)')

    def _handle_sell_candidate(self, stock_code, current_positions):
        """매도 대상 종목에 대한 신호를 처리합니다."""
        # 종목이 signals에 초기화되지 않았다면 초기화
        if stock_code not in self.signals:
            self.signals[stock_code] = {
                'signal': None,
                'signal_date': None,
                'traded_today': False,
                'target_quantity': 0
            }
        
        # 수정: signal_date가 None이면 current_daily_date 사용
        signal_date = self.signals[stock_code].get('signal_date')
        if signal_date is None:
            # signal_date가 None이면 현재 날짜를 사용 (임시 처리)
            # 실제로는 이 부분이 호출되면 안 되지만, 안전성을 위해 추가
            logging.warning(f"{stock_code}: signal_date가 None입니다. 매도 신호 생성을 건너킵니다.")
            return
        
        # 수정: 전일 종가를 목표가격으로 설정 (장전 판단을 위해)
        daily_data = self._get_historical_data_up_to('daily', stock_code, signal_date, lookback_period=1)
        if daily_data.empty or 'close' not in daily_data.columns or daily_data['close'].empty:
            logging.warning(f"[{signal_date}] {stock_code}: 일봉 종가 데이터를 가져올 수 없어 매도 신호의 목표 가격을 설정할 수 없습니다.")
            current_price_daily = 0.0 # 또는 적절한 기본값
        else:
            current_price_daily = daily_data['close'].iloc[-1]
            
        self.signals[stock_code].update({
            'signal': 'sell',
            'target_price': current_price_daily  # 목표가격 추가 (전일 종가)
        })
        
        if stock_code in current_positions:
            logging.info(f'매도 신호 - {stock_code} (보유중): 목표가격 {current_price_daily:,.0f}원 (전일 종가)')
        else:
            logging.debug(f'매도 신호 - {stock_code} (미보유): 목표가격 {current_price_daily:,.0f}원 (전일 종가)')


    # 모멘텀 전략으로 보낼 것 -> 관심종목 필터링으로 용도변경경
    def _select_buy_candidates(self, momentum_scores, safe_asset_momentum):
        """모멘텀 스코어를 기반으로 매수 대상 종목을 선정합니다."""
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = set()
        
        for rank, (stock_code, score) in enumerate(sorted_stocks, 1):
            if rank <= self.strategy_params['num_top_stocks'] and score > safe_asset_momentum:
                buy_candidates.add(stock_code)

        return buy_candidates, sorted_stocks
    

    def _get_bar_at_time(self, data_type: str, stock_code: str, target_dt: datetime) -> Optional[Dict[str, Any]]:
        """
        주어진 시간(target_dt)에 해당하는 정확한 OHLCV 바를 TradingManager에서 가져와 반환합니다.
        """
        if data_type == 'daily':
            # 일봉 데이터는 TradingManager에서 특정 날짜의 데이터를 요청
            df = self.manager.fetch_daily_ohlcv(stock_code, target_dt.date(), target_dt.date())
            if not df.empty:
                # 인덱스를 datetime으로 변경하여 정확한 날짜 매칭
                df.index = pd.to_datetime(df['date'])
                if target_dt.date() in df.index.date:
                    # 해당 날짜의 데이터 (첫 번째 행만 가져옴)
                    row = df[df['date'] == target_dt.date()].iloc[0]
                    return row.to_dict()
            return None

        elif data_type == 'minute':
            # 분봉 데이터는 TradingManager에서 특정 시간의 데이터를 요청
            # backtest.py의 data_store['minute'][stock_code]는 이제 하나의 DataFrame을 가집니다.
            # 이 DataFrame은 여러 날짜의 데이터를 포함할 수 있습니다.
            df = self.data_store['minute'].get(stock_code)
            if df is not None and not df.empty:
                # 정확히 해당 datetime과 일치하는 행을 인덱스를 사용하여 찾음
                # 인덱스가 DatetimeIndex이고, target_dt가 datetime 객체이므로 직접 비교 가능
                if target_dt in df.index:
                    row = df.loc[target_dt]
                    return row.to_dict()
                else:
                    logger.warning(f"[{target_dt}] {stock_code}: 해당 시각의 분봉 데이터가 인덱스에 없습니다. (가장 가까운 시간 찾기 필요 시 로직 추가)")
                    return None
            return None

        else:
            logger.error(f"알 수 없는 데이터 타입: {data_type}")
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
            # data_store['minute'][stock_code]는 이제 하나의 DataFrame을 가집니다.
            df = self.data_store['minute'].get(stock_code)
            if df is None or df.empty:
                return pd.DataFrame()
            
            # 현재 시각(current_dt)까지의 데이터만 필터링
            filtered_df = df.loc[df.index <= current_dt]
            if lookback_period:
                return filtered_df.tail(lookback_period)
            return filtered_df
        return pd.DataFrame()


    def _calculate_target_quantity(self, stock_code, current_price, num_stocks=None):
        """
        주어진 가격에서 동일비중 투자에 필요한 수량을 계산합니다.
        
        Args:
            stock_code (str): 종목 코드
            current_price (float): 현재 가격
            num_stocks (int, optional): 분배할 종목 수. None인 경우 strategy_params['num_top_stocks'] 사용
            
        Returns:
            int: 매수 가능한 주식 수량
        """
        # current_price가 0이면 계산 불가
        if current_price <= 0:
            logging.warning(f"{stock_code} 종목 매수 수량 계산 불가: 현재 가격이 0 이하입니다.")
            return 0

        # 분배할 종목 수 결정
        if num_stocks is None:
            num_stocks = self.strategy_params.get('num_top_stocks', 1)
        # 포트폴리오 가치 기준 종목당 투자금 계산
        # 현재가 정보 수집 (일봉 데이터 기준)
        current_prices_for_summary = {}
        for code in self.data_store['daily']:
            daily_data = self._get_historical_data_up_to('daily', code, pd.Timestamp.today(), lookback_period=1)
            if not daily_data.empty:
                current_prices_for_summary[code] = daily_data['close'].iloc[-1]

        portfolio_value = self.broker.get_portfolio_value(current_prices_for_summary)
        per_stock_investment = portfolio_value / num_stocks
        available_cash = self.broker.initial_cash
        commission_rate = self.broker.commission_rate
        max_buyable_amount = available_cash / (1 + commission_rate)
        actual_investment_amount = min(per_stock_investment, max_buyable_amount)
        quantity = int(actual_investment_amount / current_price)
        if quantity > 0:
            total_cost = current_price * quantity * (1 + commission_rate)
            if total_cost > available_cash:
                quantity -= 1
        if quantity > 0:
            logging.info(f"{stock_code} 종목 매수 수량 계산: {quantity}주 (종목당 투자금: {per_stock_investment:,.0f}원, 가용현금: {available_cash:,.0f}원, 현재가: {current_price:,.0f}원)")
        else:
            logging.warning(f"{stock_code} 종목 매수 불가: 현금 부족 (종목당 투자금: {per_stock_investment:,.0f}원, 가용현금: {available_cash:,.0f}원, 현재가: {current_price:,.0f}원)")
        
        return quantity

    def _log_rebalancing_summary(self, current_daily_date, buy_candidates, current_positions, sell_candidates=None):
        """리밸런싱 요약을 로깅합니다."""
        if sell_candidates is None:
            sell_candidates = set()
            
        logging.info(f'[{current_daily_date}] === 리밸런싱 요약 ===')
        logging.info(f'매수 후보: {len(buy_candidates)}개 - {sorted(buy_candidates)}')
        logging.info(f'매도 후보: {len(sell_candidates)}개 - {sorted(sell_candidates)}')
        logging.info(f'현재 보유: {len(current_positions)}개 - {sorted(current_positions)}')
        
        # 매수 신호 개수
        buy_signals = sum(1 for signal in self.signals.values() if signal.get('signal') == 'buy')
        # 매도 신호 개수
        sell_signals = sum(1 for signal in self.signals.values() if signal.get('signal') == 'sell')
        # 홀딩 신호 개수
        hold_signals = sum(1 for signal in self.signals.values() if signal.get('signal') == 'hold')
        
        logging.info(f'생성된 신호 - 매수: {buy_signals}개, 매도: {sell_signals}개, 홀딩: {hold_signals}개')
        logging.info(f'=== 리밸런싱 요약 완료 ===')


    def execute_time_cut_buy(self, stock_code, current_dt, current_price, target_quantity, max_deviation_ratio):
        """
        타임컷 강제매수를 실행합니다.
        
        Args:
            stock_code (str): 종목 코드
            current_dt (datetime): 현재 시각
            current_price (float): 현재 가격
            target_quantity (int): 매수 수량
            max_price_diff_ratio (float): 최대 허용 괴리율 (기본값: 0.01 = 1%)
            
        Returns:
            bool: 매수 실행 여부
        """
        if stock_code not in self.signals:
            return False
            
        target_price = self.signals[stock_code].get('target_price', current_price)
        
        # target_price가 0이거나 유효하지 않은 경우 매매를 건너뜁니다.
        if target_price is None or target_price <= 0:
            logging.warning(f'[타임컷 매수] {current_dt.isoformat()} - {stock_code}: 목표 가격이 0이거나 유효하지 않아 매매를 건너뜁니다.')
            return False

        price_diff_ratio = abs(target_price - current_price) * 100 / target_price
        
        if price_diff_ratio <= max_deviation_ratio:
            logging.info(f'[타임컷 강제매수] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:.2f}, 매수가: {current_price:.2f}, 괴리율: {price_diff_ratio:.2%}')
            self.broker.execute_order(stock_code, 'buy', current_price, target_quantity, current_dt)
            self.reset_signal(stock_code)
            return True
        else:
            logging.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:.2f}, 매수가: {current_price:.2f}, 괴리율: {price_diff_ratio:.2%} ({max_deviation_ratio:.1%} 초과)')
            return False

    def execute_time_cut_sell(self, stock_code, current_dt, current_price, current_position_size, max_deviation_ratio):
        """
        타임컷 강제매도를 실행합니다.
        
        Args:
            stock_code (str): 종목 코드
            current_dt (datetime): 현재 시각
            current_price (float): 현재 가격
            current_position_size (int): 현재 보유 수량
            max_price_diff_ratio (float): 최대 허용 괴리율 (기본값: 0.01 = 1%)
            
        Returns:
            bool: 매도 실행 여부
        """
        if stock_code not in self.signals:
            return False
            
        target_price = self.signals[stock_code].get('target_price', current_price)
        
        # target_price가 0이거나 유효하지 않은 경우 매매를 건너뜁니다.
        if target_price is None or target_price <= 0:
            logging.warning(f'[타임컷 매도] {current_dt.isoformat()} - {stock_code}: 목표 가격이 0이거나 유효하지 않아 매매를 건너뜁니다.')
            return False

        price_diff_ratio = abs(target_price - current_price) * 100 / target_price 
        
        if price_diff_ratio <= max_deviation_ratio:
            logging.info(f'[타임컷 강제매도] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:.2f}, 매도가: {current_price:.2f}, 괴리율: {price_diff_ratio:.2%}')
            self.broker.execute_order(stock_code, 'sell', current_price, current_position_size, current_dt)
            self.reset_signal(stock_code)
            return True
        else:
            logging.info(f'[타임컷 미체결] {current_dt.isoformat()} - {stock_code} 목표가: {target_price:.2f}, 매도가: {current_price:.2f}, 괴리율: {price_diff_ratio:.2%} ({max_deviation_ratio:.1%} 초과)')
            return False

