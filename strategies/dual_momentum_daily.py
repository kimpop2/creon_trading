import datetime
import logging
import pandas as pd
import numpy as np

# from util.utils import calculate_momentum, get_next_weekday # 주석 처리 또는 제거 (이미 utils.py에서 임포트)
from util.utils import * # utils.py에 있는 모든 함수를 임포트한다고 가정
# DailyStrategy 추상 클래스 임포트
from strategies.strategy_base import DailyStrategy

class DualMomentumDaily(DailyStrategy): # DailyStrategy 상속
    def __init__(self, data_store, strategy_params, broker):
        # DailyStrategy의 __init__ 호출하여 data_store, strategy_params, broker, position_info 초기화
        # DualMomentumDaily는 position_info를 직접 사용하지 않지만, BaseStrategy의 인터페이스를 맞추기 위해 전달
        super().__init__(data_store, strategy_params, broker) 
        
        self.momentum_period = strategy_params['momentum_period']
        self.rebalance_weekday = strategy_params['rebalance_weekday']
        self.num_top_stocks = strategy_params['num_top_stocks']
        self.safe_asset_code = strategy_params['safe_asset_code']
        #self.equal_weight_amount = strategy_params['equal_weight_amount']
        
        # momentum_signals는 이 전략 클래스 내부에서만 관리하며, Backtester를 통해 MinuteStrategy로 전달됨
        self.momentum_signals = {} # {stock_code: {'signal': 'buy'/'sell'/'hold', 'signal_date': date, 'traded_today': False, 'current_price': float}}
        
        # 모든 종목에 대한 모멘텀 시그널을 미리 초기화 (백테스터에서 일봉 데이터를 추가할 때 호출됨)
        # 이 부분은 Backtester에서 add_daily_data 호출 시 DualMomentumDaily 내부에서 직접 호출되므로,
        # 초기화 시점에 모든 종목이 다 로드되지 않았을 수 있어 주의 필요.
        # 따라서 Backtester의 add_daily_data에서 이 메서드를 호출하는 것이 더 적절.
        # 여기서는 __init__에서 불필요하게 호출되지 않도록 주석 처리
        # self._initialize_momentum_signals_for_all_stocks()


    def _initialize_momentum_signals_for_all_stocks(self):
        """
        모든 종목에 대해 초기 모멘텀 시그널 정보를 설정합니다.
        이는 data_store에 새로운 종목이 추가될 때마다 호출되어야 합니다.
        """
        for stock_code in self.data_store['daily'].keys():
            if stock_code not in self.momentum_signals:
                self.momentum_signals[stock_code] = {
                    'signal': 'hold',
                    'signal_date': None,
                    'traded_today': False, # 당일 해당 종목의 매매 발생 여부
                    'current_price': 0 # 현재 가격 (모멘텀 계산용)
                }
        logging.debug(f"DualMomentumDaily: 모든 종목의 모멘텀 시그널 초기화 완료. 종목 수: {len(self.momentum_signals)}")


    def run_weekly_momentum_logic(self, current_daily_date):
        """
        기존 주간 모멘텀 로직을 실행하는 메서드.
        주어진 날짜가 리밸런싱 요일(금요일)이고, 리밸런싱이 아직 이뤄지지 않았다면 모멘텀 계산 및 포트폴리오 재조정 시도.
        """
        # 매일 시작 시 'traded_today' 플래그 초기화
        for stock_code in self.momentum_signals:
            self.momentum_signals[stock_code]['traded_today'] = False

        if current_daily_date.weekday() == self.rebalance_weekday:
            # 해당 날짜가 리밸런싱 요일(금요일)인 경우에만 로직 실행
            
            logging.info(f"[{current_daily_date.isoformat()}] 리밸런싱 요일: 듀얼 모멘텀 전략 실행 준비 중...")
            
            # 1. 모든 종목의 모멘텀 스코어 계산
            momentum_scores = {}
            for stock_code, daily_df in self.data_store['daily'].items():
                # 현재 날짜까지의 데이터만 사용 (미래 데이터 방지)
                past_data = daily_df.loc[daily_df.index.normalize() <= pd.Timestamp(current_daily_date).normalize()]
                
                # 모멘텀 계산을 위한 충분한 데이터가 있는지 확인
                if len(past_data) >= self.momentum_period:
                    # 최신 close 가격 업데이트
                    latest_close = past_data['close'].iloc[-1] if not past_data.empty else 0
                    if stock_code in self.momentum_signals:
                        self.momentum_signals[stock_code]['current_price'] = latest_close

                    momentum = calculate_momentum(past_data, self.momentum_period).iloc[-1]
                    momentum_scores[stock_code] = momentum
                else:
                    logging.debug(f"[{current_daily_date.isoformat()}] {stock_code}: 모멘텀 계산을 위한 충분한 데이터 부족. (현재: {len(past_data)}행, 필요: {self.momentum_period}행)")
            
            # 2. 상위 N개 종목 선정
            if not momentum_scores:
                logging.warning(f"[{current_daily_date.isoformat()}] 모멘텀 스코어를 계산할 수 있는 종목이 없습니다. 포트폴리오 조정 건너뜜.")
                # 시그널을 모두 'hold'로 초기화 (현재 보유 종목이 있다면 계속 보유)
                for stock_code in self.momentum_signals:
                    self.momentum_signals[stock_code]['signal'] = 'hold'
                return

            sorted_momentum_stocks = sorted(momentum_scores.items(), key=lambda item: item[1], reverse=True)
            top_n_stocks = [stock for stock, score in sorted_momentum_stocks if score > 0][:self.num_top_stocks]
            
            logging.info(f"[{current_daily_date.isoformat()}] 상위 {self.num_top_stocks}개 모멘텀 종목 (모멘텀 > 0): {[s[0] for s in sorted_momentum_stocks if s[1] > 0][:self.num_top_stocks]}")

            # 현재 보유 종목 리스트 (매도 시그널을 위해 필요)
            current_holdings = list(self.broker.positions.keys())
            
            # 3. 매도 시그널 생성 (보유 종목 중 선정되지 않은 종목)
            for stock_code in current_holdings:
                if stock_code not in top_n_stocks and stock_code != self.safe_asset_code:
                    if self.broker.get_position_size(stock_code) > 0: # 실제 보유 중인 경우에만 매도 시그널
                        self.momentum_signals[stock_code]['signal'] = 'sell'
                        self.momentum_signals[stock_code]['signal_date'] = current_daily_date 
                        logging.info(f"[{current_daily_date.isoformat()}] 매도 시그널 발생: {stock_code} (모멘텀 하락 또는 미선정)")
                    else:
                        self.momentum_signals[stock_code]['signal'] = 'hold' # 이미 보유하지 않으면 hold
                elif stock_code == self.safe_asset_code and stock_code not in top_n_stocks:
                    # 안전자산이 상위 종목에 없고 보유 중이라면 매도 시그널
                    if self.broker.get_position_size(stock_code) > 0:
                        self.momentum_signals[stock_code]['signal'] = 'sell'
                        self.momentum_signals[stock_code]['signal_date'] = current_daily_date 
                        logging.info(f"[{current_daily_date.isoformat()}] 매도 시그널 발생: {stock_code} (안전자산 포지션 정리)")
                    else:
                        self.momentum_signals[stock_code]['signal'] = 'hold' # 이미 보유하지 않으면 hold
                else: # 상위 N 종목에 포함되거나 안전자산이 포함되어 있다면 hold
                     self.momentum_signals[stock_code]['signal'] = 'hold' # 현재 시점에 매매는 없음

            # 4. 매수 시그널 생성 (선정된 종목 중 미보유 종목)
            if top_n_stocks:
                for stock_code in top_n_stocks:
                    if self.broker.get_position_size(stock_code) == 0: # 현재 보유하고 있지 않은 경우에만 매수 시그널
                        self.momentum_signals[stock_code]['signal'] = 'buy'
                        self.momentum_signals[stock_code]['signal_date'] = current_daily_date
                        logging.info(f"[{current_daily_date.isoformat()}] 매수 시그널 발생: {stock_code}")
                    elif stock_code in current_holdings: # 이미 보유 중인 경우 (재조정 불필요 시)
                        self.momentum_signals[stock_code]['signal'] = 'hold' # 현재 시점에 매매는 없음
            else: # 상위 N개 종목이 없다면, 모든 포지션을 정리하고 안전자산으로 이동
                logging.info(f"[{current_daily_date.isoformat()}] 모멘텀 종목이 없어 안전자산으로 전환합니다.")
                # 모든 보유 종목 매도 시그널
                for stock_code in current_holdings:
                    if self.broker.get_position_size(stock_code) > 0:
                        self.momentum_signals[stock_code]['signal'] = 'sell'
                        self.momentum_signals[stock_code]['signal_date'] = current_daily_date
                        logging.info(f"[{current_daily_date.isoformat()}] 매도 시그널 발생 (안전자산 전환): {stock_code}")
                
                # 안전자산 매수 시그널 (현재 보유 중이 아니라면)
                if self.broker.get_position_size(self.safe_asset_code) == 0:
                    self.momentum_signals[self.safe_asset_code]['signal'] = 'buy'
                    self.momentum_signals[self.safe_asset_code]['signal_date'] = current_daily_date
                    logging.info(f"[{current_daily_date.isoformat()}] 매수 시그널 발생: 안전자산 ({self.safe_asset_code})")
                else: # 이미 보유 중이라면 hold
                    self.momentum_signals[self.safe_asset_code]['signal'] = 'hold'

        else: # 리밸런싱 요일이 아닌 경우 모든 시그널 'hold'로 유지
            for stock_code in self.momentum_signals:
                # 기존 'buy' 또는 'sell' 시그널이 다음 거래일까지 유효하게 유지될 수 있도록 (분봉 전략이 처리하도록)
                # 'traded_today'만 초기화하고 'signal'은 유지하거나,
                # 아니면 매일 'hold'로 초기화하고 시그널이 필요할 때만 업데이트하는 방식을 고려해야 함.
                # 현재 로직에서는 매일 초기화하는 것이 아니라, 시그널이 발생할 때만 업데이트되므로 이 부분은 그대로 유지
                # self.momentum_signals[stock_code]['signal'] = 'hold' # 이 부분을 주석 처리하여 시그널 유지
                self.momentum_signals[stock_code]['traded_today'] = False # 매일 초기화


    # DailyStrategy의 추상 메서드인 run_daily_logic을 구현합니다.
    # 기존 run_weekly_momentum_logic의 이름을 그대로 사용하므로, 래퍼 역할을 합니다.
    def run_daily_logic(self, current_daily_date):
        """
        DailyStrategy의 추상 메서드를 구현합니다.
        기존의 run_weekly_momentum_logic을 호출하여 일봉 전략 로직을 실행합니다.
        """
        self.run_weekly_momentum_logic(current_daily_date)

    # DailyStrategy의 추상 메서드이지만 DailyStrategy에서는 필요 없는 메서드 구현
    def run_minute_logic(self, stock_code, current_minute_dt):
        """DailyStrategy는 분봉 로직을 직접 수행하지 않습니다."""
        pass

    # DailyStrategy의 추상 메서드이지만 DailyStrategy에서는 필요 없는 메서드 구현
    def update_momentum_signals(self, momentum_signals):
        """DailyStrategy는 외부에서 모멘텀 시그널을 업데이트 받지 않습니다."""
        pass