# strategies/dual_momentum_daily_etf.py (수정된 내용)

import datetime
import logging
import pandas as pd
import numpy as np

# from util.utils import calculate_momentum  # 이 임포트는 그대로 유지
from strategies.strategy import DailyStrategy 
# calculate_momentum, calculate_rsi 등 필요한 함수는 utils.py에서 임포트됩니다.
from util.utils import calculate_momentum, calculate_rsi # 명시적으로 임포트

class DualMomentumDaily(DailyStrategy):
    def __init__(self, data_store, strategy_params, broker):
        super().__init__(data_store, strategy_params, broker)
        self.data_store = data_store
        self.signals = {} # {stock_code: {'signal': 'buy'/'sell'/'hold', 'target_quantity': X, 'signal_date': date, 'traded_today': False}}
        self.last_rebalance_date = None

        self.momentum_period = self.strategy_params.get('momentum_period', 120)
        self.rebalance_weekday = self.strategy_params.get('rebalance_weekday', 0)
        self.num_top_stocks = self.strategy_params.get('num_top_stocks', 10)
        self.safe_asset_code = self.strategy_params.get('safe_asset_code')
        self.market_index_code = self.strategy_params.get('market_index_code')
        self.inverse_etf_code = self.strategy_params.get('inverse_etf_code')
        self.risk_free_rate = self.strategy_params.get('risk_free_rate', 0.0)

        self._initialize_signals_for_all_stocks() 
        logging.info(f"DualMomentumDaily 전략 초기화 완료. 모멘텀 기간: {self.momentum_period}일, 리밸런싱 요일: {self.rebalance_weekday} (월=0)")

    def _initialize_signals_for_all_stocks(self):
        for stock_code in self.data_store['daily'].keys():
            if stock_code not in self.signals:
                self.signals[stock_code] = {'signal': 'hold', 'target_quantity': 0, 'signal_date': None, 'traded_today': False}
        for etf_code in [self.safe_asset_code, self.market_index_code, self.inverse_etf_code]:
            if etf_code and etf_code not in self.signals:
                self.signals[etf_code] = {'signal': 'hold', 'target_quantity': 0, 'signal_date': None, 'traded_today': False}
        logging.debug("모든 종목의 초기 시그널 설정 완료.")

    def run_daily_logic(self, current_date):
        """
        일봉 데이터를 기반으로 듀얼 모멘텀 전략을 실행하고 매수/매도 시그널을 생성합니다.
        시그널은 self.signals 딕셔너리에 저장됩니다.
        """
        if current_date.weekday() != self.rebalance_weekday:
            logging.debug(f"[{current_date.isoformat()}] 리밸런싱 요일이 아님. (요일: {current_date.weekday()})")
            return

        if self.last_rebalance_date == current_date:
            logging.debug(f"[{current_date.isoformat()}] 이미 오늘 리밸런싱 완료. 스킵.")
            return

        logging.info(f"[{current_date.isoformat()}] 듀얼 모멘텀 리밸런싱 로직 실행.")
        self.last_rebalance_date = current_date

        # 1. 절대 모멘텀 계산 (시장 지수 ETF 사용)
        # _get_historical_data_up_to는 DataFrame을 반환합니다.
        market_data = self._get_historical_data_up_to('daily', self.market_index_code, current_date, self.momentum_period + 1)
        
        if market_data.empty or len(market_data) < self.momentum_period + 1:
            logging.warning(f"[{current_date.isoformat()}] 시장 지수 ETF ({self.market_index_code}) 데이터 부족. 절대 모멘텀 계산 불가.")
            self._handle_no_momentum_strategy(current_date)
            return

        # 수정된 부분: calculate_momentum에 DataFrame 전체를 전달
        # utils.py의 calculate_momentum은 DataFrame의 'close' 컬럼을 기대합니다.
        market_momentum = calculate_momentum(market_data, self.momentum_period) # <--- 이 부분을 수정
        
        # calculate_momentum이 Series를 반환하고, 그 Series의 마지막 값을 사용해야 합니다.
        # utils.py의 calculate_momentum은 pct_change(period)를 반환하므로, 그 결과 Series의 마지막 값을 사용합니다.
        market_momentum_value = market_momentum.iloc[-1]
        
        is_bull_market = market_momentum_value > self.risk_free_rate 

        # 2. 강세장/약세장 시나리오에 따른 포지션 결정
        if is_bull_market:
            logging.info(f"[{current_date.isoformat()}] 강세장 판단 (시장 모멘텀: {market_momentum_value:.2f}%). 종목 선택 시작.")
            # 2-1. 강세장: 개별 종목 상대 모멘텀 계산
            stock_momentums = {}
            valid_stock_codes = []

            for stock_code in self.data_store['daily'].keys():
                if stock_code in [self.safe_asset_code, self.market_index_code, self.inverse_etf_code]:
                    continue

                stock_data = self._get_historical_data_up_to('daily', stock_code, current_date, self.momentum_period + 1)
                if stock_data.empty or len(stock_data) < self.momentum_period + 1:
                    logging.debug(f"[{current_date.isoformat()}] {stock_code} 데이터 부족. 모멘텀 계산에서 제외.")
                    continue
                
                try:
                    # 수정된 부분: calculate_momentum에 DataFrame 전체를 전달
                    momentum_series = calculate_momentum(stock_data, self.momentum_period)
                    momentum = momentum_series.iloc[-1] # 결과 Series의 마지막 값
                    if momentum > 0:
                        stock_momentums[stock_code] = momentum
                        valid_stock_codes.append(stock_code)
                except Exception as e:
                    logging.warning(f"[{current_date.isoformat()}] {stock_code} 모멘텀 계산 오류: {e}")
                    continue

            selected_stocks = sorted(stock_momentums.items(), key=lambda x: x[1], reverse=True)[:self.num_top_stocks]
            selected_stock_codes = [code for code, mom in selected_stocks]

            logging.info(f"[{current_date.isoformat()}] 강세장 포트폴리오 (상위 {self.num_top_stocks}개): {selected_stock_codes}")

            for stock_code in self.broker.get_positions().keys():
                if stock_code not in selected_stock_codes and stock_code != self.safe_asset_code:
                    self._set_signal(stock_code, 'sell', 0, current_date)
                    logging.info(f"[{current_date.isoformat()}] 매도 시그널 생성 (강세장, 미선택): {stock_code}")
            
            for stock_code in selected_stock_codes:
                if stock_code != self.safe_asset_code:
                    self._set_signal(stock_code, 'buy', None, current_date) 
                    logging.info(f"[{current_date.isoformat()}] 매수 시그널 생성 (강세장, 선택됨): {stock_code}")

            if self.safe_asset_code in self.broker.get_positions():
                self._set_signal(self.safe_asset_code, 'sell', self.broker.get_position_size(self.safe_asset_code), current_date)
                logging.info(f"[{current_date.isoformat()}] 강세장 진입. 안전자산({self.safe_asset_code}) 매도 시그널 생성.")
            else:
                self._set_signal(self.safe_asset_code, 'hold', 0, current_date)

            if self.inverse_etf_code in self.broker.get_positions():
                self._set_signal(self.inverse_etf_code, 'sell', self.broker.get_position_size(self.inverse_etf_code), current_date)
                logging.info(f"[{current_date.isoformat()}] 강세장 진입. 인버스 ETF({self.inverse_etf_code}) 매도 시그널 생성.")
            else:
                self._set_signal(self.inverse_etf_code, 'hold', 0, current_date)

        else: # 약세장
            logging.info(f"[{current_date.isoformat()}] 약세장 판단 (시장 모멘텀: {market_momentum_value:.2f}%). 안전자산/인버스 ETF 비중 확대.")
            for stock_code in self.broker.get_positions().keys():
                if stock_code not in [self.safe_asset_code, self.inverse_etf_code]:
                    self._set_signal(stock_code, 'sell', self.broker.get_position_size(stock_code), current_date)
                    logging.info(f"[{current_date.isoformat()}] 매도 시그널 생성 (약세장, 개별 종목): {stock_code}")
            
            for stock_code in self.data_store['daily'].keys():
                if stock_code not in [self.safe_asset_code, self.market_index_code, self.inverse_etf_code]:
                    self._set_signal(stock_code, 'hold', 0, current_date)

            if market_momentum_value < 0: 
                self._set_signal(self.inverse_etf_code, 'buy', None, current_date) 
                logging.info(f"[{current_date.isoformat()}] 약세장 진입. 인버스 ETF({self.inverse_etf_code}) 매수 시그널 생성.")
                if self.safe_asset_code in self.broker.get_positions():
                    self._set_signal(self.safe_asset_code, 'sell', self.broker.get_position_size(self.safe_asset_code), current_date)
                    logging.info(f"[{current_date.isoformat()}] 약세장 진입. 안전자산({self.safe_asset_code}) 매도 시그널 생성.")
                else:
                    self._set_signal(self.safe_asset_code, 'hold', 0, current_date)
            else: 
                self._set_signal(self.safe_asset_code, 'buy', None, current_date) 
                logging.info(f"[{current_date.isoformat()}] 강세장 조건 미충족. 안전자산({self.safe_asset_code}) 매수 시그널 생성.")
                if self.inverse_etf_code in self.broker.get_positions():
                    self._set_signal(self.inverse_etf_code, 'sell', self.broker.get_position_size(self.inverse_etf_code), current_date)
                    logging.info(f"[{current_date.isoformat()}] 강세장 조건 미충족. 인버스 ETF({self.inverse_etf_code}) 매도 시그널 생성.")
                else:
                    self._set_signal(self.inverse_etf_code, 'hold', 0, current_date)

        logging.info(f"[{current_date.isoformat()}] 일봉 전략 시그널 업데이트 완료.")

    def _handle_no_momentum_strategy(self, current_date):
        """데이터 부족 등으로 모멘텀 계산이 불가능할 때의 대체 전략 (전량 안전자산)"""
        logging.warning(f"[{current_date.isoformat()}] 모멘텀 계산 불가능. 모든 개별 종목 매도 후 안전자산 매수 시그널 생성.")
        for stock_code in self.broker.get_positions().keys():
            if stock_code not in [self.safe_asset_code, self.inverse_etf_code]:
                self._set_signal(stock_code, 'sell', self.broker.get_position_size(stock_code), current_date)

        for stock_code in self.data_store['daily'].keys():
            if stock_code not in [self.safe_asset_code, self.market_index_code, self.inverse_etf_code]:
                self._set_signal(stock_code, 'hold', 0, current_date)

        self._set_signal(self.safe_asset_code, 'buy', None, current_date) 
        logging.info(f"[{current_date.isoformat()}] 안전자산({self.safe_asset_code}) 매수 시그널 생성 (데이터 부족).")

        if self.inverse_etf_code in self.broker.get_positions():
            self._set_signal(self.inverse_etf_code, 'sell', self.broker.get_position_size(self.inverse_etf_code), current_date)
            logging.info(f"[{current_date.isoformat()}] 인버스 ETF({self.inverse_etf_code}) 매도 시그널 생성 (데이터 부족).")
        else:
            self._set_signal(self.inverse_etf_code, 'hold', 0, current_date)

    def _set_signal(self, stock_code, signal_type, target_quantity, signal_date):
        if stock_code not in self.signals: 
            self.signals[stock_code] = {'signal': 'hold', 'target_quantity': 0, 'signal_date': None, 'traded_today': False}

        self.signals[stock_code]['signal'] = signal_type
        self.signals[stock_code]['target_quantity'] = target_quantity
        self.signals[stock_code]['signal_date'] = signal_date
        self.signals[stock_code]['traded_today'] = False