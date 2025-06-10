import datetime
import logging
import pandas as pd
import numpy as np
import time

from backtest.broker import Broker
from util.utils import calculate_performance_metrics, get_next_weekday 
# 전략 추상 클래스 임포트
from strategies.strategy_base import DailyStrategy, MinuteStrategy 

class Backtester:
    def __init__(self, api_client, initial_cash):
        self.api_client = api_client
        self.broker = Broker(initial_cash, commission_rate=0.0003) # 수수료 0.03%
        self.data_store = {'daily': {}, 'minute': {}} # {stock_code: DataFrame}
        self.position_info = {} # {stock_code: {'highest_price': float, 'entry_date': datetime.date}}
        self.portfolio_values = [] # (datetime, value) 튜플 저장
        self.initial_cash = initial_cash

        self.strategy_params = {
            'momentum_period': 60, # 60일 모멘텀
            'rebalance_weekday': 4, # 0:월, 4:금 (주간 리밸런싱)
            'num_top_stocks': 5, # 상위 5개 종목
            'safe_asset_code': 'A439870',  # 안전자산 코드 (국고채 ETF) # KODEX 200 (코스피200 ETF) 또는 KODEX 인버스 (경기 침체 시)
            #'equal_weight_amount': 2_000_000, # 종목당 2백만원 균등 배분
            'minute_rsi_period': 14, # 분봉 RSI 기간
            'minute_rsi_overbought': 70, # 과매수 기준
            'minute_rsi_oversold': 30, # 과매도 기준
            'stop_loss_ratio': -5, # 일반 손절 -5%
            'trailing_stop_ratio': -3, # 트레일링 스탑 -3% (고점 대비)
            'early_stop_loss': -2, # 매수 후 5거래일 이내 -2% 손절
            'max_losing_positions': 3, # 최대 손실 허용 포지션 개수 (미사용)
            'initial_cash': initial_cash # RSI 전략에서 포트폴리오 손절 계산을 위함 (현재는 사용 안함)
        }
        
        # 단일 전략 인스턴스를 저장하도록 변경
        self.daily_strategy: DailyStrategy = None
        self.minute_strategy: MinuteStrategy = None

        logging.info(f"백테스터 초기화 완료. 초기 현금: {self.initial_cash:,.0f}원")


    def set_strategies(self, daily_strategy: DailyStrategy = None, minute_strategy: MinuteStrategy = None):
        """
        백테스팅에 사용할 일봉 및 분봉 전략 인스턴스를 설정합니다.
        각 전략 인스턴스는 이미 data_store, broker 등을 주입받은 상태여야 합니다.
        """
        if daily_strategy:
            if not isinstance(daily_strategy, DailyStrategy):
                raise TypeError("daily_strategy는 DailyStrategy 타입의 인스턴스여야 합니다.")
            self.daily_strategy = daily_strategy
            logging.info(f"일봉 전략 '{daily_strategy.__class__.__name__}'이(가) 설정되었습니다.")

        if minute_strategy:
            if not isinstance(minute_strategy, MinuteStrategy):
                raise TypeError("minute_strategy는 MinuteStrategy 타입의 인스턴스여야 합니다.")
            self.minute_strategy = minute_strategy
            logging.info(f"분봉 전략 '{minute_strategy.__class__.__name__}'이(가) 설정되었습니다.")

        if not self.daily_strategy and not self.minute_strategy:
            logging.warning("설정된 일봉 또는 분봉 전략이 없습니다. 백테스트가 제대로 동작하지 않을 수 있습니다.")


    def add_daily_data(self, stock_code, daily_df):
        """백테스터에 종목별 일봉 데이터를 추가합니다."""
        self.data_store['daily'][stock_code] = daily_df
        # 새로운 종목이 추가될 때마다 모멘텀 시그널도 초기화 (DualMomentumDaily 내부에서 처리)
        self.daily_strategy._initialize_momentum_signals_for_all_stocks()

    def get_next_business_day(self, date):
        """일봉 데이터를 기반으로 다음 거래일을 찾습니다."""
        next_day = date + datetime.timedelta(days=1)
        max_attempts = 10 # 최대 10일까지 다음 거래일을 찾아봄 (주말, 공휴일 등 고려)
        
        while max_attempts > 0:
            has_data = False
            # 모든 종목의 일봉 데이터를 확인하여 해당 날짜가 거래일인지 판단
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
            max_attempts -= 1
        
        logging.warning(f"{date.strftime('%Y-%m-%d')} 이후 {10}일 이내에 거래일을 찾을 수 없습니다.")
        return None

    def _get_minute_data_for_signal_dates(self, stock_code, signal_date):
        # """
        # 일봉 전략 시그널 발생일의 다음 거래일 분봉 데이터를 미리 로드합니다.
        # trade_date에 해당하는 분봉 데이터만 로드하여 self.data_store['minute']에 저장.
        # signal_dates_and_codes: {stock_code: [signal_date1, signal_date2, ...]}
        # """
        """
        매수/매도 시그널이 발생한 날짜와 다음 거래일의 분봉 데이터를 조회합니다.
        크레온 API 호출을 통해 데이터를 가져와 data_store에 저장하고 반환합니다.
        """
        # 다음 거래일 찾기
        next_trading_day = self.get_next_business_day(signal_date)
        if next_trading_day is None:
            logging.warning(f"{signal_date} 이후의 다음 거래일을 찾을 수 없습니다 - {stock_code}")
            return pd.DataFrame()
            
        # 시그널 발생일과 다음 거래일의 분봉 데이터 로드
        # 주의: 여기서는 실제 백테스팅 시점에 필요한 날짜의 분봉 데이터를 불러옵니다.
        # 실제 환경에서는 실시간 데이터 피드를 사용해야 합니다.
        dates_to_load = [signal_date, next_trading_day]
        
        dfs_to_concat = []
        for date in dates_to_load:
            date_str = date.strftime('%Y%m%d')
            
            # 해당 날짜의 분봉 데이터가 이미 있는지 확인
            if stock_code in self.data_store['minute'] and date in self.data_store['minute'][stock_code]:
                dfs_to_concat.append(self.data_store['minute'][stock_code][date])
                continue
            
            # 해당 날짜가 거래일인지 확인
            daily_df = self.data_store['daily'].get(stock_code)
            if daily_df is not None and not daily_df.empty and pd.Timestamp(date).normalize() in daily_df.index:
                minute_df_day = self.api_client.get_minute_ohlcv(stock_code, date_str, date_str, interval=1)
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

    def run(self, start_date, end_date):
        # 포트폴리오 가치 기록을 위한 리스트
        portfolio_values = []
        dates = []
        
        # 모든 종목의 일봉 데이터를 합쳐서 전체 거래일 목록을 생성
        all_daily_dates = pd.DatetimeIndex([])
        for stock_code, daily_df in self.data_store['daily'].items():
            if not daily_df.empty:
                all_daily_dates = all_daily_dates.union(pd.DatetimeIndex(daily_df.index).normalize())

        # 백테스트 기간 내의 거래일만 필터링하고 정렬
        daily_dates_to_process = all_daily_dates[
            (all_daily_dates >= pd.Timestamp( start_date).normalize()) & \
            (all_daily_dates <= pd.Timestamp(end_date).normalize())
        ].sort_values()

        if daily_dates_to_process.empty:
            logging.error("지정된 백테스트 기간 내에 일봉 데이터가 없습니다. 종료합니다.")
            return pd.Series(), {} # 빈 Series와 dict 반환

        # 백테스트 진행 루프 (매일 단위로 진행)
        for current_daily_date_full in daily_dates_to_process:
            current_daily_date = current_daily_date_full.date()
            logging.info(f"\n--- 처리 중인 날짜: {current_daily_date.isoformat()} ---")

            # 매일 시작 시 모든 종목의 'traded_today' 플래그 초기화
            for stock_code in self.daily_strategy.momentum_signals:
                self.daily_strategy.momentum_signals[stock_code]['traded_today'] = False

            # 주간 모멘텀 로직 실행 (DualMomentumDaily 클래스에서)
            self.daily_strategy.run_weekly_momentum_logic(current_daily_date)
            
            # RSI 분봉 트레이더에 최신 모멘텀 시그널 업데이트
            self.minute_strategy.update_momentum_signals(self.daily_strategy.momentum_signals)

            # 매수/매도 시그널이 있는 종목들에 대해서만 분봉 데이터 처리
            # (시그널 발생일의 다음 거래일부터 매매 시도)
            for stock_code, signal_info in self.daily_strategy.momentum_signals.items():
                # 이미 오늘 거래가 이루어졌다면 다음 종목으로 넘어감 (손절 등으로)
                if signal_info.get('traded_today', False):
                    continue

                if signal_info['signal'] in ['buy', 'sell']:
                    signal_date = signal_info['signal_date']
                    
                    # 시그널 발생일의 다음 거래일이 현재 처리 중인 일봉 날짜와 같은 경우에만 분봉 로직 실행
                    next_trading_day_for_signal = self.get_next_business_day(signal_date)
                    if next_trading_day_for_signal and next_trading_day_for_signal == current_daily_date:
                        # 매도 시그널인데 보유 종목이 없으면 건너뛰기
                        if signal_info['signal'] == 'sell' and self.broker.get_position_size(stock_code) <= 0:
                            continue

                        minute_data = self._get_minute_data_for_signal_dates(stock_code, signal_date) # API 호출로 분봉 데이터 로드
                        if not minute_data.empty:
                            # 해당 일자의 분봉 데이터만 필터링하여 순회
                            minute_data_today = minute_data.loc[minute_data.index.normalize() == pd.Timestamp(current_daily_date).normalize()]
                            for minute_dt in minute_data_today.index:
                                if minute_dt > end_date: # 백테스트 종료 시간을 넘어서면 중단
                                    break
                                # RSI 분봉 트레이더를 통해 실제 매수/매도 실행
                                self.minute_strategy.run_minute_logic(stock_code, minute_dt)
                                if self.daily_strategy.momentum_signals[stock_code]['traded_today']:
                                    # 해당 종목에 대해 오늘 거래가 완료되었으면 (매수 또는 매도) 다음 종목으로 넘어감
                                    break
                        else:
                            logging.warning(f"[{current_daily_date.isoformat()}] {stock_code}: 시그널({signal_info['signal']})이 있으나 분봉 데이터가 없어 매매를 시도할 수 없습니다.")


            # 일별 종료 시 포트폴리오 가치 계산 및 기록
            current_prices = {}
            for stock_code in self.data_store['daily']:
                daily_bar = self.data_store['daily'][stock_code].loc[self.data_store['daily'][stock_code].index.normalize() == current_daily_date_full.normalize()]
                if not daily_bar.empty:
                    current_prices[stock_code] = daily_bar['close'].iloc[0]
                else:
                    # 해당 날짜에 데이터가 없으면, 전날 마지막 가격으로 대체 시도 (정확도는 떨어짐)
                    # 실제 백테스트에서는 데이터가 없는 날짜는 건너뛰거나 미리 처리해야 함
                    last_valid_idx = self.data_store['daily'][stock_code].index.normalize() <= current_daily_date_full.normalize()
                    if last_valid_idx.any():
                        current_prices[stock_code] = self.data_store['daily'][stock_code].loc[last_valid_idx]['close'].iloc[-1]
                    else:
                        current_prices[stock_code] = 0 # 데이터 없으면 0으로 간주 (포트폴리오 가치에 영향)

            portfolio_value = self.broker.get_portfolio_value(current_prices)
            portfolio_values.append(portfolio_value)
            dates.append(current_daily_date_full)

        # 포트폴리오 가치 시계열 데이터 생성
        portfolio_value_series = pd.Series(portfolio_values, index=dates)
        
        # 성과 지표 계산
        metrics = calculate_performance_metrics(portfolio_value_series, risk_free_rate=0.03) # 무위험 수익률은 필요에 따라 조정
        
        # 최종 결과 출력
        logging.info("\n=== 백테스트 결과 ===")
        logging.info(f"시작일: { start_date.date().isoformat()}")
        logging.info(f"종료일: {end_date.date().isoformat()}")
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