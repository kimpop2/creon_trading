# manager/business_manager.py

import logging
import pandas as pd
from datetime import datetime, date, timedelta
import sys
import os
import json
from typing import Optional, List

# project_root를 sys.path에 추가하여 모듈 임포트 가능하게 함
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from manager.db_manager import DBManager
from manager.data_manager import DataManager # DataManager의 메서드를 활용하기 위해 임포트
from util.strategies_util import calculate_sma, calculate_rsi, calculate_ema, calculate_macd

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 필요 시 DEBUG로 설정

class BusinessManager:
    """
    백테스팅 결과를 조회하고 시각화에 필요한 데이터를 가공하는 역할을 담당합니다.
    DBManager와 DataManager의 기능을 활용합니다.
    """
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager # DataManager 인스턴스 주입
        self.db_manager = data_manager.db_manager # DataManager의 DBManager 재사용
        self.api_client = data_manager.api_client
        logger.info("BusinessManager 초기화 완료.")

    def fetch_backtest_runs(self, run_id: int = None, start_date: date = None, end_date: date = None) -> pd.DataFrame:
        """
        DB에서 백테스트 실행 정보를 조회합니다.
        Args:
            run_id (int, optional): 조회할 백테스트 실행 ID. Defaults to None.
            start_date (date, optional): 백테스트 시작일 필터링 (이상). Defaults to None.
            end_date (date, optional): 백테스트 종료일 필터링 (이하). Defaults to None.
        Returns:
            pd.DataFrame: 백테스트 실행 정보.
        """
        logger.debug(f"백테스트 실행 정보 조회 요청: run_id={run_id}, start_date={start_date}, end_date={end_date}")
        df = self.db_manager.fetch_backtest_run(run_id, start_date, end_date)
        
        # JSON 문자열로 저장된 파라미터를 딕셔너리로 변환
        if 'params_json_daily' in df.columns:
            df['params_json_daily'] = df['params_json_daily'].apply(lambda x: json.loads(x) if pd.notna(x) and x else {})
        if 'params_json_minute' in df.columns:
            df['params_json_minute'] = df['params_json_minute'].apply(lambda x: json.loads(x) if pd.notna(x) and x else {})

        # 수익률을 100분율로 변환 (Decimal 타입을 float로 변환)
        if 'cumulative_return' in df.columns:
            df['cumulative_return'] = df['cumulative_return'].astype(float) * 100
        if 'max_drawdown' in df.columns:
            df['max_drawdown'] = df['max_drawdown'].astype(float) * 100
            
        # 연수익률(Annualized Return) 계산
        if 'cumulative_return' in df.columns and 'start_date' in df.columns and 'end_date' in df.columns:
            def calculate_annualized_return(row):
                start = pd.to_datetime(row['start_date'])
                end = pd.to_datetime(row['end_date'])
                days = (end - start).days
                if days == 0:
                    return 0
                
                final_return_decimal = row['cumulative_return'] / 100
                annualized = ((1 + final_return_decimal) ** (365 / days)) - 1
                return annualized * 100

            df['annualized_return'] = df.apply(calculate_annualized_return, axis=1)
        else:
            df['annualized_return'] = 0

        logger.debug(f"조회된 백테스트 실행 정보 (총 {len(df)}개): {df.head()}")
        return df

    def fetch_backtest_performance(self, run_id: int, start_date: date = None, end_date: date = None) -> pd.DataFrame:
        """
        DB에서 백테스트 일별 성능 지표를 조회합니다.
        Args:
            run_id (int): 조회할 백테스트 실행 ID.
            start_date (date, optional): 성능 기록 시작 날짜 필터링. Defaults to None.
            end_date (date, optional): 성능 기록 종료 날짜 필터링. Defaults to None.
        Returns:
            pd.DataFrame: 일별 성능 지표.
        """
        logger.debug(f"백테스트 성능 지표 조회 요청: run_id={run_id}, start_date={start_date}, end_date={end_date}")
        df = self.db_manager.fetch_backtest_performance(run_id, start_date, end_date)
        
        # 수익률을 100분율로 변환 (Decimal 타입을 float로 변환)
        if 'daily_return' in df.columns:
            df['daily_return'] = df['daily_return'].astype(float) * 100
        if 'cumulative_return' in df.columns:
            df['cumulative_return'] = df['cumulative_return'].astype(float) * 100
        if 'drawdown' in df.columns:
            df['drawdown'] = df['drawdown'].astype(float) * 100
            
        logger.debug(f"조회된 백테스트 성능 지표 (총 {len(df)}개): {df.head()}")
        return df

    def fetch_backtest_trades(self, run_id: int, stock_code: str = None, start_datetime: datetime = None, end_datetime: datetime = None) -> pd.DataFrame:
        """
        DB에서 백테스트 개별 거래 내역을 조회합니다.
        Args:
            run_id (int): 조회할 백테스트 실행 ID.
            stock_code (str, optional): 조회할 종목 코드. Defaults to None.
            start_datetime (datetime, optional): 거래 시작 시각. Defaults to None.
            end_datetime (datetime, optional): 거래 종료 시각. Defaults to None.
        Returns:
            pd.DataFrame: 개별 거래 내역.
        """
        logger.debug(f"백테스트 거래 내역 조회 요청: run_id={run_id}, stock_code={stock_code}, start_datetime={start_datetime}, end_datetime={end_datetime}")
        df = self.db_manager.fetch_backtest_trade(run_id, stock_code, start_datetime, end_datetime)
        
        if df.empty:
            return df

        # trade_datetime 컬럼을 datetime 객체로 변환
        if 'trade_datetime' in df.columns:
            df['trade_datetime'] = pd.to_datetime(df['trade_datetime'])
            
        # Decimal 타입을 float로 변환
        decimal_cols = ['trade_price', 'trade_amount', 'commission', 'tax', 'realized_profit_loss']
        for col in decimal_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        logger.debug(f"조회된 백테스트 거래 내역 (총 {len(df)}개): {df.head()}")
        return df

    def get_traded_stocks_summary(self, run_id: int) -> pd.DataFrame:
        """
        특정 백테스트의 매매 이력을 기반으로 종목별 매매 횟수, 실현 손익, 평균 수익률을 집계합니다.
        Args:
            run_id (int): 백테스트 실행 ID.
        Returns:
            pd.DataFrame: 종목별 매매 요약 정보.
        """
        logger.debug(f"종목별 매매 요약 정보 집계 요청: run_id={run_id}")
        trades_df = self.fetch_backtest_trades(run_id)
        
        if trades_df.empty:
            logger.info(f"Run ID {run_id}에 대한 거래 내역이 없습니다.")
            return pd.DataFrame(columns=['stock_code', 'trade_count', 'total_realized_profit_loss', 'avg_return_per_trade'])

        # 매매 횟수 (매수/매도 각각 1회로 간주)
        trade_counts = trades_df.groupby('stock_code').size().reset_index(name='trade_count')

        # 실현 손익 (매도 거래만 해당) - Decimal을 float로 변환
        realized_profits = trades_df[trades_df['trade_type'] == 'SELL'].groupby('stock_code')['realized_profit_loss'].sum().reset_index(name='total_realized_profit_loss')
        # Decimal 타입을 float로 변환
        realized_profits['total_realized_profit_loss'] = realized_profits['total_realized_profit_loss'].astype(float)

        # 종목별 평균 수익률 계산 (매수-매도 쌍 기준)
        trade_returns_list = []
        # 매수 거래 데이터를 딕셔너리로 저장 (trade_id: {price, quantity})
        buy_trades = trades_df[trades_df['trade_type'] == 'BUY'].set_index('trade_id')

        for index, sell_trade in trades_df[trades_df['trade_type'] == 'SELL'].iterrows():
            entry_trade_id = sell_trade['entry_trade_id']
            if pd.notna(entry_trade_id) and entry_trade_id in buy_trades.index:
                try:
                    buy_trade = buy_trades.loc[entry_trade_id]
                    entry_price = float(buy_trade['trade_price'])  # Decimal을 float로 변환
                    exit_price = float(sell_trade['trade_price'])   # Decimal을 float로 변환
                    
                    if entry_price > 0:
                        trade_return = ((exit_price - entry_price) / entry_price) * 100
                        trade_returns_list.append({'stock_code': sell_trade['stock_code'], 'trade_return': trade_return})
                except (KeyError, IndexError) as e:
                    logger.warning(f"매수-매도 쌍 매칭 중 오류 발생: {e}, entry_trade_id={entry_trade_id}")
                    continue
            
        if trade_returns_list:
            trade_returns_df = pd.DataFrame(trade_returns_list)
            avg_returns = trade_returns_df.groupby('stock_code')['trade_return'].mean().reset_index(name='avg_return_per_trade')
        else:
            avg_returns = pd.DataFrame(columns=['stock_code', 'avg_return_per_trade'])

        # 결과 병합
        summary_df = pd.merge(trade_counts, realized_profits, on='stock_code', how='left').fillna(0)
        summary_df = pd.merge(summary_df, avg_returns, on='stock_code', how='left').fillna(0)
        
        # 수익금/수익률을 소수점 2자리로 제한 (float 타입으로 안전하게 처리)
        summary_df['total_realized_profit_loss'] = summary_df['total_realized_profit_loss'].astype(float).round(2)
        summary_df['avg_return_per_trade'] = summary_df['avg_return_per_trade'].astype(float).round(2)

        logger.debug(f"집계된 종목별 매매 요약 정보: {summary_df.head()}")
        return summary_df

    def fetch_daily_ohlcv_with_indicators(self, stock_code: str, start_date: date, end_date: date, daily_strategy_params: dict) -> pd.DataFrame:
        """
        특정 종목의 일봉 OHLCV 데이터를 가져오고, 일봉 전략의 지표(SMA 등)를 계산하여 추가합니다.
        Args:
            stock_code (str): 종목 코드.
            start_date (date): 데이터 조회 시작일.
            end_date (date): 데이터 조회 종료일.
            daily_strategy_params (dict): 일봉 전략의 파라미터.
        Returns:
            pd.DataFrame: 일봉 OHLCV 데이터 및 계산된 지표.
        """
        logger.debug(f"일봉 OHLCV 및 지표 조회 요청: {stock_code}, {start_date} ~ {end_date}, 전략 파라미터: {daily_strategy_params}")
        
        # DataManager를 통해 일봉 OHLCV 데이터 로드
        # 지표 계산에 필요한 충분한 과거 데이터 확보를 위해 조회 시작일을 앞당김
        # 예: SMA long_period가 60일이면 최소 60일 이전부터 데이터를 가져와야 정확한 계산 가능
        lookback_buffer_days = max(daily_strategy_params.get('long_sma_period', 0), daily_strategy_params.get('lookback_period', 0)) + 30 # 충분한 버퍼
        adjusted_start_date = start_date - timedelta(days=lookback_buffer_days)
        
        daily_df = self.data_manager.cache_daily_ohlcv(stock_code, adjusted_start_date, end_date)
        
        if daily_df.empty:
            logger.warning(f"종목 {stock_code}의 일봉 데이터를 찾을 수 없습니다.")
            return pd.DataFrame()

        # 인덱스 이름 설정 (Matplotlib finance plot용)
        daily_df.index.name = 'Date'
        
        # 일봉 전략 이름 확인 및 지표 계산
        daily_strategy_name = daily_strategy_params.get('strategy_name') # params_json_daily에 strategy_name을 포함해야 함
        
        logger.info(f"일봉 전략 분석: 이름='{daily_strategy_name}', 파라미터={daily_strategy_params}")

        if daily_strategy_name == 'SMADaily':
            short_period = daily_strategy_params.get('short_sma_period')
            long_period = daily_strategy_params.get('long_sma_period')
            volume_period = daily_strategy_params.get('volume_ma_period')
            
            if short_period and long_period:
                daily_df[f'SMA_{short_period}'] = calculate_sma(daily_df['close'], period=short_period)
                daily_df[f'SMA_{long_period}'] = calculate_sma(daily_df['close'], period=long_period)
                logger.info(f"SMADaily 지표 (SMA_{short_period}, SMA_{long_period}) 계산 완료.")
            if volume_period:
                daily_df[f'Volume_MA_{volume_period}'] = calculate_sma(daily_df['volume'], period=volume_period)
                logger.info(f"SMADaily 지표 (Volume_MA_{volume_period}) 계산 완료.")

        elif daily_strategy_name == 'TripleScreenDaily':
            ema_short_period = daily_strategy_params.get('ema_short_period', 12)
            ema_long_period = daily_strategy_params.get('ema_long_period', 26)
            macd_signal_period = daily_strategy_params.get('macd_signal_period', 9)
            
            daily_df[f'EMA_{ema_short_period}'] = calculate_ema(daily_df['close'], period=ema_short_period)
            daily_df[f'EMA_{ema_long_period}'] = calculate_ema(daily_df['close'], period=ema_long_period)
            
            macd_df = calculate_macd(daily_df['close'], short_period=ema_short_period, long_period=ema_long_period, signal_period=macd_signal_period)
            daily_df = pd.concat([daily_df, macd_df], axis=1)
            logger.info(f"TripleScreenDaily 지표 (EMA, MACD) 계산 완료.")
            
        # 다른 일봉 전략이 있다면 여기에 추가
        # elif daily_strategy_name == 'AnotherDailyStrategy':
        #     ...

        # 요청된 기간의 데이터만 필터링하여 반환
        return daily_df[daily_df.index.normalize() >= pd.Timestamp(start_date).normalize()]

    def fetch_minute_ohlcv_with_indicators(self, stock_code: str, trade_date: date, minute_strategy_params: dict) -> pd.DataFrame:
        """
        특정 종목의 분봉 OHLCV 데이터를 가져오고, 분봉 전략의 지표(RSI 등)를 계산하여 추가합니다.
        매매 전날부터 매매 당일까지 2일치 분봉 데이터를 가져옵니다.
        Args:
            stock_code (str): 종목 코드.
            trade_date (date): 매매가 발생한 날짜 (조회 기준일).
            minute_strategy_params (dict): 분봉 전략의 파라미터.
        Returns:
            pd.DataFrame: 분봉 OHLCV 데이터 및 계산된 지표.
        """
        logger.debug(f"분봉 OHLCV 및 지표 조회 요청: {stock_code}, 매매일={trade_date}, 전략 파라미터: {minute_strategy_params}")

        # 매매 전날부터 매매 당일까지의 데이터 조회
        # prev_trading_day를 정확히 찾기 위해 market_calendar 사용
        trading_days = self.db_manager.get_all_trading_days(trade_date - timedelta(days=7), trade_date) # 넉넉하게 일주일 전부터
        trading_days = sorted([d for d in trading_days if d <= pd.Timestamp(trade_date).normalize()], reverse=True)
        
        from_date = None
        if len(trading_days) >= 2:
            from_date = trading_days[1].date() # 매매일 직전 영업일
        elif len(trading_days) == 1: # 매매일이 첫 영업일인 경우
            from_date = trading_days[0].date() # 매매일 당일
        else:
            logger.warning(f"매매일 {trade_date}의 이전 영업일을 찾을 수 없습니다. {trade_date} 당일만 조회합니다.")
            from_date = trade_date

        to_date = trade_date # 매매 당일까지

        # DataManager를 통해 분봉 OHLCV 데이터 로드
        # 지표 계산에 필요한 충분한 과거 데이터 확보를 위해 조회 시작일을 앞당김
        lookback_buffer_minutes = minute_strategy_params.get('minute_rsi_period', 0) * 2 # 충분한 버퍼 (예: RSI 14분이면 28분)
        # from_date의 9시부터 to_date의 15시 30분까지 데이터를 가져옴
        
        # cache_minute_ohlcv는 from_date, to_date가 date 타입이어야 함
        minute_df = self.data_manager.cache_minute_ohlcv(stock_code, from_date, to_date)
        
        if minute_df.empty:
            logger.warning(f"종목 {stock_code}의 분봉 데이터를 찾을 수 없습니다 (기간: {from_date} ~ {to_date}).")
            return pd.DataFrame()

        # 인덱스 이름 설정 (Matplotlib finance plot용)
        minute_df.index.name = 'Datetime'

        # 분봉 전략 이름 확인 및 지표 계산
        minute_strategy_name = minute_strategy_params.get('strategy_name') # params_json_minute에 strategy_name을 포함해야 함
        
        if minute_strategy_name == 'RSIMinute':
            rsi_period = minute_strategy_params.get('minute_rsi_period')
            if rsi_period:
                # RSI 계산 로직 (util.strategies_util.calculate_rsi 사용)
                minute_df['RSI'] = calculate_rsi(minute_df['close'], period=rsi_period)
                logger.debug(f"RSIMinute 지표 (RSI_{rsi_period}) 계산 완료.")
        # 다른 분봉 전략이 있다면 여기에 추가
        # elif minute_strategy_name == 'AnotherMinuteStrategy':
        #     ...

        # 요청된 기간의 데이터만 필터링하여 반환 (매매 전날 9시부터 매매 당일 장 마감까지)
        start_filter_dt = datetime.combine(from_date, datetime.min.time().replace(hour=9, minute=0))
        end_filter_dt = datetime.combine(to_date, datetime.max.time().replace(hour=15, minute=30)) # 15:30 장 마감

        filtered_df = minute_df[(minute_df.index >= start_filter_dt) & (minute_df.index <= end_filter_dt)]
        
        return filtered_df

    # --- BusinessManager에서 이동: 실시간/과거 데이터, 종목/캘린더/재무정보 관련 메서드 ---
    def get_realtime_price(self, stock_code: str) -> Optional[float]:
        price = self.api_client.get_current_price(stock_code)
        if price is not None:
            logger.debug(f"실시간 현재가 조회: {stock_code} - {price:,.0f}원")
        else:
            logger.warning(f"실시간 현재가 조회 실패: {stock_code}")
        return float(price) if price is not None else None

    def get_realtime_minute_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        df = self.api_client.get_current_minute_data(stock_code)
        if df is not None and not df.empty:
            logger.debug(f"실시간 1분봉 데이터 조회: {stock_code}, {len(df)}개")
        else:
            logger.warning(f"실시간 1분봉 데이터 조회 실패 또는 없음: {stock_code}")
        return df

    def get_historical_ohlcv(self, stock_code: str, period: str, count: int) -> Optional[pd.DataFrame]:
        df = self.api_client.get_price_data(stock_code, period, count)
        if df is not None and not df.empty:
            logger.debug(f"과거 OHLCV 데이터 조회: {stock_code}, 주기: {period}, 개수: {count}, {len(df)}개")
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime']).dt.date
            elif 'Date' in df.columns:
                df['datetime'] = pd.to_datetime(df['Date']).dt.date
                df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
                df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        else:
            logger.warning(f"과거 OHLCV 데이터 조회 실패 또는 없음: {stock_code}, 주기: {period}, 개수: {count}")
        return df

    def get_all_stock_codes(self, market_type: Optional[str] = None) -> List[str]:
        codes = self.api_client.get_all_stock_codes(market_type)
        logger.info(f"총 {len(codes)}개의 종목 코드 로드 완료 (시장: {market_type if market_type else '전체'})")
        return codes

    def update_market_calendar(self, from_date: date, to_date: date) -> bool:
        logger.info(f"거래일 캘린더 업데이트 시작: {from_date} ~ {to_date}")
        trading_days = self.api_client.get_market_calendar(from_date, to_date)
        if trading_days:
            calendar_df = pd.DataFrame([{'trade_date': d} for d in trading_days])
            if self.db_manager.insert_df_to_db(calendar_df, "market_calendar", option="append", is_index=False):
                logger.info(f"{len(trading_days)}개의 거래일이 market_calendar 테이블에 업데이트되었습니다.")
                return True
            else:
                logger.error("market_calendar 테이블 업데이트 실패.")
                return False
        logger.warning("Creon API에서 거래일 캘린더를 가져오지 못했습니다.")
        return False

    def update_stock_info(self, stock_code: str):
        logger.info(f"{stock_code} stock_info 테이블의 최신 재무 데이터 업데이트 중...")
        finance_df = self.api_client.get_latest_financial_data(stock_code)
        if finance_df.empty:
            logger.info(f"{stock_code} Creon API에서 조회된 재무 데이터가 없습니다.")
            return
        if self.db_manager.insert_df_to_db(finance_df, "stock_info", option="update", is_index=False, on_conflict_cols=['stock_code']):
            logger.info(f"{stock_code} 종목의 재무 데이터가 stock_info 테이블에 성공적으로 업데이트되었습니다.")
        else:
            logger.error(f"{stock_code} 종목의 재무 데이터 업데이트 실패.")

    # --- Trade/Portfolio Management Methods ---
    def save_daily_signals(self, signals, signal_date):
        """DBManager를 통해 일봉 매매 신호를 저장합니다."""
        return self.db_manager.save_daily_signals(signals, signal_date)

    def load_daily_signals_for_today(self, signal_date):
        """DBManager를 통해 오늘 날짜의 일봉 매매 신호를 로드합니다."""
        # trader.py에서 load_로 호출하므로, db_manager의 fetch_를 호출하도록 브릿지
        return self.db_manager.fetch_daily_signals_for_today(signal_date)

    def save_trade_log(self, log_entry):
        """DBManager를 통해 거래 로그를 저장합니다."""
        return self.db_manager.save_trade_log(log_entry)

    def save_daily_portfolio_snapshot(self, snapshot_date, portfolio_value, cash, positions):
        """DBManager를 통해 일일 포트폴리오 스냅샷을 저장합니다."""
        return self.db_manager.save_daily_portfolio_snapshot(snapshot_date, portfolio_value, cash, positions)

    def save_current_positions(self, positions):
        """DBManager를 통해 현재 보유 포지션을 저장합니다."""
        return self.db_manager.save_current_positions(positions)

    def load_current_positions(self):
        """DBManager를 통해 현재 보유 포지션을 로드합니다."""
        # brokerage.py에서 load_로 호출하므로, db_manager의 fetch_를 호출하도록 브릿지
        return self.db_manager.fetch_current_positions()
