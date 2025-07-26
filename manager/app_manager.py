# app/app_manager.py (최종 통합본)

import logging
import pandas as pd
from datetime import datetime, date, timedelta
import json
from typing import Optional, List, Dict, Any

# --- 프로젝트의 다른 모듈 임포트 ---
from manager.db_manager import DBManager
# BacktestManager에서 사용하던 유틸리티 임포트
from util.strategies_util import calculate_sma, calculate_rsi, calculate_ema, calculate_macd

logger = logging.getLogger(__name__)

class AppManager:
    """
    DB 데이터 기반의 모든 비즈니스 로직을 총괄하는 중앙 관리자.
    BacktestManager의 기능을 내장하여 독립적으로 동작합니다.
    """
    def __init__(self):
        self.db_manager = DBManager()
        self.stock_info_map: Dict[str, str] = {}
        # AppManager가 생성될 때 직접 종목 정보를 로드
        self._load_stock_info_map()
        logger.info("AppManager 초기화 완료 (DB 전용 독립 모드).")

    def close(self):
        """애플리케이션 종료 시 DB 연결을 정리합니다."""
        self.db_manager.close()
        logger.info("AppManager를 통해 DB 연결이 종료되었습니다.")
        
    # ========================================================
    # 아래는 BacktestManager에서 가져온 메서드들
    # ========================================================

    def _load_stock_info_map(self):
        """DB에서 모든 종목 정보를 가져와 self.stock_info_map에 직접 저장합니다."""
        logger.debug("종목 정보 맵(딕셔너리) 로딩 시작")
        try:
            stock_info_df = self.db_manager.fetch_stock_info() 
            if not stock_info_df.empty:
                self.stock_info_map = pd.Series(stock_info_df.stock_name.values, index=stock_info_df.stock_code).to_dict()
                logger.debug(f"{len(self.stock_info_map)}개의 종목 정보 로딩 완료")
            else:
                self.stock_info_map = {}
        except Exception as e:
            logger.error(f"종목 정보 로딩 중 오류 발생: {e}")
            self.stock_info_map = {}

    def get_stock_info_map(self) -> Dict[str, str]:
        """캐시된 종목 코드-이름 맵을 반환합니다."""
        return self.stock_info_map

    def get_backtest_runs(self, run_id: int = None, start_date: date = None, end_date: date = None) -> pd.DataFrame:
        df = self.db_manager.fetch_backtest_run(run_id, start_date, end_date)
        if df.empty: return df
        
        for col in ['params_json_daily', 'params_json_minute']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and x else {})

        for col in ['cumulative_return', 'max_drawdown']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') * 100
        
        if 'cumulative_return' in df.columns and 'start_date' in df.columns and 'end_date' in df.columns:
            df['annualized_return'] = df.apply(self._calculate_annualized_return, axis=1)
        else:
            df['annualized_return'] = 0.0
        return df

    def _calculate_annualized_return(self, row) -> float:
        start, end = pd.to_datetime(row['start_date']), pd.to_datetime(row['end_date'])
        days = (end - start).days
        if days <= 0: return 0.0
        final_return = row['cumulative_return'] / 100.0
        return (((1 + final_return) ** (365 / days)) - 1) * 100

    def get_backtest_performance(self, run_id: int) -> pd.DataFrame:
        df = self.db_manager.fetch_backtest_performance(run_id)
        if df.empty: return df
        for col in ['daily_return', 'cumulative_return', 'drawdown']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') * 100
        return df

    def get_backtest_trades(self, run_id: int) -> pd.DataFrame:
        df = self.db_manager.fetch_backtest_trade(run_id)
        if df.empty: return df
        df['trade_datetime'] = pd.to_datetime(df['trade_datetime'])
        for col in ['trade_price', 'trade_amount', 'commission', 'tax', 'realized_profit_loss']:
            if col in df.columns: 
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def get_traded_stocks_summary(self, run_id: int) -> pd.DataFrame:
        trades_df = self.get_backtest_trades(run_id)
        if trades_df.empty:
            return pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_count', 'total_realized_profit_loss', 'avg_return_per_trade'])

        trade_counts = trades_df.groupby('stock_code').size().reset_index(name='trade_count')
        realized_profits = trades_df[trades_df['trade_type'] == 'SELL'].groupby('stock_code')['realized_profit_loss'].sum().reset_index(name='total_realized_profit_loss')
        
        buy_trades = trades_df[trades_df['trade_type'] == 'BUY'].set_index('trade_id')
        trade_returns_list = []
        for index, sell_trade in trades_df[trades_df['trade_type'] == 'SELL'].iterrows():
            entry_trade_id = sell_trade.get('entry_trade_id')
            if pd.notna(entry_trade_id) and entry_trade_id in buy_trades.index:
                buy_trade = buy_trades.loc[entry_trade_id]
                entry_price = float(buy_trade['trade_price'])
                exit_price = float(sell_trade['trade_price'])
                if entry_price > 0:
                    trade_return = ((exit_price - entry_price) / entry_price) * 100
                    trade_returns_list.append({'stock_code': sell_trade['stock_code'], 'trade_return': trade_return})
        
        avg_returns = pd.DataFrame(trade_returns_list).groupby('stock_code')['trade_return'].mean().reset_index(name='avg_return_per_trade') if trade_returns_list else pd.DataFrame(columns=['stock_code', 'avg_return_per_trade'])

        summary_df = pd.merge(trade_counts, realized_profits, on='stock_code', how='left').fillna(0)
        summary_df = pd.merge(summary_df, avg_returns, on='stock_code', how='left').fillna(0)
        
        stock_map = self.get_stock_info_map()
        summary_df['stock_name'] = summary_df['stock_code'].map(stock_map).fillna(summary_df['stock_code'])

        return summary_df

    def get_daily_ohlcv_with_indicators(self, stock_code: str, start_date: date, end_date: date, daily_strategy_params: dict) -> pd.DataFrame:
        lookback_buffer_days = max(daily_strategy_params.get('long_sma_period', 0), daily_strategy_params.get('lookback_period', 0)) + 30
        adjusted_start_date = start_date - timedelta(days=lookback_buffer_days)
        daily_df = self.db_manager.fetch_daily_price(stock_code, adjusted_start_date, end_date)
        
        if daily_df.empty: return pd.DataFrame()
        daily_df.index.name = 'Date'
        
        strategy_name = daily_strategy_params.get('strategy_name')
        if strategy_name == 'SMADaily':
            daily_df[f'SMA_{daily_strategy_params.get("short_sma_period")}'] = calculate_sma(daily_df['close'], period=daily_strategy_params.get("short_sma_period"))
            daily_df[f'SMA_{daily_strategy_params.get("long_sma_period")}'] = calculate_sma(daily_df['close'], period=daily_strategy_params.get("long_sma_period"))
        elif strategy_name == 'TripleScreenDaily':
            macd_df = calculate_macd(daily_df['close'], short_period=daily_strategy_params.get('ema_short_period', 12), long_period=daily_strategy_params.get('ema_long_period', 26), signal_period=daily_strategy_params.get('macd_signal_period', 9))
            daily_df = pd.concat([daily_df, macd_df], axis=1)

        return daily_df[daily_df.index.normalize() >= pd.Timestamp(start_date).normalize()]

    def get_minute_ohlcv_with_indicators(self, stock_code: str, trade_date: date, minute_strategy_params: dict) -> pd.DataFrame:
        trading_days = sorted([d.date() for d in self.db_manager.get_all_trading_days(trade_date - timedelta(days=7), trade_date) if d.date() <= trade_date])
        from_date = trading_days[-2] if len(trading_days) >= 2 else trade_date
        
        minute_df = self.db_manager.fetch_minute_price(stock_code, from_date, trade_date)
        if minute_df.empty: return pd.DataFrame()
        minute_df.index.name = 'Datetime'
        
        strategy_name = minute_strategy_params.get('strategy_name')
        if strategy_name == 'RSIMinute':
            minute_df['RSI'] = calculate_rsi(minute_df['close'], period=minute_strategy_params.get('minute_rsi_period'))
            
        start_filter_dt = datetime.combine(from_date, datetime.min.time().replace(hour=9, minute=0))
        end_filter_dt = datetime.combine(trade_date, datetime.max.time().replace(hour=15, minute=30))
        return minute_df[(minute_df.index >= start_filter_dt) & (minute_df.index <= end_filter_dt)]