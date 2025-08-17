# manager/data_manager.py (신규 파일)

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # 맨 위에 추가
from pykrx import stock
from datetime import date, timedelta
from typing import Dict, List, Set, Optional, Any
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from config.sector_stocks import sector_stocks

logger = logging.getLogger(__name__)

class DataManager:
    """
    BacktestManager와 TradingManager의 공통 데이터 처리 로직을 담는 부모 클래스.
    DB 캐싱 및 API를 통한 데이터 수집의 핵심 로직을 포함합니다.
    """
    def __init__(self, api_client: Optional[CreonAPIClient], db_manager: DBManager):
        self.api_client = api_client
        self.db_manager = db_manager
        self.stock_info_map: Dict[str, str] = {}
        self.pykrx_master_df = None
        self._load_stock_info_map()
        logger.info(f"DataManager 초기화 완료. {len(self.stock_info_map)}개 종목 정보 로드.")

    
    def get_db_manager(self) -> DBManager:
        """[래퍼] ReportGenerator 등 외부 모듈에 DBManager 인스턴스를 전달해야 할 때 사용합니다."""
        return self.db_manager
    

    def close(self):
        """DB 연결을 종료합니다."""
        if self.db_manager:
            self.db_manager.close()
            logger.info("DataManager를 통해 DB 연결을 종료했습니다.")


    def _load_stock_info_map(self):
        """DB에서 모든 종목 정보를 가져와 내부 딕셔너리에 캐시합니다."""
        try:
            stock_info_df = self.db_manager.fetch_stock_info()
            if not stock_info_df.empty:
                self.stock_info_map = pd.Series(stock_info_df.stock_name.values, index=stock_info_df.stock_code).to_dict()
        except Exception as e:
            logger.error(f"종목 정보 로딩 중 오류 발생: {e}")
            self.stock_info_map = {}


    def fetch_stock_info(self, stock_codes: list = None) -> pd.DataFrame:
        """[래퍼] DBManager의 fetch_stock_info 메서드를 호출합니다."""
        return self.db_manager.fetch_stock_info(stock_codes)
    

    def get_stock_name(self, stock_code: str) -> str:
        """내부 캐시에서 종목명을 조회합니다."""
        return self.stock_info_map.get(stock_code, "알 수 없음")


    def get_universe_codes(self) -> List[str]:
        """설정 파일에서 유니버스 종목 목록을 가져와 코드로 변환합니다."""
        if not self.api_client:
            logger.warning("API 클라이언트가 없어 유니버스 코드를 변환할 수 없습니다.")
            return []
        all_stock_names = {name for stocks in sector_stocks.values() for name, theme in stocks}
        universe_codes = [code for name in sorted(list(all_stock_names)) if (code := self.api_client.get_stock_code(name))]
        return universe_codes


    def cache_daily_ohlcv(self, stock_code: str, from_date: date, to_date: date, all_trading_dates: Optional[Set] = None) -> pd.DataFrame:
        """일봉 데이터를 캐싱하고 반환합니다. DB에 없는 데이터는 API를 통해 가져옵니다."""
        db_df = self.db_manager.fetch_daily_price(stock_code, from_date, to_date)
        db_existing_dates = set(db_df.index.normalize()) if not db_df.empty else set()
        
        if all_trading_dates is None:
            all_trading_dates = set(self.db_manager.get_all_trading_days(from_date, to_date))

        missing_dates = sorted(list(all_trading_dates - db_existing_dates))
        if not missing_dates:
            return db_df

        api_df = self._fetch_and_store_daily_range(stock_code, missing_dates[0].date(), missing_dates[-1].date())
        final_df = pd.concat([db_df, api_df]).sort_index()
        return final_df[~final_df.index.duplicated(keep='last')]


    def cache_minute_ohlcv(self, stock_code: str, from_date: date, to_date: date, all_trading_dates: Optional[Set] = None) -> pd.DataFrame:
        """분봉 데이터를 캐싱하고 반환합니다. DB에 없는 데이터는 API를 통해 가져옵니다."""
        db_df = self.db_manager.fetch_minute_price(stock_code, from_date, to_date)
        db_existing_dates = {pd.Timestamp(d).normalize() for d in db_df.index.date} if not db_df.empty else set()

        if all_trading_dates is None:
            all_trading_dates = set(self.db_manager.get_all_trading_days(from_date, to_date))
            
        missing_dates = sorted(list(all_trading_dates - db_existing_dates))
        if not missing_dates:
            return db_df

        api_df = self._fetch_and_store_minute_range(stock_code, missing_dates[0].date(), missing_dates[-1].date())
        final_df = pd.concat([db_df, api_df]).sort_index()
        return final_df[~final_df.index.duplicated(keep='last')]


    def _fetch_and_store_daily_range(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """[Helper] API에서 일봉 데이터를 가져와 DB에 저장하고 반환합니다."""
        api_df = self.api_client.get_daily_ohlcv(stock_code, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
        if not api_df.empty:
            df_for_db = api_df.reset_index()
            data_to_save = df_for_db.to_dict('records')
            for row in data_to_save:
                row['stock_code'] = stock_code
                row['date'] = row['date'].date()
            self.db_manager.save_daily_price(data_to_save)
        return api_df


    def _fetch_and_store_minute_range(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """[Helper] API에서 분봉 데이터를 가져와 DB에 저장하고 반환합니다."""
        api_df = self.api_client.get_minute_ohlcv(stock_code, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
        if not api_df.empty:
            df_for_db = api_df.reset_index()
            data_to_save = df_for_db.to_dict('records')
            for row in data_to_save:
                row['stock_code'] = stock_code
            self.db_manager.save_minute_price(data_to_save)
        return api_df

    # --- BacktestManager 고유 기능 ---
    # data_manager.py의 prepare_pykrx_data_for_period 함수

    def prepare_pykrx_data_for_period(self, start_date: date, end_date: date):
        """
        [수정] Creon API를 사용하여 KOSPI 지수 데이터를 로드하고,
        '거래대금' 필드를 생성한 후 마스터 데이터프레임에 저장합니다.
        """
        if self.pykrx_master_df is not None:
            logger.info("마스터 데이터가 이미 로드되어 있습니다.")
            return

        logger.info(f"Creon API를 통해 마스터 데이터 로딩 시작 (기간: {start_date} ~ {end_date})...")
        
        fetch_start_date = start_date - timedelta(days=830)
        start_str = fetch_start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')

        # 1. Creon API를 통해 KOSPI 지수(U001) 데이터 조회
        df_ohlcv = self.cache_daily_ohlcv("U001", start_str, end_str)
        
        if df_ohlcv.empty:
            raise ValueError("Creon API를 통해 마스터 KOSPI 데이터 조회에 실패했습니다.")
        
        # 2. Creon API 컬럼명을 한글로 변경
        column_rename_map = {
            'open': '시가',
            'high': '고가',
            'low': '저가',
            'close': '종가',
            'volume': '거래량'
            # 'value' 항목은 직접 계산하므로 여기서는 제외합니다.
        }
        df_ohlcv.rename(columns=column_rename_map, inplace=True)
        
        # --- ▼ [핵심] '종가'와 '거래량'을 사용하여 '거래대금' 컬럼 생성 ▼ ---
        if '종가' in df_ohlcv.columns and '거래량' in df_ohlcv.columns:
            df_ohlcv['거래대금'] = df_ohlcv['종가'] * df_ohlcv['거래량']
        else:
            logger.warning("'종가' 또는 '거래량' 컬럼이 없어 '거래대금'을 생성할 수 없습니다.")
            df_ohlcv['거래대금'] = 0 # 거래대금 필드가 없는 경우를 대비해 0으로 채움
        # --- ▲ [핵심] 종료 ▲ ---
        
        self.pykrx_master_df = df_ohlcv
        
        logger.info(f"마스터 데이터 로딩 완료. 총 {len(self.pykrx_master_df)}일치 데이터 저장.")

    def get_market_data_for_hmm(self, start_date: date = None, end_date: date = None, days: int = 730) -> pd.DataFrame:
        """
        [최종 개선] 코드 효율성, 가독성, 특징 중복성을 개선한 HMM Feature Set 생성 함수입니다.
        """
        if self.pykrx_master_df is None:
            raise RuntimeError("pykrx 마스터 데이터가 로드되지 않았습니다. prepare_pykrx_data_for_period()를 먼저 호출해야 합니다.")

        logger.debug("HMM 학습용 데이터를 마스터 데이터프레임에서 슬라이싱합니다.")

        # --- 1. 데이터 준비 ---
        final_end_date = end_date if end_date else date.today() - timedelta(days=1)
        if start_date:
            final_start_date = start_date
        else:
            final_start_date = final_end_date - timedelta(days=days)
        
        # 계산에 필요한 과거 데이터까지 포함하여 조회
        query_start_date = final_start_date - timedelta(days=250) # 200일 이평선 계산을 위해 여유있게 조회
        
        df = self.pykrx_master_df.loc[query_start_date:final_end_date].copy()
        
        if df.empty:
            logger.warning(f"HMM 학습용 KOSPI 데이터를 마스터 DF에서 슬라이싱할 수 없습니다.")
            return pd.DataFrame()
        
        # --- 2. 기본 변수 사전 계산 ---
        daily_returns = df['종가'].pct_change()

        # --- 3. Feature 계산 ---

        # 3-1. 모멘텀 및 가속도 특징
        window_20d = 20
        sma20 = df['종가'].rolling(window=window_20d).mean()
        sma60 = df['종가'].rolling(window=60).mean()
        df['E_ma_divergence'] = (sma20 - sma60) / sma60 * 100
        
        vol_short = daily_returns.rolling(window=window_20d).std()
        vol_long = daily_returns.rolling(window=60).std()
        df['K_volatility_ratio'] = vol_short / vol_long

        sma50 = df['종가'].rolling(window=50).mean()
        sma200 = df['종가'].rolling(window=200).mean()
        df['L_trend_strength'] = (sma50 - sma200) / sma200 * 100

        # 3-4. 시장 참여도 특징
        # [개선] .apply(axis=1) 대신 where를 사용하여 성능 최적화
        up_day_volume = df['거래대금'].where(daily_returns > 0, 0)
        down_day_volume = df['거래대금'].where(daily_returns < 0, 0)
        up_flow = up_day_volume.rolling(window_20d).sum()
        down_flow = down_day_volume.rolling(window_20d).sum()
        df['M_market_breadth_ratio'] = up_flow / (up_flow + down_flow)

        ema_12 = df['종가'].ewm(span=12, adjust=False).mean()
        ema_26 = df['종가'].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df['N_macd_histogram'] = macd_line - signal_line

        # 2-3. 위기 감지 특징 (VKOSPI 대체 로직)
        high_low = df['고가'] - df['저가']
        high_close = np.abs(df['고가'] - df['종가'].shift())
        low_close = np.abs(df['저가'] - df['종가'].shift())
        true_range = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        drawdown_pct = (df['종가'] - df['고가'].rolling(window=20).max()) / df['고가'].rolling(window=20).max()
        df['O_crisis_index'] = np.abs(drawdown_pct) * (atr / df['종가'])

        # --- 4. 최종 데이터 정리 ---
        final_features_df = df[[
            'E_ma_divergence',
            'K_volatility_ratio',
            'L_trend_strength',       # 장기 추세 방향 (Bull/Bear)  
            'M_market_breadth_ratio', # 상대적 변동성 (Low/High Vol)
            'N_macd_histogram',       # 추세 모멘텀 (Bull/Bear)
            'O_crisis_index'          # 위기 수준 변동성 (Crisis)
        ]].copy()
        
        # 날짜 필터링 및 결측치 처리
        final_features_df = final_features_df.loc[final_start_date:final_end_date]
        final_features_df.replace([np.inf, -np.inf], 0, inplace=True) # NaN 대신 0으로 채워 데이터 손실 최소화
        final_features_df.fillna(0, inplace=True)

        # --- 5. 스케일링 ---
        if final_features_df.empty:
            return pd.DataFrame()
            
        scaler = StandardScaler()
        scaled_features = pd.DataFrame(
            scaler.fit_transform(final_features_df), 
            index=final_features_df.index, 
            columns=final_features_df.columns
        )
        logger.debug(f"HMM 학습용 데이터 스케일링 완료. 최종 {len(scaled_features)}일치 데이터")
        
        return scaled_features

    # --- [신규] 4. 분석 데이터 준비 함수 (코드 중복 제거용) ---
    def prepare_analysis_data(self, model_name: str) -> pd.DataFrame:
        """지정된 모델의 국면 데이터와 KOSPI 데이터를 로드하여 분석용 데이터프레임을 준비합니다."""
        model_info = self.db_manager.fetch_hmm_model_by_name(model_name)
        if not model_info:
            logger.error(f"'{model_name}' 모델을 DB에서 찾을 수 없습니다.")
            return pd.DataFrame()

        regime_df = self.db_manager.fetch_daily_regimes(model_info['model_id'])
        if regime_df.empty:
            logger.error(f"모델 ID {model_info['model_id']}에 대한 국면 데이터가 없습니다.")
            return pd.DataFrame()

        start_date = regime_df['date'].min()
        end_date = regime_df['date'].max()
        kospi_df = self.db_manager.fetch_daily_price("U001", start_date, end_date)

        kospi_df.rename(columns={'close': '종가'}, inplace=True)
        kospi_df.index = pd.to_datetime(kospi_df.index)
        regime_df['date'] = pd.to_datetime(regime_df['date'])
        
        analysis_df = pd.merge(kospi_df[['종가']], regime_df, left_index=True, right_on='date', how='inner')
        analysis_df.sort_values(by='date', inplace=True)
        
        return analysis_df
    
    
    def get_daily_factors(self, start_date: date, end_date: date, stock_code: str) -> pd.DataFrame:
        """
        [수정 완료] pykrx를 사용하여 특정 기간의 일별 팩터 데이터를 조회하고 계산합니다.
        pykrx에서 데이터가 반환되지 않는 경우를 안전하게 처리합니다.
        """
        logger.info(f"[{stock_code}] pykrx에서 팩터 데이터 조회 시작: {start_date} ~ {end_date}")

        pykrx_stock_code = stock_code.replace('A', '')
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        lookback_start_date = (start_date - timedelta(days=365 + 100)).strftime('%Y%m%d')

        df_ohlcv = stock.get_market_ohlcv(lookback_start_date, end_date_str, pykrx_stock_code)
        if df_ohlcv.empty:
            return pd.DataFrame()
        if '거래대금' not in df_ohlcv.columns:
            df_ohlcv['거래대금'] = df_ohlcv['종가'] * df_ohlcv['거래량']
            
        df_kospi = stock.get_index_ohlcv(lookback_start_date, end_date_str, "1001")
        
        df_ohlcv.rename(columns={'시가':'open', '고가':'high', '저가':'low', '종가':'close', '거래량':'volume', '거래대금': 'trading_value'}, inplace=True)
        final_df = df_ohlcv.copy()

        # --- [수정] pykrx 데이터 조회 및 병합 로직 (안전하게 처리) ---
        try:
            df_fundamental = stock.get_market_fundamental(start_date_str, end_date_str, pykrx_stock_code)
            if not df_fundamental.empty and all(c in df_fundamental.columns for c in ['PER', 'PBR', 'DIV']):
                df_fundamental.rename(columns={'PER':'per', 'PBR':'pbr', 'DIV':'dividend_yield'}, inplace=True)
                final_df = pd.merge(final_df, df_fundamental[['per', 'pbr', 'dividend_yield']], left_index=True, right_index=True, how='left')
        except Exception as e:
            logger.warning(f"[{stock_code}] Fundamental 데이터 조회/병합 실패: {e}")

        try:
            df_trading = stock.get_market_trading_volume_by_date(start_date_str, end_date_str, pykrx_stock_code)
            if not df_trading.empty and all(c in df_trading.columns for c in ['외국인합계', '기관합계']):
                df_trading.rename(columns={'외국인합계':'foreigner_net_buy', '기관합계':'institution_net_buy'}, inplace=True)
                final_df = pd.merge(final_df, df_trading[['foreigner_net_buy', 'institution_net_buy']], left_index=True, right_index=True, how='left')
        except Exception as e:
            logger.warning(f"[{stock_code}] Trading Volume 데이터 조회/병합 실패: {e}")

        try:
            df_short = stock.get_shorting_status_by_date(start_date_str, end_date_str, pykrx_stock_code)
            if not df_short.empty and '공매도' in df_short.columns:
                df_short.rename(columns={'공매도':'short_volume'}, inplace=True)
                final_df = pd.merge(final_df, df_short[['short_volume']], left_index=True, right_index=True, how='left')
        except Exception as e:
            logger.warning(f"[{stock_code}] Shorting Status 데이터 조회/병합 실패: {e}")
        # --- 수정 끝 ---

        final_df['ma20'] = final_df['close'].rolling(window=20).mean()
        final_df['dist_from_ma20'] = (final_df['close'] / final_df['ma20'] - 1) * 100
        daily_returns = final_df['close'].pct_change()
        final_df['historical_volatility_20d'] = daily_returns.rolling(window=20).std() * np.sqrt(252)
        market_returns = df_kospi['종가'].pct_change()
        market_returns.name = 'market_return'
        combined_returns = pd.merge(daily_returns.rename('close'), market_returns, left_index=True, right_index=True, how='left')
        final_df['relative_strength'] = (combined_returns['close'] - combined_returns['market_return']) * 100
        rolling_cov = combined_returns['close'].rolling(window=60).cov(combined_returns['market_return'])
        rolling_var = combined_returns['market_return'].rolling(window=60).var()
        final_df['beta_coefficient'] = rolling_cov / rolling_var
        final_df['high_52w'] = final_df['high'].rolling(window=52*5, min_periods=1).max()
        is_new_high = final_df['high'] >= final_df['high_52w'].shift(1)
        new_high_dates = pd.Series(final_df.index, index=final_df.index).where(is_new_high).ffill()
        final_df['days_since_52w_high'] = (final_df.index - new_high_dates).dt.days

        final_df['stock_code'] = stock_code
        final_df = final_df.loc[start_date_str:end_date_str]
        
        final_df.reset_index(inplace=True)
        
        if '날짜' in final_df.columns:
            final_df.rename(columns={'날짜': 'date'}, inplace=True)
        
        final_df.drop_duplicates(subset=['date'], keep='first', inplace=True)
        
        final_columns = [
            'date', 'stock_code', 'foreigner_net_buy', 'institution_net_buy', 'program_net_buy',
            'trading_intensity', 'credit_ratio', 'short_volume', 'trading_value', 'per', 'pbr',
            'psr', 'dividend_yield', 'relative_strength', 'beta_coefficient',
            'days_since_52w_high', 'dist_from_ma20', 'historical_volatility_20d',
            'q_revenue_growth_rate', 'q_op_income_growth_rate'
        ]
        
        for col in final_columns:
            if col not in final_df.columns:
                final_df[col] = np.nan
        # 최종 반환 전에 모든 NaN 값을 0으로 채움
        final_df.fillna(0, inplace=True)
        return final_df[final_columns]

    # def get_daily_factors(self, start_date: date, end_date: date, stock_code: str) -> pd.DataFrame:
    #     """
    #     pykrx를 사용하여 특정 기간의 일별 팩터 데이터를 조회하고 계산합니다. (진짜 최종 수정본)
    #     """
    #     logger.info(f"[{stock_code}] pykrx에서 팩터 데이터 조회 시작: {start_date} ~ {end_date}")

    #     pykrx_stock_code = stock_code.replace('A', '')
    #     start_date_str = start_date.strftime('%Y%m%d')
    #     end_date_str = end_date.strftime('%Y%m%d')
    #     lookback_start_date = (start_date - timedelta(days=365 + 100)).strftime('%Y%m%d')

    #     df_ohlcv = stock.get_market_ohlcv(lookback_start_date, end_date_str, pykrx_stock_code)
    #     if df_ohlcv.empty:
    #         return pd.DataFrame()
    #     if '거래대금' not in df_ohlcv.columns:
    #         df_ohlcv['거래대금'] = df_ohlcv['종가'] * df_ohlcv['거래량']
            
    #     df_fundamental = stock.get_market_fundamental(start_date_str, end_date_str, pykrx_stock_code)
    #     df_trading = stock.get_market_trading_volume_by_date(start_date_str, end_date_str, pykrx_stock_code)
    #     df_short = stock.get_shorting_status_by_date(start_date_str, end_date_str, pykrx_stock_code)
    #     df_kospi = stock.get_index_ohlcv(lookback_start_date, end_date_str, "1001")
        
    #     df_ohlcv.rename(columns={'시가':'open', '고가':'high', '저가':'low', '종가':'close', '거래량':'volume', '거래대금': 'trading_value'}, inplace=True)
    #     final_df = df_ohlcv.copy()

    #     df_fundamental.rename(columns={'PER':'per', 'PBR':'pbr', 'DIV':'dividend_yield'}, inplace=True)
    #     final_df = pd.merge(final_df, df_fundamental[['per', 'pbr', 'dividend_yield']], left_index=True, right_index=True, how='left')

    #     df_trading.rename(columns={'외국인합계':'foreigner_net_buy', '기관합계':'institution_net_buy'}, inplace=True)
    #     final_df = pd.merge(final_df, df_trading[['foreigner_net_buy', 'institution_net_buy']], left_index=True, right_index=True, how='left')

    #     if not df_short.empty and '공매도' in df_short.columns:
    #         df_short.rename(columns={'공매도':'short_volume'}, inplace=True)
    #         final_df = pd.merge(final_df, df_short[['short_volume']], left_index=True, right_index=True, how='left')
        
    #     final_df['ma20'] = final_df['close'].rolling(window=20).mean()
    #     final_df['dist_from_ma20'] = (final_df['close'] / final_df['ma20'] - 1) * 100
    #     daily_returns = final_df['close'].pct_change()
    #     final_df['historical_volatility_20d'] = daily_returns.rolling(window=20).std() * np.sqrt(252)
    #     market_returns = df_kospi['종가'].pct_change()
    #     market_returns.name = 'market_return'
    #     combined_returns = pd.merge(daily_returns, market_returns, left_index=True, right_index=True, how='left')
    #     final_df['relative_strength'] = (combined_returns['close'] - combined_returns['market_return']) * 100
    #     rolling_cov = combined_returns['close'].rolling(window=60).cov(combined_returns['market_return'])
    #     rolling_var = combined_returns['market_return'].rolling(window=60).var()
    #     final_df['beta_coefficient'] = rolling_cov / rolling_var
    #     final_df['high_52w'] = final_df['high'].rolling(window=52*5, min_periods=1).max()
    #     is_new_high = final_df['high'] >= final_df['high_52w'].shift(1)
    #     new_high_dates = pd.Series(final_df.index, index=final_df.index).where(is_new_high).ffill()
    #     final_df['days_since_52w_high'] = (final_df.index - new_high_dates).dt.days

    #     final_df['stock_code'] = stock_code
    #     final_df = final_df.loc[start_date:end_date]
        
    #     final_df.reset_index(inplace=True)
        
    #     # ✨ [핵심 수정] pykrx가 인덱스를 '날짜'로 반환하는 경우에 대비해 컬럼명을 'date'로 강제 변경
    #     if '날짜' in final_df.columns:
    #         final_df.rename(columns={'날짜': 'date'}, inplace=True)
        
    #     final_df.drop_duplicates(subset=['date'], keep='first', inplace=True)
        
    #     final_columns = [
    #         'date', 'stock_code', 'foreigner_net_buy', 'institution_net_buy', 'program_net_buy',
    #         'trading_intensity', 'credit_ratio', 'short_volume', 'trading_value', 'per', 'pbr',
    #         'psr', 'dividend_yield', 'relative_strength', 'beta_coefficient',
    #         'days_since_52w_high', 'dist_from_ma20', 'historical_volatility_20d',
    #         'q_revenue_growth_rate', 'q_op_income_growth_rate'
    #     ]
        
    #     for col in final_columns:
    #         if col not in final_df.columns:
    #             final_df[col] = np.nan
        
    #     return final_df[final_columns]
        
    def _fetch_and_store_factors(self, start_date: date, end_date: date, stock_code: str) -> pd.DataFrame:
        """
        pykrx에서 팩터 데이터를 가져와 DB에 저장하고, 그 결과를 반환합니다.
        """
        logger.info(f"[{stock_code}] 신규 팩터 데이터 조회 및 저장: {start_date} ~ {end_date}")
        krx_df = self.get_daily_factors(start_date, end_date, stock_code)

        if not krx_df.empty:
            # DataFrame을 딕셔너리 리스트로 변환하여 전달
            success = self.db_manager.save_daily_factors(krx_df.to_dict('records'))
            
            if success:
                logger.info(f"[{stock_code}] {len(krx_df)}개 팩터 데이터 DB 저장 완료.")
            else:
                logger.error(f"[{stock_code}] 팩터 데이터 DB 저장 실패.")
        
        return krx_df

    def cache_factors(self, start_date: date, end_date: date, stock_code: str) -> pd.DataFrame:
        """
        [수정 완료] 캐시(DB)를 우선 확인하고, 없는 데이터만 pykrx에서 가져와 채운 후 최종 데이터를 반환합니다.
        """
        logger.info(f"[{stock_code}] 일별 팩터 데이터 캐싱 요청: {start_date} ~ {end_date}")
        
        db_df = self.db_manager.fetch_daily_factors(start_date, end_date, stock_code)
        
        db_existing_dates = set(pd.to_datetime(db_df['date']).dt.normalize()) if not db_df.empty else set()

        all_trading_dates = set(self.db_manager.get_all_trading_days(start_date, end_date))
        missing_dates = sorted(list(all_trading_dates - db_existing_dates))

        if not missing_dates:
            logger.info(f"[{stock_code}] 요청된 기간의 모든 팩터 데이터가 DB에 존재합니다.")
            return db_df

        logger.info(f"[{stock_code}] {len(missing_dates)}일치 팩터 데이터가 누락되어 추가 조회합니다.")
        
        api_df = pd.DataFrame()
        if missing_dates:
            start_range = missing_dates[0].date()
            end_range = missing_dates[-1].date()
            try:
                api_df = self._fetch_and_store_factors(start_range, end_range, stock_code)
            except Exception as e:
                logger.error(f"API로부터 팩터 데이터 가져오기 실패: {stock_code} - {str(e)}")

        if not api_df.empty:
            final_df = pd.concat([db_df, api_df], ignore_index=True)
            final_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
            final_df.sort_values(by='date', inplace=True)
            return final_df.reset_index(drop=True)
        else:
            return db_df
            
    # def cache_factors(self, start_date: date, end_date: date, stock_code: str) -> pd.DataFrame:
    #     """
    #     [최종 수정] 캐시(DB)를 우선 확인하고, 없는 데이터만 pykrx에서 가져와 채운 후 최종 데이터를 반환합니다.
    #     """
    #     logger.info(f"[{stock_code}] 일별 팩터 데이터 캐싱 요청: {start_date} ~ {end_date}")
        
    #     db_df = self.db_manager.fetch_daily_factors(start_date, end_date, stock_code)
        
    #     # [수정] 'date' 컬럼에서 기존 날짜를 가져옵니다.
    #     db_existing_dates = set(pd.to_datetime(db_df['date']).dt.normalize()) if not db_df.empty else set()

    #     all_trading_dates = set(self.db_manager.get_all_trading_days(start_date, end_date))
    #     missing_dates = sorted(list(all_trading_dates - db_existing_dates))

    #     if not missing_dates:
    #         logger.info(f"[{stock_code}] 요청된 기간의 모든 팩터 데이터가 DB에 존재합니다.")
    #         return db_df

    #     logger.info(f"[{stock_code}] {len(missing_dates)}일치 팩터 데이터가 누락되어 추가 조회합니다.")
        
    #      # 4. 누락된 데이터가 있을 경우, API 호출
    #     if missing_dates:
    #         api_fetched_dfs = []
    #         # 사용자의 간소화된 로직: 누락된 날짜의 최소/최대 구간을 한번에 조회
    #         start_range = missing_dates[0].date()
    #         end_range = missing_dates[-1].date()

    #         try:
    #             api_df = self._fetch_and_store_factors(start_range, end_range, stock_code)
    #             if not api_df.empty:
    #                 api_fetched_dfs.append(api_df)
    #         except Exception as e:
    #             logger.error(f"API로부터 팩터 데이터 가져오기 실패: {stock_code} - {str(e)}")

    #         # ✨ [핵심 수정] 데이터 통합 및 반환 로직
    #         if api_fetched_dfs:
    #             new_data_df = pd.concat(api_fetched_dfs, ignore_index=True)
    #             final_df = pd.concat([db_df, new_data_df], ignore_index=True)
    #             # 'date' 컬럼 기준으로 중복 제거 및 정렬
    #             final_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    #             final_df.sort_values(by='date', inplace=True)
    #             return final_df.reset_index(drop=True)
    #         else:
    #             # 새로 가져온 데이터가 없으면 기존 DB 데이터만 반환
    #             return db_df
    #     else:
    #         # 누락된 데이터가 없으면 DB 데이터만 반환
    #         return db_df


    def fetch_market_calendar(self, from_date: date, to_date: date) -> pd.DataFrame:
        """DB에서 시장 캘린더 데이터를 조회합니다."""
        return self.db_manager.fetch_market_calendar(from_date, to_date)

    def get_all_trading_days(self, from_date: date, to_date: date) -> list:
        """[래퍼] DB에서 모든 영업일 리스트를 조회합니다."""
        return self.db_manager.get_all_trading_days(from_date, to_date)
    
    def get_previous_trading_day(self, current_date: date) -> Optional[date]:
        """
        [수정] TradingManager를 통해 DB에서 직접 이전 영업일을 조회합니다.
        """
        # broker -> manager를 통해 DB 조회 기능에 접근
        prev_day = self.db_manager.get_previous_trading_day(current_date)
        if prev_day is None:
            logger.warning(f"{current_date}의 이전 영업일을 찾을 수 없습니다.")
        return prev_day
           

    def fetch_average_trading_values(self, universe_codes: List[str], start_date: date, end_date: date) -> Dict[str, float]:
        """[래퍼] DBManager의 fetch_average_trading_values 메서드를 호출합니다."""
        return self.db_manager.fetch_average_trading_values(universe_codes, start_date, end_date)


    def fetch_latest_factors_for_universe(self, universe_codes: List[str], current_date: date) -> pd.DataFrame:
        """[래퍼] DBManager의 fetch_latest_factors_for_universe 메서드를 호출합니다."""
        return self.db_manager.fetch_latest_factors_for_universe(universe_codes, current_date)
    


    def fetch_hmm_model_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        """[래퍼] HMM 모델 정보를 이름으로 조회합니다."""
        return self.db_manager.fetch_hmm_model_by_name(model_name)


    def fetch_strategy_profiles_by_model(self, model_id: int) -> pd.DataFrame:
        """[래퍼] 특정 모델 ID에 대한 모든 전략 프로파일을 조회합니다."""
        return self.db_manager.fetch_strategy_profiles_by_model(model_id)


    def fetch_latest_wf_model(self) -> Optional[Dict[str, Any]]:
        """[래퍼] 이름이 'wf_model_'로 시작하는 HMM 모델 중,
        가장 최근에 생성된(updated_at 기준) 모델 정보를 반환합니다."""
        return self.db_manager.fetch_latest_wf_model()