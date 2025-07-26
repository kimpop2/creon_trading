# manager/setup_manager.py 파일 전체를 아래 내용으로 교체합니다.

import logging
from datetime import date, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np # 수치 계산을 위해 numpy 임포트
from pykrx import stock
# 프로젝트 루트 경로를 sys.path에 추가 (임포트를 위함)
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from config.sector_stocks import sector_stocks
# 로거 설정
logger = logging.getLogger(__name__)

# --- 상수 정의 ---
KOSPI_INDEX_CODE = 'U001'  # 코스피 업종 지수 코드

class SetupManager:
    """
    장 마감 후 일별 팩터 데이터를 생성하고 관리하는 비즈니스 로직을 담당.
    Creon API와 DBManager를 통해 필요한 데이터를 가져와 가공하고 저장.
    """
    def __init__(self, api_client: CreonAPIClient, db_manager: DBManager):
        self.api_client = api_client
        self.db_manager = db_manager
        logger.info("SetupManager 초기화 완료.")
        self.api_client = api_client
        self.db_manager = db_manager
        
        self.stock_info_map: Dict[str, str] = {} # 종목 코드:종목명 매핑을 위한 캐시 ---
        self._load_stock_info_map() # 생성 시점에 종목명 캐시를 미리 만듭니다.
    def close(self):
        """DBManager의 연결을 종료합니다."""
        if self.db_manager:
            self.db_manager.close()
            logger.info("BacktestManager를 통해 DB 연결을 종료했습니다.")

    def _load_stock_info_map(self): # [수정] 반환 타입 힌트 제거
        """
        [수정됨] DB에서 모든 종목 정보를 가져와 self.stock_info_map에 직접 저장합니다.
        """
        logger.debug("종목 정보 맵(딕셔너리) 로딩 시작")
        try:
            stock_info_df = self.db_manager.fetch_stock_info() 
            if not stock_info_df.empty:
                stock_map = pd.Series(stock_info_df.stock_name.values, index=stock_info_df.stock_code).to_dict()
                
                # [핵심 수정] return 하는 대신, self.stock_info_map에 직접 할당합니다.
                self.stock_info_map = stock_map
                
                logger.debug(f"{len(self.stock_info_map)}개의 종목 정보 로딩 완료")
            else:
                logger.warning("DB에서 종목 정보를 가져오지 못했습니다. stock_info 테이블이 비어있을 수 있습니다.")
                self.stock_info_map = {} # 비어있는 경우에도 초기화
        except Exception as e:
            logger.error(f"종목 정보 로딩 중 오류 발생: {e}")
            self.stock_info_map = {} # 오류 발생 시에도 초기화
        

    def get_stock_name(self, stock_code: str) -> str:
        """
        미리 생성된 내부 캐시(self.stock_info_map)에서 종목명을 매우 빠르게 조회합니다.
        캐시에 없는 경우 "알 수 없음"을 반환합니다.
        """
        # API 호출 대신, 메모리에 있는 딕셔너리에서 즉시 값을 찾습니다.
        return self.stock_info_map.get(stock_code, "알 수 없음")
    
    def get_universe_codes(self) -> List[str]:
        """
        config/sector_stocks.py 에서 유니버스 종목 목록을 가져와
        종목명을 코드로 변환한 뒤, 유니크한 종목 코드 리스트를 반환합니다.
        """
        logger.info("유니버스 종목 코드 목록 생성을 시작합니다.")

        # 1. sector_stocks 설정에서 모든 종목 이름을 수집하고 중복을 제거합니다.
        all_stock_names = set()
        for stocks_in_sector in sector_stocks.values():
            for stock_name, theme in stocks_in_sector:
                all_stock_names.add(stock_name)
        
        logger.debug(f"고유 유니버스 종목명 {len(all_stock_names)}개 수집 완료.")

        # 2. 종목명을 종목 코드로 변환합니다.
        universe_codes = []
        for name in sorted(list(all_stock_names)): # 정렬하여 처리 순서 고정
            code = self.api_client.get_stock_code(name)
            
            # 3. 코드가 유효한 경우에만 리스트에 추가하고, 없으면 경고 로그를 남깁니다.
            if code:
                universe_codes.append(code)
            else:
                logger.warning(f"유니버스 구성 중 종목명을 코드로 변환할 수 없습니다: '{name}'. 해당 종목을 건너뜁니다.")
        
        logger.info(f"최종 유니버스 종목 코드 {len(universe_codes)}개 생성 완료.")
        return universe_codes
    
    

    def get_daily_factors(self, start_date: date, end_date: date, stock_code: str) -> pd.DataFrame:
        """
        pykrx를 사용하여 특정 기간의 일별 팩터 데이터를 조회하고 계산합니다. (진짜 최종 수정본)
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
            
        df_fundamental = stock.get_market_fundamental(start_date_str, end_date_str, pykrx_stock_code)
        df_trading = stock.get_market_trading_volume_by_date(start_date_str, end_date_str, pykrx_stock_code)
        df_short = stock.get_shorting_status_by_date(start_date_str, end_date_str, pykrx_stock_code)
        df_kospi = stock.get_index_ohlcv(lookback_start_date, end_date_str, "1001")
        
        df_ohlcv.rename(columns={'시가':'open', '고가':'high', '저가':'low', '종가':'close', '거래량':'volume', '거래대금': 'trading_value'}, inplace=True)
        final_df = df_ohlcv.copy()

        df_fundamental.rename(columns={'PER':'per', 'PBR':'pbr', 'DIV':'dividend_yield'}, inplace=True)
        final_df = pd.merge(final_df, df_fundamental[['per', 'pbr', 'dividend_yield']], left_index=True, right_index=True, how='left')

        df_trading.rename(columns={'외국인합계':'foreigner_net_buy', '기관합계':'institution_net_buy'}, inplace=True)
        final_df = pd.merge(final_df, df_trading[['foreigner_net_buy', 'institution_net_buy']], left_index=True, right_index=True, how='left')

        if not df_short.empty and '공매도' in df_short.columns:
            df_short.rename(columns={'공매도':'short_volume'}, inplace=True)
            final_df = pd.merge(final_df, df_short[['short_volume']], left_index=True, right_index=True, how='left')
        
        final_df['ma20'] = final_df['close'].rolling(window=20).mean()
        final_df['dist_from_ma20'] = (final_df['close'] / final_df['ma20'] - 1) * 100
        daily_returns = final_df['close'].pct_change()
        final_df['historical_volatility_20d'] = daily_returns.rolling(window=20).std() * np.sqrt(252)
        market_returns = df_kospi['종가'].pct_change()
        market_returns.name = 'market_return'
        combined_returns = pd.merge(daily_returns, market_returns, left_index=True, right_index=True, how='left')
        final_df['relative_strength'] = (combined_returns['close'] - combined_returns['market_return']) * 100
        rolling_cov = combined_returns['close'].rolling(window=60).cov(combined_returns['market_return'])
        rolling_var = combined_returns['market_return'].rolling(window=60).var()
        final_df['beta_coefficient'] = rolling_cov / rolling_var
        final_df['high_52w'] = final_df['high'].rolling(window=52*5, min_periods=1).max()
        is_new_high = final_df['high'] >= final_df['high_52w'].shift(1)
        new_high_dates = pd.Series(final_df.index, index=final_df.index).where(is_new_high).ffill()
        final_df['days_since_52w_high'] = (final_df.index - new_high_dates).dt.days

        final_df['stock_code'] = stock_code
        final_df = final_df.loc[start_date:end_date]
        
        final_df.reset_index(inplace=True)
        
        # ✨ [핵심 수정] pykrx가 인덱스를 '날짜'로 반환하는 경우에 대비해 컬럼명을 'date'로 강제 변경
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
        
        return final_df[final_columns]
        
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
        [최종 수정] 캐시(DB)를 우선 확인하고, 없는 데이터만 pykrx에서 가져와 채운 후 최종 데이터를 반환합니다.
        """
        logger.info(f"[{stock_code}] 일별 팩터 데이터 캐싱 요청: {start_date} ~ {end_date}")
        
        db_df = self.db_manager.fetch_daily_factors(start_date, end_date, stock_code)
        
        # [수정] 'date' 컬럼에서 기존 날짜를 가져옵니다.
        db_existing_dates = set(pd.to_datetime(db_df['date']).dt.normalize()) if not db_df.empty else set()

        all_trading_dates = set(self.db_manager.get_all_trading_days(start_date, end_date))
        missing_dates = sorted(list(all_trading_dates - db_existing_dates))

        if not missing_dates:
            logger.info(f"[{stock_code}] 요청된 기간의 모든 팩터 데이터가 DB에 존재합니다.")
            return db_df

        logger.info(f"[{stock_code}] {len(missing_dates)}일치 팩터 데이터가 누락되어 추가 조회합니다.")
        
         # 4. 누락된 데이터가 있을 경우, API 호출
        if missing_dates:
            api_fetched_dfs = []
            # 사용자의 간소화된 로직: 누락된 날짜의 최소/최대 구간을 한번에 조회
            start_range = missing_dates[0].date()
            end_range = missing_dates[-1].date()

            try:
                api_df = self._fetch_and_store_factors(start_range, end_range, stock_code)
                if not api_df.empty:
                    api_fetched_dfs.append(api_df)
            except Exception as e:
                logger.error(f"API로부터 팩터 데이터 가져오기 실패: {stock_code} - {str(e)}")

            # ✨ [핵심 수정] 데이터 통합 및 반환 로직
            if api_fetched_dfs:
                new_data_df = pd.concat(api_fetched_dfs, ignore_index=True)
                final_df = pd.concat([db_df, new_data_df], ignore_index=True)
                # 'date' 컬럼 기준으로 중복 제거 및 정렬
                final_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
                final_df.sort_values(by='date', inplace=True)
                return final_df.reset_index(drop=True)
            else:
                # 새로 가져온 데이터가 없으면 기존 DB 데이터만 반환
                return db_df
        else:
            # 누락된 데이터가 없으면 DB 데이터만 반환
            return db_df

    # def cache_factors(self, start_date: date, end_date: date, stock_code: str) -> pd.DataFrame:
    #     """
    #     캐시(DB)를 우선 확인하고, 없는 데이터만 pykrx에서 가져와 채운 후 최종 데이터를 반환합니다.
    #     """
    #     logger.info(f"[{stock_code}] 일별 팩터 데이터 캐싱 요청: {start_date} ~ {end_date}")
        
    #     # db_manager에 fetch_daily_factors가 구현되어 있다고 가정
    #     db_df = self.db_manager.fetch_daily_factors(start_date, end_date, stock_code)
    #     db_existing_dates = set(db_df.index.normalize()) if not db_df.empty else set()

    #     # 2. 누락된 날짜 계산
    #     all_trading_dates = set(self.db_manager.get_all_trading_days(start_date, end_date))
    #     missing_dates = sorted(list(all_trading_dates - db_existing_dates))

    #     # 3. 최종 조회일(to_date) 강제 포함 (데이터 무결성 보장)
    #     # to_date_ts = pd.Timestamp(end_date).normalize()
    #     # if to_date_ts in all_trading_dates and to_date_ts not in {pd.Timestamp(d).normalize() for d in missing_dates}:
    #     #     missing_dates.append(to_date_ts)

    #     # 4. 누락/최종일 데이터가 있을 경우, 전용 헬퍼 함수 호출
    #     if missing_dates:
    #         api_fetched_dfs = []
    #         # 누락된 날짜들의 시작과 끝을 찾아 API 호출
    #         start_range = missing_dates[0].date()
    #         end_range = missing_dates[-1].date()
            
    #         try:
    #             # 일봉 전용 헬퍼 함수 호출
    #             api_df = self._fetch_and_store_factors(start_range, end_range, stock_code)
    #             if not api_df.empty:
    #                 api_fetched_dfs.append(api_df)
    #         except Exception as e:
    #             logger.error(f"API로부터 일봉 데이터 가져오기 실패: {stock_code} - {str(e)}")

    #         # 5. 데이터 통합 및 반환
    #         final_df = pd.concat([db_df] + api_fetched_dfs)
    #         final_df = final_df[~final_df.index.duplicated(keep='last')]
    #         return final_df.sort_index()
    #     else:
    #         # 5. 누락된 데이터가 없으면 DB 데이터만 반환
    #         return db_df

    
    def run_daily_factor_update(self, target_date: date) -> bool:
        logger.info(f"--- {target_date} 기준 일별 팩터 업데이트 작업 시작 ---")

        try:
            # 1. 대상 종목 및 시장 지수 데이터 로드
            all_stocks = self._get_all_target_stocks()
            if not all_stocks:
                logger.warning("팩터를 계산할 대상 종목이 없습니다. 작업을 중단합니다.")
                return False
            logger.info(f"총 {len(all_stocks)}개 종목에 대한 팩터 계산을 시작합니다.")

            lookback_start_date = target_date - pd.DateOffset(years=1)
            kospi_df = self.db_manager.fetch_daily_price(KOSPI_INDEX_CODE, lookback_start_date, target_date)
            if kospi_df.empty:
                logger.error("상대강도 계산에 필요한 코스피 지수 데이터를 가져올 수 없습니다. 작업을 중단합니다.")
                return False

            # 2. 소스 데이터 대량 조회 (MarketEye)
            marketeye_data_map = self.api_client.get_market_eye_datas(stock_codes=all_stocks)
            if not marketeye_data_map:
                logger.error("MarketEye로부터 소스 데이터를 가져오지 못했습니다. 작업을 중단합니다.")
                return False

            all_factors_list = []
            for stock_code in all_stocks:
                marketeye_data = marketeye_data_map.get(stock_code)
                if not marketeye_data:
                    logger.warning(f"[{stock_code}]에 대한 MarketEye 데이터가 없습니다. 건너뜁니다.")
                    continue

                daily_ohlcv_df = self.db_manager.fetch_daily_price(stock_code, lookback_start_date, target_date)
                if len(daily_ohlcv_df) < 20: # 최소 20일 데이터는 있어야 의미있는 계산 가능
                    logger.warning(f"[{stock_code}]의 일봉 데이터가 20일 미만이라 팩터 계산을 건너뜁니다.")
                    continue

                # 3. 팩터 계산
                factors = self._calculate_all_factors(
                    stock_code=stock_code,
                    target_date=target_date,
                    marketeye_data=marketeye_data,
                    daily_ohlcv_df=daily_ohlcv_df,
                    kospi_df=kospi_df
                )
                all_factors_list.append(factors)

            # 4. 계산된 팩터 DB에 일괄 저장
            if not all_factors_list:
                logger.warning("저장할 팩터 데이터가 생성되지 않았습니다.")
                return True

            success = self.db_manager.save_daily_factors(all_factors_list)
            if success:
                logger.info(f"--- 총 {len(all_factors_list)}건의 일별 팩터 데이터 저장 완료 ---")
            else:
                logger.error("--- 일별 팩터 데이터 저장 실패 ---")
            return success

        except Exception as e:
            logger.critical(f"일별 팩터 업데이트 작업 중 심각한 오류 발생: {e}", exc_info=True)
            return False

    def _get_all_target_stocks(self) -> List[str]:
        return self.db_manager.get_all_stock_codes()

    def _calculate_all_factors(self, stock_code: str, target_date: date, marketeye_data: dict, daily_ohlcv_df: pd.DataFrame, kospi_df: pd.DataFrame) -> Dict[str, Any]:
        logger.debug(f"[{stock_code}] 팩터 계산 중...")

        factors = {
            'date': target_date,
            'stock_code': stock_code,
            # MarketEye에서 직접 가져오는 값들
            'foreigner_net_buy': marketeye_data.get('foreigner_net_buy'),
            'institution_net_buy': marketeye_data.get('institution_net_buy'),
            'program_net_buy': marketeye_data.get('program_net_buy'),
            'trading_intensity': marketeye_data.get('trading_intensity'),
            'credit_ratio': marketeye_data.get('credit_ratio'),
            'short_volume': marketeye_data.get('short_volume'),
            'trading_value': marketeye_data.get('trading_value'),
            'per': marketeye_data.get('per'),
            'dividend_yield': marketeye_data.get('dividend_yield'),
            'beta_coefficient': marketeye_data.get('beta_coefficient'),
            'q_revenue_growth_rate': marketeye_data.get('q_revenue_growth_rate'),
            'q_op_income_growth_rate': marketeye_data.get('q_op_income_growth_rate'),
        }

        # --- MarketEye 데이터를 이용한 계산 팩터 (PBR, PSR) ---
        bps = marketeye_data.get('bps')
        sps = marketeye_data.get('sps')
        current_price = marketeye_data.get('current_price')

        factors['pbr'] = (current_price / bps) if bps and current_price and bps > 0 else None
        factors['psr'] = (current_price / sps) if sps and current_price and sps > 0 else None

        # --- 일봉 데이터를 이용한 계산 팩터 ---
        try:
            # 20일 이평선 이격도
            sma20 = daily_ohlcv_df['close'].rolling(window=20).mean().iloc[-1]
            factors['dist_from_ma20'] = ((current_price - sma20) / sma20 * 100) if sma20 > 0 else None

            # 20일 역사적 변동성
            daily_returns = daily_ohlcv_df['close'].pct_change()
            volatility = daily_returns.rolling(window=20).std().iloc[-1]
            factors['historical_volatility_20d'] = volatility * np.sqrt(252) if volatility else None # 연율화

            # 52주 신고가 후 경과일
            df_52w = daily_ohlcv_df.last('52W')
            date_of_52w_high = df_52w['high'].idxmax()
            factors['days_since_52w_high'] = (pd.Timestamp(target_date) - date_of_52w_high).days
            #final_df['days_since_52w_high'] = (final_df.index - new_high_dates).dt.days # <- '.dt' 추가
            # 시장 대비 상대강도 점수 (RS)
            stock_return = daily_returns.iloc[-1]
            market_return = kospi_df['close'].pct_change().iloc[-1]
            factors['relative_strength'] = (stock_return - market_return) * 100 if pd.notna(stock_return) and pd.notna(market_return) else None

        except (IndexError, KeyError) as e:
            logger.warning(f"[{stock_code}] 일봉 데이터 기반 팩터 계산 중 오류 (데이터 부족 가능성): {e}")
        except Exception as e:
            logger.error(f"[{stock_code}] 일봉 데이터 기반 팩터 계산 중 예외 발생: {e}", exc_info=True)

        return factors