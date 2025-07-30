# manager/trading_manager.py (수정된 부분)

import logging
import pandas as pd
from datetime import datetime, date, timedelta, time
import time as pytime # time.sleep 사용을 위해 임포트
import sys
import os
import json
from typing import Optional, List, Dict, Any, Tuple

# --- 로거 설정 ---
logger = logging.getLogger(__name__)

# sys.path에 프로젝트 루트 추가 (db_manager 및 creon_api 임포트를 위함)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
logger.debug(f"Project root added to sys.path: {project_root}")

from manager.db_manager import DBManager
from api.creon_api import CreonAPIClient # CreonAPIClient 클래스 이름 일치
from util.strategies_util import calculate_sma, calculate_rsi, calculate_ema, calculate_macd
from config.sector_stocks import sector_stocks
from config.settings import COMMON_PARAMS
class TradingManager:
    """
    자동매매 시스템의 데이터 관련 모든 비즈니스 로직을 담당하는 클래스.
    DB, Creon API와의 통신을 관리하고, 가공된 데이터를 제공하며,
    실시간 데이터를 처리하고 시스템 상태를 영속화합니다.
    """
    def __init__(self, 
                 api_client: CreonAPIClient, 
                 db_manager: DBManager):
        
        self.api_client = api_client
        self.db_manager = db_manager
        
        self.stock_info_map: Dict[str, str] = {} # 종목 코드:종목명 매핑을 위한 캐시 ---
        self._load_stock_info_map() # 생성 시점에 종목명 캐시를 미리 만듭니다.

        self.realtime_ohlcv_cache: Dict[str, pd.DataFrame] = {} # 종목별 실시간 분봉 데이터를 저장

        logger.info(f"TradingManager 초기화 완료: {len(self.stock_info_map)} 종목, CreonAPIClient 및 DBManager 연결")

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
        
    # def _load_stock_names_from_db(self):
    #     """
    #     시스템 시작 시 DB의 stock_info 테이블에서 모든 종목 정보를 가져와
    #     self.stock_info_map 딕셔너리에 캐시합니다.
    #     """
    #     logger.info("DB로부터 전체 종목명 캐시를 시작합니다.")
    #     try:
    #         # DBManager를 통해 모든 종목 정보를 DataFrame으로 조회
    #         stock_info_df = self.db_manager.fetch_stock_info() 
    #         if not stock_info_df.empty:
    #             # 'stock_code'를 키로, 'stock_name'을 값으로 하는 딕셔너리 생성
    #             self.stock_info_map = pd.Series(
    #                 stock_info_df.stock_name.values, 
    #                 index=stock_info_df.stock_code
    #             ).to_dict()
    #             logger.info(f"종목명 {len(self.stock_info_map)}건 캐시 완료.")
    #         else:
    #             logger.warning("DB에서 종목 정보를 가져오지 못했습니다. stock_info 테이블이 비어있을 수 있습니다.")
    #     except Exception as e:
    #         logger.error(f"종목 정보 캐시 중 오류 발생: {e}")


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
    
    def cache_daily_ohlcv(self, stock_code: str, from_date: date, to_date: date) -> pd.DataFrame:
        """
        [최종 검토 완료] DB와 증권사 API를 사용하여 특정 종목의 일봉 데이터를 캐싱하고 반환합니다.
        일봉 전용 헬퍼 함수 `_fetch_and_store_daily_range`를 호출합니다.
        """
        logger.debug(f"일봉 데이터 캐싱 시작: {stock_code} ({from_date.strftime('%Y%m%d')} ~ {to_date.strftime('%Y%m%d')})")

        # 1. DB에서 기존 데이터 조회
        db_df = self.db_manager.fetch_daily_price(stock_code, from_date, to_date)
        db_existing_dates = set(db_df.index.normalize()) if not db_df.empty else set()
        
        # 2. 누락된 날짜 계산
        all_trading_dates = set(self.db_manager.get_all_trading_days(from_date, to_date))
        missing_dates = sorted(list(all_trading_dates - db_existing_dates))
        
        # 3. 최종 조회일(to_date) 강제 포함 (데이터 무결성 보장)
        to_date_ts = pd.Timestamp(to_date).normalize()
        if to_date_ts in all_trading_dates and to_date_ts not in {pd.Timestamp(d).normalize() for d in missing_dates}:
            missing_dates.append(to_date_ts)

        # 4. 누락/최종일 데이터가 있을 경우, 전용 헬퍼 함수 호출
        if missing_dates:
            api_fetched_dfs = []
            # 누락된 날짜들의 시작과 끝을 찾아 API 호출
            start_range = missing_dates[0].date()
            end_range = missing_dates[-1].date()
            
            try:
                # 일봉 전용 헬퍼 함수 호출
                api_df = self._fetch_and_store_daily_range(stock_code, start_range, end_range)
                if not api_df.empty:
                    api_fetched_dfs.append(api_df)
            except Exception as e:
                logger.error(f"API로부터 일봉 데이터 가져오기 실패: {stock_code} - {str(e)}")

            # 5. 데이터 통합 및 반환
            final_df = pd.concat([db_df] + api_fetched_dfs)
            final_df = final_df[~final_df.index.duplicated(keep='last')]
            return final_df.sort_index()
        else:
            # 5. 누락된 데이터가 없으면 DB 데이터만 반환
            return db_df

    def cache_minute_ohlcv(self, stock_code: str, from_date: date, to_date: date, interval: int = 1) -> pd.DataFrame:
        """
        [최종 검토 완료] 분봉 데이터를 캐싱하고 반환합니다. 
        분봉 전용 헬퍼 함수 `_fetch_and_store_minute_range`를 호출합니다.
        """
        logger.debug(f"분봉 데이터 캐싱 시작: {stock_code} ({from_date} ~ {to_date}), Interval: {interval}분")
        
        # 1. DB에서 기존 데이터 조회
        db_df = self.db_manager.fetch_minute_price(stock_code, from_date, to_date)
        db_existing_dates = {pd.Timestamp(d).normalize() for d in db_df.index.date} if not db_df.empty else set()
        
        # 2. 누락된 날짜 계산
        all_trading_dates = set(self.db_manager.get_all_trading_days(from_date, to_date))
        missing_dates = sorted(list(all_trading_dates - db_existing_dates))

        # 3. 최종 조회일(to_date) 강제 포함 (데이터 무결성 보장)
        to_date_ts = pd.Timestamp(to_date).normalize()
        if to_date_ts in all_trading_dates and to_date_ts not in {pd.Timestamp(d).normalize() for d in missing_dates}:
            missing_dates.append(to_date_ts)
        
        if not missing_dates:
            return db_df

        # 4. 누락/최종일 데이터가 있을 경우, 전용 헬퍼 함수 호출
        api_fetched_dfs = []
        start_range = missing_dates[0].date()
        end_range = missing_dates[-1].date()

        try:
            # 분봉 전용 헬퍼 함수 호출
            api_df = self._fetch_and_store_minute_range(stock_code, start_range, end_range)
            if not api_df.empty:
                api_fetched_dfs.append(api_df)
        except Exception as e:
            logger.error(f"API 데이터 가져오기 실패: {stock_code} - {str(e)}")

        # 5. 데이터 통합 및 반환
        if api_fetched_dfs:
            final_df = pd.concat([db_df] + api_fetched_dfs).sort_index()
            final_df = final_df[~final_df.index.duplicated(keep='last')]
            return final_df
        else:
            return db_df

    def _fetch_and_store_daily_range(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """[신규] '일봉' 데이터만 가져와 DB에 저장하고 반환하는 전용 함수"""
        logger.debug(f"API로부터 {stock_code} 일봉 데이터 요청: {start_date} ~ {end_date}")
        
        try:
            api_df_part = self.api_client.get_daily_ohlcv(stock_code, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
            
            if api_df_part.empty:
                return pd.DataFrame()

            api_df_part['stock_code'] = stock_code

            df_for_saving = api_df_part.reset_index()
            data_to_save_list = []
            for _, row in df_for_saving.iterrows():
                record = {
                    'stock_code': stock_code, 'date': row['date'].date(),
                    'open': row['open'], 'high': row['high'], 'low': row['low'],
                    'close': row['close'], 'volume': row['volume'],
                    'change_rate': row.get('change_rate', 0.0), 'trading_value': row.get('trading_value', 0)
                }
                data_to_save_list.append(record)
            
            self.db_manager.save_daily_price(data_to_save_list)
            logger.debug(f"API로부터 {stock_code}의 일봉 데이터 {len(api_df_part)}개 DB 저장 완료")
            
            return api_df_part
            
        except Exception as e:
            logger.error(f"API 일봉 호출 또는 DB 저장 실패: {stock_code} - {e}", exc_info=True)
            return pd.DataFrame()

    def _fetch_and_store_minute_range(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """[신규] '분봉' 데이터만 가져와 DB에 저장하고 반환하는 전용 함수"""
        logger.debug(f"API로부터 {stock_code} 분봉 데이터 요청: {start_date} ~ {end_date}")
        
        try:
            api_df_part = self.api_client.get_minute_ohlcv(stock_code, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
            
            if api_df_part.empty:
                return pd.DataFrame()

            api_df_part['stock_code'] = stock_code
            api_df_part['change_rate'] = 0.0
            api_df_part['trading_value'] = 0

            df_for_saving = api_df_part.reset_index()
            data_to_save_list = []
            for _, row in df_for_saving.iterrows():
                record = {
                    'stock_code': stock_code, 'datetime': row['datetime'],
                    'open': row['open'], 'high': row['high'], 'low': row['low'],
                    'close': row['close'], 'volume': row['volume'],
                    'change_rate': row.get('change_rate', 0.0), 'trading_value': row.get('trading_value', 0)
                }
                data_to_save_list.append(record)
            
            self.db_manager.save_minute_price(data_to_save_list)
            logger.debug(f"API로부터 {stock_code}의 분봉 데이터 {len(api_df_part)}개 DB 저장 완료")
            
            return api_df_part
            
        except Exception as e:
            logger.error(f"API 분봉 호출 또는 DB 저장 실패: {stock_code} - {e}", exc_info=True)
            return pd.DataFrame()

    # def get_stock_info_map(self) -> dict:
    #     """
    #     DB의 stock_info 테이블에서 모든 종목의 코드와 이름을 가져와
    #     {'종목코드': '종목명'} 형태의 딕셔너리로 반환합니다.
    #     """
    #     logger.debug("종목 정보 맵(딕셔너리) 로딩 시작")
    #     try:
    #         # fetch_stock_info()를 인자 없이 호출하여 모든 종목 정보를 가져옵니다.
    #         stock_info_df = self.db_manager.fetch_stock_info() 
    #         if not stock_info_df.empty:
    #             # 'stock_code'를 인덱스로, 'stock_name'을 값으로 하는 딕셔너리 생성
    #             stock_map = pd.Series(stock_info_df.stock_name.values, index=stock_info_df.stock_code).to_dict()
    #             logger.debug(f"{len(stock_map)}개의 종목 정보 로딩 완료")
    #             return stock_map
    #         else:
    #             logger.warning("DB에서 종목 정보를 가져오지 못했습니다. stock_info 테이블이 비어있을 수 있습니다.")
    #             return {}
    #     except Exception as e:
    #         logger.error(f"종목 정보 로딩 중 오류 발생: {e}")
    #         return {}
        



    # --- 데이터 수집 및 캐싱 관련 메소드 ---
    def fetch_daily_ohlcv(self, stock_code: str, from_date: date, to_date: date) -> pd.DataFrame:
        """
        DB에서 일봉 OHLCV 데이터를 조회합니다.
        필요시 Creon API를 통해 데이터를 업데이트할 수 있습니다.
        """
        df = self.db_manager.fetch_daily_price(stock_code, from_date, to_date)
        if df.empty:
            logger.warning(f"DB에 {stock_code}의 일봉 데이터가 없습니다. Creon API를 통해 조회 시도합니다.")
            # Creon API를 통해 데이터 조회 및 저장 로직 추가
            # get_price_data는 count 기반이므로, 기간 기반으로 가져오려면 여러 번 호출해야 할 수 있음.
            # 여기서는 단순화를 위해 마지막 365개 일봉을 가져오는 것으로 가정.
            df_from_api = self.api_client.get_price_data(stock_code, 'D', (to_date - from_date).days + 1)
            if not df_from_api.empty:
                # DB에 저장
                df_from_api['stock_code'] = stock_code
                df_from_api['stock_name'] = self.get_stock_name(stock_code)
                df_from_api.rename(columns={'datetime': 'date'}, inplace=True) # 컬럼명 일치
                df_from_api['date'] = df_from_api['date'].dt.date # datetime을 date로 변환
                # DBManager의 insert_df_to_db는 if_exists='append'만 지원하므로, 중복 방지를 위해 직접 SQL 사용
                conn = self.db_manager.get_db_connection()
                if conn:
                    try:
                        with conn.cursor() as cursor:
                            for index, row in df_from_api.iterrows():
                                sql = """
                                    INSERT INTO daily_price (stock_code, stock_name, date, open, high, low, close, volume)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                    ON DUPLICATE KEY UPDATE
                                        open = VALUES(open), high = VALUES(high), low = VALUES(low), close = VALUES(close), volume = VALUES(volume),
                                        updated_at = CURRENT_TIMESTAMP
                                """
                                params = (
                                    row['stock_code'], row['stock_name'], row['date'],
                                    row['open'], row['high'], row['low'], row['close'], row['volume']
                                )
                                cursor.execute(sql, params)
                            conn.commit()
                            logger.info(f"종목 {stock_code}의 일봉 데이터 {len(df_from_api)}건 API 조회 및 DB 업데이트 완료.")
                            df = self.db_manager.fetch_daily_price(stock_code, from_date, to_date) # 새로 저장된 데이터 다시 로드
                    except Exception as e:
                        logger.error(f"일봉 데이터 API 조회 및 DB 저장 중 오류: {e}")
                        conn.rollback()
            else:
                logger.warning(f"Creon API에서도 {stock_code}의 일봉 데이터를 가져올 수 없습니다.")
        return df

    def fetch_minute_ohlcv(self, stock_code: str, from_datetime: datetime, to_datetime: datetime) -> pd.DataFrame:
        """
        DB에서 분봉 OHLCV 데이터를 조회합니다.
        필요시 Creon API를 통해 데이터를 업데이트할 수 있습니다.
        """
        df = self.db_manager.fetch_minute_price(stock_code, from_datetime, to_datetime)
        if df.empty:
            logger.warning(f"DB에 {stock_code}의 분봉 데이터가 없습니다. Creon API를 통해 조회 시도합니다.")
            # Creon API를 통해 데이터 조회 및 저장 로직 추가
            # get_price_data는 count 기반이므로, 기간 기반으로 가져오려면 여러 번 호출해야 할 수 있음.
            # 여기서는 편의상 최근 200개 분봉을 가져오는 것으로 가정.
            df_from_api = self.api_client.get_price_data(stock_code, 'm', 200) # 200분봉
            if not df_from_api.empty:
                # DB에 저장
                df_from_api['stock_code'] = stock_code
                df_from_api['stock_name'] = self.get_stock_name(stock_code)
                # DBManager의 insert_df_to_db는 if_exists='append'만 지원하므로, 중복 방지를 위해 직접 SQL 사용
                conn = self.db_manager.get_db_connection()
                if conn:
                    try:
                        with conn.cursor() as cursor:
                            for index, row in df_from_api.iterrows():
                                sql = """
                                    INSERT INTO minute_price (stock_code, stock_name, datetime, open, high, low, close, volume)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                    ON DUPLICATE KEY UPDATE
                                        open = VALUES(open), high = VALUES(high), low = VALUES(low), close = VALUES(close), volume = VALUES(volume),
                                        updated_at = CURRENT_TIMESTAMP
                                """
                                params = (
                                    row['stock_code'], row['stock_name'], row['datetime'],
                                    row['open'], row['high'], row['low'], row['close'], row['volume']
                                )
                                cursor.execute(sql, params)
                            conn.commit()
                            logger.info(f"종목 {stock_code}의 분봉 데이터 {len(df_from_api)}건 API 조회 및 DB 업데이트 완료.")
                            df = self.db_manager.fetch_minute_price(stock_code, from_datetime, to_datetime) # 새로 저장된 데이터 다시 로드
                    except Exception as e:
                        logger.error(f"분봉 데이터 API 조회 및 DB 저장 중 오류: {e}")
                        conn.rollback()
            else:
                logger.warning(f"Creon API에서도 {stock_code}의 분봉 데이터를 가져올 수 없습니다.")
        return df

    def fetch_market_calendar(self, from_date: date, to_date: date) -> pd.DataFrame:
        return self.db_manager.fetch_market_calendar(from_date, to_date)
        
    def get_previous_trading_day(self, current_date: date) -> Optional[date]:
        """
        [수정] TradingManager를 통해 DB에서 직접 이전 영업일을 조회합니다.
        """
        # broker -> manager를 통해 DB 조회 기능에 접근
        prev_day = self.db_manager.get_previous_trading_day(current_date)
        if prev_day is None:
            logger.warning(f"{current_date}의 이전 영업일을 찾을 수 없습니다.")
        return prev_day
    
    def fetch_latest_daily_portfolio(self) -> Optional[Dict[str, Any]]:
        return self.db_manager.fetch_latest_daily_portfolio()

    def fetch_trading_logs(self, start_date: date, end_date: date) -> pd.DataFrame:
        return self.db_manager.fetch_trading_logs(start_date, end_date)

    def get_db_manager(self) -> DBManager:
        """DBManager 인스턴스를 전달하기 위한 메서드"""
        return self.db_manager

    def close_db_connection(self):
        """DB 연결을 종료합니다."""
        self.db_manager.close()    
    
    def get_account_balance(self) -> Dict[str, Any]:
        """
        Creon API를 통해 현재 계좌 잔고를 조회하고 반환합니다.
        실제 계좌 정보를 Trading 클래스에 제공합니다.
        """
        balance_info = self.api_client.get_current_account_balance()
        if balance_info:
            logger.info(f"계좌 잔고 조회 성공: 현금 {balance_info.get('cash_balance'):,.0f}원")
            return balance_info
        logger.error("계좌 잔고 조회 실패.")
        return {}

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Creon API를 통해 현재 보유 종목 정보를 조회하고 반환합니다.
        current_positions 테이블과 동기화하는 로직이 필요합니다.
        """
        api_positions = self.api_client.get_current_positions()
        
        # DB의 current_positions와 API 데이터를 동기화
        db_positions = {pos['stock_code']: pos for pos in self.db_manager.fetch_current_positions()}

        # 1. API에는 있고 DB에는 없는 경우 (새로 매수) -> DB에 저장
        # 2. DB에는 있고 API에는 없는 경우 (전량 매도 또는 보유량 0) -> DB에서 삭제
        # 3. 둘 다 있는 경우 (수량, 평단가 등 변경) -> DB 업데이트

        updated_codes = set()
        for api_pos in api_positions:
            stock_code = api_pos['stock_code']
            stock_name = api_pos['stock_name']
            quantity = api_pos.get('quantity', 0)
            sell_avail_qty = api_pos.get('sell_avail_qty')
            eval_profit_loss = api_pos.get('eval_profit_loss')
            eval_return_rate = api_pos.get('eval_return_rate')
            # DB에서 기존 포지션 정보를 조회
            db_pos = db_positions.get(stock_code)
            entry_date = db_pos['entry_date'] if db_pos and 'entry_date' in db_pos else datetime.now().date()
            position_data = {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'quantity': quantity,
                # sell_avail_qty가 None이면 전체 수량(quantity)을 사용, 아니면 원래 값을 사용
                'sell_avail_qty': sell_avail_qty if sell_avail_qty is not None else 0,
                'avg_price': api_pos.get('avg_price', 0.0),
                # 평가손익과 수익률이 None이면 0.0으로 처리
                'eval_profit_loss': eval_profit_loss if eval_profit_loss is not None else 0.0,
                'eval_return_rate': eval_return_rate if eval_return_rate is not None else 0.0,
                'entry_date': entry_date # 계산된 entry_date 사용
            }
            self.db_manager.save_current_position(position_data)
            updated_codes.add(stock_code)
        logger.info(f"Creon API로부터 현재 보유 종목 {len(api_positions)}건 동기화 완료.")

        # DB에만 있는 종목은 삭제
        for db_code in db_positions.keys():
            if db_code not in updated_codes:
                self.db_manager.delete_current_position(db_code)
                logger.info(f"DB에서 삭제된 보유 종목: {db_code} (API에 없음)")

        # 최종적으로 DB에서 최신 상태를 가져와 반환
        return self.db_manager.fetch_current_positions()

    def get_unfilled_orders(self) -> List[Dict[str, Any]]:
        """
        Creon API를 통해 미체결 주문 내역을 조회하고 반환합니다.
        trading_log 테이블과 동기화하거나, 실시간 미체결 내역 구독을 통해 관리합니다.
        """
        unfilled_orders = self.api_client.get_unfilled_orders()
        # 미체결 주문은 trading_log에 '접수' 또는 '부분체결' 상태로 기록되어 있어야 함
        # 이 함수는 주로 실시간 미체결 내역 구독 서비스와 연동되어 상태를 업데이트할 것임.
        if unfilled_orders:
            logger.info(f"Creon API로부터 미체결 주문 {len(unfilled_orders)}건 조회 완료.")
        else:
            logger.debug("현재 미체결 주문이 없습니다.")
        return unfilled_orders

    # --- 상태 정보 영속화 관련 메소드 (DBManager 래핑) ---

    def save_daily_signals(self, signal_data: Dict[str, Any]) -> bool:
        """단일 매매 신호를 DB에 저장/업데이트합니다."""
        return self.db_manager.save_daily_signal(signal_data)

    def load_daily_signals(self, signal_date: date, is_executed: Optional[bool] = None, signal_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        특정 날짜의 매매 신호를 DB에서 로드하여 딕셔너리 형태로 반환합니다.
        """
        signals_list = self.db_manager.fetch_daily_signals(signal_date, is_executed)
        loaded_signals = {}
        for s in signals_list:
            # signal_type 인자가 제공되었고, 현재 신호의 타입과 일치하지 않으면 건너_ㅂ니다.
            if signal_type is not None and s['signal_type'] != signal_type:
                continue
            loaded_signals[s['stock_code']] = {
                'signal_type': s['signal_type'],
                'target_price': s['target_price'],
                'target_quantity': s['target_quantity'],
                'strategy_name': s['strategy_name'],
                'is_executed': s['is_executed'],
                'signal_id': s['signal_id'],
                'executed_order_id': s['executed_order_id'] # 추가
            }
        logger.info(f"{signal_date}의 활성 신호 {len(loaded_signals)}건 로드 완료.")
        return loaded_signals

    # --- 추가 유틸리티 메소드 (BacktestManager에서 유용했던 기능들) ---
    def get_market_df(self, target_date: date) -> pd.DataFrame:
        """
        주어진 날짜의 시장 데이터를 DataFrame으로 반환합니다.
        (현재는 backtest_manager의 역할을 TradingManager가 대체)
        """
        # TODO: 실제 시장 데이터 (코스피/코스닥 지수, 시총, 거래량 등)를 Creon API에서 가져와
        # DB에 저장하고 여기서 조회하는 로직 필요. 지금은 더미 데이터나 빈 DataFrame 반환.
        logger.warning("get_market_df는 현재 더미 데이터를 반환합니다. 실제 시장 데이터를 가져오는 로직 구현이 필요합니다.")
        # CreonAPIClient의 get_market_data를 사용하여 KOSPI/KOSDAQ 지수 데이터를 가져올 수 있음.
        # 예: self.api_client.get_market_data(1, target_date.strftime('%Y%m%d')) # KOSPI
        df = self.get_market_df
        if not df is None:
            return df
        else:
            return pd.DataFrame()

    def load_daily_data_for_strategy(self, stock_codes: List[str], target_date: date, lookback_days: int) -> Dict[str, pd.DataFrame]:
        """
        일봉 전략이 사용할 일봉 데이터를 로드합니다.
        (backtest_manager의 get_daily_data_for_strategy와 유사)
        """
        daily_data = {}
        from_date = target_date - timedelta(days=lookback_days)
        for code in stock_codes:
            # fetch_daily_ohlcv가 DB 조회 후 없으면 API 조회 및 저장까지 처리
            df = self.fetch_daily_ohlcv(code, from_date, target_date)
            if not df.empty:
                daily_data[code] = df
            else:
                logger.warning(f"종목 {code}에 대한 일봉 데이터가 부족하여 전략 분석에서 제외합니다.")
        return daily_data

    def get_universe_stocks(self, start_date: date, end_date: date) -> List[str]:
        """
        DB에서 일별 테마 종목을 조회하여 유니버스에 포함될 종목 리스트를 반환합니다.
        (db_manager.fetch_daily_theme_stock을 래핑)
        """
        # TODO: num_themes와 stocks_per_theme 인자를 활용하여 조회 로직을 db_manager에 추가하거나,
        # db_manager에서 가져온 결과에 대해 필터링 로직 추가.
        # 현재 db_manager.fetch_daily_theme_stock은 상위 3개만 가져오므로 이 부분은 일치함.
        stocks_info = self.db_manager.fetch_daily_theme_stock(start_date, end_date)
        stock_codes = [code for code, name in stocks_info]
        logger.info(f"{start_date} {end_date} 날짜의 유니버스 종목 {len(stock_codes)}개 로드 완료.")
        return stock_codes


    def get_current_prices(self, stock_codes: List[str]) -> Dict[str, float]:
        """
        [수정] CreonAPIClient의 일괄 조회 메서드를 호출하여 현재가를 가져옵니다.
        """
        if not stock_codes:
            return {}
        # 새로 추가한 일괄 조회 메서드를 직접 호출
        return self.api_client.get_current_prices_bulk(stock_codes)
    

    def get_market_data_for_hmm(self, current_date: date, days: int = 365) -> pd.DataFrame:
        """
        [신규] HMM 모델 학습에 필요한 시장 데이터를 조회합니다.
        주로 시장 지수의 일일 수익률과 같은 거시 지표를 사용합니다.
        
        Args:
            current_date (date): 데이터 조회의 기준이 되는 날짜.
            days (int): 기준일로부터 과거 몇 일간의 데이터를 가져올지 결정.
            
        Returns:
            pd.DataFrame: HMM 학습에 사용될 관찰 변수들이 담긴 데이터프레임.
                          (예: 'daily_return' 컬럼 포함)
        """
        logger.info(f"HMM 학습용 시장 데이터 조회를 시작합니다. (기준일: {current_date}, 기간: {days}일)")
        
        market_index_code = COMMON_PARAMS.get('market_index_code', 'U001') # 설정에서 시장 코드 가져오기
        end_date = current_date
        start_date = end_date - timedelta(days=days)

        # DB에서 시장 지수의 일봉 데이터를 가져옵니다.
        # fetch_daily_price는 인덱스가 date인 DataFrame을 반환한다고 가정합니다.
        market_df = self.db_manager.fetch_daily_price(
            stock_code=market_index_code,
            start_date=start_date,
            end_date=end_date
        )

        if market_df.empty:
            logger.error(f"HMM 학습용 시장 데이터({market_index_code})를 조회할 수 없습니다.")
            return pd.DataFrame()

        # HMM의 관찰 변수인 '일일 수익률'을 계산합니다.
        market_df['daily_return'] = market_df['close'].pct_change()
        
        # (선택) 변동성 지수(VKOSPI) 등 다른 관찰 변수가 있다면 여기에 추가합니다.
        # 예: market_df['volatility'] = ...
        
        # hmmlearn 라이브러리는 NaN 값이 없는 numpy 배열을 기대하므로,
        # 계산 과정에서 발생한 NaN 값을 제거하고 필요한 컬럼만 선택합니다.
        hmm_input_df = market_df[['daily_return']].dropna()
        
        logger.info(f"HMM 학습용 데이터 준비 완료. {len(hmm_input_df)}개의 데이터 포인트 생성.")
        
        return hmm_input_df

    def save_trading_log(self, log_data: Dict[str, Any]) -> bool:
        """거래 로그를 DB에 저장하도록 요청합니다."""
        return self.db_manager.save_trading_log(log_data)

    def save_daily_portfolio(self, portfolio_data: Dict[str, Any]) -> bool:
        """일별 포트폴리오 정보를 DB에 저장하도록 요청합니다."""
        return self.db_manager.save_daily_portfolio(portfolio_data)  
              
    def save_current_position(self, position_data: Dict[str, Any]) -> bool:
        """일별 포트폴리오 정보를 DB에 저장하도록 요청합니다."""
        return self.db_manager.save_current_position(position_data)
    
    
    def fetch_average_trading_values(self, universe_codes: List[str], start_date: date, end_date: date) -> Dict[str, float]:
        """[래퍼] DBManager의 fetch_average_trading_values 메서드를 호출합니다."""
        return self.db_manager.fetch_average_trading_values(universe_codes, start_date, end_date)

    def fetch_latest_factors_for_universe(self, universe_codes: List[str], current_date: date) -> pd.DataFrame:
        """[래퍼] DBManager의 fetch_latest_factors_for_universe 메서드를 호출합니다."""
        return self.db_manager.fetch_latest_factors_for_universe(universe_codes, current_date)

    def fetch_stock_info(self, stock_codes: list = None) -> pd.DataFrame:
        """[래퍼] DBManager의 fetch_stock_info 메서드를 호출합니다."""
        return self.db_manager.fetch_stock_info(stock_codes)
    
    # def cache_minute_price(self, stock_code: str, current_dt: datetime) -> Optional[Dict[str, Any]]:
    #     """
    #     CreonAPIClient를 통해 특정 종목의 현재가와 최신 분봉 데이터를 가져와
    #     realtime_ohlcv_cache를 업데이트하고 DB(minute_price)에 저장합니다.
    #     (백테스트의 get_ohlcv_data_for_strategy를 자동매매용으로 변환)
    #     """
    #     # 1. 현재가 조회 (실시간 시세 또는 TR 요청)
    #     current_data = self.api_client.get_current_price(stock_code) # TR 요청 또는 실시간 구독 데이터
    #     if not current_data:
    #         logger.warning(f"종목 {stock_code}의 현재가 데이터를 가져올 수 없습니다.")
    #         return None

    #     # 2. 최신 분봉 데이터 업데이트 (get_ohlcv_data_for_strategy의 분봉 업데이트 로직 대체)
    #     # 보통 장중에는 1분마다 TR 요청 (복잡하므로, 여기서는 단순화)
    #     # 실제로는 CreonAPIClient에서 실시간 분봉 데이터를 구독하거나, 1분 단위 TR 요청으로 데이터를 쌓아야 함.
    #     # DB의 minute_price 테이블을 최신 데이터로 업데이트하는 것이 목표.
    #     minute_bar = self.api_client.get_latest_minute_bar(stock_code) # TR 요청 (GetChartData)
    #     if minute_bar:
    #         # 직접 SQL로 UPSERT (중복 방지)
    #         try:
    #             self.db_manager.execute_sql(f"""
    #                 INSERT INTO minute_price (stock_code, stock_name, datetime, open, high, low, close, volume)
    #                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    #                 ON DUPLICATE KEY UPDATE
    #                     open = VALUES(open), high = VALUES(high), low = VALUES(low), close = VALUES(close), volume = VALUES(volume),
    #                     updated_at = CURRENT_TIMESTAMP
    #             """, (
    #                 stock_code, self.get_stock_name(stock_code), minute_bar['time'],
    #                 minute_bar['open'], minute_bar['high'], minute_bar['low'], minute_bar['close'], minute_bar['volume']
    #             ))
    #             logger.debug(f"종목 {stock_code}의 최신 분봉 데이터 ({minute_bar['time']}) DB 업데이트 완료.")
    #         except Exception as e:
    #             logger.error(f"종목 {stock_code}의 분봉 데이터 DB 업데이트 오류: {e}")

    #         # 캐시 업데이트 (실제 사용 시에는 필요한 데이터만 캐시)
    #         # 간단하게 마지막 분봉만 반환하도록 구현
    #         return minute_bar
    #     else:
    #         logger.warning(f"종목 {stock_code}의 최신 분봉 데이터를 가져올 수 없습니다.")
    #         return None
