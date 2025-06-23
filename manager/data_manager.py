# manager/data_manager.py

import logging
import pandas as pd
from datetime import datetime, date, timedelta
import time
import sys
import os

# --- 로거 설정 (스크립트 최상단에서 설정하여 항상 보이도록 함) ---
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 테스트 시 DEBUG로 설정하여 모든 로그 출력 - 제거
# sys.path에 프로젝트 루트 추가 (db_manager 및 creon_api 임포트를 위함)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
sys.path.insert(0, project_root)
logger.debug(f"Project root added to sys.path: {project_root}")

from manager.db_manager import DBManager
# 가정: CreonAPIClient는 실제 증권사 API와 통신하는 클래스입니다.
from api.creon_api import CreonAPIClient # CreonAPI 클래스 이름 일치


class DataManager:
    def __init__(self):
        self.db_manager = DBManager()
        self.api_client = CreonAPIClient()
        logger.info("DataManager 초기화 완료: DBManager 및 CreonAPIClient 연결")

    def close(self):
        """DBManager의 연결을 종료합니다."""
        if self.db_manager:
            self.db_manager.close()
            logger.info("DataManager를 통해 DB 연결을 종료했습니다.")

    def fetch_market_calendar_initial_data(self, years: int = 3):
        """
        Creon API를 통해 과거 특정 기간의 거래일 데이터를 가져와 market_calendar 테이블에 초기 데이터를 삽입합니다.
        이미 존재하는 날짜는 중복 삽입하지 않습니다.
        
        :param years: 현재부터 몇 년 전까지의 데이터를 가져올지 지정합니다.
        """
        logger.info(f"Market Calendar 초기 데이터 로드 시작: 오늘로부터 과거 {years}년치")
        
        today = date.today()
        from_date = today - timedelta(days=years * 365) # 간단하게 365일 * 년수로 계산 (윤년 고려 안함)
        
        # 1. DB에 이미 존재하는 거래일 조회
        existing_trading_days_in_db = self.db_manager.get_all_trading_days(from_date, today)
        logger.info(f"DB에 이미 존재하는 거래일 수: {len(existing_trading_days_in_db)}개")
        
        # 2. Creon API를 통해 전체 거래일 조회 (대부분 삼성전자 A005930 사용)
        all_api_trading_days = self.api_client.get_all_trading_days_from_api(from_date, today, stock_code='A005930')
        if not all_api_trading_days:
            logger.error("Creon API로부터 거래일 데이터를 가져오지 못했습니다. Market Calendar 초기화 실패.")
            return
        logger.info(f"Creon API로부터 받은은 거래일 수: {len(all_api_trading_days)}개")

        # # 3. DB에 없는 거래일만 필터링, set 연산 -> list -> sort
        new_trading_days = sorted(list(set(all_api_trading_days) - set(existing_trading_days_in_db)) )

        if not new_trading_days:
            logger.info("새로 추가할 거래일 데이터가 없습니다. Market Calendar는 최신 상태입니다.")
            return
        logger.info(f"새로 추가할 거래일 수: {len(new_trading_days)}개")

        # # 4. DataFrame 생성 및 DB에 삽입
        data_to_insert = []
        for trading_date in new_trading_days:
            data_to_insert.append({
                'date': trading_date,
                'is_holiday': True,
                'description': '영업일'
            })

        if data_to_insert:
            df_to_insert = pd.DataFrame(data_to_insert)
            # DataFrame의 'trading_date' 컬럼을 datetime.date 타입으로 정확히 설정
            df_to_insert['date'] = df_to_insert['date'].apply(lambda x: x.isoformat()) # MySQL DATE 타입에 맞추기 위해 문자열로 변환
            
            # DBManager의 insert_df_to_db는 date 인덱스가 아닌 컬럼을 기대하므로, is_index=False
            self.db_manager.insert_df_to_db('market_calendar', df_to_insert, option="append", is_index=False)
            logger.info(f"Market Calendar 테이블에 {len(df_to_insert)}개의 새로운 거래일 데이터 삽입 완료.")
        else:
            logger.info("삽입할 새로운 거래일 데이터가 없습니다.")


    def cache_daily_ohlcv(self, stock_code: str, from_date: date, to_date: date) -> pd.DataFrame:
        """
        DB와 증권사 API를 사용하여 특정 종목의 일봉 데이터를 캐싱하고 반환합니다.
        :param stock_code: 종목 코드
        :param from_date: 조회 시작일 (datetime.date 객체)
        :param to_date: 조회 종료일 (datetime.date 객체)
        :return: Pandas DataFrame (DB와 API 데이터를 합친 최종 데이터)
        """
        logger.debug(f"일봉 데이터 캐싱 시작: {stock_code} ({from_date.strftime('%Y%m%d')} ~ {to_date.strftime('%Y%m%d')})")

        # 1. DB에서 데이터 조회 시도
        db_df = self.db_manager.fetch_daily_price(stock_code, from_date, to_date)
        
        # db_df.index는 datetime.date 객체로 구성된 Index임을 가정합니다.
        db_existing_dates = set(db_df.index.to_list()) if not db_df.empty else set()
        # set 으로 min, max 구할 수 있나?
        if db_existing_dates:
            logger.debug(f"DB에서 {stock_code}의 일봉 데이터 {len(db_df)}개 로드됨. ({min(db_existing_dates).strftime('%Y-%m-%d')} ~ {max(db_existing_dates).strftime('%Y-%m-%d')})")
        else:
            logger.debug(f"DB에서 {stock_code}의 기존 일봉 데이터가 없습니다.")
        
        # 2. market_calendar에서 필요한 모든 실제 거래일 조회
        all_trading_dates = set(self.db_manager.get_all_trading_days(from_date, to_date))

        # 3. DB에 누락된 날짜 (market_calendar에는 있지만 DB daily_price에는 없는 날짜) 계산
        missing_dates = sorted(list(all_trading_dates - db_existing_dates))
        missing_dates = [d for d in missing_dates if pd.Timestamp(from_date) <= pd.Timestamp(d) <= pd.Timestamp(to_date)]
        
        # 4. 누락된 데이터 API 호출 및 DB 저장 (연속된 구간 처리)
        if missing_dates:
            logger.debug(f"DB에 누락된 일봉 데이터 발견: {len(missing_dates)}개 날짜. API 호출 시작.")
            api_fetched_dfs = []
            current_start = missing_dates[0]
            current_end = missing_dates[-1]
            api_fetched_dfs.append(self._fetch_and_store_daily_range(stock_code, current_start, current_end, 'daily'))
            
            if api_fetched_dfs:
                # 모든 데이터가 DB에 저장되었으므로, 최종적으로 DB에서 조회하여 반환
                final_df = self.db_manager.fetch_daily_price(stock_code, from_date, to_date)
                logger.debug(f"{stock_code}의 최종 일봉 데이터 {len(final_df)}개 준비 완료.")
                return final_df
            else:
                logger.warning(f"API로부터 {stock_code}에 대한 추가 일봉 데이터를 가져오지 못했습니다. DB 데이터만 반환합니다.")
                return db_df
        else:
            logger.debug(f"{stock_code}의 모든 일봉 데이터가 DB에 존재합니다. API 호출 없이 DB 데이터만 반환합니다.")
            return db_df

    def cache_minute_ohlcv(self, stock_code: str, from_date: date, to_date: date, interval: int = 1) -> pd.DataFrame:
        """
        분봉 데이터를 캐싱하고 반환합니다. DB에 있는 데이터는 재사용하고, 없는 데이터만 API에서 가져옵니다.
        """
        logger.debug(f"분봉 데이터 캐싱 시작: {stock_code} ({from_date} ~ {to_date}), Interval: {interval}분")
        
        # 1. DB에서 데이터 조회 (한 번만 조회)
        db_df = self.db_manager.fetch_minute_price(stock_code, from_date, to_date)
        
        # 2. DB에 있는 날짜들을 pd.Timestamp로 변환하여 set으로 저장
        db_existing_dates = set()
        if not db_df.empty:
            # 인덱스의 날짜 부분만 추출하여 set으로 변환
            db_existing_dates = {pd.Timestamp(d).normalize() for d in db_df.index.date}
            logger.debug(f"DB에서 {stock_code}의 분봉 데이터가 존재하는 날짜: {len(db_existing_dates)}개")
            if db_existing_dates:
                logger.debug(f"데이터 범위: {min(db_existing_dates).strftime('%Y-%m-%d')} ~ {max(db_existing_dates).strftime('%Y-%m-%d')}")
        
        # 3. 거래일 목록 조회 (이미 pd.Timestamp 객체)
        all_trading_dates = set(self.db_manager.get_all_trading_days(from_date, to_date))
        
        # 4. 거래일 중 DB에 없는 날짜만 필터링
        dates_to_fetch_from_api = sorted(list(all_trading_dates - db_existing_dates))

        if not dates_to_fetch_from_api:
            logger.debug(f"모든 요청 날짜({from_date} ~ {to_date})의 분봉 데이터가 DB에 존재합니다. API 호출 안함.")
            return db_df
        
        logger.debug(f"DB에 누락된 분봉 데이터 발견: {len(dates_to_fetch_from_api)}개 날짜")
        logger.debug(f"API에서 분봉 데이터를 가져올 날짜들: {[d.strftime('%Y-%m-%d') for d in dates_to_fetch_from_api]}")

        # 5. 누락된 데이터를 API에서 가져오기
        api_fetched_dfs = []
        if dates_to_fetch_from_api:
            # 연속된 날짜들을 하나의 구간으로 처리
            current_start = dates_to_fetch_from_api[0]
            current_end = dates_to_fetch_from_api[-1]
            
            try:
                api_df = self._fetch_and_store_daily_range(stock_code, current_start, current_end, 'minute')
                if not api_df.empty:
                    api_fetched_dfs.append(api_df)
            except Exception as e:
                logger.error(f"API 데이터 가져오기 실패: {stock_code} - {str(e)}")

        # 6. 최종 데이터 반환
        if api_fetched_dfs:
            # API에서 가져온 데이터와 DB 데이터를 합침
            final_df = pd.concat([db_df] + api_fetched_dfs).sort_index()
            # 중복 제거 (혹시 모를 중복 데이터 처리)
            final_df = final_df[~final_df.index.duplicated(keep='first')]
            logger.debug(f"{stock_code}의 최종 분봉 데이터 {len(final_df)}개 준비 완료.")
            return final_df
        else:
            logger.debug(f"API에서 추가 데이터를 가져오지 못했습니다. DB 데이터만 반환합니다.")
            return db_df

    def _fetch_and_store_daily_range(self, stock_code: str, start_date: date, end_date: date, data_type: str) -> pd.DataFrame:
        """
        API에서 데이터를 가져와 DB에 저장합니다. API 호출 실패 시 빈 DataFrame을 반환합니다.
        """
        logger.debug(f"API로부터 {stock_code} {data_type} 데이터 요청: {start_date} ~ {end_date}")
        
        api_df_part = pd.DataFrame()
        try:
            # API 호출
            if data_type == 'daily':
                api_df_part = self.api_client.get_daily_ohlcv(stock_code, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
            elif data_type == 'minute':
                api_df_part = self.api_client.get_minute_ohlcv(stock_code, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
                # 분봉 데이터 API 호출 시 지연 적용
                if not api_df_part.empty:
                    time.sleep(0.1)  # 0.1초로 변경
            else:
                logger.error(f"알 수 없는 data_type: {data_type}")
                return pd.DataFrame()
            
            if api_df_part.empty:
                return pd.DataFrame()
            
            # DB 저장을 위한 데이터 변환
            data_to_save_list = []
            for index_val, row in api_df_part.iterrows():
                record = {
                    'stock_code': stock_code,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'change_rate': row.get('change_rate', 0.0),
                    'trading_value': row.get('trading_value', 0)
                }
                
                if data_type == 'daily':
                    record['date'] = index_val.date()
                elif data_type == 'minute':
                    record['datetime'] = index_val
                
                data_to_save_list.append(record)
            
            # DB 저장 시도
            try:
                if data_type == 'daily':
                    self.db_manager.save_daily_price(data_to_save_list)
                elif data_type == 'minute':
                    self.db_manager.save_minute_price(data_to_save_list)
                logger.debug(f"API로부터 {stock_code}의 {data_type} 데이터 {len(api_df_part)}개 DB 저장 완료")
            except Exception as e:
                logger.error(f"DB 저장 실패: {stock_code} - {str(e)}")
                # DB 저장 실패해도 API 데이터는 반환
            
            return api_df_part
            
        except Exception as e:
            logger.error(f"API 호출 실패: {stock_code} - {str(e)}")
            return pd.DataFrame()

    def get_stock_info_map(self) -> dict:
        """
        DB의 stock_info 테이블에서 모든 종목의 코드와 이름을 가져와
        {'종목코드': '종목명'} 형태의 딕셔너리로 반환합니다.
        """
        logger.debug("종목 정보 맵(딕셔너리) 로딩 시작")
        try:
            # fetch_stock_info()를 인자 없이 호출하여 모든 종목 정보를 가져옵니다.
            stock_info_df = self.db_manager.fetch_stock_info() 
            if not stock_info_df.empty:
                # 'stock_code'를 인덱스로, 'stock_name'을 값으로 하는 딕셔너리 생성
                stock_map = pd.Series(stock_info_df.stock_name.values, index=stock_info_df.stock_code).to_dict()
                logger.debug(f"{len(stock_map)}개의 종목 정보 로딩 완료")
                return stock_map
            else:
                logger.warning("DB에서 종목 정보를 가져오지 못했습니다. stock_info 테이블이 비어있을 수 있습니다.")
                return {}
        except Exception as e:
            logger.error(f"종목 정보 로딩 중 오류 발생: {e}")
            return {}
        
    def update_all_stock_info(self):
        """
        Creon API에서 모든 종목 정보를 가져와 DB의 stock_info 테이블에 저장/업데이트합니다.
        매매의 기본이 되는 모든 종목의 기본 정보를 셋업합니다.
        """
        logger.info("모든 종목 기본 정보 업데이트를 시작합니다.")
        if not self.api_client.connected:
            logger.error("Creon API가 연결되어 있지 않아 종목 정보를 가져올 수 없습니다.")
            return False

        try:
            filtered_codes = self.api_client.get_filtered_stock_list()
            stock_info_list = []
            for code in filtered_codes:
                name = self.api_client.get_stock_name(code)
                market_type = 'KOSPI' if code in self.api_client.cp_code_mgr.GetStockListByMarket(1) else 'KOSDAQ'

                stock_info_list.append({
                    'stock_code': code,
                    'stock_name': name,
                    'market_type': market_type,
                    'sector': None, # CpCodeMgr.GetStockSector() 등으로 가져올 수 있으나, 현재는 제외
                    'per': None, 'pbr': None, 'eps': None, 'roe': None, 'debt_ratio': None, # 초기값 None
                    'sales': None, 'operating_profit': None, 'net_profit': None,
                    'recent_financial_date': None
                })

            if stock_info_list:
                # 직접 DBManager를 통해 저장
                if self.db_manager.save_stock_info(stock_info_list):
                    logger.info(f"{len(stock_info_list)}개의 종목 기본 정보를 성공적으로 DB에 업데이트했습니다.")
                    return True
                else:
                    logger.error("종목 기본 정보 DB 저장에 실패했습니다.")
                    return False
            else:
                logger.warning("가져올 종목 정보가 없습니다. Creon HTS 연결 상태 및 종목 필터링 조건을 확인하세요.")
                return False
        except Exception as e:
            logger.error(f"모든 종목 기본 정보 업데이트 중 오류 발생: {e}", exc_info=True)
            return False

    def update_market_calendar(self, start_date: date = None, end_date: date = None):
        """
        주식시장 캘린더 정보를 업데이트합니다.
        :param start_date: 시작 날짜 (기본값: 1년 전)
        :param end_date: 종료 날짜 (기본값: 1년 후)
        """
        logger.info("주식시장 캘린더 업데이트를 시작합니다.")
        
        if not start_date:
            start_date = date.today() - timedelta(days=365)
        if not end_date:
            end_date = date.today() + timedelta(days=365)
            
        try:
            # Creon API에서 거래일 정보 가져오기
            trading_days = self.api_client.get_all_trading_days_from_api(start_date, end_date)
            
            calendar_data = []
            current_date = start_date
            while current_date <= end_date:
                is_holiday = current_date not in trading_days
                calendar_data.append({
                    'date': current_date,
                    'is_holiday': is_holiday,
                    'description': '공휴일' if is_holiday else '거래일'
                })
                current_date += timedelta(days=1)
            
            if calendar_data:
                # 리스트를 DataFrame으로 변환하고 중복 키 오류 방지를 위해 "replace" 옵션 사용
                calendar_df = pd.DataFrame(calendar_data)
                if self.db_manager.save_market_calendar(calendar_df, option="replace"):
                    logger.info(f"주식시장 캘린더 {len(calendar_data)}개 데이터를 성공적으로 업데이트했습니다.")
                    return True
                else:
                    logger.error("주식시장 캘린더 DB 저장에 실패했습니다.")
                    return False
            else:
                logger.warning("업데이트할 캘린더 데이터가 없습니다.")
                return True
        except Exception as e:
            logger.error(f"주식시장 캘린더 업데이트 중 오류 발생: {e}", exc_info=True)
            return False

    def update_financial_data_for_stock_info(self, stock_code):
        """
        특정 종목의 최신 재무 데이터를 CreonAPIClient (MarketEye)에서 가져와
        DB의 stock_info 테이블에 업데이트합니다.
        """
        logger.info(f"{stock_code} stock_info 테이블의 최신 재무 데이터 업데이트 중...")
        
        try:
            # Creon API Client에서 최신 재무 데이터 가져오기
            finance_df = self.api_client.get_latest_financial_data(stock_code)

            if finance_df.empty:
                logger.info(f"{stock_code} Creon API에서 조회된 재무 데이터가 없습니다.")
                return

            # MarketEye에서 가져온 DataFrame을 stock_info 테이블의 컬럼에 맞게 조정
            finance_df['operating_profit'] = finance_df['operating_profit'] / 1_000_000
            finance_df['net_profit'] = finance_df['net_profit'] / 1_000_000

            # 직접 DBManager를 통해 저장
            self.db_manager.save_stock_info(finance_df.to_dict(orient='records'))

            logger.info(f"{stock_code} stock_info 테이블의 최신 재무 데이터가 성공적으로 업데이트되었습니다.")

        except Exception as e:
            logger.error(f"stock_info 재무 데이터 업데이트 중 오류 발생: {e}", exc_info=True)
        logger.info(f"{stock_code} 재무 데이터 업데이트 완료.")
