# manager/data_manager.py

import logging
import pandas as pd
from datetime import datetime, date, timedelta
import time
import sys
import os


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
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


    def load_market_calendar_initial_data(self, years: int = 3):
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
        logger.info(f"일봉 데이터 캐싱 시작: {stock_code} ({from_date.strftime('%Y%m%d')} ~ {to_date.strftime('%Y%m%d')})")

        # 1. DB에서 데이터 조회 시도
        db_df = self.db_manager.fetch_daily_price(stock_code, from_date, to_date)
        
        # db_df.index는 datetime.date 객체로 구성된 Index임을 가정합니다.
        db_existing_dates = set(db_df.index.to_list()) if not db_df.empty else set()
        # set 으로 min, max 구할 수 있나?
        if db_existing_dates:
            logger.info(f"DB에서 {stock_code}의 일봉 데이터 {len(db_df)}개 로드됨. ({min(db_existing_dates).strftime('%Y-%m-%d')} ~ {max(db_existing_dates).strftime('%Y-%m-%d')})")
        else:
            logger.info(f"DB에서 {stock_code}의 기존 일봉 데이터가 없습니다.")
        
        # 2. market_calendar에서 필요한 모든 실제 거래일 조회
        all_trading_dates = set(self.db_manager.get_all_trading_days(from_date, to_date))
        # -------------------  print('all_trading_dates', self.db_manager.get_all_trading_days(from_date, to_date))

        # 3. DB에 누락된 날짜 (market_calendar에는 있지만 DB daily_price에는 없는 날짜) 계산
        missing_dates = sorted(list(all_trading_dates - db_existing_dates))
        missing_dates = [d for d in missing_dates if from_date <= d <= to_date]
        
        # 4. 누락된 데이터 API 호출 및 DB 저장 (연속된 구간 처리)
        if missing_dates:
            logger.info(f"DB에 누락된 일봉 데이터 발견: {len(missing_dates)}개 날짜. API 호출 시작.")
            api_fetched_dfs = []
            current_start = missing_dates[0]
            current_end = missing_dates[-1]
            api_fetched_dfs.append(self._fetch_and_store_daily_range(stock_code, current_start, current_end, 'daily'))
            
            if api_fetched_dfs:
                # 모든 데이터가 DB에 저장되었으므로, 최종적으로 DB에서 조회하여 반환
                final_df = self.db_manager.fetch_daily_price(stock_code, from_date, to_date)
                logger.info(f"{stock_code}의 최종 일봉 데이터 {len(final_df)}개 준비 완료.")
                return final_df
            else:
                logger.warning(f"API로부터 {stock_code}에 대한 추가 일봉 데이터를 가져오지 못했습니다. DB 데이터만 반환합니다.")
                return db_df
        else:
            logger.info(f"{stock_code}의 모든 일봉 데이터가 DB에 존재합니다. API 호출 없이 DB 데이터만 반환합니다.")
            return db_df

    def cache_minute_ohlcv(self, stock_code: str, from_date: date, to_date: date, interval: int = 1) -> pd.DataFrame:
        logger.info(f"분봉 데이터 캐싱 시작: {stock_code} ({from_date} ~ {to_date}), Interval: {interval}분")
        ################ 수정정
        db_df = self.db_manager.fetch_minute_price(stock_code, from_date, to_date)
        db_existing_dates = set(db_df.index.date) if not db_df.empty else set()
        # 3. 요청 범위에 해당하는 '모든 날짜'들을 리스트로 만듭니다. (datetime.date 객체들의 리스트)
        # pd.date_range를 사용하면 더 깔끔합니다.
        all_target_dates_list = pd.date_range(start=from_date, end=to_date, freq='D').date.tolist()
        
        # 4. API에서 가져와야 할 날짜들 결정 (set 연산 사용)
        # all_target_dates_list를 set으로 변환하여 차집합 연산을 수행합니다.
        dates_to_fetch_from_api = [
            d for d in all_target_dates_list 
            if d not in db_existing_dates # d in db_existing_dates로 제대로 비교되는지 확인
        ]

        if not dates_to_fetch_from_api: # API 호출이 필요한 날짜가 없는 경우
            logger.info(f"모든 요청 날짜({from_date} ~ {to_date})의 분봉 데이터가 DB에 존재합니다. API 호출 건너뜐.")
            return db_df # DB에서 가져온 데이터만 반환
        
        logger.info(f"DB에 누락된 분봉 데이터 발견: {len(dates_to_fetch_from_api)}개 날짜. API 호출 시작.")
        logger.info(f"API에서 분봉 데이터를 가져올 날짜들: {dates_to_fetch_from_api}")        
        if db_existing_dates:
            logger.info(f"DB에서 {stock_code}의 분봉 데이터가 존재하는 날짜: {len(db_existing_dates)}개. ({min(db_existing_dates)} ~ {max(db_existing_dates)})")
        else:
            logger.info(f"DB에서 {stock_code}의 기존 분봉 데이터가 없습니다.")

        all_trading_dates = set(self.db_manager.get_all_trading_days(from_date, to_date))
        
        # if not all_trading_dates:
        #     logger.warning(f"거래일 캘린더를 로드할 수 없습니다. API로만 분봉 데이터를 가져와 캐싱합니다.")
        #     api_df = self.api_client.get_minute_ohlcv(stock_code, from_date.strftime('%Y%m%d'), to_date.strftime('%Y%m%d'), interval)
        #     if not api_df.empty:
        #         df_to_save = api_df.copy()
        #         df_to_save['datetime'] = df_to_save.index.date
        #         df_to_save['stock_code'] = stock_code
        #         #df_to_save['interval'] = interval
        #         df_to_save.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
        #         if 'trading_value' not in df_to_save.columns: df_to_save['trading_value'] = 0
        #         if 'change_rate' not in df_to_save.columns: df_to_save['change_rate'] = 0.0
        #         self.db_manager.insert_df_to_db('minute_price', df_to_save, option="append", is_index=False)
        #         return api_df
        #     return pd.DataFrame()

        missing_dates = sorted(list(all_trading_dates - db_existing_dates))

        if missing_dates:
            logger.info(f"DB에 누락된 분봉 데이터 발견: {len(missing_dates)}개 날짜. API 호출 시작.")
            
            api_fetched_dfs = []
            current_start = missing_dates[0]
            #current_end = missing_dates[0]
            current_end = current_start + timedelta(days=13)
            # for i in range(1, len(missing_dates)):
            #     if missing_dates[i] == current_end + timedelta(days=1):
            #         current_end = missing_dates[i]
            #     else:
            #         api_fetched_dfs.append(self._fetch_and_store_daily_range(stock_code, current_start, current_end, 'minute'))
            #         current_start = missing_dates[i]
            #         current_end = missing_dates[i]
            
            api_fetched_dfs.append(self._fetch_and_store_daily_range(stock_code, current_start, current_end, 'minute'))

            api_fetched_dfs = [df for df in api_fetched_dfs if not df.empty]

            if api_fetched_dfs:
                final_df = self.db_manager.fetch_minute_price(stock_code, from_date, to_date)
                logger.info(f"{stock_code}의 최종 분봉 데이터 {len(final_df)}개 준비 완료.")
                return final_df
            else:
                logger.warning(f"API로부터 {stock_code}에 대한 추가 분봉 데이터를 가져오지 못했습니다. DB 데이터만 반환합니다.")
                return db_df
        else:
            logger.info(f"{stock_code}의 모든 분봉 데이터가 DB에 존재합니다. API 호출 없이 DB 데이터만 반환합니다.")
            return db_df

        # missing_dates = sorted(list(all_trading_dates - db_existing_dates))

        # # 4. 누락된 데이터 API 호출 및 DB 저장 (연속된 구간 처리)
        # if missing_dates:
        #     logger.info(f"DB에 누락된 일봉 데이터 발견: {len(missing_dates)}개 날짜. API 호출 시작.")
        #     api_fetched_dfs = []
        #     current_start = missing_dates[0]
        #     current_end = missing_dates[-1]
        #     api_fetched_dfs.append(self._fetch_and_store_daily_range(stock_code, current_start, current_end, 'daily'))
            
        #     if api_fetched_dfs:
        #         # 모든 데이터가 DB에 저장되었으므로, 최종적으로 DB에서 조회하여 반환
        #         final_df = self.db_manager.fetch_minute_price(stock_code, from_date, to_date)
        #         logger.info(f"{stock_code}의 최종 분봉 데이터 {len(final_df)}개 준비 완료.")
        #         return final_df
        #     else:
        #         logger.warning(f"API로부터 {stock_code}에 대한 추가 분봉 데이터를 가져오지 못했습니다. DB 데이터만 반환합니다.")
        #         return db_df
        # else:
        #     logger.info(f"{stock_code}의 모든 분봉 데이터가 DB에 존재합니다. API 호출 없이 DB 데이터만 반환합니다.")
        #     return db_df
        

    # _fetch_and_store_daily_range를 일반화하여 일봉/분봉 모두 처리하도록 수정
    def _fetch_and_store_daily_range(self, stock_code: str, start_date: date, end_date: date, data_type: str) -> pd.DataFrame:
        """
        주어진 범위의 일봉 또는 분봉 데이터를 API로부터 가져와 DB에 저장하고 반환합니다.
        저장 시 UPSERT (ON DUPLICATE KEY UPDATE) 처리를 위해 db_manager의 save_daily_price/save_minute_price를 사용합니다.
        :param stock_code: 종목 코드
        :param start_date: 조회 시작일 (datetime.date 객체)
        :param end_date: 조회 종료일 (datetime.date 객체)
        :param data_type: 'daily' 또는 'minute'
        :return: Pandas DataFrame (API로부터 받은 원본 데이터)
        """
        logger.debug(f"API로부터 {stock_code} {data_type} 데이터 요청: {start_date} ~ {end_date}")
        
        api_df_part = pd.DataFrame()
        if data_type == 'daily':
            api_df_part = self.api_client.get_daily_ohlcv(stock_code, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
        elif data_type == 'minute':
            # Creon API의 get_minute_ohlcv가 시작일~종료일 지원을 가정합니다.
            # 만약 Creon API가 일별 조회만 지원한다면, 이 함수 내에서 일별로 루프를 돌려야 합니다.
            # 여기서는 API가 기간 조회를 지원한다고 가정하고 직접 호출합니다.
            api_df_part = self.api_client.get_minute_ohlcv(stock_code, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
        else:
            logger.error(f"알 수 없는 data_type: {data_type}")
            return pd.DataFrame()
        
        time.sleep(0.3) # API 요청 간격 유지

        if not api_df_part.empty:
            # Pandas DataFrame을 db_manager.save_xx_price 함수가 기대하는 딕셔너리 리스트로 변환
            data_to_save_list = []
            
            # DataFrame 인덱스 처리 (날짜 또는 날짜+시간)
            # _get_price_data에서 이미 인덱스 타입이 적절하게 처리되었으므로, 여기서는 이를 활용
            for index_val, row in api_df_part.iterrows():
                record = {
                    'stock_code': stock_code,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    # change_rate와 trading_value는 API에서 제공하지 않을 경우 0으로 기본값 설정
                    'change_rate': row.get('change_rate', 0.0), # .get() 사용 (DataFrame row도 딕셔너리처럼 접근 가능)
                    'trading_value': row.get('trading_value', 0)
                }
                
                if data_type == 'daily':
                    # daily_price 테이블의 'date' 컬럼은 datetime.date 객체 또는 YYYY-MM-DD 문자열 기대
                    # DatetimeIndex의 요소는 Timestamp 객체이므로 .date()로 datetime.date 추출
                    record['date'] = index_val.date()
                elif data_type == 'minute':
                    # minute_price 테이블의 'datetime' 컬럼은 datetime.datetime 객체 또는 YYYY-MM-DD HH:MM:SS 문자열 기대
                    # DatetimeIndex의 요소는 Timestamp 객체이므로 그대로 사용 (datetime.datetime과 호환)
                    record['datetime'] = index_val
                
                data_to_save_list.append(record)
            
            # DBManager의 해당 저장 함수 호출
            if data_type == 'daily':
                if self.db_manager.save_daily_price(data_to_save_list):
                    logger.info(f"API로부터 {stock_code}의 일봉 데이터 {len(api_df_part)}개 가져와 DB에 UPSERT 완료. ({start_date} ~ {end_date})")
                else:
                    logger.error(f"DB에 {stock_code}의 일봉 데이터 저장/업데이트 실패. ({start_date} ~ {end_date})")
            elif data_type == 'minute':
                if self.db_manager.save_minute_price(data_to_save_list):
                    logger.info(f"API로부터 {stock_code}의 분봉 데이터 {len(api_df_part)}개 가져와 DB에 UPSERT 완료. ({start_date} ~ {end_date})")
                else:
                    logger.error(f"DB에 {stock_code}의 분봉 데이터 저장/업데이트 실패. ({start_date} ~ {end_date})")
            
            # 원본 API DataFrame을 그대로 반환 (인덱스 타입 변경 로직 제거)
            # _get_price_data에서 이미 인덱스 타입이 적절하게 처리되었으므로 추가 변환은 불필요
            return api_df_part
        else:
            logger.warning(f"API로부터 {stock_code}의 {data_type} 데이터를 가져오지 못했습니다. ({start_date} ~ {end_date})")
            return pd.DataFrame()


# __main__ 부분은 이전과 동일하게 유지됩니다.
if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers: # 루트 로거에 핸들러가 없다면 추가
        console_handler = logging.StreamHandler(sys.stdout)

    logger.info("DataManager 실제 객체로 초기화 시작...")
    try:
        data_manager = DataManager()
        logger.info("DataManager 인스턴스 실제 객체로 초기화 완료.")
    except Exception as e:
        logger.error(f"DataManager 초기화 중 오류 발생: {e}")
        logger.error("DBManager 또는 CreonAPIClient 초기화에 문제가 없는지 확인하세요.")
        sys.exit(1)

    # --- Market Calendar 초기 데이터 로드 테스트 ---
    logger.info(f"\n--- Market Calendar 초기 데이터 로드 테스트 시작: 오늘 이전 3년치 ---")
    try:
        data_manager.load_market_calendar_initial_data(years=1)
    except Exception as e:
        logger.error(f"Market Calendar 초기 데이터 로드 중 오류 발생: {e}")
        
    # time.sleep(1) # API 호출 제한을 위한 간격

    stock_code = 'A005930'
    
    today = date.today()
    start_date_daily = today - timedelta(days=5)
    start_date_minute = today - timedelta(days=5)

    # # --- 1. cache_daily_ohlcv 테스트 ---
    # logger.info(f"\n--- cache_daily_ohlcv 테스트 시작: 종목코드 {stock_code}, 기간 {start_date_daily} ~ {today} ---")
    # try:
    #     daily_df = data_manager.cache_daily_ohlcv(stock_code, start_date_daily, today)
    #     if not daily_df.empty:
    #         logger.info(f"일봉 데이터 캐싱 및 로드 성공. 총 {len(daily_df)}개 데이터.")
    #         logger.info(f"첫 5개 데이터:\n{daily_df.head()}")
    #         logger.info(f"마지막 5개 데이터:\n{daily_df.tail()}")
    #         logger.info(f"데이터 범위: {daily_df.index.min()} ~ {daily_df.index.max()}")
    #     else:
    #         logger.warning(f"일봉 데이터를 가져오지 못했습니다: {stock_code}")
    # except Exception as e:
    #     logger.error(f"cache_daily_ohlcv 테스트 중 오류 발생: {e}")
        
    # time.sleep(1)

    # # # # --- 2. cache_minute_ohlcv 테스트 ---
    # logger.info(f"\n--- cache_minute_ohlcv 테스트 시작: 종목코드 {stock_code}, 기간 {start_date_minute} ~ {today}, 1분봉 ---")
    # try:
    #     minute_df = data_manager.cache_minute_ohlcv(stock_code, start_date_minute, today, interval=1)
    #     print(minute_df)
    #     if not minute_df.empty:
    #         logger.info(f"분봉 데이터 캐싱 및 로드 성공. 총 {len(minute_df)}개 데이터.")
    #         logger.info(f"첫 5개 데이터:\n{minute_df.head()}")
    #         logger.info(f"마지막 5개 데이터:\n{minute_df.tail()}")
    #         logger.info(f"데이터 범위: {minute_df.index.min()} ~ {minute_df.index.max()}")
    #     else:
    #         logger.warning(f"분봉 데이터를 가져오지 못했습니다: {stock_code}")
    # except Exception as e:
    #     logger.error(f"cache_minute_ohlcv 테스트 중 오류 발생: {e}")

    # logger.info("\n모든 DataManager 실제 객체 테스트 완료.")