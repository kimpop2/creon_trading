# api_client/creon_api.py

import win32com.client
import ctypes
import time
import logging
import pandas as pd
#import datetime # datetime 모듈 전체를 임포트하여 datetime.timedelta 사용 가능
import re
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any

API_REQUEST_INTERVAL = 0.2

# 로거 설정 (기존 설정 유지)
logger = logging.getLogger(__name__)

class CreonAPIClient:
    def __init__(self):
        self.connected = False
        self.cp_code_mgr = None
        self.cp_cybos = None
        self.request_interval = API_REQUEST_INTERVAL
        self.stock_name_dic = {}
        self.stock_code_dic = {}
        self.account_number = None  # 계좌번호
        self.account_flag = None    # 주식상품 구분

        self._connect_creon_and_init_trade()  # 연결 및 거래 초기화 통합
        if self.connected:
            self.cp_code_mgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")
            logger.info("CpCodeMgr COM object initialized.")
            self._make_stock_dic()


    def _connect_creon_and_init_trade(self):
        """Creon Plus에 연결하고 COM 객체 및 거래 초기화를 수행합니다."""
        if not ctypes.windll.shell32.IsUserAnAdmin():
            logger.warning("관리자 권한으로 실행되지 않았습니다. 일부 Creon 기능이 제한될 수 있습니다.")

        self.cp_cybos = win32com.client.Dispatch("CpUtil.CpCybos")
        if self.cp_cybos.IsConnect:
            self.connected = True
            logger.info("Creon Plus가 이미 연결되어 있습니다.")
        else:
            logger.info("Creon Plus 연결 시도 중...")
            max_retries = 10
            for i in range(max_retries):
                if self.cp_cybos.IsConnect:
                    self.connected = True
                    logger.info("Creon Plus 연결 성공.")
                    break
                else:
                    logger.warning(f"Creon Plus 연결 대기 중... ({i+1}/{max_retries})")
                    time.sleep(2)
            if not self.connected:
                logger.error("Creon Plus 연결 실패. HTS가 실행 중이고 로그인되어 있는지 확인하세요.")
                raise ConnectionError("Creon Plus 연결 실패.")

        try:
            cpTradeUtil = win32com.client.Dispatch('CpTrade.CpTdUtil')
            if cpTradeUtil.TradeInit(0) != 0:
                logger.error("주문 초기화 실패!")
                raise RuntimeError("Creon TradeInit 실패.")

            self.account_number = cpTradeUtil.AccountNumber[0]
            # GoodsList는 튜플을 반환하므로 첫 번째 요소를 가져옴 (대부분 '1' for 주식)
            self.account_flag = cpTradeUtil.GoodsList(self.account_number, 1)[0]
            logger.info(f"Creon API 계좌 정보 확인: 계좌번호={self.account_number}, 상품구분={self.account_flag}")

        except Exception as e:
            logger.error(f"Creon TradeUtil 초기화 또는 계좌 정보 가져오는 중 오류 발생: {e}", exc_info=True)
            raise  # 초기화 실패 시 예외 발생


    def _check_creon_status(self):
        """Creon API 사용 가능한지 상태를 확인합니다."""
        if not self.connected:
            logger.error("Creon Plus가 연결되지 않았습니다.")
            return False
        # 추가적인 요청 제한 확인 로직은 필요에 따라 여기에 구현
        return True

        # 요청 제한 개수 확인 (현재는 사용하지 않음 - 단순 시뮬레이션 목적)
        # remain_count = self.cp_cybos.GetLimitRequestRemainTime()
        # if remain_count <= 0:
        #    logger.warning(f"Creon API request limit reached. Waiting for 1 second.")
        #    time.sleep(1)
        #    remain_count = self.cp_cybos.GetLimitRequestRemainTime()
        #    if remain_count <= 0:
        #        logger.error("Creon API request limit still active after waiting. Cannot proceed.")
        #        return False
        return True

    def _is_spac(self, code_name):
        """종목명에 숫자+'호' 패턴이 있으면 스펙주로 판단합니다."""
        return re.search(r'\d+호', code_name) is not None

    def _is_preferred_stock(self, code):
        """우선주 판단, 코드 뒷자리가 0이 아님"""
        return code[-1] != '0'

    def _is_reits(self, code_name):
        """종목명에 '리츠'가 포함되면 리츠로 판단합니다."""
        return "리츠" in code_name

    def _make_stock_dic(self):
        """주식 종목 정보를 딕셔너리로 저장합니다. 스펙주, 우선주, 리츠 제외."""
        logger.info("종목 코드/명 딕셔너리 생성 시작")
        if not self.cp_code_mgr:
            logger.error("cp_code_mgr is not initialized. Cannot make stock dictionary.")
            return

        try:
            kospi_codes = self.cp_code_mgr.GetStockListByMarket(1)
            kosdaq_codes = self.cp_code_mgr.GetStockListByMarket(2)
            all_codes = kospi_codes + kosdaq_codes
            
            processed_count = 0
            for code in all_codes:
                code_name = self.cp_code_mgr.CodeToName(code)
                if not code_name: # 종목명이 없으면 유효하지 않은 종목으로 간주
                    continue

                # 1. 섹션 종류 필터링: 보통주(0)만 포함
                # Creon API GetStockSectionKind: 0:전체, 1:보통주, 2:선물, 3:옵션, 4:주식옵션, 5:ELW, 6:테마
                # NOTE: GetStockSectionKind는 GetStockSecKind (0:KOSPI, 1:KOSDAQ)와 다릅니다.
                if self.cp_code_mgr.GetStockSectionKind(code) != 1: # 보통주(1)가 아니면 다음 종목으로 건너뛰기
                    continue

                # 2. 이름 기반 필터링 (섹션 종류가 1이어도 이름으로 추가 확인)
                if (self._is_spac(code_name) or
                    self._is_preferred_stock(code) or
                    self._is_reits(code_name)):
                    continue

                # 3. 관리/투자경고/거래정지 등 상태 필터링
                # GetStockControlKind: 0:정상, 1:관리, 2:투자경고, 3:투자위험, 4:투자주의 등
                if self.cp_code_mgr.GetStockControlKind(code) != 0: 
                    continue
                # GetStockSupervisionKind: 0:정상, 1:투자유의
                if self.cp_code_mgr.GetStockSupervisionKind(code) != 0: 
                    continue
                # GetStockStatusKind: 0:정상, 2:거래정지, 3:거래중단
                if self.cp_code_mgr.GetStockStatusKind(code) in [2, 3]: 
                    continue
                
                self.stock_name_dic[code_name] = code
                self.stock_code_dic[code] = code_name
                processed_count += 1

            logger.info(f"종목 코드/명 딕셔너리 생성 완료. 총 {processed_count}개 종목 저장.")

        except Exception as e:
            logger.error(f"_make_stock_dic 중 오류 발생: {e}", exc_info=True)

    def get_stock_name(self, find_code: str) -> Optional[str]:
        """종목코드로 종목명을 반환 합니다."""
        return self.stock_code_dic.get(find_code, None)

    def get_stock_code(self, find_name: str) -> Optional[str]:
        """종목명으로 종목목코드를 반환 합니다."""
        return self.stock_name_dic.get(find_name, None)
    
    # def get_filtered_stock_list(self):
    #     """필터링된 모든 종목 코드를 리스트로 반환합니다."""
    #     return list(self.stock_code_dic.keys())

    def get_price_data(self, code: str, period: str, count: int) -> pd.DataFrame:
        """
        지정된 종목의 차트 데이터를 요청하고 DataFrame으로 반환합니다.

        Args:
            code (str): 종목코드 (e.g., 'A005930')
            period (str): 주기 ('D':일봉, 'W':주봉, 'M':월봉, 'm':분봉, 'T':틱봉)
            count (int): 요청할 데이터 개수

        Returns:
            pandas.DataFrame: 요청된 차트 데이터 (오류 발생 시 빈 DataFrame)
        """
        logger.info(f"종목 [{code}] 차트 데이터 요청 시작: 주기={period}, 개수={count}")

        try:
            objChart = win32com.client.Dispatch('CpSysDib.StockChart')

            # Set common input values
            objChart.SetInputValue(0, code)
            objChart.SetInputValue(1, ord('2'))  # 요청구분 2:개수 (1:기간)
            objChart.SetInputValue(4, count)     # 요청할 데이터 개수
            objChart.SetInputValue(6, ord(period)) # 주기 : D, W, M, m, T
            objChart.SetInputValue(9, ord('1'))  # 수정주가 사용 (1:적용, 0:미적용)

            # Define fields based on chart period type
            # Fields: [0:날짜, 1:시간, 2:시가, 3:고가, 4:저가, 5:종가, 8:거래량]
            # Note: GetDataValue indices will correspond to the order in this list
            if period in ['m', 'T']:
                # 분/틱 주기 시 시간 필드 포함
                chart_fields = [0, 1, 2, 3, 4, 5, 8]
                if period == 'm':
                    objChart.SetInputValue(7, 1) # 분봉 주기 (1분봉) - CpSysDib.StockChart는 1분봉만 가능
            else:
                # 일/주/월 주기 시 시간 필드 없음
                chart_fields = [0, 2, 3, 4, 5, 8]
            
            objChart.SetInputValue(5, chart_fields) # 요청 항목 설정

            # Request data
            ret = objChart.BlockRequest()

            # Handle COM object request errors
            if ret != 0:
                logger.error(f"종목 [{code}] 차트 요청 BlockRequest 오류: {ret}", exc_info=True)
                return pd.DataFrame()

            # Check API communication status
            rqStatus = objChart.GetDibStatus()
            rqMsg = objChart.GetDibMsg1()
            if rqStatus != 0:
                logger.error(f"종목 [{code}] 차트 요청 통신 오류: 상태={rqStatus}, 메시지={rqMsg}", exc_info=True)
                return pd.DataFrame()

            # Get received data count
            data_count = objChart.GetHeaderValue(3)
            logger.debug(f"종목 [{code}] 차트 데이터 {data_count}개 수신 완료.")

            if data_count == 0:
                logger.warning(f"종목 [{code}]에 대한 차트 데이터가 없습니다.")
                return pd.DataFrame()

            # Extract data and prepare for DataFrame
            data_records = []
            for i in range(data_count):
                record = {}
                
                date_val = str(objChart.GetDataValue(chart_fields.index(0), i)) # 날짜

                if period in ['m', 'T']:
                    time_val = str(objChart.GetDataValue(chart_fields.index(1), i)).zfill(6) # 시간 (HHMMSS)
                    # Combine date and time for full datetime string
                    datetime_str = f"{date_val}{time_val}"
                    try:
                        record['datetime'] = datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
                    except ValueError:
                        # If time is HHMM, try that format
                        try:
                            datetime_str = f"{date_val}{time_val[:4]}" # Take first 4 digits for HHMM
                            record['datetime'] = datetime.strptime(datetime_str, '%Y%m%d%H%M')
                        except ValueError:
                             logger.warning(f"Failed to parse datetime for {code}: {datetime_str}")
                             record['datetime'] = None # Or handle as error
                else:
                    try:
                        record['datetime'] = datetime.strptime(date_val, '%Y%m%d')
                    except ValueError:
                        logger.warning(f"Failed to parse date for {code}: {date_val}")
                        record['datetime'] = None

                # Extract OHLCV values using their original field numbers' index in chart_fields
                record['open'] = objChart.GetDataValue(chart_fields.index(2), i)
                record['high'] = objChart.GetDataValue(chart_fields.index(3), i)
                record['low'] = objChart.GetDataValue(chart_fields.index(4), i)
                record['close'] = objChart.GetDataValue(chart_fields.index(5), i)
                record['volume'] = objChart.GetDataValue(chart_fields.index(8), i)
                
                data_records.append(record)
            
            # Create DataFrame
            df = pd.DataFrame(data_records)

            # Set datetime as index and sort
            if 'datetime' in df.columns:
                df = df.dropna(subset=['datetime']) # Drop rows where datetime parsing failed
                if not df.empty:
                    df['datetime'] = pd.to_datetime(df['datetime']) # Ensure it's datetime object
                    df = df.set_index('datetime').sort_index(ascending=True) # Sort ascending for time series
            
            logger.debug(f"종목 [{code}] 차트 데이터 DataFrame 생성 완료. shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"종목 [{code}] 차트 데이터 요청 및 처리 중 예상치 못한 오류 발생: {e}", exc_info=True)
            return pd.DataFrame()


    def _get_price_data(self, stock_code, period, from_date_str, to_date_str, interval=1):
        """
        Creon API에서 주식 차트 데이터를 가져오는 내부 범용 메서드.
        :param stock_code: 종목 코드 (예: 'A005930')
        :param period: 'D': 일봉, 'W': 주봉, 'M': 월봉, 'm': 분봉
        :param from_date_str: 시작일 (YYYYMMDD 형식 문자열)
        :param to_date_str: 종료일 (YYYYMMDD 형식 문자열)
        :param interval: 분봉일 경우 주기 (기본 1분)
        :return: Pandas DataFrame
        """
        if not self._check_creon_status():
            # 연결 실패 시에도 필요한 컬럼을 가진 빈 DataFrame 반환
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        objChart = win32com.client.Dispatch('CpSysDib.StockChart')
        
        # 입력 값 설정
        objChart.SetInputValue(0, stock_code)
        objChart.SetInputValue(1, ord('1'))      # 요청구분 1:기간 2: 개수 (우리는 기간으로 요청)
        objChart.SetInputValue(2, int(to_date_str))  # 2: To 날짜 (long)
        objChart.SetInputValue(3, int(from_date_str)) # 3: From 날짜 (long)
        objChart.SetInputValue(6, ord(period))   # 주기
        objChart.SetInputValue(9, ord('1'))      # 수정주가 사용

        # 요청 항목 설정 (주기에 따라 달라짐)
        # backtrader에서 사용할 최종 컬럼명과 매핑될 초기 컬럼명 정의
        standard_ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']

        if period == 'm':
            objChart.SetInputValue(7, interval)  # 분틱차트 주기 (1분)
            # Creon API 필드 인덱스: 날짜(0), 시간(1), 종가(5), 고가(3), 저가(4), 시가(2), 거래량(8)
            # GetDataValue 인덱스: 0, 1, 2, 3, 4, 5, 6
            requested_fields = [0, 1, 2, 3, 4, 5, 8] # 날짜, 시간, 시가, 고가, 저가, 종가, 거래량 (이 순서대로 GetDataValue에서 추출)
            # DataList에 담을 딕셔너리의 키 (GetDataValue 인덱스에 매핑)
            data_keys = ['datetime', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'] 
            # Note: 'stock_code'는 직접 추가, 'datetime'은 날짜+시간 조합, 'open_price' 등은 GetDataValue 순서.
            # GetDataValue(2, i)는 시가(open_price), GetDataValue(3,i)는 고가(high_price) 등
        else: # 일봉, 주봉, 월봉
            # 요청 항목: 날짜(0), 시가(2), 고가(3), 저가(4), 종가(5), 거래량(8)
            requested_fields = [0, 2, 3, 4, 5, 8] 
            data_keys = ['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
            # Note: 'stock_code'는 직접 추가, 'date'는 날짜, 'trading_value'는 필요시 추가
        
        objChart.SetInputValue(5, requested_fields) # 요청할 데이터

        data_list = []
        
        while True:
            objChart.BlockRequest()
            time.sleep(self.request_interval) # 과도한 요청 방지 및 제한 시간 준수

            rq_status = objChart.GetDibStatus()
            rq_msg = objChart.GetDibMsg1()

            if rq_status != 0:
                logger.error(f"CpStockChart: 데이터 요청 실패. 통신상태: {rq_status}, 메시지: {rq_msg}")
                if rq_status == 5: # '해당 기간의 데이터 없음'
                    logger.warning(f"No data for {stock_code} in specified period ({from_date_str}~{to_date_str}).")
                # 오류 또는 데이터 없음 시, 빈 DataFrame에 표준 OHLCV 컬럼을 붙여서 반환
                return pd.DataFrame(columns=standard_ohlcv_columns)

            received_len = objChart.GetHeaderValue(3) # 현재 BlockRequest로 수신된 데이터 개수
            if received_len == 0:
                # 데이터가 전혀 없을 때도 표준 컬럼을 가진 빈 DataFrame 반환
                return pd.DataFrame(columns=standard_ohlcv_columns) 

            for i in range(received_len):
                row_data = {'stock_code': stock_code}
                if period == 'm':
                    date_val = objChart.GetDataValue(0, i) # 날짜 (YYYYMMDD, int)
                    time_val = objChart.GetDataValue(1, i) # 시간 (HHMM, int, 예: 901, 1000)
                    
                    # time_val을 4자리 문자열로 포맷팅 (예: 901 -> '0901')
                    time_str_padded = str(time_val).zfill(4) 
                    
                    try:
                        # 날짜와 시간을 합쳐 datetime 객체 생성
                        dt_obj = datetime.strptime(f"{date_val}{time_str_padded}", '%Y%m%d%H%M')
                        row_data['datetime'] = dt_obj
                    except ValueError as e:
                        logger.error(f"Error parsing minute datetime for {stock_code}: {date_val}{time_str_padded}. Error: {e}")
                        continue # 잘못된 날짜/시간 포맷은 건너뜀

                    # GetDataValue 인덱스 매핑 (requested_fields 순서에 따름)
                    row_data['open'] = objChart.GetDataValue(2, i) # 시가
                    row_data['high'] = objChart.GetDataValue(3, i) # 고가
                    row_data['low'] = objChart.GetDataValue(4, i)  # 저가
                    row_data['close'] = objChart.GetDataValue(5, i)# 종가
                    row_data['volume'] = objChart.GetDataValue(6, i)     # 거래량
                else: # 일봉, 주봉, 월봉
                    date_val = objChart.GetDataValue(0, i)
                    row_data['date'] = datetime.strptime(str(date_val), '%Y%m%d').date() # 일봉은 date 컬럼 (datetime.date 객체)
                    row_data['open'] = objChart.GetDataValue(1, i)
                    row_data['high'] = objChart.GetDataValue(2, i)
                    row_data['low'] = objChart.GetDataValue(3, i)
                    row_data['close'] = objChart.GetDataValue(4, i)
                    row_data['volume'] = objChart.GetDataValue(5, i)
                    row_data['change_rate'] = None # 요청하지 않은 필드
                    row_data['trading_value'] = 0 # 요청하지 않은 필드
                
                data_list.append(row_data)
            
            if not objChart.Continue: # 연속 조회할 데이터가 없으면 루프 종료
                break

        df = pd.DataFrame(data_list)
        
        if df.empty:
            # 데이터는 없지만, 성공적으로 루프를 빠져나왔을 경우에도 표준 컬럼을 가진 빈 DataFrame 반환
            return pd.DataFrame(columns=standard_ohlcv_columns)

        # 데이터가 있다면 컬럼명 변경 및 인덱스 설정
        if period == 'm':
            df = df.sort_values(by='datetime').set_index('datetime')
        else: # 일봉, 주봉, 월봉
            df['date'] = pd.to_datetime(df['date']) # date 컬럼이 현재는 date 객체일 것이므로 datetime으로 변환
            df = df.sort_values(by='date').set_index('date') # 'date' 컬럼을 인덱스로 설정
            df.index = df.index.normalize()

        # backtrader에서 요구하는 컬럼명으로 변경
        # df.rename(columns={
        #     'open_price': 'open',
        #     'high_price': 'high',
        #     'low_price': 'low',
        #     'close_price': 'close',
        #     'volume': 'volume'
        # }, inplace=True)
        
        # 핵심 수정: 숫자 컬럼들을 float 타입으로 명시적으로 변환
        for col in standard_ohlcv_columns: # ['open', 'high', 'low', 'close', 'volume']
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        return df

    def get_daily_ohlcv(self, code, from_date, to_date):
        # _get_price_data를 호출하여 일봉 데이터 가져오기
        df = self._get_price_data(code, 'D', from_date, to_date)
        # _get_price_data에서 이미 rename 처리
        # 빈 DataFrame일 경우에도 'open', 'high', 'low', 'close', 'volume' 컬럼이 보장됨
        logger.debug(f"Creon API 일봉 {from_date}~{to_date} {len(df)}건 로드.")

        return df[['open', 'high', 'low', 'close', 'volume']] 

    def get_minute_ohlcv(self, code, from_date, to_date, interval=1):
        # _get_price_data를 호출하여 분봉 데이터 가져오기
        df = self._get_price_data(code, 'm', from_date, to_date, interval)
        # _get_price_data에서 이미 rename 처리
        # 빈 DataFrame일 경우에도 'open', 'high', 'low', 'close', 'volume' 컬럼이 보장됨
        logger.debug(f"Creon API {interval}분봉 {from_date}~{to_date} {len(df)}건 로드.")
        return df[['open', 'high', 'low', 'close', 'volume']]

    def get_all_trading_days_from_api(self, from_date: date, to_date: date, stock_code: str = 'A005930') -> list[date]:
        """
        Creon API의 일봉 데이터 조회를 통해 특정 기간의 모든 거래일(날짜)을 가져옵니다.
        _get_price_data에서 반환되는 DatetimeIndex를 활용합니다.

        :param from_date: 조회 시작일 (datetime.date 객체)
        :param to_date: 조회 종료일 (datetime.date 객체)
        :param stock_code: 거래일을 조회할 기준 종목 코드 (기본값: 삼성전자 'A005930')
        :return: 거래일 날짜를 담은 list (datetime.date 객체들), 실패 시 빈 리스트
        """
        logger.info(f"Creon API를 통해 거래일 캘린더 조회 시작: {stock_code} ({from_date} ~ {to_date})")

        from_date_str = from_date.strftime('%Y%m%d')
        to_date_str = to_date.strftime('%Y%m%d')

        # _get_price_data는 일봉 데이터를 DatetimeIndex 인덱스로 가진 DataFrame을 반환합니다.
        # 이 인덱스의 각 요소는 pandas.Timestamp 객체이며, normalize()에 의해 시간 정보는 00:00:00으로 설정됩니다.
        ohlcv_df = self._get_price_data(stock_code, 'D', from_date_str, to_date_str)

        if ohlcv_df.empty:
            logger.warning(f"Creon API로부터 {stock_code}의 일봉 데이터를 가져오지 못했습니다. 거래일 없음.")
            return []
        
        # DatetimeIndex의 .date 속성을 사용하여 각 Timestamp에서 datetime.date 객체를 추출합니다.
        # 이 과정은 pandas의 DatetimeIndex가 datetime.date 객체와 호환되도록 설계되어 있어 안전합니다.
        trading_days = ohlcv_df.index.date.tolist()
        
        # _get_price_data에서 이미 인덱스 기준으로 정렬되지만, 최종적으로 정렬 및 중복 제거
        trading_days = sorted(list(set(trading_days)))

        logger.info(f"Creon API로부터 총 {len(trading_days)}개의 거래일 캘린더 데이터를 가져왔습니다.")
        return trading_days
    

    def get_current_price(self, stock_code: str) -> Optional[float]:
        """
        실시간 현재가를 조회합니다 (CpSysDib.StockMst 사용).
        """
        logger.debug(f"Fetching current price for {stock_code}")
        try:
            objStockMst = win32com.client.Dispatch("DsCbo1.StockMst")
            objStockMst.SetInputValue(0, stock_code)
            
            ret = objStockMst.BlockRequest()
            if ret == 0:
                # 필드 10: 현재가 (종가)
                current_price = float(objStockMst.GetHeaderValue(11)) # 종가 (실시간은 보통 현재가와 동일)
                logger.debug(f"Current price for {stock_code}: {current_price}")
                return current_price
            else:
                logger.error(f"BlockRequest failed for current price {stock_code}: {ret}")
                return None
        except Exception as e:
            logger.error(f"Error fetching current price for {stock_code}: {e}", exc_info=True)
            return None

    def get_current_minute_data(self, stock_code: str, count: int = 1) -> Optional[pd.DataFrame]:
        """
        실시간 1분봉 데이터를 조회합니다 (get_price_data 재활용).
        실시간 스트리밍이 아닌 요청 시점의 가장 최신 1분봉 데이터를 가져오는 방식.
        """
        logger.debug(f"Fetching current minute data for {stock_code}, count={count}")
        # CpSysDib.StockChart의 1분봉 조회 기능을 활용
        df = self.get_price_data(stock_code, 'm', count)
        if df is not None and not df.empty:
            # 가장 최근 데이터만 필요하다면 (count=1 기준)
            # df = df.tail(1)
            return df
        return None
    def get_latest_financial_data(self, stock_code) -> pd.DataFrame:
        """
        종목의 최신 재무 데이터를 조회합니다 (CpSysDib.MarketEye 사용).
        백테스팅의 creon_api.py의 get_latest_financial_data와 유사하게 구현.
        """
        logger.info(f"{stock_code} 종목의 최신 재무 데이터를 가져오는 중...")
        objMarketEye = win32com.client.Dispatch("CpSysDib.MarketEye")

        req_fields = [
            0,   # Field 0: 종목코드
            1,   # Field 1: 종목명
            11,  # Field 11: 현재가
            20,  # Field 20: PER
            21,  # Field 21: PBR
            22,  # Field 22: EPS
            67,  # Field 67: ROE
            70,  # Field 70: 부채비율
            110, # Field 110: 매출액(억)
            111, # Field 111: 영업이익(억)
            112, # Field 112: 당기순이익(억)
            161, # Field 161: 최근 결산년월 (YYYYMM 형식)
            4    # Field 4: 상장주식수 (시가총액 계산용)
        ]

        # 요청 필드 및 종목 코드 설정
        objMarketEye.SetInputValue(0, req_fields)
        objMarketEye.SetInputValue(1, stock_code)

        # 데이터 요청 (BlockRequest는 동기 방식으로 응답을 기다림)
        ret = objMarketEye.BlockRequest()
        if ret != 0:
            logger.error(f"재무 데이터 BlockRequest 실패 ({stock_code}): {ret}")
            return pd.DataFrame() # 빈 DataFrame 반환

        # 요청 상태 확인
        status = objMarketEye.GetDibStatus()
        msg = objMarketEye.GetDibMsg1()
        if status != 0:
            logger.error(f"재무 데이터 요청 에러 ({stock_code}): 상태={status}, 메시지={msg}")
            return pd.DataFrame()

        # 반환된 항목의 수 가져오기 (단일 종목 코드 요청 시 보통 1)
        cnt = objMarketEye.GetHeaderValue(2)
        
        data = []
        # 반환된 각 항목을 순회 (단일 종목 코드의 경우 보통 한 번 실행)
        for i in range(cnt):
            # GetDataValue(req_fields_인덱스, item_인덱스)를 사용하여 데이터 조회
            # 인덱스는 req_fields 리스트 내의 순서에 해당하며, Creon API의 원래 필드 번호와 매칭됨
            stock_code_res = objMarketEye.GetDataValue(0, i)  # 종목코드
            stock_name_res = objMarketEye.GetDataValue(1, i)  # 종목명
            current_price = objMarketEye.GetDataValue(2, i)   # 현재가
            per = objMarketEye.GetDataValue(3, i)             # PER
            pbr = objMarketEye.GetDataValue(4, i)             # PBR
            eps = objMarketEye.GetDataValue(5, i)             # EPS
            roe = objMarketEye.GetDataValue(6, i)             # ROE
            debt_ratio = objMarketEye.GetDataValue(7, i)      # 부채비율
            sales_billion = objMarketEye.GetDataValue(8, i)   # 매출액
            operating_profit_billion = objMarketEye.GetDataValue(9, i) # 영업이익
            net_profit_billion = objMarketEye.GetDataValue(10, i) # 당기순이익
            recent_financial_date_str = str(objMarketEye.GetDataValue(11, i)) # 최근 결산년월
            listed_stock = objMarketEye.GetDataValue(12, i)   # 상장주식수

            # 시가총액 계산
            market_cap = listed_stock * current_price
            if self.cp_code_mgr and self.cp_code_mgr.IsBigListingStock(stock_code_res):
                market_cap *= 1000  # Creon API 문서에 따라 대형주 시가총액 조정 (필요한 경우)
            print(f"{stock_code_res} {stock_name_res} 시총: {market_cap:,} 원")

            recent_financial_date = None
            if len(recent_financial_date_str) == 6: # 예상 형식: YYYYMM
                try:
                    recent_financial_date = datetime.strptime(recent_financial_date_str, '%Y%m').date()
                except ValueError:
                    logger.warning(f"재무 일자 파싱 실패: {recent_financial_date_str} (종목: {stock_code_res})")

            # '억' 단위 데이터를 '원' 단위로 변환
            sales = float(sales_billion) * 100_000_000
            operating_profit = float(operating_profit_billion) * 100_000_000
            net_profit = float(net_profit_billion) * 100_000_000

            finance = {
                'stock_code': stock_code_res,
                'stock_name': stock_name_res,
                'current_price': float(current_price),
                'per': float(per) if per != 0 else None,
                'pbr': float(pbr) if pbr != 0 else None,
                'eps': float(eps) if eps != 0 else None,
                'roe': float(roe) if roe != 0 else None,
                'debt_ratio': float(debt_ratio) if debt_ratio != 0 else None,
                'sales': sales,
                'operating_profit': operating_profit,
                'net_profit': net_profit,
                'recent_financial_date': recent_financial_date,
                'market_cap': market_cap # 데이터프레임에 시가총액 추가
            }
            data.append(finance)
        
        df = pd.DataFrame(data)
        logger.info(f"{stock_code} 종목의 재무 데이터 조회를 성공적으로 완료했습니다.")
        return df

    # --- 주문 관련 메서드 ---
    def send_order(self, stock_code: str, order_type: str, price: float, quantity: int, order_kind: str = '01', org_order_no: str = '') -> Optional[str]:
        """
        주문 전송 (매수/매도/정정/취소).
        order_type: 'buy', 'sell'
        order_kind: '01'(보통), '03'(시장가) 등 Creon 주문 종류 코드
        org_order_no: 정정/취소 시 원주문번호
        """
        if not self.connected:
            logger.error("Creon API is not connected. Cannot send order.")
            return None

        objOrder = win32com.client.Dispatch("CpTrade.CpTd0311")
        
        # 입력 값 설정
        if order_type == 'buy':
            objOrder.SetInputValue(0, ord('2')) # '2': 매수
        elif order_type == 'sell':
            objOrder.SetInputValue(0, ord('1')) # '1': 매도
        else:
            logger.error(f"Unsupported order type: {order_type}")
            return None
        
        objOrder.SetInputValue(1, self.account_number) # 계좌번호
        objOrder.SetInputValue(2, self.account_flag) # 상품구분
        objOrder.SetInputValue(3, stock_code)       # 종목코드
        objOrder.SetInputValue(4, int(quantity))    # 주문수량
        objOrder.SetInputValue(5, int(price))       # 주문가격(단가) (시장가는 의미 없음)
        objOrder.SetInputValue(7, ord('0'))         # 주문 조건 (0:기본) - IOC/FOK 등 필요시 수정
        objOrder.SetInputValue(8, order_kind)       # 주문 종류: 01-보통, 03-시장가
        
        # '보통' 주문 시에만 유효한 가격 필드
        # objOrder.SetInputValue(7, "0")  # '0': 주문조건 구분 코드 (없음)
        # objOrder.SetInputValue(8, "01") # '01': 신용주문 구분코드 (대출,신용 등은 01:보통)

        # 정정/취소 주문 시 원주문번호 필요
        if org_order_no:
            objOrder.SetInputValue(9, org_order_no) # 원주문번호 (정정/취소 시 사용)

        # 주문 요청
        ret = objOrder.BlockRequest()
        if ret != 0:
            logger.error(f"Order BlockRequest failed for {stock_code} {order_type} {quantity}@{price}: {ret}")
            return None
        
        status = objOrder.GetDibStatus()
        msg = objOrder.GetDibMsg1()
        if status != 0:
            logger.error(f"Order request error for {stock_code}: Status={status}, Msg={msg}")
            return None

        # 주문 성공 시 반환 값
        # GetHeaderValue(4) : 주문번호
        # GetHeaderValue(5) : 주문수량
        order_id = str(objOrder.GetHeaderValue(4))
        order_qty = int(objOrder.GetHeaderValue(5))
        logger.info(f"Order sent successfully: {order_type.upper()} {stock_code}, Qty: {order_qty}, Price: {price}, OrderID: {order_id}")
        return order_id

    def get_account_balance(self) -> Dict[str, float]:
        """
        계좌 잔고 (현금) 및 예수금 정보를 조회합니다.
        """
        logger.debug("Fetching account balance...")
        try:
            objCash = win32com.client.Dispatch("CpTrade.CpTdNew5331A")
            objCash.SetInputValue(0, self.account_number)
            objCash.SetInputValue(1, self.account_flag)
            
            ret = objCash.BlockRequest()
            if ret != 0:
                logger.error(f"BlockRequest failed for account balance: {ret}")
                return {"cash": 0.0, "deposit": 0.0}

            status = objCash.GetDibStatus()
            msg = objCash.GetDibMsg1()
            if status != 0:
                logger.error(f"Account balance request error: Status={status}, Msg={msg}")
                return {"cash": 0.0, "deposit": 0.0}

            # 예수금, 매도 가능 금액 등 조회
            cash = float(objCash.GetHeaderValue(9)) # 주문가능금액
            deposit = float(objCash.GetHeaderValue(13)) # 예수금
            logger.info(f"Account Balance: Cash={cash:,.0f}, Deposit={deposit:,.0f}")
            return {"cash": cash, "deposit": deposit}
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}", exc_info=True)
            return {"cash": 0.0, "deposit": 0.0}

    def get_portfolio_positions(self) -> List[Dict[str, Any]]:
        """
        현재 보유 종목 리스트 및 상세 정보를 조회합니다.
        """
        logger.debug("Fetching portfolio positions...")
        try:
            objRp = win32com.client.Dispatch("CpTrade.CpTd6033")
            objRp.SetInputValue(0, self.account_number)
            objRp.SetInputValue(1, self.account_flag)
            objRp.SetInputValue(2, 50) # 요청할 개수 (최대 50개)

            positions = []
            while True:
                ret = objRp.BlockRequest()
                if ret != 0:
                    logger.error(f"BlockRequest failed for portfolio positions: {ret}")
                    break

                status = objRp.GetDibStatus()
                msg = objRp.GetDibMsg1()
                if status != 0:
                    logger.error(f"Portfolio positions request error: Status={status}, Msg={msg}")
                    break

                cnt = objRp.GetHeaderValue(7) # 수신 개수
                for i in range(cnt):
                    stock_code = objRp.GetDataValue(12, i) # 종목코드
                    stock_name = objRp.GetDataValue(0, i) # 종목명
                    current_qty = int(objRp.GetDataValue(7, i)) # 잔고수량
                    avg_price = float(objRp.GetDataValue(9, i)) # 매입단가
                    
                    # 현재가는 별도로 조회 필요 (StockMst 사용)
                    # 여기서는 일단 잔고 정보만 가져오고, 현재가는 BusinessManager에서 별도로 호출
                    positions.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'size': current_qty,
                        'avg_price': avg_price
                    })

                if not objRp.Continue: # 연속 데이터가 없으면
                    break
                time.sleep(self.request_interval)

            logger.info(f"Fetched {len(positions)} portfolio positions.")
            return positions
        except Exception as e:
            logger.error(f"Error fetching portfolio positions: {e}", exc_info=True)
            return []

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        특정 주문의 체결 상태를 조회합니다 (CpTrade.CpTd0311 - 주문확인).
        order_id: 주문번호
        """
        logger.debug(f"Fetching order status for order ID: {order_id}")
        try:
            objReq = win32com.client.Dispatch("CpTrade.CpTd0311")
            objReq.SetInputValue(0, self.account_number)
            objReq.SetInputValue(1, self.account_flag)
            objReq.SetInputValue(2, order_id) # 원주문번호 (조회할 주문번호)

            ret = objReq.BlockRequest()
            if ret != 0:
                logger.error(f"BlockRequest failed for order status {order_id}: {ret}")
                return {"status": "ERROR", "message": f"BlockRequest failed: {ret}"}
            
            status = objReq.GetDibStatus()
            msg = objReq.GetDibMsg1()
            if status != 0:
                logger.error(f"Order status request error {order_id}: Status={status}, Msg={msg}")
                return {"status": "ERROR", "message": f"API error: {msg}"}

            # 반환 필드 확인 (CpTrade.CpTd0311 설명서 참고)
            # 1: 주문상태 (접수, 체결, 확인, 거부 등)
            # 5: 체결수량
            # 6: 체결가격
            order_status = objReq.GetHeaderValue(1)
            executed_qty = int(objReq.GetHeaderValue(5))
            executed_price = float(objReq.GetHeaderValue(6))

            logger.info(f"Order {order_id} Status: {order_status}, Executed Qty: {executed_qty}, Price: {executed_price}")
            return {
                "status": order_status,
                "executed_quantity": executed_qty,
                "executed_price": executed_price
            }
        except Exception as e:
            logger.error(f"Error fetching order status for {order_id}: {e}", exc_info=True)
            return {"status": "ERROR", "message": str(e)}