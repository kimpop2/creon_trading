# api_client/creon_api.py

import win32com.client
import ctypes
import time
import logging
import pandas as pd
#import datetime # datetime 모듈 전체를 임포트하여 datetime.timedelta 사용 가능
import re
from datetime import datetime, date, timedelta
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
        self._connect_creon()
        if self.connected:
            self.cp_code_mgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")
            logger.info("CpCodeMgr COM object initialized.")
            self._make_stock_dic()

    def _connect_creon(self):
        """Creon Plus에 연결하고 COM 객체를 초기화합니다."""
        if ctypes.windll.shell32.IsUserAnAdmin():
            logger.info("Running with administrator privileges.")
        else:
            logger.warning("Not running with administrator privileges. Some Creon functions might be restricted.")

        self.cp_cybos = win32com.client.Dispatch("CpUtil.CpCybos")
        if self.cp_cybos.IsConnect:
            self.connected = True
            logger.info("Creon Plus is already connected.")
        else:
            logger.info("Attempting to connect to Creon Plus...")
            # self.cp_cybos.PlusConnect() # 보통 HTS가 실행되어 있으면 자동 연결되므로 주석 처리
            max_retries = 10
            for i in range(max_retries):
                if self.cp_cybos.IsConnect:
                    self.connected = True
                    logger.info("Creon Plus connected successfully.")
                    break
                else:
                    logger.warning(f"Waiting for Creon Plus connection... ({i+1}/{max_retries})")
                    time.sleep(2)
            if not self.connected:
                logger.error("Failed to connect to Creon Plus. Please ensure HTS is running and logged in.")
                raise ConnectionError("Creon Plus connection failed.")

    def _check_creon_status(self):
        """Creon API 사용 가능한지 상태를 확인합니다."""
        if not self.connected:
            logger.error("Creon Plus is not connected.")
            return False

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

    def get_stock_name(self, find_code):
        """종목코드로 종목명을 반환 합니다."""
        return self.stock_code_dic.get(find_code, None)

    def get_stock_code(self, find_name):
        """종목명으로 종목목코드를 반환 합니다."""
        return self.stock_name_dic.get(find_name, None)

    def get_filtered_stock_list(self):
        """필터링된 모든 종목 코드를 리스트로 반환합니다."""
        return list(self.stock_code_dic.keys())

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
    