# backtesting/data_manager/stock_data_manager.py

import logging
import pandas as pd
from datetime import datetime, timedelta, date
import os
import sys

# sys.path에 프로젝트 루트 추가 (모듈 임포트를 위함)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from db.db_manager import DBManager
from api.creon_api import CreonAPIClient
# from config.settings import DEFAULT_OHLCV_DAYS_TO_FETCH # 향후 사용될 수 있음

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StockDataManager:
    def __init__(self, db_manager: DBManager, creon_api_client: CreonAPIClient):
        self.db_manager = db_manager
        self.creon_api_client = creon_api_client
        logger.info("StockDataManager 초기화 완료.")

    def update_all_stock_info(self):
        """
        Creon API에서 모든 종목 정보를 가져와 DB의 stock_info 테이블에 저장/업데이트합니다.
        기본 정보(코드, 이름, 시장)만 먼저 채우고, 재무 데이터는 별도 메서드로 업데이트합니다.
        """
        logger.info("모든 종목 기본 정보 업데이트를 시작합니다.")
        if not self.creon_api_client.connected:
            logger.error("Creon API가 연결되어 있지 않아 종목 정보를 가져올 수 없습니다.")
            return False

        try:
            filtered_codes = self.creon_api_client.get_filtered_stock_list()
            stock_info_list = []
            for code in filtered_codes:
                name = self.creon_api_client.get_stock_name(code)
                market_type = 'KOSPI' if code in self.creon_api_client.cp_code_mgr.GetStockListByMarket(1) else 'KOSDAQ'

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
                # DBManager의 save_stock_info를 호출하여 기본 정보만 먼저 저장 (재무 필드는 None)
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

    def update_daily_ohlcv(self, stock_code, start_date=None, end_date=None):
        """
        특정 종목의 일봉 데이터를 Creon API에서 가져와 DB에 저장/업데이트합니다.
        기존 DB에 데이터가 있다면 최신 날짜 이후의 데이터만 가져와서 추가합니다.
        :param stock_code: 종목 코드
        :param start_date: 조회 시작 날짜 (datetime.date 객체). None이면 DB 최신 날짜 + 1일 부터 조회.
        :param end_date: 조회 종료 날짜 (datetime.date 객체). None이면 오늘 날짜까지 조회.
        """
        logger.info(f"{stock_code} 일봉 데이터 업데이트를 시작합니다.")

        if not self.creon_api_client.connected:
            logger.error("Creon API가 연결되어 있지 않아 일봉 데이터를 가져올 수 없습니다.")
            return False

        if not end_date:
            end_date = date.today()

        db_latest_date = self.db_manager.get_latest_daily_data_date(stock_code)

        fetch_start_date = start_date
        if db_latest_date:
            # DB에 이미 데이터가 있다면, 최신 날짜 다음 날부터 가져옵니다.
            if fetch_start_date: # 시작 날짜가 지정된 경우
                fetch_start_date = max(fetch_start_date, db_latest_date + timedelta(days=1))
            else: # 시작 날짜가 지정되지 않은 경우
                fetch_start_date = db_latest_date + timedelta(days=1)
        elif not fetch_start_date:
            # DB에 데이터가 없고, 시작 날짜도 지정되지 않았다면, 5년 전부터 가져옵니다.
            fetch_start_date = end_date - timedelta(days=365 * 5) # 기본 5년치

        if fetch_start_date > end_date:
            logger.info(f"{stock_code} 일봉 데이터는 최신 상태입니다. 업데이트할 데이터가 없습니다.")
            return True

        start_date_str = fetch_start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')

        ohlcv_df = self.creon_api_client.get_daily_ohlcv(stock_code, start_date_str, end_date_str)

        if ohlcv_df.empty:
            logger.info(f"{stock_code} 기간 {start_date_str}~{end_date_str} 동안 Creon API에서 조회된 일봉 데이터가 없습니다.")
            return True

        # 등락률(change_rate) 계산
        # DB에서 이전 종가를 가져와서 계산하거나, 조회된 데이터프레임 내에서 계산
        ohlcv_df = ohlcv_df.sort_values(by='date', ascending=True).reset_index(drop=True)
        # 첫 데이터의 이전 종가는 DB에서 가져와야 하지만, 현재는 DataFrame 내에서만 처리
        # 만약 fetch_start_date가 DB의 첫 데이터가 아니라면, 이전 데이터의 종가를 가져와야 정확한 등락률 계산 가능
        # 복잡도를 위해 여기서는 조회된 DataFrame 내에서만 계산하고, 첫 행의 등락률은 0으로 처리합니다.
        ohlcv_df['prev_close_price'] = ohlcv_df['close_price'].shift(1)
        ohlcv_df['change_rate'] = ((ohlcv_df['close_price'] - ohlcv_df['prev_close_price']) / ohlcv_df['prev_close_price'] * 100).round(2)
        ohlcv_df.loc[0, 'change_rate'] = 0.0 # 첫 데이터의 등락률은 0으로 설정

        # 필요한 컬럼만 선택하여 DB에 저장할 형태로 변환
        save_data = ohlcv_df[['stock_code', 'date', 'open_price', 'high_price',
                              'low_price', 'close_price', 'volume', 'change_rate', 'trading_value']].to_dict(orient='records')

        if save_data:
            if self.db_manager.save_daily_data(save_data):
                logger.info(f"{stock_code} 일봉 데이터 {len(save_data)}개를 성공적으로 DB에 업데이트했습니다.")
                return True
            else:
                logger.error(f"{stock_code} 일봉 데이터 DB 저장에 실패했습니다.")
                return False
        else:
            logger.info(f"{stock_code} 업데이트할 새로운 일봉 데이터가 없습니다.")
            return True

    def update_minute_ohlcv(self, stock_code, start_datetime=None, end_datetime=None, interval=1):
        """
        특정 종목의 분봉 데이터를 Creon API에서 가져와 DB에 저장/업데이트합니다.
        기존 DB에 데이터가 있다면 최신 시각 이후의 데이터만 가져와서 추가합니다.
        :param stock_code: 종목 코드
        :param start_datetime: 조회 시작 시각 (datetime.datetime 객체). None이면 DB 최신 시각 + 1분 부터 조회.
        :param end_datetime: 조회 종료 시각 (datetime.datetime 객체). None이면 현재 시각까지 조회.
        :param interval: 분봉 주기 (기본 1분)
        """
        logger.info(f"{stock_code} {interval}분봉 데이터 업데이트를 시작합니다.")

        if not self.creon_api_client.connected:
            logger.error("Creon API가 연결되어 있지 않아 분봉 데이터를 가져올 수 없습니다.")
            return False

        if not end_datetime:
            end_datetime = datetime.now()

        db_latest_datetime = self.db_manager.get_latest_minute_data_datetime(stock_code)

        fetch_start_datetime = start_datetime
        if db_latest_datetime:
            # DB에 이미 데이터가 있다면, 최신 시각 다음 분부터 가져옵니다.
            if fetch_start_datetime: # 시작 시각이 지정된 경우
                fetch_start_datetime = max(fetch_start_datetime, db_latest_datetime + timedelta(minutes=interval))
            else: # 시작 시각이 지정되지 않은 경우
                fetch_start_datetime = db_latest_datetime + timedelta(minutes=interval)
        elif not fetch_start_datetime:
            # DB에 데이터가 없고, 시작 시각도 지정되지 않았다면, 최근 7일치만 가져옵니다.
            fetch_start_datetime = end_datetime - timedelta(days=7) # 기본 7일치

        if fetch_start_datetime > end_datetime:
            logger.info(f"{stock_code} 분봉 데이터는 최신 상태입니다. 업데이트할 데이터가 없습니다.")
            return True

        start_date_str = fetch_start_datetime.strftime('%Y%m%d')
        end_date_str = end_datetime.strftime('%Y%m%d')

        ohlcv_df = self.creon_api_client.get_minute_ohlcv(stock_code, start_date_str, end_date_str, interval)

        if ohlcv_df.empty:
            logger.info(f"{stock_code} 기간 {start_date_str}~{end_date_str} 동안 Creon API에서 조회된 분봉 데이터가 없습니다.")
            return True

        # 필요한 컬럼만 선택하여 DB에 저장할 형태로 변환
        save_data = ohlcv_df[['stock_code', 'datetime', 'open_price', 'high_price',
                              'low_price', 'close_price', 'volume']].to_dict(orient='records')

        if save_data:
            if self.db_manager.save_minute_data(save_data):
                logger.info(f"{stock_code} 분봉 데이터 {len(save_data)}개를 성공적으로 DB에 업데이트했습니다.")
                return True
            else:
                logger.error(f"{stock_code} 분봉 데이터 DB 저장에 실패했습니다.")
                return False
        else:
            logger.info(f"{stock_code} 업데이트할 새로운 분봉 데이터가 없습니다.")
            return True

    def update_financial_data_for_stock_info(self, stock_code): # 메서드명 변경 및 역할 명확화
        """
        특정 종목의 최신 재무 데이터를 CreonAPIClient (MarketEye)에서 가져와
        DB의 stock_info 테이블에 업데이트합니다.
        """
        logger.info(f"{stock_code} stock_info 테이블의 최신 재무 데이터 업데이트 중...")
        
        try:
            # Creon API Client에서 최신 재무 데이터 가져오기
            finance_df = self.creon_api_client.get_latest_financial_data(stock_code)

            if finance_df.empty:
                logger.info(f"{stock_code} Creon API에서 조회된 재무 데이터가 없습니다.")
                return

            # MarketEye에서 가져온 DataFrame을 stock_info 테이블의 컬럼에 맞게 조정
            # sales, operating_profit, net_profit은 MarketEye에서 원단위로 제공될 수 있으므로
            # 여기서는 백만 원 단위로 변환합니다. (MarketEye 문서 기준)
            # Creon API 필드 86(매출액): 백만, 91(영업이익): 원, 88(당기순이익): 원
            # stock_info 테이블에 모두 백만 원 단위로 저장하는 것이 일관적입니다.
            finance_df['operating_profit'] = finance_df['operating_profit'] / 1_000_000
            finance_df['net_profit'] = finance_df['net_profit'] / 1_000_000

            # DBManager의 save_stock_info (ON DUPLICATE KEY UPDATE 활용)를 호출하여
            # stock_info 테이블의 재무 관련 컬럼을 업데이트합니다.
            # 이 메서드는 전체 stock_info 스키마에 맞춰 인자를 받으므로, DataFrame을 그대로 전달합니다.
            self.db_manager.save_stock_info(finance_df.to_dict(orient='records'))

            logger.info(f"{stock_code} stock_info 테이블의 최신 재무 데이터가 성공적으로 업데이트되었습니다.")

        except Exception as e:
            logger.error(f"stock_info 재무 데이터 업데이트 중 오류 발생: {e}", exc_info=True)
        logger.info(f"{stock_code} 재무 데이터 업데이트 완료.")