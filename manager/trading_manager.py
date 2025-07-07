# manager/trading_manager.py (수정된 부분)

import logging
import pandas as pd
from datetime import datetime, date, timedelta, time
import time as time_module # time.sleep 사용을 위해 임포트
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

        # 실시간 데이터를 위한 캐시 (필요에 따라 확장)
        self.realtime_ohlcv_cache: Dict[str, pd.DataFrame] = {} # 종목별 실시간 분봉 데이터를 저장

        # DB 데이터로 종목정보 구성        
        self.stock_names: Dict[str, str] = {} # 종목 코드: 종목명 매핑 캐시
        self._load_stock_names() # 종목명 캐시 초기화
        logger.info(f"TradingManager 초기화 완료: {len(self.stock_names)} 종목, CreonAPIClient 및 DBManager 연결")

    def close(self):
        """DBManager의 연결을 종료합니다."""
        if self.db_manager:
            self.db_manager.close()
            logger.info("BacktestManager를 통해 DB 연결을 종료했습니다.")
    
    def _load_stock_names(self):
        """DB에서 모든 종목 코드와 이름을 로드하여 캐시합니다."""
        self.stock_names = self.get_stock_info_map()
        logger.info(f"종목명 {len(self.stock_names)}건 캐시 완료.")
    
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
        
    def get_stock_name(self, stock_code: str) -> str:
        """종목 코드로 종목명을 조회합니다. 캐시에 없으면 DB에서 조회 후 캐시합니다."""
        if stock_code not in self.stock_names:
            stock_info = self.db_manager.fetch_stock_info(stock_code)
            if stock_info:
                self.stock_names[stock_code] = stock_info['stock_name']
            else:
                self.stock_names[stock_code] = "알 수 없음" # 또는 에러 처리
        return self.stock_names.get(stock_code, "알 수 없음")

    def get_all_stock_list(self) -> List[Tuple[str, str]]:
        """캐시된 모든 종목의 코드와 이름을 반환합니다."""
        return [(code, name) for code, name in self.stock_names.items()]

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

    # def fetch_minute_ohlcv(self, stock_code: str, from_datetime: datetime, to_datetime: datetime) -> pd.DataFrame:
    #     """
    #     DB에서 분봉 OHLCV 데이터를 조회합니다.
    #     필요시 Creon API를 통해 데이터를 업데이트할 수 있습니다.
    #     """
    #     df = self.db_manager.fetch_minute_price(stock_code, from_datetime, to_datetime)
    #     if df.empty:
    #         logger.warning(f"DB에 {stock_code}의 분봉 데이터가 없습니다. Creon API를 통해 조회 시도합니다.")
    #         # Creon API를 통해 데이터 조회 및 저장 로직 추가
    #         # get_price_data는 count 기반이므로, 기간 기반으로 가져오려면 여러 번 호출해야 할 수 있음.
    #         # 여기서는 편의상 최근 200개 분봉을 가져오는 것으로 가정.
    #         df_from_api = self.api_client.get_price_data(stock_code, 'm', 200) # 200분봉
    #         if not df_from_api.empty:
    #             # DB에 저장
    #             df_from_api['stock_code'] = stock_code
    #             df_from_api['stock_name'] = self.get_stock_name(stock_code)
    #             # DBManager의 insert_df_to_db는 if_exists='append'만 지원하므로, 중복 방지를 위해 직접 SQL 사용
    #             conn = self.db_manager.get_db_connection()
    #             if conn:
    #                 try:
    #                     with conn.cursor() as cursor:
    #                         for index, row in df_from_api.iterrows():
    #                             sql = """
    #                                 INSERT INTO minute_price (stock_code, stock_name, datetime, open, high, low, close, volume)
    #                                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    #                                 ON DUPLICATE KEY UPDATE
    #                                     open = VALUES(open), high = VALUES(high), low = VALUES(low), close = VALUES(close), volume = VALUES(volume),
    #                                     updated_at = CURRENT_TIMESTAMP
    #                             """
    #                             params = (
    #                                 row['stock_code'], row['stock_name'], row['datetime'],
    #                                 row['open'], row['high'], row['low'], row['close'], row['volume']
    #                             )
    #                             cursor.execute(sql, params)
    #                         conn.commit()
    #                         logger.info(f"종목 {stock_code}의 분봉 데이터 {len(df_from_api)}건 API 조회 및 DB 업데이트 완료.")
    #                         df = self.db_manager.fetch_minute_price(stock_code, from_datetime, to_datetime) # 새로 저장된 데이터 다시 로드
    #                 except Exception as e:
    #                     logger.error(f"분봉 데이터 API 조회 및 DB 저장 중 오류: {e}")
    #                     conn.rollback()
    #         else:
    #             logger.warning(f"Creon API에서도 {stock_code}의 분봉 데이터를 가져올 수 없습니다.")
    #     return df

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
            position_data = {
                'stock_code': stock_code, # Already defined from api_pos['stock_code']
                'stock_name': stock_name,
                'quantity': api_pos['quantity'],
                'sell_avail_qty': api_pos['sell_avail_qty'],
                'average_buy_price': api_pos['average_buy_price'],
                'eval_profit_loss': api_pos['eval_profit_loss'],
                'eval_return_rate': api_pos['eval_return_rate'],
                'entry_date': datetime.now().date() # Ensure datetime is imported
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
        주어진 종목 코드 리스트에 대한 현재 시장 가격을 조회합니다.
        (실시간 데이터 또는 TR 요청)
        """
        prices = {}
        prices = self.api_client.get_current_prices(stock_codes)
        if prices:
            return prices
        else:
            logger.warning(f"종목 {stock_codes}의 현재 시장 가격을 가져올 수 없습니다.")

            
