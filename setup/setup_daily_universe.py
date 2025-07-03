import sys
import os
from datetime import date, timedelta, datetime
import logging
import time
import pandas as pd # market_calendar 처리용으로 필요

# 프로젝트 루트 경로를 sys.path에 추가 (manager 디렉토리에서 실행 가능하도록)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 필요한 모듈 임포트
from manager.setup_manager import SetupManager
from manager.backtest_manager import BacktestManager
from manager.db_manager import DBManager # fetch_market_calendar 사용을 위해 직접 임포트

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyUniverseFiller:
    """
    daily_universe 테이블에 과거 데이터를 채우는 작업을 관리하는 클래스.
    일봉 데이터 사전 캐싱 및 SetupManager 반복 실행을 포함합니다.
    """
    def __init__(self, start_date: date, end_date: date, setup_parameters: dict = None):
        """
        DailyUniverseFiller를 초기화합니다.
        :param start_date: 데이터 채우기를 시작할 전체 기간의 시작 날짜
        :param end_date: 데이터 채우기를 종료할 전체 기간의 종료 날짜
        :param setup_parameters: SetupManager에 전달할 사용자 정의 파라미터 (없으면 기본값 사용)
        """
        self.start_fill_date = start_date
        self.end_fill_date = end_date
        
        # SetupManager에 전달할 파라미터 설정 (기본값 제공)
        self.setup_parameters = setup_parameters if setup_parameters is not None else {
            'weight_price_trend': 50,
            'weight_trading_volume': 20,
            'weight_volatility': 15,
            'weight_theme_mention': 15,
            'ma_window_short': 5,
            'ma_window_long': 20,
            'volume_recent_days': 10,     # 거래량 점수 계산을 위한 최근 거래대금 기간
            'atr_window': 14,            # 변동성 측정 ATR 기간
            'daily_range_ratio_window': 7 # 일중 변동폭 비율 기간 (최소 데이터 요구 사항 및 평균 계산 기간)
        }
        
    def prepare_theme_daily_price(self):
        """
        daily_universe 테이블에 등록될 종목들의 일봉 데이터를 미리 캐싱합니다.
        `daily_universe` 테이블에서 해당 기간의 종목 코드를 가져와 `daily_price` 테이블에 데이터를 셋팅합니다.
        캐시 함수 특성상 이미 있는 데이터는 다시 가져오지 않으므로 효율적입니다.
        """
        logger.info(f"--- 일봉 데이터 사전 캐싱 시작 (daily_universe 종목 기준) ---")
        logger.info(f"전체 조회 기간: {self.start_fill_date} ~ {self.end_fill_date}")

        # BacktestManager 인스턴스를 생성하고, 그 안에 포함된 db_manager를 사용
        backtest_manager = BacktestManager()
        self.db_manager = backtest_manager.db_manager 
        
        try:
            # 1. daily_universe 테이블에서 전체 기간에 해당하는 고유 종목 코드 가져오기
            # 올바른 메서드명: fetch_stock_codes_from_daily_theme_by_date_range
            unique_stock_info = self.db_manager.fetch_daily_theme_stock(
                self.start_fill_date, self.end_fill_date
            )
            
            if not unique_stock_info: # 변경된 변수명 사용
                logger.warning("지정된 기간 내 daily_universe에 등록된 종목이 없습니다. 일봉 캐싱을 건너뜁니다.")
                return

            logger.info(f"총 {len(unique_stock_info)}개의 종목에 대해 일봉 데이터를 캐싱합니다.") # 변경된 변수명 사용

            # 2. 각 종목에 대해 일봉 데이터 캐싱할 기간 설정
            cache_from_date = self.start_fill_date - timedelta(days=40) # 20일치로 수정
            cache_to_date = self.end_fill_date

            logger.info(f"캐싱할 일봉 데이터 기간: {cache_from_date} ~ {cache_to_date}")

            # 3. BacktestManager의 cache_daily_ohlcv를 사용하여 데이터 캐싱
            # unique_stock_info의 각 항목은 (stock_code, stock_name) 튜플이므로 언패킹하여 사용합니다.
            for i, (stock_code, stock_name) in enumerate(unique_stock_info): # <--- 이 부분이 중요하게 수정되었습니다.
                try:
                    # cache_daily_ohlcv는 이미 DB에 있는 데이터는 건너뛰므로 효율적입니다.
                    backtest_manager.cache_daily_ohlcv(stock_code, cache_from_date, cache_to_date) # <--- stock_code를 직접 사용
                    if (i + 1) % 50 == 0 or (i + 1) == len(unique_stock_info): # 변경된 변수명 사용
                        logger.info(f"... {i+1}/{len(unique_stock_info)} 종목 일봉 데이터 캐싱 중...") # 변경된 변수명 사용
                except Exception as e:
                    logger.error(f"종목 {stock_name} ({stock_code})의 일봉 데이터 캐싱 중 오류 발생: {e}", exc_info=True) # <--- stock_name과 stock_code를 직접 사용
                
                # API 요청 빈도 조절이 필요할 경우 주석 해제 (예: time.sleep(0.05))
                # time.sleep(0.05) 
                
        except Exception as e:
            logger.critical(f"일봉 데이터 사전 캐싱 중 치명적인 오류 발생: {e}", exc_info=True)
        finally:
            # DBManager 연결을 닫습니다. (BacktestManager 내부의 db_manager 인스턴스)
            # if self.db_manager: 
            #     self.db_manager.close() # 명시적으로 연결 닫기
            logger.info("일봉 데이터 사전 캐싱 완료.")

    def run_daily_universe_filling_process(self):
        """
        지정된 기간 동안 `daily_universe` 테이블에 데이터를 채웁니다.
        영업일(trading day)에만 SetupManager를 실행하여 점수를 계산하고 저장합니다.
        """
        logger.info(f"--- daily_universe 테이블 데이터 채우기 시작 ---")
        logger.info(f"전체 채우기 기간: {self.start_fill_date} ~ {self.end_fill_date}")
        
        try:
            # 시장 캘린더 데이터 로드
            market_calendar_df = self.db_manager.fetch_market_calendar(self.start_fill_date, self.end_fill_date)
            
            if market_calendar_df.empty:
                logger.error("시장 캘린더 데이터를 가져올 수 없습니다. daily_universe 채우기를 중단합니다.")
                return

            # 영업일만 필터링하고 날짜를 리스트로 변환하여 정렬
            trading_dates = market_calendar_df[market_calendar_df['is_holiday'] == 0]['date'].dt.date.tolist()
            trading_dates.sort() # 날짜가 오름차순으로 정렬되도록 보장

            if not trading_dates:
                logger.warning("지정된 기간 내 영업일이 없습니다. daily_universe 채우기를 건너뜁니다.")
                return

            logger.info(f"총 {len(trading_dates)}개의 영업일에 대해 daily_universe를 채웁니다.")
            # SetupManager 인스턴스는 각 날짜별로 새로 생성하여 독립적인 상태를 유지
            manager = SetupManager(setup_parameters=self.setup_parameters, db_manager=self.db_manager)

            # 영업일을 순회하며 SetupManager 실행
            for current_date in trading_dates:
                # current_date가 전체 채우기 기간 내에 있는지 다시 확인 (필요에 따라)
                if current_date < self.start_fill_date or current_date > self.end_fill_date:
                    continue # 범위를 벗어나는 날짜는 건너뜀 (정렬되어 있으므로 처음/끝에서만 발생)

                iteration_to_date = current_date
                # 각 날짜의 from_date는 해당 날짜로부터 30일 이전으로 설정
                iteration_from_date = current_date - timedelta(days=30)
                
                logger.info(f"\n======== SetupManager 실행: 데이터 기준 날짜 {iteration_to_date} (영업일) =========================")
                logger.info(f"점수 계산 기간: {iteration_from_date} ~ {iteration_to_date}")

                try:
                    manager.run_all_processes(from_date=iteration_from_date, to_date=iteration_to_date)
                    logger.info(f"날짜 {iteration_to_date}의 daily_universe 데이터 처리가 완료되었습니다.")
                except Exception as e:
                    logger.error(f"날짜 {iteration_to_date} SetupManager 실행 중 오류 발생: {e}", exc_info=True)
                
                # 과도한 요청 방지를 위해 잠시 대기 (필요에 따라 조절)
                time.sleep(1) # 각 일자 처리 후 1초 대기

        except Exception as e:
            logger.critical(f"daily_universe 테이블 채우기 중 치명적인 오류 발생: {e}", exc_info=True)
        finally:
            if self.db_manager: # db_manager 인스턴스가 성공적으로 생성되었다면 연결 닫기
                self.db_manager.close()
            logger.info("--- daily_universe 테이블 데이터 채우기 완료 ---")

    def run(self):
        """
        전체 프로세스를 실행합니다 (일봉 데이터 사전 캐싱 -> daily_universe 데이터 채우기).
        """
        self.prepare_theme_daily_price()
        self.run_daily_universe_filling_process()


if __name__ == "__main__":
    # --- 전체 데이터 채우기 기간 설정 (main 함수 실행 시 기준) ---
    start_fill_date_main = date(2025, 7, 2) # 예시 시작 날짜 (사용자 정의 가능)
    end_fill_date_main = datetime.today().date() # 오늘 날짜

    # SetupManager에 적용할 고정 파라미터 설정 (main 함수에서 전달)
    custom_setup_params = {
        'weight_price_trend': 50,
        'weight_trading_volume': 20,
        'weight_volatility': 15,
        'weight_theme_mention': 15,
        'ma_window_short': 5,
        'ma_window_long': 20,
        'volume_recent_days': 10,
        'atr_window': 14,
        'daily_range_ratio_window': 7
    }

    # DailyUniverseFiller 인스턴스 생성 및 실행
    filler = DailyUniverseFiller(
        start_date=start_fill_date_main,
        end_date=end_fill_date_main,
        setup_parameters=custom_setup_params
    )
    filler.run()