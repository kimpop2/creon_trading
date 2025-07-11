# setup/startup.py
import logging
import sys
import os
from datetime import date, datetime, timedelta
import win32com.client
from pywinauto import application
import time

# 프로젝트 루트 경로를 sys.path에 추가 (다른 모듈 임포트를 위함)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# config.settings에서 설정 로드
from config.settings import (
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    CREON_ID, CREON_PWD, CREON_CERT_PWD, # 추가된 크레온 로그인 정보
    LOG_LEVEL, LOG_FILE_STARTUP # LOG_FILE_STARTUP 추가
)

# manager 모듈 임포트 (예: TradingManager, DBManager 등)
from manager.db_manager import DBManager
from api.creon_api import CreonAPIClient
from util.notifier import Notifier
from manager.trading_manager import TradingManager

# 로거 설정
def setup_startup_logging():
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    # 기본 로거를 가져옴
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 기존 핸들러 제거 (basicConfig가 여러 번 호출될 경우 중복을 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 파일 핸들러 (UTF-8)
    file_handler = logging.FileHandler(LOG_FILE_STARTUP, encoding='utf-8')
    #file_handler = logging.FileHandler(LOG_FILE_STARTUP, encoding='cp949')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)   
    
    # 스트림 핸들러 (UTF-8)
    # sys.stdout의 인코딩을 시스템과 일치시키기 위해 명시적으로 'utf-8' 지정
    # 그러나 Windows 콘솔이 UTF-8을 제대로 지원하지 않으면 여전히 깨질 수 있음
    # 그럴 경우, 배치 파일에서 chcp 65001 설정이 필수
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.encoding = 'utf-8' # ⭐ 여기를 utf-8로 명시적으로 설정

    root_logger.addHandler(stream_handler)
    logging.getLogger('urllib3').setLevel(logging.WARNING) # 불필요한 로깅 줄이기

setup_startup_logging() # 로깅 설정 호출
logger = logging.getLogger(__name__)


def connect_creon_auto_login(reconnect=True):
    """
    크레온 HTS를 자동 로그인하고 CpCybos COM 객체를 반환합니다.
    재연결(reconnect=True) 시 기존 크레온 프로세스를 강제 종료합니다.
    """
    if reconnect:
        logger.info("기존 Creon 프로세스 강제 종료 시도...")
        try:
            os.system('taskkill /IM ncStarter* /F /T')
            os.system('taskkill /IM CpStart* /F /T')
            os.system('taskkill /IM DibServer* /F /T')
            os.system('wmic process where "name like \'%%coStarter.exe%%\'" call terminate')
            os.system('wmic process where "name like \'%%CpStart%%\'" call terminate')
            os.system('wmic process where "name like \'%%DibServer%%\'" call terminate')
            time.sleep(2) # 프로세스 종료 대기
        except Exception as e:
            logger.warning(f"Creon 프로세스 종료 중 오류 발생 (무시 가능): {e}")

    CpCybos = win32com.client.Dispatch("CpUtil.CpCybos")

    if CpCybos.IsConnect:
        logger.info('Creon Plus가 이미 연결되어 있습니다.')
    else:
        logger.info('Creon Plus 연결 시도 중 (자동 로그인)...')
        try:
            app = application.Application()
            # settings.py에서 로그인 정보 로드
            login_cmd = 'C:\\CREON\\STARTER\\coStarter.exe /prj:cp /id:{id} /pwd:{pwd} /pwdcert:{pwdcert} /autostart'.format(
                id=CREON_ID, pwd=CREON_PWD, pwdcert=CREON_CERT_PWD
            )
            app.start(login_cmd)
            logger.info("coStarter.exe 실행. 로그인 대기 중...")

            # 연결 될때까지 무한루프
            max_wait_time = 180 # 최대 60초 대기
            start_time = time.time()
            while True:
                if CpCybos.IsConnect:
                    logger.info('Creon Plus 연결 성공.')
                    break
                if time.time() - start_time > max_wait_time:
                    logger.error(f"Creon Plus 연결 시간 초과 ({max_wait_time}초). 로그인 실패.")
                    return None
                time.sleep(1)
        except Exception as e:
            logger.error(f"Creon 자동 로그인 중 오류 발생: {e}", exc_info=True)
            return None
    return CpCybos


def run_startup_tasks():
    logger.info("========================================")
    logger.info("      자동매매 시스템 시작 전 작업 시작     ")
    logger.info("========================================")

    notifier = Notifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    notifier.send_message("⚙️ 자동매매 시스템 시작 전 준비 작업이 시작됩니다.")

    creon_api = None
    db_manager = None
    try:
        # 1. Creon HTS 자동 로그인 및 연결
        logger.info("Creon HTS 자동 로그인 시도...")
        cp_cybos_obj = connect_creon_auto_login(reconnect=True) # 재연결 시도
        if cp_cybos_obj is None or not cp_cybos_obj.IsConnect:
            logger.critical("Creon HTS 자동 로그인 및 연결 실패. 시작 전 작업 중단.")
            notifier.send_message("❌ Creon HTS 자동 로그인 및 연결 실패. 시작 전 작업 중단.")
            return False
        logger.info("Creon HTS 자동 로그인 및 연결 성공.")

        # CreonAPIClient는 내부적으로 CpCybos 객체를 사용하여 연결 상태를 확인
        creon_api = CreonAPIClient() # 새로운 CreonAPIClient 인스턴스 생성
        if not creon_api._check_creon_status(): # CreonAPIClient 내부에서 연결 확인
            logger.critical("CreonAPIClient가 Creon Plus에 연결되지 않았습니다. 시작 전 작업 중단.")
            notifier.send_message("❌ CreonAPIClient 연결 실패. 시작 전 작업 중단.")
            return False
        logger.info("CreonAPIClient 초기화 및 연결 확인 완료.")


        # 2. DBManager 초기화
        db_manager = DBManager()
        if not db_manager.get_db_connection():
            logger.critical("DB 연결 실패. 데이터 수집을 진행할 수 없습니다.")
            notifier.send_message("❌ DB 연결 실패. 시작 전 작업 중단.")
            return False
        logger.info("DB 연결 성공.")

        # TradingManager 초기화 (데이터 수집 기능 활용)
        trading_manager = TradingManager(creon_api, db_manager)
        logger.info("TradingManager 초기화 완료.")

        # 3. 일봉 데이터 수집 (예시: 모든 종목에 대해 1년치 데이터 업데이트)
        # logger.info("최신 일봉 데이터 수집 시작 (모든 종목)...")
        # all_stock_codes = trading_manager.get_all_stock_list() # (code, name) 튜플 리스트
        # today = datetime.now().date()
        # one_year_ago = today - timedelta(days=365)

        # for stock_code, stock_name in all_stock_codes:
        #     try:
        #         # TradingManager의 fetch_daily_ohlcv가 내부적으로 DB 조회 후 없으면 API 조회 및 저장 처리
        #         df_daily = trading_manager.fetch_daily_ohlcv(stock_code, one_year_ago, today)
        #         if df_daily.empty:
        #             logger.warning(f"종목 {stock_name}({stock_code})의 일봉 데이터 수집 실패 또는 데이터 없음.")
        #         else:
        #             logger.debug(f"종목 {stock_name}({stock_code}) 일봉 데이터 {len(df_daily)}건 수집/확인 완료.")
        #     except Exception as e:
        #         logger.error(f"종목 {stock_name}({stock_code}) 일봉 데이터 수집 중 오류 발생: {e}")
        # logger.info("일봉 데이터 수집 완료.")

        # 4. 유니버스 생성/업데이트 (daily_universe 테이블)
        logger.info("유니버스(매매 대상 종목군) 생성/업데이트 시작...")
        # TradingManager에 유니버스 업데이트 로직이 있다면 호출
        # 예: trading_manager.update_daily_universe(today)
        # 현재 TradingManager에는 get_universe_stocks만 있으므로,
        # 여기서는 유니버스 생성 로직이 별도로 필요하거나, get_universe_stocks가 내부적으로 업데이트를 포함해야 함.
        # 이 부분은 사용자의 기존 'util.make_up_universe' 모듈의 기능을 통합해야 합니다.
        # 임시로 메시지만 출력
        logger.info("유니버스 생성/업데이트 로직 실행 (구현 필요).")
        notifier.send_message("✨ 유니버스 종목 선정 및 업데이트 완료 (구현에 따라).")

        # 5. 뉴스 및 텔레그램 실시간 피드 시작 (이 부분은 직접 구현 필요)
        logger.info("뉴스 및 텔레그램 실시간 피드 시작 (실제 로직 필요)...")
        # 예: news_parser.start_realtime_feed()
        # 예: telegram_receiver.start_listening()
        notifier.send_message("📢 뉴스 및 실시간 피드 모니터링 시작 (기능 구현 필요).")

        logger.info("========================================")
        logger.info("      자동매매 시스템 시작 전 작업 완료     ")
        logger.info("========================================")
        notifier.send_message("✅ 자동매매 시스템 시작 전 준비 작업이 성공적으로 완료되었습니다.")
        return True

    except Exception as e:
        logger.critical(f"시작 전 작업 중 치명적인 오류 발생: {e}", exc_info=True)
        notifier.send_message(f"🚨 자동매매 시스템 시작 전 작업 중 오류 발생: {e}")
        return False
    finally:
        if creon_api:
            creon_api.cleanup()
        if db_manager:
            db_manager.close() # db_manager의 close 메서드 호출

if __name__ == "__main__":
    run_startup_tasks()

