# setup/creon_auto_login.py
import logging
import sys
import os
import time
import win32com.client
from pywinauto import application

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('creon_auto_login')

# config.settings에서 설정 로드 (보안 강화)
try:
    from config.settings import CREON_ID, CREON_PWD, CREON_CERT_PWD
except ImportError:
    logger.critical("보안 오류: config/settings.py 파일에서 CREON 로그인 정보를 찾을 수 없습니다. 스크립트를 종료합니다.")
    sys.exit(1) # 설정 파일이 없으면 즉시 종료

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
            login_cmd = 'C:\\CREON\\STARTER\\coStarter.exe /prj:cp /id:{id} /pwd:{pwd} /pwdcert:{pwdcert} /autostart'.format(
                id=CREON_ID, pwd=CREON_PWD, pwdcert=CREON_CERT_PWD
            )
            app.start(login_cmd)
            logger.info("coStarter.exe 실행. 로그인 대기 중...")

            max_wait_time = 180 # 최대 3분 대기
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

if __name__ == '__main__':
    logger.info("Creon 자동 로그인 스크립트 시작...")
    cp_cybos = connect_creon_auto_login(reconnect=True)
    if cp_cybos and cp_cybos.IsConnect:
        logger.info("Creon 자동 로그인 성공.")
        sys.exit(0) # 성공 시 종료 코드 0
    else:
        logger.error("Creon 자동 로그인 실패.")
        sys.exit(1) # 실패 시 종료 코드 1
