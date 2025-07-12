# feeds/invester_feed.py

import logging
import time
from datetime import datetime, date
import sys
import os
import win32com.client  # Creon API 사용
import ctypes  # 관리자 권한 확인용
from typing import Dict, Any, List, Optional

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from feeds.base_feed import BaseFeed
from feeds.db_feed import DBFeed

# 로거 설정
logger = logging.getLogger(__name__)

# --- Creon API 관련 전역 객체 및 유틸리티 함수 ---
# g_objCodeMgr과 g_objCpStatus는 전역으로 한 번만 Dispatch
g_objCodeMgr = None
g_objCpStatus = None

def init_creon_plus_check() -> bool:
    """
    Creon Plus의 연결 상태 및 관리자 권한 실행 여부를 확인합니다.
    """
    global g_objCodeMgr, g_objCpStatus
    if g_objCodeMgr is None:
        g_objCodeMgr = win32com.client.Dispatch('CpUtil.CpCodeMgr')
    if g_objCpStatus is None:
        g_objCpStatus = win32com.client.Dispatch('CpUtil.CpCybos')

    # 프로세스가 관리자 권한으로 실행 여부
    if not ctypes.windll.shell32.IsUserAnAdmin():
        logger.error('오류: 일반권한으로 실행됨. 관리자 권한으로 실행해 주세요')
        return False

    # 연결 여부 체크
    if g_objCpStatus.IsConnect == 0:
        logger.error("PLUS가 정상적으로 연결되지 않음.")
        return False

    logger.info("Creon Plus 연결 및 관리자 권한 확인 완료.")
    return True

# --- CpInvester 클래스 (투자자_잠정_상위_분.py에서 통합) ---
class CpInvester:
    """
    Cp7210: 종목별 투자자 매매동향(잠정) 데이터를 요청하고 수신하는 클래스.
    """
    def __init__(self, creon_api_client: CreonAPIClient):
        self.creon = creon_api_client  # CreonAPIClient 인스턴스
        self.flag_market = {'전체': '0', '코스피': '1', '코스닥': '2', '업종': '3', '관심종목': '4'}
        self.flag_qty_amt = {'수량': 0, '금액': 1}
        self.flag_invester = {
            '종목': '0', '외국인': '1', '기관계': '2', '보험,기타금융': '3', '투신': '4',
            '은행': '5', '연기금': '6', '국가,지자체': '7', '기타법인': '8'
        }
        
        # CpSysDib.CpSvr7210d COM 객체 Dispatch
        self.objRq = win32com.client.Dispatch("CpSysDib.CpSvr7210d")
        logger.debug("CpInvester: CpSysDib.CpSvr7210d dispatched.")

    def SetInputValue(self, market: str = '전체', qty_amt: str = '수량', invester: str = '외국인'):
        """
        요청할 데이터의 입력 값을 설정합니다.
        :param market: 시장 구분 (전체, 코스피, 코스닥 등)
        :param qty_amt: 수량 또는 금액 기준
        :param invester: 투자자 구분 (외국인, 기관계 등)
        """
        self.objRq.SetInputValue(0, self.flag_market.get(market, '0'))  # '0':전체 '1':코스피 '2':코스닥 '3':업종 '4':관심종목
        self.objRq.SetInputValue(1, self.flag_qty_amt.get(qty_amt, 0))  # 0:수량 1:금액
        self.objRq.SetInputValue(2, self.flag_invester.get(invester, '1')) # '0':종목 '1':외국인 '2':기관계 ...

    def Request(self, ret_data: List[Dict[str, Any]], max_count: int = 200) -> bool:
        """
        투자자 매매동향(잠정) 데이터를 요청하고 ret_data 리스트에 추가합니다.
        :param ret_data: 수집된 데이터를 추가할 리스트
        :param max_count: 요청할 최대 데이터 개수
        :return: 데이터 요청 성공 여부
        """
        self.objRq.BlockRequest()  # 데이터 요청

        status = self.objRq.GetDibStatus()
        msg = self.objRq.GetDibMsg1()
        if status != 0:
            logger.error(f"CpInvester: Request failed: Status={status}, Msg={msg}")
            return False

        # 데이터 수신
        ret_date = self.objRq.GetHeaderValue(0)  # 일자
        ret_time = self.objRq.GetHeaderValue(1)  # 시간
        ret_count = self.objRq.GetHeaderValue(2) # 수신 개수

        logger.debug(f"CpInvester: Received {ret_count} items for {ret_date} {ret_time}. Total collected: {len(ret_data) + ret_count}")

        for i in range(ret_count):
            # Keys are mapped to English DB field names directly for consistency
            item = {
                'date': ret_date,
                'time': ret_time,
                'stock_code': self.objRq.GetDataValue(0, i),
                'stock_name': self.objRq.GetDataValue(1, i),
                'current_price': self.objRq.GetDataValue(2, i),
                'change_from_prev_day': self.objRq.GetDataValue(3, i),
                'change_rate': self.objRq.GetDataValue(4, i),
                'volume_total': self.objRq.GetDataValue(5, i),
                'net_foreign': self.objRq.GetDataValue(6, i),
                'net_institutional': self.objRq.GetDataValue(7, i),
                'net_insurance_etc': self.objRq.GetDataValue(8, i),
                'net_trust': self.objRq.GetDataValue(9, i),
                'net_bank': self.objRq.GetDataValue(10, i),
                'net_pension': self.objRq.GetDataValue(11, i),
                'net_gov_local': self.objRq.GetDataValue(12, i),
                'net_other_corp': self.objRq.GetDataValue(13, i)
            }
            ret_data.append(item)

        # CpSysDib.CpSvr7210d doesn't support continuous querying, so all data is returned in one BlockRequest.
        
        return True

# --- InvesterFeed 클래스 ---
class InvesterFeed(BaseFeed):
    """
    Creon API를 통해 종목별 투자자 매매 동향(잠정) 데이터를 수집하고 DB에 저장하는 Feed 모듈.
    investor_trends 테이블에 데이터를 저장합니다.
    """
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        super().__init__("InvesterFeed", redis_host, redis_port)
        self.api_client = CreonAPIClient()
        self.db_feed = DBFeed()
        
        # 통합된 CpInvester 클래스 사용
        self.cp_invester = CpInvester(self.api_client) 
        self.investor_data_buffer: List[Dict[str, Any]] = [] # 수신된 투자자 동향 임시 저장 버퍼

        logger.info("InvesterFeed initialized.")

    def _collect_and_save_investor_trends(self):
        """
        투자자 매매 동향 데이터를 Creon API에서 가져와 DB에 저장합니다.
        CpInvester의 Request 메서드를 활용합니다.
        """
        logger.info("Collecting investor trends data.")
        ret_data_from_creon = [] # CpInvester.Request가 데이터를 채울 리스트

        # '전체' 시장, '수량' 기준, '외국인' 투자자 기준으로 요청 (예시)
        self.cp_invester.SetInputValue(market='전체', qty_amt='수량', invester='외국인')
        
        # Request 메서드 호출 (데이터는 ret_data_from_creon에 채워짐)
        success = self.cp_invester.Request(ret_data_from_creon, max_count=200) # 최대 200개 종목 데이터 요청 (예시)

        if success and ret_data_from_creon:
            data_to_save = []
            current_date = datetime.now().date()
            current_time = datetime.now().time()
            for item in ret_data_from_creon:
                # CpInvester에서 반환된 item (이제 영어 키 사용) 구조에 맞게 매핑
                data_to_save.append({
                    'stock_code': item['stock_code'],
                    'date': current_date, # 당일 날짜 (DB에 저장 시 사용)
                    'time': current_time, # 현재 시간 (DB에 저장 시 사용)
                    'current_price': item['current_price'],
                    'volume_total': item['volume_total'],
                    'net_foreign': item['net_foreign'],
                    'net_institutional': item['net_institutional'],
                    'net_insurance_etc': item['net_insurance_etc'],
                    'net_trust': item['net_trust'],
                    'net_bank': item['net_bank'],
                    'net_pension': item['net_pension'],
                    'net_gov_local': item['net_gov_local'],
                    'net_other_corp': item['net_other_corp'],
                    'data_type': '수량' # 요청 시 설정한 기준
                })
            
            if self.db_feed.save_investor_trends(data_to_save):
                logger.info(f"Saved {len(data_to_save)} investor trends records to DB.")
                self.publish_event('invester_feed_events', {
                    'type': 'INVESTOR_TRENDS_COLLECTED',
                    'count': len(data_to_save),
                    'timestamp': datetime.now().isoformat()
                })
            else:
                logger.error("Failed to save investor trends to DB.")
        else:
            logger.warning("No investor trends data collected or request failed.")

    def run(self):
        """
        InvesterFeed의 메인 실행 루프.
        주기적으로 투자자 매매 동향 데이터를 수집하고 DB에 저장합니다.
        """
        logger.info(f"InvesterFeed process started.")
        # Creon Plus 연결 및 관리자 권한 확인
        if not init_creon_plus_check():
            logger.error("Creon Plus is not connected or not running with admin privileges. InvesterFeed cannot run.")
            return

        # CreonAPIClient의 연결 상태를 다시 확인 (init_creon_plus_check와는 별개)
        if not self.api_client.connected:
            logger.error("CreonAPIClient is not connected. Attempting to connect...")
            pass

        while not self.stop_event.is_set():
            try:
                self._collect_and_save_investor_trends()
                self._wait_for_stop_signal(interval=300) # 5분마다 실행 (예시)
            except Exception as e:
                logger.error(f"[{self.feed_name}] Error in run loop: {e}", exc_info=True)
                time.sleep(5)

        logger.info(f"InvesterFeed process stopped.")
        self.db_feed.close()

# 실행 예시 (별도 프로세스로 실행될 때)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    invester_feed = InvesterFeed()
    try:
        invester_feed.run()
    except KeyboardInterrupt:
        logger.info("InvesterFeed interrupted by user.")
    finally:
        invester_feed.stop()