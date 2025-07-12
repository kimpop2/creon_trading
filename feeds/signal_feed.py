# feeds/signal_feed.py

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

# --- Creon 실시간 이벤트 수신 클래스 (News_공시수신.py 및 ex_특징주포착...py에서 통합) ---
class CpEvent:
    """
    PLUS로부터 실시간 이벤트(공시, 특징주 등)를 수신 받아 처리하는 기본 클래스.
    이 클래스는 win32com.client.WithEvents에 의해 동적으로 사용됩니다.
    """
    def set_params(self, client: Any, name: str, caller: Any):
        self.client = client  # CP 실시간 통신 object
        self.name = name      # 서비스가 다른 이벤트를 구분하기 위한 이름 (예: 'marketwatch')
        self.caller = caller  # 콜백을 위해 SignalFeed 인스턴스를 보관

        # 특징주 포착 신호 카테고리 맵 (ex_특징주포착...py에서 가져옴)
        self.diccode = {
            10: '외국계증권사창구첫매수', 11: '외국계증권사창구첫매도', 12: '외국인순매수', 13: '외국인순매도',
            14: '기관순매수', 15: '기관순매도', 16: '개인순매수', 17: '개인순매도',
            21: '전일대비등락율상위', 22: '전일대비등락율하위', 23: '거래량급증', 24: '거래량급감',
            25: '거래대금급증', 26: '거래대금급감', 27: '상한가', 28: '하한가',
            29: '신고가', 30: '신저가', 31: '상한가근접', 32: '하한가근접',
            33: '상승반전', 34: '하락반전', 35: '골든크로스', 36: '데드크로스',
            37: '이동평균선정배열', 38: '이동평균선역배열', 39: 'MACD골든크로스', 40: 'MACD데드크로스',
            41: 'RSI과매도', 42: 'RSI과매수', 43: '볼린저밴드상단돌파', 44: '볼린저밴드하단돌파',
            45: '스토캐스틱슬로우%K골든크로스', 46: '스토캐스틱슬로우%K데드크로스',
            51: '단기이평선돌파', 52: '중기이평선돌파', 53: '장기이평선돌파',
            54: '단기이평선이탈', 55: '중기이평선이탈', 56: '장기이평선이탈',
            57: '거래량회전율급증', 58: '외국인매수강화', 59: '기관매수강화',
            60: '외국인매도강화', 61: '기관매도강화', 62: '개인매수강화', 63: '개인매도강화',
            70: '주요공시', 71: '투자경고종목지정', 72: '투자주의종목지정', 73: '단기과열종목지정',
            74: '거래정지', 75: '관리종목지정', 76: '상장폐지', 77: '액면분할', 78: '액면병합',
            79: '무상증자', 80: '유상증자', 81: '배당락', 82: '권리락',
            83: '자사주취득', 84: '자사주처분', 85: '유상감자', 86: '무상감자',
            87: '합병', 88: '분할', 89: '주식소각', 90: '공개매수',
            91: '매출액영업이익변동', 92: '최대주주변경', 93: '감사보고서제출', 94: '반기보고서제출',
            95: '사업보고서제출', 96: '주요계약체결', 97: '계열회사편입', 98: '계열회사제외',
            99: '투자주의환기종목', 100: '투자경고해제', 101: '단기과열해제',
            # 기타 필요한 코드 추가
        }
        # dic_signal = {
        #         10: {'signal_name': '외국계증권사창구첫매수', 'signal_score': 7},   # 긍정적 시그널, 초기 매수세
        #         11: {'signal_name': '외국계증권사창구첫매도', 'signal_score': 3},   # 부정적 시그널, 초기 매도세
        #         12: {'signal_name': '외국인순매수', 'signal_score': 8},       # 강한 긍정적 시그널, 지속적인 매수
        #         13: {'signal_name': '외국인순매도', 'signal_score': 2},       # 강한 부정적 시그널, 지속적인 매도
        #         21: {'signal_name': '전일거래량갱신', 'signal_score': 5},       # 중립적 시그널, 관심 증가 가능성
        #         22: {'signal_name': '최근5일거래량최고갱신', 'signal_score': 7},   # 긍정적 시그널, 활발한 거래
        #         23: {'signal_name': '최근5일매물대돌파', 'signal_score': 8},   # 긍정적 시그널, 저항선 돌파 가능성
        #         24: {'signal_name': '최근60일매물대돌파', 'signal_score': 9},  # 매우 긍정적 시그널, 강력한 돌파 가능성
        #         28: {'signal_name': '최근5일첫상한가', 'signal_score': 9},     # 매우 긍정적 시그널, 강한 상승 추세 시작
        #         29: {'signal_name': '최근5일신고가갱신', 'signal_score': 8},     # 긍정적 시그널, 상승 추세 지속
        #         30: {'signal_name': '최근5일신저가갱신', 'signal_score': 2},     # 부정적 시그널, 하락 추세 지속
        #         31: {'signal_name': '상한가직전', 'signal_score': 9},       # 매우 긍정적 시그널, 강한 매수세
        #         32: {'signal_name': '하한가직전', 'signal_score': 1},       # 매우 부정적 시그널, 강한 매도세
        #         41: {'signal_name': '주가 5MA 상향돌파', 'signal_score': 6},    # 단기 긍정적 시그널, 추세 전환 가능성
        #         42: {'signal_name': '주가 5MA 하향돌파', 'signal_score': 4},    # 단기 부정적 시그널, 추세 전환 가능성
        #         43: {'signal_name': '거래량 5MA 상향돌파', 'signal_score': 6},    # 단기 긍정적 시그널, 관심 증가
        #         44: {'signal_name': '주가데드크로스(5MA < 20MA)', 'signal_score': 3}, # 부정적 시그널, 단기 하락 추세
        #         45: {'signal_name': '주가골든크로스(5MA > 20MA)', 'signal_score': 7}, # 긍정적 시그널, 단기 상승 추세
        #         46: {'signal_name': 'MACD 매수-Signal(9) 상향돌파', 'signal_score': 7}, # 긍정적 시그널, 매수 추세 시작
        #         47: {'signal_name': 'MACD 매도-Signal(9) 하향돌파', 'signal_score': 3}, # 부정적 시그널, 매도 추세 시작
        #         48: {'signal_name': 'CCI 매수-기준선(-100) 상향돌파', 'signal_score': 6}, # 긍정적 시그널, 과매도 탈출
        #         49: {'signal_name': 'CCI 매도-기준선(100) 하향돌파', 'signal_score': 4},  # 부정적 시그널, 과매수 진입
        #         50: {'signal_name': 'Stochastic(10,5,5)매수- 기준선상향돌파', 'signal_score': 6}, # 긍정적 시그널, 과매도 탈출
        #         51: {'signal_name': 'Stochastic(10,5,5)매도- 기준선하향돌파', 'signal_score': 4},  # 부정적 시그널, 과매수 진입
        #         52: {'signal_name': 'Stochastic(10,5,5)매수- %K%D 교차', 'signal_score': 7}, # 긍정적 시그널, 매수 신호 강화
        #         53: {'signal_name': 'Stochastic(10,5,5)매도- %K%D 교차', 'signal_score': 3}, # 부정적 시그널, 매도 신호 강화
        #         54: {'signal_name': 'Sonar 매수-Signal(9) 상향돌파', 'signal_score': 6},    # 긍정적 시그널, 추세 전환 가능성
        #         55: {'signal_name': 'Sonar 매도-Signal(9) 하향돌파', 'signal_score': 4},    # 부정적 시그널, 추세 전환 가능성
        #         56: {'signal_name': 'Momentum 매수-기준선(100) 상향돌파', 'signal_score': 7}, # 긍정적 시그널, 상승 모멘텀 강화
        #         57: {'signal_name': 'Momentum 매도-기준선(100) 하향돌파', 'signal_score': 3}, # 부정적 시그널, 하락 모멘텀 강화
        #         58: {'signal_name': 'RSI(14) 매수-Signal(9) 상향돌파', 'signal_score': 6},   # 긍정적 시그널, 매수세 강화
        #         59: {'signal_name': 'RSI(14) 매도-Signal(9) 하향돌파', 'signal_score': 4},   # 부정적 시그널, 매도세 강화
        #         60: {'signal_name': 'Volume Oscillator 매수-Signal(9) 상향돌파', 'signal_score': 6}, # 긍정적 시그널, 거래량 증가 추세
        #         61: {'signal_name': 'Volume Oscillator 매도-Signal(9) 하향돌파', 'signal_score': 4}, # 부정적 시그널, 거래량 감소 추세
        #         62: {'signal_name': 'Price roc 매수-Signal(9) 상향돌파', 'signal_score': 6},  # 긍정적 시그널, 가격 상승 가속화
        #         63: {'signal_name': 'Price roc 매도-Signal(9) 하향돌파', 'signal_score': 4},  # 부정적 시그널, 가격 하락 가속화
        #         64: {'signal_name': '일목균형표매수-전환선 > 기준선상향교차', 'signal_score': 7}, # 긍정적 시그널, 추세 전환 및 상승 지속
        #         65: {'signal_name': '일목균형표매도-전환선 < 기준선하향교차', 'signal_score': 3}, # 부정적 시그널, 추세 전환 및 하락 지속
        #         66: {'signal_name': '일목균형표매수-주가가선행스팬상향돌파', 'signal_score': 8}, # 강력한 긍정적 시그널, 추세 상승 강화
        #         67: {'signal_name': '일목균형표매도-주가가선행스팬하향돌파', 'signal_score': 2}, # 강력한 부정적 시그널, 추세 하락 강화
        #         68: {'signal_name': '삼선전환도-양전환', 'signal_score': 7},       # 긍정적 시그널, 추세 전환 가능성
        #         69: {'signal_name': '삼선전환도-음전환', 'signal_score': 3},       # 부정적 시그널, 추세 전환 가능성
        #         70: {'signal_name': '캔들패턴-상승반전형', 'signal_score': 7},     # 긍정적 시그널, 하락 추세 종료 및 상승 시작 가능성
        #         71: {'signal_name': '캔들패턴-하락반전형', 'signal_score': 3},     # 부정적 시그널, 상승 추세 종료 및 하락 시작 가능성
        #         81: {'signal_name': '단기급락후 5MA 상향돌파', 'signal_score': 8},  # 긍정적 시그널, 반등 가능성 높음
        #         82: {'signal_name': '주가이동평균밀집-5%이내', 'signal_score': 6},  # 중립적이나 방향성 결정 임박, 변동성 확대 가능성
        #         83: {'signal_name': '눌림목재상승-20MA 지지', 'signal_score': 8}   # 긍정적 시그널, 안정적인 지지 후 상승 기대
        #     }

    def OnReceived(self):
        """
        PLUS로부터 실제로 이벤트가 수신될 때 호출되는 메서드.
        공시, 특징주(기술적/수급 신호) 데이터를 파싱하여 SignalFeed의 버퍼에 추가합니다.
        """
        if self.name == 'marketwatch':
            current_time = datetime.now()
            
            # CpMarketWatch (8092)는 뉴스/공시/특징주를 모두 처리
            # GetHeaderValue(0): 종목코드, GetHeaderValue(1): 종목명 (이름은 GetDataValue로 가져옴)
            # GetDataValue(0,i): 시간, GetDataValue(1,i): 업데이트 구분, GetDataValue(2,i): 카테고리
            # GetDataValue(3,i): 특이사항 (뉴스/공시 제목/신호 내용)

            # 수신된 데이터 개수
            cnt = self.client.GetHeaderValue(2) 
            
            for i in range(cnt):
                code = self.client.GetDataValue(0, i)
                name = g_objCodeMgr.CodeToName(code) if g_objCodeMgr else "Unknown"
                time_val = self.client.GetDataValue(1, i) # 시간 (HHMM)
                category_code = self.client.GetDataValue(2, i) # 신호 카테고리 코드
                special_note = self.client.GetDataValue(3, i) # 특이사항 (뉴스/공시 제목/신호 내용)

                # 시간 포맷팅
                h, m = divmod(time_val, 100)
                event_datetime = current_time.replace(hour=h, minute=m, second=0, microsecond=0)

                signal_type_desc = self.diccode.get(category_code, f"Unknown Signal ({category_code})")
                
                # 공시/뉴스 데이터는 news_raw 테이블에 저장
                # category_code 70번대 이상은 주로 공시/관리종목 관련
                if 70 <= category_code <= 101: 
                    news_item = {
                        'source': 'Creon Announcement/MarketWatch',
                        'datetime': event_datetime,
                        'title': f"[{signal_type_desc}] {special_note}",
                        'content': special_note,
                        'url': None,
                        'related_stocks': [code] if code and code != '000000' else []
                    }
                    self.caller.news_data_buffer.append(news_item)
                    logger.info(f"[CpEvent] Received announcement/news: {news_item['title'][:50]}...")
                else: # 그 외는 매매 신호 (기술적, 수급 등)
                    signal_item = {
                        'signal_date': event_datetime.date(),
                        'stock_code': code,
                        'stock_name': name,
                        'signal_type': 'BUY' if category_code in [10, 12, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 51, 52, 53, 57, 58, 59] else 'SELL', # 매수/매도 임시 구분 (더 정교한 로직 필요)
                        'strategy_name': 'CreonMarketWatch',
                        'target_price': None, # 실시간 신호는 목표가 없을 수 있음
                        'target_quantity': None,
                        'is_executed': False,
                        'executed_order_id': None,
                        'created_at': current_time,
                        'updated_at': current_time,
                        'signal_description': signal_type_desc + ": " + special_note
                    }
                    self.caller.signal_data_buffer.append(signal_item)
                    logger.info(f"[CpEvent] Received market watch signal: {signal_item['signal_description']} for {code}")

class CpPublish:
    """
    Creon Plus 실시간 데이터를 구독하고 해지하는 기본 클래스.
    """
    def __init__(self, name: str, serviceID: str):
        self.name = name
        self.obj = win32com.client.Dispatch(serviceID)
        self.bIsSB = False # 구독 상태 플래그

    def Subscribe(self, var: str, caller: Any):
        """
        실시간 데이터 구독을 시작합니다.
        :param var: 구독할 입력 값 (예: 종목 코드, 빈 문자열로 전체 특징주)
        :param caller: OnReceived 콜백을 받을 객체 (SignalFeed 인스턴스)
        """
        if self.bIsSB:
            self.Unsubscribe() # 이미 구독 중이면 해지 후 재구독

        if len(var) > 0:
            self.obj.SetInputValue(0, var) # 입력 값 설정 (0: 전체, 1: 코스피, 2: 코스닥 등)

        # win32com.client.WithEvents를 사용하여 이벤트 핸들러 연결
        handler = win32com.client.WithEvents(self.obj, CpEvent)
        handler.set_params(self.obj, self.name, caller) # CpEvent에 파라미터 설정
        self.obj.Subscribe() # 구독 시작
        self.bIsSB = True
        logger.debug(f"Subscribed to {self.name} with var: '{var}'")

    def Unsubscribe(self):
        """
        실시간 데이터 구독을 해지합니다.
        """
        if self.bIsSB:
            self.obj.Unsubscribe()
            self.bIsSB = False
            logger.debug(f"Unsubscribed from {self.name}")

class CpRpMarketWatch:
    """
    CpSysDib.CpMarketWatch (특징주 포착 통신) 서비스를 위한 Request 클래스.
    이 클래스는 실시간 특징주/공시 데이터를 요청하고 구독을 관리합니다.
    """
    def __init__(self):
        # CpSysDib.CpMarketWatch 객체는 뉴스/공시/특징주를 통합적으로 처리
        self.objMarketWatch = win32com.client.Dispatch('CpSysDib.CpMarketWatch')
        # CpPB8092news는 뉴스 수신용 (여기서는 CpMarketWatch를 통해 통합 처리되므로 직접 사용 안 함)
        # CpPB8092news는 NewsFeed에서 사용
        self.objpbMarket = CpPublish('marketwatch', 'Dscbo1.CpSvr8092S') # 8092 서비스 ID

    def Request(self, code: str, caller: Any) -> bool:
        """
        실시간 특징주/공시 데이터 구독을 요청합니다.
        :param code: 구독할 종목 코드 (빈 문자열이면 전체 시장)
        :param caller: OnReceived 콜백을 받을 객체 (SignalFeed 인스턴스)
        """
        # 기존 구독 해지 (필요 시)
        self.objpbMarket.Unsubscribe()
        
        # 새로운 구독 시작
        # SetInputValue(0, '0') : 전체 시장 (코스피+코스닥)
        # SetInputValue(1, '1') : 특징주 포착 (CpMarketWatch의 서비스 타입)
        # SetInputValue(2, '0') : 전체 종목 (특정 종목 필터링은 code 인자로)
        # SetInputValue(3, '1') : 실시간 데이터 요청
        # SetInputValue(4, '0') : 전체 카테고리 (특정 카테고리 필터링은 code 인자로)
        
        # CpMarketWatch는 SetInputValue를 통해 요청 조건을 설정
        # News_공시수신.py의 CpRpMarketWatch.Request는 objpbMarket.Subscribe(code, caller)를 호출
        # 즉, CpPublish의 Subscribe를 통해 실제 COM 객체 구독이 이루어짐
        # 따라서 여기서는 objpbMarket의 Subscribe를 호출
        try:
            # CpMarketWatch 서비스의 SetInputValue는 다양한 필터링 옵션을 가짐
            # 여기서는 'code' 인자를 통해 전체 시장 또는 특정 종목을 구독하도록 함
            # (CpMarketWatch는 종목 코드 대신 필터링 조건을 SetInputValue로 받음)
            # 샘플 코드에서는 code를 SetInputValue(0, var)로 전달
            self.objpbMarket.Subscribe(code, caller)
            logger.info(f"Requested real-time market watch for code '{code}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to request market watch subscription: {e}", exc_info=True)
            return False

# --- SignalFeed 클래스 (기존 signal_feed.py) ---
class SignalFeed(BaseFeed):
    """
    Creon API를 통해 실시간 공시 및 특징주(MarketWatch) 신호를 수집하고 DB에 저장하는 Feed 모듈.
    news_raw (공시), daily_signals (특징주 중 차트/기술적 신호) 테이블에 저장합니다.
    """
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        super().__init__("SignalFeed", redis_host, redis_port)
        self.api_client = CreonAPIClient()
        self.db_feed = DBFeed()

        # 공시/특징주 수신 객체 초기화
        self.objMarketWatch = CpRpMarketWatch()
        
        self.news_data_buffer: List[Dict[str, Any]] = []  # 공시/뉴스 임시 저장 버퍼
        self.signal_data_buffer: List[Dict[str, Any]] = [] # 특징주 신호 임시 저장 버퍼

        # CpEvent 콜백을 위한 인스턴스 생성 및 파라미터 설정
        # CpEvent의 caller로 SignalFeed 인스턴스(self)를 전달하여 버퍼에 접근
        self.signal_event_handler = CpEvent()
        self.signal_event_handler.set_params(self.objMarketWatch.objpbMarket.obj, self.objMarketWatch.objpbMarket.name, self)

        logger.info("SignalFeed initialized.")

    def _subscribe_creon_signals(self) -> bool:
        """Creon 실시간 특징주/공시 구독을 시작합니다."""
        if not self.api_client.connected:
            logger.error("Creon API Client is not connected. Cannot subscribe to signals.")
            return False
        
        try:
            # CpRpMarketWatch의 Request 메서드 호출
            # ""는 전체 종목을 의미한다고 가정 (Creon API 문서 확인 필요)
            self.objMarketWatch.Request("", self) 
            logger.info("Subscribed to Creon real-time market watch signals.")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to Creon signals: {e}", exc_info=True)
            return False

    def _process_buffers(self):
        """버퍼에 쌓인 데이터들을 DB에 저장하고 이벤트를 발행합니다."""
        if self.news_data_buffer:
            if self.db_feed.save_news_raw(self.news_data_buffer):
                logger.info(f"Saved {len(self.news_data_buffer)} raw announcement/news records to DB.")
                self.publish_event('signal_feed_events', {
                    'type': 'ANNOUNCEMENT_NEWS_COLLECTED',
                    'count': len(self.news_data_buffer),
                    'timestamp': datetime.now().isoformat()
                })
                self.news_data_buffer.clear()
            else:
                logger.error("Failed to save raw announcement/news to DB.")

        if self.signal_data_buffer:
            # daily_signals 테이블에 저장
            # save_daily_signals 메서드를 DBFeed에 추가하거나, execute_sql을 직접 사용
            # 여기서는 execute_sql을 직접 사용하는 예시를 보여줌 (DBFeed에 save_daily_signals 추가 권장)
            sql = """
            INSERT INTO daily_signals (signal_date, stock_code, stock_name, signal_type, strategy_name,
                                       target_price, target_quantity, is_executed, executed_order_id,
                                       created_at, updated_at, signal_description)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                is_executed=VALUES(is_executed), executed_order_id=VALUES(executed_order_id),
                updated_at=CURRENT_TIMESTAMP, signal_description=VALUES(signal_description)
            """
            data_to_save = []
            for s_item in self.signal_data_buffer:
                data_to_save.append((
                    s_item['signal_date'], s_item['stock_code'], s_item['stock_name'], s_item['signal_type'],
                    s_item['strategy_name'], s_item['target_price'], s_item['target_quantity'],
                    s_item['is_executed'], s_item['executed_order_id'],
                    s_item['created_at'], s_item['updated_at'], s_item['signal_description']
                ))
            
            if self.db_feed.execute_sql(sql, data_to_save) is not None:
                logger.info(f"Saved {len(self.signal_data_buffer)} market watch signals to DB.")
                self.publish_event('signal_feed_events', {
                    'type': 'MARKET_WATCH_SIGNAL_COLLECTED',
                    'count': len(self.signal_data_buffer),
                    'timestamp': datetime.now().isoformat()
                })
                self.signal_data_buffer.clear()
            else:
                logger.error("Failed to save market watch signals to DB.")

    def run(self):
        """
        SignalFeed의 메인 실행 루프.
        Creon API를 통해 실시간 공시 및 특징주 신호를 수신하고 주기적으로 DB에 저장합니다.
        """
        logger.info(f"SignalFeed process started.")
        # Creon Plus 연결 및 관리자 권한 확인
        if not init_creon_plus_check():
            logger.error("Creon Plus is not connected or not running with admin privileges. SignalFeed cannot run.")
            return

        # CreonAPIClient의 연결 상태를 다시 확인 (init_creon_plus_check와는 별개)
        if not self.api_client.connected:
            logger.error("CreonAPIClient is not connected. Attempting to connect...")
            pass # _subscribe_creon_signals에서 다시 확인하므로 여기서는 생략 가능

        if not self._subscribe_creon_signals():
            logger.error("SignalFeed cannot start without Creon signal subscription.")
            return

        while not self.stop_event.is_set():
            try:
                self._process_buffers()
                self._wait_for_stop_signal(interval=5)  # 5초마다 버퍼 처리 시도
            except Exception as e:
                logger.error(f"[{self.feed_name}] Error in run loop: {e}", exc_info=True)
                time.sleep(5)

        logger.info(f"SignalFeed process stopped.")
        # 종료 시 구독 해지
        if hasattr(self.objMarketWatch, 'objpbMarket') and self.objMarketWatch.objpbMarket:
            self.objMarketWatch.objpbMarket.Unsubscribe()
        self.db_feed.close()

# 실행 예시 (별도 프로세스로 실행될 때)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    signal_feed = SignalFeed()
    try:
        signal_feed.run()
    except KeyboardInterrupt:
        logger.info("SignalFeed interrupted by user.")
    finally:
        signal_feed.stop()
