# api_client/creon_api.py

import win32com.client
import ctypes
import time
import logging
import pandas as pd
#import datetime # datetime 모듈 전체를 임포트하여 datetime.timedelta 사용 가능
import re
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Callable

API_REQUEST_INTERVAL = 0.2

# 로거 설정 (기존 설정 유지)
logger = logging.getLogger(__name__)

# 실시간 이벤트 처리를 위한 클래스들
class CpEvent:
    """실시간 이벤트 수신 클래스 (예제 기반)"""
    def set_params(self, client, name, parent):
        self.client = client  # CP 실시간 통신 object
        self.name = name  # 서비스가 다른 이벤트를 구분하기 위한 이름
        self.parent = parent  # callback 을 위해 보관
        self.concdic = {"1": "체결", "2": "확인", "3": "거부", "4": "접수"}

    def OnReceived(self):
        """PLUS로부터 실제로 이벤트(체결/주문 응답/시세 이벤트 등)를 수신 받아 처리하는 함수"""
        if self.name == "td0314":
            """주문 Request에 대한 응답 처리"""
            logger.info("[CpEvent]주문응답")
            self.parent.OrderReply()
            return

        elif self.name == "conclusion":
            """주문 체결 실시간 업데이트"""
            # 주문 체결 실시간 업데이트
            i3 = self.client.GetHeaderValue(3)     # 체결 수량
            i4 = self.client.GetHeaderValue(4)     # 가격
            i5 = self.client.GetHeaderValue(5)     # 주문번호
            i6 = self.client.GetHeaderValue(6)     # 원주문번호
            i9 = self.client.GetHeaderValue(9)     # 종목코드
            i12 = self.client.GetHeaderValue(12)   # 매수/매도 구분 1 매도 2매수
            i14 = self.client.GetHeaderValue(14)   # 체결 플래그 1 체결 2 확인...
            i16 = self.client.GetHeaderValue(16)   # 정정/취소 구분코드 (1 정상, 2 정정 3 취소)

            logger.info(f"[CpEvent]{self.concdic.get(i14)}, 수량 {i3}, 가격 {i4}, 주문번호 {i5}, 원주문 {i6}, 코드 {i9}")

            if i14 == "1":  # 체결
                """체결"""
                if not (i5 in self.parent.diOrderList):  # 미체결 리스트에 없다.
                    logger.warning(f"[CpEvent]주문번호 찾기 실패 {i5}")
                    return

                item = self.parent.diOrderList[i5]  # 주문번호로 미체결 내용 가져옴
                if (item.amount - i3 > 0):  # 일부 체결 (미체결수량 - 체결수량 : 미체결 남음)
                    # 기존 데이터 업데이트
                    item.amount -= i3           # 미체결 수량을 체결수량 만큼 감소
                    item.modAvali = item.amount # 미체결 수량
                    item.ContAmount += i3       # 주문수량 = 미체결 수량 +체결 수량
                else:                   # 전체 체결 시 미체결 번호 삭제
                    self.parent.deleteOrderNum(i5)

                logger.info(f"[CpEvent]미체결 개수 {len(self.parent.orderList)}")

            elif i14 == "2":  # 확인
                """확인"""
                # 원주문 번호로 찾기
                if not (i6 in self.parent.diOrderList):
                    logger.warning(f"[CpEvent]원주문번호 찾기 실패 {i6}")
                    
                    # IOC/FOK의 경우 취소 주문을 낸적이 없어도 자동으로 취소 확인이 들어 온다.
                    if i5 in self.parent.diOrderList and (i16 == "3"):
                        self.parent.deleteOrderNum(i5)
                        self.parent.ForwardPB("cancelpb", i5)
                    return

                item = self.parent.diOrderList[i6]  # 이체결 리스트

                if i16 == "2":  # 정정 확인
                    """미체결 업데이트"""
                    logger.info(f"[CpEvent]정정확인 {item.amount} {i3}")
                    if (item.amount - i3 > 0):
                        """일부정정"""
                        # 기존 데이터 업데이트
                        item.amount -= i3
                        item.modAvali = item.amount

                        # 새로운 미체결 추가
                        item2 = orderData()
                        item2.code = i9
                        item2.name = self.parent.get_stock_name(i9)
                        item2.orderNum = i5
                        item2.orderPrev = i6
                        item2.buysell = i12
                        item2.modAvali = item2.amount = i3
                        item2.price = i4
                        item2.orderFlag = self.client.GetHeaderValue(18)
                        item2.debugPrint()

                        self.parent.diOrderList[i5] = item2
                        self.parent.orderList.append(item2)

                    else:
                        """잔량정정 : 주문/원주문 번호 변경"""
                        # 잔량 정정 인 경우 ==> 업데이트
                        item.orderNum = i5  # 주문번호 변경
                        item.orderPrev = i6  # 원주문번호 변경

                        item.modAvali = item.amount = i3
                        item.price = i4
                        item.orderFlag = self.client.GetHeaderValue(18)
                        item.debugPrint()

                        # 주문번호가 변경되어 기존 key는 제거
                        self.parent.diOrderList[i5] = item
                        del self.parent.diOrderList[i6]

                elif i16 == "3":  # 취소 확인 ==> 미체결 찾아 지운다.
                    """주문취소 확인 시 미체결 지움"""
                    self.parent.deleteOrderNum(i6)
                    self.parent.ForwardPB("cancelpb", i6)

                logger.info(f"[CpEvent]미체결 개수 {len(self.parent.orderList)}")

            elif i14 == "3":  # 거부
                """거부 시"""
                logger.warning("[CpEvent]거부")

            elif i14 == "4":  # 접수
                """접수 시 정정/취소접수는 무시, 신규접수만 처리"""
                if not (i16 == "1"):
                    logger.info("[CpEvent]정정이나 취소 접수는 일단 무시한다.")
                    return

                """신규접수 처리"""
                item = orderData()
                item.code = i9
                item.name = self.parent.get_stock_name(i9)
                item.orderNum = i5
                item.buysell = i12
                item.modAvali = item.amount = i3
                item.price = i4
                item.orderFlag = self.client.GetHeaderValue(18)
                item.debugPrint()
                self.parent.diOrderList[i5] = item
                self.parent.orderList.append(item)

                logger.info(f"[CpEvent]미체결 개수 {len(self.parent.orderList)}")

            return


# 미체결 주문 정보 저장 구조체
class orderData:
    def __init__(self):
        self.code = ""          # 종목코드
        self.name = ""          # 종목명
        self.orderNum = 0       # 주문번호
        self.orderPrev = 0      # 원주문번호
        self.orderDesc = ""     # 주문구분내용
        self.amount = 0         # 주문수량
        self.price = 0          # 주문 단가
        self.ContAmount = 0     # 체결수량
        self.credit = ""        # 신용 구분 "현금" "유통융자" "자기융자" "유통대주" "자기대주"
        self.modAvali = 0       # 정정/취소 가능 수량
        self.buysell = ""       # 매매구분 코드  1 매도 2 매수
        self.creditdate = ""    # 대출일
        self.orderFlag = ""     # 주문호가 구분코드
        self.orderFlagDesc = "" # 주문호가 구분 코드 내용

        # 데이터 변환용
        self.concdic = {"1": "체결", "2": "확인", "3": "거부", "4": "접수"}
        self.buyselldic = {"1": "매도", "2": "매수"}

    def debugPrint(self):
        logger.info(f"{self.code}, {self.name}, 주문번호 {self.orderNum}, 원주문 {self.orderPrev}, {self.orderDesc}, "
                   f"주문수량 {self.amount}, 주문단가 {self.price}, 체결수량 {self.ContAmount}, {self.credit}, "
                   f"정정가능수량 {self.modAvali}, 매수매도: {self.buyselldic.get(self.buysell)}, "
                   f"대출일 {self.creditdate}, 주문호가구분 {self.orderFlag} {self.orderFlagDesc}")


# SB/PB 요청 ROOT 클래스
class CpPublish:
    def __init__(self, name, serviceID):
        self.name = name
        self.obj = win32com.client.Dispatch(serviceID)
        self.bIsSB = False

    def __del__(self):
        self.Unsubscribe()

    def Subscribe(self, var, parent):
        if self.bIsSB:
            self.Unsubscribe()

        if len(var) > 0:
            self.obj.SetInputValue(0, var)

        handler = win32com.client.WithEvents(self.obj, CpEvent)
        handler.set_params(self.obj, self.name, parent)

        self.obj.Subscribe()
        self.bIsSB = True

    def Unsubscribe(self):
        if self.bIsSB:
            self.obj.Unsubscribe()
        self.bIsSB = False


# 실시간 주문 체결 구독 클래스
class CpConclution(CpPublish):
    def __init__(self):
        super().__init__("conclusion", "DsCbo1.CpConclusion")


# 취소 주문 요청에 대한 응답 이벤트 처리 클래스
class CpPB0314:
    def __init__(self, obj):
        self.name = "td0314"
        self.obj = obj

    def Subscribe(self, parent):
        handler = win32com.client.WithEvents(self.obj, CpEvent)
        handler.set_params(self.obj, self.name, parent)


# 주식 주문 취소 클래스
class CpRPOrder:
    def __init__(self, account_number, account_flag):
        self.acc = account_number
        self.accFlag = account_flag
        self.objCancelOrder = win32com.client.Dispatch("CpTrade.CpTd0314")  # 취소
        self.callback = None
        self.bIsRq = False
        self.RqOrderNum = 0     # 취소 주문 중인 주문 번호

    # 주문 취소 통신 - Request를 이용하여 취소 주문
    def RequestCancel(self, ordernum, code, amount, callback):
        if self.bIsRq:
            logger.warning("RequestCancel - 통신 중이라 주문 불가")
            return False

        self.callback = callback
        logger.info(f"[CpRPOrder/RequestCancel]취소주문 {ordernum} {code} {amount}")
        self.objCancelOrder.SetInputValue(1, ordernum)  # 원주문 번호
        self.objCancelOrder.SetInputValue(2, self.acc)  # 계좌번호
        self.objCancelOrder.SetInputValue(3, self.accFlag)  # 상품구분
        self.objCancelOrder.SetInputValue(4, code)  # 종목코드
        self.objCancelOrder.SetInputValue(5, amount)  # 정정 수량, 0이면 잔량 취소임

        # 취소주문 요청
        ret = 0
        while True:
            ret = self.objCancelOrder.Request()
            if ret == 0:
                break

            logger.warning(f"[CpRPOrder/RequestCancel] 주문 요청 실패 ret: {ret}")
            if ret == 4:
                # 연속 통신 초과 처리
                cp_cybos = win32com.client.Dispatch("CpUtil.CpCybos")
                remainTime = cp_cybos.LimitRequestRemainTime
                logger.info(f"연속 통신 초과에 의해 재 통신처리: {remainTime / 1000}초 대기")
                time.sleep(remainTime / 1000)
                continue
            else:   # 1 통신 요청 실패 3 그 외의 오류 4: 주문요청제한 개수 초과
                return False

        self.bIsRq = True
        self.RqOrderNum = ordernum

        # 주문 응답(이벤트로 수신)
        self.objReply = CpPB0314(self.objCancelOrder)
        self.objReply.Subscribe(self)
        return True

    # 취소 주문 - BlockRequest를 이용해서 취소 주문
    def BlockRequestCancel(self, ordernum, code, amount, callback):
        self.callback = callback
        logger.info(f"[CpRPOrder/BlockRequestCancel]취소주문2 {ordernum} {code} {amount}")
        self.objCancelOrder.SetInputValue(1, ordernum)  # 원주문 번호
        self.objCancelOrder.SetInputValue(2, self.acc)  # 계좌번호
        self.objCancelOrder.SetInputValue(3, self.accFlag)  # 상품구분
        self.objCancelOrder.SetInputValue(4, code)  # 종목코드
        self.objCancelOrder.SetInputValue(5, amount)  # 정정 수량, 0이면 잔량 취소임

        # 취소주문 요청
        ret = 0
        while True:
            ret = self.objCancelOrder.BlockRequest()
            if ret == 0:
                break
            logger.warning(f"[CpRPOrder/BlockRequestCancel] 주문 요청 실패 ret: {ret}")
            if ret == 4:
                # 연속 통신 초과 처리
                cp_cybos = win32com.client.Dispatch("CpUtil.CpCybos")
                remainTime = cp_cybos.LimitRequestRemainTime
                logger.info(f"연속 통신 초과에 의해 재 통신처리: {remainTime / 1000}초 대기")
                time.sleep(remainTime / 1000)
                continue
            else:   # 1 통신 요청 실패 3 그 외의 오류 4: 주문요청제한 개수 초과
                return False

        logger.info(f"[CpRPOrder/BlockRequestCancel] 주문결과 {self.objCancelOrder.GetDibStatus()} {self.objCancelOrder.GetDibMsg1()}")
        if self.objCancelOrder.GetDibStatus() != 0:
            return False
        return True

    # 주문 취소 Request에 대한 응답 처리
    def OrderReply(self):
        self.bIsRq = False

        if self.objCancelOrder.GetDibStatus() != 0:
            logger.error(f"[CpRPOrder/OrderReply]통신상태 {self.objCancelOrder.GetDibStatus()} {self.objCancelOrder.GetDibMsg1()}")
            self.callback.ForwardReply(-1, 0)
            return False

        orderPrev = self.objCancelOrder.GetHeaderValue(1)
        code = self.objCancelOrder.GetHeaderValue(4)
        orderNum = self.objCancelOrder.GetHeaderValue(6)
        amount = self.objCancelOrder.GetHeaderValue(5)

        logger.info(f"[CpRPOrder/OrderReply] 주문 취소 reply, 취소한 주문: {orderPrev} {code} {orderNum} {amount}")

        # 주문 취소를 요청한 클래스로 포워딩 한다.
        if self.callback is not None:
            self.callback.ForwardReply(0, orderPrev)


# 미체결 조회 서비스
class Cp5339:
    def __init__(self, account_number, account_flag):
        self.objRq = win32com.client.Dispatch("CpTrade.CpTd5339")
        self.acc = account_number
        self.accFlag = account_flag

    def Request5339(self, dicOrderList, orderList):
        self.objRq.SetInputValue(0, self.acc)
        self.objRq.SetInputValue(1, self.accFlag)
        self.objRq.SetInputValue(4, "0")  # 전체
        self.objRq.SetInputValue(5, "1")  # 정렬 기준 - 역순
        self.objRq.SetInputValue(6, "0")  # 전체
        self.objRq.SetInputValue(7, 20)   # 요청 개수 - 최대 20개

        logger.info("[Cp5339] 미체결 데이터 조회 시작")
        # 미체결 연속 조회를 위해 while 문 사용
        while True:
            ret = self.objRq.BlockRequest()
            if self.objRq.GetDibStatus() != 0:
                logger.error(f"통신상태 {self.objRq.GetDibStatus()} {self.objRq.GetDibMsg1()}")
                return False

            if ret == 2 or ret == 3:
                logger.error(f"통신 오류 {ret}")
                return False

            # 통신 초과 요청 방지에 의한 오류인 경우
            while ret == 4:  # 연속 주문 오류 임. 이 경우는 남은 시간동안 반드시 대기해야 함.
                cp_cybos = win32com.client.Dispatch("CpUtil.CpCybos")
                remainTime = cp_cybos.LimitRequestRemainTime
                logger.info(f"연속 통신 초과에 의해 재 통신처리: {remainTime / 1000}초 대기")
                time.sleep(remainTime / 1000)
                ret = self.objRq.BlockRequest()

            # 수신 개수
            cnt = self.objRq.GetHeaderValue(5)
            logger.info(f"[Cp5339] 수신 개수 {cnt}")
            if cnt == 0:
                break

            for i in range(cnt):
                item = orderData()
                item.orderNum = self.objRq.GetDataValue(1, i)
                item.orderPrev = self.objRq.GetDataValue(2, i)
                item.code = self.objRq.GetDataValue(3, i)  # 종목코드
                item.name = self.objRq.GetDataValue(4, i)  # 종목명
                item.orderDesc = self.objRq.GetDataValue(5, i)  # 주문구분내용
                item.amount = self.objRq.GetDataValue(6, i)  # 주문수량
                item.price = self.objRq.GetDataValue(7, i)  # 주문단가
                item.ContAmount = self.objRq.GetDataValue(8, i)  # 체결수량
                item.credit = self.objRq.GetDataValue(9, i)  # 신용구분
                item.modAvali = self.objRq.GetDataValue(11, i)  # 정정취소 가능수량
                item.buysell = self.objRq.GetDataValue(13, i)  # 매매구분코드
                item.creditdate = self.objRq.GetDataValue(17, i)  # 대출일
                item.orderFlagDesc = self.objRq.GetDataValue(19, i)  # 주문호가구분코드내용
                item.orderFlag = self.objRq.GetDataValue(21, i)  # 주문호가구분코드

                # 사전과 배열에 미체결 item을 추가
                dicOrderList[item.orderNum] = item
                orderList.append(item)

            # 연속 처리 체크 - 다음 데이터가 없으면 중지
            if not self.objRq.Continue:
                logger.info("[Cp5339] 연속 조회 여부: 다음 데이터가 없음")
                break

        return True

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
        
        # 미체결 주문 관리 (예제 기반)
        self.diOrderList = {}  # 미체결 내역 딕셔너리 - key: 주문번호, value - 미체결 레코드
        self.orderList = []    # 미체결 내역 리스트 - 순차 조회 등을 위한 미체결 리스트
        
        # 미체결 통신 object
        self.obj5339 = None
        # 주문 취소 통신 object
        self.objOrder = None
        
        # 실시간 주문 체결 구독
        self.conclusion_subscriber = None
        self.callbacks = {}  # 콜백 함수 저장

        self._connect_creon_and_init_trade()  # 연결 및 거래 초기화 통합
        if self.connected:
            self.cp_code_mgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")
            logger.info("CpCodeMgr COM object initialized.")
            self._make_stock_dic()
            
            # 미체결 관리 객체 초기화
            self.obj5339 = Cp5339(self.account_number, self.account_flag)
            self.objOrder = CpRPOrder(self.account_number, self.account_flag)
            
            # 실시간 주문 체결 구독 시작
            self.conclusion_subscriber = CpConclution()
            self.conclusion_subscriber.Subscribe("", self)

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
            # 종목코드 정규화
            normalized_code = self._normalize_stock_code(stock_code)
            
            objStockMst = win32com.client.Dispatch("DsCbo1.StockMst")
            objStockMst.SetInputValue(0, normalized_code)
            
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

    def _normalize_stock_code(self, stock_code: str) -> str:
        """
        종목코드를 Creon API 형식으로 정규화합니다.
        """
        # 이미 'A'로 시작하면 그대로 반환
        if stock_code.startswith('A'):
            return stock_code
        
        # 'A'로 시작하지 않으면 'A'를 앞에 추가
        return f"A{stock_code}"

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
    # 호가 단위에 맞춰 반올림
    def round_to_tick(self, price):
        if price < 1000:
            return round(price) # 1원 단위 (1,000원 미만)
        elif price < 2000:
            return round(price) # 1원 단위 (1,000원 이상 2,000원 미만)
        elif price < 5000:
            return round(price / 5) * 5 # 5원 단위 (2,000원 이상 5,000원 미만)
        elif price < 10000:
            return round(price / 10) * 10 # 10원 단위 (5,000원 이상 10,000원 미만)
        elif price < 20000:
            return round(price / 10) * 10 # 10원 단위 (10,000원 이상 20,000원 미만)
        elif price < 50000:
            return round(price / 50) * 50 # 50원 단위 (20,000원 이상 50,000원 미만)
        elif price < 100000:
            return round(price / 100) * 100 # 100원 단위 (50,000원 이상 100,000원 미만)
        elif price < 200000:
            return round(price / 100) * 100 # 100원 단위 (100,000원 이상 200,000원 미만)
        elif price < 500000:
            return round(price / 500) * 500 # 500원 단위 (200,000원 이상 500,000원 미만)
        else:
            return round(price / 1000) * 1000 # 1,000원 단위 (500,000원 이상)
    
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
        # 호가에 맞는 주문가격 계산
        hoga = 0
        if price > 0 :
            hoga = self.round_to_tick(int(price))    
        # 입력 값 설정 (string : ord() 금지)
        if order_type == 'buy':
            objOrder.SetInputValue(0, '2') # '2': 매수
        elif order_type == 'sell':
            objOrder.SetInputValue(0, '1') # '1': 매도
        else:
            logger.error(f"Unsupported order type: {order_type}")
            return None
        
        objOrder.SetInputValue(1, self.account_number) # 계좌번호
        objOrder.SetInputValue(2, self.account_flag) # 상품구분
        objOrder.SetInputValue(3, stock_code)       # 종목코드
        objOrder.SetInputValue(4, int(quantity))    # 주문수량
        objOrder.SetInputValue(5, hoga)       # 주문가격(단가) (시장가는 의미 없음)
        objOrder.SetInputValue(7, '0')              # 주문 조건 (0:기본) - IOC/FOK 등 필요시 수정
        objOrder.SetInputValue(8, order_kind)       # 주문 종류: 01-지정가, 03-시장가

        # 정정/취소 주문 시 원주문번호 필요
        if org_order_no:
            objOrder.SetInputValue(9, org_order_no) # 원주문번호 (정정/취소 시 사용)

        # 주문 요청
        ret = objOrder.BlockRequest()
        logger.info(f"BlockRequest 결과: {ret}")
        if ret != 0:
            logger.error(f"Order BlockRequest failed for {stock_code} {order_type} {quantity}@{price}: {ret}")
            return None
        
        status = objOrder.GetDibStatus()
        msg = objOrder.GetDibMsg1()
        logger.info(f"주문 상태: Status={status}, Msg={msg}")
        if status != 0:
            logger.error(f"Order request error for {stock_code}: Status={status}, Msg={msg}")
            return None

        # 주문 성공 시 반환 값
        # value = objOrder.GetHeaderValue(type)
        # type 숫자 에해당하는헤더데이터를반환합니다
        # 0 - (string) 주문종류코드
        # 1 - (string) 계좌번호
        # 2 - (string) 상품관리구분코드
        # 3 - (string) 종목코드
        # 4 - (long) 주문수량
        # 5 - (long) 주문단가
        # 8 - (long) 주문번호
        # 9 - (string) 계좌명
        # 10 - (string) 종목명
        # 12 - (string) 주문조건구분코드, 0: 기본 1: IOC 2:FOK
        # 13 - (string) 주문호가구분코드, 01: 지정가 03: 시장가
        # 14 - (long) 조건단가
        order_id = str(objOrder.GetHeaderValue(8)) # 주문번호
        order_qty = int(objOrder.GetHeaderValue(4))# 주문수량
        logger.info(f"주문성공: {order_type.upper()} {stock_code}, Qty: {order_qty}, Price: {price}, OrderID: {order_id}")
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

                cnt = objRp.GetHeaderValue(7)
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

    def generate_calendar_data(self, trading_days: list[date], start_date: date, end_date: date) -> list[dict]:
        """
        trading_days는 이미 date 타입 리스트임을 가정
        trading_days는 이미 정렬되어 있음을 가정
        """
        trading_days = set(trading_days)

        calendar_data = []
        current_date = start_date
        while current_date <= end_date:
            # current_date가 datetime.datetime이면 date로 변환
            if isinstance(current_date, datetime.datetime):
                date_to_check = current_date.date()
            else:
                date_to_check = current_date

            is_holiday = date_to_check not in trading_days
            calendar_data.append({
                'date': date_to_check,
                'is_holiday': is_holiday,
                'description': '공휴일' if is_holiday else '거래일'
            })
            current_date += timedelta(days=1)

        return calendar_data

    def get_current_cash(self):
        """
        계좌의 주문 가능 현금(잔고)만 반환합니다.
        """
        balance = self.get_account_balance()
        return balance.get('cash', 0.0)

    def is_connected(self):
        """
        Creon API 연결 상태 반환
        """
        return self.connected

    def get_account_positions_dict(self):
        """
        보유 종목 정보를 {code: {quantity, purchase_price}} 형태로 반환합니다.
        """
        positions = self.get_portfolio_positions()
        return {
            p['stock_code']: {
                'quantity': p['size'],
                'purchase_price': p['avg_price']
            }
            for p in positions
        }

    def cancel_order(self, order_id: str) -> bool:
        """
        주문 취소
        """
        logger.info(f"주문 취소 시작: {order_id}")
        
        try:
            if not self._check_creon_status():
                return False

            # CpTrade.CpTd0314 객체 생성 (주문 취소용)
            obj0314 = win32com.client.Dispatch("CpTrade.CpTd0314")
            
            # 주문 취소 정보 설정
            obj0314.SetInputValue(1, order_id)        # 원주문번호
            obj0314.SetInputValue(2, self.account_number)  # 계좌번호
            obj0314.SetInputValue(3, self.account_flag)    # 상품구분
            obj0314.SetInputValue(4, 0)               # 취소수량 (0: 전체취소)
            obj0314.SetInputValue(8, "01")            # 주문호가구분 (01: 보통)

            # 주문 취소 요청
            obj0314.BlockRequest()
            
            # 통신 상태 확인
            rqStatus = obj0314.GetDibStatus()
            rqRet = obj0314.GetDibMsg1()
            if rqStatus != 0:
                logger.error(f"주문 취소 통신 오류: {rqStatus}, {rqRet}")
                return False

            # 취소 결과 확인
            result = obj0314.GetDibMsg1()
            if "정상처리" in result or "취소완료" in result:
                logger.info(f"주문 취소 성공: {order_id}")
                return True
            else:
                logger.error(f"주문 취소 실패: {result}")
                return False

        except Exception as e:
            logger.error(f"주문 취소 오류: {e}")
            return False

    def get_unfilled_orders(self) -> List[Dict[str, Any]]:
        """
        미체결 주문 목록 조회
        """
        logger.info("미체결 주문 목록 조회 시작")
        
        try:
            if not self._check_creon_status():
                return []

            # CpTrade.CpTd5339 객체 생성
            obj5339 = win32com.client.Dispatch("CpTrade.CpTd5339")
            obj5339.SetInputValue(0, self.account_number)  # 계좌번호
            obj5339.SetInputValue(1, self.account_flag)    # 상품구분
            obj5339.SetInputValue(2, 50)                   # 요청건수

            unfilled_orders = []
            
            # 연속 조회로 모든 미체결 주문 조회
            while True:
                obj5339.BlockRequest()
                
                # 통신 상태 확인
                rqStatus = obj5339.GetDibStatus()
                rqRet = obj5339.GetDibMsg1()
                if rqStatus != 0:
                    logger.error(f"미체결 주문 조회 통신 오류: {rqStatus}, {rqRet}")
                    break

                cnt = obj5339.GetHeaderValue(7)
                if cnt == 0:
                    break

                for i in range(cnt):
                    # 미체결 주문만 필터링 (미체결수량 > 0)
                    mod_avali = obj5339.GetDataValue(9, i)  # 정정/취소 가능 수량
                    if mod_avali > 0:
                        order_data = {
                            'order_id': obj5339.GetDataValue(5, i),      # 주문번호
                            'stock_code': obj5339.GetDataValue(12, i),   # 종목코드
                            'stock_name': obj5339.GetDataValue(0, i),    # 종목명
                            'side': 'sell' if obj5339.GetDataValue(13, i) == '1' else 'buy',  # 매수/매도 구분
                            'quantity': obj5339.GetDataValue(7, i),      # 주문수량
                            'price': obj5339.GetDataValue(6, i),         # 주문단가
                            'filled_quantity': obj5339.GetDataValue(8, i),  # 체결수량
                            'unfilled_quantity': mod_avali,              # 미체결수량
                            'order_time': obj5339.GetDataValue(3, i),    # 주문시간
                            'credit_type': obj5339.GetDataValue(1, i),   # 신용구분
                            'order_flag': obj5339.GetDataValue(14, i),   # 주문호가구분
                            'timestamp': datetime.now()
                        }
                        unfilled_orders.append(order_data)

                if not obj5339.Continue:  # 연속 데이터가 없으면
                    break

            logger.info(f"미체결 주문 {len(unfilled_orders)}건 조회 완료")
            return unfilled_orders

        except Exception as e:
            logger.error(f"미체결 주문 조회 오류: {e}")
            return []

 
    def get_current_prices(self, stock_codes: List[str]) -> Dict[str, float]:
        """
        여러 종목의 현재가를 한번에 조회 (DsCbo1.StockMst2 사용)
        """
        logger.info(f"복수 종목 현재가 조회 시작: {len(stock_codes)}개 종목")
        
        try:
            if not self._check_creon_status():
                return {}

            # DsCbo1.StockMst2 객체 생성 (복수 종목 조회용)
            objStockMst2 = win32com.client.Dispatch("DsCbo1.StockMst2")
            
            # 최대 110종목까지 처리 가능
            if len(stock_codes) > 110:
                logger.warning(f"종목 수가 110개를 초과합니다. 처음 110개만 처리합니다.")
                stock_codes = stock_codes[:110]
            
            # 종목코드를 콤마로 연결
            stock_codes_str = ','.join(stock_codes)
            objStockMst2.SetInputValue(0, stock_codes_str)
            
            objStockMst2.BlockRequest()
            
            # 통신 상태 확인
            rqStatus = objStockMst2.GetDibStatus()
            rqRet = objStockMst2.GetDibMsg1()
            if rqStatus != 0:
                logger.error(f"복수 종목 현재가 조회 통신 오류: {rqStatus}, {rqRet}")
                return {}

            # 결과 처리
            current_prices = {}
            cnt = objStockMst2.GetHeaderValue(0)  # count
            
            for i in range(cnt):
                stock_code = objStockMst2.GetDataValue(0, i)  # 종목코드
                current_price = objStockMst2.GetDataValue(3, i)  # 현재가
                current_prices[stock_code] = current_price

            logger.info(f"복수 종목 현재가 조회 완료: {len(current_prices)}개 종목")
            return current_prices

        except Exception as e:
            logger.error(f"복수 종목 현재가 조회 오류: {e}")
            return {}

    def get_current_prices_detailed(self, stock_codes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        여러 종목의 상세 현재가 정보를 한번에 조회 (시고저현종가 포함)
        """
        logger.info(f"복수 종목 상세 현재가 조회 시작: {len(stock_codes)}개 종목")
        
        try:
            if not self._check_creon_status():
                return {}

            # DsCbo1.StockMst2 객체 생성
            objStockMst2 = win32com.client.Dispatch("DsCbo1.StockMst2")
            
            # 최대 110종목까지 처리 가능
            if len(stock_codes) > 110:
                logger.warning(f"종목 수가 110개를 초과합니다. 처음 110개만 처리합니다.")
                stock_codes = stock_codes[:110]
            
            # 종목코드를 콤마로 연결
            stock_codes_str = ','.join(stock_codes)
            objStockMst2.SetInputValue(0, stock_codes_str)
            
            objStockMst2.BlockRequest()
            
            # 통신 상태 확인
            rqStatus = objStockMst2.GetDibStatus()
            rqRet = objStockMst2.GetDibMsg1()
            if rqStatus != 0:
                logger.error(f"복수 종목 상세 현재가 조회 통신 오류: {rqStatus}, {rqRet}")
                return {}

            # 결과 처리
            detailed_prices = {}
            cnt = objStockMst2.GetHeaderValue(0)  # count
            
            for i in range(cnt):
                stock_code = objStockMst2.GetDataValue(0, i)  # 종목코드
                stock_name = objStockMst2.GetDataValue(1, i)  # 종목명
                time_val = objStockMst2.GetDataValue(2, i)    # 시간(HHMM)
                current_price = objStockMst2.GetDataValue(3, i)  # 현재가
                change = objStockMst2.GetDataValue(4, i)      # 전일대비
                status = objStockMst2.GetDataValue(5, i)      # 상태구분
                open_price = objStockMst2.GetDataValue(6, i)  # 시가
                high_price = objStockMst2.GetDataValue(7, i)  # 고가
                low_price = objStockMst2.GetDataValue(8, i)   # 저가
                ask_price = objStockMst2.GetDataValue(9, i)   # 매도호가
                bid_price = objStockMst2.GetDataValue(10, i)  # 매수호가
                volume = objStockMst2.GetDataValue(11, i)     # 거래량
                trading_value = objStockMst2.GetDataValue(12, i)  # 거래대금
                prev_close = objStockMst2.GetDataValue(19, i) # 전일종가
                
                # 상태구분 코드를 텍스트로 변환
                status_text = {
                    '1': '상한', '2': '상승', '3': '보합', '4': '하한', '5': '하락',
                    '6': '기세상한', '7': '기세상승', '8': '기세하한', '9': '기세하락'
                }.get(status, '알수없음')
                
                detailed_prices[stock_code] = {
                    'stock_name': stock_name,
                    'time': time_val,
                    'current_price': current_price,
                    'change': change,
                    'status': status,
                    'status_text': status_text,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': current_price,  # 현재가를 종가로 사용
                    'ask': ask_price,
                    'bid': bid_price,
                    'volume': volume,
                    'trading_value': trading_value,
                    'prev_close': prev_close,
                    'timestamp': datetime.now()
                }

            logger.info(f"복수 종목 상세 현재가 조회 완료: {len(detailed_prices)}개 종목")
            return detailed_prices

        except Exception as e:
            logger.error(f"복수 종목 상세 현재가 조회 오류: {e}")
            return {}

    def get_daily_ohlcv_from_current(self, stock_codes: List[str]) -> Dict[str, Dict[str, float]]:
        """
        복수 종목의 현재가 정보로부터 당일 일봉 OHLCV 데이터 생성
        """
        logger.info(f"복수 종목 당일 일봉 OHLCV 생성 시작: {len(stock_codes)}개 종목")
        
        try:
            # 상세 현재가 정보 조회
            detailed_prices = self.get_current_prices_detailed(stock_codes)
            
            # OHLCV 데이터로 변환
            daily_ohlcv = {}
            for stock_code, price_data in detailed_prices.items():
                daily_ohlcv[stock_code] = {
                    'open': price_data['open'],
                    'high': price_data['high'],
                    'low': price_data['low'],
                    'close': price_data['current_price'],
                    'volume': price_data['volume']
                }
            
            logger.info(f"복수 종목 당일 일봉 OHLCV 생성 완료: {len(daily_ohlcv)}개 종목")
            return daily_ohlcv

        except Exception as e:
            logger.error(f"복수 종목 당일 일봉 OHLCV 생성 오류: {e}")
            return {}

    def get_filled_orders(self) -> List[Dict[str, Any]]:
        """
        체결된 주문 목록 조회
        """
        logger.info("체결된 주문 목록 조회 시작")
        
        try:
            if not self._check_creon_status():
                return []

            # CpTrade.CpTd5339 객체 생성 (미체결 조회용)
            obj5339 = win32com.client.Dispatch("CpTrade.CpTd5339")
            obj5339.SetInputValue(0, self.account_number)  # 계좌번호
            obj5339.SetInputValue(1, self.account_flag)    # 상품구분
            obj5339.SetInputValue(2, 50)                   # 요청건수

            filled_orders = []
            
            # 연속 조회로 모든 체결 주문 조회
            while True:
                obj5339.BlockRequest()
                
                # 통신 상태 확인
                rqStatus = obj5339.GetDibStatus()
                rqRet = obj5339.GetDibMsg1()
                if rqStatus != 0:
                    logger.error(f"체결 주문 조회 통신 오류: {rqStatus}, {rqRet}")
                    break

                cnt = obj5339.GetHeaderValue(7)
                if cnt == 0:
                    break

                for i in range(cnt):
                    # 체결된 주문만 필터링 (체결수량 > 0)
                    cont_amount = obj5339.GetDataValue(8, i)  # 체결수량
                    if cont_amount > 0:
                        order_data = {
                            'order_id': obj5339.GetDataValue(5, i),      # 주문번호
                            'stock_code': obj5339.GetDataValue(12, i),   # 종목코드
                            'stock_name': obj5339.GetDataValue(0, i),    # 종목명
                            'side': 'sell' if obj5339.GetDataValue(13, i) == '1' else 'buy',  # 매수/매도 구분
                            'quantity': cont_amount,                     # 체결수량
                            'price': obj5339.GetDataValue(7, i),         # 체결단가
                            'order_time': obj5339.GetDataValue(3, i),    # 주문시간
                            'credit_type': obj5339.GetDataValue(1, i),   # 신용구분
                            'timestamp': datetime.now()
                        }
                        filled_orders.append(order_data)

                if not obj5339.Continue:  # 연속 데이터가 없으면
                    break

            logger.info(f"체결된 주문 {len(filled_orders)}건 조회 완료")
            return filled_orders

        except Exception as e:
            logger.error(f"체결된 주문 조회 오류: {e}")
            return []


    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            # 실시간 구독 해지
            if hasattr(self, 'conclusion_subscriber') and self.conclusion_subscriber:
                self.conclusion_subscriber.Unsubscribe()
            
            # 콜백 함수 정리
            if hasattr(self, 'callbacks'):
                self.callbacks.clear()
            
            logger.info("Creon API 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"리소스 정리 오류: {e}")

    def __del__(self):
        """소멸자"""
        self.cleanup()
