# api_client/creon_api.py

import win32com.client
import pythoncom # 파일 상단에 import 되어 있는지 확인
import ctypes
import time
import logging
import pandas as pd
import re
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Callable, Tuple
import threading
from enum import Enum

# API 요청 간격 (크레온 API 제한 준수)
API_REQUEST_INTERVAL = 1.5

# 로거 설정
logger = logging.getLogger(__name__)

# --- 주문 관련 Enum ---
class OrderStatus(Enum):
    NOTHING = 1         # 별 일 없는 상태
    NEW_ORDER = 2       # 신규 주문 낸 상태
    ORDER_CONFIRM = 3   # 신규 주문 처리 확인 (접수)
    MODIFY_ORDER = 4    # 정정 주문 낸 상태
    CANCEL_ORDER = 5    # 취소 주문 낸 상태

class OrderType(Enum):
    BUY = "2"    # Creon API code for Buy (매수)
    SELL = "1"   # Creon API code for Sell (매도)
    MODIFY = "modify" # Custom type for modify logic (정정)
    CANCEL = "cancel" # Custom type for cancel logic (취소)

# --- 실시간 이벤트 핸들러 ---
class CpEvent:
    """
    Creon API로부터 실시간 이벤트를 수신하고 CreonAPIClient의 콜백 메서드를 호출합니다.
    """
    def set_params(self, client_obj, event_name: str, parent_instance, stock_code: Optional[str] = None):
        self.client = client_obj
        self.name = event_name
        self.parent = parent_instance # CreonAPIClient 인스턴스
        self.stock_code = stock_code
        self.concdic = {"1": "체결", "2": "확인", "3": "거부", "4": "접수"}
        self.buyselldic = {"1" : "sell", "2" : "buy"}

    def OnReceived(self):
        """PLUS로부터 실시간 이벤트를 수신 받아 처리하는 함수"""
        # 💡 주문 체결/응답 수신
        if self.name == "conclusion":
            conflag = self.client.GetHeaderValue(14)    # 주문상태 {"1": "체결", "2": "확인", "3": "거부", "4": "접수"}
            order_id = self.client.GetHeaderValue(5)
            quantity = self.client.GetHeaderValue(3)    # <-- 변경
            price = self.client.GetHeaderValue(4)
            stock_code = self.client.GetHeaderValue(9)
            buy_sell = self.client.GetHeaderValue(12)
            balance = self.client.GetHeaderValue(23)

            conflags_str = self.concdic.get(str(conflag), "알수없음") # 주문상태 숫자->한글문자
            buy_sell_str = self.buyselldic.get(str(buy_sell), "알수없음")

            logger.info(f"[CpEvent] 주문 체결/응답 수신: {conflags_str} {buy_sell_str} 종목:{stock_code} 가격:{price:,.0f} 수량:{quantity} 주문번호:{order_id} 잔고:{balance}") # <-- 변경

            if self.parent.conclusion_callback:
                self.parent.conclusion_callback({
                    'order_status': conflags_str,
                    'order_id': order_id,
                    'stock_code': stock_code,
                    'price': price,
                    'quantity': quantity,  # <-- 변경
                    'balance': balance,
                    'order_type': buy_sell_str
                })

        # 실시간 현재가 이벤트 처리
        elif self.name == "stockcur":
            exFlag = self.client.GetHeaderValue(19)  # 예상체결 플래그
            cprice = self.client.GetHeaderValue(13)  # 현재가
            
            # 장중이 아니면 처리 안함. (예상체결 플래그 2: 장중)
            if exFlag != ord('2'):
                return
            
            if self.parent.price_update_callback:
                self.parent.price_update_callback(self.stock_code, cprice, time.time())

        # 실시간 10차 호가 이벤트 처리
        elif self.name == "stockbid":
            offer_prices = [self.client.GetHeaderValue(i) for i in range(0, 19, 2)]
            bid_prices = [self.client.GetHeaderValue(i) for i in range(1, 20, 2)]
            offer_amounts = [self.client.GetHeaderValue(i) for i in range(20, 39, 2)]
            bid_amounts = [self.client.GetHeaderValue(39 - i) for i in range(0, 19, 2)]
            
            if self.parent.bid_update_callback:
                self.parent.bid_update_callback(self.stock_code, offer_prices, bid_prices, offer_amounts, bid_amounts)

# --- 실시간 구독 클래스들의 공통 부모 ---
class CpPublish:
    """
    Creon API 실시간 구독 객체들의 기본 클래스.
    """
    def __init__(self, com_obj_prog_id: str, event_name: str):
        self.obj = win32com.client.Dispatch(com_obj_prog_id)
        self.event_handler = None
        self.stock_code = None
        self.event_name = event_name

    def Subscribe(self, parent, stock_code: Optional[str] = None):
        """실시간 구독을 시작합니다."""
        if self.event_handler: 
            self.Unsubscribe()

        self.stock_code = stock_code
        if stock_code:
            self.obj.SetInputValue(0, stock_code)
        
        self.event_handler = win32com.client.WithEvents(self.obj, CpEvent)
        self.event_handler.set_params(self.obj, self.event_name, parent, stock_code)

        self.obj.Subscribe()
        logger.info(f"실시간 구독 시작: {self.event_name} for {stock_code if stock_code else '계좌 전체'}")

    def Unsubscribe(self):
        """실시간 구독을 해지합니다."""
        if self.obj and self.event_handler:
            self.obj.Unsubscribe()
            log_msg = f"실시간 구독 해지: {self.event_name}"
            if self.stock_code:
                log_msg += f" for {self.stock_code}"
            logger.info(log_msg)
        self.event_handler = None
        self.stock_code = None

# --- 특정 실시간 구독 클래스들 ---
class ConclusionSubscriber(CpPublish):
    """주문 체결 실시간 구독"""
    def __init__(self):
        super().__init__("DsCbo1.CpConclusion", "conclusion")

class StockCurSubscriber(CpPublish):
    """주식 현재가 실시간 구독"""
    def __init__(self):
        super().__init__("DsCbo1.StockCur", "stockcur")

class StockBidSubscriber(CpPublish):
    """주식 10차 호가 실시간 구독"""
    def __init__(self):
        super().__init__("Dscbo1.StockJpBid", "stockbid")


class CreonAPIClient:
    """
    Creon Plus API와 통신하는 클라이언트 클래스.
    """
    _api_request_lock = threading.Lock()
    _realtime_sub_lock = threading.Lock()

    def __init__(self):
        self.connected = False
        self.cp_code_mgr = None
        self.cp_cybos = None
        self.obj_trade = None
        
        self.stock_name_dic = {}
        self.stock_code_dic = {}
        self.account_number = None
        self.account_flag = None
        self._connect_creon_and_init_trade()

        if self.connected:
            self.cp_code_mgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")
            logger.info("CpCodeMgr COM object initialized.")
            self._make_stock_dic()

        # 실시간 구독 객체 관리
        self.conclusion_subscriber: Optional[ConclusionSubscriber] = None
        self.stock_cur_subscribers: Dict[str, StockCurSubscriber] = {} 
        self.stock_bid_subscribers: Dict[str, StockBidSubscriber] = {} 
        
        # 실시간 이벤트 콜백 함수
        self.conclusion_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.price_update_callback: Optional[Callable[[str, int, float], None]] = None
        self.bid_update_callback: Optional[Callable[[str, List[int], List[int], List[int], List[int]], None]] = None

        # 실시간 현재가 출력 주기 제어용
        self._last_price_print_time_per_stock: Dict[str, float] = {}

        self._init_conclusion_subscriber()

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
            self.obj_trade = win32com.client.Dispatch('CpTrade.CpTdUtil')
            if self.obj_trade.TradeInit(0) != 0:
                logger.error("주문 초기화 실패 (TradeInit)!")
                raise RuntimeError("Creon TradeInit 실패.")

            self.account_number = self.obj_trade.AccountNumber[0]
            self.account_flag = self.obj_trade.GoodsList(self.account_number, 1)[0]
            logger.info(f"Creon API 계좌 정보 확인: 계좌번호={self.account_number}, 상품구분={self.account_flag}")

        except Exception as e:
            logger.error(f"Creon TradeUtil 초기화 또는 계좌 정보 가져오는 중 오류 발생: {e}", exc_info=True)
            raise

    def _execute_block_request(self, com_object: Any, method_name: str = "BlockRequest") -> Tuple[int, str]:
        """
        COM 객체에 대한 BlockRequest를 실행하고 공통 오류를 처리합니다.
        """
        with CreonAPIClient._api_request_lock:
            try:
                time.sleep(API_REQUEST_INTERVAL)

                ret = getattr(com_object, method_name)()
                
                if ret != 0:
                    obj_identifier = getattr(com_object, '__class__', None).__name__ or str(type(com_object))
                    error_msg = f"COM 객체 {obj_identifier} {method_name} 호출 실패. 반환 코드: {ret}"
                    logger.error(error_msg)
                    return ret, error_msg

                status = com_object.GetDibStatus()
                msg = com_object.GetDibMsg1()

                if status != 0:
                    obj_identifier = getattr(com_object, '__class__', None).__name__ or str(type(com_object))
                    error_msg = f"COM 객체 {obj_identifier} {method_name} 통신 오류: 상태={status}, 메시지={msg}"
                    logger.error(error_msg)
                    return status, msg
                
                return 0, "Success"
            except Exception as e:
                obj_identifier = getattr(com_object, '__class__', None).__name__ or str(type(com_object))
                logger.error(f"COM 객체 {obj_identifier} {method_name} 실행 중 예외 발생: {e}", exc_info=True)
                return -1, f"내부 예외 발생: {str(e)}"

    def _check_creon_status(self):
        """Creon API 사용 가능한지 상태를 확인합니다."""
        if not self.connected:
            logger.error("Creon Plus가 연결되지 않았습니다.")
            return False
        return True
    
    # def get_stock_name(self, find_code: str) -> Optional[str]:
    #     """종목 코드로 종목명을 조회합니다."""
    #     if not self.cp_code_mgr:
    #         logger.error("cp_code_mgr is not initialized.")
    #         return None
    #     return self.cp_code_mgr.CodeToName(find_code)

    # def get_stock_code(self, find_name: str) -> Optional[str]:
    #     """종목명으로 종목 코드를 조회합니다."""
    #     if not self.cp_code_mgr:
    #         logger.error("cp_code_mgr is not initialized.")
    #         return None
    #     return self.cp_code_mgr.NameToCode(find_name)
    
    def _is_spac(self, code_name):
        return re.search(r'\d+호', code_name) is not None

    def _is_preferred_stock(self, code):
        return code[-1] != '0'

    def _is_reits(self, code_name):
        return "리츠" in code_name

    def _make_stock_dic(self):
        """주식 종목 정보를 딕셔너리로 저장합니다."""
        logger.info("종목 코드/명 딕셔너리 생성 시작")
        if not self.cp_code_mgr:
            logger.error("cp_code_mgr is not initialized.")
            return

        try:
            all_codes = self.cp_code_mgr.GetStockListByMarket(1) + self.cp_code_mgr.GetStockListByMarket(2)
            processed_count = 0
            for code in all_codes:
                code_name = self.cp_code_mgr.CodeToName(code)
                if not code_name or self.cp_code_mgr.GetStockSectionKind(code) != 1 or \
                   self._is_spac(code_name) or self._is_preferred_stock(code) or self._is_reits(code_name):
                    continue
                
                self.stock_name_dic[code_name] = code
                self.stock_code_dic[code] = code_name
                processed_count += 1
            logger.info(f"종목 코드/명 딕셔너리 생성 완료. 총 {processed_count}개 종목 저장.")
        except Exception as e:
            logger.error(f"_make_stock_dic 중 오류 발생: {e}", exc_info=True)

    def get_stock_name(self, find_code: str) -> Optional[str]:
        return self.stock_code_dic.get(find_code)

    def get_stock_code(self, find_name: str) -> Optional[str]:
        return self.stock_name_dic.get(find_name, find_name)

    def get_current_price_and_quotes(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        특정 종목의 현재가(종가), 10차 호가 및 각 호가의 잔량을 조회합니다.
        """
        logger.info(f"종목 [{stock_code}] 현재가 및 10차 호가 상세 조회 중...")
        
        # 1. 현재가 기본 정보 조회 (StockMst)
        obj_stock_mst = win32com.client.Dispatch("DsCbo1.StockMst")
        obj_stock_mst.SetInputValue(0, stock_code)
        status_code_mst, message_mst = self._execute_block_request(obj_stock_mst)

        if status_code_mst != 0:
            logger.error(f"StockMst 요청 실패: {message_mst}")
            return None
        
        current_price_data = {
            'stock_name': obj_stock_mst.GetHeaderValue(1),
            'time': obj_stock_mst.GetHeaderValue(4),
            'current_price': obj_stock_mst.GetHeaderValue(11),
            'open_price': obj_stock_mst.GetHeaderValue(13),
            'high_price': obj_stock_mst.GetHeaderValue(14),
            'low_price': obj_stock_mst.GetHeaderValue(15),
            'volume': obj_stock_mst.GetHeaderValue(18),
            'diff': obj_stock_mst.GetHeaderValue(2),
            'diff_rate': obj_stock_mst.GetHeaderValue(3),
        }

        # 2. 10차 호가 및 잔량 조회 (StockJpBid2)
        obj_stock_jpbid2 = win32com.client.Dispatch("DsCbo1.StockJpBid2")
        obj_stock_jpbid2.SetInputValue(0, stock_code)
        status_code_bid, message_bid = self._execute_block_request(obj_stock_jpbid2)

        if status_code_bid != 0:
            logger.error(f"StockJpBid2 요청 실패: {message_bid}")
            return None 

        offer_prices = [obj_stock_jpbid2.GetDataValue(0, i) for i in range(10)]
        bid_prices = [obj_stock_jpbid2.GetDataValue(1, i) for i in range(10)]
        offer_amounts = [obj_stock_jpbid2.GetDataValue(2, i) for i in range(10)]
        bid_amounts = [obj_stock_jpbid2.GetDataValue(3, i) for i in range(10)]

        result = {
            **current_price_data,
            'offer_prices': offer_prices,
            'bid_prices': bid_prices,
            'offer_amounts': offer_amounts,
            'bid_amounts': bid_amounts
        }
        
        logger.info(f"종목 [{stock_code}] 현재가 및 10차 호가 상세 조회 완료. 현재가: {result['current_price']}")
        return result
    
    def get_price_data(self, code: str, period: str, count: int) -> pd.DataFrame:
        logger.info(f"종목 [{code}] 차트 데이터 요청 시작: 주기={period}, 개수={count}")
        try:
            objChart = win32com.client.Dispatch('CpSysDib.StockChart')
            objChart.SetInputValue(0, code)
            objChart.SetInputValue(1, ord('2'))
            objChart.SetInputValue(4, count)
            objChart.SetInputValue(6, ord(period))
            objChart.SetInputValue(9, ord('1')) # 0: 무수정, 1: 수정주가

            chart_fields = [0, 1, 2, 3, 4, 5, 8] if period in ['m', 'T'] else [0, 2, 3, 4, 5, 8]
            if period == 'm':
                objChart.SetInputValue(7, 1)
            objChart.SetInputValue(5, chart_fields)

            status_code, msg = self._execute_block_request(objChart)
            if status_code != 0:
                logger.error(f"종목 [{code}] 차트 요청 오류: {msg}")
                return pd.DataFrame()

            data_count = objChart.GetHeaderValue(3)
            if data_count == 0:
                logger.warning(f"종목 [{code}]에 대한 차트 데이터가 없습니다.")
                return pd.DataFrame()

            data_records = []
            for i in range(data_count):
                record = {}
                date_val = str(objChart.GetDataValue(chart_fields.index(0), i))
                dt_str = date_val
                dt_format = '%Y%m%d'
                if period in ['m', 'T']:
                    time_val = str(objChart.GetDataValue(chart_fields.index(1), i)).zfill(4)
                    dt_str = f"{date_val}{time_val}"
                    dt_format = '%Y%m%d%H%M'
                
                try:
                    record['datetime'] = datetime.strptime(dt_str, dt_format)
                except ValueError:
                    logger.warning(f"날짜/시간 파싱 실패: {dt_str}")
                    continue
                
                record['open'] = objChart.GetDataValue(chart_fields.index(2), i)
                record['high'] = objChart.GetDataValue(chart_fields.index(3), i)
                record['low'] = objChart.GetDataValue(chart_fields.index(4), i)
                record['close'] = objChart.GetDataValue(chart_fields.index(5), i)
                record['volume'] = objChart.GetDataValue(chart_fields.index(8), i)
                data_records.append(record)
            
            df = pd.DataFrame(data_records)
            if 'datetime' in df.columns and not df.empty:
                df = df.dropna(subset=['datetime']).set_index('datetime').sort_index(ascending=True)
            return df
        except Exception as e:
            logger.error(f"종목 [{code}] 차트 데이터 처리 중 오류 발생: {e}", exc_info=True)
            return pd.DataFrame()

    def _get_price_data(self, stock_code, period, from_date_str, to_date_str, interval=1):
        if not self._check_creon_status():
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        objChart = win32com.client.Dispatch('CpSysDib.StockChart')
        objChart.SetInputValue(0, stock_code)
        objChart.SetInputValue(1, ord('1'))
        objChart.SetInputValue(2, int(to_date_str))
        objChart.SetInputValue(3, int(from_date_str))
        objChart.SetInputValue(6, ord(period))
        objChart.SetInputValue(9, ord('1')) # 0: 무수정, 1: 수정주가
        
        if period == 'm':
            objChart.SetInputValue(7, interval)
            requested_fields = [0, 1, 2, 3, 4, 5, 8]
        else:
            requested_fields = [0, 2, 3, 4, 5, 8]
        objChart.SetInputValue(5, requested_fields)

        data_list = []
        while True:
            status_code, msg = self._execute_block_request(objChart)
            if status_code != 0:
                logger.error(f"데이터 요청 실패: {msg}")
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

            received_len = objChart.GetHeaderValue(3)
            if received_len == 0: break

            for i in range(received_len):
                row_data = {}
                if period == 'm':
                    date_val = objChart.GetDataValue(0, i)
                    time_val = str(objChart.GetDataValue(1, i)).zfill(4)
                    try:
                        row_data['datetime'] = datetime.strptime(f"{date_val}{time_val}", '%Y%m%d%H%M')
                    except ValueError: continue
                    row_data['open'] = objChart.GetDataValue(2, i)
                    row_data['high'] = objChart.GetDataValue(3, i)
                    row_data['low'] = objChart.GetDataValue(4, i)
                    row_data['close'] = objChart.GetDataValue(5, i)
                    row_data['volume'] = objChart.GetDataValue(6, i)
                else:
                    row_data['date'] = datetime.strptime(str(objChart.GetDataValue(0, i)), '%Y%m%d').date()
                    row_data['open'] = objChart.GetDataValue(1, i)
                    row_data['high'] = objChart.GetDataValue(2, i)
                    row_data['low'] = objChart.GetDataValue(3, i)
                    row_data['close'] = objChart.GetDataValue(4, i)
                    row_data['volume'] = objChart.GetDataValue(5, i)
                data_list.append(row_data)
            
            if not objChart.Continue: break
        
        if not data_list:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        df = pd.DataFrame(data_list)
        idx_col = 'datetime' if period == 'm' else 'date'
        df[idx_col] = pd.to_datetime(df[idx_col])
        df = df.sort_values(by=idx_col).set_index(idx_col)
        if period != 'm': df.index = df.index.normalize()

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        return df

    def get_daily_ohlcv(self, code, from_date, to_date):
        return self._get_price_data(code, 'D', from_date, to_date)

    def get_minute_ohlcv(self, code, from_date, to_date, interval=1):
        return self._get_price_data(code, 'm', from_date, to_date, interval)
        
    def get_all_trading_days_from_api(self, from_date: date, to_date: date, stock_code: str = 'A005930') -> list[date]:
        from_date_str = from_date.strftime('%Y%m%d')
        to_date_str = to_date.strftime('%Y%m%d')
        ohlcv_df = self._get_price_data(stock_code, 'D', from_date_str, to_date_str)
        if ohlcv_df.empty: return []
        return sorted(list(set(ohlcv_df.index.date.tolist())))

    def get_latest_financial_data(self, stock_code) -> pd.DataFrame:
        logger.info(f"{stock_code} 종목의 최신 재무 데이터를 가져오는 중...")
        objMarketEye = win32com.client.Dispatch("CpSysDib.MarketEye")

        req_fields = [0, 1, 11, 20, 21, 22, 67, 70, 110, 111, 112, 161, 4]
        objMarketEye.SetInputValue(0, req_fields)
        objMarketEye.SetInputValue(1, stock_code)

        status_code, msg = self._execute_block_request(objMarketEye)
        if status_code != 0:
            logger.error(f"재무 데이터 요청 에러 ({stock_code}): {msg}")
            return pd.DataFrame()

        data = []
        try:
            current_price = objMarketEye.GetDataValue(2, 0)
            listed_stock = objMarketEye.GetDataValue(12, 0)
            market_cap = listed_stock * current_price

            finance = {
                'stock_code': objMarketEye.GetDataValue(0, 0),
                'per': float(objMarketEye.GetDataValue(3, 0) or 0),
                'pbr': float(objMarketEye.GetDataValue(4, 0) or 0),
                'eps': float(objMarketEye.GetDataValue(5, 0) or 0),
                'roe': float(objMarketEye.GetDataValue(6, 0) or 0),
                'debt_ratio': float(objMarketEye.GetDataValue(7, 0) or 0),
                'sales': float(objMarketEye.GetDataValue(8, 0) or 0) * 1e8,
                'operating_profit': float(objMarketEye.GetDataValue(9, 0) or 0) * 1e8,
                'net_profit': float(objMarketEye.GetDataValue(10, 0) or 0) * 1e8,
                'market_cap': market_cap
            }
            data.append(finance)
        except Exception as e:
            logger.error(f"재무 데이터 파싱 오류: {e}")
            return pd.DataFrame()
        
        return pd.DataFrame(data)

    def round_to_tick(self, price: float) -> int:
        price = int(price)
        if price < 2000: return round(price)
        elif price < 5000: return round(price / 5) * 5
        elif price < 20000: return round(price / 10) * 10
        elif price < 50000: return round(price / 50) * 50
        elif price < 200000: return round(price / 100) * 100
        elif price < 500000: return round(price / 500) * 500
        else: return round(price / 1000) * 1000

    def send_order(self, stock_code: str, order_type: OrderType, quantity: int, price: int = 0, origin_order_id: Optional[int] = 0, order_condition: str = "0", order_unit: str = "01") -> Dict[str, Any]: # <-- 변경
        """주식 주문 (매수, 매도, 정정, 취소)을 전송합니다."""
        logger.info(f"주문 요청 - 유형: {order_type.name}, 종목: {stock_code}, 수량: {quantity}, 가격: {price}, 원주문번호: {origin_order_id}") # <-- 변경

        com_obj = None
        # 매수/매도 주문
        if order_type in [OrderType.BUY, OrderType.SELL]:
            com_obj = win32com.client.Dispatch("CpTrade.CpTd0311")
            com_obj.SetInputValue(0, order_type.value)  # 매수/매도
            com_obj.SetInputValue(1, self.account_number)
            com_obj.SetInputValue(2, self.account_flag)
            com_obj.SetInputValue(3, stock_code)
            com_obj.SetInputValue(4, quantity) # <-- 변경
            com_obj.SetInputValue(5, self.round_to_tick(price) if price > 0 else 0)
            com_obj.SetInputValue(7, order_condition)
            com_obj.SetInputValue(8, order_unit)
        # 정정 주문    
        elif order_type == OrderType.MODIFY:
            com_obj = win32com.client.Dispatch("CpTrade.CpTd0313")
            com_obj.SetInputValue(1, origin_order_id)
            com_obj.SetInputValue(2, self.account_number)
            com_obj.SetInputValue(3, self.account_flag)
            com_obj.SetInputValue(4, stock_code)
            com_obj.SetInputValue(5, quantity) # <-- 변경
            com_obj.SetInputValue(6, self.round_to_tick(price) if price > 0 else 0)
        # 취소주문    
        elif order_type == OrderType.CANCEL:
            com_obj = win32com.client.Dispatch("CpTrade.CpTd0314")
            com_obj.SetInputValue(1, origin_order_id)
            com_obj.SetInputValue(2, self.account_number)
            com_obj.SetInputValue(3, self.account_flag)
            com_obj.SetInputValue(4, stock_code)
            com_obj.SetInputValue(5, quantity) # <-- 변경
        else:
            return {'status': 'fail', 'message': '지원하지 않는 주문 유형', 'order_id': None}

        status_code, message = self._execute_block_request(com_obj)
        if status_code != 0:
            return {'status': 'fail', 'message': message, 'order_id': None}

        result_order_id = com_obj.GetHeaderValue(8 if order_type in [OrderType.BUY, OrderType.SELL] else 7) if order_type != OrderType.CANCEL else origin_order_id
        return {'status': 'success', 'message': message, 'order_id': result_order_id}

    def get_account_balance(self) -> Optional[Dict[str, Any]]:
        """계좌의 현금 잔고 및 예수금 정보를 조회합니다."""
        logger.info("계좌 잔고 조회 중...")
        obj_cash = win32com.client.Dispatch("CpTrade.CpTdNew5331A")
        obj_cash.SetInputValue(0, self.account_number)
        obj_cash.SetInputValue(1, self.account_flag)
        
        status_code, message = self._execute_block_request(obj_cash)
        if status_code != 0:
            return None

        return {
            'cash_balance': obj_cash.GetHeaderValue(9),
            'deposit': obj_cash.GetHeaderValue(10),
            'withdrawal_possible': obj_cash.GetHeaderValue(11),
            'loan_amount': obj_cash.GetHeaderValue(12)
        }

    def get_portfolio_positions(self) -> List[Dict[str, Any]]:
        """현재 보유 종목 정보를 조회합니다."""
        logger.info("보유 종목 조회 중...")
        obj_pos = win32com.client.Dispatch("CpTrade.CpTd6033")
        obj_pos.SetInputValue(0, self.account_number)
        obj_pos.SetInputValue(1, self.account_flag)
        obj_pos.SetInputValue(2, 50)
        
        positions = []
        while True:
            status_code, message = self._execute_block_request(obj_pos)
            if status_code != 0: break
            
            cnt = obj_pos.GetHeaderValue(7)
            if not isinstance(cnt, int) or cnt <= 0: break

            for i in range(cnt):
                try:
                    positions.append({
                        'stock_code': obj_pos.GetDataValue(12, i),
                        'stock_name': obj_pos.GetDataValue(0, i),
                        'quantity': int(obj_pos.GetDataValue(7, i)),
                        'avg_price': float(obj_pos.GetDataValue(17, i)),
                        'eval_profit_loss': float(obj_pos.GetDataValue(10, i)),
                        'sell_avail_qty': int(obj_pos.GetDataValue(15, i))
                    })
                except Exception as e:
                    logger.error(f"보유 종목 데이터 처리 중 오류: {e}", exc_info=True)
            if not obj_pos.Continue: break
        
        return positions

    def get_unfilled_orders(self) -> List[Dict[str, Any]]:
        """미체결 주문 정보를 조회합니다."""
        logger.info("미체결 주문 조회 중...")
        obj_unfilled = win32com.client.Dispatch("CpTrade.CpTd5339")
        obj_unfilled.SetInputValue(0, self.account_number)
        obj_unfilled.SetInputValue(1, self.account_flag)
        obj_unfilled.SetInputValue(2, 50)

        unfilled_orders = []
        while True:
            status_code, message = self._execute_block_request(obj_unfilled)
            if status_code != 0: break

            cnt = obj_unfilled.GetHeaderValue(7) 
            if not isinstance(cnt, int) or cnt <= 0: break

            for i in range(cnt):
                try:
                    if obj_unfilled.GetDataValue(9, i) > 0:
                        unfilled_orders.append({
                            'order_id': obj_unfilled.GetDataValue(5, i),
                            'stock_code': obj_unfilled.GetDataValue(12, i),
                            'side': 'sell' if obj_unfilled.GetDataValue(13, i) == '1' else 'buy',
                            'quantity': obj_unfilled.GetDataValue(7, i),
                            'price': obj_unfilled.GetDataValue(6, i),
                            'filled_quantity': obj_unfilled.GetDataValue(8, i),
                            'unfilled_quantity': obj_unfilled.GetDataValue(9, i),
                        })
                except Exception as e:
                    logger.error(f"미체결 주문 데이터 처리 중 오류: {e}", exc_info=True)
            if not obj_unfilled.Continue: break
        
        return unfilled_orders

    def get_unexecuted_orders(self, stock_code: str):
        all_unfilled_orders = self.get_unfilled_orders()
        return [order for order in all_unfilled_orders if order.get('stock_code') == stock_code]

    def get_current_cash(self):
        balance = self.get_account_balance()
        return balance.get('cash_balance', 0.0) if balance else 0.0

    def is_connected(self):
        return self.connected

    def get_account_positions_dict(self):
        positions = self.get_portfolio_positions()
        return {p['stock_code']: {'quantity': p['quantity'], 'purchase_price': p['avg_price']} for p in positions}
    
    def get_current_price(self, code: str) -> Optional[Dict[str, Any]]:
        objStockMst = win32com.client.Dispatch("Dscbo1.StockMst")
        objStockMst.SetInputValue(0, code)
        status_code, msg = self._execute_block_request(objStockMst)
        if status_code != 0: return None
        return {'code': code, 'current_price': objStockMst.GetHeaderValue(11)}

    # --- 실시간 구독 관리 ---
    def _init_conclusion_subscriber(self):
        with self._realtime_sub_lock:
            if not self.conclusion_subscriber:
                self.conclusion_subscriber = ConclusionSubscriber()
                self.conclusion_subscriber.Subscribe(self)

    def set_conclusion_callback(self, callback: Callable[[Dict[str, Any]], None]):
        self.conclusion_callback = callback
    
    def set_price_update_callback(self, callback: Callable[[str, int, float], None]):
        self.price_update_callback = callback

    def set_bid_update_callback(self, callback: Callable[[str, List[int], List[int], List[int], List[int]], None]):
        self.bid_update_callback = callback

    def subscribe_realtime_price(self, stock_code: str):
        with self._realtime_sub_lock:
            if stock_code in self.stock_cur_subscribers: return
            subscriber = StockCurSubscriber()
            subscriber.Subscribe(self, stock_code)
            self.stock_cur_subscribers[stock_code] = subscriber

    def unsubscribe_realtime_price(self, stock_code: str):
        with self._realtime_sub_lock:
            if stock_code not in self.stock_cur_subscribers: return
            subscriber = self.stock_cur_subscribers.pop(stock_code)
            subscriber.Unsubscribe()

    def subscribe_realtime_bid(self, stock_code: str):
        with self._realtime_sub_lock:
            if stock_code in self.stock_bid_subscribers: return
            subscriber = StockBidSubscriber()
            subscriber.Subscribe(self, stock_code)
            self.stock_bid_subscribers[stock_code] = subscriber

    def unsubscribe_realtime_bid(self, stock_code: str):
        with self._realtime_sub_lock:
            if stock_code not in self.stock_bid_subscribers: return
            subscriber = self.stock_bid_subscribers.pop(stock_code)
            subscriber.Unsubscribe()
    
    def unsubscribe_all_realtime_data(self):
        with self._realtime_sub_lock:
            for code in list(self.stock_cur_subscribers.keys()):
                self.unsubscribe_realtime_price(code)
            for code in list(self.stock_bid_subscribers.keys()):
                self.unsubscribe_realtime_bid(code)
    
    # --- 정리 ---
    def cleanup(self) -> None:
        """
        사용한 리소스를 정리하고, 모든 실시간 구독을 해지하며,
        COM 스레드가 정상적으로 종료되도록 보장합니다.
        """
        try:
            if self.conclusion_subscriber:
                self.conclusion_subscriber.Unsubscribe()
                self.conclusion_subscriber = None

            self.unsubscribe_all_realtime_data()

            self.conclusion_callback = None
            self.price_update_callback = None
            self.bid_update_callback = None

            logger.info("모든 실시간 구독 해지 및 콜백 정리 완료.")

            # 💡 [중요] 아래 두 줄이 주석 처리되지 않고 반드시 실행되어야 합니다.
            logger.info("COM 스레드 정상 종료를 위해 대기 및 메시지 처리...")
            time.sleep(1) 
            pythoncom.PumpWaitingMessages()
            
            logger.info("CreonAPIClient 리소스 정리 최종 완료.")

        except Exception as e:
            logger.error(f"CreonAPIClient 리소스 정리 중 오류 발생: {e}", exc_info=True)

    def __del__(self):
        self.cleanup()