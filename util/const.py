# util/const.py
def get_order_code(order_str: str) -> str:
    for key, val in order_dic.items():
        if key == order_str:
            return val
    return None  # 값을 찾지 못한 경우 None 반환

def get_order_name(order_str: str) -> str:
    for key, val in order_dic.items():
        if val == order_str:
            return key
    return None  # 값을 찾지 못한 경우 None 반환
# 주문상수 변환
order_dic = {
    '매도': '1', '매수': '2',                       # 주문 유형 (order_type)
    '체결': '1', '확인': '2', '거부': '3', '접수': '4', # 주문 결과 (result)
    '해당없음': '00', '유통융자': '01', '자기융자': '02',  # 신용 구분 (credit)
    '유통대주': '03', '자기대주': '04', '주식담보대출': '05', # 신용 구분 (credit) 계속
    '정상주문': '1', '정정주문': '2', '취소주문': '3',     # 주문 종류 (order_action)
    '현금': '1', '신용': '2', '선물대용': '3', '공매도': '4', # 결제 방법 (pay_method)
    '기본': '0', 'IOC': '1', 'FOK': '2',             # IOC/FOK 구분 (ioc_fok)
    '지정가': '01', '임의': '02', '시장가': '03',         # 주문 호가 구분 (order_class)
    '조건부': '05', '최유리': '12', '최우선': '13'      # 주문 호가 구분 (order_class) 계속
}
# 대신증권 신호 상수 변환 점수표
dic_signal = {
    10: {'signal_name': '외국계증권사창구첫매수', 'signal_score': 7},   # 긍정적 시그널, 초기 매수세
    11: {'signal_name': '외국계증권사창구첫매도', 'signal_score': 3},   # 부정적 시그널, 초기 매도세
    12: {'signal_name': '외국인순매수', 'signal_score': 8},       # 강한 긍정적 시그널, 지속적인 매수
    13: {'signal_name': '외국인순매도', 'signal_score': 2},       # 강한 부정적 시그널, 지속적인 매도
    21: {'signal_name': '전일거래량갱신', 'signal_score': 5},       # 중립적 시그널, 관심 증가 가능성
    22: {'signal_name': '최근5일거래량최고갱신', 'signal_score': 7},   # 긍정적 시그널, 활발한 거래
    23: {'signal_name': '최근5일매물대돌파', 'signal_score': 8},   # 긍정적 시그널, 저항선 돌파 가능성
    24: {'signal_name': '최근60일매물대돌파', 'signal_score': 9},  # 매우 긍정적 시그널, 강력한 돌파 가능성
    28: {'signal_name': '최근5일첫상한가', 'signal_score': 9},     # 매우 긍정적 시그널, 강한 상승 추세 시작
    29: {'signal_name': '최근5일신고가갱신', 'signal_score': 8},     # 긍정적 시그널, 상승 추세 지속
    30: {'signal_name': '최근5일신저가갱신', 'signal_score': 2},     # 부정적 시그널, 하락 추세 지속
    31: {'signal_name': '상한가직전', 'signal_score': 9},       # 매우 긍정적 시그널, 강한 매수세
    32: {'signal_name': '하한가직전', 'signal_score': 1},       # 매우 부정적 시그널, 강한 매도세
    41: {'signal_name': '주가 5MA 상향돌파', 'signal_score': 6},    # 단기 긍정적 시그널, 추세 전환 가능성
    42: {'signal_name': '주가 5MA 하향돌파', 'signal_score': 4},    # 단기 부정적 시그널, 추세 전환 가능성
    43: {'signal_name': '거래량 5MA 상향돌파', 'signal_score': 6},    # 단기 긍정적 시그널, 관심 증가
    44: {'signal_name': '주가데드크로스(5MA < 20MA)', 'signal_score': 3}, # 부정적 시그널, 단기 하락 추세
    45: {'signal_name': '주가골든크로스(5MA > 20MA)', 'signal_score': 7}, # 긍정적 시그널, 단기 상승 추세
    46: {'signal_name': 'MACD 매수-Signal(9) 상향돌파', 'signal_score': 7}, # 긍정적 시그널, 매수 추세 시작
    47: {'signal_name': 'MACD 매도-Signal(9) 하향돌파', 'signal_score': 3}, # 부정적 시그널, 매도 추세 시작
    48: {'signal_name': 'CCI 매수-기준선(-100) 상향돌파', 'signal_score': 6}, # 긍정적 시그널, 과매도 탈출
    49: {'signal_name': 'CCI 매도-기준선(100) 하향돌파', 'signal_score': 4},  # 부정적 시그널, 과매수 진입
    50: {'signal_name': 'Stochastic(10,5,5)매수- 기준선상향돌파', 'signal_score': 6}, # 긍정적 시그널, 과매도 탈출
    51: {'signal_name': 'Stochastic(10,5,5)매도- 기준선하향돌파', 'signal_score': 4},  # 부정적 시그널, 과매수 진입
    52: {'signal_name': 'Stochastic(10,5,5)매수- %K%D 교차', 'signal_score': 7}, # 긍정적 시그널, 매수 신호 강화
    53: {'signal_name': 'Stochastic(10,5,5)매도- %K%D 교차', 'signal_score': 3}, # 부정적 시그널, 매도 신호 강화
    54: {'signal_name': 'Sonar 매수-Signal(9) 상향돌파', 'signal_score': 6},    # 긍정적 시그널, 추세 전환 가능성
    55: {'signal_name': 'Sonar 매도-Signal(9) 하향돌파', 'signal_score': 4},    # 부정적 시그널, 추세 전환 가능성
    56: {'signal_name': 'Momentum 매수-기준선(100) 상향돌파', 'signal_score': 7}, # 긍정적 시그널, 상승 모멘텀 강화
    57: {'signal_name': 'Momentum 매도-기준선(100) 하향돌파', 'signal_score': 3}, # 부정적 시그널, 하락 모멘텀 강화
    58: {'signal_name': 'RSI(14) 매수-Signal(9) 상향돌파', 'signal_score': 6},   # 긍정적 시그널, 매수세 강화
    59: {'signal_name': 'RSI(14) 매도-Signal(9) 하향돌파', 'signal_score': 4},   # 부정적 시그널, 매도세 강화
    60: {'signal_name': 'Volume Oscillator 매수-Signal(9) 상향돌파', 'signal_score': 6}, # 긍정적 시그널, 거래량 증가 추세
    61: {'signal_name': 'Volume Oscillator 매도-Signal(9) 하향돌파', 'signal_score': 4}, # 부정적 시그널, 거래량 감소 추세
    62: {'signal_name': 'Price roc 매수-Signal(9) 상향돌파', 'signal_score': 6},  # 긍정적 시그널, 가격 상승 가속화
    63: {'signal_name': 'Price roc 매도-Signal(9) 하향돌파', 'signal_score': 4},  # 부정적 시그널, 가격 하락 가속화
    64: {'signal_name': '일목균형표매수-전환선 > 기준선상향교차', 'signal_score': 7}, # 긍정적 시그널, 추세 전환 및 상승 지속
    65: {'signal_name': '일목균형표매도-전환선 < 기준선하향교차', 'signal_score': 3}, # 부정적 시그널, 추세 전환 및 하락 지속
    66: {'signal_name': '일목균형표매수-주가가선행스팬상향돌파', 'signal_score': 8}, # 강력한 긍정적 시그널, 추세 상승 강화
    67: {'signal_name': '일목균형표매도-주가가선행스팬하향돌파', 'signal_score': 2}, # 강력한 부정적 시그널, 추세 하락 강화
    68: {'signal_name': '삼선전환도-양전환', 'signal_score': 7},       # 긍정적 시그널, 추세 전환 가능성
    69: {'signal_name': '삼선전환도-음전환', 'signal_score': 3},       # 부정적 시그널, 추세 전환 가능성
    70: {'signal_name': '캔들패턴-상승반전형', 'signal_score': 7},     # 긍정적 시그널, 하락 추세 종료 및 상승 시작 가능성
    71: {'signal_name': '캔들패턴-하락반전형', 'signal_score': 3},     # 부정적 시그널, 상승 추세 종료 및 하락 시작 가능성
    81: {'signal_name': '단기급락후 5MA 상향돌파', 'signal_score': 8},  # 긍정적 시그널, 반등 가능성 높음
    82: {'signal_name': '주가이동평균밀집-5%이내', 'signal_score': 6},  # 중립적이나 방향성 결정 임박, 변동성 확대 가능성
    83: {'signal_name': '눌림목재상승-20MA 지지', 'signal_score': 8}   # 긍정적 시그널, 안정적인 지지 후 상승 기대
}

