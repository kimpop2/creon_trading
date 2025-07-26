import pandas as pd
from pykrx import stock
from datetime import date

def run_pykrx_unit_test(stock_code: str, start_date_str: str, end_date_str: str):
    """
    get_daily_factors에서 사용하는 모든 pykrx 함수를 호출하여 데이터 상태를 검증합니다.
    """
    ticker = stock_code.replace('A', '')
    print("="*50)
    print(f"[*] 테스트 시작: 종목코드={stock_code}({ticker}), 기간={start_date_str}~{end_date_str}")
    print("="*50)

    # 모든 함수 호출을 테스트하기 위한 딕셔너리
    functions_to_test = {
        "get_market_ohlcv": stock.get_market_ohlcv,
        "get_market_fundamental": stock.get_market_fundamental,
        "get_market_trading_volume_by_date": stock.get_market_trading_volume_by_date,
        "get_shorting_status_by_date": stock.get_shorting_status_by_date
    }

    for i, (name, func) in enumerate(functions_to_test.items()):
        print(f"\n[{i+1}] {name} 테스트...")
        try:
            # 모든 함수는 fromdate, todate, ticker 파라미터를 공통으로 가짐
            df = func(start_date_str, end_date_str, ticker)
            
            if df.empty:
                print(" -> 결과: 비어있는 DataFrame이 반환되었습니다.")
                continue

            print(f" -> 결과: 총 {len(df)}개 행의 데이터가 반환되었습니다.")
            
            # 중복된 날짜(인덱스) 검사
            duplicates = df.index.duplicated(keep=False)
            num_duplicates = duplicates.sum()
            
            print(f" -> 중복된 날짜 검사: {num_duplicates}개의 중복된 날짜가 발견되었습니다.")
            if num_duplicates > 0:
                print("▼▼▼▼▼ 중복된 데이터 ▼▼▼▼▼")
                print(df[duplicates].sort_index()) # 보기 편하게 정렬
                print("▲▲▲▲▲ 중복된 데이터 ▲▲▲▲▲")

        except Exception as e:
            print(f" -> 오류 발생: {e}")

    print("\n" + "="*50)
    print("[*] 테스트 종료")
    print("="*50)


if __name__ == "__main__":
    # 오류가 발생한 종목과 날짜를 그대로 사용
    TEST_STOCK_CODE = 'A265520'
    TEST_START_DATE = '20240708'
    TEST_END_DATE = '20250728'

    run_pykrx_unit_test(TEST_STOCK_CODE, TEST_START_DATE, TEST_END_DATE)