import win32com.client
import pythoncom

print("--- Creon API 환경 진단을 시작합니다 ---")

# --- 1. CpUtil.CpCodeMgr 객체 테스트 ---
try:
    print("\n[1] CpUtil.CpCodeMgr 객체를 테스트합니다...")
    # 객체 생성 시도
    cp_code_mgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")
    print(" -> 성공: CpUtil.CpCodeMgr 객체 생성 완료.")
    
    # 메서드 호출 시도
    samsung_code = cp_code_mgr.NameToCode("삼성전자")
    print(f" -> 성공: NameToCode('삼성전자') -> {samsung_code}")
    
    samsung_name = cp_code_mgr.CodeToName("A005930")
    print(f" -> 성공: CodeToName('A005930') -> {samsung_name}")
    
    print(" -> 결과: CpUtil.CpCodeMgr 객체가 정상입니다.")

except Exception as e:
    print(f" -> 실패: CpUtil.CpCodeMgr 객체 테스트 중 오류 발생: {e}")
    # 오류 발생 시, 객체가 가진 실제 메서드 목록을 출력해봅니다.
    try:
        print(" -> 객체가 가진 실제 메서드 목록을 확인합니다...")
        print(dir(cp_code_mgr))
    except:
        pass


# --- 2. CpUtil.CpStockCode 객체 테스트 ---
try:
    print("\n[2] CpUtil.CpStockCode 객체를 테스트합니다...")
    # 객체 생성 시도
    cp_stock_code = win32com.client.Dispatch("CpUtil.CpStockCode")
    print(" -> 성공: CpUtil.CpStockCode 객체 생성 완료.")
    
    # 메서드 호출 시도
    samsung_code = cp_stock_code.NameToCode("삼성전자")
    print(f" -> 성공: NameToCode('삼성전자') -> {samsung_code}")
    
    samsung_name = cp_stock_code.CodeToName("A005930")
    print(f" -> 성공: CodeToName('A005930') -> {samsung_name}")

    print(" -> 결과: CpUtil.CpStockCode 객체가 정상입니다.")

except Exception as e:
    print(f" -> 실패: CpUtil.CpStockCode 객체 테스트 중 오류 발생: {e}")


print("\n--- 진단 완료 ---")