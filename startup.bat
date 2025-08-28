:: =================================================================
:: 장 시작 전 준비 스케줄 (startup.bat)
:: - 매일 오전 8시 30분경 실행을 권장합니다.
:: =================================================================
@echo off
CHCP 65001 > NUL
SET PYTHONIOENCODING=utf-8

:: --- 환경 설정 ---
SET "PROJECT_ROOT=C:\project\creon_trading"
SET "LOG_DIR=%PROJECT_ROOT%\logs"
SET "LOG_FILE=%LOG_DIR%\startup_%date:~0,4%%date:~5,2%%date:~8,2%.log"

:: 로그 디렉토리 생성
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: 32비트 Conda 환경
SET "CONDA_32_PATH=C:\Anaconda3"
SET "CONDA_32_ENV=system_trading_py37_32" 

echo. >> "%LOG_FILE%"
echo [LOG] %DATE% %TIME% - ========= 장 시작 전 준비 작업 시작 ========= >> "%LOG_FILE%"
echo ========= 장 시작 전 준비 작업 시작 =========

:: --- 32비트 환경 활성화 ---
echo.
echo 32bit Conda 환경(%CONDA_32_ENV%) 활성화...
call "%CONDA_32_PATH%\Scripts\activate.bat" "%CONDA_32_ENV%" 
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda 환경 활성화 실패. 종료합니다.
    GOTO :EOF
)

:: --- 단계 1: Creon 자동 로그인 (32비트 환경) ---
echo [1/3] Creon 자동 로그인
echo [LOG] %DATE% %TIME% - [1/3] Creon 자동 로그인 시작... >> "%LOG_FILE%"
python "%PROJECT_ROOT%\setup\creon_auto_login.py" >> "%LOG_FILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Creon 자동 로그인 실패. 후속 작업을 중단합니다.
    GOTO :EOF
)
echo [LOG] %DATE% %TIME% - [1/3] Creon 자동 로그인 완료. >> "%LOG_FILE%"

:: --- 단계 2: 시스템 상태 확인 (32비트 환경) ---
echo [2/3] 당일 투자금 및 투자전략 계획 작성
echo [LOG] %DATE% %TIME% - [2/3] 투자계획 생성... >> "%LOG_FILE%"
python "%PROJECT_ROOT%\trading\hmm_brain.py" >> "%LOG_FILE%" 2>&1
echo [LOG] %DATE% %TIME% - [2/3] 투자계획 생성 완료. >> "%LOG_FILE%"

:: --- 단계 3: 거래 준비 및 상태 확인 (32비트 환경) ---
echo [3/3] 자동매매 
echo [LOG] %DATE% %TIME% - [3/3] 자동매매 시작... >> "%LOG_FILE%"
:: [수정] start 명령어를 사용하여 "AutoTradingSystem" 이라는 제목으로 새 창에서 실행
:: 이렇게 해야 monitor.bat가 창 제목으로 프로세스를 감지할 수 있습니다.
start "AutoTradingSystem" python "%PROJECT_ROOT%\trading\hmm_trading.py" >> "%LOG_FILE%" 2>&1
echo [LOG] %DATE% %TIME% - [3/3] 자동매매 시작됨. >> "%LOG_FILE%"


:: --- 32비트 환경 비활성화 ---
:: 자동매매 프로그램(hmm_trading.py)이 새 창에서 독립적으로 실행되므로,
:: 이 배치 파일은 Conda 환경을 비활성화하고 즉시 종료됩니다.
echo.
echo Conda 환경 비활성화...
call conda deactivate 
echo.
echo ========= 모든 장 시작 전 준비 작업 완료 =========
echo [LOG] %DATE% %TIME% - ========= 모든 장 시작 전 준비 작업 완료 ========= >> "%LOG_FILE%"

:EOF