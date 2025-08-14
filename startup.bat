@echo off
CHCP 65001 > NUL
SET PYTHONIOENCODING=utf-8

:: =================================================================
:: 장 시작 전 준비 스케줄 (startup.bat)
:: - 매일 오전 8시 30분경 실행을 권장합니다.
:: =================================================================

:: --- 환경 설정 ---
:: --- 환경 설정 ---
SET "PROJECT_ROOT=C:\project\cursor_ai\creon_trading"
SET "LOG_DIR=%PROJECT_ROOT%\logs"
SET "LOG_FILE=%LOG_DIR%\startup_%date:~0,4%%date:~5,2%%date:~8,2%.log"

:: 32비트 Conda 환경
SET "CONDA_32_PATH=C:\Anaconda3"
SET "CONDA_32_ENV=system_trading_py37_32" 
SET PYTHONIOENCODING=utf-8
:: 로그 디렉토리 생성
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

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
echo [1/3] Creon 자동 로그인 시작...
echo [LOG] %DATE% %TIME% - [1/3] Creon 자동 로그인 시작... >> "%LOG_FILE%"
python "%PROJECT_ROOT%\setup\creon_auto_login.py" >> "%LOG_FILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Creon 자동 로그인 실패. 후속 작업을 중단합니다.
    GOTO :EOF
)
echo [LOG] %DATE% %TIME% - [1/3] Creon 자동 로그인 완료. >> "%LOG_FILE%"

:: --- 단계 2: 시스템 상태 확인 (32비트 환경) ---
:: echo [2/3] 거래 시스템 상태 확인 시작...
:: echo [LOG] %DATE% %TIME% - [2/3] 거래 시스템 상태 확인 시작... >> "%LOG_FILE%"
:: python "%PROJECT_ROOT%\setup\startup_process.py" >> "%LOG_FILE%" 2>&1
:: echo [LOG] %DATE% %TIME% - [2/3] 거래 시스템 상태 확인 완료. >> "%LOG_FILE%"

:: --- 단계 3: 거래 준비 및 상태 확인 (32비트 환경) ---
echo [3/3] 거래 시스템 준비 및 상태 확인 시작...
echo [LOG] %DATE% %TIME% - [3/3] 거래 시스템 준비 시작... >> "%LOG_FILE%"
python "%PROJECT_ROOT%\trading\hmm_trading.py" >> "%LOG_FILE%" 2>&1
echo [LOG] %DATE% %TIME% - [3/3] 거래 시스템 준비 완료. >> "%LOG_FILE%"


:: --- 32비트 환경 비활성화 ---
echo.
echo Conda 환경 비활성화...
call conda deactivate 
echo.
echo ========= 모든 장 시작 전 준비 작업 완료 =========
echo [LOG] %DATE% %TIME% - ========= 모든 장 시작 전 준비 작업 완료 ========= >> "%LOG_FILE%"

:EOF