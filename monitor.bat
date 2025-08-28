@echo off
CHCP 65001 > NUL
SET PYTHONIOENCODING=utf-8
SETLOCAL ENABLEDELAYEDEXPANSION

:: =================================================================
:: 자동매매 프로세스 모니터링 (monitor.bat)
:: =================================================================

:: --- 환경 설정 ---
SET "PROJECT_ROOT=C:\project\creon_trading"
SET "LOG_DIR=%PROJECT_ROOT%\logs"
SET "MONITOR_LOG_FILE=%LOG_DIR%\monitor_%date:~0,4%%date:~5,2%%date:~8,2%.log"

:: 32비트 Conda 환경
SET "CONDA_32_PATH=C:\Anaconda3"
SET "CONDA_32_ENV=system_trading_py37_32"

SET "MAIN_APP_TITLE=AutoTradingSystem"
SET "LOGIN_SCRIPT=%PROJECT_ROOT%\setup\creon_auto_login.py"
SET "TRADING_SCRIPT=%PROJECT_ROOT%\trading\hmm_trading.py"
SET "TRADING_SCRIPT_NAME=hmm_trading.py"

:: --- 모니터링 시작 ---
echo [%date% %time%] 자동매매 모니터링 시작 >> "%MONITOR_LOG_FILE%"

:: --- 장 시간 확인 (09:00 ~ 15:30) ---
for /f "tokens=1-2 delims=: " %%a in ("%time%") do (
    SET HOUR=%%a
    SET MINUTE=%%b
)
if %HOUR% LSS 9 goto end_monitor
if %HOUR% GTR 15 goto end_monitor
if %HOUR%==15 if %MINUTE% GTR 30 goto end_monitor

:: --- [수정] 프로세스 확인 방식을 WMIC 명령어로 변경 ---
:: 창 제목 대신 실행 명령어 라인에 스크립트 이름이 있는지 확인하여 안정성 확보
wmic process where "name='python.exe' and commandline like '%%%TRADING_SCRIPT_NAME%%%'" get Caption | find "%TRADING_SCRIPT_NAME%" > NUL
IF !ERRORLEVEL! EQU 0 (
    echo [%date% %time%] 자동매매 프로세스 정상 실행 중 (WMIC). >> "%MONITOR_LOG_FILE%"
    GOTO :end_monitor
) ELSE (
    echo [%date% %time%] 자동매매 프로세스가 실행되고 있지 않음. 재시작을 시도합니다. >> "%MONITOR_LOG_FILE%"
)


:: --- 자동매매 시스템 재시작 ---
echo [%date% %time%] Conda 환경(%CONDA_32_ENV%) 활성화... >> "%MONITOR_LOG_FILE%"
call "%CONDA_32_PATH%\Scripts\activate.bat" "%CONDA_32_ENV%"
IF !ERRORLEVEL! NEQ 0 (
    echo [%date% %time%] [ERROR] Conda 환경 활성화 실패. 재시작 중단. >> "%MONITOR_LOG_FILE%"
    GOTO :end_monitor
)

echo [%date% %time%] Creon 자동 로그인 시도... >> "%MONITOR_LOG_FILE%"
python "%LOGIN_SCRIPT%" >> "%MONITOR_LOG_FILE%" 2>&1
IF !ERRORLEVEL! NEQ 0 (
    echo [%date% %time%] [ERROR] Creon 자동 로그인 실패. 재시작 중단. >> "%MONITOR_LOG_FILE%"
    GOTO :end_monitor
)

echo [%date% %time%] 자동매매 스크립트 재시작... >> "%MONITOR_LOG_FILE%"
start "%MAIN_APP_TITLE%" python "%TRADING_SCRIPT%" >> "%MONITOR_LOG_FILE%" 2>&1
echo [%date% %time%] 자동매매 시스템 재시작 완료. >> "%MONITOR_LOG_FILE%"


:end_monitor
echo [%date% %time%] 모니터링 완료 >> "%MONITOR_LOG_FILE%"
ENDLOCAL