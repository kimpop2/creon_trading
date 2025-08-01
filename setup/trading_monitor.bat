REM /trading_monitor.bat
REM 이 배치 파일은 장중에 주기적으로 실행되어 
REM main.py 프로세스의 상태를 확인하고, 비정상 종료 시 재시작합니다. 
REM 이 파일을 Windows 작업 스케줄러에 10분마다 등록해야 합니다.

@echo off
setlocal enabledelayedexpansion
CHCP 65001 > NUL

REM --- 설정 ---
SET "PROJECT_DIR=%~dp0"
SET "CONDA_PATH=C:\Anaconda3"
SET "CONDA_ENV_NAME=system_trading_py37_32"
SET "MAIN_APP_SCRIPT=main.py"
SET "MAIN_APP_TITLE=AutoTradingSystem" REM main.py 실행 시 사용할 창 제목 (trading_start.bat과 동일해야 함)
SET "PID_FILE=%PROJECT_DIR%\auto_trading.pid"

REM 로그 디렉토리 생성
SET "LOG_DIR=%PROJECT_DIR%logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
SET "MONITOR_LOG_FILE=%LOG_DIR%\monitor_%date:~0,4%%date:~5,2%%date:~8,2%.log"

echo [%date% %time%] 자동매매 모니터링 시작 >> "%MONITOR_LOG_FILE%"

REM 현재 시각 확인
for /f "tokens=1-3 delims=:." %%a in ("%time%") do (
    set HOUR=%%a
    set MINUTE=%%b
    set SECOND=%%c
)

REM 시간을 24시간 형식으로 변환 (오전 1시 -> 01, 오후 1시 -> 13)
if "%HOUR:~0,1%"==" " set HOUR=0%HOUR:~1,1%

REM 장 시간 확인 (09:00 ~ 15:30)
REM 장 시간이 아니면 모니터링 종료
if %HOUR% LSS 9 goto :end_monitor
if %HOUR% GTR 15 goto :end_monitor
if %HOUR%==15 if %MINUTE% GTR 30 goto :end_monitor

REM 1. 자동매매 프로세스 상태 확인 (PID 파일 및 창 제목 기준)
SET "IS_RUNNING=0"
IF EXIST "%PID_FILE%" (
    FOR /F "tokens=*" %%i IN (%PID_FILE%) DO SET PID_FROM_FILE=%%i
    
    REM PID가 유효한지 확인 (tasklist 사용)
    tasklist /FI "PID eq !PID_FROM_FILE!" /NH 2>NUL | find /I "!PID_FROM_FILE!" >NUL
    IF !ERRORLEVEL! EQU 0 (
        REM PID는 존재하지만, 해당 프로세스가 우리가 원하는 main.py인지 창 제목으로 다시 확인
        tasklist /V /FI "PID eq !PID_FROM_FILE!" /FO CSV /NH | findstr /I /C:"\"!MAIN_APP_TITLE!\"" >NUL
        IF !ERRORLEVEL! EQU 0 (
            echo [%date% %time%] 자동매매 프로세스 정상 실행 중 (PID: !PID_FROM_FILE!) >> "%MONITOR_LOG_FILE%"
            SET "IS_RUNNING=1"
        ) ELSE (
            echo [%date% %time%] PID !PID_FROM_FILE!는 존재하지만, 예상된 창 제목이 아님. 비정상 종료로 간주. >> "%MONITOR_LOG_FILE%"
        )
    ) ELSE (
        echo [%date% %time%] PID 파일의 프로세스(!PID_FROM_FILE!)가 존재하지 않음. 비정상 종료 감지. >> "%MONITOR_LOG_FILE%"
    )
) ELSE (
    echo [%date% %time%] PID 파일 (%PID_FILE%) 없음 - 자동매매 시스템 시작 필요. >> "%MONITOR_LOG_FILE%"
)

IF !IS_RUNNING! EQU 1 GOTO :end_monitor

REM 2. 자동매매 시스템 재시작
echo [%date% %time%] 자동매매 시스템 재시작 시작 >> "%MONITOR_LOG_FILE%"

REM 기존 PID 파일 삭제
IF EXIST "%PID_FILE%" DEL "%PID_FILE%"

REM Conda 가상환경 활성화
call "%CONDA_PATH%\Scripts\activate.bat" "%CONDA_ENV_NAME%"
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] %DATE% %TIME% - Conda 환경 활성화 실패. 재시작 중단. >> "%MONITOR_LOG_FILE%"
    GOTO :end_monitor
)
echo [%date% %time%] Conda 환경 %CONDA_ENV_NAME% 활성화됨. >> "%MONITOR_LOG_FILE%"

REM 자동매매 시스템 시작
cd /d "%PROJECT_DIR%"
start "%MAIN_APP_TITLE%" /B python "%PROJECT_APP_SCRIPT%" >> "%LOG_DIR%\main_trading_%date:~0,4%%date:~5,2%%date:~8,2%.log" 2>&1
echo [%date% %time%] main.py (창 제목: %MAIN_APP_TITLE%) 백그라운드에서 재시작됨. >> "%MONITOR_LOG_FILE%"

REM 새로 시작된 프로세스의 PID 저장
REM 창 제목으로 프로세스 찾기
timeout /t 5 /nobreak >NUL REM 프로세스 시작 대기
FOR /F "tokens=2" %%i IN ('tasklist /V /FI "WINDOWTITLE eq !MAIN_APP_TITLE!*" /IM python.exe /FO CSV /NH') DO (
    SET "NEW_PID=%%i"
    SET "NEW_PID=!NEW_PID:"=!"
    IF NOT "!NEW_PID!"=="" (
        ECHO !NEW_PID! > "%PID_FILE%"
        echo [%date% %time%] 자동매매 시스템 재시작 완료 (새 PID: !NEW_PID!) >> "%MONITOR_LOG_FILE%"
        GOTO :end_monitor
    )
)
echo [ERROR] %DATE% %TIME% - 새 PID를 찾을 수 없음. 재시작 실패 가능성. >> "%MONITOR_LOG_FILE%"

:end_monitor
echo [%date% %time%] 모니터링 완료 >> "%MONITOR_LOG_FILE%"
endlocal