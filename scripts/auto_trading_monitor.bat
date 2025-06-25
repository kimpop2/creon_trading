@echo off
REM 자동매매 시스템 모니터링 및 재시작 배치 파일
REM 윈도우 스케줄러에서 10분마다 실행

setlocal enabledelayedexpansion

REM 설정
set PROJECT_DIR=C:\project\cursor_ai\creon_trading
set LOG_DIR=%PROJECT_DIR%\logs
set PID_FILE=%PROJECT_DIR%\auto_trading.pid
set LOG_FILE=%LOG_DIR%\monitor_%date:~0,4%%date:~5,2%%date:~8,2%.log

REM 로그 디렉토리 생성
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [%date% %time%] 자동매매 모니터링 시작 >> "%LOG_FILE%"

REM 1. 자동매매 프로세스 상태 확인
if exist "%PID_FILE%" (
    for /f "tokens=*" %%i in (%PID_FILE%) do set PID=%%i
    
    REM 프로세스 존재 여부 확인
    tasklist /FI "PID eq !PID!" 2>NUL | find /I "!PID!" >NUL
    if !errorlevel! equ 0 (
        echo [%date% %time%] 자동매매 프로세스 정상 실행 중 (PID: !PID!) >> "%LOG_FILE%"
        goto :end
    ) else (
        echo [%date% %time%] 자동매매 프로세스 비정상 종료 감지 (PID: !PID!) >> "%LOG_FILE%"
    )
) else (
    echo [%date% %time%] PID 파일 없음 - 자동매매 시스템 시작 필요 >> "%LOG_FILE%"
)

REM 2. 장 시간 확인 (9:00-15:30)
for /f "tokens=1-3 delims=:." %%a in ("%time%") do (
    set HOUR=%%a
    set MINUTE=%%b
)

REM 시간을 24시간 형식으로 변환
if "%HOUR:~0,1%"==" " set HOUR=0%HOUR:~1,1%

REM 장 시간이 아니면 종료
if %HOUR% LSS 9 goto :end
if %HOUR% GTR 15 goto :end
if %HOUR%==15 if %MINUTE% GTR 30 goto :end

REM 3. 자동매매 시스템 재시작
echo [%date% %time%] 자동매매 시스템 재시작 시작 >> "%LOG_FILE%"

REM 기존 PID 파일 삭제
if exist "%PID_FILE%" del "%PID_FILE%"

REM Python 가상환경 활성화 (필요시)
REM call %PROJECT_DIR%\venv\Scripts\activate.bat

REM 자동매매 시스템 시작
cd /d "%PROJECT_DIR%"
start /B python main_auto_trading.py > "%LOG_DIR%\auto_trading_%date:~0,4%%date:~5,2%%date:~8,2%.log" 2>&1

REM 프로세스 ID 저장
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| find "main_auto_trading.py"') do (
    set NEW_PID=%%i
    set NEW_PID=!NEW_PID:"=!
    echo !NEW_PID! > "%PID_FILE%"
    echo [%date% %time%] 자동매매 시스템 재시작 완료 (PID: !NEW_PID!) >> "%LOG_FILE%"
)

:end
echo [%date% %time%] 모니터링 완료 >> "%LOG_FILE%"
endlocal 