REM /trading_start.bat
REM 이 배치 파일은 매일 아침 startup.py를 실행하고, main.py를 시작하며, 
REM 장 마감 후 closing.py를 실행하는 역할을 합니다. 
REM main.py의 모니터링 및 재시작은 trading_monitor.bat이 담당하므로, 
REM  여기서는 main.py의 강제 종료 로직을 제거합니다.
@echo off
CHCP 65001 > NUL
REM UTF-8 인코딩 설정 (한글 깨짐 방지)

REM --- 설정 ---
SET "PROJECT_ROOT=%~dp0"
SET "CONDA_PATH=C:\Anaconda3"
SET "CONDA_ENV_NAME=system_trading_py37_32"
SET "MAIN_APP_TITLE=AutoTradingSystem" REM main.py 실행 시 사용할 창 제목

REM 로그 파일 경로 설정 (배치 스크립트 실행 로그)
SET "BATCH_LOG=%PROJECT_ROOT%logs\trading_start_batch_%date:~0,4%%date:~5,2%%date:~8,2%.log"

REM 로그 디렉토리 생성
if not exist "%PROJECT_ROOT%logs" mkdir "%PROJECT_ROOT%logs"

REM 현재 시각을 YYYY-MM-DD HH:MM:SS 형식으로 로깅
echo [LOG] %DATE% %TIME% - trading_start.bat 스크립트 시작. >> "%BATCH_LOG%"

REM --- 1. 08:30:00: 시작 전 준비 작업 (startup.py 실행) ---
echo [LOG] %DATE% %TIME% - 08:30까지 대기 (startup.py 실행). >> "%BATCH_LOG%"
powershell -command "do { $now = Get-Date; Start-Sleep 1 } while ($now.Hour -lt 8 -or ($now.Hour -eq 8 -and $now.Minute -lt 30))"
echo [LOG] %DATE% %TIME% - setup/startup.py 실행 중... >> "%BATCH_LOG%"

REM Conda 환경 활성화
call "%CONDA_PATH%\Scripts\activate.bat" "%CONDA_ENV_NAME%"
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] %DATE% %TIME% - Conda 환경 활성화 실패. 종료. >> "%BATCH_LOG%"
    GOTO :EOF
)
echo [LOG] %DATE% %TIME% - Conda 환경 %CONDA_ENV_NAME% 활성화됨. >> "%BATCH_LOG%"

REM startup.py 실행
python "%PROJECT_ROOT%setup\startup.py" >> "%BATCH_LOG%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] %DATE% %TIME% - startup.py 실행 실패. startup.log 확인 필요. >> "%BATCH_LOG%"
) ELSE (
    echo [LOG] %DATE% %TIME% - startup.py 실행 완료. >> "%BATCH_LOG%"
)

REM --- 2. 08:50:00: 자동매매 시스템 메인 실행 (main.py 실행) ---
echo [LOG] %DATE% %TIME% - 08:50까지 대기 (main.py 실행). >> "%BATCH_LOG%"
powershell -command "do { $now = Get-Date; Start-Sleep 1 } while ($now.Hour -lt 8 -or ($now.Hour -eq 8 -and $now.Minute -lt 50))"
echo [LOG] %DATE% %TIME% - main.py 실행 중... >> "%BATCH_LOG%"

REM main.py를 백그라운드에서 새로운 창으로 실행. 창 제목을 설정하여 모니터링 및 종료에 활용.
start "%MAIN_APP_TITLE%" /B python "%PROJECT_ROOT%main.py" >> "%PROJECT_ROOT%logs\main_trading_%date:~0,4%%date:~5,2%%date:~8,2%.log" 2>&1
echo [LOG] %DATE% %TIME% - main.py (창 제목: %MAIN_APP_TITLE%) 백그라운드에서 시작됨. >> "%BATCH_LOG%"

REM --- 3. 15:40:00: 자동매매 시스템 종료 후 작업 (closing.py 실행) ---
REM main.py는 Trading 클래스 내부 로직에 의해 15:30에 스스로 종료될 것으로 예상.
echo [LOG] %DATE% %TIME% - 15:40까지 대기 (closing.py 실행). >> "%BATCH_LOG%"
powershell -command "do { $now = Get-Date; Start-Sleep 1 } while ($now.Hour -lt 15 -or ($now.Hour -eq 15 -and $now.Minute -lt 40))"
echo [LOG] %DATE% %TIME% - closing.py 실행 중... >> "%BATCH_LOG%"

REM Conda 환경 비활성화 (main.py가 종료된 후에 실행되도록)
call "%CONDA_PATH%\Scripts\deactivate.bat"
echo [LOG] %DATE% %TIME% - Conda 환경 비활성화됨. >> "%BATCH_LOG%"

REM closing.py 실행
python "%PROJECT_ROOT%closing.py" >> "%BATCH_LOG%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] %DATE% %TIME% - closing.py 실행 실패. closing.log 확인 필요. >> "%BATCH_LOG%"
) ELSE (
    echo [LOG] %DATE% %TIME% - closing.py 실행 완료. >> "%BATCH_LOG%"
)

echo [LOG] %DATE% %TIME% - trading_start.bat 스크립트 완료. >> "%BATCH_LOG%"

:EOF
endlocal