@echo off
CHCP 6501 > NUL
:: =================================================================
:: 장 마감 후 데이터 셋업 스케줄 (schedule_closing.bat)
:: - 화면에 진행 상황을 표시하고, 상세 내용은 로그 파일에 저장합니다.
:: - PYTHONIOENCODING=utf-8 설정을 추가하여 로그 파일 인코딩을 통일합니다.
:: =================================================================

:: --- 환경 설정 ---
SET "PROJECT_ROOT=C:\project\cursor_ai\creon_trading"
SET "LOG_DIR=%PROJECT_ROOT%\logs"
SET "LOG_FILE=%LOG_DIR%\closing_%date:~0,4%%date:~5,2%%date:~8,2%.log"

:: 64비트 파이썬 경로 (Anaconda 또는 일반 Python)
SET "PYTHON_64_PATH=C:\Python310\python.exe"

:: 32비트 Conda 환경
SET "CONDA_32_PATH=C:\Anaconda3"
SET "CONDA_32_ENV=system_trading_py37_32"

:: 로그 디렉토리 생성
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo. >> "%LOG_FILE%"
echo [LOG] %DATE% %TIME% - ========= 장 마감 후 데이터 셋업 시작 ========= >> "%LOG_FILE%"
echo ========= 장 마감 후 데이터 셋업 시작 =========

:: --- 단계 1: daily_theme 업데이트 (64비트 환경) ---
echo [1/6] daily_theme 업데이트 시작 (64bit Python)...
echo [LOG] %DATE% %TIME% - [1/6] daily_theme 업데이트 시작... >> "%LOG_FILE%"

:: [수정] 64비트 파이썬 경로 존재 여부 확인
if not exist "%PYTHON_64_PATH%" (
    echo [ERROR] 64비트 파이썬 경로를 찾을 수 없습니다: "%PYTHON_64_PATH%"
    echo [ERROR] %DATE% %TIME% - 64비트 파이썬 경로 오류. >> "%LOG_FILE%"
    GOTO :EOF
)

:: [수정] 파이썬 출력 인코딩을 UTF-8로 설정
SET PYTHONIOENCODING=utf-8
call %PYTHON_64_PATH% "%PROJECT_ROOT%\setup\3_docx_to_daily_theme_64x.py" >> "%LOG_FILE%" 2>&1
echo [LOG] %DATE% %TIME% - [1/6] daily_theme 업데이트 완료. >> "%LOG_FILE%"

:: --- 32비트 환경 활성화 ---
echo.
echo 32bit Conda 환경(%CONDA_32_ENV%) 활성화...
echo [LOG] %DATE% %TIME% - 32bit Conda 환경 활성화... >> "%LOG_FILE%"
call "%CONDA_32_PATH%\Scripts\activate.bat" "%CONDA_32_ENV%"
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda 환경 활성화 실패. 종료합니다.
    echo [ERROR] %DATE% %TIME% - Conda 환경 활성화 실패. 종료. >> "%LOG_FILE%"
    GOTO :EOF
)

:: --- 단계 2: Creon 자동 로그인 (32비트 환경) ---
echo [2/6] Creon 자동 로그인 시작...
echo [LOG] %DATE% %TIME% - [2/6] Creon 자동 로그인 시작... >> "%LOG_FILE%"
python "%PROJECT_ROOT%\setup\creon_auto_login.py" >> "%LOG_FILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Creon 자동 로그인 실패. 후속 작업을 중단합니다.
    echo [ERROR] %DATE% %TIME% - Creon 자동 로그인 실패. 후속 작업을 중단합니다. >> "%LOG_FILE%"
    GOTO :EOF
)
echo [LOG] %DATE% %TIME% - [2/6] Creon 자동 로그인 완료. >> "%LOG_FILE%"

:: --- 단계 3: 기본 정보 업데이트 (32비트 환경) ---
echo [3/7] Market Calendar, Stock Info 업데이트 시작...
echo [LOG] %DATE% %TIME% - [3/7] 기본 정보 업데이트 시작... >> "%LOG_FILE%"
python "%PROJECT_ROOT%\setup\setup_market_info.py" >> "%LOG_FILE%" 2>&1
echo [LOG] %DATE% %TIME% - [3/7] 기본 정보 업데이트 완료. >> "%LOG_FILE%"

:: --- 단계 4: 최근 테마주 데이터 수집 (32비트 환경) ---
echo [4/7] 최근 테마주 데이터 수집 시작...
echo [LOG] %DATE% %TIME% - [4/7] 최근 테마주 데이터 수집 시작... >> "%LOG_FILE%"
python "%PROJECT_ROOT%\setup\setup_theme.py" >> "%LOG_FILE%" 2>&1
echo [LOG] %DATE% %TIME% - [4/7] 최근 테마주 데이터 수집 완료. >> "%LOG_FILE%"

:: --- 단계 5: 주요 종목 주가 및 팩터 데이터 수집 (32비트 환경) ---
echo [5/7] 주요 종목 주가 및 pykrx 팩터 데이터 수집 시작...
echo [LOG] %DATE% %TIME% - [5/7] 주요 종목 주가 및 pykrx 팩터 데이터 수집 시작... >> "%LOG_FILE%"
python "%PROJECT_ROOT%\setup\setup_price_factors.py" >> "%LOG_FILE%" 2>&1
echo [LOG] %DATE% %TIME% - [5/7] 주요 종목 주가 및 pykrx 팩터 데이터 수집 완료. >> "%LOG_FILE%"

:: --- [신규] 단계 6: MarketEye 스냅샷 -> daily_factors 동기화 (32비트 환경) ---
echo [6/7] daily_factors 스냅샷 동기화 시작...
echo [LOG] %DATE% %TIME% - [6/7] daily_factors 스냅샷 동기화 시작... >> "%LOG_FILE%"
python "%PROJECT_ROOT%\setup\sync_factors_from_snapshot.py" >> "%LOG_FILE%" 2>&1
echo [LOG] %DATE% %TIME% - [6/7] daily_factors 스냅샷 동기화 완료. >> "%LOG_FILE%"

:: --- 단계 7: 테마 분석 및 유니버스 후보 선정 (32비트 환경) ---
echo [7/7] 테마 분석 및 유니버스 후보 선정 시작...
echo [LOG] %DATE% %TIME% - [7/7] 테마 분석 시작... >> "%LOG_FILE%"
python "%PROJECT_ROOT%\setup\setup_daily_universe.py" >> "%LOG_FILE%" 2>&1
echo [LOG] %DATE% %TIME% - [7/7] 테마 분석 완료. >> "%LOG_FILE%"

:: --- 32비트 환경 비활성화 ---
echo.
echo Conda 환경 비활성화...
call conda deactivate
echo [LOG] %DATE% %TIME% - Conda 환경 비활성화. >> "%LOG_FILE%"
echo.
echo ========= 모든 장 마감 후 작업 완료 =========
echo [LOG] %DATE% %TIME% - ========= 모든 장 마감 후 작업 완료 ========= >> "%LOG_FILE%"

:EOF
