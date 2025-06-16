@echo off
chcp 65001 > nul
echo 옵티마이저 통합 테스트를 시작합니다...
echo.

REM Python 환경이 활성화되어 있는지 확인
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python이 설치되어 있지 않습니다.
    echo Python을 설치하고 PATH에 추가해주세요.
    pause
    exit /b 1
)

REM 상위 디렉토리로 이동
cd ..

REM 테스트 실행
echo 테스트 실행 중...
python -m unittest discover -s tests/integration -v

REM 테스트 결과 확인
if %ERRORLEVEL% equ 0 (
    echo.
    echo 모든 테스트가 성공적으로 완료되었습니다.
) else (
    echo.
    echo 일부 테스트가 실패했습니다.
)

echo.
pause 