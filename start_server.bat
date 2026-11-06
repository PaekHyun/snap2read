@echo off
rem snap2read 서버 실행 — miniforge (tf) 환경에서 python server.py
rem 모델을 직접 고르려면 이 파일 사용 (시작 시 Ollama 모델 선택 프롬프트 표시)
rem 고정 실행은 start_server_14b.bat (16GB GPU) / start_server_8b.bat (8GB GPU)
rem 더블클릭 한 번으로 서버 시작. 창을 닫으면 서버도 종료됩니다.

set "ENV_PY=%USERPROFILE%\miniforge3\envs\tf\python.exe"
if not exist "%ENV_PY%" set "ENV_PY=%USERPROFILE%\miniconda3\envs\tf\python.exe"
if not exist "%ENV_PY%" set "ENV_PY=C:\ProgramData\miniforge3\envs\tf\python.exe"

if not exist "%ENV_PY%" (
    echo [오류] miniforge 'tf' 환경의 python.exe 를 찾지 못했습니다.
    echo        실제 경로가 다르면 이 파일의 ENV_PY 값을 수정하세요.
    pause
    exit /b 1
)

cd /d "%~dp0"
echo snap2read 서버 시작 — %ENV_PY%
"%ENV_PY%" server.py
pause
