@echo off
rem snap2read 서버 실행 — Free API 고정 (Gemini → Groq 자동 폴백)
rem 더블클릭 한 번으로 서버 시작. 창을 닫으면 서버도 종료됩니다.
rem ※ config.py에 API 키(GEMINI_API_KEY 또는 GROQ_API_KEY)가 있어야 합니다.
rem 로그는 logs\ 폴더에 날짜별로 저장됩니다 (화면에도 동시 표시).

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
if not exist logs mkdir logs
set "LOGFILE=logs\server_%date:~0,4%-%date:~5,2%-%date:~8,2%.log"
echo snap2read 서버 시작 (LLM: Free API — Gemini/Groq) — %ENV_PY%
echo 로그 저장: %LOGFILE%
powershell -NoProfile -Command "& '%ENV_PY%' server.py --model free 2>&1 | ForEach-Object { $_; Add-Content -Path '%LOGFILE%' -Value $_ }"
pause
