@echo off
rem snap2read 서버 실행 — Free API 고정 (Gemini → Groq 자동 폴백)
rem 더블클릭 한 번으로 서버 시작. 창을 닫으면 서버도 종료됩니다.
rem ※ config.py에 API 키(GEMINI_API_KEY 또는 GROQ_API_KEY)가 있어야 합니다.
rem 로그는 logs\ 폴더에 날짜별로 저장됩니다 (화면에도 동시 표시).

cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File run_server.ps1 -Model free
