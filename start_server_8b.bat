@echo off
rem snap2read 서버 실행 — qwen3:8b 고정 (8GB GPU용)
rem 더블클릭 한 번으로 서버 시작. 창을 닫으면 서버도 종료됩니다.
rem 로그는 logs\ 폴더에 날짜별로 저장됩니다 (화면에도 동시 표시).

cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File run_server.ps1 -Model qwen3:8b
