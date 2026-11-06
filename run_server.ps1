# snap2read 서버 실행 + 로그 저장 (PowerShell)
# bat에서 호출: powershell -NoProfile -ExecutionPolicy Bypass -File run_server.ps1 -Model free
param(
    [string]$Model = ""        # 비우면 server.py 기본(선택 프롬프트), "free"/"qwen3:14b"/"qwen3:8b"
)

$ErrorActionPreference = "Continue"

# miniforge 환경 python 탐색 — 환경 이름은 'ocr' (tf는 구버전 호환용으로 유지)
$envPy = "$env:USERPROFILE\miniforge3\envs\ocr\python.exe"
if (-not (Test-Path $envPy)) { $envPy = "$env:USERPROFILE\miniconda3\envs\ocr\python.exe" }
if (-not (Test-Path $envPy)) { $envPy = "C:\ProgramData\miniforge3\envs\ocr\python.exe" }
if (-not (Test-Path $envPy)) { $envPy = "$env:USERPROFILE\miniforge3\envs\tf\python.exe" }
if (-not (Test-Path $envPy)) { $envPy = "$env:USERPROFILE\miniconda3\envs\tf\python.exe" }
if (-not (Test-Path $envPy)) { $envPy = "C:\ProgramData\miniforge3\envs\tf\python.exe" }
if (-not (Test-Path $envPy)) {
    Write-Host "[오류] miniforge 'tf' 환경의 python.exe 를 찾지 못했습니다."
    Read-Host "종료하려면 Enter"
    exit 1
}

# 로그 파일 (날짜별, 추가 모드)
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }
$logFile = "logs\server_$(Get-Date -Format 'yyyy-MM-dd').log"

if ($Model) {
    Write-Host "snap2read 서버 시작 (LLM: $Model) — $envPy"
} else {
    Write-Host "snap2read 서버 시작 — $envPy"
}
Write-Host "로그 저장: $logFile"

# 서버 실행 — 화면에도 표시하고 파일에도 기록 (Tee-Object 사용)
$args = @("server.py")
if ($Model) { $args += @("--model", $Model) }

& $envPy @args 2>&1 | Tee-Object -FilePath $logFile -Append

Read-Host "서버가 종료됐습니다. 창을 닫으려면 Enter"
