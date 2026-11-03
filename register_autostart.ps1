# register_autostart.ps1
# snap2read 서버를 Windows 로그온 시 자동 실행으로 등록/제거하는 스크립트
#
# 등록:  powershell -ExecutionPolicy Bypass -File .\register_autostart.ps1
# 제거:  powershell -ExecutionPolicy Bypass -File .\register_autostart.ps1 -Remove
# 즉시 시작: Start-ScheduledTask -TaskName snap2read

param([switch]$Remove)

$ErrorActionPreference = "Stop"
$TaskName = "snap2read"

# 이 스크립트 위치 = 프로젝트 폴더 (server.py와 같은 폴더에 두고 사용)
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ServerPy   = Join-Path $ProjectDir "server.py"

if ($Remove) {
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($t) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "[OK] 자동실행 작업 '$TaskName' 제거 완료"
    } else {
        Write-Host "등록된 작업이 없습니다."
    }
    exit 0
}

if (-not (Test-Path $ServerPy)) { throw "server.py 를 찾을 수 없습니다: $ServerPy" }

# python.exe 찾기 (PATH 기준)
$Python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) { throw "python.exe 를 PATH에서 찾지 못했습니다. Python 설치 후 다시 실행하세요." }

# 이미 등록돼 있으면 갱신(멱등)
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "기존 작업이 있어 갱신합니다."
}

$Action   = New-ScheduledTaskAction -Execute $Python -Argument "`"$ServerPy`"" -WorkingDirectory $ProjectDir
$Trigger  = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$Settings = New-ScheduledTaskSettingsSet `
                -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                -ExecutionTimeLimit ([TimeSpan]::Zero) `
                -MultipleInstances IgnoreNew `
                -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
    -Settings $Settings -Description "snap2read photo-to-speech server (auto start at logon)" | Out-Null

Write-Host "[OK] 등록 완료 — 로그온 시 서버가 자동 실행됩니다."
Write-Host "  작업 이름 : $TaskName"
Write-Host "  서버      : $ServerPy"
Write-Host "  python    : $Python"
Write-Host "  즉시 시작 : Start-ScheduledTask -TaskName $TaskName"
Write-Host "  제거      : powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Remove"
Write-Host ""
Write-Host "콘솔 창을 숨기려면 작업 스케줄러에서 프로그램을 pythonw.exe 로 변경하세요."
