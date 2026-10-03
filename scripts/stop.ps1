$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$processFile = Join-Path $projectRoot 'data\processes.json'
if (!(Test-Path -LiteralPath $processFile)) { return }
$savedProcesses = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
$runtimePath = Join-Path $projectRoot '.venv\Scripts\python.exe'
foreach ($recordedId in @($savedProcesses.api, $savedProcesses.worker)) {
    $parentProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $recordedId"
    if (!$parentProcess -or !$parentProcess.CommandLine.Contains($runtimePath)) { continue }
    # Windows venv launchers create a child Python process. Only stop this app's children.
    Get-CimInstance Win32_Process -Filter "ParentProcessId = $recordedId" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($runtimePath) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Stop-Process -Id $recordedId -Force -ErrorAction SilentlyContinue
}
Remove-Item -LiteralPath $processFile
Write-Host 'Local API and worker stopped.'
