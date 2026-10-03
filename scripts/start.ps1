$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$runtime = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $runtime)) { throw 'Run scripts\setup.ps1 first' }
New-Item -ItemType Directory -Force 'data' | Out-Null
# Bounded BLAS threads keep the worker within memory on small machines.
$env:OPENBLAS_NUM_THREADS = '2'; $env:OMP_NUM_THREADS = '4'; $env:PYTHONIOENCODING = 'utf-8'
if (!(Test-Path 'frontend\dist\index.html')) { throw 'Build frontend first: cd frontend; npm run build' }
if (Test-Path 'data\processes.json') { & (Join-Path $PSScriptRoot 'stop.ps1') }
$workerProcess = Start-Process -FilePath $runtime -ArgumentList '-m','backend.worker' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput 'data\worker.stdout.log' -RedirectStandardError 'data\worker.stderr.log'
$apiProcess = Start-Process -FilePath $runtime -ArgumentList '-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8000' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput 'data\api.stdout.log' -RedirectStandardError 'data\api.stderr.log'
@{ worker=$workerProcess.Id; api=$apiProcess.Id } | ConvertTo-Json | Set-Content 'data\processes.json'
Write-Host 'Open http://127.0.0.1:8000. Logs are in data/. Keep both processes running while editing.'
