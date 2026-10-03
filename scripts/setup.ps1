$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
if (!(Test-Path '.venv\Scripts\python.exe')) { python -m venv .venv }
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed' }
if (!(Get-Command ffmpeg -ErrorAction SilentlyContinue) -or !(Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    New-Item -ItemType Directory -Force '.tools' | Out-Null
    $ffArchive = Join-Path $projectRoot '.tools\ffmpeg.zip'
    if (!(Test-Path -LiteralPath $ffArchive)) { Invoke-WebRequest 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile $ffArchive }
    # Only command-line binaries, not the player/docs.
    & '.\.venv\Scripts\python.exe' -c "import zipfile,pathlib; z=zipfile.ZipFile('.tools/ffmpeg.zip'); [(pathlib.Path('.tools')/pathlib.Path(n).name).write_bytes(z.read(n)) for n in z.namelist() if n.endswith(('/ffmpeg.exe','/ffprobe.exe'))]"
    if ($LASTEXITCODE -ne 0) { throw 'FFmpeg extraction failed' }
    Remove-Item -LiteralPath $ffArchive
}
& '.\.venv\Scripts\python.exe' -m scripts.setup_caption_model
if ($LASTEXITCODE -ne 0) { throw 'Caption model setup failed' }
Push-Location frontend
npm.cmd ci
if ($LASTEXITCODE -ne 0) { Pop-Location; throw 'Frontend dependencies failed' }
npm.cmd run build
if ($LASTEXITCODE -ne 0) { Pop-Location; throw 'Frontend build failed' }
Pop-Location
Write-Host 'Ready. Run scripts\start.ps1, then connect Gemini in the app.'
