$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = (Get-Command 'python.exe' -ErrorAction Stop).Source
$tempPath = Join-Path $projectRoot 'data\benchmarks\notsofar1-desktop-20261109\tmp'
New-Item -ItemType Directory -Force -Path $tempPath | Out-Null
$env:TEMP = $tempPath
$env:TMP = $tempPath
$env:TMPDIR = $tempPath
$env:PYTHONPATH = @($projectRoot, "$projectRoot\src", "$projectRoot\models\benchmark-monitor-deps", "$projectRoot\models\asr-deps", "$projectRoot\models\media-deps", "$projectRoot\.runtime-deps", "$projectRoot\.test-deps") -join [IO.Path]::PathSeparator

Write-Host 'NOTSOFAR-1 Desktop benchmark: preparing local sample and activity reference...' -ForegroundColor Cyan
& $python 'scripts\prepare_notsofar1_sample.py'
if ($LASTEXITCODE -ne 0) { throw "Sample preparation failed with exit code $LASTEXITCODE" }

Write-Host 'Starting 6 diarization runs (2 audio derivations × 3 system modes)...' -ForegroundColor Cyan
& $python 'scripts\run_audio_benchmarks.py' '--config' 'configs\notsofar1_desktop_suite_verified.json' '--phase' 'diarization'
if ($LASTEXITCODE -ne 0) { throw "Benchmark failed with exit code $LASTEXITCODE" }

Write-Host 'Benchmark finished. Results are in data\benchmarks\notsofar1-desktop-20261109\verified\diarization.json' -ForegroundColor Green
Read-Host 'Press Enter to close this window'
