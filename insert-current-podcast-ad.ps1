param(
  [string]$AudioPath = 'C:\Users\madha\Downloads\neuroscientist-s-guide-to-10x-your-focus-memory-dr-sahar-yousef-fo559-raj-sham_vxRstE6T.mp3',
  [string]$TranscriptPath = 'C:\Users\madha\Documents\ChatGPT\dsplabproject\data\listening-room\uploads\4bd041770de143cbb843feff31ec8dcb\result\transcript.json',
  [string]$OutputPath = (Join-Path $PSScriptRoot 'field-notes-ad-inserted-1104-879.mp3'),
  [int]$Port = 4175,
  [string]$FFmpegPath = ''
)

$ErrorActionPreference = 'Stop'
$insertAt = 664.879
$scriptText = 'A quick pause, brought to you by Field Notes Club. When a useful idea or goal comes up, writing it down can help you keep track of it. Field Notes Club makes compact pocket journals from recycled paper, designed to fit in a jacket pocket. Your first set includes a free pocket notebook. Now, back to the conversation.'

if (-not (Test-Path -LiteralPath $AudioPath -PathType Leaf)) { throw "Podcast audio not found: $AudioPath" }
if (-not (Test-Path -LiteralPath $TranscriptPath -PathType Leaf)) { throw "Saved transcript not found: $TranscriptPath. This script does not run diarization." }
$ffmpeg = if ($FFmpegPath -and (Test-Path -LiteralPath $FFmpegPath -PathType Leaf)) { Get-Item -LiteralPath $FFmpegPath } else { Get-Command ffmpeg -ErrorAction SilentlyContinue }
$ffmpegExe = if ($ffmpeg) { if ($ffmpeg.Source) { $ffmpeg.Source } else { $ffmpeg.FullName } } else { '' }
if (-not $ffmpegExe) {
  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($python) { $bundled = & $python.Source -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>$null; if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $bundled -PathType Leaf)) { $ffmpegExe = $bundled.Trim() } }
}
if (-not $ffmpegExe) { throw 'The pip package "ffmpeg" is only a Python wrapper; it does not include ffmpeg.exe. Install the bundled binary with: python -m pip install imageio-ffmpeg. Or extract your FFmpeg download and pass its bin\ffmpeg.exe path with -FFmpegPath.' }

$actualHash = (Get-FileHash -LiteralPath $AudioPath -Algorithm SHA256).Hash.ToLowerInvariant()
$transcript = Get-Content -LiteralPath $TranscriptPath -Raw | ConvertFrom-Json
if ($actualHash -ne [string]$transcript.source_sha256) { throw 'The selected MP3 does not match the saved diarization transcript; refusing to insert at an unverified timestamp.' }
$turns = $transcript.variants.overlap_recovery.turns
$anchor = @($turns | Where-Object {
  [math]::Abs([double]$_.end - $insertAt) -lt 0.002 -and
  $_.text -like '*We can make anything happen*' -and
  -not $_.uncertain -and -not $_.overlap
}) | Select-Object -First 1
if (-not $anchor) { throw 'The exact clean host-turn boundary at 11:04.879 was not found in the saved transcript. No audio was changed.' }
$following = @($turns | Where-Object { [double]$_.start -ge $insertAt } | Sort-Object { [double]$_.start } | Select-Object -First 1)
if (-not $following -or $following[0].speaker -eq $anchor.speaker) { throw 'The next diarized speaker turn does not confirm a speaker change at this boundary. No audio was changed.' }

$status = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/status" -TimeoutSec 5
if (-not $status.gnani) { throw "Gnani is not configured on the Airwave server at port $Port. Start the existing server and retry." }
$tempAd = Join-Path $env:TEMP ("airwave-fieldnotes-{0}.wav" -f [guid]::NewGuid().ToString('N'))
try {
  $request = @{ text = $scriptText; language = 'en-IN'; useHostVoice = $true } | ConvertTo-Json -Compress
  Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/tts" -Method Post -ContentType 'application/json' -Body $request -OutFile $tempAd -TimeoutSec 120 | Out-Null
  $signature = [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($tempAd)[0..3])
  if ($signature -ne 'RIFF') { throw 'Gnani did not return a WAV file; no episode output was written.' }

  $filter = "[0:a]atrim=start=0:end=$insertAt,asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo[before];[1:a]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[ad];[0:a]atrim=start=$insertAt,asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo[after];[before][ad][after]concat=n=3:v=0:a=1[out]"
  & $ffmpegExe -hide_banner -loglevel error -y -i $AudioPath -i $tempAd -filter_complex $filter -map '[out]' -c:a libmp3lame -q:a 2 $OutputPath
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) { throw 'FFmpeg failed to write the personalized episode.' }
  Write-Output "Inserted the Field Notes Club sponsor read after the verified host turn at 11:04.879."
  Write-Output "Podcast duration: $([math]::Round([double]$transcript.duration,1)) seconds; ad adds runtime."
  Write-Output "Output: $OutputPath"
  Write-Output "Voice: host clone when available on the running Airwave server; otherwise its configured Gnani voice."
}
finally {
  Remove-Item -LiteralPath $tempAd -Force -ErrorAction SilentlyContinue
}
