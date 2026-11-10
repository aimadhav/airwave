@echo off
setlocal
cd /d "%~dp0"
if exist "models\dsp-test-runtime\Scripts\python.exe" (
  "models\dsp-test-runtime\Scripts\python.exe" "scripts\serve_recording_viewer.py" --open-browser
) else (
  python "scripts\serve_recording_viewer.py" --open-browser
)
if errorlevel 1 (
  echo.
  echo The local app could not start. Check the Python setup and whether the port is already in use.
  pause
)
