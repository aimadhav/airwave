# Local DSP project used by Airwave

This folder contains the Python speech-processing project that Airwave calls for local transcription and speaker diarization. The original project source remains available here alongside Airwave so the demo's local speech pipeline can be set up from one repository.

Airwave expects this service on `127.0.0.1:5095`. It uses the recording upload API to submit a short FFmpeg crop, polls the returned job, and reads the timestamped transcript. The Node server does not send the podcast audio to Groq.

## Windows setup

From this folder, create the local runtime and install the speech-processing dependencies:

```powershell
py -3.11 -m venv models\dsp-test-runtime
models\dsp-test-runtime\Scripts\python.exe -m pip install -r requirements.txt -r requirements-diarization.txt
models\dsp-test-runtime\Scripts\python.exe scripts\setup_diarization.py --with-stt
```

Then start the local service:

```powershell
.\Start-Listening-Room.cmd
```

The script downloads model assets into ignored local folders when requested. Model binaries, virtual environments, and recording data are not committed in this repo.

## How Airwave connects

The service's upload API is implemented in `scripts/serve_recording_viewer.py`. Airwave checks `/api/upload-config`, posts cropped audio to `/api/uploads`, polls `/api/jobs/{jobId}`, then reads `/recordings/{jobId}/transcript.json`. The default upload configuration uses loopback port `5095`.

For the complete product flow, see [`../HOW_IT_WORKS.md`](../HOW_IT_WORKS.md). For dependency and feature options, see the requirements files and `Documentation/`.
