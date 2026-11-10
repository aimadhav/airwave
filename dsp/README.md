# Airwave local speech service

This folder contains the DSP and local recording-service code Airwave needs for timestamped transcription and speaker diarization. Airwave sends the selected audio crop to this service on `127.0.0.1:5095`; episode audio remains on your computer.

## Windows setup

From this folder, create the local Python runtime, install the required packages, and download the speech models:

```powershell
py -3.11 -m venv models\dsp-test-runtime
models\dsp-test-runtime\Scripts\python.exe -m pip install -r requirements.txt -r requirements-diarization.txt -r requirements-recording-viewer.txt
models\dsp-test-runtime\Scripts\python.exe scripts\setup_diarization.py --with-stt
```

Then start the service:

```powershell
.\Start-Listening-Room.cmd
```

The service listens on loopback port `5095`. Airwave submits an audio crop, polls its job status, then reads the completed timestamped transcript. Model files, recordings, and the virtual environment are created locally and ignored by Git.

## Optional overlap recovery

Overlap recovery is optional. To install its extra dependencies and model assets:

```powershell
models\dsp-test-runtime\Scripts\python.exe -m pip install -r requirements-overlap-recovery.txt
models\dsp-test-runtime\Scripts\python.exe scripts\setup_overlap_separator.py
```

## Airwave integration

The API service is `scripts/serve_recording_viewer.py`; `Start-Listening-Room.cmd` launches it. Airwave uses:

- `GET /api/upload-config` to check service availability.
- `POST /api/uploads` to submit the cropped audio and receive a job ID.
- `GET /api/jobs/{jobId}` to check progress.
- `GET /recordings/{jobId}/transcript.json` to retrieve timed words and speaker turns.

For the end-to-end product flow, see [`../HOW_IT_WORKS.md`](../HOW_IT_WORKS.md).
