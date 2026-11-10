"""Bridge a local upload to the existing isolated processing command."""

import json
from pathlib import Path
import subprocess
from recording_viewer.exceptions import RecordingViewerError
from recording_viewer.models.upload import StoredRecording
from recording_viewer.upload_config import UploadConfig


class SubprocessRecordingProcessor:
    """Trusted executable/config only; no shell or HTTP-controlled commands."""

    def __init__(self, root: Path, config: UploadConfig) -> None:
        self.root = root.resolve()
        self.config = config

    def process(self, job_id: str, recording: StoredRecording) -> None:
        """Run real DSP and timestamped STT, preserving logs privately."""
        config = self.config
        executable = self.root / config.python_executable
        if not executable.is_file():
            raise RecordingViewerError("The configured local processing runtime is missing")
        values = json.loads((self.root / config.base_config).read_text(encoding="utf-8"))
        values.update(source=str(recording.source), output_directory=str(recording.result.relative_to(self.root)),
                      session_id=f"upload_{job_id}", max_duration_seconds=config.max_audio_seconds)
        job_config = recording.source.parent / "processing-config.json"
        job_config.write_text(json.dumps(values, indent=2), encoding="utf-8")
        with (recording.source.parent / "processing.log").open("w", encoding="utf-8") as log:
            subprocess.run([str(executable), str(self.root / "scripts/build_recording_viewer.py"),
                            "--config", str(job_config)], cwd=self.root, check=True, shell=False,
                           stdout=log, stderr=subprocess.STDOUT, timeout=config.processing_timeout_seconds)
        if not (recording.result / "index.html").is_file():
            raise RecordingViewerError("Processing did not produce a completed transcript page")
