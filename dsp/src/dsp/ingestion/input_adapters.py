"""Input adapters for random audio and Google Meet artifacts."""

from dataclasses import replace
import shutil
from pathlib import Path
import subprocess
from typing import Protocol
import uuid

from dsp.contracts import AudioInput, SourceType
from dsp.exceptions import AudioLoadError, UnsupportedAudioError


class AudioInputAdapter(Protocol):
    """Adapter contract for turning an external source into canonical input metadata."""

    def load(self, source: Path) -> AudioInput:
        """Validate and describe an audio source."""


def _session_id(source: Path, prefix: str) -> str:
    return f"{prefix}_{source.stem.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"


class WavInputAdapter:
    """Validate a native PCM WAV input."""

    def load(self, source: Path) -> AudioInput:
        source = Path(source)
        if not source.exists() or not source.is_file():
            raise AudioLoadError(f"Audio source does not exist: {source}")
        if source.suffix.lower() != ".wav":
            raise UnsupportedAudioError(f"WavInputAdapter requires a .wav file: {source}")
        return AudioInput(
            session_id=_session_id(source, "audio"),
            source_path=source,
            audio_path=source,
            transcript_path=None,
            source_type="audio_file",
            metadata={},
        )


class MediaInputAdapter:
    """Describe audio/media inputs and convert them to WAV through FFmpeg."""

    SUPPORTED_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg", ".flac"}

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else None

    def load(self, source: Path) -> AudioInput:
        source = Path(source)
        if not source.exists() or not source.is_file():
            raise AudioLoadError(f"Media source does not exist: {source}")
        if source.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise UnsupportedAudioError(
                f"Unsupported media format {source.suffix!r}; expected WAV, MP3, M4A or MP4"
            )
        session_id = _session_id(source, "audio")
        if source.suffix.lower() == ".wav":
            audio_path = source
        else:
            if shutil.which("ffmpeg") is None:
                raise UnsupportedAudioError(
                    "MP3, M4A and MP4 require FFmpeg. Install FFmpeg or provide a WAV file."
                )
            if self.output_dir is None:
                raise AudioLoadError("MediaInputAdapter requires output_dir for converted audio")
            audio_path = self.output_dir / session_id / "normalized_input.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                str(audio_path),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                raise AudioLoadError(f"FFmpeg could not decode {source}: {exc.stderr[-500:]}") from exc
        return AudioInput(
            session_id=session_id,
            source_path=source,
            audio_path=audio_path,
            transcript_path=None,
            source_type="audio_file",
            metadata={"input_format": source.suffix.lower().lstrip(".")},
        )


class GoogleMeetArtifactAdapter:
    """Describe an uploaded Meet recording and optional transcript artifact."""

    def __init__(self, media_adapter: MediaInputAdapter | None = None) -> None:
        self.media_adapter = media_adapter or MediaInputAdapter()

    def load(
        self,
        recording: Path | None = None,
        transcript: Path | None = None,
        metadata: dict[str, str] | None = None,
        session_id: str | None = None,
    ) -> AudioInput:
        if recording is None and transcript is None:
            raise AudioLoadError("Google Meet input requires a recording or transcript")
        if recording is not None:
            audio_input = self.media_adapter.load(recording)
            resolved_session_id = session_id or audio_input.session_id
            audio_path = audio_input.audio_path
            source_path = audio_input.source_path
        else:
            transcript = Path(transcript)
            if not transcript.exists() or not transcript.is_file():
                raise AudioLoadError(f"Transcript source does not exist: {transcript}")
            resolved_session_id = session_id or _session_id(transcript, "meet")
            audio_path = None
            source_path = None
        if transcript is not None:
            transcript = Path(transcript)
            if not transcript.exists() or not transcript.is_file():
                raise AudioLoadError(f"Transcript source does not exist: {transcript}")
        return AudioInput(
            session_id=resolved_session_id,
            source_path=source_path,
            audio_path=audio_path,
            transcript_path=transcript,
            source_type="google_meet",
            metadata=dict(metadata or {}),
        )
