"""Unified Google Meet and random-audio DSP pipeline."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ai_layer.models.base_models import TimeRange
from ai_layer.models.transcription_models import (
    AudioClipPayload,
    AudioClipSegment,
    RawUtterance,
    STTTranscriptOutput,
    StitchedTurn,
)
from dsp.clipping.clip_writer import write_clips
from dsp.contracts import (
    AudioBuffer,
    AudioInput,
    MeetTranscriptEntry,
    SpeakerSegment,
    TranscriptParseResult,
)
from dsp.diarization import DiarizationProvider, MockDiarizer, SingleSpeakerDiarizer
from dsp.ingestion.audio_loader import load_wav
from dsp.ingestion.input_adapters import GoogleMeetArtifactAdapter, MediaInputAdapter
from dsp.meet.transcript_parser import parse_transcript
from dsp.preprocessing.normalizer import normalize_peak, resample_linear, resample_antialiased, to_mono
from dsp.timeline import TimelineBuilder
from dsp.vad.energy_vad import EnergyVAD
from dsp.diarization.config import LocalDiarizationConfig
from dsp.exceptions import DiarizationConfigurationError


@dataclass(frozen=True)
class DSPConfig:
    """Runtime choices for the offline-first pipeline."""

    canonical_sample_rate: int = 16_000
    input_mode: Literal["auto", "google_meet", "audio_file"] = "auto"
    random_audio_provider: Literal["mock", "single_speaker", "local_model"] = "single_speaker"
    local_model: LocalDiarizationConfig = field(default_factory=LocalDiarizationConfig)
    meet_transcript_preferred: bool = True
    overlap_detection_enabled: bool = True
    source_separation_enabled: bool = False
    save_intermediate_audio: bool = True
    save_diagnostics: bool = True
    vad_frame_length_ms: float = 30.0
    vad_hop_length_ms: float = 10.0
    vad_threshold_db: float = -35.0
    vad_min_speech_ms: float = 200.0
    vad_min_silence_ms: float = 250.0
    vad_adaptive_noise_floor: bool = True
    vad_noise_floor_percentile: float = 10.0
    vad_noise_margin_db: float = 3.0

    def __post_init__(self) -> None:
        if self.random_audio_provider not in {"mock", "single_speaker", "local_model"}:
            raise DiarizationConfigurationError("Unknown diarization provider")
        if self.canonical_sample_rate != 16000:
            raise DiarizationConfigurationError("The canonical audio contract requires 16000 Hz")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "DSPConfig":
        """Build configuration from the ``dsp`` section of project config."""

        return cls(
            local_model=LocalDiarizationConfig.from_mapping(dict(values.get("local_model", {}))),
            canonical_sample_rate=int(values.get("canonical_sample_rate", 16_000)),
            input_mode=str(values.get("input_mode", "auto")),  # type: ignore[arg-type]
            random_audio_provider=str(values.get("random_audio_provider", "single_speaker")),  # type: ignore[arg-type]
            meet_transcript_preferred=bool(values.get("meet_transcript_preferred", True)),
            overlap_detection_enabled=bool(values.get("overlap_detection_enabled", True)),
            source_separation_enabled=bool(values.get("source_separation_enabled", False)),
            save_intermediate_audio=bool(values.get("save_intermediate_audio", True)),
            save_diagnostics=bool(values.get("save_diagnostics", True)),
            vad_frame_length_ms=float(values.get("vad_frame_length_ms", 30.0)),
            vad_hop_length_ms=float(values.get("vad_hop_length_ms", 10.0)),
            vad_threshold_db=float(values.get("vad_threshold_db", -35.0)),
            vad_min_speech_ms=float(values.get("vad_min_speech_ms", 200.0)),
            vad_min_silence_ms=float(values.get("vad_min_silence_ms", 250.0)),
            vad_adaptive_noise_floor=bool(values.get("vad_adaptive_noise_floor", True)),
            vad_noise_floor_percentile=float(values.get("vad_noise_floor_percentile", 10.0)),
            vad_noise_margin_db=float(values.get("vad_noise_margin_db", 3.0)),
        )


@dataclass
class DSPPipelineResult:
    """All outputs needed by either the audio or native-transcript AI path."""

    session_id: str
    source_type: Literal["google_meet", "audio_file"]
    payload: AudioClipPayload
    native_entries: list[MeetTranscriptEntry] = field(default_factory=list)
    native_raw_text: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_stt_transcript_output(self) -> STTTranscriptOutput:
        """Convert native Meet entries to the existing AI transcript contract."""

        utterances: list[RawUtterance] = []
        turns: list[StitchedTurn] = []
        for index, entry in enumerate(self.native_entries):
            utterance_id = f"{self.session_id}_utterance_{index:04d}"
            turn_id = f"{self.session_id}_turn_{index:04d}"
            time_range = TimeRange(entry.start_time, entry.end_time)
            audio_path = ""
            for segment in self.payload.segments:
                if segment.preliminary_speaker_id == entry.speaker_id and abs(
                    segment.start_time - entry.start_time
                ) < 0.001:
                    audio_path = segment.clip_path
                    break
            utterances.append(
                RawUtterance(
                    utterance_id=utterance_id,
                    speaker_id=entry.speaker_id,
                    text=entry.text,
                    time_range=time_range,
                    audio_path=audio_path,
                )
            )
            turns.append(
                StitchedTurn(
                    turn_id=turn_id,
                    speaker_id=entry.speaker_id,
                    text=entry.text,
                    time_range=time_range,
                    source_utterance_ids=[utterance_id],
                )
            )
        return STTTranscriptOutput(
            session_id=self.session_id,
            turns=turns,
            raw_utterances=utterances,
        )


class DSPPipeline:
    """Process Meet artifacts or arbitrary audio into the AI handoff contract."""

    def __init__(
        self,
        config: DSPConfig | None = None,
        diarization_provider: DiarizationProvider | None = None,
    ) -> None:
        self.config = config or DSPConfig()
        self._diarization_provider = diarization_provider or self._create_diarizer()
        self._timeline = TimelineBuilder()
        self._vad = EnergyVAD(
            frame_length_ms=self.config.vad_frame_length_ms,
            hop_length_ms=self.config.vad_hop_length_ms,
            threshold_db=self.config.vad_threshold_db,
            min_speech_ms=self.config.vad_min_speech_ms,
            min_silence_ms=self.config.vad_min_silence_ms,
            adaptive_noise_floor=self.config.vad_adaptive_noise_floor,
            noise_floor_percentile=self.config.vad_noise_floor_percentile,
            noise_margin_db=self.config.vad_noise_margin_db,
        )

    def process_audio_file(
        self,
        source: str | Path,
        output_dir: str | Path,
        session_id: str | None = None,
    ) -> DSPPipelineResult:
        """Process a random audio/media file without speaker metadata."""

        output_dir = Path(output_dir)
        input_info = MediaInputAdapter(output_dir=output_dir).load(Path(source))
        if session_id:
            input_info = _with_session_id(input_info, session_id)
        audio = self._prepare_audio(input_info.audio_path)
        speech = self._vad.detect(audio)
        diarized = self._diarization_provider.diarize(audio, speech)
        segments, warnings = self._timeline.validate(diarized, audio.duration_seconds)
        if self.config.random_audio_provider == "local_model":
            warnings.append("Speaker labels are acoustic estimates, not identities. Model confidence is unavailable (stored as 0). Overlap clips contain the original mixed audio; source separation is not enabled.")
        clipped = write_clips(audio, segments, output_dir / input_info.session_id / "clips")
        return self._result(
            input_info=input_info,
            audio=audio,
            segments=clipped,
            warnings=warnings,
        )

    def process_google_meet(
        self,
        recording: str | Path | None = None,
        transcript: str | Path | None = None,
        output_dir: str | Path | None = None,
        metadata: dict[str, str] | None = None,
        session_id: str | None = None,
    ) -> DSPPipelineResult:
        """Process an uploaded Google Meet recording and optional transcript."""

        output_dir = Path(output_dir) if output_dir else Path("data/dsp_sessions")
        media_adapter = MediaInputAdapter(output_dir=output_dir)
        input_info = GoogleMeetArtifactAdapter(media_adapter).load(
            recording=Path(recording) if recording else None,
            transcript=Path(transcript) if transcript else None,
            metadata=metadata,
            session_id=session_id,
        )
        parsed = parse_transcript(input_info.transcript_path) if input_info.transcript_path else TranscriptParseResult()
        audio = self._prepare_audio(input_info.audio_path) if input_info.audio_path else None
        warnings = list(parsed.warnings)
        if parsed.entries and self.config.meet_transcript_preferred:
            duration = audio.duration_seconds if audio else max(item.end_time for item in parsed.entries)
            segments, timeline_warnings = self._timeline.from_meet_entries(parsed.entries, duration)
            warnings.extend(timeline_warnings)
        elif audio is not None:
            speech = self._vad.detect(audio)
            diarized = self._diarization_provider.diarize(audio, speech)
            segments, timeline_warnings = self._timeline.validate(diarized, audio.duration_seconds)
            warnings.extend(timeline_warnings)
            warnings.append("No usable native Meet transcript; acoustic diarization fallback used")
        else:
            segments = []
            warnings.append("Meet transcript contains no usable timed entries")
        if audio is not None and output_dir is not None:
            segments = write_clips(audio, segments, output_dir / input_info.session_id / "clips")
        return self._result(
            input_info=input_info,
            audio=audio,
            segments=segments,
            entries=parsed.entries,
            raw_text=parsed.raw_text,
            warnings=warnings,
        )

    def _prepare_audio(self, path: Path | None) -> AudioBuffer:
        if path is None:
            raise ValueError("Audio path is required for audio processing")
        audio = load_wav(path)
        audio = to_mono(audio)
        resampler = resample_antialiased if self.config.random_audio_provider == "local_model" else resample_linear
        audio = resampler(audio, self.config.canonical_sample_rate)
        return normalize_peak(audio)

    def _create_diarizer(self) -> DiarizationProvider:
        if self.config.random_audio_provider == "mock":
            return MockDiarizer()
        if self.config.random_audio_provider == "local_model":
            # Composition root only: domain algorithms depend on the backend protocol.
            from dsp.diarization.services.local_provider import LocalModelDiarizer
            if self.config.local_model.backend == "nemo_native":
                from dsp.diarization.adapters.nemo_native_backend import NemoNativeSpeakerBackend
                return LocalModelDiarizer(NemoNativeSpeakerBackend(self.config.local_model.native))
            from dsp.diarization.adapters.sherpa_backend import SherpaSpeakerBackend
            return LocalModelDiarizer(SherpaSpeakerBackend(self.config.local_model))
        return SingleSpeakerDiarizer()

    def _result(
        self,
        input_info: AudioInput,
        audio: AudioBuffer | None,
        segments: list[SpeakerSegment],
        entries: list[MeetTranscriptEntry] | None = None,
        raw_text: str = "",
        warnings: list[str] | None = None,
    ) -> DSPPipelineResult:
        payload_segments = [
            AudioClipSegment(
                segment_id=segment.segment_id,
                clip_path=str(segment.clip_path or ""),
                start_time=segment.start_time,
                end_time=segment.end_time,
                preliminary_speaker_id=segment.speaker_id,
                acoustic_confidence=segment.confidence,
                speaker_source=segment.speaker_source,
                is_overlap=segment.is_overlap,
            )
            for segment in segments
        ]
        source_speaker = "unknown"
        if segments and all(segment.speaker_source == "meet_transcript" for segment in segments):
            source_speaker = "meet_transcript"
        elif segments and all(segment.speaker_source == "unknown" for segment in segments):
            source_speaker = "unknown"
        elif segments:
            source_speaker = "diarization"
        payload = AudioClipPayload(
            session_id=input_info.session_id,
            source_audio_path=str(input_info.source_path or input_info.audio_path or ""),
            sample_rate=audio.metadata.sample_rate if audio else self.config.canonical_sample_rate,
            channels=audio.metadata.channels if audio else 1,
            duration_seconds=audio.duration_seconds if audio else max(
                (entry.end_time for entry in entries or []), default=0.0
            ),
            segments=payload_segments,
            source_type=input_info.source_type,
            speaker_source=source_speaker,
            warnings=list(warnings or []),
        )
        return DSPPipelineResult(
            session_id=input_info.session_id,
            source_type=input_info.source_type,
            payload=payload,
            native_entries=list(entries or []),
            native_raw_text=raw_text,
            warnings=list(warnings or []),
        )


def _with_session_id(input_info: AudioInput, session_id: str) -> AudioInput:
    return AudioInput(
        session_id=session_id,
        source_path=input_info.source_path,
        audio_path=input_info.audio_path,
        transcript_path=input_info.transcript_path,
        source_type=input_info.source_type,
        metadata=input_info.metadata,
    )
