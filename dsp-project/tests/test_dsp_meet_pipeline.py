"""Tests for Meet artifacts, random audio and the unified DSP payload."""

import json
from pathlib import Path
import wave

import numpy as np

from dsp import DSPConfig, DSPPipeline
from dsp.contracts import AudioBuffer, AudioMetadata, MeetTranscriptEntry, SpeakerSegment
from dsp.diarization import MockDiarizer
from dsp.meet.transcript_parser import parse_json_transcript, parse_vtt_transcript
from dsp.timeline import TimelineBuilder


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1, 1)
    writer = wave.open(str(path), "wb")
    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(sample_rate)
    writer.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    writer.close()


def test_vtt_parser_merges_adjacent_same_speaker() -> None:
    result = parse_vtt_transcript(
        """WEBVTT

00:00:00.200 --> 00:00:00.550
<v Alice>Hello there</v>

00:00:00.550 --> 00:00:00.900
<v Alice>team</v>

00:00:00.900 --> 00:00:01.200
<v Bob>Hi</v>
"""
    )

    assert len(result.entries) == 2
    assert result.entries[0].speaker_name == "Alice"
    assert result.entries[0].text == "Hello there team"
    assert result.entries[0].end_time == 0.9
    assert result.entries[1].speaker_name == "Bob"


def test_json_parser_reports_missing_timestamps_and_speaker() -> None:
    result = parse_json_transcript(
        json.dumps(
            {
                "entries": [
                    {"startTime": 0, "endTime": 1, "text": "Unknown voice"},
                    {"speakerName": "Alice", "text": "Missing time"},
                ]
            }
        )
    )

    assert len(result.entries) == 1
    assert result.entries[0].speaker_id == "SPK_UNKNOWN"
    assert any("no speaker" in warning for warning in result.warnings)
    assert any("no usable timestamp" in warning for warning in result.warnings)


def test_timeline_preserves_overlap_and_marks_both_segments() -> None:
    entries = [
        MeetTranscriptEntry("one", "spk_a", "Alice", 1.0, 3.0, "hello"),
        MeetTranscriptEntry("two", "spk_b", "Bob", 2.0, 4.0, "yes"),
    ]

    segments, warnings = TimelineBuilder().from_meet_entries(entries, duration_seconds=5.0)

    assert all(segment.is_overlap for segment in segments)
    assert len([warning for warning in warnings if "overlaps" in warning]) == 2


def test_random_audio_pipeline_creates_anonymous_clips(tmp_path: Path) -> None:
    samples = np.zeros(16_000, dtype=np.float32)
    time = np.arange(len(samples), dtype=np.float32) / 16_000
    samples[2_000:7_000] = 0.4 * np.sin(2 * np.pi * 240 * time[2_000:7_000])
    samples[10_000:15_000] = 0.3 * np.sin(2 * np.pi * 320 * time[10_000:15_000])
    source = tmp_path / "random.wav"
    _write_wav(source, samples)

    result = DSPPipeline(
        DSPConfig(random_audio_provider="single_speaker", vad_min_silence_ms=100.0)
    ).process_audio_file(source, tmp_path / "output", session_id="random_001")

    assert result.source_type == "audio_file"
    assert len(result.payload.segments) == 2
    assert all(segment.preliminary_speaker_id == "SPEAKER_00" for segment in result.payload.segments)
    assert all(segment.speaker_source == "unknown" for segment in result.payload.segments)
    assert all(Path(segment.clip_path).exists() for segment in result.payload.segments)


def test_mock_diarizer_supports_three_speaker_demo(tmp_path: Path) -> None:
    samples = np.zeros(3 * 16_000, dtype=np.float32)
    time = np.arange(len(samples), dtype=np.float32) / 16_000
    for index, frequency in enumerate((180, 240, 320)):
        start = index * 16_000
        end = start + 8_000
        samples[start:end] = 0.35 * np.sin(2 * np.pi * frequency * time[start:end])
    source = tmp_path / "three_speakers.wav"
    _write_wav(source, samples)

    result = DSPPipeline(
        DSPConfig(random_audio_provider="mock", vad_min_silence_ms=100.0),
        diarization_provider=MockDiarizer(speaker_count=3),
    ).process_audio_file(source, tmp_path / "output", session_id="three_speakers")

    assert {segment.preliminary_speaker_id for segment in result.payload.segments} == {
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_02",
    }


def test_meet_pipeline_prefers_native_transcript_and_creates_clips(tmp_path: Path) -> None:
    samples = np.zeros(16_000, dtype=np.float32)
    time = np.arange(len(samples), dtype=np.float32) / 16_000
    samples[1_000:5_000] = 0.4 * np.sin(2 * np.pi * 240 * time[1_000:5_000])
    samples[6_000:10_000] = 0.3 * np.sin(2 * np.pi * 320 * time[6_000:10_000])
    recording = tmp_path / "meeting.wav"
    _write_wav(recording, samples)
    transcript = tmp_path / "meeting.vtt"
    transcript.write_text(
        """WEBVTT

00:00:00.050 --> 00:00:00.250
<v Alice>We should start.</v>

00:00:00.400 --> 00:00:00.650
<v Bob>I agree.</v>
""",
        encoding="utf-8",
    )

    result = DSPPipeline().process_google_meet(
        recording=recording,
        transcript=transcript,
        output_dir=tmp_path / "output",
        session_id="meet_001",
    )

    assert [segment.preliminary_speaker_id for segment in result.payload.segments] == [
        "spk_alice",
        "spk_bob",
    ]
    assert result.payload.speaker_source == "meet_transcript"
    assert result.to_stt_transcript_output().turns[0].text == "We should start."


def test_transcript_only_meet_input_reaches_native_ai_contract(tmp_path: Path) -> None:
    transcript = tmp_path / "meeting.json"
    transcript.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "entryId": "e1",
                        "speakerName": "Alice",
                        "startTime": 0,
                        "endTime": 2,
                        "text": "Welcome",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = DSPPipeline().process_google_meet(
        transcript=transcript,
        session_id="meet_transcript_only",
    )

    assert result.payload.duration_seconds == 2.0
    assert result.payload.segments[0].clip_path == ""
    assert result.to_stt_transcript_output().turns[0].speaker_id == "spk_alice"
