import wave
from pathlib import Path

from ai_layer.audio_ingestion.audio_segmenter import AudioSegmenter
from ai_layer.audio_ingestion.sample_generator import generate_synthetic_acoustic_wav


def test_sample_generator_and_segmenter(tmp_path: Path):
    wav_path = tmp_path / "sample_test.wav"
    output_clips_dir = tmp_path / "clips"

    # Generate synthetic wav
    result_path = generate_synthetic_acoustic_wav(wav_path, sample_rate=16000)
    assert result_path.exists()
    assert result_path.stat().st_size > 0

    # Verify wave properties
    with wave.open(str(result_path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        duration = handle.getnframes() / handle.getframerate()
        assert duration > 5.0

    # Segment the wave
    segmenter = AudioSegmenter()
    clips = segmenter.segment_wav(
        wav_path=result_path,
        output_dir=output_clips_dir,
        session_id="session_test",
    )

    assert len(clips) >= 2
    for clip in clips:
        assert Path(clip.audio_path).exists()
        assert clip.time_range.start_time_seconds < clip.time_range.end_time_seconds
        assert clip.preliminary_speaker_id.startswith("Speaker_")
