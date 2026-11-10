"""Offline excerpt integrity tests; no corpus download or inference required."""

from pathlib import Path
from dataclasses import replace
import wave

import pytest

from dsp.evaluation import TimedSpeaker
from scripts.prepare_ami_excerpt import AMIExcerptConfig, ROOT, clip_reference, crop_wav


def test_reference_shift_clamp_and_overlap() -> None:
    """Keep overlap and discard outside regions without changing identities."""
    result = clip_reference([
        TimedSpeaker(8, 12, "a"), TimedSpeaker(11, 17, "b"),
        TimedSpeaker(1, 2, "c"), TimedSpeaker(20, 21, "d"),
    ], 10, 5)
    assert result == [TimedSpeaker(0, 2, "a"), TimedSpeaker(1, 5, "b")]


def test_pcm_crop_is_exact_and_does_not_overwrite(tmp_path: Path) -> None:
    """The output contains original frames, not generated/repeated content."""
    source, target = tmp_path / "source.wav", tmp_path / "excerpt.wav"
    pcm = bytes(range(100))
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(1)
        writer.setframerate(10)
        writer.writeframes(pcm)
    result = crop_wav(source, target, 2, 3)
    assert result["duration_seconds"] == 3
    with wave.open(str(target), "rb") as reader:
        assert reader.readframes(30) == pcm[20:50]
    with pytest.raises(FileExistsError):
        crop_wav(source, target, 2, 3)


def test_truncated_prefix_cannot_become_a_valid_excerpt(tmp_path: Path) -> None:
    """Reject unavailable frames even when the header advertises a long file."""
    source, target = tmp_path / "prefix.part", tmp_path / "excerpt.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(1)
        writer.setframerate(10)
        writer.writeframes(bytes(range(100)))
    source.write_bytes(source.read_bytes()[:-50])
    with pytest.raises(ValueError, match="complete requested range"):
        crop_wav(source, target, 6, 3)
    assert not target.exists()


@pytest.mark.parametrize("start,duration", [(-1, 2), (0, 0), (float("nan"), 2)])
def test_invalid_excerpt_interval(start: float, duration: float) -> None:
    with pytest.raises(ValueError):
        clip_reference([], start, duration)


def test_config_rejects_invalid_times_and_escaped_paths() -> None:
    """Configured artifact paths cannot escape the ignored corpus folder."""
    config = AMIExcerptConfig.load(ROOT / "configs/ami_four_speaker_test.json")
    assert config.expected_reference_speakers == 4
    with pytest.raises(ValueError, match="duration_seconds"):
        replace(config, duration_seconds=-1)
    with pytest.raises(ValueError, match="inside data/benchmarks"):
        replace(config, output_audio="../outside.wav")
    with pytest.raises(ValueError, match="positive integer"):
        replace(config, expected_reference_speakers=True)
