"""Prepare one continuous AMI excerpt and aligned human-derived RTTM.

No inference or model-based selection happens here. A downloaded WAV prefix is
acceptable only if it contains every sample of the requested interval. The
prefix hash must never be advertised as the hash of the complete source WAV.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsp.evaluation import TimedSpeaker, read_rttm


@dataclass(frozen=True)
class AMIExcerptConfig:
    """Typed input/provenance boundary for this offline corpus preparation."""

    meeting_id: str
    audio_url: str
    reference_url: str
    source_audio: str
    source_reference: str
    output_audio: str
    output_reference: str
    manifest: str
    start_seconds: float
    duration_seconds: float
    expected_reference_speakers: int
    selection_reason: str
    audio_license: str
    reference_note: str
    reference_setup_url: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError("Invalid start_seconds")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("Invalid duration_seconds")
        if not isinstance(self.expected_reference_speakers, int) or isinstance(self.expected_reference_speakers, bool) or self.expected_reference_speakers < 1:
            raise ValueError("expected_reference_speakers must be a positive integer")
        for relative in (self.source_audio, self.source_reference, self.output_audio,
                         self.output_reference, self.manifest):
            if not (ROOT / relative).resolve().is_relative_to(ROOT / "data" / "benchmarks"):
                raise ValueError("Corpus preparation paths must stay inside data/benchmarks")

    @classmethod
    def load(cls, path: Path) -> "AMIExcerptConfig":
        """Reject unknown/missing configuration fields before processing files."""
        values = json.loads(path.read_text(encoding="utf-8"))
        return cls(**values)


def digest(path: Path) -> str:
    """Hash the actual local bytes, including intentionally partial downloads."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def clip_reference(
    entries: list[TimedSpeaker], start: float, duration: float,
) -> list[TimedSpeaker]:
    """Intersect reference intervals with the excerpt and reset its time origin."""
    if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0:
        raise ValueError("Excerpt start/duration must be finite and valid")
    result: list[TimedSpeaker] = []
    for entry in entries:
        left, right = max(start, entry.start), min(start + duration, entry.end)
        if right > left:
            result.append(TimedSpeaker(left - start, right - start, entry.speaker))
    return sorted(result, key=lambda row: (row.start, row.end, row.speaker))


def crop_wav(source: Path, destination: Path, start: float, duration: float) -> dict[str, Any]:
    """Copy exact PCM frames; reject incomplete ranges instead of padding silence."""
    if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0:
        raise ValueError("Invalid excerpt interval")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing excerpt: {destination}")
    with wave.open(str(source), "rb") as reader:
        rate, channels, width = reader.getframerate(), reader.getnchannels(), reader.getsampwidth()
        first, count = round(start * rate), round(duration * rate)
        if first + count > reader.getnframes():
            raise ValueError("Requested range exceeds the WAV header duration")
        reader.setpos(first)
        pcm = reader.readframes(count)
        if len(pcm) != count * channels * width:
            raise ValueError("Downloaded prefix does not contain the complete requested range")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(pcm)
    return {"sample_rate": rate, "channels": channels, "sample_width": width,
            "frames": count, "duration_seconds": count / rate}


def main() -> None:
    """Use externally configured source paths, provenance and excerpt times."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ami_four_speaker_test.json")
    args = parser.parse_args()
    config = AMIExcerptConfig.load(args.config)
    start, duration = config.start_seconds, config.duration_seconds
    source, reference = ROOT / config.source_audio, ROOT / config.source_reference
    output, output_reference = ROOT / config.output_audio, ROOT / config.output_reference
    manifest_path = ROOT / config.manifest
    entries = clip_reference(read_rttm(reference), start, duration)
    speaker_time: dict[str, float] = {}
    for row in entries:
        speaker_time[row.speaker] = speaker_time.get(row.speaker, 0.0) + row.end - row.start
    if len(speaker_time) != config.expected_reference_speakers:
        raise ValueError(f"Excerpt has {len(speaker_time)} reference speakers, not the configured count")
    if output_reference.exists() or manifest_path.exists():
        raise FileExistsError("Existing excerpt metadata must not be overwritten")
    audio_info = crop_wav(source, output, start, duration)
    output_reference.write_text("\n".join(
        f"SPEAKER {output.stem} 1 {row.start:.6f} {row.end-row.start:.6f} <NA> <NA> {row.speaker} <NA> <NA>"
        for row in entries) + "\n", encoding="utf-8")
    manifest = {
        **asdict(config), "audio": audio_info, "reference_speakers": sorted(speaker_time),
        "reference_speaker_seconds": speaker_time, "reference_segments": len(entries),
        "source_download_is_prefix": source.suffix == ".part",
        "hash_scope": (
            "Local source prefix bytes, reference bytes, and complete derived excerpt; not the full source meeting."
            if source.suffix == ".part" else
            "Complete downloaded source asset, reference bytes, and complete derived excerpt; the source asset may itself be an excerpt."
        ),
        "sha256": {"downloaded_source_bytes": digest(source), "source_reference": digest(reference),
                   "excerpt_audio": digest(output), "excerpt_reference": digest(output_reference)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
