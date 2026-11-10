"""Prepare the checked-in NOTSOFAR CSS sample as mono audio plus RTTM activity labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / "Desktop" / "audiodataset" / "NOTSOFAR1-Challenge" / "sample_data" / "css_train_set"
DEFAULT_OUTPUT = ROOT / "data/benchmarks/notsofar1-desktop-20261109"


def sha256(path: Path) -> str:
    """Return a content hash for provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write or validate deterministic PCM16 output without changing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(values), initial=0.0))
    if not np.isfinite(values).all() or peak == 0:
        raise ValueError("Sample has non-finite values or is silent")
    values = values / max(peak, 1.0)
    frames = np.rint(values * 32767).astype("<i2").tobytes()
    if path.exists():
        with wave.open(str(path), "rb") as existing:
            if (existing.getnchannels() != 1 or existing.getsampwidth() != 2
                    or existing.getframerate() != sample_rate or existing.readframes(existing.getnframes()) != frames):
                raise ValueError(f"Existing generated audio differs from current source: {path}")
        return
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)


def activity_rttm(activity: np.ndarray, path: Path, recording: str) -> int:
    """Encode contiguous score==1 frames; score 0 is transitional/borderline, not speech."""
    rows = 0
    lines: list[str] = []
    for speaker_index in range(activity.shape[1]):
        active = activity[:, speaker_index] == 1
        padded = np.pad(active.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends):
            if end > start:
                lines.append(
                    f"SPEAKER {recording} 1 {start / 16000:.6f} "
                    f"{(end - start) / 16000:.6f} <NA> <NA> speaker_{speaker_index + 1} <NA> <NA>\n"
                )
                rows += 1
    content = "".join(lines)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing generated reference differs from current source: {path}")
    else:
        path.write_text(content, encoding="utf-8", newline="\n")
    return rows


def prepare(source: Path, output: Path) -> dict[str, object]:
    """Validate source shapes, write channel-0/mean inputs and speaker activity RTTM."""
    stem = "0000_Libri.clean-500.book_00082_chp_0009_reader_02124_53"
    metadata_path = source / f"{stem}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    columns = metadata["columns"]
    samples, channels = columns["mixture"]["shape"]
    frames, speakers = columns["gt_spk_activity_scores"]["shape"]
    if samples != frames or samples != 160000 or channels != 7 or speakers != 3:
        raise ValueError("Unexpected NOTSOFAR sample dimensions")
    mixture_path = source / f"{stem}.mixture"
    activity_path = source / f"{stem}.gt_spk_activity_scores"
    mixture = np.fromfile(mixture_path, dtype="<i2").reshape(samples, channels)
    activity = np.fromfile(activity_path, dtype=np.int8).reshape(frames, speakers)
    if not set(np.unique(activity)).issubset({-1, 0, 1}):
        raise ValueError("Unexpected NOTSOFAR activity score values")

    input_dir = output / "inputs"
    ref_path = output / "notsofar1_css_sample.rttm"
    # The official sample stores 16 kHz samples and a per-row scale factor.
    scale = float(columns["mixture_scale"]["values"])
    normalized = mixture.astype(np.float64) / scale
    audio_paths = {
        "channel0": input_dir / "notsofar1_css_channel0.wav",
        "seven_channel_mean": input_dir / "notsofar1_css_seven_channel_mean.wav",
    }
    write_pcm16(audio_paths["channel0"], normalized[:, 0], 16000)
    write_pcm16(audio_paths["seven_channel_mean"], normalized.mean(axis=1), 16000)
    interval_count = activity_rttm(activity, ref_path, "notsofar1_css_sample")
    manifest = {
        "dataset": "NOTSOFAR-1 simulated CSS training sample",
        "source_sample": stem,
        "source_rate_hz": 16000,
        "source_duration_seconds": samples / 16000,
        "source_channels": channels,
        "reference_speaker_slots": speakers,
        "active_reference_speakers": int(np.count_nonzero(np.any(activity == 1, axis=0))),
        "reference_activity_values": {"-1": "not speaking", "0": "borderline transition excluded", "1": "speaking"},
        "reference_interval_count": interval_count,
        "reference_file": str(ref_path.relative_to(ROOT)),
        "source_files": {
            "metadata_sha256": sha256(metadata_path),
            "mixture_sha256": sha256(mixture_path),
            "activity_sha256": sha256(activity_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for label, path in audio_paths.items()
        },
        "scoring_note": "A 10-second simulated speech-mixture sample. This evaluates speaker activity/diarization only; no human transcript is supplied, so WER/tcpWER cannot be measured.",
    }
    (output / "notsofar1-source-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"Prepared {samples / 16000:.1f}s, {channels}-channel source; {interval_count} speaker intervals.", flush=True)
    print(f"Reference: {ref_path}", flush=True)
    return manifest


def main() -> None:
    """Parse paths and prepare the sample without modifying its source dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
