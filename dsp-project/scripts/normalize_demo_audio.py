"""Make a louder playback copy without modifying the benchmark recording."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if (ROOT / ".runtime-deps").is_dir():
    sys.path.insert(0, str(ROOT / ".runtime-deps"))

import numpy as np
from dsp.ingestion.audio_loader import load_wav
from dsp.preprocessing.normalizer import normalize_peak, to_mono


def normalize_copy(source: Path, output: Path, target_peak: float) -> dict[str, object]:
    """Preserve timing; apply DC removal and one whole-recording gain only."""
    if not math.isfinite(target_peak) or not 0 < target_peak <= 1:
        raise ValueError("Target peak must be finite and in (0, 1]")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Existing playback artifacts must not be overwritten")
    audio = to_mono(load_wav(source))
    original_peak = float(np.max(np.abs(audio.samples), initial=0))
    normalized = normalize_peak(audio, target_peak)
    pcm = np.rint(np.clip(normalized.samples, -1, 1) * 32767).astype("<i2")
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as writer:
        writer.setparams((1, 2, audio.metadata.sample_rate, len(pcm), "NONE", "not compressed"))
        writer.writeframes(pcm.tobytes())
    report: dict[str, object] = {
        "source": str(source), "output": str(output), "duration_seconds": audio.duration_seconds,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_peak_dbfs": 20 * math.log10(original_peak) if original_peak else None,
        "target_peak_dbfs": 20 * math.log10(target_peak),
        "operation": "DC removal plus global peak normalization; no editing, denoising or separation",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Read explicit input, output and gain target from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--peak", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize_copy(args.source, args.output, args.peak), indent=2))


if __name__ == "__main__":
    main()
