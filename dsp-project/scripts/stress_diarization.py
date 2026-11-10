"""Generate transparent robustness probes from real downloaded recordings.

The repeat test is a synthetic long-session stress test, not a new real meeting
or independent accuracy dataset. Noise/gain/rate tests retain original RTTM.
"""

import json
from pathlib import Path
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if (ROOT / ".runtime-deps").is_dir():
    sys.path.insert(0, str(ROOT / ".runtime-deps"))

import numpy as np
from scipy.signal import resample_poly


def read_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2 or source.getnchannels() != 1:
            raise ValueError("Probe source must be mono PCM16")
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768, source.getframerate()


def write_pcm(path: Path, samples: np.ndarray, rate: int) -> None:
    values = samples if samples.ndim == 2 else samples[:, None]
    with wave.open(str(path), "wb") as target:
        target.setnchannels(values.shape[1])
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes((np.clip(values, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    config = json.loads((ROOT / "configs/diarization_stress.json").read_text(encoding="utf-8"))
    samples, rate = read_pcm(ROOT / config["annotated_audio"])
    folder = ROOT / "data/benchmarks/stress"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["noise_seed"])
    noise = rng.standard_normal(len(samples)).astype(np.float32)
    noise *= np.sqrt(np.mean(samples**2)) / (10 ** (config["noise_snr_db"] / 20) * np.sqrt(np.mean(noise**2)))
    noisy = samples + noise
    # Avoid introducing hard clipping into the noise experiment.
    noisy /= max(1.0, float(np.max(np.abs(noisy))))
    variants = {
        "quiet": (samples * config["quiet_gain"], rate),
        "noise_15db": (noisy, rate),
        "stereo_48k": (np.column_stack([resample_poly(samples, 3, 1)] * 2), config["resample_rate"]),
    }
    repeat, repeat_rate = read_pcm(ROOT / config["repeat_audio"])
    variants["repeated_four_speakers"] = (np.tile(repeat, config["repeat_count"]), repeat_rate)
    results: list[dict[str, object]] = []
    for name, (values, variant_rate) in variants.items():
        path = folder / f"{name}.wav"
        write_pcm(path, values, variant_rate)
        command = [sys.executable, str(ROOT / "scripts/benchmark_diarization.py"), str(path)]
        if name != "repeated_four_speakers":
            command += ["--reference", config["annotated_reference"]]
        outcome = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        result = json.loads(outcome.stdout)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    (ROOT / "data/benchmark_results/stress_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
