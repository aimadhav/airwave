"""Controlled separation check with known sources from exclusive speech regions.

This is NOT a human transcription benchmark: isolated-ASR comparisons are silver
references. SI-SDR uses the actual known source signals, not ASR confidence.
"""

import argparse
from dataclasses import asdict
from itertools import permutations
import json
from pathlib import Path
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "src", ROOT / "models/asr-deps", ROOT / "models/media-deps", ROOT / ".runtime-deps"):
    sys.path.insert(0, str(dependency))
import numpy as np
from recording_viewer.adapters.faster_whisper import FasterWhisperTranscriber
from recording_viewer.adapters.local_separator import LocalSpeechSeparator
from recording_viewer.adapters.voice_embedder import OnnxVoiceEmbedder
from recording_viewer.config import RecordingViewerConfig
from recording_viewer.models import VoiceRegion
from recording_viewer.recovery_config import RecoveryConfig
from recording_viewer.services.overlap_recovery import OverlapRecovery
from recording_viewer.services.recovery_timeline import activity_spans


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant SDR in dB; reference has to be an actual isolated source."""
    estimate = estimate.astype(np.float64) - float(estimate.mean())
    reference = reference.astype(np.float64) - float(reference.mean())
    epsilon = np.finfo(float).eps
    target = np.dot(estimate, reference) / (np.dot(reference, reference) + epsilon) * reference
    return float(10 * np.log10((np.dot(target, target) + epsilon) / (np.sum((estimate - target) ** 2) + epsilon)))


def main() -> None:
    """Use all qualifying pairs, report failures; never cherry-pick a winner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, default=ROOT / "data/listening-room/sample-podcast-overlap")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/recording_viewer_overlap_demo.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/overlap-evaluation")
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--max-pairs", type=int, default=3)
    args = parser.parse_args()
    if args.clip_seconds <= 0 or args.max_pairs < 1 or not args.output.resolve().is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Invalid bounded evaluation settings")
    config = RecordingViewerConfig.load(args.config)
    policy = RecoveryConfig.load(ROOT / config.recovery_config)
    args.output.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.recording / "processing.wav"), "rb") as recording:
        rate = recording.getframerate()
        samples = np.frombuffer(recording.readframes(recording.getnframes()), dtype="<i2").astype(np.float32) / 32768
    rows = json.loads((args.recording / "speaker-timelines.json").read_text())[policy.backend]
    spans = activity_spans([VoiceRegion(**row) for row in rows], len(samples) / rate)
    candidates: dict[str, tuple[float, float]] = {}
    for start, end, speakers in sorted(spans, key=lambda row: -(row[1] - row[0])):
        if len(speakers) == 1 and end - start >= args.clip_seconds + 2 * policy.reference_guard_seconds:
            candidates.setdefault(speakers[0], (start + policy.reference_guard_seconds, end - policy.reference_guard_seconds))
    speakers = sorted(candidates)
    separator = LocalSpeechSeparator(ROOT, policy, args.output / "scratch")
    recognizer = FasterWhisperTranscriber(config)
    embedder = OnnxVoiceEmbedder(policy)
    results: list[dict[str, object]] = []
    for first_index, first in enumerate(speakers):
        for second in speakers[first_index + 1:]:
            if len(results) >= args.max_pairs:
                break
            sources = [samples[round(candidates[s][0]*rate):round(candidates[s][0]*rate)+round(args.clip_seconds*rate)].copy()
                       for s in (first, second)]
            mixture = (sources[0] + sources[1]) / 2
            started = time.monotonic()
            output = separator.separate(mixture, rate)
            scores = [[si_sdr(track, source) for source in sources] for track in output.tracks]
            assignment = max(permutations(range(2)), key=lambda p: sum(scores[i][j] for i, j in enumerate(p)))
            baselines = [si_sdr(mixture, source) for source in sources]
            improvements = [scores[i][j] - baselines[j] for i, j in enumerate(assignment)]
            isolated = [[asdict(word) for word in recognizer.transcribe_samples(source, rate)] for source in sources]
            mixed = [asdict(word) for word in recognizer.transcribe_samples(mixture, rate)]
            separated = [[asdict(word) for word in recognizer.transcribe_samples(track, rate)] for track in output.tracks]
            # A separate session includes isolated enrollment material, a gap,
            # then the mixture. This tests the complete identity/confidence gate.
            n, gap = len(mixture), rate
            test_audio = np.concatenate((sources[0], np.zeros(gap), sources[1], np.zeros(gap), mixture)).astype(np.float32)
            core_start = (2*n + 2*gap) / rate
            test_regions = [VoiceRegion("ref_a", first, 0, n/rate), VoiceRegion("ref_b", second, (n+gap)/rate, (2*n+gap)/rate),
                            VoiceRegion("mix_a", first, core_start, core_start+n/rate), VoiceRegion("mix_b", second, core_start, core_start+n/rate)]
            recovery = OverlapRecovery(policy, separator, embedder, recognizer).recover(test_audio, rate, test_regions)
            result = {"speakers": [first, second], "source_starts": [candidates[s][0] for s in (first, second)],
                "duration_seconds": args.clip_seconds, "si_sdr_db": scores, "track_to_source": assignment,
                "mixture_si_sdr_db": baselines, "si_sdr_improvement_db": improvements,
                "mean_si_sdr_improvement_db": float(np.mean(improvements)), "isolated_asr_silver_only": isolated,
                "mixture_asr": mixed, "separated_asr": separated, "recovery": asdict(recovery),
                "elapsed_seconds": time.monotonic()-started}
            results.append(result)
            np.savez(args.output / f"pair_{len(results):02d}.npz", source_0=sources[0], source_1=sources[1],
                     mixture=mixture, track_0=output.tracks[0], track_1=output.tracks[1])
            print(f"Pair {first}/{second}: SI-SDR improvement {np.mean(improvements):.2f} dB; recovery {[r.status+':'+r.reason for r in recovery.reports]}", flush=True)
            (args.output / "results.json").write_text(json.dumps({"limitations": "Synthetic overlaps of diarizer-selected exclusive clips. No human reference transcripts; no real-overlap WER claim.", "pairs": results}, indent=2), encoding="utf-8")
    if not results:
        raise ValueError("No sufficiently long exclusive source pairs available")


if __name__ == "__main__":
    main()
