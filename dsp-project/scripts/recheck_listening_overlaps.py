"""Fixed short-context native recheck experiment; never modifies the recording."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
for name in ("src", "models/asr-deps", ".runtime-deps"):
    sys.path.insert(0, str(ROOT / name))
from dsp.diarization.adapters.nemo_native_backend import parse_native_segments
from dsp.diarization.native_config import NativeDiarizationConfig
from dsp.diarization.interfaces.inference import ModelSpeakerInterval
from overlap_lab.exceptions import LabInputError
from overlap_lab.models import ClipSpec
from overlap_lab.models.listening import ListeningReview
from overlap_lab.services.detection_evaluation import count_predictions, detection_cases
from recording_viewer.models import VoiceRegion
from recording_viewer.services.recovery_timeline import activity_spans


def native_regions(intervals: list[ModelSpeakerInterval], duration: float) -> list[VoiceRegion]:
    """Map the canonical inference DTO, not assumed provider-specific field names."""
    if not math.isfinite(duration) or duration <= 0:
        raise LabInputError("Native clip duration must be finite and positive")
    return [VoiceRegion(str(i), str(row.cluster), row.start, min(row.end, duration))
            for i, row in enumerate(intervals) if row.start < duration]


def main() -> None:
    """Use a fixed preset, without selecting thresholds against held-out labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/listening_native_recheck.json")
    parser.add_argument("--experiment-name", default="native-recheck-r2")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.experiment_name):
        raise LabInputError("Experiment name must be a simple local folder name")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    policy = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("context_seconds", "minimum_concurrent_seconds", "timeout_seconds"):
        if isinstance(policy[key], bool) or not isinstance(policy[key], (int, float)) or not math.isfinite(policy[key]) or policy[key] <= 0:
            raise LabInputError("Invalid native recheck policy")
    defaults = json.loads((ROOT / policy["native_defaults"]).read_text(encoding="utf-8"))
    native = NativeDiarizationConfig(**{**defaults, **policy["overrides"]})
    native.validate_assets()
    reviews = [ListeningReview.from_mapping(r) for r in snapshot["reviews"]["items"].values()]
    cases, unresolved = detection_cases(reviews)
    upload = json.loads((ROOT / "configs/recording_upload.json").read_text(encoding="utf-8"))
    recording_id = snapshot["recording_id"]
    # Recording ID and exact core intervals are validated canonical contracts.
    if any(not isinstance(r.clip, ClipSpec) for r in reviews) or len(recording_id) != 32 or any(c not in "0123456789abcdef" for c in recording_id):
        raise LabInputError("Invalid frozen recording identity")
    recording = (ROOT / upload["upload_directory"] / recording_id / "result").resolve()
    if not recording.is_relative_to((ROOT / "data/listening-room").resolve()):
        raise LabInputError("Recording must stay in private local storage")
    identity = json.loads((recording / "source.json").read_text(encoding="utf-8"))
    if identity["sha256"] != snapshot["reviews"]["source_hash"]:
        raise LabInputError("Recording no longer matches the frozen labels")
    output = args.snapshot.resolve().parent / args.experiment_name
    if not output.is_relative_to((ROOT / "data").resolve()):
        raise LabInputError("Reports must stay in private local storage")
    output.mkdir(exist_ok=True)
    inputs, results = output / "clips", output / "outputs"
    inputs.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    crops: dict[str, tuple[float, float]] = {}
    with wave.open(str(recording / "processing.wav"), "rb") as source:
        rate, channels, width = source.getframerate(), source.getnchannels(), source.getsampwidth()
        if (rate, channels, width) != (16000, 1, 2):
            raise LabInputError("Recheck requires canonical PCM16 audio")
        duration = source.getnframes()/rate
        for review in reviews:
            start, end = max(0., review.clip.start-policy["context_seconds"]), min(duration, review.clip.end+policy["context_seconds"])
            if review.clip.end > duration or end-start > native.max_full_attention_seconds:
                raise LabInputError("Recheck interval exceeds native bounds")
            first, last = round(start*rate), round(end*rate)
            crops[review.clip.id] = first/rate, last/rate
            source.setpos(first)
            with wave.open(str(inputs / f"{review.clip.id}.wav"), "wb") as target:
                target.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                target.writeframes(source.readframes(last-first))
    command = [str(ROOT / native.executable), "diarize", str(inputs), "--model", str(ROOT / native.model),
               "--preset", native.preset, "--device", "cpu", "--format", "json", "--output-dir", str(results), "--offline"]
    for name, flag in (("onset", "--onset"), ("offset", "--offset"), ("pad_onset_seconds", "--pad-onset"),
                       ("pad_offset_seconds", "--pad-offset"), ("min_duration_on", "--min-duration-on"), ("min_duration_off", "--min-duration-off")):
        value = getattr(native, name)
        if value is not None:
            command += [flag, str(value)]
    with (ROOT / native.model).open("rb") as model_file:
        model_hash = hashlib.file_digest(model_file, "sha256").hexdigest()
    provenance = {"policy": policy, "native_config": asdict(native), "labels": snapshot["reviews_sha256"],
                  "model_sha256": model_hash, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "command": command}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    manifest = output / "result.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["signature"] != fingerprint:
            raise LabInputError("Existing recheck policy differs; preserve it and use a new experiment folder")
        print(json.dumps(saved), flush=True)
        return
    if list(results.rglob("*.json")):
        raise LabInputError("Unmanifested outputs already exist; use a fresh experiment name to preserve provenance")
    print(f"Rechecking {len(reviews)} short contextual clips with one shared native model", flush=True)
    started = time.monotonic()
    with (output / "processing.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=policy["timeout_seconds"], shell=False)
    rows: list[dict[str, object]] = []
    decisions: dict[str, bool] = {}
    for review in reviews:
        matches = [p for p in results.rglob("*.json") if p.name in {f"{review.clip.id}.json", f"{review.clip.id}.wav.json"}]
        if len(matches) != 1:
            raise LabInputError(f"Native recheck output missing or ambiguous for {review.clip.id}")
        intervals = parse_native_segments(matches[0].read_text(encoding="utf-8"))
        offset, end = crops[review.clip.id]
        regions = native_regions(intervals, end-offset)
        seconds = sum(max(0., min(b+offset, review.clip.end)-max(a+offset, review.clip.start))
                      for a,b,speakers in activity_spans(regions, end-offset) if len(speakers) > 1)
        predicted = seconds >= policy["minimum_concurrent_seconds"]
        if review.verdict not in {"unsure", "other_voices"}:
            decisions[review.clip.id] = predicted
        rows.append({"clip_id": review.clip.id, "human_verdict": review.verdict,
                     "overlap_seconds_in_core": seconds, "predicted_overlap": predicted})
    report = {"signature": fingerprint, "provenance": provenance, "policy": policy, "counts": asdict(count_predictions(cases, decisions)),
              "cases": rows, "excluded_unresolved": unresolved, "elapsed_seconds": time.monotonic()-started,
              "experimental_not_applied": True, "note": "Fixed diagnostic on a tiny non-blinded detector-positive sample; no full-recording accuracy or recall claim."}
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
