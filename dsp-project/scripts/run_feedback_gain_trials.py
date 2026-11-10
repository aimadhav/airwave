"""Run trusted local A/B gain trials against a frozen listening-review snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from overlap_lab.config import LabConfig
from overlap_lab.evaluation_config import DetectionEvaluationConfig
from overlap_lab.exceptions import LabInputError
from overlap_lab.models import ClipSpec
from recording_viewer.recovery_config import RecoveryConfig


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    """New comparison artifacts only; never rewrite audio or user references."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--recovery-policy", type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    policy_path = args.recovery_policy.resolve()
    if not policy_path.is_relative_to(ROOT / "configs"):
        raise LabInputError("Experiment policy must be an operator-owned local config")
    policy = RecoveryConfig.load(policy_path)
    if policy.separator_gain_mode != "shared_peak":
        raise LabInputError("This experiment requires the explicit shared-peak gain policy")
    lab = LabConfig.load(ROOT / "configs/overlap_lab.json")
    evaluation = DetectionEvaluationConfig(**{**snapshot["evaluation_policy"],
        "minimum_durations": tuple(snapshot["evaluation_policy"]["minimum_durations"]),
        "contexts": tuple(snapshot["evaluation_policy"]["contexts"])})
    storage = (ROOT / lab.storage_directory / "runs").resolve()
    if not storage.is_relative_to(ROOT / "data"):
        raise LabInputError("Trials must stay in private data storage")
    signature = hashlib.sha256(json.dumps({"reviews": snapshot["reviews_sha256"],
        "files": {str(p.relative_to(ROOT)): file_hash(p) for p in (
            policy_path, ROOT / "scripts/run_overlap_lab.py", ROOT / "src/recording_viewer/services/source_gain.py",
            ROOT / "src/recording_viewer/adapters/local_separator.py", ROOT / policy.embedding_model,
            ROOT / "models/stt/faster-whisper-base.en/model.bin", ROOT / policy.separator_directory / "manifest.json")}}, sort_keys=True).encode()).hexdigest()
    baseline_inputs: dict[str, tuple[float, Path]] = {}
    completed: dict[tuple[str, float], dict[str, object]] = {}
    for result_path in storage.glob("*/result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["recording_id"] != snapshot["recording_id"] or result["source_hash"] != snapshot["reviews"]["source_hash"]:
                continue
            clip = result["request"]["reference"]["clip"]
            human = snapshot["reviews"]["items"].get(clip["id"])
            if not human or ClipSpec.from_mapping(clip) != ClipSpec.from_mapping(human["clip"]):
                continue
            if result.get("feedback_experiment", {}).get("signature") == signature:
                completed[(clip["id"], result["request"]["context_seconds"])] = result
            if result.get("separator_gain_mode", "raw") == "raw" and result["request"]["context_seconds"] == evaluation.contexts[0]:
                stamp = result_path.stat().st_mtime
                if clip["id"] not in baseline_inputs or stamp > baseline_inputs[clip["id"]][0]:
                    baseline_inputs[clip["id"]] = stamp, result_path.parent / "input.json"
        except (OSError, ValueError, KeyError, TypeError):
            continue
    cases = sorted(snapshot["reviews"]["items"].values(), key=lambda r: (r["verdict"] != "two_overlapping", r["clip"]["start"]))
    if any(r["clip"]["id"] not in baseline_inputs for r in cases):
        raise LabInputError("Complete the raw baseline for every frozen review before A/B trials")
    for context in evaluation.contexts:
        for review in cases:
            if context != evaluation.contexts[0] and review["verdict"] != "two_overlapping":
                continue  # larger-context positive-only diagnostic, not a measured negative-control test
            clip = review["clip"]
            if (clip["id"], context) in completed:
                print(f"Reused completed gain trial {clip['id']} context={context}", flush=True)
                continue
            payload = json.loads(baseline_inputs[clip["id"]][1].read_text(encoding="utf-8"))
            payload["request"]["context_seconds"] = context
            payload["request"]["variants"] = ["sepformer--base"]
            payload["request"]["reference"]["reviewed"] = False
            payload["request"]["reference"]["texts"] = {speaker: "" for speaker in clip["speakers"]}
            payload["lab_config"]["recovery_config"] = str(policy_path.relative_to(ROOT))
            job_id = uuid.uuid4().hex
            folder = storage / job_id
            folder.mkdir()
            input_path, output_path = folder / "input.json", folder / "result.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            print(f"Gain trial started {clip['id']} context={context}, job={job_id}", flush=True)
            started = time.monotonic()
            with (folder / "processing.log").open("w", encoding="utf-8") as log:
                subprocess.run([str(ROOT / lab.python_executable), str(ROOT / "scripts/run_overlap_lab.py"),
                    "--input", str(input_path), "--output", str(output_path)], stdout=log, stderr=subprocess.STDOUT,
                    cwd=ROOT, check=True, timeout=lab.job_timeout_seconds, shell=False)
            result = json.loads(output_path.read_text(encoding="utf-8"))
            result["feedback_experiment"] = {"signature": signature, "reviews_sha256": snapshot["reviews_sha256"],
                "human_verdict": review["verdict"], "elapsed_seconds": time.monotonic()-started,
                "notice": "No reviewed word reference. Human activity labels are evaluation metadata, not used by inference."}
            temporary = output_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(output_path)
            variant = next(v for v in result["variants"] if v["kind"] == "separated")
            print(json.dumps({"clip": clip["id"], "human": review["verdict"], "context": context, "job_id": job_id,
                "status": variant["status"], "reason": variant["reason"], "raw_peak": result["separator_raw_peak"],
                "gain": result["separator_applied_gain"], "output_peak": result["separator_output_peak"],
                "tracks": variant["tracks"]}), flush=True)


if __name__ == "__main__":
    main()
