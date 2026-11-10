"""Run a predeclared local comparison without tuning on the check recordings."""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def validated_name(value: object) -> str:
    """Names are identifiers, never user-controlled output paths."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        raise ValueError("Case/profile names must be simple identifiers")
    return value


def select_native_profile(rows: list[dict[str, Any]], native_names: set[str]) -> str:
    """Use development DER only; deterministic tie-break by profile name."""
    candidates = [row for row in rows if row["profile"] in native_names]
    if not candidates or any(row["der"] is None for row in candidates):
        raise ValueError("Native selection requires scored development candidates")
    return str(min(candidates, key=lambda row: (row["der"], row["profile"]))["profile"])


def run_comparison(manifest_path: Path) -> dict[str, Any]:
    """Invoke the existing benchmark in isolated subprocesses, then summarize."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timeout = manifest["benchmark_timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("benchmark_timeout_seconds must be finite and positive")
    output = (ROOT / manifest["output"]).resolve()
    if not output.is_relative_to((ROOT / "data/benchmark_results").resolve()):
        raise ValueError("Comparison output must stay within data/benchmark_results")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {output}")
    profiles = {validated_name(row["name"]): row["config"] for row in manifest["profiles"]}
    if len(profiles) != len(manifest["profiles"]):
        raise ValueError("Duplicate profile names")
    baseline, native_baseline = manifest["baseline"], manifest["native_baseline"]
    if baseline not in profiles or native_baseline not in profiles or baseline == native_baseline:
        raise ValueError("Two distinct declared baselines are required")
    cases = [manifest["development"], *manifest["checks"]]
    case_names = [validated_name(row["name"]) for row in cases]
    if len(set(case_names)) != len(case_names):
        raise ValueError("Duplicate recording names")
    for row in cases:
        for key in ("audio", "reference"):
            if not (ROOT / row[key]).is_file():
                raise FileNotFoundError(row[key])
    for filename in profiles.values():
        if not (ROOT / filename).is_file():
            raise FileNotFoundError(filename)
    rows: list[dict[str, Any]] = []

    def evaluate(case: dict[str, str], profile: str) -> None:
        tag = f"{case['name']}_{profile}"
        command = [sys.executable, str(ROOT / "scripts/benchmark_diarization.py"), case["audio"],
                   "--reference", case["reference"], "--config", profiles[profile],
                   "--tag", tag, "--output", str(output)]
        try:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                       check=False, timeout=timeout, shell=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Comparison timed out for {tag} after {timeout}s") from error
        if completed.returncode:
            raise RuntimeError(f"Comparison failed for {tag}: {completed.stderr[-2000:]}")
        report = json.loads((output / tag / "report.json").read_text(encoding="utf-8"))
        row = {"case": case["name"], "profile": profile,
               "der": report["evaluation"]["0.0"]["der"],
               "collared_der": report["evaluation"]["0.25"]["der"],
               "speakers": len(report["speakers"]), "seconds": report["elapsed_seconds"],
               "miss_seconds": report["evaluation"]["0.0"]["miss_seconds"],
               "confusion_seconds": report["evaluation"]["0.0"]["confusion_seconds"],
               "clip_validation_errors": report["clip_validation_errors"]}
        rows.append(row)
        print(json.dumps(row), flush=True)

    for profile in profiles:
        evaluate(manifest["development"], profile)
    chosen = select_native_profile(rows, set(profiles) - {baseline})
    for case in manifest["checks"]:
        for profile in dict.fromkeys([baseline, native_baseline, chosen]):
            evaluate(case, profile)
    summary = {"selection_rule": manifest["selection_rule"], "chosen_native_profile": chosen,
               "production_defaults_changed": False, "results": rows}
    (output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """Use an explicit, versioned comparison manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    run_comparison(args.manifest)


if __name__ == "__main__":
    main()
