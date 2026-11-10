"""Freeze human labels and report offline detection/recovery experiments."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from overlap_lab.adapters.file_studies import FileStudyRepository
from overlap_lab.adapters.listening_store import FileListeningReviewRepository
from overlap_lab.config import LabConfig
from overlap_lab.evaluation_config import DetectionEvaluationConfig
from overlap_lab.exceptions import LabInputError
from overlap_lab.models import ClipSpec
from overlap_lab.models.listening import ListeningReview
from overlap_lab.services.detection_evaluation import evaluate_duration_filter


def document_hash(document: object) -> str:
    """Canonical content fingerprint, independent of JSON whitespace."""
    return hashlib.sha256(json.dumps(document, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write generated private experiment artifacts, never reference input files."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def clock(seconds: float) -> str:
    return f"{int(seconds//60)}:{seconds%60:05.2f}"


def collect_results(storage: Path, recording_id: str, snapshot: dict[str, object],
                    policy: DetectionEvaluationConfig) -> list[dict[str, object]]:
    """Read completed matching local jobs; do not modify or rerun them."""
    found: dict[tuple[str, float, str], tuple[float, dict[str, object]]] = {}
    reviews = snapshot["reviews"]["items"]
    for path in storage.glob("*/result.json"):
        if not re.fullmatch(r"[a-f0-9]{32}", path.parent.name):
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            request = result["request"]
            clip = request["reference"]["clip"]
            review = reviews.get(clip["id"])
            if (result["recording_id"] != recording_id or result["source_hash"] != snapshot["reviews"]["source_hash"]
                    or review is None or ClipSpec.from_mapping(clip) != ClipSpec.from_mapping(review["clip"])
                    or request["context_seconds"] not in policy.contexts):
                continue
            variant = next((v for v in result["variants"] if v["id"] == "sepformer--base"), None)
            if variant is None:
                continue
            row = {"clip_id": clip["id"], "start": clip["start"], "end": clip["end"],
                   "human_verdict": review["verdict"], "context_seconds": request["context_seconds"],
                   "job_id": path.parent.name, "status": variant["status"], "reason": variant["reason"],
                   "metrics": variant["metrics"], "tracks": variant["tracks"], "speakers": variant["speakers"],
                   "scores": variant["scores"], "separation_seconds": result["separation_seconds"],
                   "recognition_matching_seconds": variant["elapsed_seconds"],
                   "model_sha256": result["model_sha256"], "separator_revision": result["separator_revision"],
                   "gain_mode": result.get("separator_gain_mode", "raw"),
                   "raw_peak": result.get("separator_raw_peak"), "applied_gain": result.get("separator_applied_gain"),
                   "output_peak": result.get("separator_output_peak"), "worker_sha256": result.get("worker_sha256"),
                   "experiment_signature": result.get("feedback_experiment", {}).get("signature")}
            key, modified = (clip["id"], request["context_seconds"], row["gain_mode"]), path.stat().st_mtime
            if key not in found or modified > found[key][0]:
                found[key] = modified, row
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted((row for _, row in found.values()), key=lambda row: (row["gain_mode"], row["context_seconds"], row["start"]))


def summarize_recovery(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Separate true-overlap recovery from incorrectly accepted negative controls."""
    groups: dict[tuple[str, float], list[dict[str, object]]] = {}
    seen: set[tuple[str, float, str]] = set()
    for row in results:
        identity = row["clip_id"], row["context_seconds"], row["gain_mode"]
        if identity in seen:
            raise LabInputError("Duplicate recovery comparison case")
        seen.add(identity)
        groups.setdefault((row["gain_mode"], row["context_seconds"]), []).append(row)
    summaries = []
    for (mode, context), rows in sorted(groups.items()):
        positive = [r for r in rows if r["human_verdict"] == "two_overlapping"]
        negative = [r for r in rows if r["human_verdict"] in {"one_speaker", "two_sequential", "no_speech"}]
        summaries.append({"gain_mode": mode, "context_seconds": context,
            "confirmed_overlaps_tested": len(positive),
            "confirmed_overlaps_accepted": sum(r["status"] == "accepted" for r in positive),
            "negative_controls_tested": len(negative),
            "negative_controls_incorrectly_accepted": sum(r["status"] == "accepted" for r in negative),
            "failed_runs": sum(r["status"] == "failed" for r in rows)})
    return summaries


def markdown_report(snapshot: dict[str, object], evaluation: dict[str, object],
                    results: list[dict[str, object]], base_url: str,
                    native_report: dict[str, object] | None = None) -> str:
    """Keep model outputs and human listening labels visibly distinct."""
    recording_id = snapshot["recording_id"]
    lines = ["# Listening-feedback experiment", "", f"Recording: {snapshot['filename']}",
        f"Human listening revision: **{snapshot['reviews']['revision']}**; snapshot SHA-256: `{snapshot['reviews_sha256']}`.", "",
        "Human labels are frozen here for evaluation; original listening and word annotations were not modified. Historical runs remain unchanged. See the engineering document for any separately promoted gain fix.", "",
        "## Detection-only duration ablation", "",
        f"Candidate duration selected on calibration data only: **{evaluation['selected_minimum_seconds']} seconds**.",
        f"Calibration cases: {len(evaluation['calibration_ids'])}; held-out cases: {len(evaluation['held_out_ids'])}.", "",
        "| Evaluation set | Rule | True overlaps retained | False overlap alarms | Correct rejections | Missed labelled overlaps |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for label, name in (("Held out", "held_out"), ("All reviewed (descriptive)", "all_reviewed")):
        for version in ("baseline", "selected"):
            counts = evaluation[f"{name}_{version}"]
            lines.append(f"| {label} | {version} | {counts['true_positive']} | {counts['false_positive']} | {counts['true_negative']} | {counts['false_negative']} |")
    lines += ["", "This is an **experimental duration-only prioritization rule**, not an acoustic overlap detector, and it has NOT been applied to the recording. It can miss real short interruptions. A convenient 12-case sample does not establish recording-wide accuracy or recall.", "",
        "## Real separation and recognition runs", "",
        "These are automatic gate decisions. A rejected human-confirmed overlap is not thereby a false overlap; an accepted negative control is not thereby true simultaneous speech. No verified word transcripts were supplied for these runs, so WER is unavailable.", "",
        "| Gain mode | Context | Confirmed overlaps passing gates | Negative controls incorrectly passing | Failed runs |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in summarize_recovery(results):
        negatives = (f"{row['negative_controls_incorrectly_accepted']}/{row['negative_controls_tested']}"
                     if row["negative_controls_tested"] else "not tested")
        lines.append(f"| {row['gain_mode']} | {row['context_seconds']} s | {row['confirmed_overlaps_accepted']}/{row['confirmed_overlaps_tested']} | {negatives} | {row['failed_runs']} |")
    lines += ["", "Passing gates is NOT verified transcription accuracy. Each pair of gain modes must be compared at the same context. Larger-context positive-only runs do not establish safety on negative controls.", "",
        "| Interval | Human label | Gain mode | Context | Gate | Reason | Local job |", "| --- | --- | --- | ---: | --- | --- | --- |"]
    for row in results:
        lines.append(f"| {clock(row['start'])}–{clock(row['end'])} | {row['human_verdict']} | {row['gain_mode']} | {row['context_seconds']} s | {row['status']} | {row['reason']} | `{row['job_id']}` |")
    lines += ["", f"[Open saved comparisons]({base_url.rstrip('/')}/lab/?recording={recording_id})", "",
              "Choose the matching time in **Inspect results → Comparison history**. The most recent configured number of jobs is shown; older files are not deleted.", "",
              "## Machine track text on human-confirmed overlaps", ""]
    for row in results:
        if row["human_verdict"] == "two_overlapping":
            lines += [f"### {clock(row['start'])}, {row['gain_mode']}, {row['context_seconds']} s context: {row['status']}", ""]
            for track in row["tracks"]:
                speaker = track.get("assigned_speaker") or "unassigned"
                lines.append(f"- Track {track['index']+1} ({speaker}): {json.dumps(track['text'], ensure_ascii=False)}")
                lines.append(f"  [Listen]({base_url.rstrip('/')}/lab-audio/{row['job_id']}/{track['audio']})")
            lines.append("")
    if native_report is not None:
        counts = native_report["counts"]
        lines += ["## Independent short-context native recheck", "",
            f"Fixed local native profile: {counts['true_positive']} confirmed overlaps retained, {counts['false_positive']} false overlap alarms, {counts['true_negative']} correct rejections, {counts['false_negative']} missed labelled overlaps.", "",
            "This diagnostic changes activity thresholds and removes padding, not the speech separator. It is experimental and was NOT applied. The output JSON records the model hash, complete command, configuration and case-level concurrent durations.", ""]
    lines += ["## Limits", "", *[f"- {item}" for item in evaluation["limitations"]], "",
              "Any larger-context positive-only run is a diagnostic: it has not been checked on negative controls at that context and must not be promoted solely because more tracks pass.", "",
              "The JSON artifacts preserve labels, split, thresholds, model fingerprints, gate metrics and run IDs.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--review-revision", type=int, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/listening_feedback_evaluation.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--native-recheck-report", type=Path)
    args = parser.parse_args()
    policy = DetectionEvaluationConfig.load(args.config)
    lab = LabConfig.load(ROOT / "configs/overlap_lab.json")
    upload = json.loads((ROOT / "configs/recording_upload.json").read_text(encoding="utf-8"))
    viewer = json.loads((ROOT / lab.viewer_config).read_text(encoding="utf-8"))
    storage = ROOT / lab.storage_directory
    studies = FileStudyRepository(ROOT / viewer["output_directory"], ROOT / upload["upload_directory"], storage / "references", lab)
    reviews = FileListeningReviewRepository(studies, storage / "listening-reviews", lab).reviews(args.recording)
    study = studies.study(args.recording)
    if reviews["revision"] != args.review_revision:
        raise LabInputError("Listening labels changed; choose a new explicit experiment revision")
    output = (ROOT / policy.output_directory / args.recording / f"revision_{args.review_revision}").resolve()
    if not output.is_relative_to((ROOT / "data").resolve()):
        raise LabInputError("Feedback reports must remain in private local data storage")
    output.mkdir(parents=True, exist_ok=True)
    identity = output / "snapshot.json"
    snapshot = {"schema_version": 1, "recording_id": args.recording, "filename": study["filename"],
                "created_utc": datetime.now(timezone.utc).isoformat(), "reviews": reviews,
                "reviews_sha256": document_hash(reviews), "evaluation_policy": asdict(policy),
                "policy_sha256": document_hash(asdict(policy))}
    if identity.exists():
        existing = json.loads(identity.read_text(encoding="utf-8"))
        if any(existing[key] != snapshot[key] for key in ("reviews_sha256", "policy_sha256", "recording_id")):
            raise LabInputError("Existing experiment has different labels or policy; do not overwrite its snapshot")
        snapshot = existing
    else:
        write_json(identity, snapshot)
    evaluation = evaluate_duration_filter([ListeningReview.from_mapping(r) for r in reviews["items"].values()], policy)
    write_json(output / "duration-evaluation.json", evaluation)
    print(json.dumps({"output": str(output), "threshold_seconds": evaluation["selected_minimum_seconds"],
                      "held_out": evaluation["held_out_selected"], "all_reviewed": evaluation["all_reviewed_selected"]}), flush=True)
    if not args.prepare_only:
        results = collect_results(storage / "runs", args.recording, snapshot, policy)
        write_json(output / "separation-runs.json", results)
        write_json(output / "recovery-summary.json", summarize_recovery(results))
        native_report = None
        if args.native_recheck_report:
            native_path = args.native_recheck_report.resolve()
            if not native_path.is_relative_to(output):
                raise LabInputError("Native report must belong to this frozen experiment")
            native_report = json.loads(native_path.read_text(encoding="utf-8"))
            if native_report["provenance"]["labels"] != snapshot["reviews_sha256"]:
                raise LabInputError("Native recheck refers to different human labels")
        (output / "REPORT.md").write_text(markdown_report(snapshot, evaluation, results, upload["local_app_url"], native_report), encoding="utf-8")
        print(f"Report includes {len(results)} matching completed local experiments", flush=True)


if __name__ == "__main__":
    main()
