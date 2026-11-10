"""Export allowlisted, non-mutating pipeline evidence for the public walkthrough."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recording_viewer.exceptions import RecordingViewerError

POLICY_KEYS = (
    "enabled", "backend", "context_seconds", "max_core_seconds", "minimum_overlap_seconds",
    "max_windows", "reference_seconds", "reference_guard_seconds", "max_references",
    "minimum_references", "minimum_embedding_seconds", "minimum_cosine", "minimum_cosine_margin",
    "maximum_stream_correlation", "minimum_stream_rms", "minimum_energy_ratio",
    "minimum_reconstruction_db", "minimum_word_probability", "minimum_mean_probability",
    "maximum_duplicate_similarity", "minimum_duplicate_words", "separator_repository",
    "separator_revision", "separator_sample_rate", "separator_threads", "separator_timeout_seconds",
    "separator_gain_mode", "separator_peak_limit", "embedding_threads",
)
ASR_KEYS = ("model_repository", "model_revision", "device", "compute_type", "cpu_threads",
            "beam_size", "vad_filter", "condition_on_previous_text", "language",
            "minimum_word_coverage", "waveform_bins", "maximum_no_speech_probability",
            "minimum_segment_log_probability", "maximum_compression_ratio")
REASONS = {
    "ambiguous_speaker_match", "low_recognition_confidence", "no_secondary_words_recovered",
    "overlap_too_short", "quality_gates_passed_not_human_verified", "unsupported_simultaneous_speaker_count",
    "window_budget_exceeded", "insufficient_clean_references", "insufficient_voice_for_matching",
    "silent_source", "duplicated_sources", "negligible_secondary_source", "inconsistent_sources",
    "invalid_source_shape", "wrong_source_sample_rate", "recognition_outside_crop",
    "missing_recognition_confidence", "duplicated_transcripts", "disabled",
}
METRICS = {"energy_ratio", "stream_correlation", "reconstruction_db", "track_0_cosine",
           "track_1_cosine", "track_0_margin", "track_1_margin"}


def finite(value: object) -> bool:
    """Reject booleans and invalid numerical evidence."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def build_audit(provenance: dict[str, Any], recovery: dict[str, Any],
                current_policy: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """Whitelist public evidence; never copy raw local paths, logs or review stores."""
    try:
        duration = provenance["duration_seconds"]
        if not finite(duration) or duration <= 0 or not re.fullmatch(r"[a-f0-9]{64}", provenance["source_sha256"]):
            raise ValueError("Invalid source identity or duration")
        reports: list[dict[str, Any]] = []
        for row in recovery["result"]["reports"]:
            window = row["window"]
            a, b, left, right = (window[k] for k in ("start", "end", "crop_start", "crop_end"))
            if not all(finite(v) for v in (a, b, left, right)) or not 0 <= left <= a < b <= right <= duration:
                raise ValueError("Invalid saved ownership interval")
            if row["status"] not in {"accepted", "rejected", "skipped", "failed"}:
                raise ValueError("Invalid saved decision")
            speakers = window["speakers"]
            if not all(re.fullmatch(r"SPEAKER_[0-9]+", s) for s in speakers):
                raise ValueError("Unexpected speaker identifier")
            reports.append({"id": f"overlap_{len(reports):04d}", "start": a, "end": b,
                "crop_start": left, "crop_end": right, "speakers": speakers, "status": row["status"],
                "reason": row["reason"] if row["reason"] in REASONS else "provider_failure",
                "metrics": {k: v for k, v in row.get("metrics", {}).items() if k in METRICS and finite(v)}})
        attempts = [r for r in reports if r["status"] != "skipped"]
        return {"schema_version": 1, "verified_on": facts["verified_on"], "code_baseline": facts["code_baseline"],
            "source_sha256": provenance["source_sha256"], "duration_seconds": duration,
            "raw_word_tokens": provenance["words_recognized"],
            "snapshot_state": "historical_before_shared_peak_gain_fix_not_rerun",
            "asr": {k: v for k, v in provenance["configuration"].items() if k in ASR_KEYS},
            "saved_recovery_policy": {k: v for k, v in recovery["policy"].items() if k in POLICY_KEYS},
            "current_recovery_policy": {k: v for k, v in current_policy.items() if k in POLICY_KEYS},
            "runtime_facts": facts["runtime_facts"], "primary_sources": facts["primary_sources"],
            "counts": dict(Counter(r["status"] for r in reports)),
            "reason_counts": dict(Counter(r["reason"] for r in reports)),
            "last_attempt_end": max((r["end"] for r in attempts), default=0),
            "reports": reports}
    except (KeyError, TypeError, ValueError) as error:
        raise RecordingViewerError("Cannot export invalid pipeline audit evidence") from error


def export(root: Path, config_path: Path) -> Path:
    """Read frozen evidence and emit only the sanitized public artifact."""
    try:
        facts = json.loads(config_path.read_text(encoding="utf-8"))
        folder = (root / facts["recording_directory"]).resolve()
        output = (root / facts["output"]).resolve()
        if not folder.is_relative_to((root / "data/listening-room").resolve()) or not output.is_relative_to((root / "web/recording-viewer").resolve()):
            raise ValueError("Audit input/output outside the configured artifact boundaries")
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        public = json.loads((root / facts["public_demo_directory"] / "transcript.json").read_text(encoding="utf-8"))
        if public["source_sha256"] != provenance["source_sha256"]:
            raise ValueError("Audit does not belong to the public recording")
        recovery = json.loads((folder / "overlap-recovery.json").read_text(encoding="utf-8"))
        policy = json.loads((root / facts["current_recovery_policy"]).read_text(encoding="utf-8"))
        document = build_audit(provenance, recovery, policy, facts)
        document["evidence_sha256"] = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("provenance.json", "overlap-recovery.json", "speaker-timelines.json", "recognized-words.json")}
        output.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")
        return output
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RecordingViewerError("Cannot export podcast pipeline audit") from error


def main() -> None:
    """Export evidence without invoking any speech models or modifying reviews."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/podcast_audit.json")
    args = parser.parse_args()
    print(export(ROOT, args.config))


if __name__ == "__main__":
    main()
