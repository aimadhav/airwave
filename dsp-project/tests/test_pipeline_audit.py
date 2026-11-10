"""Evidence serialization and educational tools require no live models or APIs."""

import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from recording_viewer.exceptions import RecordingViewerError
from scripts.export_podcast_audit import build_audit

ROOT = Path(__file__).resolve().parents[1]


def fixtures() -> tuple[dict, dict, dict, dict]:
    """Minimal synthetic data, including deliberately forbidden private fields."""
    provenance = {"duration_seconds": 10, "source_sha256": "a"*64, "words_recognized": 4,
        "configuration": {"model_repository": "fixture", "source": "C:/private/secret.wav", "token": "secret-token"}}
    recovery = {"policy": {"context_seconds": 1, "separator_python": "C:/private/runtime"},
        "result": {"reports": [{"window": {"start": 2, "end": 3, "crop_start": 1, "crop_end": 4,
            "speakers": ["SPEAKER_00", "SPEAKER_01"]}, "status": "rejected", "reason": "ambiguous_speaker_match",
            "metrics": {"track_0_cosine": .2, "private_path": "secret-path", "track_1_cosine": float("nan")}}]}}
    current = {"separator_gain_mode": "shared_peak", "separator_python": "secret-path"}
    facts = {"verified_on": "2026-10-18", "code_baseline": "fixture", "runtime_facts": {}, "primary_sources": []}
    return provenance, recovery, current, facts


def test_audit_allowlists_paths_secrets_and_metrics_without_mutating_inputs() -> None:
    values = fixtures()
    before = copy.deepcopy(values)
    result = build_audit(*values)
    text = json.dumps(result, allow_nan=False)
    assert "private" not in text and "secret" not in text and "separator_python" not in text
    assert result["counts"] == {"rejected": 1}
    assert result["last_attempt_end"] == 3
    assert result["reports"][0]["metrics"] == {"track_0_cosine": .2}
    assert values[0] == before[0] and values[2:] == before[2:]
    assert "separator_gain_mode" not in result["saved_recovery_policy"]
    assert result["current_recovery_policy"]["separator_gain_mode"] == "shared_peak"


@pytest.mark.parametrize("changes", [{"start": -1}, {"end": 15}, {"crop_start": 2.5},
    {"crop_end": 2.8}, {"start": float("nan")}, {"speakers": ["Real person's name"]}])
def test_invalid_saved_intervals_are_rejected(changes: dict[str, object]) -> None:
    values = fixtures()
    values[1]["result"]["reports"][0]["window"].update(changes)
    with pytest.raises(RecordingViewerError):
        build_audit(*values)


def test_provider_failure_does_not_export_raw_private_error() -> None:
    values = fixtures()
    values[1]["result"]["reports"][0].update(status="failed", reason="C:/private/key: secret-token")
    assert build_audit(*values)["reports"][0]["reason"] == "provider_failure"


def test_public_audit_matches_saved_demo_and_exposes_coverage_limit() -> None:
    audit = json.loads((ROOT / "web/recording-viewer/pipeline-audit.json").read_text(encoding="utf-8"))
    demo = json.loads((ROOT / "demos/podcast-57min/results.json").read_text(encoding="utf-8"))
    assert audit["source_sha256"] == demo["source_sha256"]
    assert len(audit["reports"]) == 470 and sum(audit["counts"].values()) == 470
    assert audit["reason_counts"]["window_budget_exceeded"] == 287
    assert audit["reason_counts"]["ambiguous_speaker_match"] == 32
    assert audit["last_attempt_end"] == 378.389
    f = audit["runtime_facts"]
    assert f["native_chunk_frames"]*f["native_coarse_frame_seconds"] == 21.12
    assert audit["snapshot_state"] == "historical_before_shared_peak_gain_fix_not_rerun"


def test_pipeline_diagrams_filters_and_calculator_with_node() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js needed for the isolated explanation-tool test")
    subprocess.run([node, "--test", str(ROOT / "tests/browser/pipeline-notebook.test.cjs")],
                   capture_output=True, check=True, text=True, timeout=30)


def test_walkthrough_has_accessible_diagrams_and_no_active_backend() -> None:
    page = (ROOT / "web/recording-viewer/pipeline.html").read_text(encoding="utf-8")
    script = (ROOT / "web/recording-viewer/pipeline.js").read_text(encoding="utf-8")
    assert 'role="img"' in page and '<desc id="flow-desc">' in page
    for chapter in ("chunks", "vad", "separation", "gates", "audit", "fixes", "evidence"):
        assert f'id="{chapter}"' in page
    assert "fetch(" not in script and "innerHTML" not in script
    assert "not a live model dashboard" in page and "not a rerun" in page
