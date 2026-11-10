"""Selection and orchestration tests require no real models or remote services."""

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
import scripts.run_dsp_comparison as comparison


def test_selection_uses_der_not_speaker_count() -> None:
    rows = [{"profile": "native_a", "der": 0.4, "speakers": 4},
            {"profile": "native_b", "der": 0.2, "speakers": 5},
            {"profile": "sherpa", "der": 0.1, "speakers": 4}]
    assert comparison.select_native_profile(rows, {"native_a", "native_b"}) == "native_b"
    with pytest.raises(ValueError):
        comparison.select_native_profile([], {"native_a"})


@pytest.mark.parametrize("name", ["../outside", "nested/path", "bad.name", "", 3])
def test_names_cannot_be_paths(name: object) -> None:
    with pytest.raises(ValueError):
        comparison.validated_name(name)


def test_configured_comparison_and_no_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    for name in ("development.wav", "check.wav", "reference.rttm", "config.yaml"):
        (tmp_path / name).write_text("fixture", encoding="utf-8")
    manifest = tmp_path / "comparison.json"
    manifest.write_text(json.dumps({"output": "data/benchmark_results/test", "baseline": "sherpa", "benchmark_timeout_seconds": 900,
        "native_baseline": "native_default", "selection_rule": "development only",
        "development": {"name": "dev", "audio": "development.wav", "reference": "reference.rttm"},
        "checks": [{"name": "check", "audio": "check.wav", "reference": "reference.rttm"}],
        "profiles": [{"name": name, "config": "config.yaml"} for name in ("sherpa", "native_default", "native_tuned")]}), encoding="utf-8")
    calls: list[str] = []

    def fake_run(command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        assert options["timeout"] == 900 and options["shell"] is False
        tag = command[command.index("--tag") + 1]
        calls.append(tag)
        directory = Path(command[command.index("--output") + 1]) / tag
        directory.mkdir(parents=True)
        der = 0.2 if tag.endswith("native_tuned") else 0.4
        (directory / "report.json").write_text(json.dumps({"speakers": ["anonymous"], "elapsed_seconds": 1,
            "clip_validation_errors": [], "evaluation": {"0.0": {"der": der, "miss_seconds": 0,
            "confusion_seconds": 0}, "0.25": {"der": der}}}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = comparison.run_comparison(manifest)
    assert result["chosen_native_profile"] == "native_tuned"
    assert calls == ["dev_sherpa", "dev_native_default", "dev_native_tuned",
                     "check_sherpa", "check_native_default", "check_native_tuned"]
    assert result["production_defaults_changed"] is False
    with pytest.raises(FileExistsError):
        comparison.run_comparison(manifest)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), True])
def test_bad_timeout_is_rejected(tmp_path: Path, timeout: object) -> None:
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"benchmark_timeout_seconds": timeout}), encoding="utf-8")
    with pytest.raises(ValueError, match="finite and positive"):
        comparison.run_comparison(manifest)
