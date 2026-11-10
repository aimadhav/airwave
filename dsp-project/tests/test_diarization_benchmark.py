"""Offline benchmark configuration and reproducibility checks; no model downloads."""

import hashlib
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest
import yaml

import scripts.benchmark_diarization as benchmark
from dsp.exceptions import DiarizationConfigurationError


def test_fingerprint_identifies_exact_bytes(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"fixture model data")
    result = benchmark.file_fingerprint(source)
    assert result["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["size_bytes"] == source.stat().st_size


def test_embedding_override_does_not_change_config_file(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump({"dsp": {"local_model": {
        "embedding_model": "baseline.onnx", "cluster_threshold": 0.9,
    }}}), encoding="utf-8")
    before = source.read_bytes()
    config = benchmark.load_benchmark_config(source, "local_model", -1, 0.7, Path("candidate.onnx"))
    assert config.local_model.embedding_model == "candidate.onnx"
    assert config.local_model.cluster_threshold == 0.7
    assert config.local_model.num_speakers == -1
    assert source.read_bytes() == before
    baseline = benchmark.load_benchmark_config(source, "single_speaker", -1, None, None)
    assert baseline.local_model.embedding_model == "baseline.onnx"
    assert baseline.local_model.cluster_threshold == 0.9


def test_invalid_experiment_config_fails_fast(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    source.write_text("dsp: {}", encoding="utf-8")
    with pytest.raises(DiarizationConfigurationError):
        benchmark.load_benchmark_config(source, "local_model", 0, 0.9, None)


def test_single_speaker_benchmark_offline_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rate = 16000
    samples = (0.3 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)).astype(np.float32)
    source = tmp_path / "speech.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        writer.writeframes((samples * 32767).astype("<i2").tobytes())
    config = tmp_path / "config.yaml"
    config.write_text("dsp: {}", encoding="utf-8")
    arguments = ["benchmark", str(source), "--provider", "single_speaker", "--config", str(config),
                 "--tag", "offline_fixture", "--output", str(tmp_path / "results")]
    monkeypatch.setattr(sys, "argv", arguments)
    benchmark.main()
    report = json.loads((tmp_path / "results/offline_fixture/report.json").read_text(encoding="utf-8"))
    assert report["clip_validation_errors"] == []
    assert report["fingerprints"]["input"]["sha256"] == benchmark.file_fingerprint(source)["sha256"]
    assert "evaluation" not in report  # No invented accuracy when reference labels are absent.
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


@pytest.mark.parametrize("tag", ["../escape", "nested/path", "nested\\path", "..", "bad."])
def test_experiment_tag_cannot_escape_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tag: str,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("dsp: {}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["benchmark", "unused.wav", "--config", str(config), "--tag", tag])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
