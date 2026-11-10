"""Mock-first regressions for separator output entering normalized-PCM consumers."""

from dataclasses import replace
from pathlib import Path
import subprocess

import numpy as np
import pytest

from recording_viewer.adapters.local_separator import LocalSpeechSeparator
from recording_viewer.exceptions import RecoveryError
from recording_viewer.models.recovery import SeparationOutput
from recording_viewer.recovery_config import RecoveryConfig
from recording_viewer.services.source_gain import shared_peak_gain


def test_one_gain_preserves_phase_relative_levels_and_inputs() -> None:
    loud = np.array([-40., 20., 0.], dtype=np.float32)
    quiet = np.array([2., -1., 0.], dtype=np.float32)
    originals = loud.copy(), quiet.copy()
    tracks, gain, peak = shared_peak_gain((loud, quiet), .98)
    assert peak == 40 and gain == pytest.approx(.98 / 40)
    assert max(np.max(np.abs(t)) for t in tracks) == pytest.approx(.98)
    np.testing.assert_allclose(tracks[1], -.05 * tracks[0])
    for original, source, actual in zip(originals, (loud, quiet), tracks):
        np.testing.assert_array_equal(source, original)
        assert actual.dtype == np.float32 and not np.shares_memory(source, actual)


@pytest.mark.parametrize("amplitude", [0., 1e-8, .4, .98])
def test_quiet_tracks_and_silence_are_never_amplified(amplitude: float) -> None:
    source = np.array([amplitude, -amplitude], dtype=np.float32)
    tracks, gain, _ = shared_peak_gain((source, source * .01), 1.)
    assert gain == 1
    np.testing.assert_array_equal(tracks[0], source)


@pytest.mark.parametrize("invalid", [0., -1., 1.01, float("nan"), float("inf"), True, "0.9"])
def test_invalid_peak_limit_is_a_domain_error(invalid: object) -> None:
    with pytest.raises(RecoveryError):
        shared_peak_gain((np.ones(2),), invalid)


@pytest.mark.parametrize("tracks", [(), (np.array([]),), (np.ones((2, 2)),),
    (np.ones(2), np.ones(3)), (np.array([np.nan]),), (np.array([np.inf]),)])
def test_invalid_tracks_are_rejected(tracks: tuple[np.ndarray, ...]) -> None:
    with pytest.raises(RecoveryError):
        shared_peak_gain(tracks, .98)


@pytest.mark.parametrize("mode,expected_gain", [("raw", 1.), ("shared_peak", .98 / 40)])
def test_adapter_calibrates_before_consumers_without_external_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, expected_gain: float,
) -> None:
    config = replace(RecoveryConfig.load(Path("configs/overlap_recovery.json")),
                     separator_python="runtime/python.exe", separator_directory="model", separator_gain_mode=mode)
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime/python.exe").touch()
    (tmp_path / "model").mkdir()
    (tmp_path / "model/manifest.json").write_text("{}", encoding="utf-8")

    def fake_worker(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output = Path(command[command.index("--output") + 1])
        np.savez(output, track_0=np.array([40., -20.], dtype=np.float32),
                 track_1=np.array([2., -1.], dtype=np.float32))
        assert "PYTHONPATH" not in kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("recording_viewer.adapters.local_separator.subprocess.run", fake_worker)
    separator = LocalSpeechSeparator(tmp_path, config, tmp_path / "scratch")
    result = separator.separate(np.array([.2, -.1], dtype=np.float32), 16000)
    assert result.sample_rate == 16000 and result.raw_peak == 40
    assert result.applied_gain == pytest.approx(expected_gain)
    np.testing.assert_allclose(result.tracks[0], np.array([40., -20.]) * expected_gain)
    np.testing.assert_allclose(result.tracks[1], np.array([2., -1.]) * expected_gain)
    assert list((tmp_path / "scratch").iterdir()) == []


def test_contract_and_old_config_remain_backward_compatible() -> None:
    config = RecoveryConfig.load(Path("configs/overlap_recovery.json"))
    values = dict(config.__dict__)
    values.pop("separator_gain_mode")
    values.pop("separator_peak_limit")
    assert RecoveryConfig(**values).separator_gain_mode == "raw"
    output = SeparationOutput((np.ones(3),), 16000)
    assert output.applied_gain == 1 and output.raw_peak is None


@pytest.mark.parametrize("changes", [{"separator_gain_mode": "independent_boost"},
    {"separator_peak_limit": 0}, {"separator_peak_limit": True}, {"separator_peak_limit": float("nan")}])
def test_bad_gain_policy_fails_fast(changes: dict[str, object]) -> None:
    with pytest.raises(RecoveryError):
        replace(RecoveryConfig.load(Path("configs/overlap_recovery.json")), **changes)
