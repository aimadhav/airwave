"""Local profile enrollment/persistence is mock-first and needs no real models."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from recording_viewer.adapters.mock_recovery import MockVoiceEmbedder
from recording_viewer.adapters.voice_profile_store import JsonVoiceProfileStore, MemoryVoiceProfileStore
from recording_viewer.exceptions import RecoveryError, VoiceProfileError
from recording_viewer.models.voice_profiles import ProfileContext, VoiceProfileSet
from recording_viewer.profile_config import VoiceProfileExportConfig
from recording_viewer.recovery_config import RecoveryConfig
from recording_viewer.services.recovery_quality import match_tracks
from recording_viewer.services.voice_profiles import enroll_profiles
from scripts.save_voice_profiles import save_existing


def context() -> ProfileContext:
    """Distinct deterministic synthetic hashes; no user audio is needed."""
    return ProfileContext("a"*64, "b"*64, "c"*64, "d"*64, "e"*64,
                          "nemo_native", 16000, 8000, "titanet_polyphase_bandlimit_v1")


def bundle() -> VoiceProfileSet:
    """Two separable synthetic speakers with two reference vectors each."""
    return enroll_profiles(np.ones(16000*8, dtype=np.float32), context(),
        {"SPEAKER_00": [(0, 2), (2, 4)], "SPEAKER_01": [(4, 6), (6, 8)]},
        MockVoiceEmbedder([np.array(v) for v in ([2., 0.], [4., 0.], [0., 3.], [0., 9.])]), 2)


def test_enrollment_saves_each_reference_and_unit_centroids() -> None:
    result = bundle()
    assert result.profiles[0].centroid == (1., 0.)
    assert result.profiles[1].centroid == (0., 1.)
    assert [len(p.references) for p in result.profiles] == [2, 2]
    assert result.profiles[1].references[0].start == 4
    assert all(p.identity_status == "unverified" for p in result.profiles)
    assert VoiceProfileSet.from_document(json.loads(json.dumps(result.to_document()))) == result


@pytest.mark.parametrize("kind", ["memory", "json"])
def test_repository_round_trip_and_overwrite_refusal(tmp_path: Path, kind: str) -> None:
    store = MemoryVoiceProfileStore() if kind == "memory" else JsonVoiceProfileStore(tmp_path)
    value = bundle()
    key = store.save(value)
    assert key == value.context.key and store.load(key) == value
    before = (tmp_path / f"{key}.json").read_bytes() if kind == "json" else b""
    with pytest.raises(VoiceProfileError, match="already exist"):
        store.save(replace(value, created_at="2026-10-26T00:00:00+00:00"))
    assert store.load(key) == value
    if kind == "json":
        assert (tmp_path / f"{key}.json").read_bytes() == before
        assert not list(tmp_path.glob("*.tmp"))


def test_saved_vectors_work_with_existing_recovery_matcher(tmp_path: Path) -> None:
    store = JsonVoiceProfileStore(tmp_path)
    loaded = store.load(store.save(bundle()))
    loaded.require_compatible("d"*64, 16000, 8000, "titanet_polyphase_bandlimit_v1")
    profiles = {p.speaker_id: np.asarray(p.centroid) for p in loaded.profiles}
    assignment, _ = match_tracks([np.array([0., 1.]), np.array([1., 0.])],
        ("SPEAKER_00", "SPEAKER_01"), profiles, RecoveryConfig.load(Path("configs/overlap_recovery.json")))
    assert assignment == ("SPEAKER_01", "SPEAKER_00")


@pytest.mark.parametrize("encoder,rate,band,version", [
    ("f"*64, 16000, 8000, "titanet_polyphase_bandlimit_v1"),
    ("d"*64, 8000, 8000, "titanet_polyphase_bandlimit_v1"),
    ("d"*64, 16000, 16000, "titanet_polyphase_bandlimit_v1"),
    ("d"*64, 16000, 8000, "different_preprocessing"),
])
def test_incompatible_features_are_rejected(encoder: str, rate: int, band: int, version: str) -> None:
    with pytest.raises(VoiceProfileError, match="mismatch"):
        bundle().require_compatible(encoder, rate, band, version)


def test_versioned_keys_prevent_cross_recording_speaker_number_collisions() -> None:
    original = context()
    assert original.key != replace(original, source_sha256="f"*64).key
    assert original.key != replace(original, timeline_sha256="f"*64).key
    assert original.key != replace(original, bandlimit_rate=16000).key
    assert original.key != replace(original, encoder_sha256="f"*64).key


@pytest.mark.parametrize("vector", [[0., 0.], [float("nan"), 1.], [float("inf"), 1.]])
def test_invalid_provider_vectors_never_reach_storage(vector: list[float]) -> None:
    with pytest.raises(VoiceProfileError):
        enroll_profiles(np.ones(32000), context(), {"SPEAKER_00": [(0, 1), (1, 2)]},
                        MockVoiceEmbedder([np.array(vector), np.array(vector)]), 2)


def test_inconsistent_reference_dimensions_are_typed_errors() -> None:
    with pytest.raises(VoiceProfileError, match="dimensions"):
        enroll_profiles(np.ones(32000), context(), {"SPEAKER_00": [(0, 1), (1, 2)]},
                        MockVoiceEmbedder([np.ones(2), np.ones(3)]), 2)


def test_no_clean_reference_does_not_create_a_fake_profile() -> None:
    with pytest.raises(VoiceProfileError, match="No speaker"):
        enroll_profiles(np.ones(16000), context(), {}, MockVoiceEmbedder([]), 2)
    with pytest.raises(VoiceProfileError, match="outside"):
        enroll_profiles(np.ones(16000), context(), {"SPEAKER_00": [(0, 2)]}, MockVoiceEmbedder([]), 2)


def test_reference_provider_failures_are_explicit_and_can_leave_partial_profiles() -> None:
    class FailureThenSuccess:
        """A failed clean reference must not be treated as a valid voice vector."""

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
            self.calls += 1
            if self.calls == 1:
                raise RecoveryError("Synthetic provider failure")
            return np.array([1., 0.])

    result = enroll_profiles(np.ones(64000), context(),
        {"SPEAKER_00": [(0, 1), (1, 2)], "SPEAKER_01": [(2, 3), (3, 4)]}, FailureThenSuccess(), 2)
    assert [p.speaker_id for p in result.profiles] == ["SPEAKER_01"]
    assert len(result.warnings) == 2


@pytest.mark.parametrize("changes", [{"schema_version": 2}, {"schema_version": True}, {"profiles": []},
    {"created_at": "not-a-date"}, {"created_at": "2026-10-26T00:00:00"}, {"unexpected": "field"}])
def test_malformed_saved_documents_fail_fast(changes: dict[str, object]) -> None:
    document = bundle().to_document()
    document.update(changes)
    with pytest.raises(VoiceProfileError):
        VoiceProfileSet.from_document(document)


def test_store_rejects_path_traversal_and_corruption(tmp_path: Path) -> None:
    store = JsonVoiceProfileStore(tmp_path)
    with pytest.raises(VoiceProfileError):
        store.load("../outside")
    (tmp_path / ("f"*64 + ".json")).write_text(json.dumps(bundle().to_document()))
    with pytest.raises(VoiceProfileError, match="context"):
        store.load("f"*64)
    (tmp_path / ("a"*64 + ".json")).write_text("bad-json")
    with pytest.raises(VoiceProfileError):
        store.load("a"*64)


def test_export_rejects_public_output_before_any_read_or_write(tmp_path: Path) -> None:
    config = VoiceProfileExportConfig("data/listening-room/fixture", "demos/public-profiles", "policy.json",
                                     "titanet_polyphase_bandlimit_v1")
    with pytest.raises(VoiceProfileError, match="private"):
        save_existing(tmp_path, config)
    assert not (tmp_path / "demos").exists()
