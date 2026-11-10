"""Offline, mock-first acceptance and adversarial tests for overlap recovery."""

from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from recording_viewer.adapters.mock_recovery import MockArrayTranscriber, MockSpeechSeparator, MockVoiceEmbedder
from recording_viewer.exceptions import RecoveryError
from recording_viewer.models import TranscriptWord, VoiceRegion
from recording_viewer.models.recovery import AttributedWord, RecoveryResult, WindowReport
from recording_viewer.recovery_config import RecoveryConfig
from recording_viewer.services.overlap_recovery import OverlapRecovery
from recording_viewer.services.recovery_quality import duplicate_text, match_tracks, signal_checks, unit
from recording_viewer.services.recovery_timeline import activity_spans, plan_windows, reference_spans, splice_recovery


@pytest.fixture
def policy() -> RecoveryConfig:
    return RecoveryConfig.load(Path("configs/overlap_recovery.json"))


def regions() -> list[VoiceRegion]:
    return [VoiceRegion("a", "A", 0, 6), VoiceRegion("b", "B", 5, 12)]


def signals(length: int) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(length) / 16000
    return (.2 * np.sin(2 * np.pi * 201 * time)).astype(np.float32), (.15 * np.sin(2 * np.pi * 341 * time)).astype(np.float32)


def engine(policy: RecoveryConfig, words: list[list[TranscriptWord]] | None = None,
           vectors: list[np.ndarray] | None = None) -> tuple[OverlapRecovery, np.ndarray]:
    left, right = signals(3 * 16000)  # overlap core 5–6; context crop 4–7
    service = OverlapRecovery(policy, MockSpeechSeparator((left, right)),
        MockVoiceEmbedder(vectors or [np.array(v) for v in ([1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 0.], [0., 1.])]),
        MockArrayTranscriber(words or [[TranscriptWord(1.1, 1.4, "Hello", .9)], [TranscriptWord(1.2, 1.7, " World", .9)]]))
    samples = np.zeros(12 * 16000, dtype=np.float32)
    samples[4 * 16000:7 * 16000] = left + right
    return service, samples


def test_full_mock_pipeline_preserves_both_simultaneous_transcripts(policy: RecoveryConfig) -> None:
    service, samples = engine(policy)
    result = service.recover(samples, 16000, regions())
    assert result.reference_counts == {"A": 2, "B": 2}
    report = result.reports[0]
    assert report.status == "accepted"
    assert [w.speaker for w in report.words] == ["A", "B"]
    assert report.words[0].word.start == 5.1
    baseline = [TranscriptWord(1, 2, "Before"), TranscriptWord(5.1, 5.4, " mixed"), TranscriptWord(8, 9, " After")]
    turns = splice_recovery(baseline, regions(), result, .5)
    assert "mixed" not in " ".join(t.text for t in turns)
    assert sum(t.word_count for t in turns) == 4
    separated = [t for t in turns if t.transcript_source == "separated_asr"]
    assert len(separated) == 2 and all(t.overlap for t in separated)
    assert {t.text for t in separated} == {"Hello", "World"}


def test_permutation_matching_does_not_trust_track_order(policy: RecoveryConfig) -> None:
    assignment, _ = match_tracks([np.array([0., 1.]), np.array([1., 0.])], ("A", "B"),
        {"A": np.array([1., 0.]), "B": np.array([0., 1.])}, policy)
    assert assignment == ("B", "A")


def test_same_speaker_duplicate_regions_are_not_overlap(policy: RecoveryConfig) -> None:
    rows = [VoiceRegion("a", "A", 0, 5), VoiceRegion("b", "A", 2, 6)]
    assert plan_windows(rows, 6, policy) == []
    assert activity_spans(rows, 6) == [(0, 6, ("A",))]


def test_long_overlaps_have_disjoint_cores_and_bounded_context(policy: RecoveryConfig) -> None:
    rows = [VoiceRegion("a", "A", 0, 21), VoiceRegion("b", "B", 0, 21)]
    windows = plan_windows(rows, 21, policy)
    assert len(windows) == 4
    assert all(a.end == b.start for a, b in zip(windows, windows[1:]))
    assert all(w.end - w.start <= policy.max_core_seconds for w in windows)
    assert windows[0].crop_start == 0 and windows[-1].crop_end == 21


def test_references_exclude_overlap_and_guard_boundaries(policy: RecoveryConfig) -> None:
    spans = reference_spans(regions(), 12, policy)
    assert all(end < 5 for _, end in spans["A"])
    assert all(start > 6 for start, _ in spans["B"])


@pytest.mark.parametrize("words,reason", [
    ([[TranscriptWord(1.1, 1.4, "a", .1)], [TranscriptWord(1.1, 1.4, "b", .9)]], "low_recognition_confidence"),
    ([[TranscriptWord(1.1, 1.4, "a")], [TranscriptWord(1.1, 1.4, "b", .9)]], "missing_recognition_confidence"),
    ([[TranscriptWord(1.1, 1.4, "a", .9)], []], "no_secondary_words_recovered"),
    ([[TranscriptWord(3.1, 3.4, "a", .9)], []], "recognition_outside_crop"),
    ([[TranscriptWord(1.1, 1.4, " yes we can", .9)], [TranscriptWord(1.1, 1.4, " yes we can", .9)]], "duplicated_transcripts"),
])
def test_recognition_quality_rejections_preserve_baseline(policy: RecoveryConfig, words: list[list[TranscriptWord]], reason: str) -> None:
    service, samples = engine(policy, words)
    result = service.recover(samples, 16000, regions())
    assert result.reports[0].reason == reason
    assert result.reports[0].status == "rejected"
    baseline = [TranscriptWord(5.1, 5.4, " keep this")]
    assert sum(t.word_count for t in splice_recovery(baseline, regions(), result, .5)) == 1


def test_no_clean_reference_does_not_invent_identity(policy: RecoveryConfig) -> None:
    service, samples = engine(policy)
    result = service.recover(samples, 16000, [VoiceRegion("a", "A", 0, 12), VoiceRegion("b", "B", 0, 12)])
    assert all(r.reason == "insufficient_clean_references" for r in result.reports)


def test_three_speakers_never_forced_into_two_tracks(policy: RecoveryConfig) -> None:
    service, samples = engine(policy)
    result = service.recover(samples, 16000, [VoiceRegion(str(i), str(i), 0, 12) for i in range(3)])
    assert all(r.reason == "unsupported_simultaneous_speaker_count" for r in result.reports)


def test_silent_duplicated_and_invalid_tracks_rejected(policy: RecoveryConfig) -> None:
    a, b = signals(16000)
    assert signal_checks(a+b, (a, np.zeros_like(a)), policy)[0] == "silent_source"
    assert signal_checks(a+b, (a, a), policy)[0] == "duplicated_sources"
    assert signal_checks(a+b, (a[:-1], b), policy)[0] == "invalid_source_shape"
    assert signal_checks(a+b, (a*np.nan, b), policy)[0] == "invalid_source_shape"
    assert signal_checks(a+b, (a, b), policy)[0] == ""


def test_ambiguous_embeddings_rejected(policy: RecoveryConfig) -> None:
    assignment, _ = match_tracks([np.ones(2), np.ones(2)], ("A", "B"), {"A": np.array([1., 0.]), "B": np.array([0., 1.])}, policy)
    assert assignment is None
    with pytest.raises(RecoveryError):
        unit(np.zeros(2))


def test_provider_failure_is_audited_and_not_promoted(policy: RecoveryConfig) -> None:
    service, samples = engine(policy)

    class Failure:
        def separate(self, samples: np.ndarray, sample_rate: int) -> None:
            raise RecoveryError("fixture timeout")

    service.separator = Failure()
    assert service.recover(samples, 16000, regions()).reports[0].status == "failed"


def test_disabled_and_budget_are_explicit(policy: RecoveryConfig) -> None:
    service, samples = engine(replace(policy, enabled=False))
    assert service.recover(samples, 16000, regions()).reports[0].reason == "disabled"
    service, samples = engine(replace(policy, max_windows=1, max_core_seconds=.5))
    assert service.recover(samples, 16000, regions()).reports[-1].reason == "window_budget_exceeded"


@pytest.mark.parametrize("field,value", [("max_windows", 0), ("minimum_cosine", float("nan")), ("enabled", "yes"),
    ("minimum_word_probability", -1), ("minimum_references", 100), ("separator_timeout_seconds", float("inf"))])
def test_invalid_configuration_fails_fast(policy: RecoveryConfig, field: str, value: object) -> None:
    with pytest.raises(RecoveryError):
        replace(policy, **{field: value})


def test_touching_boundaries_and_out_of_bounds_timeline(policy: RecoveryConfig) -> None:
    assert plan_windows([VoiceRegion("a", "A", 0, 2), VoiceRegion("b", "B", 2, 4)], 4, policy) == []
    with pytest.raises(RecoveryError):
        plan_windows(regions(), 10, policy)


def test_short_backchannels_are_not_rejected_as_duplicate_text(policy: RecoveryConfig) -> None:
    assert not duplicate_text("yes", "yes", policy)


def test_third_speaker_is_excluded_from_inference_context(policy: RecoveryConfig) -> None:
    rows = [VoiceRegion("a", "A", 2, 5), VoiceRegion("b", "B", 3, 4),
            VoiceRegion("c", "C", 0, 2.8), VoiceRegion("d", "D", 4.2, 6)]
    target = next(w for w in plan_windows(rows, 6, policy) if w.speakers == ("A", "B"))
    assert target.crop_start == 2.8 and target.crop_end == 4.2


def test_core_midpoint_owns_boundary_words_exactly_once(policy: RecoveryConfig) -> None:
    window = plan_windows(regions(), 12, policy)[0]
    recovered = AttributedWord("A", TranscriptWord(4.8, 5.4, " boundary", .9), window.id, .8)
    result = RecoveryResult((WindowReport(window, "accepted", "fixture", {}, (recovered,)),), {})
    turns = splice_recovery([TranscriptWord(4.8, 5.4, " baseline")], regions(), result, .5)
    assert sum(t.word_count for t in turns) == 1
    row = next(t for t in turns if t.transcript_source == "separated_asr" and t.word_count)
    assert row.start == 4.8  # Do not hide recognized words before the ownership core.


def test_overlapping_recovery_cores_are_invalid(policy: RecoveryConfig) -> None:
    window = plan_windows(regions(), 12, policy)[0]
    result = RecoveryResult((WindowReport(window, "accepted", "fixture", {}), WindowReport(window, "accepted", "fixture", {})), {})
    with pytest.raises(RecoveryError):
        splice_recovery([], regions(), result, .5)


def test_embedding_dimension_changes_fail_explicitly(policy: RecoveryConfig) -> None:
    with pytest.raises(RecoveryError, match="dimensions"):
        match_tracks([np.ones(3), np.ones(3)], ("A", "B"), {"A": np.ones(2), "B": np.ones(2)}, policy)


def test_short_skipped_windows_do_not_consume_inference_budget(policy: RecoveryConfig) -> None:
    service, samples = engine(replace(policy, max_windows=1))
    rows = [*regions(), VoiceRegion("tiny", "B", 4.8, 4.81)]
    result = service.recover(samples, 16000, rows)
    assert result.reports[0].reason == "overlap_too_short"
    assert result.reports[-1].status == "accepted"
