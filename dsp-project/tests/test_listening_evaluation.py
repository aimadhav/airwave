"""Pure offline evaluation: activity labels are not transcripts or model input."""

from dataclasses import replace
from pathlib import Path

import pytest

from dsp.diarization.interfaces.inference import ModelSpeakerInterval
from overlap_lab.evaluation_config import DetectionEvaluationConfig
from overlap_lab.config import LabConfig
from overlap_lab.adapters.mock_adapter import MockExperimentProcessor
from overlap_lab.exceptions import LabInputError
from overlap_lab.models import ClipSpec, LabJob
from overlap_lab.models.detection_evaluation import DetectionCase, DetectionCounts
from overlap_lab.models.listening import ListeningReview
from overlap_lab.services.detection_evaluation import (
    count_predictions, detection_cases, detection_counts, evaluate_duration_filter, split_cases,
)
from scripts.recheck_listening_overlaps import native_regions
from scripts.evaluate_listening_feedback import summarize_recovery
from overlap_lab.services.lab_service import LabService


def policy() -> DetectionEvaluationConfig:
    return DetectionEvaluationConfig.load(Path("configs/listening_feedback_evaluation.json"))


def reviews() -> list[ListeningReview]:
    return [ListeningReview(ClipSpec(f"p{i}", 2*i, 2*i+1.4, ("A", "B")), "two_overlapping", ("A", "B"))
            for i in range(3)] + [ListeningReview(ClipSpec(f"n{i}", 20+2*i, 20+2*i+.7, ("A", "B")), "one_speaker", ("A",))
                                  for i in range(9)]


def test_labels_turn_taking_and_unresolved_are_not_conflated() -> None:
    labels = [ListeningReview(ClipSpec(str(i), i, i+1, ("A", "B")), verdict,
                             ("A", "B") if verdict in {"two_overlapping", "two_sequential"} else ())
              for i, verdict in enumerate(("two_overlapping", "two_sequential", "one_speaker", "no_speech", "unsure", "other_voices"))]
    cases, unresolved = detection_cases(labels)
    assert [c.overlapping for c in cases] == [True, False, False, False]
    assert unresolved == ["4", "5"]
    with pytest.raises(LabInputError):
        detection_cases([labels[0], labels[0]])


def test_split_is_deterministic_disjoint_stratified_and_order_independent() -> None:
    cases, _ = detection_cases(reviews())
    calibration, holdout = split_cases(cases, policy())
    assert split_cases(list(reversed(cases)), policy()) == (calibration, holdout)
    assert len(calibration) == 8 and len(holdout) == 4
    assert not {c.id for c in calibration} & {c.id for c in holdout}
    assert sum(c.overlapping for c in calibration) == 2 and sum(c.overlapping for c in holdout) == 1


def test_duration_rule_uses_calibration_only_and_does_not_change_labels() -> None:
    labels = reviews()
    original = list(labels)
    result = evaluate_duration_filter(labels, policy())
    assert labels == original and result["experimental_not_applied"] is True
    assert result["selected_minimum_seconds"] == .9
    assert result["held_out_selected"] == dict(true_positive=1, false_positive=0, true_negative=3, false_negative=0)
    # Make holdout negatives very long; selected threshold must not react.
    changed = [replace(r, clip=replace(r.clip, end=r.clip.start+1.6))
               if r.clip.id in result["held_out_ids"] else r for r in labels]
    other = evaluate_duration_filter(changed, policy())
    assert other["selected_minimum_seconds"] == result["selected_minimum_seconds"]
    assert other["held_out_selected"]["false_positive"] == 3


def test_boolean_predictions_must_cover_every_case_exactly() -> None:
    cases = [DetectionCase("yes", 1, True), DetectionCase("no", 1, False)]
    assert count_predictions(cases, {"yes": True, "no": False}) == DetectionCounts(1, 0, 1, 0)
    assert detection_counts(cases, .5) == DetectionCounts(1, 1, 0, 0)
    assert detection_counts([replace(c, overlapping=not c.overlapping) for c in cases], .5) == DetectionCounts(1, 1, 0, 0)
    for decisions in ({"yes": True}, {"yes": True, "no": 0}, {"yes": True, "no": False, "extra": True}):
        with pytest.raises(LabInputError):
            count_predictions(cases, decisions)
    with pytest.raises(LabInputError):
        count_predictions([cases[0], cases[0]], {"yes": True})


def test_tiny_classes_do_not_silently_create_a_benchmark() -> None:
    with pytest.raises(LabInputError):
        split_cases([DetectionCase("a", 1, True), DetectionCase("b", 1, False)], policy())
    cases, _ = detection_cases(reviews())
    with pytest.raises(LabInputError):
        split_cases(cases, replace(policy(), minimum_calibration_positives=3))


@pytest.mark.parametrize("changes", [{"minimum_durations": ()}, {"minimum_durations": (1., .5)},
    {"minimum_durations": (float("nan"),)}, {"minimum_durations": (True,)},
    {"held_out_fraction": 0}, {"held_out_fraction": float("inf")}, {"split_seed": ""},
    {"minimum_calibration_positives": True}, {"contexts": (-1.,)}, {"contexts": (float("nan"),)}])
def test_invalid_policy_is_typed(changes: dict[str, object]) -> None:
    with pytest.raises(LabInputError):
        replace(policy(), **changes)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_case_and_threshold_are_typed(value: float) -> None:
    with pytest.raises(LabInputError):
        DetectionCase("x", value, False)
    with pytest.raises(LabInputError):
        detection_counts([], value)


def test_native_recheck_uses_actual_contract_fields_and_clamps_context_end() -> None:
    regions = native_regions([ModelSpeakerInterval(.1, 2, 0), ModelSpeakerInterval(1, 4, 1),
                              ModelSpeakerInterval(5, 6, 2)], 3)
    assert [(r.speaker, r.start, r.end) for r in regions] == [("0", .1, 2), ("1", 1, 3)]
    with pytest.raises(LabInputError):
        native_regions([], float("nan"))


def test_recovery_summary_does_not_call_accepted_turn_taking_true_overlap() -> None:
    rows = [{"clip_id": str(i), "gain_mode": "shared_peak", "context_seconds": context,
             "human_verdict": verdict, "status": status}
            for i, (context, verdict, status) in enumerate(((1., "two_overlapping", "accepted"),
                (1., "two_sequential", "accepted"), (1., "one_speaker", "rejected"),
                (1., "unsure", "accepted"), (3., "two_overlapping", "rejected")))]
    one_second, three_seconds = summarize_recovery(rows)
    assert one_second["confirmed_overlaps_accepted"] == 1
    assert one_second["negative_controls_tested"] == 2
    assert one_second["negative_controls_incorrectly_accepted"] == 1
    assert three_seconds["negative_controls_tested"] == 0
    with pytest.raises(LabInputError):
        summarize_recovery([rows[0], rows[0]])


def test_recent_persisted_trials_are_not_hidden_by_old_server_cache() -> None:
    old = LabJob("old", "recording", "completed", "old fixture", {})
    recent = LabJob("new", "recording", "completed", "new fixture", {})

    class PersistedProcessor(MockExperimentProcessor):
        def history(self, recording_id: str) -> list[LabJob]:
            return [recent] if recording_id == "recording" else []

        def load(self, job_id: str) -> LabJob | None:
            return {"old": old, "new": recent}.get(job_id)

    service = LabService(None, PersistedProcessor(), replace(LabConfig.load(Path("configs/overlap_lab.json")), max_history=1))
    service._jobs[old.id] = old  # Simulate a long-running server with completed jobs.
    try:
        assert service.history("recording") == [recent]
        active = LabJob("active", "recording", "running", "fixture", None)
        service._jobs[active.id] = active
        assert service.history("recording") == [active]
    finally:
        service.close()
