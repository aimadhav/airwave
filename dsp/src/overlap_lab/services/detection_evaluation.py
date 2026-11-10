"""Evaluate a conservative duration ablation without changing any pipeline."""

from dataclasses import asdict
import hashlib
import math
from overlap_lab.evaluation_config import DetectionEvaluationConfig
from overlap_lab.exceptions import LabInputError
from overlap_lab.models.detection_evaluation import DetectionCase, DetectionCounts
from overlap_lab.models.listening import ListeningReview


def detection_cases(reviews: list[ListeningReview]) -> tuple[list[DetectionCase], list[str]]:
    """Treat explicit single/turn-taking/no-speech labels as negatives only."""
    if len({r.clip.id for r in reviews}) != len(reviews):
        raise LabInputError("Duplicate listening case IDs")
    cases: list[DetectionCase] = []
    unresolved: list[str] = []
    for review in reviews:
        if review.verdict in {"unsure", "other_voices"}:
            unresolved.append(review.clip.id)
        else:
            cases.append(DetectionCase(review.clip.id, review.clip.end-review.clip.start,
                                       review.verdict == "two_overlapping"))
    return cases, unresolved


def split_cases(cases: list[DetectionCase], policy: DetectionEvaluationConfig) -> tuple[list[DetectionCase], list[DetectionCase]]:
    """Deterministic class-stratified split; no threshold sees holdout outcomes."""
    if len({c.id for c in cases}) != len(cases):
        raise LabInputError("Duplicate detection case IDs")
    calibration: list[DetectionCase] = []
    held_out: list[DetectionCase] = []
    for truth in (False, True):
        group = sorted((c for c in cases if c.overlapping == truth),
                       key=lambda c: hashlib.sha256(f"{policy.split_seed}:{c.id}".encode()).digest())
        if len(group) < 2:
            raise LabInputError("Need at least two confirmed examples of each class for a split")
        count = min(len(group)-1, max(1, round(len(group)*policy.held_out_fraction)))
        held_out.extend(group[:count])
        calibration.extend(group[count:])
    if sum(c.overlapping for c in calibration) < policy.minimum_calibration_positives:
        raise LabInputError("Too few confirmed calibration overlaps; do not tune a threshold")
    return sorted(calibration, key=lambda c: c.id), sorted(held_out, key=lambda c: c.id)


def detection_counts(cases: list[DetectionCase], minimum_seconds: float) -> DetectionCounts:
    """Count predicted positives retained by duration; no label affects prediction."""
    if isinstance(minimum_seconds, bool) or not isinstance(minimum_seconds, (int, float)) or not math.isfinite(minimum_seconds) or minimum_seconds <= 0:
        raise LabInputError("A positive duration threshold is required")
    return count_predictions(cases, {c.id: c.duration_seconds >= minimum_seconds for c in cases})


def count_predictions(cases: list[DetectionCase], decisions: dict[str, bool]) -> DetectionCounts:
    """Score an explicit prediction per case; refuse partial or leaked mappings."""
    if len({c.id for c in cases}) != len(cases) or set(decisions) != {c.id for c in cases} or any(type(v) is not bool for v in decisions.values()):
        raise LabInputError("Exactly one boolean prediction is required per case")
    predictions = [(c.overlapping, decisions[c.id]) for c in cases]
    return DetectionCounts(sum(t and p for t,p in predictions), sum(not t and p for t,p in predictions),
                           sum(not t and not p for t,p in predictions), sum(t and not p for t,p in predictions))


def evaluate_duration_filter(reviews: list[ListeningReview], policy: DetectionEvaluationConfig) -> dict[str, object]:
    """Select only on calibration data; report tiny holdout separately and honestly.

    A duration threshold is a compute/prioritization ablation, not a validated
    acoustic overlap detector. It can remove real, brief interruptions elsewhere.
    """
    cases, unresolved = detection_cases(reviews)
    calibration, held_out = split_cases(cases, policy)
    trials = [{"minimum_seconds": t, "counts": asdict(detection_counts(calibration, t))}
              for t in policy.minimum_durations]
    # Preserve calibration positives first, reduce false positives second, then
    # prefer the smallest threshold achieving the same calibration result.
    winner = min(trials, key=lambda row: (row["counts"]["false_negative"],
                                         row["counts"]["false_positive"], row["minimum_seconds"]))
    selected = winner["minimum_seconds"]
    baseline = policy.minimum_durations[0]
    return {"experimental_not_applied": True, "method": "duration_only_ablation",
            "baseline_minimum_seconds": baseline, "selected_minimum_seconds": selected,
            "calibration_ids": [c.id for c in calibration], "held_out_ids": [c.id for c in held_out],
            "excluded_unresolved_ids": unresolved, "calibration_trials": trials,
            "held_out_baseline": asdict(detection_counts(held_out, baseline)),
            "held_out_selected": asdict(detection_counts(held_out, selected)),
            "all_reviewed_baseline": asdict(detection_counts(cases, baseline)),
            "all_reviewed_selected": asdict(detection_counts(cases, selected)),
            "case_decisions": [{**asdict(c), "baseline_overlap": c.duration_seconds >= baseline,
                                "experimental_overlap": c.duration_seconds >= selected} for c in cases],
            "limitations": ["Only detector-positive candidates were reviewed; full-recording recall is unknown.",
                "The 12-case convenience sample is small and not a population accuracy estimate.",
                "The human/agent already saw the overall labels; this is not a blinded independent benchmark.",
                "Short real overlaps elsewhere may be removed by a duration-only rule.",
                "Recovery gate acceptance is not evidence of actual overlap or correct words."]}
