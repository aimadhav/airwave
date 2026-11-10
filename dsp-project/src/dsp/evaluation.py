"""Interval-exact, permutation-invariant diarization scoring, including overlap.

DER = (missed speaker-time + false alarm speaker-time + confused speaker-time)
/ reference speaker-time. Optional collars exclude boundary-near time. This is
an explicit local scorer, not a claim of official benchmark compliance.
"""

from dataclasses import dataclass
from pathlib import Path
import math


@dataclass(frozen=True)
class TimedSpeaker:
    start: float
    end: float
    speaker: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end) or self.start < 0 or self.end <= self.start:
            raise ValueError("Invalid evaluation interval")


def read_rttm(path: Path) -> list[TimedSpeaker]:
    entries: list[TimedSpeaker] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == "SPEAKER":
            start = float(fields[3])
            entries.append(TimedSpeaker(start, start + float(fields[4]), fields[7]))
    return entries


def diarization_error(reference: list[TimedSpeaker], prediction: list[TimedSpeaker], collar: float = 0.0) -> dict[str, object]:
    """Score exact intervals with globally optimal label assignment."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not math.isfinite(collar) or collar < 0:
        raise ValueError("Collar must be finite and non-negative")
    ref_ids = sorted({s.speaker for s in reference})
    hyp_ids = sorted({s.speaker for s in prediction})
    ref_boundaries = {v for s in reference for v in (s.start, s.end)}
    boundaries = sorted({0.0} | ref_boundaries | {v for s in prediction for v in (s.start, s.end)} |
                        {max(0.0, v + sign * collar) for v in ref_boundaries for sign in (-1, 1)})
    chunks: list[tuple[float, set[str], set[str]]] = []
    weights = np.zeros((len(ref_ids), len(hyp_ids)))
    for start, end in zip(boundaries, boundaries[1:]):
        center = (start + end) / 2
        if collar and any(abs(center - boundary) < collar for boundary in ref_boundaries):
            continue
        refs = {s.speaker for s in reference if s.start <= center < s.end}
        hyps = {s.speaker for s in prediction if s.start <= center < s.end}
        chunks.append((end - start, refs, hyps))
        for ref in refs:
            for hyp in hyps:
                weights[ref_ids.index(ref), hyp_ids.index(hyp)] += end - start
    rows, cols = linear_sum_assignment(-weights)
    mapping = {hyp_ids[col]: ref_ids[row] for row, col in zip(rows, cols)}
    miss = false_alarm = confusion = denominator = 0.0
    for duration, refs, hyps in chunks:
        correct = len(refs & {mapping.get(hyp, "") for hyp in hyps})
        denominator += duration * len(refs)
        miss += duration * max(0, len(refs) - len(hyps))
        false_alarm += duration * max(0, len(hyps) - len(refs))
        confusion += duration * (min(len(refs), len(hyps)) - correct)
    return {"der": (miss + false_alarm + confusion) / denominator if denominator else None,
            "miss_seconds": miss, "false_alarm_seconds": false_alarm,
            "confusion_seconds": confusion, "reference_speaker_seconds": denominator,
            "label_mapping": mapping, "collar_seconds": collar, "overlap_included": True}
