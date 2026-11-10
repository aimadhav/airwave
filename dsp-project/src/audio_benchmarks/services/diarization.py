"""Interval-exact local DER/JER and activity metrics with explicit scoring scope."""

import math
from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment

from audio_benchmarks.exceptions import BenchmarkInputError
from audio_benchmarks.models import Interval


def activity_cells(reference: list[Interval], hypothesis: list[Interval], duration: float,
                   collar: float = 0.0, overlap_included: bool = True) -> list[tuple[float, set[str], set[str]]]:
    """Partition a declared whole-file UEM, excluding collars/ref overlap if requested."""
    if (not math.isfinite(duration) or duration <= 0 or not math.isfinite(collar) or collar < 0
            or any(row.end > duration + 1e-6 for row in reference + hypothesis)):
        raise BenchmarkInputError("Invalid scoring duration, collar, or out-of-range interval")
    ref_edges = {v for row in reference for v in (row.start, row.end)}
    edges = sorted({0.0, duration} | ref_edges | {v for row in hypothesis for v in (row.start, row.end)} |
        {min(duration, max(0, v + sign*collar)) for v in ref_edges for sign in (-1, 1)})
    cells: list[tuple[float, set[str], set[str]]] = []
    for a, b in zip(edges, edges[1:]):
        center = (a+b)/2
        if collar and any(abs(center-edge) < collar for edge in ref_edges):
            continue
        refs = {r.speaker for r in reference if r.start <= center < r.end}
        hyps = {r.speaker for r in hypothesis if r.start <= center < r.end}
        if overlap_included or len(refs) < 2:
            cells.append((b-a, refs, hyps))
    return cells


def detection_counts(tp: float, fp: float, fn: float, tn: float) -> dict[str, float | None]:
    """Duration-weighted binary detection; undefined denominators stay null."""
    return {"true_positive_seconds": tp, "false_positive_seconds": fp,
        "false_negative_seconds": fn, "true_negative_seconds": tn,
        "precision": tp/(tp+fp) if tp+fp else None,
        "recall": tp/(tp+fn) if tp+fn else None,
        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        "iou": tp/(tp+fp+fn) if tp+fp+fn else None}


def score_diarization(reference: list[Interval], hypothesis: list[Interval], duration: float,
                       collar: float = 0.0, overlap_included: bool = True) -> dict[str, Any]:
    """Globally map labels for DER; independently optimize IoU assignment for JER.

    JER averages over reference speakers; extra hypothesized speakers can be
    underpenalized by this metric. DER retains all false-alarm speaker time.
    This local implementation is not certified official dscore/md-eval output.
    """
    cells = activity_cells(reference, hypothesis, duration, collar, overlap_included)
    refs = sorted({s for _, active, _ in cells for s in active})
    hyps = sorted({s for _, _, active in cells for s in active})
    rindex, hindex = {s:i for i,s in enumerate(refs)}, {s:i for i,s in enumerate(hyps)}
    weights = np.zeros((len(refs), len(hyps)))
    rtime, htime = np.zeros(len(refs)), np.zeros(len(hyps))
    for seconds, rset, hset in cells:
        for r in rset:
            rtime[rindex[r]] += seconds
            for h in hset:
                weights[rindex[r], hindex[h]] += seconds
        for h in hset:
            htime[hindex[h]] += seconds
    rows, cols = linear_sum_assignment(-weights)
    mapping = {hyps[j]:refs[i] for i,j in zip(rows, cols)}
    denominator = miss = false_alarm = confusion = count_error = 0.0
    speech = [0.0]*4
    overlap = [0.0]*4
    for seconds, rset, hset in cells:
        correct = len(rset & {mapping.get(h, "") for h in hset})
        denominator += seconds*len(rset)
        miss += seconds*max(0, len(rset)-len(hset))
        false_alarm += seconds*max(0, len(hset)-len(rset))
        confusion += seconds*(min(len(rset),len(hset))-correct)
        count_error += seconds*abs(len(rset)-len(hset))
        for counters, ractive, hactive in ((speech, bool(rset), bool(hset)),
                                           (overlap, len(rset)>=2, len(hset)>=2)):
            index = 0 if ractive and hactive else 1 if hactive else 2 if ractive else 3
            counters[index] += seconds
    union = rtime[:,None]+htime[None,:]-weights
    iou = np.divide(weights, union, out=np.zeros_like(weights), where=union>0)
    jrows, jcols = linear_sum_assignment(-iou)
    jer = 1-float(iou[jrows,jcols].sum())/len(refs) if refs else None
    scored_seconds = sum(d for d,_,_ in cells)
    return {"der": (miss+false_alarm+confusion)/denominator if denominator else None,
        "jer": jer, "miss_seconds": miss, "false_alarm_seconds": false_alarm,
        "confusion_seconds": confusion, "reference_speaker_seconds": denominator,
        "scored_wall_seconds": scored_seconds, "collar_seconds": collar,
        "collar_convention": "half-width around every reference start/end",
        "overlap_included": overlap_included, "uem": [0,duration], "label_mapping": mapping,
        "speaker_count_reference": len({r.speaker for r in reference}),
        "speaker_count_hypothesis": len({r.speaker for r in hypothesis}),
        "speaker_count_absolute_error": abs(len({r.speaker for r in reference})-len({r.speaker for r in hypothesis})),
        "active_speaker_count_mae": count_error/scored_seconds if scored_seconds else None,
        "speech_detection": detection_counts(*speech), "overlap_detection": detection_counts(*overlap),
        "reference_overlap_seconds": sum(d for d,r,_ in cells if len(r)>=2)}
