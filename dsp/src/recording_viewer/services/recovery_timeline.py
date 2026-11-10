"""Sweep-line overlap detection, exclusive references and lossless splicing."""

from dataclasses import replace
import math
from recording_viewer.exceptions import RecoveryError
from recording_viewer.models import AlignedTurn, TranscriptWord, VoiceRegion
from recording_viewer.models.recovery import RecoveryResult, RecoveryWindow
from recording_viewer.recovery_config import RecoveryConfig
from recording_viewer.services.alignment import align_words


def activity_spans(regions: list[VoiceRegion], duration: float) -> list[tuple[float, float, tuple[str, ...]]]:
    """Count distinct speakers, not overlapping duplicate regions of one speaker."""
    if not math.isfinite(duration) or duration < 0:
        raise RecoveryError("Recording duration must be finite and nonnegative")
    if len({row.id for row in regions}) != len(regions):
        raise RecoveryError("Speaker region identifiers must be unique")
    if any(row.end > duration for row in regions):
        raise RecoveryError("Speaker timeline extends beyond the recording")
    events: dict[float, list[tuple[str, int]]] = {}
    for row in regions:
        events.setdefault(row.start, []).append((row.speaker, 1))
        events.setdefault(row.end, []).append((row.speaker, -1))
    active: dict[str, int] = {}
    spans: list[tuple[float, float, tuple[str, ...]]] = []
    previous = 0.0
    for time, changes in sorted(events.items()):
        speakers = tuple(sorted(s for s, count in active.items() if count > 0))
        if time > previous and speakers:
            if spans and spans[-1][1] == previous and spans[-1][2] == speakers:
                spans[-1] = (spans[-1][0], time, speakers)
            else:
                spans.append((previous, time, speakers))
        for speaker, delta in changes:
            active[speaker] = active.get(speaker, 0) + delta
        previous = time
    return spans


def plan_windows(regions: list[VoiceRegion], duration: float, config: RecoveryConfig) -> list[RecoveryWindow]:
    """Disjoint ownership cores prevent duplicated words across context windows."""
    windows: list[RecoveryWindow] = []
    spans = activity_spans(regions, duration)
    for start, end, speakers in spans:
        if len(speakers) < 2:
            continue
        cursor = start
        while cursor < end:
            right = min(cursor + config.max_core_seconds, end)
            crop_start, crop_end = max(0.0, cursor - config.context_seconds), min(duration, right + config.context_seconds)
            # A two-output separator must not receive a third context speaker.
            for a, b, context_speakers in spans:
                if set(context_speakers) - set(speakers):
                    if b <= cursor:
                        crop_start = max(crop_start, b)
                    elif a >= right:
                        crop_end = min(crop_end, a)
            windows.append(RecoveryWindow(f"overlap_{len(windows):04d}", cursor, right, crop_start, crop_end, speakers))
            cursor = right
    return windows


def reference_spans(regions: list[VoiceRegion], duration: float, config: RecoveryConfig) -> dict[str, list[tuple[float, float]]]:
    """Enroll only guarded, non-overlapping clips; never enroll mixed speech."""
    candidates: dict[str, list[tuple[float, float]]] = {}
    for start, end, speakers in activity_spans(regions, duration):
        if len(speakers) != 1:
            continue
        cursor, right = start + config.reference_guard_seconds, end - config.reference_guard_seconds
        while cursor + config.reference_seconds <= right:
            candidates.setdefault(speakers[0], []).append((cursor, cursor + config.reference_seconds))
            cursor += config.reference_seconds
    # Spread references over the recording rather than only the first utterance.
    result: dict[str, list[tuple[float, float]]] = {}
    for speaker, spans in candidates.items():
        if len(spans) <= config.max_references:
            result[speaker] = spans
        else:
            indexes = [round(i * (len(spans) - 1) / (config.max_references - 1))
                       for i in range(config.max_references)] if config.max_references > 1 else [0]
            result[speaker] = [spans[i] for i in indexes]
    return result


def splice_recovery(words: list[TranscriptWord], regions: list[VoiceRegion], result: RecoveryResult,
                    minimum_coverage: float) -> list[AlignedTurn]:
    """Replace accepted cores only; rejected overlap retains uncertain baseline.

    Word ownership uses midpoints consistently at every boundary. Full recognized
    word intervals are retained; rejected/baseline results remain in the audit.
    """
    accepted = sorted((row for row in result.reports if row.status == "accepted"), key=lambda row: row.window.start)
    for left, right in zip(accepted, accepted[1:]):
        if left.window.end > right.window.start:
            raise RecoveryError("Accepted ownership cores cannot overlap")
    cores = [(row.window.start, row.window.end) for row in accepted]
    kept_words = [word for word in words if not any(a <= (word.start + word.end) / 2 < b for a, b in cores)]
    fragments: list[VoiceRegion] = []
    for region in regions:
        spans = [(region.start, region.end)]
        for left, right in cores:
            spans = [(a, b) for start, end in spans
                     for a, b in ((start, min(end, left)), (max(start, right), end)) if b > a]
        fragments.extend(VoiceRegion(f"{region.id}_part_{i}", region.speaker, start, end)
                         for i, (start, end) in enumerate(spans))
    turns = align_words(kept_words, fragments, minimum_coverage)
    for report in accepted:
        for speaker in report.window.speakers:
            selected = sorted((row.word for row in report.words if row.speaker == speaker), key=lambda row: row.start)
            turns.append(AlignedTurn(f"{report.window.id}_{speaker}", speaker,
                min([report.window.start, *(word.start for word in selected)]),
                max([report.window.end, *(word.end for word in selected)]),
                "".join(word.text for word in selected).strip(), False, True,
                len(selected), "separated_asr", "voice_embedding"))
    # Recompute overlap for baseline fragments intersecting a recovered word row.
    return sorted((replace(turn, overlap=any(other.speaker != turn.speaker and
        min(other.end, turn.end) > max(other.start, turn.start) for other in turns)) for turn in turns),
        key=lambda row: (row.start, row.end, row.speaker))
