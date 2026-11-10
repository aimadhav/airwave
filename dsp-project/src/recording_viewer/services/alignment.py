"""Pure word-to-speaker reconciliation and overlap layout grouping."""

import math
from recording_viewer.exceptions import TranscriptInputError
from recording_viewer.models import AlignedTurn, TranscriptGroup, TranscriptWord, VoiceRegion


def union_duration(intervals: list[tuple[float, float]]) -> float:
    """Measure interval union so duplicate regions cannot bias assignment."""
    total, right = 0.0, -1.0
    for start, end in sorted(intervals):
        total += max(0.0, end - max(start, right))
        right = max(right, end)
    return total


def align_words(words: list[TranscriptWord], regions: list[VoiceRegion], minimum_coverage: float) -> list[AlignedTurn]:
    """Assign each word once and retain uncertainty during simultaneous voices.

    This is temporal alignment, not source separation. A secondary active voice
    without recognized words is retained as an unresolved row, never given a
    duplicate of the dominant transcript.
    """
    if not math.isfinite(minimum_coverage) or not 0 < minimum_coverage <= 1:
        raise TranscriptInputError("minimum_coverage must lie in (0, 1]")
    if len({row.id for row in regions}) != len(regions):
        raise TranscriptInputError("Region IDs must be unique")
    assignments: dict[str, list[TranscriptWord]] = {row.id: [] for row in regions}
    uncertain: set[str] = set()
    unknown: list[AlignedTurn] = []
    for index, word in enumerate(sorted(words, key=lambda row: (row.start, row.end))):
        candidates = [row for row in regions if min(row.end, word.end) > max(row.start, word.start)]
        if not candidates:
            unknown.append(AlignedTurn(f"unassigned_{index}", "UNKNOWN", word.start, word.end,
                                       word.text.strip(), True, False, 1))
            continue
        by_speaker: dict[str, list[tuple[float, float]]] = {}
        for row in candidates:
            by_speaker.setdefault(row.speaker, []).append((max(row.start, word.start), min(row.end, word.end)))
        scores = {speaker: union_duration(spans) for speaker, spans in by_speaker.items()}
        speaker = min(scores, key=lambda key: (-scores[key], key))
        target = min((row for row in candidates if row.speaker == speaker),
                     key=lambda row: (-(min(row.end, word.end) - max(row.start, word.start)), row.start, row.id))
        assignments[target.id].append(word)
        if len(scores) > 1 or scores[speaker] / (word.end - word.start) < minimum_coverage:
            uncertain.add(target.id)
    turns: list[AlignedTurn] = []
    for row in regions:
        overlaps = any(other.speaker != row.speaker and min(other.end, row.end) > max(other.start, row.start)
                       for other in regions)
        selected = assignments[row.id]
        text = "".join(word.text for word in selected).strip()
        turns.append(AlignedTurn(row.id, row.speaker, row.start, row.end, text,
                                  row.id in uncertain or not selected, overlaps, len(selected)))
    return sorted(turns + unknown, key=lambda row: (row.start, row.end, row.speaker))


def group_turns(turns: list[AlignedTurn]) -> list[TranscriptGroup]:
    """Group intersecting passages; touching endpoints do not imply overlap."""
    groups: list[TranscriptGroup] = []
    current: list[AlignedTurn] = []
    end = 0.0

    def finish() -> None:
        if current:
            overlap = any(a.speaker != b.speaker and min(a.end, b.end) > max(a.start, b.start)
                          for i, a in enumerate(current) for b in current[i + 1:])
            groups.append(TranscriptGroup(current[0].start, max(row.end for row in current),
                                          tuple(row.id for row in current), overlap))

    for turn in sorted(turns, key=lambda row: (row.start, row.end, row.speaker)):
        if current and turn.start >= end:
            finish()
            current = []
        current.append(turn)
        end = max(row.end for row in current)
    finish()
    return groups
