"""Small review queues and honest human labels; no model inference or training."""

from dataclasses import asdict
from overlap_lab.config import LabConfig
from overlap_lab.exceptions import LabInputError
from overlap_lab.interfaces import ListeningReviewRepository, StudyRepository
from overlap_lab.models import ClipSpec
from overlap_lab.models.listening import ListeningReview


def starter_clips(clips: list[ClipSpec], duration: float, count: int, minimum_seconds: float) -> list[str]:
    """Choose nearby time-stratified candidates, not an accuracy benchmark."""
    preferred = [c for c in clips if c.end - c.start >= minimum_seconds]
    pool = preferred if len(preferred) >= min(count, len(clips)) else clips
    remaining = {c.id: c for c in pool}
    chosen: list[ClipSpec] = []
    total = min(count, len(remaining))
    for i in range(total):
        target = duration * (i + .5) / total
        clip = min(remaining.values(), key=lambda c: (abs((c.start + c.end) / 2 - target), c.id))
        chosen.append(clip)
        del remaining[clip.id]
    return [c.id for c in sorted(chosen, key=lambda c: (c.start, c.id))]


def voice_examples(regions: list[dict[str, object]], seconds: float) -> dict[str, dict[str, float]]:
    """Find model-exclusive examples; never claim that these verify identity."""
    events: dict[float, list[tuple[str, int]]] = {}
    for region in regions:
        for time, delta in ((region["start"], 1), (region["end"], -1)):
            events.setdefault(time, []).append((region["speaker"], delta))
    active: dict[str, int] = {}
    spans: list[tuple[float, float, str]] = []
    previous = 0.0
    for time, changes in sorted(events.items()):
        speakers = [s for s, n in active.items() if n > 0]
        if len(speakers) == 1 and time > previous:
            speaker = speakers[0]
            if spans and spans[-1][1] == previous and spans[-1][2] == speaker:
                spans[-1] = (spans[-1][0], time, speaker)
            else:
                spans.append((previous, time, speaker))
        for speaker, delta in changes:
            active[speaker] = active.get(speaker, 0) + delta
        previous = time
    result: dict[str, dict[str, float]] = {}
    for start, end, speaker in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
        if speaker not in result and end - start >= seconds:
            middle = (start + end) / 2
            result[speaker] = {"start": middle - seconds / 2, "end": middle + seconds / 2}
    return result


class ListeningReviewService:
    """Coordinates a narrow listening repository, separately from WER reviews."""

    def __init__(self, studies: StudyRepository, reviews: ListeningReviewRepository, config: LabConfig) -> None:
        self.studies, self.reviews, self.config = studies, reviews, config

    def bootstrap(self, recording_id: str) -> dict[str, object]:
        """Return only the audio timeline and labels needed by the quick page."""
        study = self.studies.study(recording_id)
        clips = [ClipSpec.from_mapping(c) for c in study["clips"]
                 if 0 < c["end"] - c["start"] <= self.config.max_clip_seconds]
        return {"recordings": self.studies.list_recordings(),
                "study": {k: study[k] for k in ("id", "filename", "source_hash", "duration", "speakers", "audio_url")},
                "clips": [asdict(c) for c in clips], "reviews": self.reviews.reviews(recording_id),
                "starter_ids": starter_clips(clips, study["duration"], self.config.listening_sample_size,
                                             self.config.listening_minimum_seconds),
                "voice_examples": voice_examples(study["regions"], self.config.listening_voice_seconds),
                "context_seconds": self.config.default_context_seconds,
                "notice": "Listening labels do not create transcript references, WER scores, corrected production output, or model training."}

    def save(self, recording_id: str, source_hash: str, revision: int, review: ListeningReview) -> dict[str, object]:
        """Allow false-overlap answers even when the detector claims two voices."""
        if type(revision) is not int or revision < 0:
            raise LabInputError("Invalid listening-label revision")
        study = self.studies.study(recording_id)
        candidates = {c["id"]: c for c in study["clips"]}
        original = candidates.get(review.clip.id)
        if original is None or asdict(review.clip) != {k: tuple(original[k]) if k == "speakers" else original[k]
                                                     for k in ("id", "start", "end", "speakers")}:
            raise LabInputError("Select an unchanged detected interval from this recording")
        if review.clip.end > study["duration"] or review.clip.end - review.clip.start > self.config.max_clip_seconds:
            raise LabInputError("Listening interval exceeds the configured bounds")
        return self.reviews.save(recording_id, source_hash, revision, review)
