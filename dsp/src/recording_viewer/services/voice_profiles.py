"""Enroll supplied clean spans via a provider protocol; no storage/runtime imports."""

from datetime import datetime, timezone
import math
from collections.abc import Mapping, Sequence
import numpy as np

from recording_viewer.exceptions import RecoveryError, VoiceProfileError
from recording_viewer.interfaces.recovery import VoiceEmbedder
from recording_viewer.models.voice_profiles import ProfileContext, SpeakerVoiceProfile, VoiceProfileSet, VoiceReference


def normalized(vector: np.ndarray) -> tuple[float, ...]:
    """Use the same unit-normalize / mean / unit-normalize recipe as recovery."""
    values = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(values)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or not math.isfinite(float(norm)) or norm <= 0:
        raise VoiceProfileError("Provider returned an invalid voice embedding")
    return tuple(float(v) for v in values / norm)


def enroll_profiles(samples: np.ndarray, context: ProfileContext,
                    spans: Mapping[str, Sequence[tuple[float, float]]], embedder: VoiceEmbedder,
                    minimum_references: int) -> VoiceProfileSet:
    """Extract clean reference vectors and retain provenance for later matching."""
    if (type(minimum_references) is not int or minimum_references < 1 or samples.ndim != 1
            or not samples.size or not np.isfinite(samples).all()):
        raise VoiceProfileError("Enrollment requires finite mono audio and a positive reference minimum")
    profiles: list[SpeakerVoiceProfile] = []
    warnings: list[str] = []
    for speaker, intervals in sorted(spans.items()):
        references: list[VoiceReference] = []
        for start, end in intervals:
            if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end <= len(samples)/context.sample_rate:
                raise VoiceProfileError("Reference interval extends outside source audio")
            left, right = round(start*context.sample_rate), round(end*context.sample_rate)
            try:
                vector = normalized(embedder.embed(samples[left:right], context.sample_rate))
            except RecoveryError:
                warnings.append(f"Reference embedding failed for {speaker} at {start:.3f}s")
                continue
            references.append(VoiceReference(left/context.sample_rate, right/context.sample_rate, vector))
        if len(references) < minimum_references:
            warnings.append(f"Insufficient successful clean references for {speaker}")
            continue
        if len({len(r.embedding) for r in references}) != 1:
            raise VoiceProfileError("Provider changed reference embedding dimensions")
        centroid = normalized(np.mean([r.embedding for r in references], axis=0))
        profiles.append(SpeakerVoiceProfile(speaker, centroid, tuple(references)))
    if not profiles:
        raise VoiceProfileError("No speaker has enough clean references; no profiles were saved")
    return VoiceProfileSet(context, datetime.now(timezone.utc).isoformat(), tuple(profiles), tuple(warnings))
