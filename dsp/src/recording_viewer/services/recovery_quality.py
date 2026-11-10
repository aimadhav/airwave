"""Non-reference signal checks and permutation-safe speaker matching."""

from difflib import SequenceMatcher
from itertools import permutations
import re
import numpy as np
from recording_viewer.exceptions import RecoveryError
from recording_viewer.recovery_config import RecoveryConfig


def unit(vector: np.ndarray) -> np.ndarray:
    """Normalize a finite vector; never silently accept empty/zero embeddings."""
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or np.linalg.norm(values) <= 0:
        raise RecoveryError("Invalid speaker embedding")
    return values / np.linalg.norm(values)


def signal_checks(mixture: np.ndarray, tracks: tuple[np.ndarray, ...], config: RecoveryConfig) -> tuple[str, dict[str, float]]:
    """Reject silence, malformed, near-duplicate and inconsistent source outputs.

    Reconstruction is a necessary sanity check, NOT evidence of correct words.
    """
    metrics: dict[str, float] = {}
    if len(tracks) != 2 or any(t.ndim != 1 or len(t) != len(mixture) or not np.isfinite(t).all() for t in tracks):
        return "invalid_source_shape", metrics
    x = np.asarray(mixture, dtype=np.float64)
    streams = [np.asarray(t, dtype=np.float64) for t in tracks]
    energies = [float(np.mean(t * t)) for t in streams]
    if min(energies) < config.minimum_stream_rms ** 2:
        return "silent_source", metrics
    metrics["energy_ratio"] = min(energies) / max(energies)
    if metrics["energy_ratio"] < config.minimum_energy_ratio:
        return "negligible_secondary_source", metrics
    centered = [t - t.mean() for t in streams]
    denominator = np.linalg.norm(centered[0]) * np.linalg.norm(centered[1])
    metrics["stream_correlation"] = float(abs(np.dot(*centered)) / denominator) if denominator else 1.0
    if metrics["stream_correlation"] > config.maximum_stream_correlation:
        return "duplicated_sources", metrics
    design = np.column_stack(streams)
    gains, *_ = np.linalg.lstsq(design, x, rcond=None)
    residual = x - design @ gains
    metrics["reconstruction_db"] = float(10 * np.log10((np.mean(x * x) + np.finfo(float).eps) /
                                                      (np.mean(residual * residual) + np.finfo(float).eps)))
    if metrics["reconstruction_db"] < config.minimum_reconstruction_db:
        return "inconsistent_sources", metrics
    return "", metrics


def match_tracks(embeddings: list[np.ndarray], speakers: tuple[str, ...], profiles: dict[str, np.ndarray],
                 config: RecoveryConfig) -> tuple[tuple[str, ...] | None, dict[str, float]]:
    """Joint one-to-one assignment; require each track's own margin as well."""
    if len(embeddings) != len(speakers) or any(s not in profiles for s in speakers):
        return None, {}
    if len({unit(v).shape for v in [*embeddings, *(profiles[s] for s in speakers)]}) != 1:
        raise RecoveryError("Inconsistent speaker embedding dimensions")
    scores = np.array([[float(np.dot(unit(vector), unit(profiles[s]))) for s in speakers] for vector in embeddings])
    ranked = sorted(permutations(range(len(speakers))), key=lambda p: -sum(scores[i, j] for i, j in enumerate(p)))
    best = ranked[0]
    metrics: dict[str, float] = {}
    passed = True
    for i, index in enumerate(best):
        score = scores[i, index]
        margin = score - max(scores[i, j] for j in range(len(speakers)) if j != index)
        metrics[f"track_{i}_cosine"] = float(score)
        metrics[f"track_{i}_margin"] = float(margin)
        if score < config.minimum_cosine or margin < config.minimum_cosine_margin:
            passed = False
    return (tuple(speakers[j] for j in best) if passed else None), metrics


def duplicate_text(first: str, second: str, config: RecoveryConfig) -> bool:
    """Long near-identical transcripts can indicate cross-track voice leakage."""
    a, b = re.findall(r"\w+", first.casefold()), re.findall(r"\w+", second.casefold())
    return min(len(a), len(b)) >= config.minimum_duplicate_words and SequenceMatcher(None, a, b).ratio() >= config.maximum_duplicate_similarity
