"""Keep separated float PCM in range without boosting the weaker voice alone."""

import math
import numpy as np
from recording_viewer.exceptions import RecoveryError


def shared_peak_gain(tracks: tuple[np.ndarray, ...], peak_limit: float) -> tuple[tuple[np.ndarray, ...], float, float]:
    """Apply one attenuation factor to all tracks, preserving relative levels.

    Batch separation can have arbitrary absolute decoder gain. Never feed values
    far outside normalized PCM range into ASR/embeddings. This is not speech
    detection, denoising, source verification, or independent track loudness gain.
    Silence and already-bounded samples are not amplified.
    """
    if isinstance(peak_limit, bool) or not isinstance(peak_limit, (int, float)) or not math.isfinite(peak_limit) or not 0 < peak_limit <= 1:
        raise RecoveryError("Separator peak limit must lie in (0, 1]")
    if not tracks or any(t.ndim != 1 or not t.size or not np.isfinite(t).all() for t in tracks) or len({len(t) for t in tracks}) != 1:
        raise RecoveryError("Gain calibration requires equal-length finite mono tracks")
    peak = max(float(np.max(np.abs(np.asarray(t, dtype=np.float64)))) for t in tracks)
    gain = min(1.0, peak_limit/peak) if peak else 1.0
    return tuple((np.asarray(t, dtype=np.float64)*gain).astype(np.float32) for t in tracks), gain, peak
