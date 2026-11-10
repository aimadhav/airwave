"""Source-reference signal metrics, never substitute ASR confidence for quality."""

from itertools import permutations
from typing import Any
import numpy as np
from audio_benchmarks.exceptions import BenchmarkInputError


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float | None:
    """Zero-mean scale-invariant SDR in dB; silent inputs produce undefined scores."""
    if (estimate.ndim!=1 or reference.ndim!=1 or estimate.shape!=reference.shape or not estimate.size
            or not np.isfinite(estimate).all() or not np.isfinite(reference).all()):
        raise BenchmarkInputError("SI-SDR needs finite, equal-length mono source waveforms")
    e = estimate.astype(np.float64)-float(estimate.mean())
    r = reference.astype(np.float64)-float(reference.mean())
    eps = np.finfo(np.float64).eps
    if np.dot(e,e)<=eps or np.dot(r,r)<=eps:
        return None
    target = np.dot(e,r)/np.dot(r,r)*r
    return float(10*np.log10((np.dot(target,target)+eps)/(np.dot(e-target,e-target)+eps)))


def score_separation(mixture: np.ndarray, sources: tuple[np.ndarray,...],
                     estimates: tuple[np.ndarray,...]) -> dict[str,Any]:
    """Oracle source permutation for scoring only—not the pipeline's identity matcher."""
    if len(sources)!=2 or len(estimates)!=2:
        raise BenchmarkInputError("This separation benchmark requires exactly two sources and outputs")
    scores = [[si_sdr(track,source) for source in sources] for track in estimates]
    baseline = [si_sdr(mixture,source) for source in sources]
    if any(value is None for row in scores for value in row) or any(v is None for v in baseline):
        return {"defined":False,"reason":"silent or constant reference/output"}
    assignment = max(permutations(range(2)),key=lambda p:sum(scores[i][j] for i,j in enumerate(p)))
    improvements = [scores[i][j]-baseline[j] for i,j in enumerate(assignment)]
    return {"defined":True,"si_sdr_matrix_db":scores,"mixture_si_sdr_db":baseline,
        "track_to_source":list(assignment),"si_sdri_db":improvements,
        "mean_si_sdri_db":float(np.mean(improvements)),
        "mean_si_sdr_db":float(np.mean([scores[i][j] for i,j in enumerate(assignment)])),
        "mapping":"oracle permutation for evaluation only; not speaker-identity accuracy"}


def intelligibility_scores(sources: tuple[np.ndarray,...], estimates: tuple[np.ndarray,...],
                           assignment: list[int], rate: int) -> dict[str,Any]:
    """Optional STOI/ESTOI against the same known source waveforms and oracle permutation."""
    try:
        from pystoi import stoi
    except ImportError:
        return {"available":False,"reason":"Optional pystoi not installed"}
    ordinary=[float(stoi(sources[j],estimates[i],rate,extended=False)) for i,j in enumerate(assignment)]
    extended=[float(stoi(sources[j],estimates[i],rate,extended=True)) for i,j in enumerate(assignment)]
    if not all(np.isfinite(v) for v in ordinary+extended):
        return {"available":False,"reason":"STOI/ESTOI undefined for these waveforms"}
    return {"available":True,"stoi":ordinary,"estoi":extended,
            "mean_stoi":float(np.mean(ordinary)),"mean_estoi":float(np.mean(extended)),
            "scope":"source-reference intelligibility proxy, not word accuracy or MOS"}
