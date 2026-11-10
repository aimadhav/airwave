"""Human-reference WER/CER and concatenated minimum-permutation WER."""

from collections import defaultdict
import re
import unicodedata
from typing import Any, Sequence
import numpy as np
from scipy.optimize import linear_sum_assignment

from audio_benchmarks.models import Utterance


def words(text: str) -> list[str]:
    """NFKC/casefold, normalize apostrophes, discard punctuation, retain contractions."""
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", unicodedata.normalize("NFKC", text).casefold().replace("’", "'"))


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> dict[str, int | float | None]:
    """Exact Levenshtein alignment with deterministic substitution/deletion/insertion ties."""
    n, m = len(reference), len(hypothesis)
    distance = np.zeros((n+1,m+1), dtype=np.int32)
    distance[:,0] = np.arange(n+1)
    distance[0,:] = np.arange(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1):
            distance[i,j] = min(distance[i-1,j]+1, distance[i,j-1]+1,
                                distance[i-1,j-1]+(reference[i-1]!=hypothesis[j-1]))
    i,j,s,d,ins = n,m,0,0,0
    while i or j:
        if i and j and distance[i,j] == distance[i-1,j-1]+(reference[i-1]!=hypothesis[j-1]):
            s += reference[i-1]!=hypothesis[j-1]
            i,j = i-1,j-1
        elif i and distance[i,j] == distance[i-1,j]+1:
            d += 1; i -= 1
        else:
            ins += 1; j -= 1
    return {"reference_units": n, "substitutions": int(s), "deletions": d, "insertions": ins,
            "errors": int(s)+d+ins, "error_rate": (int(s)+d+ins)/n if n else None}


def text_scores(reference: str, hypothesis: str) -> dict[str, Any]:
    """CER includes spaces in the same normalized word string; no spelling normalization."""
    r,h = words(reference),words(hypothesis)
    return {"wer":edit_counts(r,h), "cer":edit_counts(list(" ".join(r)),list(" ".join(h))),
            "normalization":"NFKC, casefold, apostrophes normalized, punctuation discarded; CER includes spaces"}


def cpwer(reference: list[Utterance], hypothesis: list[Utterance]) -> dict[str, Any]:
    """Concatenate each speaker chronologically; optimize edit-cost speaker permutation.

    Dummy streams count ALL unmatched hypotheses as insertions and references as
    deletions. Independent from DER label mapping. Not time-constrained cpWER.
    """
    def streams(rows: list[Utterance]) -> dict[str,list[str]]:
        result: dict[str,list[str]] = defaultdict(list)
        for row in sorted(rows,key=lambda x:(x.start,x.end)):
            result[row.speaker].extend(words(row.text))
        return dict(result)
    rs,hs = streams(reference),streams(hypothesis)
    rid,hid = sorted(rs),sorted(hs)
    size = max(len(rid),len(hid))
    costs = np.zeros((size,size))
    details: dict[tuple[int,int],dict[str,Any]] = {}
    for i in range(size):
        for j in range(size):
            score = edit_counts(rs[rid[i]] if i<len(rid) else [], hs[hid[j]] if j<len(hid) else [])
            details[i,j] = score
            costs[i,j] = score["errors"]
    rows,cols = linear_sum_assignment(costs)
    count = sum(len(v) for v in rs.values())
    errors = sum(int(costs[i,j]) for i,j in zip(rows,cols))
    return {"cpwer":errors/count if count else None, "reference_words":count, "errors":errors,
        "assignments":[{"reference_speaker":rid[i] if i<len(rid) else None,
                        "hypothesis_speaker":hid[j] if j<len(hid) else None,
                        **details[i,j]} for i,j in zip(rows,cols)],
        "mapping":"minimum edit-cost one-to-one permutation, not DER mapping", "time_constrained":False}
