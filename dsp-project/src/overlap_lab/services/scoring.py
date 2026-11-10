"""Explainable word-error scoring; never confuse ASR confidence with accuracy."""

import re
import unicodedata
from dataclasses import asdict
from overlap_lab.models import WordErrorScore
from overlap_lab.models import ReferenceAnnotation


def tokenize(text: str) -> list[str]:
    """Case-insensitive Unicode words; normalize typographic apostrophes."""
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", normalized, flags=re.UNICODE)


def word_errors(reference: str, hypothesis: str) -> WordErrorScore:
    """Levenshtein S/D/I alignment. Empty-reference WER is undefined, not 0%."""
    expected, actual = tokenize(reference), tokenize(hypothesis)
    matrix = [[0] * (len(actual) + 1) for _ in range(len(expected) + 1)]
    for i in range(len(expected) + 1):
        matrix[i][0] = i
    for j in range(len(actual) + 1):
        matrix[0][j] = j
    for i, a in enumerate(expected, 1):
        for j, b in enumerate(actual, 1):
            matrix[i][j] = min(matrix[i-1][j] + 1, matrix[i][j-1] + 1, matrix[i-1][j-1] + (a != b))
    i, j = len(expected), len(actual)
    operations: list[dict[str, str]] = []
    counts = {"substitution": 0, "deletion": 0, "insertion": 0}
    while i or j:
        if i and j and matrix[i][j] == matrix[i-1][j-1] + (expected[i-1] != actual[j-1]):
            kind = "correct" if expected[i-1] == actual[j-1] else "substitution"
            operations.append({"kind": kind, "reference": expected[i-1], "hypothesis": actual[j-1]})
            i, j = i-1, j-1
        elif i and matrix[i][j] == matrix[i-1][j] + 1:
            kind = "deletion"
            operations.append({"kind": kind, "reference": expected[i-1], "hypothesis": ""})
            i -= 1
        else:
            kind = "insertion"
            operations.append({"kind": kind, "reference": "", "hypothesis": actual[j-1]})
            j -= 1
        if kind in counts:
            counts[kind] += 1
    return WordErrorScore(len(expected), counts["substitution"], counts["deletion"], counts["insertion"],
                          sum(counts.values()) / len(expected) if expected else None, tuple(reversed(operations)))


def score_speakers(reference: ReferenceAnnotation, hypotheses: dict[str, str]) -> dict[str, object] | None:
    """Only reviewed human references produce speaker-labelled WER."""
    if not reference.reviewed:
        return None
    scores = {speaker: asdict(word_errors(text, hypotheses.get(speaker, ""))) for speaker, text in reference.texts.items()}
    count = sum(score["reference_words"] for score in scores.values())
    errors = sum(score["substitutions"] + score["deletions"] + score["insertions"] for score in scores.values())
    return {"by_speaker": scores, "reference_words": count, "errors": errors,
            "wer": errors / count if count else None}
