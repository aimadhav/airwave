"""Allowlisted, read-only public artifacts; private processing state never escapes."""

from copy import deepcopy
from dataclasses import fields
import math
import re

from recording_viewer.exceptions import RecordingViewerError
from recording_viewer.models import AlignedTurn, TranscriptGroup


def public_snapshot(document: dict[str, object]) -> dict[str, object]:
    """Export known presentation fields, not private paths, logs or review stores."""
    try:
        keys = ("schema_version", "filename", "source_sha256", "duration", "sample_rate", "audio",
                "recognizer", "model_revision", "word_count", "waveform", "notice")
        result = {key: deepcopy(document[key]) for key in keys}
        if (not math.isfinite(result["duration"]) or result["duration"] <= 0
                or result["audio"] != "audio.mp3"):
            raise RecordingViewerError("The shared podcast requires a non-empty MP3 presentation")
        turn_keys = {field.name for field in fields(AlignedTurn)}
        group_keys = {field.name for field in fields(TranscriptGroup)}
        result["variants"] = {
            name: {"turns": [{key: deepcopy(value) for key, value in row.items() if key in turn_keys}
                             for row in variant["turns"]],
                   "groups": [{key: deepcopy(value) for key, value in row.items() if key in group_keys}
                              for row in variant["groups"]]}
            for name, variant in document["variants"].items()
            if name in {"nemo_native", "sherpa", "overlap_recovery"}}
        if "nemo_native" not in result["variants"]:
            raise RecordingViewerError("A native speaker timeline is required for this demo")
        result["default_backend"] = "nemo_native"
        result["read_only"] = True
        # Keep old recovery counts as historical evidence, not a post-fix rerun.
        recovery = document.get("recovery", {})
        result["recovery"] = {key: deepcopy(recovery[key]) for key in
            ("enabled", "backend", "windows", "accepted", "rejected", "skipped", "failed") if key in recovery}
        result["recovery"]["decisions"] = [{"start": row["start"], "end": row["end"], "status": row["status"],
            "reason": row["reason"] if re.fullmatch(r"[a-z_]+", row["reason"]) else "private_detail_not_exported"}
            for row in recovery.get("decisions", [])]
        result["notice"] += (" Shared frozen snapshot. Full-recording recovery predates the shared-peak gain fix; "
            "new corrected clip experiments are shown separately under Results & overlap examples. "
            "The default view is the original Nemotron timeline, not a new full-recording recovery run.")
        return result
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise RecordingViewerError("The recording snapshot does not match the public presentation contract") from error


def script_json(value: object) -> str:
    """Encode embedded JSON without allowing transcript text to close its script."""
    import json
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def transcript_markdown(document: dict[str, object]) -> str:
    """Provide a GitHub-readable transcript as well as the interactive HTML."""
    from html import escape
    turns = document["variants"]["nemo_native"]["turns"]
    speakers = sorted({row["speaker"] for row in turns if row["speaker"] != "UNKNOWN"})
    lines = ["# Full podcast transcript — original Nemotron timeline", "",
             "Machine-generated, not a verified verbatim record. Speaker numbers are acoustic estimates, not identified people.", "",
             "Words on overlapping voices can be unresolved or incorrectly assigned. The full-recording snapshot predates the later gain-fix experiments.", "",
             "[Podcast audio](audio.mp3?raw=1) · [Results](README.md) · Interactive playback: download this folder and open `index.html`.", ""]
    for row in turns:
        start, end = row["start"], row["end"]
        left, right = f"{int(start//60)}:{start%60:05.2f}", f"{int(end//60)}:{end%60:05.2f}"
        speaker = "Unassigned voice" if row["speaker"] == "UNKNOWN" else f"Speaker {speakers.index(row['speaker'])+1}"
        flags = (" · overlap" if row["overlap"] else "") + (" · uncertain attribution" if row["uncertain"] else "")
        # Render text as data even if speech contains Markdown/HTML-like syntax.
        text = escape(row["text"] or "[Voice detected; words unresolved]").replace("\n", " ")
        lines.extend((f"### {left}–{right} · {speaker}{flags}", "", f"<p>{text}</p>", ""))
    return "\n".join(lines)
