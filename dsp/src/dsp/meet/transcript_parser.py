"""Parsers for uploaded Google Meet transcript artifacts."""

import json
from pathlib import Path
import re

from dsp.contracts import MeetTranscriptEntry, TranscriptParseResult


_VTT_TIMING = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})\s+-->\s+"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})"
)
_TEXT_TIMING = re.compile(
    r"^\s*[\[\(]?(?P<start>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})\s*"
    r"(?:-->|-+)\s*(?P<end>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})[\]\)]?\s*(?P<body>.*)$"
)


def parse_transcript(path: str | Path) -> TranscriptParseResult:
    """Parse JSON, WebVTT or simple timestamped text by file extension."""

    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return parse_json_transcript(text)
    if suffix == ".vtt":
        return parse_vtt_transcript(text)
    if suffix == ".txt":
        return parse_timestamped_text(text)
    raise ValueError("Supported Meet transcript formats are .json, .vtt and .txt")


def parse_json_transcript(text: str) -> TranscriptParseResult:
    """Parse canonical entries or common Meet API-style key names."""

    data = json.loads(text)
    if isinstance(data, list):
        raw_entries = data
    elif isinstance(data, dict):
        raw_entries = data.get("entries", [])
    else:
        raise ValueError("Meet JSON transcript must be an array or an object with an entries array")
    if not isinstance(raw_entries, list):
        raise ValueError("Meet JSON transcript entries must be an array")
    warnings: list[str] = []
    entries: list[MeetTranscriptEntry] = []
    for index, raw in enumerate(raw_entries):
        start = _number(raw.get("start_time", raw.get("startTime")))
        end = _number(raw.get("end_time", raw.get("endTime")))
        if start is None or end is None:
            warnings.append(f"entry {index} has no usable timestamp and was skipped")
            continue
        speaker_name = str(raw.get("speaker_name", raw.get("speakerName", raw.get("speaker", "")))).strip()
        speaker_id = str(raw.get("speaker_id", raw.get("speakerId", ""))).strip()
        if not speaker_name and not speaker_id:
            warnings.append(f"entry {index} has no speaker and was marked unknown")
            speaker_name = "Unknown speaker"
            speaker_id = "SPK_UNKNOWN"
        elif not speaker_id:
            speaker_id = _speaker_id(speaker_name)
        elif not speaker_name:
            speaker_name = speaker_id
        entry = _make_entry(
            entry_id=str(raw.get("entry_id", raw.get("entryId", f"entry_{index:04d}"))),
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            start=start,
            end=end,
            text=str(raw.get("text", "")).strip(),
            language_code=raw.get("language_code", raw.get("languageCode")),
            warnings=warnings,
        )
        if entry is not None:
            entries.append(entry)
    return TranscriptParseResult(entries=_merge_adjacent(entries), warnings=warnings, raw_text=text)


def parse_vtt_transcript(text: str) -> TranscriptParseResult:
    """Parse WebVTT cues, including ``<v Speaker>`` and ``Speaker: text``."""

    lines = text.splitlines()
    entries: list[MeetTranscriptEntry] = []
    warnings: list[str] = []
    index = 0
    cue_index = 0
    while index < len(lines):
        match = _VTT_TIMING.search(lines[index])
        if not match:
            index += 1
            continue
        start = _parse_timestamp(match.group("start"))
        end = _parse_timestamp(match.group("end"))
        index += 1
        body_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            body_lines.append(lines[index].strip())
            index += 1
        body = " ".join(body_lines)
        speaker_name, spoken_text = _extract_speaker(body)
        if not speaker_name:
            warnings.append(f"VTT cue {cue_index} has no speaker and was marked unknown")
            speaker_name = "Unknown speaker"
            speaker_id = "SPK_UNKNOWN"
        else:
            speaker_id = _speaker_id(speaker_name)
        entry = _make_entry(
            entry_id=f"vtt_{cue_index:04d}",
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            start=start,
            end=end,
            text=spoken_text,
            language_code=None,
            warnings=warnings,
        )
        if entry is not None:
            entries.append(entry)
        cue_index += 1
    return TranscriptParseResult(entries=_merge_adjacent(entries), warnings=warnings, raw_text=text)


def parse_timestamped_text(text: str) -> TranscriptParseResult:
    """Parse lines such as ``[00:01.000 - 00:03.000] Alice: Hello``."""

    entries: list[MeetTranscriptEntry] = []
    warnings: list[str] = []
    for index, line in enumerate(text.splitlines()):
        match = _TEXT_TIMING.match(line)
        if not match:
            if line.strip():
                warnings.append(f"text line {index} has no usable timestamp and was skipped")
            continue
        body = match.group("body").strip()
        speaker_name, spoken_text = _extract_speaker(body)
        if not speaker_name:
            warnings.append(f"text line {index} has no speaker and was marked unknown")
            speaker_name = "Unknown speaker"
            speaker_id = "SPK_UNKNOWN"
        else:
            speaker_id = _speaker_id(speaker_name)
        entry = _make_entry(
            entry_id=f"text_{index:04d}",
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            start=_parse_timestamp(match.group("start")),
            end=_parse_timestamp(match.group("end")),
            text=spoken_text,
            language_code=None,
            warnings=warnings,
        )
        if entry is not None:
            entries.append(entry)
    return TranscriptParseResult(entries=_merge_adjacent(entries), warnings=warnings, raw_text=text)


def _make_entry(
    *,
    entry_id: str,
    speaker_id: str,
    speaker_name: str,
    start: float,
    end: float,
    text: str,
    language_code: str | None,
    warnings: list[str],
) -> MeetTranscriptEntry | None:
    if end <= start:
        warnings.append(f"{entry_id} has an invalid interval and was skipped")
        return None
    if not text:
        warnings.append(f"{entry_id} has no text and was skipped")
        return None
    return MeetTranscriptEntry(
        entry_id=entry_id,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        start_time=start,
        end_time=end,
        text=text,
        language_code=language_code,
    )


def _merge_adjacent(entries: list[MeetTranscriptEntry]) -> list[MeetTranscriptEntry]:
    ordered = sorted(entries, key=lambda item: (item.start_time, item.end_time))
    merged: list[MeetTranscriptEntry] = []
    for entry in ordered:
        if merged:
            previous = merged[-1]
            directly_adjacent = abs(previous.end_time - entry.start_time) <= 0.001
            if directly_adjacent and previous.speaker_id == entry.speaker_id:
                merged[-1] = MeetTranscriptEntry(
                    entry_id=previous.entry_id,
                    speaker_id=previous.speaker_id,
                    speaker_name=previous.speaker_name,
                    start_time=previous.start_time,
                    end_time=entry.end_time,
                    text=f"{previous.text} {entry.text}".strip(),
                    language_code=previous.language_code or entry.language_code,
                )
                continue
        merged.append(entry)
    return merged


def _extract_speaker(body: str) -> tuple[str, str]:
    voice_match = re.match(r"<v\s+([^>]+)>(.*)$", body, flags=re.IGNORECASE)
    if voice_match:
        return voice_match.group(1).strip(), _strip_tags(voice_match.group(2).strip())
    speaker_match = re.match(r"^([^:]{1,80}):\s*(.+)$", body)
    if speaker_match:
        return speaker_match.group(1).strip(), _strip_tags(speaker_match.group(2).strip())
    return "", _strip_tags(body)


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value).strip()


def _speaker_id(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"spk_{normalized}" if normalized else "SPK_UNKNOWN"


def _number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("seconds", value.get("value"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    raise ValueError(f"Invalid transcript timestamp: {value}")
