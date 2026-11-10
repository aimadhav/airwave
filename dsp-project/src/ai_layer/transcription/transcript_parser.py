"""Transcript Parser for Direct Transcript Ingestion.

Parses arbitrary raw or formatted transcript text into structured VerifiedTranscript
and VerifiedTurn objects, enabling direct conversation analysis without requiring audio STT.

Supports:
  1. Standard dialogue: "Sarah: Let's discuss the project."
  2. Timestamped turns: "[00:01:15 - 00:01:30] Sarah: We decided to deploy on Friday."
                        "[01:15] Sarah: We decided to deploy on Friday."
  3. Subtitle / WebVTT / SRT format: "00:00:01.000 --> 00:00:04.000\nSarah: Hello"
  4. JSON turn arrays: [{"speaker": "Sarah", "text": "..."}]
  5. Multi-line turns: continuation lines attached to the previous speaker
  6. Unstructured plain text / notes: automatic paragraph & sentence segmentation
"""

import json
import re
from dataclasses import dataclass
from typing import Sequence

from ai_layer.models.base_models import TimeRange
from ai_layer.models.speaker_models import SpeakerProfile, VerifiedTranscript, VerifiedTurn


def _timestamp_to_seconds(ts_str: str) -> float:
    """Converts HH:MM:SS[.mmm] or MM:SS[.mmm] or SS[.mmm] to float seconds."""
    ts_str = ts_str.strip().replace(",", ".")
    parts = ts_str.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        if len(parts) == 1:
            return float(parts[0])
    except ValueError:
        pass
    return 0.0


@dataclass
class ParsedTurnRaw:
    speaker_name: str
    text: str
    start_time: float | None = None
    end_time: float | None = None


class TranscriptParser:
    """Robust parser for pasting arbitrary transcripts into verified conversation turns."""

    # Matches timestamp range e.g. [00:01:10 - 00:01:25], (01:10 - 01:25), 00:00:01 --> 00:00:04
    TIMESTAMP_RANGE_REGEX = re.compile(
        r"^[\[\(]?(\d{1,2}:\d{2}(?::\d{2})?(?:[\.,]\d+)?)\s*(?:-|-->|to)\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[\.,]\d+)?)[\]\)]?",
        re.IGNORECASE,
    )
    # Matches single timestamp e.g. [00:01:15] or (01:15)
    TIMESTAMP_SINGLE_REGEX = re.compile(
        r"^[\[\(](\d{1,2}:\d{2}(?::\d{2})?(?:[\.,]\d+)?)[\]\)]"
    )
    TS_PREFIX_REGEX = re.compile(
        r"^[\[\(]?(\d{1,2}:\d{2}(?::\d{2})?(?:[\.,]\d+)?)\s*(?:-|-->|to)\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[\.,]\d+)?)[\]\)]?\s*"
    )
    TS_SINGLE_REGEX = re.compile(r"^[\[\(](\d{1,2}:\d{2}(?::\d{2})?(?:[\.,]\d+)?)[\]\)]\s*")

    # Matches speaker: text line e.g. "Sarah: Hi all" or "**Sarah:** Hi all" or "Speaker 1 (00:05): Hi all"
    SPEAKER_LINE_REGEX = re.compile(
        r"^(?:[\-\*]\s*)?(?:[\[\(][\d:\.\s\->,]+[\]\)]\s*)?(?:\*\*|__|\*|\[|\()?\s*([A-Za-z0-9_\s\.\-]{1,40}?)\s*(?:\([^\)]+\)|\[[^\]]+\])?(?::(?:\*\*|__|\*|\]|\))|(?:\*\*|__|\*|\]|\))\s*:\s*|\s*:\s*)(.+)$"
    )
    # Matches bracketed speaker e.g. "[Sarah] Hello" or "(Sarah): Hello"
    BRACKET_SPEAKER_REGEX = re.compile(
        r"^[\[\(]([A-Za-z0-9_\s\.\-]{1,35}?)[\]\)]\s*(?::|-|–|—)?\s*(.+)$"
    )
    # Matches dash speaker e.g. "Sarah - Hello everyone"
    DASH_SPEAKER_REGEX = re.compile(
        r"^([A-Za-z0-9_\s\.\-]{1,30}?)\s+(?:-|–|—)\s+(.+)$"
    )

    NON_SPEAKER_WORDS = {
        "note", "notes", "summary", "summaries", "transcript", "transcripts",
        "example", "examples", "conversation", "conversations", "output",
        "context", "here", "sure", "topic", "session", "explanation",
        "conclusion", "agenda", "action item", "action items", "decision", "decisions",
        "fact", "facts", "highlight", "highlights", "single conversation among four people",
        "single conversation", "four people", "dialogue", "meeting", "sync",
        "http", "https", "url", "step", "steps", "question", "answer",
        "important", "warning", "caution", "tip", "tips", "edit", "update",
        "ps", "p.s.", "q", "a", "reason", "result", "results", "definition",
        "method", "methods", "source", "reference", "references", "see",
        "notice", "recall", "remember", "fyi", "todo", "action", "task",
        "discussion", "chat", "script", "prompt", "assistant", "ai", "overview",
        "input", "reply", "record", "recording", "call", "audio", "video", "interview",
    }

    # Matches inline or formatted speaker headers: **Speaker:** or Speaker: or [Speaker]:
    SPEAKER_CANDIDATE_REGEX = re.compile(
        r"(?:^|\n|(?<=[.!?])\s+|(?<=:)\s+|\s+(?=(?:\*\*|__|\*|\[|\()))"
        r"(?:([\[\(][\d:\.\s\->,]+[\]\)])\s*)?"
        r"(?:"
            r"(?:\*\*|__|\*|\[|\()\s*([A-Za-z0-9_\. ]{1,25}?)\s*(?::(?:\*\*|__|\*|\]|\))|(?:\*\*|__|\*|\]|\))\s*:)"
            r"|"
            r"([A-Z][a-zA-Z0-9_\.]*(?:\s+[A-Z0-9][a-zA-Z0-9_\.]*){0,2})\s*:"
        r")\s+"
    )

    def _parse_inline_or_formatted_dialogue(self, text: str) -> list[ParsedTurnRaw] | None:
        """Parses inline, markdown-bold, or multi-speaker dialogue within paragraphs or single lines."""
        if "-->" in text:
            return None  # Let SRT subtitle parser handle it

        matches = list(self.SPEAKER_CANDIDATE_REGEX.finditer(text))
        filtered: list[re.Match] = []
        for m in matches:
            raw_name = m.group(2) or m.group(3) or ""
            name = raw_name.strip().strip("*_[]():")
            name_lower = name.lower()
            if name_lower not in self.NON_SPEAKER_WORDS and 1 <= len(name.split()) <= 4:
                if not any(
                    stop in name_lower
                    for stop in [
                        "here is", "talks to", "everyone", "conversation",
                        "four people", "discussion between", "dialogue between",
                        "speaking with", "meeting notes", "transcript of"
                    ]
                ):
                    filtered.append(m)

        if not filtered:
            return None

        # If only 1 match at very start and text has multiple lines, allow line-based parser to process
        if len(filtered) < 2 and filtered[0].start() == 0 and "\n" in text:
            return None

        raw_turns: list[ParsedTurnRaw] = []
        for i in range(len(filtered)):
            m = filtered[i]
            speaker = (m.group(2) or m.group(3) or "Speaker").strip().strip("*_[]():")
            ts_str = m.group(1)
            start_ts, end_ts = None, None
            if ts_str:
                ts_str_clean = ts_str.strip()
                r_m = self.TS_PREFIX_REGEX.match(ts_str_clean)
                if r_m:
                    start_ts = _timestamp_to_seconds(r_m.group(1))
                    end_ts = _timestamp_to_seconds(r_m.group(2))
                else:
                    s_m = self.TS_SINGLE_REGEX.match(ts_str_clean)
                    if s_m:
                        start_ts = _timestamp_to_seconds(s_m.group(1))

            start_pos = m.end()
            end_pos = filtered[i + 1].start() if i + 1 < len(filtered) else len(text)
            t_text = text[start_pos:end_pos].strip().lstrip(": \t")
            if t_text:
                raw_turns.append(
                    ParsedTurnRaw(
                        speaker_name=speaker,
                        text=t_text,
                        start_time=start_ts,
                        end_time=end_ts,
                    )
                )

        return raw_turns if raw_turns else None

    def parse_to_verified_transcript(
        self,
        text: str,
        session_id: str,
        known_speakers: Sequence[SpeakerProfile] | None = None,
    ) -> VerifiedTranscript:
        """Parses transcript text and produces a VerifiedTranscript."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return VerifiedTranscript(session_id=session_id, turns=[])

        # 1. Attempt JSON parsing first
        raw_turns = self._try_parse_json(cleaned_text)

        # 2. Attempt inline or formatted dialogue parsing (e.g. **Speaker:**, [Speaker]:, or multiple speakers in one paragraph/block)
        if not raw_turns:
            raw_turns = self._parse_inline_or_formatted_dialogue(cleaned_text)

        # 3. If not inline/formatted, attempt line-based / subtitle / plain-text parsing
        if not raw_turns:
            raw_turns = self._parse_line_based(cleaned_text)

        # 4. Fallback to unstructured text
        if not raw_turns:
            raw_turns = self._parse_unstructured_text(cleaned_text)

        # Build speaker lookup map from known_speakers
        speaker_map: dict[str, tuple[str, str]] = {}
        if known_speakers:
            for profile in known_speakers:
                speaker_map[profile.canonical_name.lower()] = (
                    profile.speaker_id,
                    profile.canonical_name,
                )
                for alias in profile.aliases:
                    speaker_map[alias.lower()] = (
                        profile.speaker_id,
                        profile.canonical_name,
                    )

        # Convert raw turns to VerifiedTurn with synthesized timestamps if missing
        verified_turns: list[VerifiedTurn] = []
        current_time = 0.0

        for index, raw in enumerate(raw_turns, start=1):
            speaker_clean = raw.speaker_name.strip()
            speaker_lower = speaker_clean.lower()

            if speaker_lower in speaker_map:
                spk_id, spk_name = speaker_map[speaker_lower]
            else:
                normalized_id = re.sub(r"[^a-zA-Z0-9_]", "", speaker_clean.lower().replace(" ", "_"))
                spk_id = f"spk_{normalized_id}" if normalized_id else f"spk_{index}"
                spk_name = speaker_clean if speaker_clean else f"Speaker {index}"

            # Calculate timestamps
            if raw.start_time is not None and raw.end_time is not None:
                start_ts = raw.start_time
                end_ts = max(raw.end_time, start_ts + 1.0)
                current_time = end_ts
            elif raw.start_time is not None:
                start_ts = raw.start_time
                word_count = max(len(raw.text.split()), 3)
                end_ts = start_ts + (word_count * 0.4)
                current_time = end_ts
            else:
                start_ts = current_time
                word_count = max(len(raw.text.split()), 3)
                duration = max(3.0, word_count * 0.4)
                end_ts = round(start_ts + duration, 2)
                current_time = end_ts + 0.5

            turn_id = f"turn_{index}"
            verified_turns.append(
                VerifiedTurn(
                    turn_id=turn_id,
                    speaker_id=spk_id,
                    speaker_name=spk_name,
                    text=raw.text.strip(),
                    time_range=TimeRange(
                        start_time_seconds=float(start_ts),
                        end_time_seconds=float(end_ts),
                    ),
                )
            )

        return VerifiedTranscript(
            session_id=session_id,
            turns=verified_turns,
            reassignments=[],
        )

    def _try_parse_json(self, text: str) -> list[ParsedTurnRaw] | None:
        """Tries to parse JSON turn array if text starts with '['."""
        if not (text.startswith("[") and text.endswith("]")):
            return None
        try:
            data = json.loads(text)
            if not isinstance(data, list) or not data:
                return None
            turns: list[ParsedTurnRaw] = []
            for item in data:
                if not isinstance(item, dict):
                    return None
                speaker = str(item.get("speaker") or item.get("speaker_name") or "Speaker")
                turn_text = str(item.get("text") or item.get("content") or "")
                start = float(item["start"]) if "start" in item else None
                end = float(item["end"]) if "end" in item else None
                turns.append(ParsedTurnRaw(speaker_name=speaker, text=turn_text, start_time=start, end_time=end))
            return turns
        except Exception:
            return None

    def _parse_line_based(self, text: str) -> list[ParsedTurnRaw]:
        """Parses lines formatted with speaker headers, timestamps, or subtitle formats."""
        lines = text.splitlines()
        raw_turns: list[ParsedTurnRaw] = []
        pending_time_range: tuple[float, float] | None = None
        current_turn: ParsedTurnRaw | None = None

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            # Check if line is a subtitle index number (SRT)
            if line.isdigit():
                continue

            # Check for standalone timestamp range: "00:00:01,000 --> 00:00:04,000" or "[00:10 - 00:25]"
            range_match = self.TIMESTAMP_RANGE_REGEX.match(line)
            if range_match:
                start_sec = _timestamp_to_seconds(range_match.group(1))
                end_sec = _timestamp_to_seconds(range_match.group(2))
                pending_time_range = (start_sec, end_sec)
                # Check if there is remaining text after the timestamp range on the same line
                remainder = line[range_match.end() :].strip().lstrip(":]- ")
                if remainder:
                    spk_match = self.SPEAKER_LINE_REGEX.match(remainder)
                    if spk_match:
                        current_turn = ParsedTurnRaw(
                            speaker_name=spk_match.group(1).strip(),
                            text=spk_match.group(2).strip(),
                            start_time=start_sec,
                            end_time=end_sec,
                        )
                        raw_turns.append(current_turn)
                        pending_time_range = None
                continue

            # Check for speaker line: "Speaker Name: Message text"
            spk_match = self.SPEAKER_LINE_REGEX.match(line)
            if not spk_match:
                spk_match = self.BRACKET_SPEAKER_REGEX.match(line)
            if not spk_match:
                spk_match = self.DASH_SPEAKER_REGEX.match(line)
            if spk_match:
                spk_name = spk_match.group(1).strip().strip("*_[]():")
                content = spk_match.group(2).strip()

                start_ts, end_ts = None, None
                if pending_time_range:
                    start_ts, end_ts = pending_time_range
                    pending_time_range = None
                else:
                    # Check if timestamp was in brackets inside speaker or line
                    single_match = self.TIMESTAMP_SINGLE_REGEX.match(line)
                    if single_match:
                        start_ts = _timestamp_to_seconds(single_match.group(1))

                current_turn = ParsedTurnRaw(
                    speaker_name=spk_name,
                    text=content,
                    start_time=start_ts,
                    end_time=end_ts,
                )
                raw_turns.append(current_turn)
                continue

            # Continuation line of previous speaker
            if current_turn is not None:
                current_turn.text = f"{current_turn.text} {line}".strip()
            else:
                # First line without speaker colon: treat as a turn with Speaker_1
                current_turn = ParsedTurnRaw(
                    speaker_name="Speaker_1",
                    text=line,
                    start_time=pending_time_range[0] if pending_time_range else None,
                    end_time=pending_time_range[1] if pending_time_range else None,
                )
                pending_time_range = None
                raw_turns.append(current_turn)

        # Decompose any turns that contain embedded multi-speaker dialogue
        decomposed: list[ParsedTurnRaw] = []
        for turn in raw_turns:
            sub = self._parse_inline_or_formatted_dialogue(turn.text)
            if sub and len(sub) > 1:
                decomposed.extend(sub)
            else:
                decomposed.append(turn)

        # Drop leading preamble turn if explicit named turns follow
        if len(decomposed) > 1 and decomposed[0].speaker_name == "Speaker_1":
            first_text_lower = decomposed[0].text.lower()
            if any(
                p in first_text_lower
                for p in ["here is", "sure.", "sure,", "transcript:", "conversation among", "dialogue:"]
            ):
                decomposed.pop(0)

        return decomposed

    def _parse_unstructured_text(self, text: str) -> list[ParsedTurnRaw]:
        """Fallback for unstructured text without speaker tags, splitting by paragraphs."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        turns: list[ParsedTurnRaw] = []
        for i, p in enumerate(paragraphs, start=1):
            turns.append(
                ParsedTurnRaw(
                    speaker_name=f"Speaker_{((i - 1) % 3) + 1}",
                    text=p.replace("\n", " "),
                )
            )
        return turns
