---
tags: [database, table]
type: entity
---

# Analysis

One analysis run within a session — a single "ask the AI about this audio" event, covering one or more audio clips, the resulting chat, and the LLM's response.

## Columns

| Column | Type | Key | Notes |
|---|---|---|---|
| `analysis_id` | UUID | **PK** | Unique identifier |
| `session_id` | UUID | **FK** → [[Sessions]].`session_id` | Which session this analysis belongs to |
| `created_at` | datetime | | Date-time of the analysis |
| `response_path` | text | | Path to the stored response file |

`response_path` points to a file containing:
- User query
- Key insights extracted from the audio
- LLM's answer

## Relationships

- **Many** `Analysis` → **One** [[Sessions]] (via `session_id`)
- **Many** `Analysis` ↔ **Many** [[Transcript]], via junction [[Analysis_Transcripts]]

> [!note] Change from original spec
> Original spec listed `File Paths (Multiple files allowed)` directly on `Analysis`. Since each of those files corresponds to an audio clip that gets its own [[Transcript]] record (and transcripts are explicitly meant to be *reusable* rather than rebuilt), this becomes a many-to-many relationship through [[Analysis_Transcripts]] instead of a raw file-path list. This also means an existing transcript can be attached to a new analysis without re-running Voice Synthesis / STT on it.

## Dataview

```dataview
TABLE session_id, created_at
FROM #database/analysis-record
SORT created_at DESC
```
