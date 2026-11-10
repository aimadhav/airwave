---
tags: [database, table]
type: entity
---

# Transcript

The stored, speaker-tagged transcript of one audio file — kept on disk so it never has to be reconstructed twice. This is the output of Voice Synthesis (diarization) + STT.

## Columns

| Column | Type | Key | Notes |
|---|---|---|---|
| `transcript_id` | UUID | **PK** | Unique identifier |
| `source_file_path` | text | | Path to the original audio file |
| `transcript_path` | text | | Path to the stored transcript (conversation-style, with timestamps + speaker tags) |
| `created_at` | datetime | | When the transcript was generated |

Note: `Speaker Id's` from the original spec is **not** a column here — a transcript can involve multiple speakers, and a speaker can appear in multiple transcripts, so that relationship is modeled as its own many-to-many table ([[Transcript_Speakers]]) rather than a column.

## Relationships

- **Many** `Transcript` ↔ **Many** [[Analysis]], via junction [[Analysis_Transcripts]]
- **Many** `Transcript` ↔ **Many** [[Speakers]], via junction [[Transcript_Speakers]]

## Dataview

```dataview
TABLE source_file_path, created_at
FROM #database/transcript-record
SORT created_at DESC
```
