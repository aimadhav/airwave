---
tags: [database, table, junction]
type: junction
---

# Transcript_Speakers

Junction table resolving the many-to-many relationship between [[Transcript]] and [[Speakers]] — a transcript can contain 2+ speakers, and a speaker recurs across many transcripts.

## Columns

| Column | Type | Key | Notes |
|---|---|---|---|
| `transcript_id` | UUID | **PK, FK** → [[Transcript]].`transcript_id` | |
| `speaker_id` | UUID | **PK, FK** → [[Speakers]].`speaker_id` | |

Composite primary key: (`transcript_id`, `speaker_id`).

Segment-level timestamps for *when* each speaker talks already live inside the transcript file itself (per the Voice Synthesis timestamp flow), so they aren't duplicated here — this table only records *which* speakers appear in *which* transcript.

## Relationships

- **Many** → **One** [[Transcript]]
- **Many** → **One** [[Speakers]]
