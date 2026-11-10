---
tags: [database, table, junction]
type: junction
---

# Analysis_Transcripts

Junction table resolving the many-to-many relationship between [[Analysis]] and [[Transcript]] — one analysis can draw on several audio clips/transcripts, and a single transcript can be reused across multiple later analyses without being rebuilt.

## Columns

| Column | Type | Key | Notes |
|---|---|---|---|
| `analysis_id` | UUID | **PK, FK** → [[Analysis]].`analysis_id` | |
| `transcript_id` | UUID | **PK, FK** → [[Transcript]].`transcript_id` | |

Composite primary key: (`analysis_id`, `transcript_id`).

## Relationships

- **Many** → **One** [[Analysis]]
- **Many** → **One** [[Transcript]]
