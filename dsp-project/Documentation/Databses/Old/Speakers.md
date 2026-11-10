---
tags: [database, table]
type: entity
---

# Speakers

Known/identified speakers, functioning like a contacts list — lets the user filter conversations down to specific people.

## Columns

| Column | Type | Key | Notes |
|---|---|---|---|
| `speaker_id` | UUID | **PK** | Unique identifier assigned during Voice Synthesis identification |
| `name` | text | | May be null until the speaker is identified/named |
| `details` | text | | Free-form notes about the speaker |
| `audio_profile_path` | text | | Path to the stored voice-print/embedding used for re-identification |

## Relationships

- **Many** `Speakers` ↔ **Many** [[Transcript]], via junction [[Transcript_Speakers]]

## Dataview

```dataview
TABLE name, details
FROM #database/speaker-record
SORT name ASC
```
