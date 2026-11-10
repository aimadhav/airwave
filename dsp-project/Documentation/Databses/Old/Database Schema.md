---
tags: [database, moc]
type: moc
---

# Database Schema — Voice/AI Personal Assistant

Map of content for the knowledge-base schema behind the Audio → Voice Synthesis → AI pipeline.

## Entity tables

- [[Sessions]] — top-level container for a thread of interaction
- [[Analysis]] — one "ask the AI" run within a session
- [[Transcript]] — stored, speaker-tagged transcript of one audio file
- [[Speakers]] — known/identified speakers (contacts-style)

## Junction tables

- [[Analysis_Transcripts]] — Analysis ↔ Transcript (many-to-many)
- [[Transcript_Speakers]] — Transcript ↔ Speakers (many-to-many)

## Entity-relationship overview

```
Sessions (1) ──< Analysis (many)
                    │
                    │ (many-to-many)
                    ▼
              Analysis_Transcripts
                    │
                    ▼
                Transcript (many)
                    │
                    │ (many-to-many)
                    ▼
              Transcript_Speakers
                    │
                    ▼
                Speakers (many)
```

## Key design changes vs. the original draft

1. **Analysis→Session FK moved**: a session has many analyses, so the FK must sit on `Analysis`, not `Sessions`.
2. **Analysis file paths → junction table**: multiple audio files per analysis, and transcripts explicitly meant to be reused, so this became a proper many-to-many (`Analysis_Transcripts`) instead of a raw list of paths.
3. **Transcript speaker list → junction table**: a transcript has multiple speakers, and a speaker spans multiple transcripts — modeled as `Transcript_Speakers` rather than a `Speaker Id's` column.

All four entity tables use a single-column surrogate `UUID` primary key; every foreign key is named `<referenced_table_singular>_id` for consistency.
