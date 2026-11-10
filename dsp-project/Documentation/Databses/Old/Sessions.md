---
tags: [database, table]
type: entity
---

# Sessions

Top-level container. A session represents one ongoing thread of interaction (e.g. one user's use of the system over time, or one grouped context).

## Columns

| Column | Type | Key | Notes |
|---|---|---|---|
| `session_id` | UUID | **PK** | Unique identifier |
| `session_name` | text | | Display name |
| `created_at` | datetime | | Creation date & time |
| `summary_path` | text | | Path to session summary file on disk |
| `author` | text | | Optional — who created/owns the session |
| `category` | text | | Optional — for filtering/grouping sessions |

## Relationships

- **One** `Sessions` → **Many** [[Analysis]], via `Analysis.session_id`

> [!note] Change from original spec
> The original draft put an `Analysis Id (foreign key)` column on **Sessions**. Since a session can have *multiple* analyses, the foreign key has to live on the "many" side ([[Analysis]]), not here — otherwise a session could only ever point to one analysis. Removed from this table.

## Dataview

```dataview
TABLE session_name, created_at, category
FROM #database/session-record
SORT created_at DESC
```
