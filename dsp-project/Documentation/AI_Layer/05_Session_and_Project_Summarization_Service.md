# Sublayer 05: Session & Project Summarization

## 1. Role & Purpose

The **Session & Project Summarization Sublayer** produces human-readable narratives of spoken interactions at two levels:
- **Session Summaries**: A consolidated narrative of a single conversation, capturing the main themes, who participated, what was concluded, and next steps.
- **Project Summaries**: A rolling, high-level status overview for long-term projects that aggregates findings across multiple related conversations over time.

---

## 2. Inputs

- **Session Data**: The complete transcript and extracted insights (facts, decisions, tasks) for the current conversation.
- **Project Context (if tagged)**: Any existing project overview and cumulative history of past sessions linked to the same project.

---

## 3. Step-by-Step Working

1. **Session Summary Generation**:
   - The sublayer reviews the full transcript alongside the extracted facts, decisions, and action items from that session.
   - It synthesizes an executive summary document organized into clear sections:
     - **Meeting Overview**: Date, duration, participants, and primary purpose.
     - **Discussion Highlights**: Key narrative points organized by topic.
     - **Decisions Made**: Concise summary of what was agreed upon.
     - **Action Items**: Checklist of assigned commitments and deadlines.
   - The summary is linked directly to the session record for quick access.

2. **Project-Level Status Rollup**:
   - When a session is associated with an overarching project, the sublayer checks the previous project summary.
   - It determines what has changed based on the new session's decisions and completed tasks.
   - It updates the high-level project narrative to reflect the latest status, current obstacles, and updated roadmaps.
   - Older superseded milestones are archived, keeping the project summary concise and focused on the current state.

---

## 4. Outputs

- **Session Summary Document**: A structured narrative document summarizing the conversation, saved and linked to the session.
- **Updated Project Overview**: A refreshed high-level summary reflecting the latest cumulative progress of the project.
