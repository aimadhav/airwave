# Sublayer 04: Database Management & Knowledge Updating

## 1. Role & Purpose

The **Database Management & Knowledge Updating Sublayer** integrates newly extracted conversation insights into the permanent database. It ensures that the system's memory stays accurate, organized, and up to date over time, handling cases where new discussions update, contradict, or reinforce past information.

---

## 2. Inputs

- **New Candidate Insights**: The extracted facts, decisions, and tasks from the current session.
- **Historical Database Records**: The existing repository of active and past knowledge gathered across previous conversations.

---

## 3. Step-by-Step Working

1. **Historical Record Lookup**:
   - For each newly extracted statement, the sublayer searches existing database records to find previously recorded information on the same subject or topic.

2. **Relationship Classification**:
   When related existing records are found, the sublayer compares the new statement with the existing record to determine their relationship:
   - **Reinforce**: The new statement confirms what was already known (e.g., repeating an ongoing policy). The existing record is marked as reaffirmed, refreshing its recency without creating an unnecessary duplicate.
   - **Supersede**: The new statement represents an update or change of plans (e.g., a past decision was to use Vendor A, but the new decision is to switch to Vendor B). The older record is updated to a "superseded" status, and the new record becomes the active truth.
   - **Contradict**: The new statement conflicts with an earlier statement without an explicit explanation of change. Both records are flagged to indicate an unresolved discrepancy for user awareness.
   - **Elaborate**: The new statement adds additional detail, dependencies, or context to an existing topic. A connection is drawn linking the new information to the existing record.
   - **Novel**: The statement describes completely new information with no prior history. It is inserted directly as a new active record.

3. **Knowledge Connection & Edge Mapping**:
   - Relational links are recorded between connected ideas (e.g., marking which newer decision replaced an older one, and why).
   - This builds a coherent web of context that preserves the history of how ideas evolved over time.

4. **Change Logging**:
   - Every modification, status change, or replacement of past information is recorded in an audit log, preserving a history of what changed, when, and from which conversation.

---

## 4. Outputs

- **Updated Knowledge Records**: New and updated facts, decisions, and tasks stored in the database with current statuses (`active`, `superseded`, or `flagged`).
- **Knowledge Relationships**: Direct connections linking related ideas and recording how past plans were modified.
- **Audit Change Log**: A chronological trail of updates showing the progression of information across conversations.
