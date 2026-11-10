# Sublayer 06: User Query Answering

## 1. Role & Purpose

The **User Query Answering Sublayer** allows users to ask natural questions about their past conversations, meetings, commitments, and discussions, providing accurate answers based strictly on the accumulated database.

---

## 2. Inputs

- **User Query**: A question in natural language (e.g., *"What did we decide about the launch date?"* or *"What tasks did Sarah commit to last week?"*).
- **Database Knowledge Base**: The stored collection of conversation transcripts, extracted facts, decisions, action items, and project summaries across both recent and historical sessions.

---

## 3. Step-by-Step Working

1. **Query Intent Analysis**:
   - The sublayer analyzes the user's question to determine what information is needed:
     - Is the user asking about a specific person, a project, a decision, or a timeline?
     - Does the question relate to a recent event or historical decisions?

2. **Context Retrieval from Database**:
   - The sublayer searches the database for knowledge records matching the topic of the query.
   - It filters and prioritizes retrieved records based on:
     - **Relevance**: How closely the stored information addresses the question.
     - **Recency and Validity**: Favoring currently active decisions over superseded older ones, while still referencing historical context if the user asks about changes over time.
     - **Connected Ideas**: Following recorded relationships between linked records to gather background context.

3. **Ground-Truth Citation Resolution**:
   - For every relevant piece of information identified, the sublayer traces back to the original conversation turn.
   - It fetches the speaker's name, the recording date, the exact timestamps, and the original spoken sentence.

4. **Answer Formulation**:
   - The sublayer composes a clear, direct answer to the user's question.
   - The answer directly incorporates citations pointing to the original statements:
     - Who said it.
     - In which session or meeting.
     - The exact timestamp and spoken quote.
   - If the requested information is absent or unresolved in the database, the sublayer explicitly states that the records do not contain the answer.

---

## 4. Outputs

- **Direct Answer**: A natural language response addressing the user's question.
- **Source Citations**: References to specific speakers, session names, and timestamps backing up the answer.
