# Sublayer 03: Conversation Analysis & Insight Extraction

## 1. Role & Purpose

The **Conversation Analysis & Insight Extraction Sublayer** reads the verified conversation transcript and distills it into meaningful, structured information. Instead of treating the conversation as an unstructured block of text, it breaks down the discussion into atomic units of knowledge: what was established as true, what choices were agreed upon, and who committed to doing what.

---

## 2. Inputs

- **Verified Conversation Transcript**: The full, ordered sequence of utterances with validated speaker identities and timestamps.
- **Session Details**: Basic metadata including session title, date, time, and list of participants.

---

## 3. Step-by-Step Working

1. **Transcript Comprehension**:
   - The sublayer analyzes the dialogue flow, following topic shifts, discussions, arguments, and conclusions reached by participants.
   - Casual chit-chat, greetings, and conversational fillers are identified and filtered out.

2. **Categorized Information Extraction**:
   The sublayer extracts specific categories of structured information from the dialogue:
   - **Facts**: Objective statements, status updates, or pieces of information affirmed during the conversation (e.g., *"The client renewed the annual contract on Monday"*).
   - **Decisions**: Explicit conclusions, agreements, or choices finalized by the group (e.g., *"The team decided to reschedule the launch date to October 15"*).
   - **Tasks & Action Items**: Commitments to take specific actions, identifying who is responsible and any mentioned deadlines (e.g., *"David will prepare the slide deck by Friday"*).

3. **Normalization & Contextualization**:
   - Extracted statements are rewritten into self-contained sentences so they remain fully understandable without reading the surrounding dialogue (e.g., replacing pronouns like *"he will finish it tomorrow"* with explicit names and dates).

4. **Provenance & Citation Linking**:
   - Every extracted fact, decision, and task is explicitly linked back to the exact utterance or group of utterances where it was spoken.
   - This ensures that every extracted insight can always be traced back to who said it and at what timestamp.

---

## 4. Outputs

- **Candidate Facts**: A list of verified factual statements made during the meeting.
- **Candidate Decisions**: A list of agreed-upon decisions.
- **Action Items & Tasks**: A checklist of commitments with assigned owners, deadlines, and source utterance references.
- **Utterance Citations**: Explicit links tying each extracted insight to specific speaker timestamps.
