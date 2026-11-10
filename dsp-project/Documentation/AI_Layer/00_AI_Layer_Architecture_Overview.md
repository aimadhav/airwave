# Layer 2: AI Layer — Architecture Overview

## 1. Role of the AI Layer

The AI Layer sits directly after the Speech Processing Layer. Its job is to take the processed audio clips and speaker IDs, understand the spoken conversation, maintain a long-term knowledge base across past and current conversations, and answer user questions.

Based on the system requirements, the AI Layer performs four primary functions:
- **Transcribe & Structure**: Converts audio clips into a chronological, speaker-attributed transcript.
- **Verify Identification**: Uses conversational cues in the text to verify and correct speaker identities.
- **Analyze**: Extracts key insights, decisions, facts, action items, and summaries from the conversation.
- **Manage Database**: Compares new information with historical records, connects related knowledge, and updates the database.
- **Answer User Queries**: Uses the accumulated database of conversations to answer questions from the user.

---

## 2. Sublayers and Their Responsibilities

The AI Layer is divided into distinct, dedicated sublayers:

```text
Processed Audio Clips (from Layer 1)
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. STT Transcription Sublayer                               │
│    Converts audio clips into a structured conversation      │
│    transcript with speaker IDs and timestamps.              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Speaker Verification Sublayer                            │
│    Inspects dialogue cues (names, introductions) to         │
│    confirm or correct speaker identity assignments.         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Conversation Analysis Sublayer                           │
│    Extracts key insights, facts, decisions, and action      │
│    items from the transcript.                               │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Database Management Sublayer                             │
│    Compares new findings against older data, connects       │
│    related information, and updates the knowledge base.     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Session & Project Summarization Sublayer                 │
│    Synthesizes session-level summaries and updates          │
│    overall project progress overviews.                      │
└─────────────────────────────────────────────────────────────┘

User Query Interface:
┌─────────────────────────────────────────────────────────────┐
│ 6. User Query Answering Sublayer                            │
│    Takes user questions, retrieves relevant context from     │
│    the database, and produces answers with citations.       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. End-to-End Operational Flow

1. **Audio Ingestion**: Layer 1 delivers a collection of audio clips. Each clip has a start timestamp, end timestamp, and a preliminary speaker ID.
2. **Transcription**: The STT sublayer transcribes each clip into text and combines them in chronological order.
3. **Identity Verification**: The verification sublayer checks if the spoken words reveal who is actually talking (e.g., introductions or addressing someone) and updates speaker labels if necessary.
4. **Insight Extraction**: The analysis sublayer reads the verified transcript and extracts discrete pieces of information: facts stated, decisions made, and tasks assigned.
5. **Knowledge Integration**: The database management sublayer checks existing database records to see if the new information reinforces, updates, or contradicts older information, and saves the updated state.
6. **Summary Generation**: Summaries are produced for the individual session and updated for any associated ongoing project.
7. **Query Answering**: When a user asks a question, the query answering sublayer searches the historical database to provide an accurate answer referencing specific conversations and speakers.

---

## 4. Modes of Operation

The architecture cleanly decouples the **STT Layer** from the **Main Analyzer**:

### Mode 1: Audio Processing Pipeline
- **Input**: Audio recordings (.wav, .mp3, .webm, mic, url) segmented into clips.
- **Workflow**: Sliced clips -> STT Transcription -> Speaker Verification -> Main Analyzer (facts, decisions, action items, summaries) -> Long-Term Database.

### Mode 2: Direct Transcript Ingestion (STT Bypass)
- **Input**: Pasted or uploaded transcript text (dialogue, timestamps, subtitles, or JSON).
- **Workflow**: Raw Text -> `TranscriptParser` -> Main Analyzer (`MainAnalyzerService`) -> Long-Term Database.
- **Benefits**: Instant analysis of pre-existing meeting notes, Google Meet / Zoom / Teams transcripts, or text logs without running speech recognition models.

