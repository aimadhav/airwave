# Description

A platform that combines speech processing with AI to manage your personal and professional life.

```text
Audio file
    ↓
Voice Synthesis
    ├─ De Noise
    ├─ Separation
    └─ Identification
           ↓
          AI
    ├─ Analyze
    ├─ Update Database
    ├─ Answer user query
    └─ Verify Identification
```

# Layers

## Audio file

Contains conversation / information

- Multiple People (2+)
- Single

## Voice Synthesis

1. De-Noise the audio file.
2. Analyze the audio to extract multiple speakers.
3. Try to mark different speakers using an unique id.
4. Create a timestamp flow of the audio with speaker id.
5. Separate these timestamps with the speaker id (Single file, allow multiple ids).

> Considering multiple people speaking at some time.

---

## AI Pipeline

1. Takes an list of the clips of the audio along with speaker id.
2. Pass each of the clips to an STT model and creates a transcript of the entire conversation with:
   - Speaker id
   - Content
   - Timestamps
   - Clip path
3. Pass the transcript to the LLM and ask it to analyze them.
4. LLM manages a database which consists of older data also.
5. The LLM gives answers to user questions by using the database.

---

## Database

- Database is our knowledge base for the AI.
- Stores conversations in an relational manner.

### Database Structure

- Multiple tables could be on dedicated purpose.
- Tables will have connections to other tables if required.

### Available Tables

#### Sessions

***Contains info about the session.***

**Columns**:

- Session Id
- Session Name
- Creation date & time
- Associated Analysis
- Session Summary Path
- Author *(Not sure if should keep this)*
- Category *(Not sure if should keep this)*

> Analysis Id (foreign key)

---

#### Analysis

Contains details regarding the analysis.

- Each session can have multiple analysis.
- Each analysis has:
  - Chat
  - Audio file

**Columns**:
- Analysis Id
- Date-Time
- File Paths *(Multiple files allowed)*
- Response Path
	*Response contains the following:*
	- User query
	- Key insights from the audio
	- LLM answer

---

#### Transcript
***This is the transcript of a audio file stored locally so that can be refereed when needed again without re constructing it again, stored in an proper conversation style With Proper Time Stamps and User_Id tags***
- File Path
- Speaker Id's
- Transcript Path

---

#### Speakers

***This table contains the info about the speakers identified by the system and holds them allowing the user to filter conversation regarding some People only  ( like contacts ) ***

- Speaker Id
- Name
- Details
- Audio Profile