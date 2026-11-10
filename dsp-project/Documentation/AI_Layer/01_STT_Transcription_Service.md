# Sublayer 01: STT Transcription

## 1. Role & Purpose

The **STT (Speech-to-Text) Transcription Sublayer** receives segmented audio clips from the speech processing layer and transforms the spoken audio into text. It organizes the resulting text chronologically to produce a coherent, full-conversation transcript attributed to individual speakers.

---

## 2. Inputs

This sublayer receives a collection of audio segments belonging to a recorded session. For each segment, the inputs are:

- **Audio Clip**: The audio data file for that specific segment.
- **Preliminary Speaker ID**: The speaker identifier assigned during voice separation.
- **Start Timestamp**: The exact second the speech segment began in the overall recording.
- **End Timestamp**: The exact second the speech segment ended in the overall recording.
- **Clip File Path**: The storage location of the audio segment.

---

## 3. Step-by-Step Working

1. **Audio Segment Ingestion**:

   - The sublayer takes the list of separated audio clips for the session.
   - It organizes the clips in sequential order based on their start timestamps.
2. **Speech-to-Text Conversion**:

   - Each audio clip is processed through a speech recognition model.
   - The spoken dialogue in the clip is decoded into written text.
   - Punctuation, capitalization, and sentence structures are generated for readability.
3. **Dialogue Context Stitching**:

   - Individual transcribed segments are assembled into a unified conversation flow.
   - Sequential segments spoken by the same person without interruption are grouped together into coherent turns.
   - Where overlapping segments occurred in the original recording, both speakers' lines are retained at their corresponding timestamps.
4. **Transcript Record Construction**:

   - For every utterance in the conversation, a structured entry is created containing:
     - Who spoke (Speaker ID)
     - What was said (Transcribed Content)
     - When it was said (Start and End Timestamps)
     - Where the original audio clip is located (Clip Path)

---

## 4. Outputs

- **Structured Utterance Records**: A complete sequence of timestamped dialogue entries representing the entire conversation.
- **Full Conversation Transcript**: A formatted transcript file stored for the session, ready to be read by the user or passed to the subsequent AI analysis sublayers.
