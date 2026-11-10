# airwave Demo

## airwave

airwave is an AI-powered podcast advertising system that transforms traditional ad breaks into **context-aware conversations**.

Instead of interrupting a podcast with a generic pre-recorded advertisement, airwave understands what the hosts are discussing and rewrites the advertisement so it naturally continues the conversation. If a podcast is talking about productivity, the ad begins from productivity. If it's about travel, the ad transitions into a travel-related message. Every advertisement remains clearly labeled as sponsored—it simply feels relevant rather than disruptive.

What makes airwave unique is that **every listener can hear a different version of the same advertisement**. The advertisement is personalized based on listener interests, language, and advertiser preferences while remaining contextually aligned with the podcast. Advertisers can choose the voice used for the ad, including a licensed host voice, celebrity voice, or a default brand voice, with support for English and multiple Indian languages.

The result is advertising that feels like a continuation of the conversation instead of a break in it.

📖 **[Read how airwave works](HOW_IT_WORKS.md)** for the complete architecture and processing pipeline.

---

# How It Works

```text
Podcast Audio
      │
      ▼
 Speech-to-Text + Speaker Diarization
      │
      ▼
Conversation Understanding
      │
      ▼
Find Natural Insertion Points
      │
      ▼
Advertiser Selection
      │
      ▼
Context-Aware Ad Generation
      │
      ▼
Personalization
(Language • Voice • Listener Interests)
      │
      ▼
Text-to-Speech
      │
      ▼
Seamless Audio Stitching
      │
      ▼
Final Podcast Stream
```

### 1. Audio Understanding

The uploaded podcast is processed using a local speech recognition pipeline that generates:

- speaker-separated transcripts
- timestamps
- speaker turns
- conversation boundaries

This creates a structured representation of the episode instead of a single transcript.

---

### 2. Finding Natural Ad Moments

airwave analyzes the conversation to identify moments where an advertisement can be inserted naturally.

The ranking combines signals such as:

- completed thoughts
- sentence endings
- speaker transitions
- pauses
- silence duration
- conversational flow

Instead of interrupting mid-sentence, airwave inserts ads only after meaningful conversational boundaries.

---

### 3. Understanding Context

The surrounding conversation is analyzed to determine:

- current topic
- intent
- entities being discussed
- conversation flow

Example:

```
Podcast

Coffee
↓

Focus
↓

Productivity
```

This context becomes the foundation for generating the advertisement.

---

### 4. Campaign Selection

airwave compares the current conversation against available advertiser campaigns.

Campaigns include information such as:

- product
- audience
- messaging
- offers
- brand constraints

The most relevant campaign is selected for that moment.

---

### 5. Context-Aware Advertisement Generation

Instead of inserting a fixed advertisement, airwave generates a new script that naturally continues the discussion.

Example:

Podcast:

> "...coffee helps improve concentration..."

↓

Generated Advertisement:

> "Speaking of staying focused..."

The transition feels connected because the advertisement starts from the current topic before introducing the sponsor.

---

### 6. Personalization

The generated advertisement is customized for each listener.

Personalization can include:

- interests
- preferred language
- preferred voice
- advertiser targeting

This means two people listening to the same podcast can hear different advertisements while the podcast itself remains unchanged.

---

### 7. Voice Generation

The generated script is converted into speech using AI text-to-speech.

Advertisers can choose:

- brand voice
- licensed host voice
- licensed celebrity voice

The same advertisement can also be generated in multiple Indian languages.

---

### 8. Audio Assembly

airwave uses FFmpeg to assemble the final audio stream by combining:

```
Original Podcast
        +
Generated Advertisement
        +
Remaining Podcast
```

The listener hears a seamless transition from podcast to sponsored message and back to the original conversation.

---

# Tech Stack

- **Node.js** — Backend
- **FFmpeg** — Audio segmentation and stream assembly
- **Local STT + Speaker Diarization** — Speech recognition and conversation segmentation
- **Groq LLMs** — Context understanding, campaign selection, and advertisement generation
- **Gnani TTS** — Speech synthesis and voice cloning
- **IndexedDB** — Transcript and inference caching
- **Browser Audio API** — Waveform visualization and playback

---

# Running Locally

1. Copy `.env.example` to `.env`
2. Start the local DSP/STT project
3. Add your `GNANI_API_KEY`
4. Add your `GROQ_API_KEY`
5. Install FFmpeg and configure `FFMPEG_PATH`
6. Run:

```bash
node server.mjs
```

7. Open:

```
http://localhost:4175
```