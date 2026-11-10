# Complete Audio DSP-to-AI Pipeline: Beginner Explainer

This document explains the complete Audio File pipeline for readers who are
not familiar with digital signal processing, speech recognition or artificial
intelligence.

It explains:

- what happens when people speak into a microphone;
- how a computer represents sound;
- how the project cleans and analyzes audio;
- how the system detects speech and speakers;
- how audio clips are sent to the AI layer;
- how speech becomes text and useful information;
- what is already implemented and what comes next.

The central idea is simple:

> We first prepare and organize the sound, then we ask AI to understand the
> organized audio.

## 1. What Problem Are We Solving?

Suppose two people have a conversation and we record it:

```text
Person A: We need to finish the testing.
Person B: I can complete it by Friday.
```

The microphone does not automatically understand the conversation. It records
one audio signal containing:

- Person A's voice;
- Person B's voice;
- silence and pauses;
- background noise;
- room echo;
- microphone noise.

The project converts that raw recording into something structured:

```text
[00:02–00:06] Speaker A:
We need to finish the testing.

[00:07–00:11] Speaker B:
I can complete it by Friday.

Action item:
Speaker B will complete testing by Friday.
```

To produce this result, the project uses two major layers.

## 2. The Two Major Layers

### Layer 1: DSP and audio processing

DSP means **Digital Signal Processing**. This layer works with the sound
waveform itself.

It answers questions such as:

- Is the audio file valid?
- When is someone speaking?
- Is the recording noisy?
- Which time interval belongs to which speaker?
- Are two people speaking at the same time?
- Which audio clip should be sent to the speech recognizer?

### Layer 2: AI processing

The AI layer works with speech and language.

It answers questions such as:

- What words were spoken?
- What decisions were made?
- What tasks were assigned?
- What is the summary of the conversation?
- What facts should be saved for later questions?

The two layers are connected, but they have different responsibilities.

```text
DSP layer:  organize the sound
AI layer:   understand the organized sound
```

## 3. Complete Pipeline Overview

```text
Audio file
    ↓
1. Ingestion and validation
    ↓
2. Standardization
    ↓
3. Noise reduction and conditioning
    ↓
4. Time-frequency analysis
    ↓
5. Voice activity detection
    ↓
6. Speaker diarization
    ↓
7. Overlap detection
    ↓
8. Source separation when necessary
    ↓
9. Speaker fingerprinting and identification
    ↓
10. Timestamped audio clip generation
    ↓
11. Speech-to-text transcription
    ↓
12. Conversation analysis
    ↓
13. Storage and search
```

Each step produces an output used by the next step. This makes the system
modular: a step can be improved without rewriting the entire pipeline.

## 4. Step 1: Audio Ingestion

### What is ingestion?

Ingestion means bringing an audio file into the system safely.

The input could be:

- a voice recording;
- a meeting recording;
- a lecture;
- an interview;
- a phone recording;
- a microphone capture.

The ingestion stage checks:

- whether the file exists;
- whether the format is supported;
- whether the file is corrupted;
- how many channels it contains;
- its sample rate;
- its duration;
- its bit depth.

### Why validate the input?

If a damaged or unsupported file reaches a later stage, the error becomes hard
to understand. Early validation gives a clear message such as:

```text
The audio file is empty or uses an unsupported format.
```

### Current project status

The first DSP implementation supports PCM WAV files and produces a typed
`AudioBuffer` containing samples and metadata. Additional format adapters can
be added later.

## 5. Step 2: Standardizing the Audio

Different recordings may use different settings:

```text
Recording A: 44.1 kHz, stereo
Recording B: 48 kHz, mono
Recording C: 8 kHz, mono
```

Processing all of them directly would make later algorithms inconsistent.

The project therefore uses a common internal format for speech processing:

- 16 kHz sample rate;
- one channel for the baseline pipeline;
- PCM WAV representation;
- floating-point samples close to the range -1 to +1.

### What is a sample rate?

Sound is continuous in the physical world. A computer measures it many times
per second. The number of measurements per second is the sample rate.

At 16 kHz:

```text
16,000 measurements are stored every second.
```

The sample rate must be high enough to preserve the frequencies needed for
speech.

### Stereo to mono

Stereo audio has two channels, usually left and right. The baseline DSP path
averages them into one channel:

```text
left channel + right channel
              ---------------- = mono channel
                       2
```

This is convenient for the first implementation. A future microphone-array
version must preserve all channels for spatial processing.

### Resampling

Resampling changes the number of samples per second. The current baseline uses
linear interpolation, which estimates values between neighboring samples.

For production use, resampling should also apply an anti-aliasing filter. This
prevents frequencies from being incorrectly folded into lower frequencies.

## 6. Step 3: Cleaning and Conditioning

Raw audio may contain:

- microphone hum;
- air-conditioner noise;
- electrical noise;
- handling noise;
- very low-frequency vibration;
- different volume levels;
- a small constant signal offset.

The conditioning stage prepares the waveform for analysis.

### Removing DC offset

The average value of an audio waveform should usually be close to zero. If the
average is not zero, the signal has a DC offset.

The system calculates the average and subtracts it from every sample:

```text
centered sample = original sample - average sample value
```

### Normalizing volume

Two microphones may record the same speaker at different volumes. Peak
normalization scales the signal to a safe standard level.

The implementation targets a peak near 0.95 instead of 1.0. The remaining
headroom reduces the risk of clipping during later operations.

### Filtering

A high-pass filter can remove extremely low-frequency rumble. Noise reduction
can reduce stationary background noise.

Filtering must be conservative. If it removes too much, it can damage speech
sounds and make transcription worse.

### Output

The system keeps both versions when possible:

```text
original audio
      ↓
normalized/enhanced audio
```

Keeping the original is important for debugging and comparison.

## 7. Step 4: Understanding Time and Frequency

There are two useful ways to look at audio.

### Time domain

The time-domain waveform shows amplitude changing over time.

```text
amplitude
    ↑       /\    /\
    │  /\  /  \__/  \
    └────────────────────────→ time
```

This helps us see duration, pauses and loudness.

### Frequency domain

The frequency domain shows which frequencies are present. A Fourier transform
converts a time-domain frame into frequency components.

```text
time signal → Fourier transform → frequency spectrum
```

Speech has changing frequency content, so one frequency spectrum for an entire
meeting is not enough.

### STFT and spectrograms

The Short-Time Fourier Transform, or STFT, divides audio into short overlapping
frames and calculates a spectrum for each frame.

The result can be displayed as a spectrogram:

```text
frequency ↑
          │ brightness = energy
          │
          └────────────────────→ time
```

The spectrogram is useful for:

- visualizing speech;
- seeing noise;
- understanding vowels and consonants;
- studying overlapping speakers;
- building source-separation masks.

The current implementation includes a NumPy STFT baseline.

## 8. Step 5: Voice Activity Detection

Voice Activity Detection, or VAD, determines which parts of the recording
contain speech-like activity.

It does not yet answer who is speaking. It only finds likely speech regions.

### How the baseline VAD works

1. Divide the signal into short frames.
2. Calculate the RMS energy of each frame.
3. Convert the energy to decibels.
4. Mark frames above a threshold as active.
5. Join active frames together.
6. Join regions separated by very short pauses.
7. Remove regions that are too short.
8. Convert sample positions into seconds.

### RMS energy

RMS measures the average magnitude of a frame:

```text
RMS = square root of the average of the squared samples
```

The baseline VAD compares frame levels with a threshold such as -35 dBFS.

### Example output

```text
00:00.00 – 00:02.10   silence
00:02.10 – 00:06.40   speech
00:06.40 – 00:07.00   silence
00:07.00 – 00:11.80   speech
```

### What VAD cannot do

Energy VAD can be confused by loud music, machinery or background noise. It
also cannot reliably separate two people who speak at the same time.

It is a deterministic baseline. A model-based VAD can be added behind the same
interface later.

## 9. Step 6: Speaker Diarization

Speaker diarization answers:

> Who spoke when?

It assigns anonymous labels to voice regions:

```text
SPEAKER_00: 00:02–00:06
SPEAKER_01: 00:07–00:11
SPEAKER_00: 00:12–00:15
```

The labels are initially anonymous. `SPEAKER_00` does not automatically mean
that the person is Raj or Priya.

### Diarization versus transcription

These are different tasks:

```text
Diarization:   who spoke and when?
Transcription: what did they say?
```

The DSP layer handles the first question. The AI layer handles the second.

### Diarization versus separation

Diarization labels time regions. Separation attempts to create separate audio
signals from a mixture.

```text
Diarization:  Speaker A spoke from 2–5 seconds.
Separation:   create an audio signal containing Speaker A.
```

The first MVP can use a deterministic mock diarizer so the rest of the
pipeline can be tested without a large model or GPU.

Later, the mock can be replaced with a local or neural provider.

## 10. Step 7: Overlap Detection

Normal conversations may contain interruptions:

```text
Speaker A:  00:10 ───────── 00:16
Speaker B:              00:13 ─────── 00:18
                         overlap
```

The timeline must allow two segments to exist at the same time. It should not
force everything into one speaker at a time.

An overlap segment should contain:

```json
{
  "speaker_id": "SPEAKER_01",
  "start_time": 13.0,
  "end_time": 16.0,
  "is_overlap": true
}
```

This information tells the next stage that ordinary transcription may be less
reliable and source separation may be useful.

## 11. Step 8: Source Separation

If two people talk at the same time, the microphone generally captures a sum:

```text
recorded audio = voice A + voice B + noise
```

Source separation tries to estimate the individual voices.

### MVP approach: controlled NMF experiment

Non-negative Matrix Factorization, or NMF, works with the non-negative
magnitude of a spectrogram.

It attempts to represent the spectrogram approximately as:

```text
magnitude spectrogram ≈ spectral patterns × time activations
```

The patterns can be grouped into estimated speakers, converted into masks and
reconstructed with an inverse STFT.

### Limitation

Single-channel NMF cannot perfectly know which frequency component belongs to
which speaker when both voices overlap in the same time-frequency region.

For the presentation, NMF is valuable because it demonstrates classical DSP
and can be evaluated on controlled mixtures with known source recordings.

### Future approaches

- multiple microphones and beamforming;
- Independent Component Analysis;
- neural source separation;
- pretrained speech-separation models.

These should be implemented behind a common `SeparationProvider` interface.

## 12. Step 9: Speaker Fingerprinting

After we have a speaker segment, we can calculate features describing the
voice.

### MFCC baseline

Mel-Frequency Cepstral Coefficients, or MFCCs, summarize the spectral shape in
a way related to human hearing.

The usual process is:

```text
audio
  → frames
  → windowing
  → FFT
  → Mel filter bank
  → logarithm
  → DCT
  → MFCC features
```

MFCCs are useful for a baseline speaker-recognition demonstration, but they
are not a unique guaranteed biometric identity.

### Matching

For a registered person, we store one or more reference feature vectors. For a
new segment, we extract features and compare them using cosine similarity.

```text
similarity = new vector compared with registered vector
```

The result may be:

```text
similarity >= threshold → known speaker
similarity < threshold  → unknown speaker
```

The system should not force a name when the confidence is low.

### Future improvement

Pretrained speaker embeddings such as ECAPA-TDNN can be used later. They are
usually more robust to different sentences and recording conditions.

## 13. Step 10: Timestamped Clip Generation

The DSP layer should create small audio clips for downstream processing.

```text
clips/
├── segment_001_SPEAKER_00.wav
├── segment_002_SPEAKER_01.wav
└── segment_003_SPEAKER_00.wav
```

Each clip is accompanied by metadata:

```json
{
  "segment_id": "segment_001",
  "speaker_id": "SPEAKER_00",
  "start_time": 2.1,
  "end_time": 6.4,
  "audio_path": "clips/segment_001_SPEAKER_00.wav",
  "confidence": 0.91,
  "is_overlap": false
}
```

The timestamps refer to the original session, not just to the individual clip.
This allows the complete conversation to be reconstructed in chronological
order.

## 14. Step 11: Speech-to-Text

The AI layer receives the generated clips. A speech-to-text provider converts
audio into written text.

Input:

```text
clip: segment_001.wav
speaker: SPEAKER_00
start: 2.1 seconds
end: 6.4 seconds
```

Output:

```text
text: "We need to finish the testing."
```

The final structured segment combines both results:

```json
{
  "segment_id": "segment_001",
  "speaker_id": "SPEAKER_00",
  "start_time": 2.1,
  "end_time": 6.4,
  "text": "We need to finish the testing.",
  "audio_path": "clips/segment_001.wav"
}
```

The DSP layer should not call a specific AI provider directly. It should pass a
stable contract to the existing transcription service.

## 15. Step 12: Conversation Analysis

Once the session has a structured transcript, the LLM can analyze it.

Possible outputs:

- short summary;
- topics;
- decisions;
- action items;
- deadlines;
- important facts;
- follow-up questions;
- speaker-specific commitments.

Example:

```text
Transcript:
Raj: We need to finish the testing.
Priya: I can complete it by Friday.

Extracted action item:
Priya will complete testing by Friday.
```

The LLM should use the transcript and metadata. It should not be expected to
repair missing timestamps or corrupted audio; those belong to earlier stages.

## 16. Step 13: Storage and Querying

The completed session can be saved in the existing storage layer.

Useful stored information includes:

- session ID;
- original audio path;
- processed audio path;
- segment IDs;
- speaker IDs;
- start and end times;
- clip paths;
- transcript text;
- confidence values;
- overlap flags;
- summaries and extracted action items.

This allows later queries such as:

```text
What did Priya agree to do?
When was the testing discussion?
Show all conversations involving Raj.
```

## 17. A Complete Example

Assume the input file is `team_meeting.wav`.

### Input

```text
Duration: 42 seconds
Speakers: Raj and Priya
Noise: air conditioner
Overlap: one interruption
```

### After ingestion

```text
Format: WAV
Sample rate: 44.1 kHz → 16 kHz
Channels: stereo → mono
```

### After preprocessing

```text
DC offset removed
Volume normalized
Low-frequency rumble reduced
```

### After VAD

```text
Speech intervals detected: 8
Silence intervals removed: 7
```

### After diarization

```text
SPEAKER_00: 5 segments
SPEAKER_01: 3 segments
```

### After fingerprint matching

```text
SPEAKER_00 → Raj, confidence 0.91
SPEAKER_01 → Priya, confidence 0.87
```

### After clip generation

```text
8 timestamped WAV clips created
1 clip marked as overlapping
```

### After transcription

```text
8 transcript segments created
```

### After LLM analysis

```text
Summary generated
2 action items extracted
1 deadline identified
```

## 18. What Is Already Implemented?

The current branch contains the first DSP foundation:

- typed `AudioBuffer` and audio metadata contracts;
- PCM WAV loading;
- stereo-to-mono conversion;
- linear resampling;
- DC-offset removal;
- peak normalization;
- NumPy STFT analysis;
- deterministic energy-based VAD;
- VAD confidence and timestamp objects;
- unit-test files and direct smoke-test coverage;
- beginner and mathematical DSP documentation.

The current branch does not yet contain the complete end-to-end pipeline.

## 19. What Comes Next?

The next implementation stages are:

1. Create speaker-segment and timeline contracts.
2. Add a deterministic mock diarizer for development and tests.
3. Preserve overlapping speaker intervals.
4. Generate timestamped audio clips.
5. Add MFCC fingerprint extraction and speaker matching.
6. Connect clips to the existing transcription service.
7. Add end-to-end integration tests.
8. Add NMF separation for controlled two-speaker mixtures.
9. Add stronger diarization and speaker-embedding providers.
10. Add a simple dashboard or CLI demonstration.

## 20. Testing the Pipeline

Testing should happen at multiple levels.

### Unit tests

Test one algorithm at a time:

- WAV loading;
- resampling;
- normalization;
- STFT dimensions;
- VAD intervals;
- timestamp validation;
- MFCC extraction;
- similarity matching.

### Integration tests

Test several stages together:

```text
audio file
  → preprocessing
  → VAD
  → diarizer
  → clip generator
```

### End-to-end tests

Use a small controlled recording or synthetic mixture and verify:

- clips are created;
- timestamps are preserved;
- speaker labels reach the transcript;
- the AI layer receives the expected contract;
- records can be stored and retrieved.

### Evaluation metrics

Different stages need different metrics:

| Stage | Example metric |
|---|---|
| Preprocessing | SNR improvement |
| VAD | Precision, recall and F1 |
| Diarization | Diarization Error Rate |
| Separation | SI-SDR or SDR |
| Speaker matching | Accuracy and unknown rejection rate |
| Transcription | Word Error Rate |
| Full system | End-to-end success rate and latency |

## 21. Important Limitations

The system cannot guarantee perfect results in every recording. Difficult
conditions include:

- two speakers talking at exactly the same time;
- strong room echo;
- very loud background noise;
- microphones placed far from speakers;
- speakers with similar voices;
- very short speech segments;
- recordings with clipping or missing data.

These limitations are normal for speech-processing systems. The project should
report them honestly and evaluate the system on controlled examples before
claiming production-level performance.

## 22. How to Present the Project

For a minor-project presentation, demonstrate one prepared two-speaker file.

Show the following sequence:

1. Upload the audio file.
2. Show file metadata.
3. Display the original waveform.
4. Display the cleaned waveform and spectrogram.
5. Highlight detected speech regions.
6. Display the speaker timeline.
7. Show speaker similarity scores.
8. Show generated audio clips.
9. Display the speaker-labelled transcript.
10. Display a summary and action item.

The most important result is not a perfect research-grade separator. It is a
clear, working demonstration that connects signal processing to AI:

```text
raw sound → organized speech clips → transcript → useful knowledge
```

## 23. Short Explanation for a Non-Technical Audience

> The microphone records a mixture of voices and noise. The DSP part cleans
> the recording, finds the parts where people speak, marks which speaker is
> active and cuts the recording into timestamped clips. The AI part reads those
> clips, converts them to text and extracts useful information such as summaries
> and tasks. The DSP part organizes the sound; the AI part understands it.

