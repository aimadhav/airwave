**DSP-BASED MULTI-SPEAKER SEPARATION AND VOICE FINGERPRINTING**

*A Clear, Step-by-Step Project Development Guide*

| Item | Description |
|---|---|
| Project domain | Digital Signal Processing, Speech Processing and Speaker Recognition |
| Main problem | Separate multiple overlapping voices and identify each speaker |
| Starting platform | Python on a laptop/desktop |
| Advanced platform | Microphone array + Raspberry Pi/edge computer |
| Recommended strategy | Build a simple DSP version first, then add advanced AI |

# 1. What Exactly Are We Building?

Imagine two people are talking at the same time in front of one or more microphones. The microphone does not normally give us two separate files. It records a combined signal. Our project tries to answer two questions: (1) Can we separate the combined recording into individual voices? and (2) After separation, can we determine whose voice each signal belongs to?

```
Person A voice ──┐
                  ├──→ Microphone(s) ──→ Mixed audio
Person B voice ──┘                          ↓
                                       DSP separation
                                        ↙        ↘
                                   Voice A      Voice B
                                      ↓            ↓
                                    MFCC         MFCC
                                      ↓            ↓
                                        Voice fingerprint
                                              ↓
                                       Speaker database
                                              ↓
                                     Speaker identification
```

Therefore, the project has two major technical modules: Speaker Separation and Voice Fingerprinting. They should be developed separately and then connected.

# 2. The Problem in Simple Mathematical Form

If Speaker 1 produces s₁(t) and Speaker 2 produces s₂(t), a simple microphone model is:

> x(t) = s₁(t) + s₂(t)

Here x(t) is the signal we actually receive. The difficult task is to estimate s₁(t) and s₂(t) from x(t). This is called the speech source-separation problem or, more generally, blind source separation.

In a real room the model is more complicated because sound reflects from walls and reaches microphones at different times. Background noise and different speaker volumes make the problem harder.

# 3. What Will Happen During the Project?

| Stage | Input | Main operation | Output |
|---|---|---|---|
| 1. Recording | Voice | Record audio | WAV files |
| 2. Preprocessing | Raw audio | Filtering/normalization | Clean audio |
| 3. Analysis | Clean/mixed audio | FFT + STFT | Spectrum/spectrogram |
| 4. Separation | Mixed audio | NMF/ICA/beamforming | Separate voices |
| 5. Fingerprinting | Separated voice | MFCC extraction | Feature vector |
| 6. Identification | Feature vector | Similarity/classifier | Speaker name |
| 7. Evaluation | Known test data | Metrics | Accuracy/results |
| 8. Real-time | Microphone input | Integrated pipeline | Live demonstration |

# 4. Phase 0 — Define the Scope

Do not begin with three or four speakers and real-time processing. Start with the smallest experiment that can prove the idea.

Recommended first target:

- Two registered speakers.
- One artificially mixed audio signal.
- Offline processing in Python.
- DSP separation using NMF as the initial single-channel method.
- MFCC-based speaker fingerprint.
- Cosine similarity for identification.

Once this works, increase complexity: real recordings, noise, unequal speaker volume, three speakers, multiple microphones, and finally real-time processing.

# 5. Phase 1 — Collect the Audio Data

What we do:

1. Select 2–3 speakers.
2. Record several clean sentences from each speaker.
3. Save files as WAV.
4. Keep training/registration recordings separate from testing recordings.
5. Use the same sample rate, preferably 16 kHz for the first experiment.

Why?

The system needs examples of each registered speaker to create a voice profile. Separate test recordings are necessary to prove that the system is not simply memorizing the original recording.

```
data/
├── clean/
│   ├── speaker_1/
│   └── speaker_2/
├── mixtures/
└── test/
```

# 6. Phase 2 — Preprocess the Audio

Raw recordings may have different loudness levels, silence, DC offset and background noise. Preprocessing makes the signal suitable for analysis.

6. Load the WAV file.
7. Convert to mono if required.
8. Resample all files to the same sampling frequency.
9. Remove DC offset.
10. Normalize amplitude.
11. Optionally apply a speech band-pass filter.
12. Optionally apply mild noise reduction.

What can occur?

- Too much filtering can remove useful speech information.
- Too little filtering leaves noise that may reduce separation quality.
- Different sampling rates can cause inconsistent features.

Solution: keep preprocessing simple at first and verify the waveform and audio by listening to it.

# 7. Phase 3 — Understand FFT and STFT

This is the first major DSP concept. A waveform tells us how amplitude changes with time. FFT tells us which frequencies are present. Speech changes with time, so we use STFT to see frequency content as it changes.

FFT:

> Time-domain signal → FFT → frequency-domain representation

STFT:

> Speech → short overlapping frames → FFT of each frame → time-frequency representation

The STFT result can be shown as a spectrogram. The horizontal axis is time, the vertical axis is frequency, and brightness/intensity represents signal strength.

Why do we need this?

Two voices may have different frequency patterns at some moments. By looking at the time-frequency plane, we can try to estimate which parts belong to each speaker.

# 8. Phase 4 — Create a Controlled Mixed Voice

Before using real simultaneous speech, create a mixture where the original sources are known. For example, take two clean recordings and add them together.

> mixed = speaker_1 + α × speaker_2

α controls the volume of Speaker 2. Test α = 1 for equal levels and then test unequal levels.

13. Load Speaker 1.
14. Load Speaker 2.
15. Make their durations compatible.
16. Normalize their levels.
17. Add the signals.
18. Save mixed.wav.
19. Keep the original two files as ground truth.

This is extremely important for the project because we know the correct answers. After separation, we can compare the estimated output with the original clean speech.

# 9. Phase 5 — Speaker Separation

This is the hardest part. The system receives mixed.wav and must estimate two individual speech signals.

## 9.1 Recommended first method: NMF

NMF stands for Non-Negative Matrix Factorization. It works on a non-negative magnitude spectrogram. The idea is to represent the spectrogram as a combination of smaller spectral patterns.

> |X| ≈ W × H

W represents spectral patterns and H represents how strongly those patterns appear over time. After factorization, we group components and create masks for the estimated speakers.

Practical flow:

20. Calculate STFT of mixed.wav.
21. Calculate the magnitude spectrogram.
22. Run NMF.
23. Select/group components corresponding to different sources.
24. Create a time-frequency mask for each speaker.
25. Apply each mask to the complex mixture STFT.
26. Use inverse STFT to reconstruct audio.
27. Listen to the outputs and compare them with the original voices.

Important limitation: because the voices overlap in frequency, NMF will not always perfectly know which component belongs to which speaker. This is why artifacts or residual cross-talk may occur.

## 9.2 ICA — when should we use it?

ICA (Independent Component Analysis) is more suitable when we have multiple microphone channels. For example, microphone 1 and microphone 2 receive different mixtures because the speakers are at different positions.

```
Mic 1 ──→ x₁(t)
Mic 2 ──→ x₂(t)
              ↓
             ICA
           ↙     ↘
     Speaker 1   Speaker 2
```

Therefore, do not claim that ICA is the best single-microphone method. If you choose ICA, use a multi-microphone experiment and clearly explain why the additional spatial information helps.

## 9.3 Beamforming — advanced ECE extension

If several microphones are arranged as an array, the same speaker reaches each microphone with different delays and amplitudes. Beamforming uses these differences to emphasize sound arriving from a chosen direction.

This makes microphone arrays a strong extension if the professor wants more hardware and DSP involvement.

# 10. What Will We See After Separation?

There are three possible outcomes.

| Outcome | Meaning | Action |
|---|---|---|
| Good separation | Each output mostly contains one speaker | Continue to fingerprinting |
| Partial separation | Each output contains the target speaker plus some other voice | Tune the separation method and parameters |
| Poor separation | Both outputs still sound like the mixture | Check STFT settings, NMF component grouping, recording quality, or switch to multi-microphone/neural separation |

Do not hide failure cases. Showing one successful case and one difficult case makes the project more scientifically credible.

# 11. Phase 6 — Voice Fingerprinting

After obtaining an individual speaker signal, we need a compact numerical representation of that voice. MFCC (Mel-Frequency Cepstral Coefficients) is the recommended first feature.

Concept:

> Voice → framing → windowing → FFT → Mel filter bank → log → DCT → MFCC

MFCC describes the spectral envelope of speech. Different speakers have different vocal-tract characteristics, pitch patterns and speaking characteristics, so their feature distributions can differ.

Important clarification:

Calling MFCC a "voice fingerprint" is a project-level description. MFCC alone is not a guaranteed unique biometric identifier. For a stronger modern system, speaker embeddings can be added later.

# 12. Phase 7 — Create the Speaker Database

The database contains voice samples and/or extracted features for registered speakers.

```
speaker_database/
├── Raj/
│   ├── sample_01.wav
│   ├── sample_02.wav
│   └── sample_03.wav
├── Speaker_2/
│   ├── sample_01.wav
│   └── sample_02.wav
└── Speaker_3/
    └── sample_01.wav
```

28. Record several sentences from each registered speaker.
29. Extract MFCCs from each recording.
30. Store the resulting features.
31. Create a representative vector/centroid or retain multiple feature samples.
32. Keep unseen recordings for testing.

# 13. Phase 8 — Identify the Speaker

For an unknown separated voice, extract the same MFCC features and compare them with the registered profiles.

> similarity = cosine_similarity(unknown_features, stored_features)

The speaker with the highest acceptable similarity is selected. If every similarity score is below a chosen threshold, the system should return "Unknown Speaker."

| Method | Why use it? | Difficulty |
|---|---|---|
| Cosine similarity | Simple and easy to explain | Low |
| Euclidean distance | Simple baseline | Low |
| DTW | Handles different speaking rates/sequences | Medium |
| SVM | Useful supervised classifier | Medium |
| Speaker embeddings | Modern and stronger representation | High |

Recommended order: cosine similarity first → SVM/other classifier → speaker embeddings.

# 14. Phase 9 — Integrate the Complete System

```
MIXED SPEECH
     ↓
PREPROCESSING
     ↓
  STFT/FFT
     ↓
SPEAKER SEPARATOR
   ↙        ↘
SPEAKER 1   SPEAKER 2
   ↓            ↓
 MFCC          MFCC
   ↓            ↓
FINGERPRINT   FINGERPRINT
   ↘            ↙
  SPEAKER DATABASE
        ↓
     MATCHING
        ↓
Speaker 1 → Person A
Speaker 2 → Person B
```

At this point, the project becomes an end-to-end system. The user provides mixed speech and the software returns separated audio plus speaker identities.

# 15. Phase 10 — Evaluation: How Do We Prove It Works?

A working demo is not enough for an academic project. We need measurable results.

| Part | Metric | What it tells us |
|---|---|---|
| Separation | SDR / SI-SDR where applicable | How close the separated source is to the true source |
| Noise robustness | SNR improvement | How much unwanted noise was reduced |
| Identification | Accuracy | Percentage of correctly identified speakers |
| Classification | Confusion matrix | Which speakers are being confused |
| Real-time | Latency / processing time | Whether the system can operate quickly enough |

For the first experiment, create several mixtures with different speaker volume ratios and different overlap percentages. Report the average performance rather than only the best example.

# 16. Phase 11 — Real-Time Implementation

Only after the offline pipeline is stable should we connect a microphone array or live microphone input.

```
Microphone array
      ↓
Live audio buffer
      ↓
Preprocessing
      ↓
    STFT
      ↓
 Separation
      ↓
Fingerprinting
      ↓
Identification
      ↓
Display + separated audio
```

For a practical college prototype, the laptop can perform the heavy computation. A Raspberry Pi can later be used as an edge device. Trying to run the complete advanced separation system on a small microcontroller at the beginning will make development unnecessarily difficult.

# 17. Phase 12 — Advanced AI Extension

After the DSP baseline is complete, add a pretrained neural speech-separation model. Possible research directions include Conv-TasNet, DPRNN, SepFormer or another current source-separation architecture.

The important academic comparison is:

> Classical DSP method ↔ Neural separation method

Compare separation quality, speaker-identification accuracy, computation time and robustness. This gives the project a stronger research component because you are measuring the trade-off between interpretable DSP and more powerful learned models.

# 18. Hardware and Software

| Category | Recommended choice | Purpose |
|---|---|---|
| Computer | Laptop/Desktop | Main development and processing |
| Microphones | 2–4 microphone array for advanced stage | Spatial information |
| Optional edge computer | Raspberry Pi | Real-time/edge demonstration |
| Programming | Python | Main implementation |
| Signal processing | NumPy, SciPy, Librosa | FFT, STFT, filters, MFCC |
| Visualization | Matplotlib | Waveform and spectrogram plots |
| ML | scikit-learn | Similarity/classification |
| Advanced AI | PyTorch | Neural separation/embeddings |
| Audio | SoundFile/sounddevice | Read/write/record audio |

# 19. Development Plan — What We Should Actually Do

| Time | Target |
|---|---|
| Week 1 | Learn FFT, STFT, spectrogram and basic filtering. |
| Week 2 | Collect speaker recordings and build the audio preprocessing pipeline. |
| Week 3 | Create two-speaker mixtures and implement NMF separation. |
| Week 4 | Evaluate and improve separation; test noise and unequal volume. |
| Week 5 | Implement MFCC extraction and create the speaker database. |
| Week 6 | Implement speaker matching and identification. |
| Week 7 | Integrate everything and generate graphs/results. |
| Week 8 | Add microphone array or real-time processing. |
| Week 9+ | Optional neural separator, speaker embeddings and comparison. |

# 20. What Problems Are Most Likely to Occur?

| Problem | Reason | Solution / Next step |
|---|---|---|
| Cannot separate voices | Strong spectral overlap | Tune NMF; test multi-mic or neural separation |
| Speaker identification wrong | Poor separated signal / weak fingerprint | Improve separation and use multiple reference samples |
| Noise causes errors | Noisy recording | Noise reduction and better microphone placement |
| One speaker dominates | Different volume levels | Normalize and test multiple level ratios |
| Real-time is slow | Algorithm is computationally expensive | Use shorter buffers, optimize code, or use edge/GPU hardware |
| Different speakers sound similar | MFCC baseline is limited | Use speaker embeddings |
| Room echo | Reverberation | Use microphone array, dereverberation or neural methods |

# 21. What Is the Novelty?

Do not claim that multi-speaker separation or speaker recognition itself is a new invention. Those are established research areas. The project contribution should instead be framed as an integrated system and experimental study.

- Design an end-to-end DSP-first pipeline combining separation and speaker identification.
- Compare a classical DSP separator with an advanced AI separator.
- Study how separation quality affects speaker identification.
- Evaluate the effect of noise, speaker volume difference and overlap.
- Optionally demonstrate multi-microphone spatial processing in an ECE-oriented implementation.

# 22. How to Present This Project to Your Professor

The most important rule: do not begin with algorithms. Begin with the real problem.

Start with this situation:

> Two people talk at the same time → one recording → voices overlap → difficult to identify each person.

Then explain that your system solves it in two stages: first separate the voices, then identify each separated voice.

## Suggested 8–10 minute presentation

| Time | What to show | What to explain |
|---|---|---|
| 0:00–1:00 | Problem | A microphone records a mixture when people speak simultaneously. |
| 1:00–2:00 | Motivation | Explain why separation and identification are useful. |
| 2:00–3:00 | Block diagram | Walk from microphone to separation to fingerprinting. |
| 3:00–5:00 | DSP method | Explain FFT/STFT and NMF/ICA clearly. |
| 5:00–6:00 | Fingerprinting | Explain MFCC and speaker matching. |
| 6:00–7:30 | Results | Show spectrograms, separated audio and identification scores. |
| 7:30–8:30 | Limitations | Explain difficult overlap, noise, echo and single-mic limitations. |
| 8:30–10:00 | Future work | Show microphone array, beamforming and AI extensions. |

# 23. Exact Way to Explain It Verbally

"Sir/Ma'am, our project addresses the multi-speaker or cocktail-party problem. When two people speak simultaneously, the microphone records a mixed signal. We first preprocess and analyze this signal in the time-frequency domain using FFT and STFT. We then estimate the individual speaker signals using a DSP-based source-separation method. Once the voices are separated, we extract MFCC features and create a numerical voice fingerprint. We compare that fingerprint with registered speaker profiles to identify the speaker. We will evaluate both separation quality and identification accuracy. After completing the DSP baseline, we can extend the system using a microphone array, beamforming or a neural speech-separation model."

# 24. Questions Your Professor May Ask

| Question | Answer |
|---|---|
| Why is this a DSP project? | Because the core pipeline involves sampling, filtering, FFT, STFT, time-frequency analysis, source separation and reconstruction. |
| Why STFT? | Speech changes with time, so we need frequency information as a function of time. |
| Why not just use AI? | We want an interpretable DSP baseline first and then compare it with AI. |
| Why MFCC? | It is a compact representation of speech spectral characteristics and is an established baseline for speaker recognition. |
| Can one microphone perfectly separate two speakers? | No. Single-channel separation is difficult and generally underdetermined; performance depends on the signal and method. |
| Why multiple microphones? | They provide spatial differences that can be used for ICA and beamforming. |
| What if separation is poor? | Improve the DSP method, use multiple microphones, or add a neural separator. |
| How will you measure success? | Use source-separation metrics plus speaker-identification accuracy and confusion matrices. |
| What is the limitation? | Overlapping frequencies, reverberation, noise, unseen speakers and computation can reduce performance. |
| What is your future scope? | Real-time microphone arrays, beamforming, speaker embeddings and neural source separation. |

# 25. Backup Plan for the Final Demonstration

Real-time audio can fail because of room noise, microphone drivers, CPU load or unexpected recording conditions. Therefore, always prepare two demonstrations.

33. Primary demo: live microphone recording.
34. Backup demo: prerecorded mixed.wav with a known successful result.
35. Show waveform and spectrogram before separation.
36. Play separated outputs.
37. Show the fingerprint/feature visualization.
38. Show the final predicted speaker names.
39. Show quantitative results from the test set.

This ensures that the professor can see the complete engineering work even if the live environment is poor.

# 26. Final Expected Output

```
INPUT
Two people speaking simultaneously

     ↓

OUTPUT 1
Separated Speaker 1 audio
Speaker 1 → Registered Person A

OUTPUT 2
Separated Speaker 2 audio
Speaker 2 → Registered Person B

DISPLAY
• Mixed waveform
• Mixed spectrogram
• Separated spectrograms
• MFCC features
• Similarity scores
• Identification result
• Accuracy / separation metrics
```

This is the final story of the project: mixed speech → signal analysis → source separation → voice fingerprint → speaker identification → measurable evaluation.

# 27. Recommended First Practical Experiment

Before buying hardware or attempting real-time operation, complete this small experiment.

40. Record Speaker A saying a 10–20 second sentence.
41. Record Speaker B saying a different 10–20 second sentence.
42. Mix the two recordings in Python.
43. Plot the mixed waveform.
44. Generate the mixed spectrogram.
45. Apply STFT + NMF separation.
46. Save two estimated WAV files.
47. Listen to both files.
48. Extract MFCCs from the separated files.
49. Compare them with stored Speaker A and Speaker B fingerprints.
50. Record the identification result and separation metric.
51. Repeat with different volume ratios and background noise.

If this experiment works, you have the foundation of the complete project. Every later feature is an extension of this pipeline rather than a completely new project.

# 28. Final Conclusion

The project should be treated as a gradual engineering system, not as one large algorithm. The safest development path is: controlled audio → preprocessing → FFT/STFT → DSP separation → MFCC fingerprinting → speaker identification → evaluation → real-time hardware → optional AI enhancement. This approach makes the project understandable, testable and defendable in front of a professor.

The strongest academic presentation is not "our AI identifies voices." It is: "we designed and evaluated a DSP-based pipeline for overlapping speech, quantified its limitations, and then proposed advanced multi-microphone and neural methods to overcome those limitations."

# Appendix A — Suggested PPT Structure

1. Title, team members and project guide
2. Real-world problem: overlapping speakers
3. Problem statement
4. Objectives
5. Existing research / literature
6. Proposed system architecture
7. Audio acquisition and preprocessing
8. FFT and STFT explanation
9. Speaker separation method
10. Voice fingerprinting with MFCC
11. Speaker database and matching
12. Experimental setup
13. Results: spectrograms and audio
14. Results: accuracy and separation metrics
15. Limitations
16. Future scope
17. Conclusion
18. Live demonstration

# Appendix B — Minimum Deliverables

- Python source code.
- Clean speaker recordings.
- Generated mixed recordings.
- Separated output recordings.
- Waveform and spectrogram figures.
- MFCC/fingerprint plots or feature files.
- Speaker database.
- Evaluation results.
- Final PPT.
- Short live/recorded demonstration.
