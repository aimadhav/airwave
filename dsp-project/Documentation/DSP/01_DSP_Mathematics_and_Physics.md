# DSP Mathematics and Physics Foundation

This document explains the physical and mathematical ideas behind the DSP
front end of the Audio File project. It is written for the implementation in
`src/dsp` and for the minor-project presentation.

The DSP layer receives a recording of air-pressure variations and converts it
into clean, timestamped speech segments for the AI layer.

```text
Sound pressure in air
        ↓ microphone transducer
Electrical signal
        ↓ sampling
Discrete-time audio x[n]
        ↓ DSP processing
Speech intervals and speaker clips
```

## 1. The Physics of Sound

Sound is a mechanical wave. A person speaking causes nearby air molecules to
compress and rarefy. A microphone converts these pressure changes into an
electrical signal.

A simple sinusoidal pressure wave can be written as:

\[
p(t) = A \sin(2\pi f t + \phi)
\]

where:

- \(A\) is amplitude, related to pressure variation and perceived loudness;
- \(f\) is frequency in hertz;
- \(t\) is time in seconds;
- \(\phi\) is phase.

Speech is not one sinusoid. It is a time-varying combination of many
frequencies. The vocal folds create a fundamental vibration, while the vocal
tract shapes resonances called formants. Consonants also contain broadband
noise-like components.

For multiple speakers, the microphone records a mixture. In the simplest
single-microphone model:

\[
x(t) = s_1(t) + s_2(t) + n(t)
\]

where \(s_1\) and \(s_2\) are speaker signals and \(n\) is environmental noise.
The DSP problem is to estimate useful information about the individual signals
from the mixture \(x(t)\).

## 2. Sampling and Quantization

Computers do not store a continuous signal \(x(t)\). They measure it at
regular intervals:

\[
x[n] = x(nT_s)
\]

where \(T_s\) is the sampling period and the sampling frequency is:

\[
f_s = \frac{1}{T_s}
\]

The current project uses 16 kHz as the canonical speech sample rate. This
means 16,000 samples are stored every second.

### Nyquist limit

The highest frequency that can be represented without aliasing is half the
sampling rate:

\[
f_{Nyquist} = \frac{f_s}{2}
\]

At 16 kHz, the Nyquist frequency is 8 kHz. Most speech information useful for
transcription is below this limit.

If a signal contains frequencies above the Nyquist limit before sampling,
those frequencies appear as incorrect lower frequencies. This is aliasing.
An anti-aliasing low-pass filter is therefore normally applied before
downsampling.

### Quantization

The microphone's continuous amplitude is rounded to a finite number of levels.
For signed 16-bit PCM, samples are represented approximately in:

\[
[-32768, 32767]
\]

The DSP code converts these integer values to floating-point samples near:

\[
[-1, 1]
\]

This makes later processing independent of the original integer bit depth.

## 3. Audio Buffer Representation

The project represents audio as a matrix:

\[
X \in \mathbb{R}^{N \times C}
\]

where:

- \(N\) is the number of frames;
- \(C\) is the number of channels;
- each row contains the simultaneous channel samples for one time instant.

For mono audio, the shape is \((N, 1)\). The `AudioBuffer` contract stores the
matrix together with sample rate, channel count, sample width and frame count.

The duration is:

\[
T = \frac{N}{f_s}
\]

This equation is used throughout the pipeline to convert sample indices into
timestamps.

## 4. Mono Conversion

For a stereo signal with left and right channels, the baseline mono signal is:

\[
x_{mono}[n] = \frac{x_L[n] + x_R[n]}{2}
\]

This is useful because the first VAD and STFT implementations operate on one
signal. A production microphone-array implementation should preserve all
channels until spatial processing has finished.

## 5. DC Offset and Normalization

### DC offset

An ideal audio waveform has an average value near zero. A constant microphone
bias adds a DC component:

\[
\mu = \frac{1}{N}\sum_{n=0}^{N-1}x[n]
\]

The centered signal is:

\[
x_c[n] = x[n] - \mu
\]

The `normalize_peak` stage removes this offset for each channel.

### Peak normalization

Let:

\[
p = \max_n |x_c[n]|
\]

For a target peak \(a\), the normalized signal is:

\[
x_{norm}[n] = a\frac{x_c[n]}{p}
\]

The implementation uses a default target peak of 0.95. Leaving some headroom
reduces the chance of clipping after later processing.

Normalization changes amplitude, but it does not change the waveform's timing
or frequency content.

## 6. Resampling

If the original rate is 44.1 kHz and the target rate is 16 kHz, the signal must
be represented at a new set of time positions. The current baseline uses
linear interpolation:

\[
x(t) \approx (1-\alpha)x[k] + \alpha x[k+1]
\]

where \(k\) and \(k+1\) surround the new time position and \(\alpha\) is the
fractional distance between them.

Linear interpolation is easy to test and sufficient for the first prototype.
A production resampler should use an anti-aliasing low-pass filter before
downsampling, such as a polyphase filter.

## 7. RMS Energy and Decibels

Speech activity can be estimated from short-time energy. For a frame of \(M\)
samples, RMS amplitude is:

\[
RMS = \sqrt{\frac{1}{M}\sum_{n=0}^{M-1}x^2[n]}
\]

RMS is useful because it measures average signal magnitude while being less
sensitive to individual peaks than peak amplitude.

Audio levels are commonly expressed in decibels relative to full scale:

\[
dBFS = 20\log_{10}(RMS)
\]

Because \(\log(0)\) is undefined, implementations use a small floor value.
The energy VAD compares frame dB values against a configured threshold, such as
-35 dBFS.

The VAD does not prove that a frame contains a particular speaker. It only
estimates whether the frame contains sufficient acoustic energy to be speech.

## 8. Filtering

A digital filter modifies selected frequency components. In general, a linear
time-invariant filter can be described by convolution:

\[
y[n] = \sum_k h[k]x[n-k]
\]

where \(h[k]\) is the impulse response.

A high-pass filter attenuates low frequencies. In this project it is intended
to reduce rumble, microphone handling noise and very low-frequency vibration.
The cutoff must be chosen carefully: an aggressive filter can remove useful
voicing information.

Noise reduction is also a trade-off. Removing too little noise leaves false
VAD detections; removing too much can damage consonants and harm STT.

## 9. Fourier Transform

The discrete Fourier transform represents a finite signal as a sum of complex
sinusoids:

\[
X[k] = \sum_{n=0}^{N-1}x[n]e^{-j2\pi kn/N}
\]

The inverse transform reconstructs the time signal:

\[
x[n] = \frac{1}{N}\sum_{k=0}^{N-1}X[k]e^{j2\pi kn/N}
\]

The frequency associated with bin \(k\) is:

\[
f_k = \frac{k f_s}{N}
\]

The frequency resolution is:

\[
\Delta f = \frac{f_s}{N}
\]

A longer FFT frame gives better frequency resolution but poorer time
resolution. Speech processing must balance both.

The project uses the real FFT for mono real-valued audio. This returns
\(N/2+1\) frequency bins because the negative-frequency half is redundant.

## 10. Short-Time Fourier Transform

Speech changes over time, so one FFT over the entire recording is not enough.
The short-time Fourier transform divides the signal into overlapping frames:

\[
X(m,k) = \sum_{n=0}^{N-1}x[n+mH]w[n]e^{-j2\pi kn/N}
\]

where:

- \(m\) is the frame index;
- \(H\) is the hop length;
- \(w[n]\) is a window function;
- \(N\) is the frame length.

The Hann window used by the implementation reduces discontinuities at frame
boundaries:

\[
w[n] = \frac{1}{2}\left(1-\cos\left(\frac{2\pi n}{N-1}\right)\right)
\]

The magnitude spectrogram is:

\[
S(m,k) = |X(m,k)|
\]

It shows how frequency energy changes over time. Vowels tend to show strong
harmonic bands; fricatives tend to spread energy over a wider frequency range.

## 11. Why STFT Helps Speaker Processing

Two speakers can overlap in time but still have different energy patterns in
the time-frequency plane. This is the basis for classical mask-based methods
such as NMF separation.

For a mixture:

\[
X = S_1 + S_2 + N
\]

we can estimate a mask \(M_i\) for source \(i\) and produce:

\[
\hat{S_i} = M_i \odot X
\]

where \(\odot\) means element-wise multiplication. An inverse STFT then
reconstructs an estimated time-domain source.

This is an estimate, not a perfect recovery. When two speakers occupy the same
time-frequency region, the information is ambiguous.

## 12. Voice Activity Detection Mathematics

The current energy VAD follows this process:

1. Divide the signal into short frames.
2. Calculate RMS for every frame.
3. Convert RMS to dBFS.
4. Mark frames above the threshold as active.
5. Merge active regions separated by short pauses.
6. Remove regions shorter than the minimum speech duration.
7. Convert frame indices into seconds.

If a frame begins at sample index \(n\), its start timestamp is:

\[
t_{start} = \frac{n}{f_s}
\]

For frame length \(L\), its approximate end timestamp is:

\[
t_{end} = \frac{n+L}{f_s}
\]

The result is a list of `VoiceActivitySegment` objects. Each object contains a
start time, end time and an energy-derived confidence score.

## 13. Limitations of Energy VAD

Energy VAD is intentionally a baseline. It can fail when:

- background noise is louder than speech;
- a speaker is very quiet;
- speech contains long low-energy sounds;
- music or machinery resembles speech energy;
- two speakers overlap;
- the recording has strong reverberation.

It does not identify speakers and does not separate voices. Its role is to
reduce the amount of audio sent to later stages and provide a deterministic
baseline for testing.

The next stages will add diarization and optional model-based VAD behind the
same contracts.

## 14. Connection to the AI Layer

The DSP layer produces audio intervals. The AI layer adds linguistic meaning.

```text
DSP result:
  speaker_id = SPEAKER_00
  start = 2.10 s
  end = 6.40 s
  clip = clips/segment_001.wav

AI result:
  text = "Let us review the project status."
```

The combined record becomes:

```json
{
  "segment_id": "segment_001",
  "speaker_id": "SPEAKER_00",
  "start_time": 2.10,
  "end_time": 6.40,
  "audio_path": "clips/segment_001.wav",
  "text": "Let us review the project status."
}
```

This separation of responsibilities is important: DSP determines where the
audio is and which acoustic source is associated with it; STT determines what
was said; the LLM extracts meaning, actions and summaries.

## 15. Presentation Summary

The physical story is:

> A speaker creates pressure variations in air. The microphone samples those
> variations. DSP cleans and analyzes the samples in time and frequency. The
> system detects speech intervals and eventually separates or identifies
> speakers. The AI layer converts the resulting clips into text and insights.

The mathematical story is:

> Continuous signals become discrete samples; frames become RMS energies and
> Fourier coefficients; those coefficients form spectrograms; thresholds and
> time relationships become speech timestamps; timestamps connect audio to
> transcription and analysis.

## 16. The implemented real multi-speaker path (October 2026)

The repository now offers a real CPU diarizer, separately from the one-speaker
fallback and deterministic fixture. It is not a hand-written MFCC classifier.
It composes a learned speech-segmentation model, a learned voice-embedding model,
and hierarchical clustering through an injected inference backend.

### Speech segmentation: beyond energy alone

Energy VAD uses a rule such as

\[
\operatorname{RMS}(x)=\sqrt{\frac{1}{N}\sum_{n=0}^{N-1}x[n]^2},\qquad
20\log_{10}(\operatorname{RMS}(x)+\epsilon)>\theta.
\]

It asks whether a frame is loud enough, not whether two frames sound like the
same person. Loud music can pass this rule; quiet speech can fail it.

The local segmentation model instead estimates voice activity from learned
patterns over overlapping audio windows. Energy-VAD output remains available
as a diagnostic but does not discard audio before neural inference. This is
why an empty energy-VAD result does not force an empty neural timeline.

### Voice embeddings: a numerical description, not a name

For a detected voice region, an embedding network maps acoustic features to
a vector:

\[
e=f_\phi(x_{\text{region}})\in\mathbb{R}^{d}.
\]

The learned parameters \(\phi\) come from the downloaded model. We do not train
this network on the test recording. Ideally, the same voice produces nearby
vectors despite different words; different voices produce more distant ones.
Background noise, a very short region, and overlap can break that ideal.

The embedding summarizes vocal acoustics: timbre and patterns of spectral
energy, not just pitch. People with similar pitch need not be the same speaker.
No coordinate of this vector means “Alice.” A cluster remains anonymous unless
a separate, authorized identity mechanism supplies evidence.

### Cosine distance and hierarchical clustering

Normalize each vector and compare its direction:

\[
\hat e_i=\frac{e_i}{\|e_i\|_2},\qquad
d(i,j)=\max(0,1-\hat e_i^T\hat e_j).
\]

For a simple illustration, \((1,0)\) and \((0.98,0.02)\) point almost the same
way, whereas \((1,0)\) and \((0,1)\) do not. Actual embeddings have many more
dimensions; these two-dimensional arrows are only intuition.

The selected runtime uses complete-linkage hierarchical clustering: group
distance is the maximum pairwise distance between members. It cuts the merge
tree at the configured threshold, or at a supplied number of groups. A larger
threshold permits more merging; a smaller one produces more fragments.
[Runtime implementation](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/fast-clustering.cc).

Our development tests moved the threshold from `0.5` to `0.9` after finding
speaker over-splitting. This is calibration on a small collection, not a
mathematical guarantee. An explicit known count constrains clustering; it must
not be reported as automatic count detection. The mathematical grouping does
not prove a person's identity or generate calibrated speaker confidence.

### Why anti-alias filtering matters

For a new sample rate of 16 kHz, the Nyquist limit is 8 kHz. Simply discarding
samples can fold higher-frequency components into that range. For example,
a 12-kHz component can alias into a misleading 4-kHz component.

The real-model path therefore uses polyphase resampling, including a low-pass
filter. Conceptually the filter performs

\[
y[n]=\sum_k h[k]x[n-k]
\]

before the downsampling part. The ratio is reduced by the greatest common
divisor of the source and destination rates. This retains duration while
preventing much of the unwanted high-frequency folding. A unit test verifies
strong attenuation of a 12-kHz sinusoid when converting 48 kHz to 16 kHz.

### Overlap is not separation

When predicted intervals \([a,b)\) and \([c,d)\) intersect, their shared duration
is

\[
\max(0,\min(b,d)-\max(a,c)).
\]

The timeline preserves both labels and marks overlap. Cropping a mono mixture
does not extract one voice from it: both overlapping clips can contain both
people. The UI uses separate lanes and hatching to make that limitation visible.
Whisper can omit or confuse a voice there; source separation is not implemented.

### Measuring diarization error honestly

Speaker labels are arbitrary. First find the global one-to-one label mapping
that maximizes reference/prediction overlap. Then score intervals between all
reference and predicted boundaries. At a moment with \(R\) reference speakers,
\(H\) predicted speakers, and \(C\) correctly mapped active speakers:

\[
M=\max(0,R-H),\quad F=\max(0,H-R),\quad E=\min(R,H)-C.
\]

Integrate each quantity over time and divide by reference speaker-time:

\[
\mathrm{DER}=\frac{\int M\,dt+\int F\,dt+\int E\,dt}{\int R\,dt}.
\]

Two people speaking for one second contribute two reference speaker-seconds.
Our zero-collar measurement on the annotated 30-second example was 5.99%,
versus 49.57% for the one-speaker fallback. Adding a ±250-ms boundary collar
excludes near-boundary evaluation time and produces a much more forgiving
score. It must always be disclosed; it is not the same experiment.

DER is not word error rate. DER measures speaker/time attribution; WER measures
inserted, deleted, or substituted words against an independently checked text
reference. We have not measured WER for the downloaded English conversation.
See `06_Real_Multispeaker_Validation.md` for sources, raw report locations,
robustness results, and the limitations of the development sample collection.
