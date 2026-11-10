# A quality-first process for meeting audio

Updated: 16 October 2026.

## Recommendation

Keep the original recording and evaluate speaker attribution, recognized words and AI analysis independently. Our measurements do not justify replacing the existing Sherpa diarizer with Nemotron globally. Nemotron is useful as an optional alternative, not a universal winner.

The highest-priority next architectural experiment is to recognize coherent speech regions once, obtain word timestamps, and align those words with a separate speaker timeline. The present pipeline retranscribes overlapping mixed-audio speaker clips, which can repeat the same words for different people.

This document distinguishes the recommended target process from features already verified. Word alignment and a stronger recognizer have **not** been implemented by the experiments below.

## 1. Preserve information at upload

Keep the uploaded recording unchanged. If it has genuine per-participant audio tracks, retain them: merging those tracks would throw away useful speaker information. Stereo alone does not prove that one speaker occupies each channel.

For Google Meet artifacts, use available native participant names, transcript timestamps and text first, retaining their provenance. Incomplete or absent metadata falls back to acoustic estimation. Random files normally have only anonymous voice labels.

Check duration, format, number of channels, silence, clipping and decode errors before processing. Do not silently turn undecodable audio into an empty “successful” transcript.

## 2. Why the model copy uses 16 kHz

16 kHz means **16,000 amplitude measurements per second**, not 16,000 words or 16-bit precision. Sample rate and bit depth are separate quantities.

By the Nyquist limit, 16 kHz audio can represent frequencies below 8 kHz. Much of the speech information these models use lies in this band, although it does not contain every detail of the original voice. The decisive reason is that our inference contracts expect 16 kHz audio, not that 16 kHz is universally the highest-quality recording format.

For example, the official [pyannote Community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1) specifies mono 16 kHz input and resamples other rates. Our existing local DSP contract similarly requires 16 kHz mono before learned inference.

Recommended handling:

1. Keep the original 44.1/48 kHz recording if that is what was uploaded.
2. Make an anti-aliased 16 kHz processing copy for models that expect it.
3. Keep timestamps tied to elapsed recording time.
4. Use original-rate audio for playback or a future model that needs a different rate.

Upsampling a poor recording does not recover missing information. Passing higher-rate samples to a model configured for 16 kHz does not improve it; simply changing a WAV header would also alter apparent timing/pitch rather than resample correctly. At the same duration, 48 kHz contains three times as many samples as 16 kHz, so compatible lower-rate processing also saves memory and compute.

Do not aggressively denoise by default. Enhancement can change the acoustic features used to distinguish voices. Any enhancement should be compared against an unchanged baseline rather than assumed beneficial.

## 3. Determine who spoke when

Use a model-driven speaker timeline, preserving simultaneous activity. Energy VAD is a cheap activity hint, but it must not discard a quiet person before a neural diarizer can inspect the signal. Our learned-provider route already treats energy VAD as diagnostic rather than a hard gate.

Keep stable anonymous labels such as `SPEAKER_00`. Acoustic similarity is not evidence of a person's real name. Use a known speaker count only if the user actually knows it; forcing four labels merely to match a demo expectation is not validation.

Keep both ordinary turns and overlap flags. Do not pretend that a clip containing two simultaneous voices is an isolated recording of one speaker.

## 4. Recognize words, then align them to the timeline

Target approach:

1. Transcribe speech chunks with enough context, instead of independently decoding every overlapping speaker clip.
2. Obtain word-level timing through a supported recognizer or forced-alignment adapter.
3. Intersect each word's time interval with speaker activity.
4. Assign a speaker only when the evidence supports it; flag ambiguous overlaps.
5. Group adjacent words into readable speaker turns while retaining word timing and provenance.

[WhisperX](https://github.com/m-bain/whisperX) combines speech recognition, forced word alignment and separate pyannote diarization. This supports the architectural direction, not a claim that installing it automatically solves our audio. Its own documentation warns that overlapping speech and diarization remain imperfect. The [Community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1) also describes an exclusive speaker timeline intended to simplify reconciliation with transcription timestamps; keep the regular overlap-aware timeline too, since an exclusive view cannot represent two simultaneous speakers.

Our present recognizer is local Whisper tiny.en. A larger recognizer is a separate experiment: it may improve recognized words but cannot by itself repair incorrect speaker assignments. Benchmark WER on a manually verified transcript before claiming an improvement. Do not describe our current whole-file helper as word-aligned: it returns one broad interval without word timing.

## 5. Treat overlap explicitly

For normal non-overlapping turns, word/timeline alignment may be enough. During overlap, a single mixed recording can omit or combine words from two people. Possible follow-up comparisons include genuine participant tracks, overlap-specific recognition and optional source separation.

Separation is not automatically beneficial: artificial artifacts may hurt recognition or voice embeddings. Preserve the original mixture, compare recognition before/after, and retain “overlap / uncertain speaker” when evidence remains insufficient.

## 6. Analyze only after transcription quality checks

Send the verified speaker/text/timestamp contract to the AI layer. Preserve citations to the source turn and audio time range. A proposed idea, a decision and an assigned task are different things; “before deciding…” must not be presented as a confirmed decision.

The current offline demo analysis uses heuristics. It demonstrates the handoff and storage, not validated LLM reasoning. Separately test real extraction on human-labelled decisions, tasks and assignees before relying on the summary.

## 7. Experiments actually run today

The versioned comparison is `configs/dsp_quality_comparison.json`, executed by `scripts/run_dsp_comparison.py`. We tested four predeclared profiles on the existing ES2002a development recording:

| Development profile | Estimated speakers | Zero-collar DER | 0.25 s collar DER |
|---|---:|---:|---:|
| Existing Sherpa/TitaNet-small | 9 | 26.87% | 18.76% |
| Native Nemotron defaults, chunked | 4 | 30.63% | 22.63% |
| Lower thresholds + short-turn/padding profile | 4 | 31.80% | 28.04% |
| Moderate thresholds + short-turn/padding profile | 4 | 32.29% | 27.64% |

These last two profiles change multiple postprocessing settings together; the scores evaluate each whole profile, not the isolated causal effect of any one threshold. Both failed to improve the native development baseline and were rejected. They remain explicitly named experimental configurations, not production settings.

The script selected native defaults using development DER only, then compared that unchanged selection with Sherpa on the check recordings:

| Recording | Reference speakers | Sherpa DER | Native-default DER | Better on this recording |
|---|---:|---:|---:|---|
| ES2002a, 150 s development clip | 4 | 26.87% | 30.63% | Sherpa |
| ES2005a, 150 s check clip | 4 | 30.45% | 18.93% | Native |
| pyannote annotated short conversation | 2 | 5.99% | 45.24% | Sherpa |

DER includes overlap and uses zero collar in this table. Lower is better. We use the project's interval scorer, not an official benchmark submission. The recordings have different annotation preparation, so compare methods within a row. These are reused exploratory samples, not a newly blinded test set or evidence of universal model rankings.

On the short conversation both models produced two labels, yet native speaker-confusion time was 7.543 seconds versus 0.003 seconds for Sherpa. Correct speaker count clearly does **not** establish correct diarization.

All eight completed runs produced clips with zero sample-rate/channel/duration validation errors. Full per-run model/input fingerprints, timelines and error components are under:

`data/benchmark_results/quality-comparison-20261016-verified/`

Final verification: **171 tests passed**, including the opt-in real diarization/Whisper integrations (`RUN_REAL_DSP=1`). The added comparison tests cover development-only selection, identifier validation, non-overwrite behavior, declared execution order and invalid timeouts. The final Git whitespace check passed; no production backend was switched.

Its `comparison.json` contains the selection and results. An earlier partial attempt under `quality-comparison-20261016/` failed because the sandbox could not access its temporary directory; do not confuse that partial attempt with the completed comparison.

The experiments did not modify the production backend or its thresholds. We did not train on the reference labels, inject speaker counts, apply new denoising, or run a paid service.

## 8. How to reproduce and evaluate the next changes

Use a consistent Python environment. On this machine the existing project virtualenv is Python 3.11, while `.runtime-deps` contains Python 3.12 extension modules. Mixing them caused collection failures unrelated to the DSP algorithm. The completed runs used a separate Python 3.12 test virtualenv under the ignored model directory and matching dependencies, without replacing the user's virtualenv or app changes. The sandbox's temporary directory also required an approved unrestricted test run.

With a correctly installed/activated environment:

```powershell
python scripts/run_dsp_comparison.py --manifest configs/dsp_quality_comparison.json
python -m pytest -q
$env:RUN_REAL_DSP = "1"
python -m pytest -q
```

Comparison output is intentionally not overwritten: choose a fresh `output` in the manifest before repeating a completed run. Real-model tests require explicit prior model/fixture setup; ordinary tests use mocks and skip those integrations.

For the next quality experiment, use a few manually transcribed recordings spanning quiet turns, overlap, similar voices and accents. Select settings on development recordings, then evaluate untouched recordings. Measure:

- **DER:** missed, false and confused speaker-time, with an explicit overlap/collar policy.
- **WER:** word substitutions, deletions and insertions against a human transcript.
- **Speaker-attributed text quality:** whether recognized words belong to the right person, including overlap and unassigned words.
- **Analysis quality:** real decisions/tasks and correct assignees against human labels.
- **Practicality:** CPU/GPU needs, elapsed time and peak memory.

Prefer changes that improve more than one recording without hiding serious regressions. Do not choose a backend automatically from its speaker count or an unavailable “confidence” score; we do not have a validated quality predictor for a new unlabelled upload.

## 9. Implementation priority

1. Keep the current working diarizers and their contracts; retain the newly reproducible comparison.
2. Add a timestamped-word DTO and a narrow alignment-provider protocol, with mocks and overlap/ambiguity tests.
3. Plug in real word timing and evaluate a stronger local recognizer independently from diarization.
4. Test speaker-word alignment on human-checked examples, then connect the output to the existing AI contract.
5. Compare an additional complete diarization pipeline such as Community-1 when its model-access conditions and dependencies are available. Do not equate our existing pyannote segmentation component with running the full Community-1 pipeline.
6. Consider source separation only if overlap remains a measured bottleneck.

This is a quality plan with a completed diarization experiment, not a claim that all target stages are already built.
