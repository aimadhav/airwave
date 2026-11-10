# Four-speaker meeting test and optional Nemotron backend

Date: 6 October 2026. All inference ran locally on this Windows CPU machine. No audio was uploaded to a transcription/AI service, and no API keys were required.

## Outcome

We downloaded a genuine continuous English meeting excerpt, added an optional end-to-end speaker diarizer, compared it with the existing pipeline using reference speaker timelines, and completed a real DSP → Whisper → analysis → storage run.

The new backend improves the new meeting substantially. It does **not** improve every recording, so the existing Sherpa backend remains the default. Larger voice embeddings alone did not help our development recording.

## 1. Recording and provenance

The source is the AMI ES2005a example distributed in the [VBx repository](https://github.com/BUTSpeechFIT/VBx/tree/57466e6e245d5cdfe2e88ee6503702ace3ffdd03/example), pinned to commit `57466e6e245d5cdfe2e88ee6503702ace3ffdd03`:

- [Source WAV](https://raw.githubusercontent.com/BUTSpeechFIT/VBx/57466e6e245d5cdfe2e88ee6503702ace3ffdd03/example/audios/16k/ES2005a.wav).
- [Reference RTTM](https://raw.githubusercontent.com/BUTSpeechFIT/VBx/57466e6e245d5cdfe2e88ee6503702ace3ffdd03/example/rttm/ES2005a.rttm).
- [AMI corpus download/licensing information](https://groups.inf.ed.ac.uk/ami/download/): retain AMI attribution and CC BY 4.0 licensing when redistributing this recording.

We copied the first 150 seconds of the distributed WAV, without rearranging turns, repeating content, padding missing audio, or generating voices. The distributed source may itself be an excerpt; we do not claim to have downloaded the entire original meeting.

The crop was chosen before diarization based on duration and reference participation, not on the best model score. Its 49 reference intervals contain four anonymous identities: `MEE017`, `MEE018`, `FEE019`, and `MEO020`. There are 182.576 reference speaker-seconds in 150 recording-seconds because people overlap. These IDs are used only for evaluation, never supplied to the diarizer.

Artifacts:

- `data/benchmarks/vbx-es2005a/clear-four-speaker-meeting-150s.wav`: original 150-second crop, 16 kHz mono PCM16.
- `data/benchmarks/vbx-es2005a/clear-four-speaker-meeting-150s.rttm`: cropped and time-aligned reference.
- `data/benchmarks/vbx-es2005a/excerpt_manifest.json`: source URLs, licensing notes, hashes, crop parameters and per-speaker participation.
- `data/benchmarks/vbx-es2005a/clear-four-speaker-meeting-150s-louder.wav`: playback copy only, with global peak normalization and DC removal. It preserves the original timing and does not denoise or separate voices.

The original peak is −10.69 dBFS; the playback target is −0.45 dBFS (peak 0.95), approximately 10.25 dB louder. Increasing gain also increases any background noise. This is not a measured intelligibility improvement or a LUFS loudness standard. We did not establish a human-rated clarity score.

SHA-256 of the original crop: `f93c3178c2c1fbae8dfd8828a998b3b0a4e5ab11b707d9c2fd8b5849a08d6893`.

SHA-256 of its aligned reference: `0abde0c66befd69d445b6c97b22a5960200d540031b8956bd0c33cd8d3130c5d`.

## 2. What changed in the algorithm

### Existing route: segmentation, voice embeddings and clustering

The Sherpa backend uses pyannote 3.0 neural segmentation to find speaker activity, TitaNet-small to describe voices numerically, and clustering to group similar voices. A clustering threshold can split one person into multiple labels or merge different people. Setting the count to four does not prove the assignments are correct.

### New optional route: directly predict each speaker's activity

[NVIDIA Nemotron-3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) predicts a time series of speaker activity, rather than relying on our existing embedding-and-clustering combination. Its model card describes up to eight speaker channels, arrival-order identity assignment, and streaming speaker caches. Multiple active channels allow overlapping speakers to be represented.

We run its official GGUF weights using the [NeMo-Speech.cpp Windows CPU runtime](https://github.com/NVIDIA/NeMo-Speech.cpp/releases/tag/v0.2.0). This avoids requiring PyTorch, a GPU, or a cloud diarization API on this machine. The quantized native implementation is not automatically equivalent to every result reported for the original model.

The application route is:

```text
Uploaded audio
  → decode/downmix/resample to 16 kHz mono
  → DC removal and peak normalization
  → local Nemotron speaker activity prediction
  → activity thresholds and segment postprocessing
  → validated anonymous speaker timeline, preserving overlap
  → timestamped mixed-audio WAV clips
  → existing AudioClipPayload
  → real offline Whisper tiny.en
  → existing analysis and storage
```

Energy VAD still runs for diagnostic information, but it does not veto neural-model speech. A quiet voice must not disappear simply because an energy detector classifies it as silence.

Important distinctions:

- **Speaker activity detection** answers which voice is active at a time.
- **Speech recognition** answers what words were spoken. Whisper is used here, not as the diarizer.
- **Speaker identification** would match a voice to a known real person. We do not do that in these experiments: labels remain `SPEAKER_00` etc.
- **Source separation** would isolate simultaneous voices. We have not implemented it in this route; overlap clips contain the original mixture.

### Architecture and safety

`NemoNativeSpeakerBackend` implements the existing `SpeakerInferenceBackend` protocol and returns the existing `ModelSpeakerInterval` DTO. `LocalModelDiarizer` remains runtime-independent. Only the DSP composition root selects the concrete adapter. No AI-layer internal implementation is imported into the new diarization service, and `AudioClipPayload` remains unchanged.

The adapter validates JSON and timestamps, translates runtime/I/O/time-out failures into DSP-specific errors, bounds subprocess execution, uses `shell=False`, and removes temporary PCM files. It rejects forced speaker counts, because this backend predicts activity automatically. Missing weights/runtime yield an installation error; inference never downloads them silently.

Native paths, timeout, preset and short-file limit live in `configs/native_diarization_config.json`. Optional overrides live in the experimental YAML profiles. Runtime binaries and weights are SHA-256-pinned in `configs/native_diarization_assets.json`; setup only downloads/verifies/extracts allowlisted files and never executes them. Models and recordings remain git-ignored.

## 3. How quality was measured

We used the project's interval-based, label-permutation-invariant scorer:

```text
DER = (missed speaker-time + false-alarm speaker-time + confused speaker-time)
      / reference speaker-time
```

Lower DER is better. We score overlap and report both exact boundaries and a 0.25-second boundary collar. The collar removes boundary-near regions before scoring. DER is not word accuracy, and `100 − DER` is not a defensible general “accuracy” percentage.

The local scorer is not a claim of official VBx/AMI benchmark protocol compliance. These are small exploratory sample comparisons, not a statistically validated leaderboard result. The model may have been trained/evaluated on related corpus material. No reference speaker labels or speaker counts were injected into inference.

Each report stores input/model SHA-256, configuration, generated segments, reference mapping, error components, clip validation and elapsed DSP time. Timing includes preprocessing, model setup/inference and clip writing; it excludes Whisper and AI analysis. CPU scheduling varies between runs.

## 4. New four-speaker meeting: measured comparison

Both runs used the same original ES2005a crop and canonical preprocessing. Automatic speaker count was enabled.

| Measurement | Existing Sherpa/TitaNet-small | Optional Nemotron, chunked `v3-offline` |
|---|---:|---:|
| Reference speakers | 4 | 4 |
| Predicted speaker labels | 6 | 4 |
| Generated clips | 29 | 58 |
| DER, zero collar, overlap included | 30.45% | 18.93% |
| DER, 0.25 s collar, overlap included | 18.10% | 10.53% |
| Missed speaker-time | 26.994 s | 26.379 s |
| False-alarm speaker-time | 6.756 s | 6.609 s |
| Confused speaker-time | 21.841 s | 1.565 s |
| DSP elapsed time | 16.24 s | 5.87 s |
| Clip format/duration errors | 0 | 0 |

Most of the gain is **more consistent speaker assignment**, not eliminating all missed speech. The zero-collar DER reduction is 11.52 percentage points, approximately 37.8% relative to this baseline on this sample.

Reports are under `data/benchmark_results/embedding-comparison-20261006/es2005a_baseline_09/` and `es2005a_nemotron_v3_offline/`.

## 5. Counterexamples and experiments that did not help

We kept the older continuous ES2002a 150-second excerpt as a development/regression recording. It uses a different, word-derived reference preparation from ES2005a, so compare methods within each recording, not raw DER between recordings as equivalent tasks.

| ES2002a method | Labels | Zero-collar DER | 0.25 s collar DER |
|---|---:|---:|---:|
| Existing TitaNet-small, threshold 0.9 (prior run) | 9 | 26.87% | 18.76% |
| Nemotron, chunked runtime defaults | 4 | 30.63% | 22.63% |
| Nemotron, chunked onset 0.5 / offset 0.3 | 4 | 30.01% | 21.75% |
| Nemotron, full attention, onset 0.5 / offset 0.3 | 4 | 28.69% | 20.66% |

Although the native backend reduces speaker confusion to less than one second on this older recording, it misses more reference speech, so total DER is worse. We must not hide that behind the correct speaker count.

Full attention did **not** generalize to the new recording: ES2005a DER was 31.77% with that profile, worse than both the chunked native route and baseline. Keep it an experiment, not the demo recommendation. `v3-offline` is a larger-chunk streaming/cache preset; it is **not** the same as full attention (`--offline`). Full attention also has a short-file positional limit, guarded by our configured 390-second bound.

We also replaced only the embedding model on ES2002a, keeping neural segmentation and automatic clustering. All of these alternative embeddings were worse than the existing 26.87% baseline:

| Embedding | Threshold | Labels | Zero-collar DER |
|---|---:|---:|---:|
| WeSpeaker ResNet34-LM | 0.6 | 12 | 44.21% |
| WeSpeaker ResNet34-LM | 0.7 | 7 | 44.32% |
| WeSpeaker ResNet34-LM | 0.8 | 6 | 44.32% |
| WeSpeaker ResNet34-LM | 0.9 | 4 | 51.33% |
| English CAM++ | 0.6 | 8 | 55.96% |
| English CAM++ | 0.7 | 8 | 55.90% |
| English CAM++ | 0.8 | 6 | 57.07% |
| English CAM++ | 0.9 | 5 | 59.60% |
| TitaNet-large | 0.6 | 15 | 34.35% |
| TitaNet-large | 0.7 | 14 | 33.42% |
| TitaNet-large | 0.8 | 14 | 33.42% |
| TitaNet-large | 0.9 | 10 | 31.47% |

This does not establish that those models are intrinsically bad: they may need different clustering, calibration or preprocessing. It establishes that swapping weights into **our current stack** did not improve this development sample. They were not promoted or tuned on the new sample.

The native chunked profile also produced two labels on the existing English two-speaker samples `1-two-speakers-en.wav` and `3-two-speakers-en.wav`, in 0.47 s and 2.02 s respectively, with zero clip-validation errors. Those files have no supplied reference RTTM in our setup, so we report no DER for them.

## 6. Full DSP / STT / AI handoff

The actual CLI run completed with session ID `demo_es2005a_native_20261006` in `data/ai_layer.db`:

- 58 real WAV clips and four anonymous speakers.
- 49 stored transcript turns, 448 recognized words, one empty turn.
- Real local Whisper tiny.en transcription, not a scripted transcript.
- Offline heuristic analysis: zero facts, two candidate decisions and one candidate action item.
- Summary and session persisted through the existing AI orchestrator.

The recognizer produced understandable discussion about designing/pricing a remote control, alongside obvious recognition errors and repeated words during overlapping speech. We do **not** report WER: no aligned reference transcript was evaluated here.

The heuristic analyzer incorrectly treats “before deciding…” discussion as decisions. Therefore the analysis objects demonstrate the handoff/storage workflow, not trustworthy extracted meeting decisions. No one should present the mock analysis as a real LLM result or verified assignment.

Payload and clips are in `data/dsp_sessions/demo_es2005a_native_20261006/`. Inspect the stored result with the `show-session` command below.

## 7. Tests and reproducible commands

The full suite passed **160 tests**, with `RUN_REAL_DSP=1`, including:

- Isolated native-adapter tests using mock subprocess output, without runtime/model downloads.
- Invalid output, speaker indices, negative/non-finite timestamps, bad configuration and timeout/process failures.
- Temporary-file cleanup and overlap-preserving canonical clip generation.
- Pinned ZIP extraction, existing-file checksum mismatch, path traversal and symlink rejection.
- Non-destructive louder playback-copy generation.
- Actual Sherpa and native DSP → real Whisper → offline AI integration on the two-speaker fixture.
- A sample-specific four-speaker native regression with overlap and reference scoring. This is not a universal four-speaker accuracy guarantee.

In an activated project environment with the documented DSP/test dependencies:

```powershell
# Explicit optional native setup: Windows CPU archive and pinned model.
python scripts/setup_diarization.py --manifest configs/native_diarization_assets.json --native

# Download and prepare the reference-labelled recording.
python scripts/setup_diarization.py --manifest configs/vbx_clear_meeting_assets.json --samples
python scripts/prepare_ami_excerpt.py --config configs/vbx_clear_meeting_test.json

# Optional louder playback copy; do not overwrite the benchmark input.
python scripts/normalize_demo_audio.py data/benchmarks/vbx-es2005a/clear-four-speaker-meeting-150s.wav data/benchmarks/vbx-es2005a/clear-four-speaker-meeting-150s-louder.wav --peak 0.95

# Reproduce the reference-scored native route, using a fresh experiment tag.
python scripts/benchmark_diarization.py data/benchmarks/vbx-es2005a/clear-four-speaker-meeting-150s.wav --reference data/benchmarks/vbx-es2005a/clear-four-speaker-meeting-150s.rttm --config configs/dsp_nemotron_experiment.yaml --tag es2005a_native_repeat

# End-to-end demo: real diarization and real STT, offline heuristic analysis.
python cli.py ingest data/benchmarks/vbx-es2005a/clear-four-speaker-meeting-150s.wav --mode mock --dsp-provider local_model --dsp-backend nemo_native --session-id your_fresh_demo_id --title "Four-speaker meeting"
python cli.py show-session demo_es2005a_native_20261006

# Ordinary isolated suite needs no models; real tests require the explicit setup.
python -m pytest -q
$env:RUN_REAL_DSP = "1"
python -m pytest -q
```

The downloader reuses valid assets. Excerpt preparation, playback export and benchmark reports deliberately reject overwriting existing artifacts. Skip preparation/export on repeat runs or choose fresh output paths/configuration. Use a fresh session ID to avoid replacing an existing CLI session. The native runtime profile is Windows-specific; another OS requires a matching runtime path.

The web server can use this backend through its loaded DSP configuration (`dsp.local_model.backend: nemo_native`), but this turn did not add or verify a browser backend-selection control. The verified demo entry point here is the CLI.

## 8. Presentation wording and remaining work

Say: “We used a real continuous four-person discussion with reference speaker timings. Our optional neural activity diarizer recovered four speaker labels automatically and reduced local diarization error from 30.45% to 18.93% on this sample. We also tested a regression recording where total error got worse, so it remains configurable rather than replacing the default universally.”

Show the louder original recording, the four-label timeline/clips, the before/after measurements, the transcript and the passing tests. Explicitly distinguish estimated speaker identity, recognized text, and heuristic analysis.

Remaining quality work:

1. Evaluate more meetings with consistent reference preparation and hold out recordings before selecting thresholds.
2. Improve quiet/backchannel speech recall without introducing false active speakers.
3. Transcribe the whole recording with word timestamps and align words to speaker activity; per-clip mixed-audio ASR currently repeats overlap content.
4. Benchmark stronger STT separately using WER, then validate real AI extraction against human-labelled decisions/actions.
5. Add an explicit browser backend selector and show backend/provenance alongside speaker labels before claiming the web demo uses the new model.

No automatic name inference, source separation, paid API processing, Google OAuth or Zoom support was added. Existing default diarization settings and the protected local beginner explainer were not changed or pushed.
