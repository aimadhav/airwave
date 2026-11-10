# NOTSOFAR-1 Desktop Sample: Diarization Benchmark

## What was tested

The test used the 10-second simulated CSS training example in the local NOTSOFAR-1 Challenge repository under `sample_data/css_train_set`. The original item is a seven-channel 16 kHz mixture with frame/sample-level activity labels for three possible speaker slots. In this specific item, two slots are active: `speaker_1` and `speaker_3`. The labels distinguish `1` (speaking), `0` (borderline transition), and `-1` (not speaking). The reference RTTM includes only intervals labelled `1`; borderline frames are excluded.

Two mono inputs were evaluated against that same reference:

1. Channel 0 from the seven-channel mixture.
2. The arithmetic mean of all seven channels. This is an exploratory downmix, not the official NOTSOFAR multi-channel system.

Each input was processed by the project's Nemotron native diarizer, Sherpa ONNX diarizer, and single-speaker fallback. All used the same downstream timeline and scorer. Runs were CPU-only and took less than one second of measured pipeline time per 10-second input in this execution.

## Results

DER counts missed speech, false alarms, and speaker confusion. The first DER column scores overlap and speaker changes with no collar around reference boundaries. The second excludes a 250 ms half-width around each reference boundary. JER is speaker-level intersection-over-union error. Speech F1 tests speech-vs-silence; overlap F1 tests whether zero, one, or multiple speakers are active. RTF is processing time divided by audio duration.

| Input | System | DER, 0 ms collar | DER, 250 ms collar | JER | Speech F1 | Overlap F1 | Speakers found / active reference | RTF |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Channel 0 | Nemotron native | 9.91% | 0.32% | 8.49% | 97.40% | 90.11% | 2 / 2 | 0.070 |
| Channel 0 | Sherpa ONNX | 18.87% | 13.47% | 30.11% | 99.61% | 87.12% | 2 / 2 | 0.084 |
| Channel 0 | Single-speaker fallback | 44.26% | 41.87% | 59.84% | 98.47% | 0.00% | 1 / 2 | 0.006 |
| Seven-channel mean | Nemotron native | 9.66% | 0.20% | 8.27% | 97.45% | 90.38% | 2 / 2 | 0.056 |
| Seven-channel mean | Sherpa ONNX | 26.61% | 21.07% | 33.05% | 98.31% | 74.27% | 2 / 2 | 0.050 |
| Seven-channel mean | Single-speaker fallback | 44.26% | 41.87% | 59.84% | 98.47% | 0.00% | 1 / 2 | 0.007 |

For this item, Nemotron had the lowest DER on both mono inputs. Sherpa was stronger on channel 0 than on the seven-channel mean. The single-speaker fallback detected speech activity well but, by design, could not represent overlapping speakers; its overlap F1 was zero.

The near-zero Nemotron DER with a 250 ms collar is boundary-sensitive: the scored interior of this very short clip is small. The no-collar scores are the more conservative headline comparison here. Both inputs derive from one utterance, so the two rows per model are not independent test examples.

## What this does and does not show

- This is a real run of the repository's **simulated training sample**, not a run on the full NOTSOFAR recorded-meeting benchmark.
- The sample contains activity labels, but no human transcript. Word error rate (WER), speaker-attributed tcpWER, and the official challenge score cannot be computed.
- These local DER/JER metrics use the project's interval scorer and are not an official NOTSOFAR leaderboard evaluation.
- The seven-channel arithmetic mean is only a simple downmix. No NOTSOFAR CSS model or microphone-array beamformer was applied.
- The result is useful as a pipeline check and a small labeled comparison; it is too short and too narrow to claim general diarization accuracy.

The repository README describes the official task metrics as tcpWER and tcORC-WER. Its full-set baseline numbers must not be compared directly with this activity-only 10-second test.

## Reproduce

From a visible PowerShell window in the project root, run the six-case matrix and watch each result print as it completes:

```powershell
./scripts/run_notsofar1_desktop_benchmark.ps1
```

The exact run summary is in `data/benchmarks/notsofar1-desktop-20261109/verified/diarization.json`; the original file hashes, conversion details, and activity-label interpretation are recorded in `data/benchmarks/notsofar1-desktop-20261109/notsofar1-source-manifest.json`.

Dataset source: [Microsoft NOTSOFAR-1 Challenge repository](https://github.com/microsoft/NOTSOFAR1-Challenge). See its README and data license for attribution and use terms.
