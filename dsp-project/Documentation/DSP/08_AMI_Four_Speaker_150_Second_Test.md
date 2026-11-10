# Four-person AMI conversation: 150-second real-audio test

Date: **6 October 2026**. This test uses one continuous English meeting excerpt, not repeated audio, concatenated speakers, synthetic speech, or a supplied native transcript. The existing default configuration was tested first and has **not been changed**.

## Downloaded sample

- Meeting: **AMI ES2002a**, headset microphone mix.
- Excerpt: **03:00–05:30 in the original meeting**, exactly **150 seconds / 2 minutes 30 seconds**.
- File: `data/benchmarks/ami-es2002a/ami-four-speakers-150s.wav`.
- Format: 16 kHz, mono, 16-bit PCM; 2,400,000 frames.
- Audio SHA-256: `e35bc5c0a16bf614e61d1412b6ea1d8782271cf9f02b68d970546f696463d6fb`.
- It is an icebreaker conversation within a four-person design-team meeting; the local recognized text discusses favourite animals. It is not a podcast or a completed project-decision discussion.

[Official source audio directory](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/) · [AMI corpus description](https://groups.inf.ed.ac.uk/ami/corpus/) · [Download and attribution information](https://groups.inf.ed.ac.uk/ami/download/).

The publisher releases the recording under CC BY 4.0. Attribute the **AMI Meeting Corpus**, J. Carletta et al., *The AMI meeting corpus: A pre-announcement* (2006). The local derived excerpt retains exactly the original samples in this interval; only its WAV container and time origin are changed.

The source server was slow, so the download was intentionally stopped after enough of the source WAV had arrived. The retained `ES2002a.Mix-Headset.download-prefix.part` contains 19,116,032 source bytes, not the complete meeting. The preparation script checks that every requested frame exists and writes a complete, independently playable 150-second WAV. It never pads unavailable source frames with silence. An unused partial manual-annotation archive also remains labelled `.zip.part`; it was not used for scoring.

`excerpt_manifest.json` records URLs, selection reason, frame count, speaker participation and SHA-256 values. Its downloaded-source hash describes the **partial source bytes**, not the full meeting. The derived WAV is complete. Downloaded audio, annotations, logs, and reports remain Git-ignored.

## Reference: how we know four speakers are present

Speaker timings come from the [BUTSpeechFIT/pyannote AMI diarization setup](https://github.com/pyannote/AMI-diarization-setup), which derives words-only speech intervals from AMI manual annotations v1.6.2. It is not a transcript produced by our model. The original reference includes pauses and overlapping speech; we intersect it with the excerpt and subtract 180 seconds from each retained interval.

The interval was selected **before inference**, using reference participation rather than searching for the model's best score. The anonymous reference IDs are not participant names.

| Reference speaker | Annotated speaker-time in this excerpt |
|---|---:|
| FEE005 | 20.49 s |
| MEE006 | 17.31 s |
| MEE007 | 8.18 s |
| MEE008 | 44.01 s |
| Total | **89.99 speaker-seconds**, in 25 reference intervals |

Four people appear, but their participation is unequal. Total speaker-time is not the 150-second recording duration: pauses do not count and simultaneous speakers can count separately.

Source RTTM SHA-256: `3153efc675b16b7c47c6e49a3b48602613b3dfcf0ca16f590fb7541176c0f0b2`.

Scoring reference: `data/benchmarks/ami-es2002a/ami-four-speakers-150s.rttm`. Its hash is `95e60f2922bbd6b393330fe7978ca966fd2018f3a4c1bcf5bfc19297d0d8da3f`.

## Results: default first, then diagnostic comparisons

All runs used real CPU inference: pyannote segmentation 3.0, TitaNet-small embeddings, and the existing sherpa-onnx clustering adapter. No API keys or manual speaker-name labels were supplied to inference.

| Run | Count supplied to model? | Threshold | Predicted labels | Clips | DSP runtime | Zero-collar DER | ±250 ms collared DER |
|---|---|---:|---:|---:|---:|---:|---:|
| **Existing default** | No | 0.9 | **9** | 25 | 17.78 s | **26.87%** | 18.76% |
| Experimental threshold | No | 1.0 | 6 | 25 | 15.94 s | 25.34% | 18.32% |
| Higher threshold | No | 1.1 | 3 | 24 | 16.21 s | 38.44% | 31.98% |
| Explicit known-count comparison | **Yes: 4** | 0.9 | 4 | 24 | 16.33 s | 38.44% | 31.98% |

**This is not a successful automatic four-speaker diarization result.** The default over-fragments voices. Increasing the threshold reduces fragmentation but eventually merges distinct voices. Forcing four groups makes the count look right but produces worse attribution. The experimental settings were evaluated on this same excerpt; they are diagnostic tuning probes, not independent improvements or new production defaults.

Every generated clip in these runs passed 16-kHz mono and interval-duration checks, within two sample periods. Correct file format does not imply correct speaker assignment.

The default runtime's real-time factor was approximately **0.119**, measured for model initialization, DSP, timeline construction and clip writing. This does **not** include STT or downstream AI. CPU load and parallel execution can affect runtime comparisons.

### What the error includes

The local interval-exact scorer uses optimal anonymous-label matching and includes overlapping speech:

```text
DER = (missed speaker-time + false-alarm speaker-time + confused speaker-time)
      / reference speaker-time
```

Default zero-collar components:

- Missed speech: **8.593 speaker-seconds**.
- False-alarm speech: **5.897 speaker-seconds**.
- Confused speaker attribution: **9.691 speaker-seconds**.
- Denominator: **89.99 reference speaker-seconds**.

The collared score excludes timing near human reference boundaries and uses a different denominator, 72.87 speaker-seconds. It is reported separately, not substituted for the primary result. Words-only annotations also exclude some vocal sounds, so audible non-word vocalizations can contribute apparent false alarms.

This is a local scorer, not an official AMI leaderboard evaluation. AMI is a standard research corpus; it may overlap training data of pretrained components. We do not claim a model-held-out or universal accuracy result. A count of four is not equivalent to correct timing or attribution, and DER is not word error rate.

## Full pipeline test: genuine STT and stored session

The default output was also run through:

```text
150-second real WAV
  → default automatic neural diarization
  → 25 timestamped WAV clips / 9 anonymous labels
  → real offline Whisper tiny English
  → existing local DSP-to-AI contract
  → offline deterministic/extractive analysis
  → SQLite storage
```

Stored session: **`ami_es2002a_150s_auto_20261006`**.

A read-only SQLite check confirmed:

- Status: `completed`.
- 25 stored transcript turns and nine anonymous labels.
- Zero extracted facts, zero decisions, and one heuristic action item.

Log: `data/benchmarks/ami-es2002a/pipeline_run.log`.

Payload: `data/dsp_sessions/ami_es2002a_150s_auto_20261006/payload.json`.

### STT and analysis findings

The recognizer produced actual words, not a scripted transcript, but there are obvious substitutions and fragments, especially on short interjections and overlap. An independent text reference was not aligned for this test, so **no WER or transcription accuracy percentage is claimed**.

The heuristic analysis treated a sentence containing “I'm going to…” in this informal discussion as an action item. That is an **apparent false positive**, not a verified project task. End-to-end completion proves the contract and storage path, not correctness of the extracted meaning. A full LLM was not used in this run.

The local web server on port 5092 was not running when checked, so this test does not claim a new browser/playback inspection. The CLI stores a DSP payload, while the web timeline uses its own upload-generated manifest; the stored CLI session alone is not proof of a populated web speaker timeline. Start the application and upload the complete WAV with real detection to inspect that path separately.

## Reproduce the tests

Use the already prepared WAV and RTTM, with the optional diarization/STT dependencies and assets installed:

```powershell
# Existing automatic settings, no known speaker count:
python scripts/benchmark_diarization.py data/benchmarks/ami-es2002a/ami-four-speakers-150s.wav --reference data/benchmarks/ami-es2002a/ami-four-speakers-150s.rttm --output data/benchmark_results/ami-four-speakers

# Keep diagnostic experiments in separate output folders:
python scripts/benchmark_diarization.py data/benchmarks/ami-es2002a/ami-four-speakers-150s.wav --reference data/benchmarks/ami-es2002a/ami-four-speakers-150s.rttm --threshold 1.0 --output data/benchmark_results/ami-four-speakers/experimental-1.0
python scripts/benchmark_diarization.py data/benchmarks/ami-es2002a/ami-four-speakers-150s.wav --reference data/benchmarks/ami-es2002a/ami-four-speakers-150s.rttm --threshold 1.1 --output data/benchmark_results/ami-four-speakers/experimental-1.1
python scripts/benchmark_diarization.py data/benchmarks/ami-es2002a/ami-four-speakers-150s.wav --reference data/benchmarks/ami-es2002a/ami-four-speakers-150s.rttm --speakers 4 --output data/benchmark_results/ami-four-speakers/known-four

# New end-to-end session; STT is real, analysis is offline/extractive:
python cli.py ingest data/benchmarks/ami-es2002a/ami-four-speakers-150s.wav --mode mock --dsp-provider local_model
```

Raw report directories contain `report.json`, `prediction.rttm`, and `clips/`. Prediction times are relative to the 150-second excerpt; add 180 seconds to locate them in the original meeting. Use a new output folder when preserving an earlier benchmark run.

### Prepare the excerpt again in a clean workspace

`configs/ami_four_speaker_test.json` externalizes the URLs, source paths, exact interval, provenance, and expected reference count. `scripts/prepare_ami_excerpt.py` performs no inference and rejects missing samples, escaped artifact paths, invalid intervals and existing output files.

Download the official source (or a sufficient byte prefix) and the reference to the configured paths, then run:

```powershell
python scripts/prepare_ami_excerpt.py
```

Do not rerun preparation over the existing artifacts: it deliberately refuses to overwrite them. `excerpt_manifest.json` is the record of the actual downloaded and derived bytes used for this test.

## Verification of the preparation code

The final offline regression run passed **110 tests**, with one optional real-model integration test skipped by default. Real inference was nevertheless executed in the four benchmark runs and the separate full CLI run described above.

Seven new excerpt tests check exact PCM slicing, time-origin shifting, overlap retention, invalid intervals, truncated-prefix rejection, no-overwrite behaviour, and typed configuration/path validation. These tests use tiny local fixtures and require no AMI downloads or external APIs.

```powershell
python -m pytest -q
python -m pytest -q tests/test_ami_excerpt.py tests/test_diarization_evaluation.py
```

## What this additional test tells us

The earlier two-speaker result and short four-speaker count check did not establish reliable multi-speaker generalization. This longer, different English conversation exposes both fragmentation and merging. Its valid four-speaker reference makes those failures measurable instead of judging by a coloured timeline alone.

Next work should focus on stronger speaker embeddings/clustering and short overlapping interjections, evaluated on several independent excerpts with full attribution scoring. A lower label count should not be accepted as an improvement without a corresponding reduction in speaker confusion. Transcription quality and contextual action extraction need separate evaluation as well.

The downloaded 150-second sample is ready for further experiments, but **the existing algorithm is not yet a reliable four-speaker demo on this excerpt**. No global algorithm settings were changed to disguise that result.
