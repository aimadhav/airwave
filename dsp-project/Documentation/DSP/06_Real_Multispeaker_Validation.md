# Real multi-speaker demo and validation

Date: 6 October 2026. This report distinguishes downloaded real audio, generated robustness probes, mock integration fixtures, and measured accuracy. No API keys were used for acoustic diarization or speech-to-text.

## What is implemented

The real path is **uploaded audio → mono 16 kHz → DC/peak conditioning → learned speech segmentation → voice embeddings → clustering → timestamped WAV clips → offline Whisper → existing AI contract → stored transcript and offline analysis**.

- `local_model`: real CPU inference using pyannote segmentation 3.0 and NeMo TitaNet-small voice embeddings through sherpa-onnx. Labels are anonymous, stable in first-appearance order, and marked `diarization`.
- `single_speaker`: dependency-light energy VAD, not recognition of speaker changes.
- `mock`: deterministic fixture, never claimed to detect actual people.
- Native Meet transcripts still take priority when supplied through the existing artifact pipeline.
- The dashboard has a separate **Real conversation demo** button, speaker-specific timeline lanes, overlap hatching, and playable WAV clips. Uploads can choose real detection and optionally supply a genuinely known speaker count.
- The CLI now uses the canonical DSP payload rather than the older segmentation route with a hard-coded participant registry.

The models do not supply calibrated speaker confidence. The existing numeric field is `0` for unavailable, with a warning; it must not be interpreted as a measured probability. Diarization is not source separation: overlap clips still contain mixed voices. Offline analysis is deterministic/extractive, not a live LLM.

## Downloaded recordings and measured results

Sources are publisher-hosted test recordings, not a podcast. All models and recordings are ignored by Git. `configs/diarization_assets.json` records exact URLs and SHA-256 hashes; the setup script verifies them and never executes downloaded code.

| Recording | Duration | Reference speakers | Automatically returned | Accuracy evidence |
|---|---:|---:|---:|---|
| pyannote annotated meeting | 30.000 s | 2, with overlapping speech | 2 | 5.99% DER, zero boundary collar, overlap included |
| English sample 1 | 16.000 s | publisher labels it two-speaker | 2 | Count and clip integrity only; no human timing reference |
| English sample 2 | 34.000 s | publisher labels it two-speaker | 2 | Count and clip integrity only |
| English conversation 3 | 54.805 s | publisher labels it two-speaker | 2 | Real STT + dashboard + storage verified; no WER reference |
| Chinese four-speaker sample | 56.861 s | publisher labels it four-speaker | 4 | Diarization only; English Whisper was not used on Chinese |

Matching a count does **not** establish correct attribution throughout a recording. Only the annotated meeting was scored for timing and attribution. Results on this small development collection are not a general accuracy guarantee or a held-out benchmark claim.

### Actual improvement over the old fallback

On the same annotated recording and scorer:

| Provider | Zero-collar DER | DER with ±250 ms reference-boundary exclusion |
|---|---:|---:|
| Energy VAD + one anonymous speaker | 49.57% | 46.39% |
| Real segmentation + embeddings + clustering | 5.99% | 0.00% |

The zero-collar score includes approximately 0.465 missed speaker-seconds, 0.989 false-alarm speaker-seconds, and 0.0034 confused speaker-seconds over 24.35 reference speaker-seconds. Overlap contributes multiple speaker-seconds at the same moment.

The 0% collared number is **not** perfect recognition: it excludes 8.01 reference speaker-seconds near boundaries. The zero-collar score is the primary result. Our local interval-exact scorer uses globally optimal label assignment; it is tested but is not presented as an official benchmark implementation.

### Robustness and longer-runtime probes

| Probe derived from downloaded audio | Duration | Labels returned | Zero-collar DER |
|---|---:|---:|---:|
| Meeting reduced to 3% amplitude, then PCM16 quantized | 30 s | 2 | 9.65% |
| Meeting with seeded Gaussian noise at 15 dB SNR | 30 s | 2 | 10.14% |
| Meeting converted to 48 kHz stereo, then canonicalized | 30 s | 2 | 6.13% |
| Four-speaker sample repeated four times | 227.443 s | 4 with final threshold | Not scored; no human RTTM |

The 3.8-minute file is a **synthetic repetition stress test**, not an independent long meeting downloaded from the internet. It exercises processing and label consistency across a longer input. The final rerun took 63.27 seconds including model initialization, conditioning, inference, timeline building, and clipping (RTF 0.278); another run took 76.34 seconds under different CPU load. All inspected clips had correct 16-kHz mono format and duration within two sample periods of their interval.

## What testing changed

1. Replaced cyclic mock speaker assignment in real mode with learned voice embeddings and clustering.
2. Kept model speech detection independent of the energy-VAD gate, which can fail on loud noise or continuous speech.
3. Added anti-aliased polyphase resampling instead of linear interpolation in real-model mode.
4. The initial threshold `0.5` split the four-speaker sample into eight labels. At `0.8` it returned four on the short file but five on repetition; `0.85` still over-split the repeat. The final `0.9` returned four on both, while preserving two on the English files and 5.99% meeting DER. **This threshold was tuned on these development examples.** Similar voices in other recordings could be merged; a known-count override is explicit rather than disguised as automatic detection.
5. Fixed the downstream identity resolver treating “I am very…” as a name. Unregistered acoustic labels remain anonymous, including when the transcription contains a name.
6. Added real offline Whisper STT and chunked long clips into bounded windows so the rest of a long segment is not silently discarded. Word errors near chunk boundaries remain possible; transcript timestamps are segment/turn timestamps, not precise word alignment.
7. Removed the dependency on cloud inference for the real demo. Missing assets fail with setup instructions; they never silently switch to fake transcription.

## Run the demo

From the repository root, with Python 3.12 (the tested version) and project dependencies installed:

```powershell
python -m pip install -r requirements.txt -r requirements-diarization.txt
python scripts/setup_diarization.py --with-stt --samples
python app.py --port 5092
```

Open `http://127.0.0.1:5092/`, click **Real conversation demo**, wait for completion, and play segments from the two speaker lanes. It processes the real 54.805-second English conversation. The synthetic fixture remains separately labelled.

The final dashboard run completed in 36.47 seconds: **two automatic speaker labels, eight playable WAV clips, and five stitched real-STT turns**. The timeline and transcript were inspected in the browser, and actual clip playback was verified. Summary text is explicitly labelled an offline extractive overview. Asking “What is generative AI?” through the query endpoint returned stored transcript evidence with three speaker/time/session citations; no external LLM was used.

Process another recording:

```powershell
python cli.py ingest recording.wav --mode mock --dsp-provider local_model
# Only if the count is genuinely known:
python cli.py ingest recording.wav --mode mock --dsp-provider local_model --num-speakers 3
```

Here `--mode mock` selects offline analysis; with `local_model`, STT is real local Whisper. It does not make the audio or speaker detection a mock. WAV is native; MP3/M4A/MP4/WebM/OGG/FLAC require FFmpeg on PATH. FFmpeg decoding was not available in this workstation's measured runs.

Reproduce accuracy and stress results:

```powershell
python scripts/benchmark_diarization.py data/benchmarks/annotated-meeting.wav --reference data/benchmarks/annotated-meeting.rttm
python scripts/benchmark_diarization.py data/benchmarks/annotated-meeting.wav --reference data/benchmarks/annotated-meeting.rttm --provider single_speaker
python scripts/stress_diarization.py
python -m pytest -q
$env:RUN_REAL_DSP='1'
python -m pytest -q tests/test_real_multispeaker_smoke.py
Remove-Item Env:RUN_REAL_DSP
```

Raw report JSON, predicted RTTM, and WAV clips live under `data/benchmark_results/`. The regression run passed **103 tests**, with the one real-model integration test skipped by default. Normal unit tests use mock inference and do not download models or need API keys. The opt-in real integration test was executed separately and passed. It checks actual diarization, clip integrity, Whisper words, anonymous attribution, and AI-layer handoff.

## Remaining limitations

- No word-error-rate score: these runs have no independently verified text reference. Whisper tiny English makes audible-word substitutions, including “generative air”; the transcript must be reviewed.
- No source separation or guaranteed recovery of both voices during overlap.
- Short interjections, similar voices, music, strong noise, accents, very large meetings, and channel cancellation can degrade results.
- A high clustering threshold trades fewer fragments for a greater risk of merging different voices. A detected count alone is not proof of diarization quality.
- Offline facts/actions are deliberately conservative. A discussion without explicit commitments can correctly show zero action items.
- CPU inference is offline/batch, not a production live Meet stream. Direct Meet OAuth, registered voice matching, multilingual STT, and production hardening remain separate work.
- Model and source-data licenses should be reviewed before redistribution or deployment. The segmentation archive's license is preserved locally; recordings are not bundled in Git.

## Primary sources

- [Sherpa diarization model documentation](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html)
- [Publisher-hosted diarization samples and models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models)
- [Annotated meeting and RTTM](https://github.com/pyannote/pyannote-audio/tree/develop/tutorials/assets)
- [Offline Whisper model documentation](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html)
- [Cosine-distance clustering implementation](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/fast-clustering.cc)

The earlier beginner pipeline explainer remains local and was not changed or pushed by this implementation.
