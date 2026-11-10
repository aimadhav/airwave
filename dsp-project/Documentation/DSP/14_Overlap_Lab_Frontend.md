# Overlap Lab: a local evaluation frontend

## Open it

Start the existing listening-room server with `Start-Listening-Room.cmd`, then open
<http://127.0.0.1:5095/lab/>. The listening-room header also has an **Overlap lab**
link. Completed uploads are available in the recording selector; the query
`?recording=<upload-id>` opens one directly.

The lab is part of the existing private local application, not a hosted Site.
No recording, transcript, reference or voice embedding is sent to a cloud API.
Nothing is automatically downloaded or published.

## The three-step workflow

### 1. Listen & label

- Pick a detected overlap, or create a custom interval based on one.
- Play the original with context, loop it, or slow it down to 0.75×.
- Adjust the scoring interval if the detector cut a word or marked a false
  interruption. The current maximum core is six seconds; two selected speakers
  must be the only active speaker IDs in that interval.
- Write a separate human transcript for each speaker. Include words whose
  midpoint lies inside the scoring interval. Context helps you hear complete
  words but is not itself scored.
- Save a draft when unsure. Only check the review box after actually listening
  and checking both references. Model output is not ground truth.

References are stored by recording ID **and source SHA-256**. Replacing the current
recording does not overwrite the old recording's reference file. Annotations have
revision numbers; a stale browser tab gets a conflict instead of silently
overwriting another tab's edits. Editing text or bounds clears the review state.

### 2. Set up comparison

The 2×2 matrix displays SepFormer/MossFormer against Whisper base.en/small.en.
Availability is determined by local assets and supported adapters, not dummy UI
state. Currently:

| Combination | Actual availability |
| --- | --- |
| SepFormer + base.en | Installed, runnable |
| SepFormer + small.en | Small model not installed |
| MossFormer + base.en | Separation adapter not connected |
| MossFormer + small.en | Separation adapter not connected; recognizer also absent |

A mixed-audio baseline is run for each selected recognizer. Context can be varied
from zero to three seconds; context stops before an unrelated third speaker.
There are no fake results or automatic model-install buttons. Installing a
compatible CTranslate2 small.en model into its registered folder enables that
recognizer after refresh. MossFormer requires implementing and registering a real
adapter; putting a weights file on disk alone does not make it runnable.

Runs execute on one local experiment worker with a bounded queue and timeout.
The upload queue is separate, so an upload and a lab run can share the CPU;
avoid doing both when measuring runtime. Refreshing reconnects to an active lab
job. Completed runs remain available after restarting the server; the UI shows
up to the configured history limit.

### 3. Inspect results

- Compare the mixed and separated recognized text.
- Listen to both separated audio tracks; their playback includes context.
- Inspect gate decisions, cosine similarities, margins and signal metrics.
- With a reviewed reference, inspect per-speaker word error rate and colored
  word-level substitutions, deletions and insertions.
- Rejected raw tracks remain visible but unassigned. You may manually assign
  them **for evaluation only** and score their recognized words. That result is
  labelled manual and does not count as successful automatic diarization.
- Export the study as JSON, including reference snapshots, comparison outputs,
  scores, manual diagnostic assignments and model fingerprints.

Each run preserves the reference snapshot it actually used. Reviewing or editing
the annotation later does not retroactively turn a draft run into a scored one.
Start a new comparison against the checked reference. The original recording and
the production transcript are never changed by a lab experiment.

## Metrics and honest comparisons

Word-error rate is `(substitutions + deletions + insertions) / reference_words`.
Case and punctuation are normalized, with Unicode word handling and normalized
apostrophes. WER can exceed 100%; it is not displayed as `100 − WER` accuracy.
An empty reference has no defined WER, but insertion counts remain meaningful.
Draft references never receive a numeric accuracy score.

The mixed baseline assigns overlapping words by timeline and marks that
attribution uncertain. Its speaker-labelled WER therefore includes this
assignment's limitations. Automatically accepted separated output is scored by
its embedding-based speaker assignment, not a text-based oracle permutation.
Manually assigning rejected tracks measures recognized words under that manual
assignment, not the system's automatic speaker-ID performance.

Separation time is shared across recognizers and displayed separately. Individual
variant times measure recognition and matching after separation, not queue time.
They include model warm-up where applicable and are development timings, not
hardware-normalized speed benchmarks. A copied separator crop and memoized
recognition outputs are reused only when their sample hashes match exactly.

The waveform is an **energy-envelope overview**, not sample-accurate amplitude
display. The highlighted region is the scoring interval; the two activity lanes
come from Nemotron's estimated timeline.

## Architecture and contracts

```text
web/overlap-lab/{index.html,lab.css,lab.js}
  → same-origin /api/lab/*
  → LabService (provider-independent coordination)
      ├─ StudyRepository → FileStudyRepository
      └─ ExperimentProcessor → LocalExperimentProcessor
          → run_overlap_lab.py (composition root / isolated worker)
              → existing separator, embedding and timestamped-ASR contracts
              → existing OverlapRecovery checks
              → pure speaker-labelled WER scoring
```

The `overlap_lab` package has independent config, DTOs, protocols, services,
adapters and typed exceptions. It does not import DSP/model runtime internals.
Concrete recording-viewer model adapters are wired only by the worker composition
root. The HTTP server remains usable without importing NumPy or PyTorch.

DTOs: `ClipSpec`, `ReferenceAnnotation`, `ExperimentRequest`, `LabJob`, and
`WordErrorScore`. Runtime policy/model registrations live in
`configs/overlap_lab.json`; provider paths are never accepted over HTTP.

The existing recovery contract gains two backward-compatible additions:

- `OverlapRecovery.recover(..., windows=None)` can evaluate specified, bounded
  ownership windows. Default production behavior still plans the whole timeline.
- `WindowReport.track_speakers: tuple[str, ...] = ()` records the actual accepted
  track permutation. The lab never infers identity by matching transcript strings.

## HTTP interface

| Method / path | Purpose |
| --- | --- |
| GET `/api/lab/bootstrap?recording=current` | Recording choices, candidate clips, capabilities, annotations, recent jobs |
| POST `/api/lab/references` | Save one reference with source hash and expected revision |
| POST `/api/lab/runs` | Submit a bounded real comparison with a reference snapshot |
| GET `/api/lab/jobs/{id}` | Job state and completed public result |
| POST `/api/lab/score` | Diagnostic scoring after explicit manual track assignment |
| GET `/lab-audio/{id}/mixture.wav` | Comparison crop playback |
| GET `/lab-audio/{id}/sepformer_track_0.wav` | Separated track playback (also track 1) |

JSON request sizes, reference character/word counts, clip length, context, queue
capacity and worker time are bounded. Scoring limits prevent enormous dynamic
programming matrices. Host/Origin checks apply to lab and upload endpoints.
Recording IDs and audio filenames are allowlisted. Configs, raw model files,
canonical WAVs, private logs, worker inputs and arbitrary filesystem paths are
not exposed. Audio supports byte ranges for seeking.

Private artifacts live under ignored `data/overlap-lab/`: source-bound reference
JSON, per-run request snapshots, processing logs, result JSON and comparison WAVs.
Model hashes and separator revision are included in the exported result.
Manual diagnostic scores currently live in the page's loaded result and the
exported study, not in the immutable original worker output; export them before
closing if you want to preserve those manual assignments.

## Verification performed

- Offline DTO, scoring, provider and HTTP tests, including wrong paths, unavailable
  models, conflicting saves, oversized scoring input and source changes.
- Completed-result reload after server restart and range-served track playback.
- Real local run on the 51.591–52.259-second passage, both through HTTP and by
  pressing **Run local comparison** in the browser.
- Mixed output: “better than that.” Separately recognized tracks: “people.” and
  “Better than that,”, assigned to different estimated speakers. These are model
  outputs, not a verified human transcript.
- Browser draft edit/save and restoration to blank, unreviewed fields. No real
  recording reference was falsely marked human-reviewed.
- Responsive viewport overflow check, JavaScript syntax checks and Python
  compilation. Browser preview is available on the running local app.

The existing beginner explainer document remains untouched and nothing was
committed, pushed, or hosted as part of this frontend task.
