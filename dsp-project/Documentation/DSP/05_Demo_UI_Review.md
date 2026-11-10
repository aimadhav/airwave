# Demo UI and evidence review

## What the demonstration proves

Run `python app.py --port 5092`, open the workspace, then choose **Synthetic fixture**.
The generated acoustic fixture passes through normalization, energy VAD, timeline
construction, WAV clipping, the DSP-to-AI contract, analysis and storage. The UI
shows original session intervals, anonymous fixture labels and playable clips.

This is an **integration demonstration**, not a diarization accuracy benchmark.
Its signals are synthetic tones, its three speaker labels are cyclic mock labels,
and its four transcript lines are scripted. Playback does not contain those words.
The separate **Real conversation demo** now uses downloaded real speech, a CPU
diarization model, and offline Whisper. See `06_Real_Multispeaker_Validation.md`
for measured results. The low-dependency library default remains one anonymous
label; the upload UI explicitly selects real multi-speaker mode.

## Measurements and limitations

At 16 kHz, a 30 ms frame contains 480 samples. A 10 ms hop advances 160 samples.
For samples x in a frame, RMS = sqrt(mean(x²)); level = 20 log10(max(RMS, 1e-8)).
The adaptive threshold is max(−35 dBFS, percentile_10(frame levels) + 3 dB).
Active regions separated by at most about 250 ms are joined; regions shorter than
200 ms are removed. These are configurable heuristics, not a learned speech model.
Music or loud background sounds can trigger them. No acoustic source separation
or voice-profile matching is currently performed. Interval overlap is not proof
that an acoustic model detected two simultaneous speakers.

## Review changes

- Responsive light workspace; distinct header/actions; consistent speaker colours.
- Clip timeline, clear fixture provenance, keyboard-operable session selection.
- Upload dialog focus isolation, focus return, Escape dismissal and focus cycling.
- Stale session and polling responses cannot replace a newly selected session.
- Audio stops when leaving the transcript or changing sessions.
- Mock insights derive from supplied transcript lines; missing evidence stays empty.
  Old stored fixture results are preserved, with a warning to rerun the demo.
- Static asset and clip routes reject traversal; unknown sessions return 404;
  malformed or non-object JSON returns 400; missing input paths are not synthesized.
- HTTP tests use a temporary database rather than adding sessions to the workspace.
- Lesson roadmap matches its sections; sample points have nonzero amplitudes;
  VAD trace crosses the threshold; speaker labels sit outside timeline bars.

## Presentation metadata contract

`GET /api/sessions/{session_id}` retains its existing fields and adds optional `dsp`:

```text
fixture: bool
analysis_mode: string
speaker_provider: string
sample_rate: integer
channels: integer
duration_seconds: number
warnings: string[]
segments: [{segment_id, filename, start_time, end_time, speaker_id,
            speaker_source, is_overlap}]
```

It is a presentation projection of `AudioClipPayload`, not a new DSP-to-AI contract.
It is stored as `data/dsp_sessions/{session_id}/web_manifest.json`. It contains only
clip filenames, never arbitrary source paths. Only listed clips may be played through
`GET /api/audio/{session_id}/{segment_id}`. Sessions created before this metadata
was added remain readable without a timeline.

## Verification

Run `python -m pytest -q` with project dependencies installed, and
`node --check web/app.js`. Browser checks cover desktop (1280×900), mobile
(390×844), clip playback, insights, the upload dialog, Escape/focus return, and the
educational guide. External STT/LLM APIs are not exercised by offline tests.
