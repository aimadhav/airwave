# Reviewed DSP viewer and overlap-lab release

Review date: 18 October 2026. Branch: `codex/dsp-audio-processing`.

## Scope

This release includes the optional native diarization adapter, local recording
viewer and uploads, conservative overlap recovery, clip-comparison lab,
listening-only review workflow, and reproducible local evaluation tools.
The downstream AI contracts and existing mock-first test paths remain available.

Raw recordings, separated tracks, model weights, installed runtimes, human
annotations and generated experiment reports are not distributed in Git.
The beginner explainer previously requested to remain local is also ignored.

## Findings corrected during review

1. **Fresh-start failure.** The server previously required an already generated
   recording page. It now creates an upload-only landing page when no page exists.
   This needs neither audio nor model inference and never overwrites an existing
   recording page. Processing still requires explicitly installed local models.
2. **Ambiguous comparison audio.** The sidebar's selected passage can differ from
   the saved comparison selected in history. Saved results now have a dedicated
   original-mixture player whose job ID matches the two separated tracks.
3. **Context/text mismatch.** The result panel now states the full playback crop,
   the scoring interval, and that interval's offsets in each player. Text and WER
   remain core-only; this change does not retrospectively regenerate transcripts.
4. **Indistinguishable history entries.** History labels now include gain mode
   and requested context, distinguishing otherwise identical timestamps. History
   ordering also preserves new persisted runs ahead of old completed server-cache
   entries while keeping running jobs first.
5. **Machine-specific source paths.** Versioned viewer policies use a portable
   example source. The typed loader optionally merges an ignored sibling
   `*.local.json` policy, allowing a private local recording path without pushing it.
6. **Empty native input.** The native adapter now rejects empty audio before
   launching the external runtime.

The previously tested shared-peak separator gain correction remains enabled.
No recognition or speaker-matching threshold was lowered to obtain more accepted
results. Cross-track voice leakage and false overlap alarms are still limitations.

## Running locally

From the repository root, after installing the project's Python dependencies:

```powershell
python scripts/serve_recording_viewer.py
```

On Windows, `Start-Listening-Room.cmd` prefers the configured local DSP virtual
environment when present. The app binds only to loopback. Its default address is
configured in `configs/recording_upload.json`, currently `http://127.0.0.1:5095/`.
An empty output directory now offers an upload page rather than refusing startup.

For actual processing, configure/install the local DSP, timestamped recognizer
and optional separator runtimes described in DSP documents 09 and 11–14.
Uploaded jobs override the example source automatically. A manually configured
recording can use an ignored file such as `configs/recording_viewer.local.json`:

```json
{
  "source": "D:/my-private-recordings/meeting.wav"
}
```

Relative paths are interpreted from the repository-root working directory.
The checked-in executable paths target the Windows development setup; operators
on other platforms must configure their corresponding interpreter/runtime paths.
No model weights or platform binaries are checked in or downloaded on upload.

## Understanding a comparison

Open **Overlap Lab → 03 Inspect results → Comparison history**. Each entry names
the core timestamp, gain mode, context and run number. Use **Original mixture for
this saved comparison**, then the two separated players, to compare the same
audio. The sidebar player is for its selected passage, not necessarily that run.

For the reviewed 54:54.49 example with one-second requested context, players cover
54:53.49–54:56.95, while displayed words cover 54:54.49–54:55.95. Thus the scoring
core is 1.00–2.46 seconds into the playback clip. A word audible outside that core
is not automatically a missing word in its displayed transcript.

“Accepted” means engineering gates passed, not that humans verified the words or
that the sources are perfectly isolated. A short phrase leaking into a track
must not automatically be attributed to its dominant speaker. This release makes
that limitation visible; it does not implement word-level leakage classification.

## Verification

- Full offline suite: **330 passed, 4 skipped**. The skipped tests are opt-in
  real-model benchmarks; the default suite does not download assets.
- Focused regressions cover initial landing creation, existing-page preservation,
  script-data escaping, typed local overrides and empty native input.
- Real-browser checks verified labelled history entries, switching between context
  variants, the matching mixture URL, actual mixture duration, and the exact
  playback/scoring explanation without changing human annotations.
- A separate empty-directory server successfully displayed the upload dialog,
  supported formats and size/duration policy without processing a recording.

The real 12-passage gain comparison is documented separately in document 16.
Its gate-pass counts are not word-error accuracy or full-recording recall.
