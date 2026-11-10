# Local reusable voice profiles

There are two different kinds of speaker memory:

1. **Nemotron's internal cache** maintains anonymous speaker channels during its
   inference session. We do not export or restore that internal runtime state.
2. **TitaNet embeddings** describe clean reference speech used by overlap recovery.
   Previously, the recovery code held the normalized references and centroids
   only in memory. Its saved reports contained reference counts and match scores,
   not reusable vectors.

`scripts/save_voice_profiles.py` now explicitly enrolls and persists the second
kind, using the existing recording and speaker timeline. It runs only reference
embedding extraction, not diarization, transcription, separation or a full rebuild.
It does not change the active processing pipeline or automatically identify people
in future recordings.

## Save the configured recording

Use the existing project Python environment and dependency paths:

```powershell
python scripts/save_voice_profiles.py --config configs/voice_profiles.json
```

The configuration selects a local recording directory and existing recovery
policy. The exporter uses that policy's exclusive reference planner: 2-second
clips, 0.15-second edge guards, up to five references spread over the recording,
and at least two successful references per enrolled speaker under current policy.
It never enrolls known overlapping intervals. The source diarization can still be
wrong, so all saved profiles remain `unverified`.

## What is saved

Each immutable JSON set contains:

- A normalized centroid and every normalized reference embedding per speaker.
- Sample-aligned source start/end times for the reference vectors.
- Source-recording, canonical-audio, timeline, encoder-model and reference-policy
  SHA-256 fingerprints.
- Backend, sample rate, bandwidth matching and preprocessing-version metadata.
- Creation time, schema version, explicit failures/warnings and anonymous IDs.

Files stay under **`data/voice-profiles/`**, already ignored by Git. They are not
included in the website or public demo. Treat them as biometric data: local and
Git-ignored does not mean encrypted. No extra reference-audio copies are created.

The context fingerprint is the repository key. Changing the source, diarization
timeline, encoder weights, reference policy or preprocessing creates a different
key, preventing reuse of a stale `Speaker 1` profile by accident. Saving never
overwrites an existing set; an identical export validates and reuses it instead
of rerunning inference.

## Reuse the saved vectors

At a composition boundary, select the saved key and load the canonical DTO:

```python
from pathlib import Path
import numpy as np
from recording_viewer.adapters.voice_profile_store import JsonVoiceProfileStore

store = JsonVoiceProfileStore(Path("data/voice-profiles"))
saved = store.load(saved_key)
saved.require_compatible(
    encoder_sha256=current_encoder_sha256,
    sample_rate=16000,
    bandlimit_rate=8000,
    preprocessing_version="titanet_polyphase_bandlimit_v1",
)
references = {p.speaker_id: np.asarray(p.centroid) for p in saved.profiles}
```

The vectors can be supplied to the existing recovery matcher after compatibility
checks. Unit tests exercise that round-trip with deterministic mock embeddings.
New probe embeddings must use the same encoder weights and preprocessing. In the
current recovery setup, both references and separated speech are bandwidth-matched
by antialiased 16 kHz → 8 kHz → 16 kHz resampling before TitaNet extraction. Do not
compare those vectors blindly with a different model or wideband-only embeddings.

Speaker numbers are **recording/timeline-scoped acoustic groups**, not global
identities. Cross-recording similarity is a hypothesis to evaluate, not proof of a
person's identity. Registering a real name or automatically relabelling future
recordings is a separate, explicit workflow and is not enabled by this export.

## Isolation and tests

New domain logic depends only on the existing `VoiceEmbedder` protocol and typed
profile contracts. Persistence has a separate repository protocol, a local atomic
no-overwrite adapter and an in-memory mock. Tests cover reference aggregation,
serialization, corruption, model/preprocessing mismatch, no-overwrite behavior,
privacy boundaries, partial provider failures and existing-matcher compatibility.
