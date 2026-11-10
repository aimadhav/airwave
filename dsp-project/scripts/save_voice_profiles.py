"""Persist existing recording speaker references locally, without rerunning ASR or diarization."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
for dependency in ("src", "models/asr-deps", "models/media-deps", ".runtime-deps"):
    sys.path.insert(0, str(ROOT / dependency))

import numpy as np
from recording_viewer.adapters.voice_embedder import OnnxVoiceEmbedder
from recording_viewer.adapters.voice_profile_store import JsonVoiceProfileStore
from recording_viewer.exceptions import VoiceProfileError
from recording_viewer.models import VoiceRegion
from recording_viewer.models.voice_profiles import ProfileContext, VoiceProfileSet
from recording_viewer.profile_config import VoiceProfileExportConfig
from recording_viewer.recovery_config import RecoveryConfig
from recording_viewer.services.recovery_timeline import reference_spans
from recording_viewer.services.voice_profiles import enroll_profiles


def fingerprint(path: Path) -> str:
    """Read inputs only; hashes disambiguate model weights and recording revisions."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_existing(root: Path, config: VoiceProfileExportConfig) -> tuple[Path, VoiceProfileSet, bool]:
    """Reuse source timelines, enroll clean references, then verify a disk round-trip."""
    folder = (root / config.recording_directory).resolve()
    output = (root / config.output_directory).resolve()
    if (not folder.is_relative_to((root / "data/listening-room").resolve())
            or not output.is_relative_to((root / "data/voice-profiles").resolve())):
        raise VoiceProfileError("Profiles must use local recording inputs and the private data/voice-profiles directory")
    try:
        policy = RecoveryConfig.load(root / config.recovery_config)
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        identity = json.loads((folder / "source.json").read_text(encoding="utf-8"))
        original = folder / ("audio" + Path(identity["filename"]).suffix.lower())
        if fingerprint(original) != identity["sha256"] or identity["sha256"] != provenance["source_sha256"]:
            raise VoiceProfileError("Recording source fingerprint changed; no profiles saved")
        audio_path, timeline_path = folder / "processing.wav", folder / "speaker-timelines.json"
        reference_policy = {k: asdict(policy)[k] for k in (
            "reference_seconds", "reference_guard_seconds", "max_references", "minimum_references")}
        context = ProfileContext(identity["sha256"], fingerprint(audio_path), fingerprint(timeline_path),
            fingerprint(root / policy.embedding_model),
            hashlib.sha256(json.dumps(reference_policy, sort_keys=True).encode()).hexdigest(),
            policy.backend, 16000, policy.separator_sample_rate, config.preprocessing_version)
        repository = JsonVoiceProfileStore(output)
        path = repository.path_for(context.key)
        if path.exists():
            existing = repository.load(context.key)
            return path, existing, True
        with wave.open(str(audio_path), "rb") as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
                raise VoiceProfileError("Voice enrollment requires the original canonical 16 kHz mono PCM16 WAV")
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32)/32768
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        regions = [VoiceRegion(**row) for row in timeline[policy.backend]]
        spans = reference_spans(regions, len(samples)/context.sample_rate, policy)
        profiles = enroll_profiles(samples, context, spans, OnnxVoiceEmbedder(policy), policy.minimum_references)
        repository.save(profiles)
        loaded = repository.load(context.key)
        if loaded != profiles:
            raise VoiceProfileError("Saved voice-profile round-trip verification failed")
        return path, loaded, False
    except (OSError, KeyError, TypeError, ValueError, wave.Error) as error:
        raise VoiceProfileError("Cannot enroll this recording's local voice profiles") from error


def main() -> None:
    """Print a summary, never the vectors themselves or private source audio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/voice_profiles.json")
    args = parser.parse_args()
    path, profiles, reused = save_existing(ROOT, VoiceProfileExportConfig.load(args.config))
    print(json.dumps({"path": str(path), "reused": reused, "source_sha256": profiles.context.source_sha256,
        "backend": profiles.context.backend, "embedding_dimension": len(profiles.profiles[0].centroid),
        "sample_rate": profiles.context.sample_rate, "bandlimit_rate": profiles.context.bandlimit_rate,
        "speakers": [{"speaker_id": p.speaker_id, "reference_count": len(p.references),
                      "identity_status": p.identity_status} for p in profiles.profiles], "warnings": profiles.warnings}, indent=2))


if __name__ == "__main__":
    main()
