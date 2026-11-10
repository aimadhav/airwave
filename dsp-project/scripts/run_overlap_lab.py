"""Composition root for real, private, clip-level comparison experiments."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "src", ROOT / "models/asr-deps", ROOT / ".runtime-deps"):
    sys.path.insert(0, str(dependency))
import numpy as np
from overlap_lab.models import ExperimentRequest
from overlap_lab.services.scoring import score_speakers
from recording_viewer.adapters.faster_whisper import FasterWhisperTranscriber
from recording_viewer.adapters.local_separator import LocalSpeechSeparator
from recording_viewer.adapters.voice_embedder import OnnxVoiceEmbedder
from recording_viewer.config import RecordingViewerConfig
from recording_viewer.interfaces.recovery import SpeechSeparator
from recording_viewer.models import TranscriptWord, VoiceRegion
from recording_viewer.models.recovery import RecoveryWindow, SeparationOutput
from recording_viewer.recovery_config import RecoveryConfig
from recording_viewer.services.alignment import align_words
from recording_viewer.services.overlap_recovery import OverlapRecovery


def sample_key(samples: np.ndarray, rate: int) -> str:
    return hashlib.sha256(str(rate).encode() + samples.tobytes()).hexdigest()


class MemoizedSeparator:
    """Reuse precisely the same crop, not arbitrary outputs by track index."""
    def __init__(self, provider: SpeechSeparator) -> None:
        self.provider = provider
        self.cache: dict[str, SeparationOutput] = {}

    def separate(self, samples: np.ndarray, sample_rate: int) -> SeparationOutput:
        key = sample_key(samples, sample_rate)
        if key not in self.cache:
            self.cache[key] = self.provider.separate(samples, sample_rate)
        return self.cache[key]


class MemoizedTranscriber:
    def __init__(self, provider: FasterWhisperTranscriber) -> None:
        self.provider = provider
        self.cache: dict[str, list[TranscriptWord]] = {}

    def transcribe_samples(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptWord]:
        key = sample_key(samples, sample_rate)
        if key not in self.cache:
            self.cache[key] = self.provider.transcribe_samples(samples, sample_rate)
        return self.cache[key]


def save_audio(path: Path, samples: np.ndarray, rate: int, gain: float) -> None:
    """Additional common playback attenuation; inference uses adapter output."""
    with wave.open(str(path), "wb") as target:
        target.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        target.writeframes(np.rint(np.clip(samples / gain, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    request = ExperimentRequest.from_mapping(payload["request"])
    study, config = payload["study"], payload["lab_config"]
    reference, clip = request.reference, request.reference.clip
    viewer_config = RecordingViewerConfig.load(ROOT / config["viewer_config"])
    policy = replace(RecoveryConfig.load(ROOT / config["recovery_config"]), context_seconds=request.context_seconds)
    canonical = Path(payload["recording_folder"]) / "processing.wav"
    with wave.open(str(canonical), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Lab requires canonical audio")
        rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if clip.end > len(samples) / rate or clip.end - clip.start > policy.max_core_seconds:
        raise ValueError("Requested clip exceeds limits")
    regions = [VoiceRegion(**row) for row in study["regions"]]
    left, right = max(0., clip.start-request.context_seconds), min(len(samples)/rate, clip.end+request.context_seconds)
    for row in regions:
        if row.speaker not in clip.speakers:
            if row.end <= clip.start:
                left = max(left, row.end)
            elif row.start >= clip.end:
                right = min(right, row.start)
    window = RecoveryWindow(clip.id, clip.start, clip.end, left, right, clip.speakers)
    offset = round(left * rate)
    crop_offset = offset / rate
    crop = samples[offset:round(right * rate)]
    separator = MemoizedSeparator(LocalSpeechSeparator(ROOT, policy, args.output.parent / "scratch"))
    started = time.monotonic()
    separated = separator.separate(crop, rate)
    separation_seconds = time.monotonic() - started
    gain = max(1., float(np.max(np.abs(crop))), *(float(np.max(np.abs(t))) for t in separated.tracks))
    save_audio(args.output.parent / "mixture.wav", crop, rate, gain)
    for i, track in enumerate(separated.tracks):
        save_audio(args.output.parent / f"sepformer_track_{i}.wav", track, rate, gain)
    embedder = OnnxVoiceEmbedder(policy)
    options = {m["id"]: m for m in config["models"]}
    results: list[dict[str, object]] = []
    used_models: dict[str, str] = {}
    for model_id in dict.fromkeys(variant.split("--")[1] for variant in request.variants):
        model = options[model_id]
        model_path = ROOT / model["directory"] / "model.bin"
        with model_path.open("rb") as source:
            used_models[model_id] = hashlib.file_digest(source, "sha256").hexdigest()
        recognizer = MemoizedTranscriber(FasterWhisperTranscriber(replace(viewer_config,
            model_directory=str(ROOT / model["directory"]), model_repository=model["repository"])))
        started = time.monotonic()
        mixed_words = recognizer.transcribe_samples(crop, rate)
        mixed_words = [TranscriptWord(w.start + crop_offset, w.end + crop_offset, w.text, w.probability)
                       for w in mixed_words if clip.start <= crop_offset + (w.start+w.end)/2 < clip.end]
        clipped_regions = [VoiceRegion(r.id, r.speaker, max(r.start, clip.start), min(r.end, clip.end)) for r in regions
                           if min(r.end, clip.end) > max(r.start, clip.start)]
        aligned = align_words(mixed_words, clipped_regions, viewer_config.minimum_word_coverage)
        hypotheses = {speaker: " ".join(t.text for t in aligned if t.speaker == speaker and t.text) for speaker in clip.speakers}
        results.append({"id": f"mixed--{model_id}", "label": f"Mixed audio + {model['label']}", "kind": "baseline",
            "status": "timing_only", "reason": "Overlapping words assigned by timing; speaker attribution is uncertain.",
            "elapsed_seconds": time.monotonic()-started, "speakers": hypotheses,
            "scores": score_speakers(reference, hypotheses), "tracks": [], "metrics": {}})
        started = time.monotonic()
        tracks = []
        for i, track in enumerate(separated.tracks):
            words = recognizer.transcribe_samples(track, rate)
            owned = [w for w in words if clip.start <= crop_offset + (w.start+w.end)/2 < clip.end]
            tracks.append({"index": i, "text": "".join(w.text for w in owned).strip(),
                "audio": f"sepformer_track_{i}.wav", "assigned_speaker": None})
        recovery = OverlapRecovery(policy, separator, embedder, recognizer).recover(samples, rate, regions, windows=[window])
        report = recovery.reports[0]
        hypotheses = {speaker: "".join(w.word.text for w in report.words if w.speaker == speaker).strip() for speaker in clip.speakers}
        # Preserve raw unassigned track text even if production gates reject it.
        for track in tracks:
            if report.status == "accepted":
                track["assigned_speaker"] = report.track_speakers[track["index"]]
        gain_label = " (shared-peak gain)" if policy.separator_gain_mode == "shared_peak" else ""
        results.append({"id": f"sepformer--{model_id}", "label": f"SepFormer{gain_label} + {model['label']}", "kind": "separated",
            "status": report.status, "reason": report.reason, "elapsed_seconds": time.monotonic()-started,
            "speakers": hypotheses if report.status == "accepted" else {},
            "scores": score_speakers(reference, hypotheses) if report.status == "accepted" else None,
            "tracks": tracks, "metrics": report.metrics, "reference_counts": recovery.reference_counts})
    result = {"schema_version": 1, "recording_id": request.recording_id, "source_hash": study["source_hash"],
        "request": asdict(request), "variants": results, "separation_seconds": separation_seconds,
        "crop_start": crop_offset, "crop_end": right, "playback_gain": gain,
        "model_sha256": used_models, "separator_revision": policy.separator_revision,
        "separator_gain_mode": policy.separator_gain_mode,
        "separator_raw_peak": separated.raw_peak, "separator_applied_gain": separated.applied_gain,
        "separator_output_peak": max(float(np.max(np.abs(t))) for t in separated.tracks),
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recovery_policy": asdict(policy),
        "notice": "Human-reference WER is shown only for reviewed references and speaker-labelled outputs. Unassigned tracks are not scored. WER can exceed 100%. Separation time is shared; variant time covers recognition and matching, not queue time."}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
