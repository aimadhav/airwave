"""Process a private recording and export a local, overlap-aware listening page."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "src", ROOT / "models/asr-deps", ROOT / "models/media-deps", ROOT / ".runtime-deps"):
    if dependency.is_dir():
        sys.path.insert(0, str(dependency))

import numpy as np
import yaml
from dsp.pipeline import DSPConfig, DSPPipeline
from recording_viewer.adapters.faster_whisper import FasterWhisperTranscriber
from recording_viewer.services.alignment import align_words, group_turns
from recording_viewer.config import RecordingViewerConfig
from recording_viewer.models import TranscriptWord, VoiceRegion
from recording_viewer.adapters.local_separator import LocalSpeechSeparator
from recording_viewer.adapters.voice_embedder import OnnxVoiceEmbedder
from recording_viewer.exceptions import RecoveryError
from recording_viewer.models.recovery import RecoveryResult, WindowReport
from recording_viewer.recovery_config import RecoveryConfig
from recording_viewer.services.overlap_recovery import OverlapRecovery
from recording_viewer.services.recovery_timeline import plan_windows, splice_recovery


def fingerprint(path: Path) -> str:
    """Fingerprint local inputs and weights without exposing their contents."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, document: object) -> None:
    """Write generated processing artifacts, not editable application source."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def signature(*values: object) -> str:
    """Invalidate expensive caches when source, policy or model selection changes."""
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def main() -> None:
    """Resume expensive stages locally, checking source identity on every run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/recording_viewer.json")
    parser.add_argument("--diarize-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-recovery", action="store_true", help="Baseline comparison; do not attempt voice separation")
    args = parser.parse_args()
    config = RecordingViewerConfig.load(args.config)
    source = Path(config.source)
    folder = (ROOT / config.output_directory).resolve()
    if not folder.is_relative_to((ROOT / "data/listening-room").resolve()):
        raise ValueError("Private viewer output must stay within data/listening-room")
    folder.mkdir(parents=True, exist_ok=True)
    cache_path = folder / "stage-cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    source_hash = fingerprint(source)
    identity = folder / "source.json"
    if identity.exists() and json.loads(identity.read_text(encoding="utf-8"))["sha256"] != source_hash:
        raise ValueError("Existing output belongs to a different recording")
    write_json(identity, {"filename": source.name, "sha256": source_hash})
    original = folder / ("audio" + source.suffix.lower())
    if not original.exists():
        shutil.copyfile(source, original)
    canonical = folder / "processing.wav"
    if not canonical.exists():
        import imageio_ffmpeg
        limit = ["-t", str(config.max_duration_seconds + 1)] if config.max_duration_seconds is not None else []
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-n", "-i", str(source), *limit,
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(canonical)],
                       check=True, capture_output=True, timeout=config.decode_timeout_seconds)
    with wave.open(str(canonical), "rb") as recording:
        duration = recording.getnframes() / recording.getframerate()
        if config.max_duration_seconds is not None and duration > config.max_duration_seconds:
            raise ValueError("This recording exceeds the configured duration limit")
        sample_rate = recording.getframerate()
        samples = np.frombuffer(recording.readframes(recording.getnframes()), dtype="<i2").astype(np.float32) / 32768
    timeline_path = folder / "speaker-timelines.json"
    dsp_policy_path = ROOT / (config.dsp_config_path or "configs/ai_layer_config.yaml")
    base = DSPConfig.from_mapping(yaml.safe_load(dsp_policy_path.read_text(encoding="utf-8"))["dsp"])
    timeline_assets = [base.local_model.segmentation_model, base.local_model.embedding_model,
                       base.local_model.native.model, base.local_model.native.executable]
    timeline_signature = signature(source_hash, asdict(base),
        {name: fingerprint(ROOT / name) if (ROOT / name).is_file() else None for name in timeline_assets}, "timeline-v2")
    if not timeline_path.exists() or cache.get("timeline") != timeline_signature:
        if args.build_only:
            raise ValueError("Timeline cache is missing or stale; run without --build-only")
        timelines: dict[str, object] = {}
        for backend in ("sherpa", "nemo_native"):
            dsp_config = replace(base, random_audio_provider="local_model", local_model=replace(base.local_model, backend=backend, num_speakers=-1))
            result = DSPPipeline(dsp_config).process_audio_file(canonical, folder / "diagnostics", f"{config.session_id}_{backend}")
            regions = [VoiceRegion(s.segment_id, s.preliminary_speaker_id, s.start_time, s.end_time) for s in result.payload.segments]
            timelines[backend] = [asdict(region) for region in regions]
            print(f"{backend}: {len(regions)} turns, {len({r.speaker for r in regions})} estimated voice groups", flush=True)
        write_json(timeline_path, timelines)
        cache["timeline"] = timeline_signature
        write_json(cache_path, cache)
    if args.diarize_only:
        return
    words_path = folder / "recognized-words.json"
    model_hash = fingerprint(ROOT / config.model_directory / "model.bin")
    word_signature = signature(source_hash, model_hash, config.model_revision, config.language, config.beam_size,
        config.vad_filter, config.condition_on_previous_text, config.maximum_no_speech_probability,
        config.minimum_segment_log_probability, config.maximum_compression_ratio, "word-probabilities-v3")
    recognizer = FasterWhisperTranscriber(config)
    if not words_path.exists() or cache.get("recognition") != word_signature:
        if args.build_only:
            raise ValueError("Recognition must finish before the page can be built")
        words = recognizer.transcribe(canonical)
        write_json(words_path, [asdict(word) for word in words])
        cache["recognition"] = word_signature
        write_json(cache_path, cache)
    words = [TranscriptWord(**row) for row in json.loads(words_path.read_text(encoding="utf-8"))]
    timelines = json.loads(timeline_path.read_text(encoding="utf-8"))
    variants: dict[str, object] = {}
    for backend, rows in timelines.items():
        regions = [VoiceRegion(**row) for row in rows]
        turns = align_words(words, regions, config.minimum_word_coverage)
        variants[backend] = {"turns": [asdict(turn) for turn in turns],
                             "groups": [asdict(group) for group in group_turns(turns)]}
    recovery_summary: dict[str, object] = {"enabled": False, "accepted": 0}
    if config.recovery_config and not args.skip_recovery:
        policy = RecoveryConfig.load(ROOT / config.recovery_config)
        regions = [VoiceRegion(**row) for row in timelines[policy.backend]]
        recovery_path = folder / "overlap-recovery.json"
        manifest = ROOT / policy.separator_directory / "manifest.json"
        # Provider and domain code changes invalidate recovery, too.
        recovery_code = [fingerprint(path) for path in sorted((ROOT / "src/recording_viewer").rglob("*.py"))]
        recovery_signature = signature(word_signature, fingerprint(timeline_path), asdict(policy), recovery_code,
                                       fingerprint(manifest) if manifest.exists() else None,
                                       fingerprint(ROOT / policy.embedding_model) if (ROOT / policy.embedding_model).is_file() else None,
                                       fingerprint(ROOT / "scripts/separate_overlap_worker.py"))
        if recovery_path.exists() and cache.get("recovery") == recovery_signature:
            recovery_document = json.loads(recovery_path.read_text(encoding="utf-8"))
        else:
            if args.build_only:
                raise ValueError("Recovery cache is missing or stale; run without --build-only")
            windows = plan_windows(regions, duration, policy)
            try:
                if not policy.enabled or not windows:
                    recovery = RecoveryResult(tuple(WindowReport(window, "skipped", "disabled", {}) for window in windows), {})
                else:
                    recovery = OverlapRecovery(policy, LocalSpeechSeparator(ROOT, policy, folder / "separation-scratch"),
                                               OnnxVoiceEmbedder(policy), recognizer).recover(samples, sample_rate, regions)
            except RecoveryError as error:
                recovery = RecoveryResult(tuple(WindowReport(window, "failed", str(error), {})
                    for window in windows), {}, (str(error),))
            turns = splice_recovery(words, regions, recovery, config.minimum_word_coverage)
            recovery_document = {"policy": asdict(policy), "result": asdict(recovery),
                "turns": [asdict(turn) for turn in turns], "groups": [asdict(group) for group in group_turns(turns)]}
            write_json(recovery_path, recovery_document)
            cache["recovery"] = recovery_signature
            write_json(cache_path, cache)
        reports = recovery_document["result"]["reports"]
        statuses = {state: sum(row["status"] == state for row in reports) for state in ("accepted", "rejected", "skipped", "failed")}
        recovery_summary = {"enabled": policy.enabled, "backend": policy.backend, "windows": len(reports), **statuses,
                            "decisions": [{"start": r["window"]["start"], "end": r["window"]["end"],
                                           "status": r["status"],
                                           "reason": "local_recovery_provider_failed" if r["status"] == "failed" else r["reason"]}
                                          for r in reports]}
        variants["overlap_recovery"] = {"turns": recovery_document["turns"], "groups": recovery_document["groups"]}
        print(f"Overlap recovery: {statuses}", flush=True)
    bins = np.array_split(samples, config.waveform_bins)
    envelope = np.array([float(np.sqrt(np.mean(block ** 2))) if block.size else 0 for block in bins])
    maximum = float(np.max(envelope, initial=0))
    document = {"schema_version": 2, "filename": source.name, "source_sha256": source_hash,
                "duration": duration, "sample_rate": sample_rate, "audio": original.name,
                "recognizer": config.model_repository, "model_revision": config.model_revision,
                "word_count": len(words), "waveform": (envelope / maximum).tolist() if maximum else envelope.tolist(),
                "variants": variants,
                "default_backend": "overlap_recovery" if recovery_summary.get("accepted") else config.default_backend,
                "recovery": recovery_summary,
                "notice": "Estimated speaker labels, not verified identities. Word timing and attribution can be wrong. Overlap recovery is experimental: only quality-gated separated tracks replace baseline words. Other overlap remains unresolved or uncertain; no missing words are invented."}
    upload_policy = json.loads((ROOT / "configs/recording_upload.json").read_text(encoding="utf-8"))
    document["local_app_url"] = upload_policy["local_app_url"]
    write_json(folder / "transcript.json", document)
    template = ROOT / "web/recording-viewer"
    encoded = json.dumps(document, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = (template / "index.html").read_text(encoding="utf-8").replace("__RECORDING_DATA__", encoded)
    (folder / "index.html").write_text(html, encoding="utf-8")
    for asset in ("viewer.css", "viewer.js", "upload.css", "upload.js", "README.txt"):
        shutil.copyfile(template / asset, folder / asset)
    write_json(folder / "provenance.json", {"source_sha256": source_hash, "configuration": asdict(config),
        "model_sha256": model_hash, "stage_signatures": cache, "overlap_recovery": recovery_summary,
        "words_recognized": len(words), "duration_seconds": duration,
        "limitations": document["notice"]})
    with zipfile.ZipFile(folder / "listening-room.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for asset in ("index.html", "viewer.css", "viewer.js", "upload.css", "upload.js", original.name, "transcript.json", "README.txt"):
            archive.write(folder / asset, arcname=asset)
    print(f"Page ready: {folder / 'index.html'}; recognized {len(words)} words across {duration:.2f}s", flush=True)


if __name__ == "__main__":
    main()
