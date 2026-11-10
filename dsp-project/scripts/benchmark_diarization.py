"""Reproducible real-audio benchmark; no fabricated accuracy without reference labels."""

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if (ROOT / ".runtime-deps").is_dir():
    sys.path.insert(0, str(ROOT / ".runtime-deps"))

import yaml
from dsp.pipeline import DSPConfig, DSPPipeline
from dsp.evaluation import TimedSpeaker, diarization_error, read_rttm


def file_fingerprint(path: Path) -> dict[str, object]:
    """Identify exact input/model bytes without relying on a mutable filename."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load_benchmark_config(
    path: Path, provider: str, speakers: int, threshold: float | None,
    embedding_model: Path | None,
) -> DSPConfig:
    """Compose experiments from external configuration without changing defaults."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))["dsp"]
    config = DSPConfig.from_mapping(values)
    local = replace(
        config.local_model, num_speakers=speakers,
        cluster_threshold=threshold if threshold is not None else config.local_model.cluster_threshold,
        embedding_model=str(embedding_model) if embedding_model is not None else config.local_model.embedding_model,
    )
    return replace(config, random_audio_provider=provider, local_model=local)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--provider", choices=["local_model", "single_speaker"], default="local_model")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--speakers", type=int, default=-1)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ai_layer_config.yaml")
    parser.add_argument("--embedding-model", type=Path, help="Optional CPU embedding model override")
    parser.add_argument("--tag", help="Unique experiment name; previous reports are never overwritten")
    parser.add_argument("--output", type=Path, default=Path("data/benchmark_results"))
    args = parser.parse_args()
    config = load_benchmark_config(args.config, args.provider, args.speakers, args.threshold, args.embedding_model)
    local = config.local_model
    session = args.tag or f"{args.audio.stem}_{args.provider}_{local.cluster_threshold}"
    if not session or session in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- ." for c in session):
        parser.error("Experiment tag must be a simple directory name")
    if session.endswith((".", " ")):
        parser.error("Experiment tag cannot end with a dot or space")
    folder = args.output / session
    if folder.exists():
        parser.error(f"Refusing to overwrite existing experiment: {folder}")
    started = perf_counter()
    result = DSPPipeline(config).process_audio_file(args.audio, args.output, session_id=session)
    elapsed = perf_counter() - started
    segments = result.payload.segments
    errors = []
    for segment in segments:
        with wave.open(segment.clip_path, "rb") as clip:
            actual = clip.getnframes() / clip.getframerate()
            if clip.getframerate() != 16000 or clip.getnchannels() != 1 or abs(actual - (segment.end_time - segment.start_time)) > 2 / 16000:
                errors.append(segment.segment_id)
    report = {
        "input": str(args.audio), "provider": args.provider, "local_config": asdict(local),
        "duration_seconds": result.payload.duration_seconds, "elapsed_seconds": elapsed,
        "real_time_factor": elapsed / result.payload.duration_seconds,
        "speakers": sorted({s.preliminary_speaker_id for s in segments}),
        "segments": [asdict(s) for s in segments], "clip_validation_errors": errors,
        "warnings": result.warnings,
        "fingerprints": {"input": file_fingerprint(args.audio)},
        "experiment_tag": session,
    }
    if args.provider == "local_model":
        report["backend"] = local.backend
        if local.backend == "nemo_native":
            report["fingerprints"].update({
                "native_executable": file_fingerprint(Path(local.native.executable)),
                "native_model": file_fingerprint(Path(local.native.model)),
            })
        else:
            report["fingerprints"].update({
                "segmentation_model": file_fingerprint(Path(local.segmentation_model)),
                "embedding_model": file_fingerprint(Path(local.embedding_model)),
            })
            report["runtime_version"] = importlib.metadata.version("sherpa-onnx")
    if args.reference:
        reference = read_rttm(args.reference)
        prediction = [TimedSpeaker(s.start_time, s.end_time, s.preliminary_speaker_id) for s in segments]
        report["evaluation"] = {str(collar): diarization_error(reference, prediction, collar) for collar in [0.0, 0.25]}
        report["fingerprints"]["reference"] = file_fingerprint(args.reference)
        report["reference_speakers"] = sorted({row.speaker for row in reference})
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (folder / "prediction.rttm").write_text("\n".join(
        f"SPEAKER {args.audio.stem} 1 {s.start_time:.3f} {s.end_time-s.start_time:.3f} <NA> <NA> {s.preliminary_speaker_id} <NA> <NA>"
        for s in segments), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"segments", "local_config", "warnings"}}, indent=2))


if __name__ == "__main__":
    main()
