"""Export only the explicitly shared podcast as a portable read-only Git demo."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recording_viewer.demo_config import PublicDemoConfig
from recording_viewer.exceptions import RecordingViewerError
from recording_viewer.services.demo_snapshot import public_snapshot, script_json, transcript_markdown


def digest(path: Path) -> str:
    """Hash the exact shared bytes rather than an upstream or decoded version."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path: Path) -> object:
    """Read a trusted artifact, translating I/O/JSON failures at the boundary."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RecordingViewerError(f"Cannot read required artifact: {path.name}") from error


def save_json(path: Path, value: object) -> None:
    """Write generated public artifacts, never a source recording or annotation."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")


def clock(seconds: float) -> str:
    """Format saved session times without changing their time base."""
    return f"{int(seconds//60)}:{seconds%60:05.2f}"


def readme(document: dict[str, object], report: dict[str, object]) -> str:
    """Keep report results immediately readable in GitHub without an HTML host."""
    lines = ["# 57-minute podcast: audio and DSP results", "",
        f"**{document['filename']}** · {clock(document['duration'])} · {document['word_count']:,} raw recognizer word tokens.", "",
        "## Open the results", "",
        "- [Download/play the complete MP3](audio.mp3?raw=1).",
        "- [Read the full speaker-labelled transcript directly in GitHub](TRANSCRIPT.md).",
        "- [Inspect the machine-readable results](results.json) or [complete presentation data](transcript.json).",
        "- Interactive player: download/clone this repository and open **`demos/podcast-57min/index.html`** in a browser. No Python, API key or speech-model installation is needed to view these saved results.",
        "- Open **`examples.html`** beside it for the three corrected overlap comparisons, with original mixture and both separated tracks.", "",
        "GitHub displays Markdown reports directly but does not execute this HTML viewer in its source-code view. A hosted interactive URL requires separate static hosting such as GitHub Pages; no hosting deployment is claimed here.", "",
        "## Full-recording snapshot", "",
        "This is the already processed full recording, not a new post-fix run. The interactive page starts with the original Nemotron timeline. The older recovery variant and Sherpa comparison remain selectable and explicitly unverified.", "",
        "| Saved variant | Transcript rows | Estimated voice groups |", "| --- | ---: | ---: |"]
    for variant in report["full_recording"]["variants"]:
        lines.append(f"| {variant['id']} | {variant['turns']} | {variant['estimated_voice_groups']} |")
    old = report["full_recording"]["historical_recovery"]
    lines += ["", f"Older full-recording recovery audit: **{old.get('accepted', 0)} accepted, {old.get('rejected', 0)} rejected, {old.get('skipped', 0)} skipped, {old.get('failed', 0)} failed**. Its budget was limited; this does not mean all overlap was separated. Accepted is a gate decision, not human-verified correctness.", "",
        "## Later gain-fix experiment — separate from the full recording", "",
        "Twelve reviewed detector-positive passages contained three true overlaps and nine negative controls. The three-second context diagnostic tested positives only.", "",
        "| Separator gain | Context | Confirmed overlaps passing gates | Negative controls incorrectly passing |",
        "| --- | ---: | ---: | ---: |"]
    for row in report["gain_comparison"]:
        negative = (f"{row['negative_controls_incorrectly_accepted']}/{row['negative_controls_tested']}"
                    if row["negative_controls_tested"] else "not tested")
        lines.append(f"| {row['gain_mode']} | {row['context_seconds']} s | {row['confirmed_overlaps_accepted']}/{row['confirmed_overlaps_tested']} | {negative} |")
    lines += ["", "## Corrected one-second-context examples", "",
        "These are saved model hypotheses. Compare each mixture with both tracks. Players include context; displayed text owns only the stated core interval. No word-error accuracy is available without checked word references.", ""]
    for example in report["examples"]:
        folder = f"examples/{example['id']}"
        lines += [f"### {clock(example['start'])}–{clock(example['end'])}: {example['status']}", "",
            f"Reason: `{example['reason']}`. Playback covers {clock(example['crop_start'])}–{clock(example['crop_end'])}.", "",
            f"[Original mixture]({folder}/mixture.wav?raw=1) · [Separated track 1]({folder}/track-1.wav?raw=1) · [Separated track 2]({folder}/track-2.wav?raw=1)", ""]
        for track in example["tracks"]:
            text = json.dumps(track["text"], ensure_ascii=False).replace("`", "'")
            lines.append(f"- Track {track['index']+1} ({track['assigned_speaker'] or 'unassigned'}): {text}")
        lines.append("")
    lines += ["## Limitations and attribution", "",
        "- Speakers are anonymous acoustic groups, not verified people. Counts may be fragmented or wrong; Sherpa's large count here is not a claim that the podcast has that many people.",
        "- Overlap detection has false positives; separation can leak another voice. A gate pass does not guarantee isolation or correct wording.",
        "- This small non-blinded, detector-positive sample does not establish full-recording accuracy or recall. The full 57-minute transcript was not manually verified.",
        "- This is a frozen, read-only demonstration. It cannot upload/process another recording or save reviews, and it does not contact a localhost backend.",
        "- Audio was supplied for this project. Ownership remains with the podcast's rights holders; the repository's source-code license does not grant rights to the podcast audio.", "",
        f"Original MP3 SHA-256: `{document['source_sha256']}`. The shared file is unchanged. `artifact-manifest.json` records hashes of all exported files. Private processing configurations, logs, local paths, caches and raw annotation stores are excluded.", ""]
    return "\n".join(lines)


def export_demo(root: Path, policy: PublicDemoConfig) -> Path:
    """Publish into a new allowlisted demo folder without mutating local results."""
    source = (root / policy.recording_directory).resolve()
    feedback = (root / policy.feedback_directory).resolve()
    output = (root / policy.output_directory).resolve()
    if (not source.is_relative_to((root / "data").resolve())
            or not feedback.is_relative_to((root / "data").resolve())
            or not output.is_relative_to((root / "demos").resolve())
            or output == (root / "demos").resolve()):
        raise RecordingViewerError("Demo inputs must stay in data and output in a named demos subdirectory")
    if output.exists():
        raise RecordingViewerError("Demo output already exists; preserve it and choose a fresh output folder")
    document = public_snapshot(read_json(source / "transcript.json"))
    if digest(source / "audio.mp3") != document["source_sha256"]:
        raise RecordingViewerError("Podcast audio does not match the stored source identity")
    snapshot = read_json(feedback / "snapshot.json")
    if snapshot["reviews"]["source_hash"] != document["source_sha256"]:
        raise RecordingViewerError("Listening experiment belongs to another recording")
    report = {"schema_version": 1, "filename": document["filename"], "source_sha256": document["source_sha256"],
        "full_recording": {"duration_seconds": document["duration"], "raw_word_tokens": document["word_count"],
            "recovery_state": "historical_before_shared_peak_gain_fix_not_rerun",
            "default_view": "nemo_native",
            "variants": [{"id": name, "turns": len(variant["turns"]),
                "estimated_voice_groups": len({row["speaker"] for row in variant["turns"] if row["speaker"] != "UNKNOWN"})}
                for name, variant in document["variants"].items()],
            "historical_recovery": {key: document["recovery"].get(key) for key in
                ("windows", "accepted", "rejected", "skipped", "failed")}},
        "gain_comparison": [{key: row[key] for key in ("gain_mode", "context_seconds", "confirmed_overlaps_tested",
            "confirmed_overlaps_accepted", "negative_controls_tested", "negative_controls_incorrectly_accepted", "failed_runs")}
            for row in read_json(feedback / "recovery-summary.json")], "examples": [],
        "limitations": "Gate acceptance is not word accuracy. Unknown full-recording recall. Cross-track leakage remains. No live processing."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="podcast-export-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        shutil.copyfile(source / "audio.mp3", stage / "audio.mp3")
        assets = root / "web/recording-viewer"
        for asset in ("viewer.css", "viewer.js", "upload.css", "upload.js", "demo-examples.js"):
            (stage / asset).write_text((assets / asset).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        for selected in policy.examples:
            job = root / "data/overlap-lab/runs" / selected.job_id
            result = read_json(job / "result.json")
            clip = result["request"]["reference"]["clip"]
            expected = snapshot["reviews"]["items"].get(selected.clip_id)
            if (result["source_hash"] != document["source_sha256"] or clip["id"] != selected.clip_id
                    or expected is None or any(clip[key] != expected["clip"][key] for key in ("id", "start", "end", "speakers"))
                    or result.get("separator_gain_mode") != "shared_peak"):
                raise RecordingViewerError("Example does not match the frozen source, core and corrected gain")
            variant = next(v for v in result["variants"] if v["id"] == "sepformer--base")
            target = stage / "examples" / selected.clip_id
            target.mkdir(parents=True)
            lengths = []
            for original, name in (("mixture.wav", "mixture.wav"), ("sepformer_track_0.wav", "track-1.wav"),
                                   ("sepformer_track_1.wav", "track-2.wav")):
                with wave.open(str(job / original), "rb") as audio:
                    if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
                        raise RecordingViewerError("Demo comparison must use canonical PCM16 clips")
                    lengths.append(audio.getnframes())
                shutil.copyfile(job / original, target / name)
            if len(set(lengths)) != 1:
                raise RecordingViewerError("Comparison players must cover identical sample ranges")
            report["examples"].append({"id": selected.clip_id, "source_job_id": selected.job_id,
                "start": clip["start"], "end": clip["end"], "crop_start": result["crop_start"], "crop_end": result["crop_end"],
                "context_seconds": result["request"]["context_seconds"], "status": variant["status"], "reason": variant["reason"],
                "gain_mode": "shared_peak", "tracks": [{key: track[key] for key in ("index", "text", "assigned_speaker")}
                                                      for track in variant["tracks"]]})
        save_json(stage / "transcript.json", document)
        save_json(stage / "results.json", report)
        (stage / "TRANSCRIPT.md").write_text(transcript_markdown(document), encoding="utf-8", newline="\n")
        (stage / "README.md").write_text(readme(document, report), encoding="utf-8", newline="\n")
        for template, target, payload in (("index.html", "index.html", document),
                                           ("demo-examples.html", "examples.html", report)):
            html = (assets / template).read_text(encoding="utf-8").replace("__RECORDING_DATA__", script_json(payload))
            (stage / target).write_text(html, encoding="utf-8", newline="\n")
        manifest = {str(path.relative_to(stage)).replace("\\", "/"): {"sha256": digest(path), "bytes": path.stat().st_size}
                    for path in sorted(stage.rglob("*")) if path.is_file()}
        save_json(stage / "artifact-manifest.json", manifest)
        stage.rename(output)
    return output


def main() -> None:
    """Require an explicit recording policy; no automatic broad data publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/podcast_demo_export.json")
    args = parser.parse_args()
    print(export_demo(ROOT, PublicDemoConfig.load(args.config)))


if __name__ == "__main__":
    main()
