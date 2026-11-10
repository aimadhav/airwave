"""Prepare a hosted read-only demo without depending on private GitHub assets."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for dependency in ("src", "models/media-deps", ".runtime-deps"):
    sys.path.insert(0, str(ROOT / dependency))

from recording_viewer.exceptions import RecordingViewerError
from recording_viewer.services.demo_snapshot import script_json


@dataclass(frozen=True)
class PodcastSiteConfig:
    demo_directory: str
    site_directory: str
    audio_filename: str
    audio_sample_rate: int
    audio_channels: int
    audio_bit_rate: str
    max_asset_bytes: int
    encode_timeout_seconds: float
    playback_load_timeout_seconds: float

    def __post_init__(self) -> None:
        if self.audio_filename != Path(self.audio_filename).name or not self.audio_filename.endswith(".m4a"):
            raise RecordingViewerError("Configure a simple M4A playback filename")
        for value in (self.audio_sample_rate, self.audio_channels, self.max_asset_bytes):
            if type(value) is not int or value <= 0:
                raise RecordingViewerError("Invalid web-audio size/rate policy")
        if (not isinstance(self.audio_bit_rate, str) or not self.audio_bit_rate[:-1].isdigit()
                or not self.audio_bit_rate.endswith("k") or int(self.audio_bit_rate[:-1]) <= 0
                or isinstance(self.encode_timeout_seconds, bool) or not isinstance(self.encode_timeout_seconds, (int, float))
                or not math.isfinite(self.encode_timeout_seconds) or self.encode_timeout_seconds <= 0
                or isinstance(self.playback_load_timeout_seconds, bool)
                or not isinstance(self.playback_load_timeout_seconds, (int, float))
                or not math.isfinite(self.playback_load_timeout_seconds) or self.playback_load_timeout_seconds <= 0):
            raise RecordingViewerError("Invalid web-audio encoding policy")

    @classmethod
    def load(cls, path: Path) -> "PodcastSiteConfig":
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, TypeError, ValueError) as error:
            raise RecordingViewerError("Invalid hosted podcast configuration") from error


def save_json(path: Path, document: object) -> None:
    """Write generated website data using stable UTF-8 and LF bytes."""
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8", newline="\n")


def prepare(root: Path, config: PodcastSiteConfig) -> Path:
    """Copy only sanitized demo artifacts and encode a separate playback asset."""
    source, site = (root / config.demo_directory).resolve(), (root / config.site_directory).resolve()
    if not source.is_relative_to((root / "demos").resolve()) or not site.is_relative_to((root / "data/sites").resolve()):
        raise RecordingViewerError("Site input must be an exported demo; checkout must stay under data/sites")
    hosting = json.loads((site / ".openai/hosting.json").read_text(encoding="utf-8"))
    if not hosting.get("project_id") or hosting.get("static", {}).get("directory") != "dist":
        raise RecordingViewerError("Persist the registered Site identity and static configuration first")
    document = json.loads((source / "transcript.json").read_text(encoding="utf-8"))
    report = json.loads((source / "results.json").read_text(encoding="utf-8"))
    audit_path = root / "web/recording-viewer/pipeline-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["source_sha256"] != document["source_sha256"]:
        raise RecordingViewerError("Pipeline audit belongs to a different recording")
    target = site / "dist"
    target.mkdir(exist_ok=True)
    for name in ("viewer.css", "viewer.js", "upload.css", "upload.js", "demo-examples.js", "TRANSCRIPT.md"):
        shutil.copyfile(source / name, target / name)
    for name in ("viewer.js", "buffered-playback.js"):
        shutil.copyfile(root / "web/recording-viewer" / name, target / name)
    for name in ("pipeline.js", "pipeline.css", "pipeline-audit.json"):
        shutil.copyfile(root / "web/recording-viewer" / name, target / name)
    audit_html = (root / "web/recording-viewer/pipeline.html").read_text(encoding="utf-8")
    (target / "pipeline.html").write_text(audit_html.replace("__PIPELINE_AUDIT__", script_json(audit)), encoding="utf-8", newline="\n")
    shutil.copytree(source / "examples", target / "examples", dirs_exist_ok=True)
    playback = target / config.audio_filename
    if not playback.exists():
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-n", "-i", str(source / "audio.mp3"),
            "-vn", "-map_metadata", "-1", "-ac", str(config.audio_channels), "-ar", str(config.audio_sample_rate),
            "-c:a", "aac", "-b:a", config.audio_bit_rate, "-movflags", "+faststart", str(playback)],
            capture_output=True, check=True, timeout=config.encode_timeout_seconds, shell=False)
    if playback.stat().st_size >= config.max_asset_bytes:
        raise RecordingViewerError("Web playback asset exceeds the configured static-file budget")
    with playback.open("rb") as audio:
        playback_hash = hashlib.file_digest(audio, "sha256").hexdigest()
    document["audio"] = config.audio_filename
    document["web_playback"] = {"codec": "AAC", "bit_rate": config.audio_bit_rate,
        "mime_type": "audio/mp4", "buffer_before_seek": True,
        "load_timeout_seconds": config.playback_load_timeout_seconds,
        "channels": config.audio_channels, "sample_rate": config.audio_sample_rate,
        "sha256": playback_hash, "original_sha256": document["source_sha256"]}
    document["notice"] += " Website playback is a separate compressed AAC copy; the original MP3 and saved transcripts were not changed."
    report["web_playback"] = document["web_playback"]
    save_json(target / "transcript.json", document)
    save_json(target / "results.json", report)
    for template, destination, payload in (("index.html", "index.html", document),
                                           ("demo-examples.html", "examples.html", report)):
        html = (root / "web/recording-viewer" / template).read_text(encoding="utf-8")
        html = html.replace("__RECORDING_DATA__", script_json(payload)).replace('href="audio.mp3"', f'href="{config.audio_filename}"')
        html = html.replace("Download full MP3", "Download web audio copy")
        if destination == "examples.html":
            html = html.replace('<div class="report-links">',
                '<div class="report-links"><a class="button" href="pipeline.html">How it works &amp; debug →</a>')
        if destination == "index.html":
            html = html.replace('<div class="masthead-actions">',
                '<div class="masthead-actions"><a class="button" href="pipeline.html">How it works &amp; debug</a>')
            html = html.replace('</head>', '<style>.masthead{height:auto;min-height:83px;gap:16px;padding-top:16px;padding-bottom:16px;flex-wrap:wrap}.masthead-actions{gap:12px;flex-wrap:wrap}</style></head>')
        html = html.replace("<strong>Original recording</strong>", "<strong>Web playback copy</strong>")
        html = html.replace('<script src="viewer.js"></script>',
                            '<script src="buffered-playback.js"></script>\n  <script src="viewer.js"></script>')
        html = html.replace("Click any timestamp to check the original.",
                            "Click any timestamp to check the web audio copy. The first play downloads the 25 MB audio once, then all timestamp jumps work locally.")
        (target / destination).write_text(html, encoding="utf-8", newline="\n")
    original_markdown = (target / "TRANSCRIPT.md").read_text(encoding="utf-8")
    (target / "TRANSCRIPT.md").write_text(original_markdown.replace("audio.mp3?raw=1", config.audio_filename)
        .replace("[Results](README.md)", "[Results](examples.html)"), encoding="utf-8", newline="\n")
    (site / "README.md").write_text("# Public podcast DSP results\n\nStatic, read-only transcript and saved overlap comparisons.\n"
        "Web audio is a separate AAC playback copy; source MP3 and pipeline results remain unchanged.\n"
        "No uploads, localhost dependencies, credentials or model inference are included.\n", encoding="utf-8", newline="\n")
    return target


def main() -> None:
    """Prepare the selected, already registered Site; never create another Site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/podcast_site.json")
    args = parser.parse_args()
    print(prepare(ROOT, PodcastSiteConfig.load(args.config)), flush=True)


if __name__ == "__main__":
    main()
