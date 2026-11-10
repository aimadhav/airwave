"""Explicit opt-in download of a pinned public model; never run by uploads."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recording_viewer.recovery_config import RecoveryConfig


def main() -> None:
    """Save weights and hashes locally without importing remote Python code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/overlap_recovery.json")
    args = parser.parse_args()
    config = RecoveryConfig.load(args.config)
    folder = (ROOT / config.separator_directory).resolve()
    if not folder.is_relative_to((ROOT / "models").resolve()):
        raise ValueError("Separator weights must stay inside the project models directory")
    folder.mkdir(parents=True, exist_ok=True)
    files = ("hyperparams.yaml", "encoder.ckpt", "masknet.ckpt", "decoder.ckpt")
    hashes: dict[str, str] = {}
    for filename in files:
        path = folder / filename
        if not path.is_file():
            url = f"https://huggingface.co/{config.separator_repository}/resolve/{config.separator_revision}/{filename}"
            print(f"Downloading pinned separator asset: {filename}", flush=True)
            partial = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=config.separator_timeout_seconds) as response, partial.open("wb") as target:
                while block := response.read(1024 * 1024):
                    target.write(block)
            partial.replace(path)
        with path.open("rb") as source:
            hashes[filename] = hashlib.file_digest(source, "sha256").hexdigest()
    manifest = {"repository": config.separator_repository, "revision": config.separator_revision,
                "sample_rate": config.separator_sample_rate, "sha256": hashes}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Separator ready: {folder}", flush=True)


if __name__ == "__main__":
    main()
