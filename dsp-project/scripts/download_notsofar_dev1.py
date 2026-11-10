"""Download the official NOTSOFAR-1 dev-set-1 release for a baseline-aligned run."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from huggingface_hub import HfApi, get_token, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "microsoft/NOTSOFAR"
SPLIT = "benchmark-datasets/dev_set/240825.1_dev1"
OUTPUT = ROOT / "data/benchmarks/notsofar1-dev1-full-20261109/source"
MINIMUM_FREE_BYTES = 20 * 1024**3


def download() -> dict[str, Any]:
    """Fetch only dev-set-1, reusing the user's cached HF credential securely."""
    token = get_token()
    if not token:
        raise RuntimeError("No cached Hugging Face read credential is available")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(OUTPUT.parent).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(f"Need at least 20 GiB free for this split; currently {free_bytes / 1024**3:.1f} GiB")

    info = HfApi().dataset_info(DATASET_ID, revision="main", token=token)
    patterns = [f"{SPLIT}/**"]
    local_path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=info.sha,
        allow_patterns=patterns,
        local_dir=OUTPUT,
        token=token,
        max_workers=4,
        tqdm_class=None,
    )
    files = [p for p in OUTPUT.rglob("*") if p.is_file() and ".cache" not in p.parts]
    total_bytes = sum(p.stat().st_size for p in files)
    meetings = sorted(p.name for p in (OUTPUT / SPLIT / "MTG").glob("MTG_*") if p.is_dir())
    if not meetings:
        raise RuntimeError("Download completed without meeting directories")
    manifest = {
        "dataset_id": DATASET_ID,
        "revision": info.sha,
        "split": SPLIT,
        "source_url": "https://huggingface.co/datasets/microsoft/NOTSOFAR/tree/main/benchmark-datasets/dev_set/240825.1_dev1",
        "local_path": str(Path(local_path).resolve()),
        "meeting_count": len(meetings),
        "file_count": len(files),
        "downloaded_bytes": total_bytes,
        "downloaded_gib": round(total_bytes / 1024**3, 3),
    }
    manifest_path = ROOT / "data/benchmarks/notsofar1-dev1-full-20261109/download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    print(f"Download complete: {len(meetings)} meetings, {len(files)} files, {total_bytes / 1024**3:.2f} GiB", flush=True)
    print(f"Dataset pinned to revision {info.sha}", flush=True)
    return manifest


if __name__ == "__main__":
    download()
