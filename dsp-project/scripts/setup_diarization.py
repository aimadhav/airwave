"""Download explicitly requested, checksum-pinned model/sample assets.

Never runs downloaded code. Archives extract only allowlisted regular files;
existing valid files are reused. Models and recordings stay git-ignored.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
import stat
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_target(relative: str) -> Path:
    target = (ROOT / relative).resolve()
    if not any(target.is_relative_to(ROOT / folder) for folder in ("models", "data/benchmarks")):
        raise ValueError(f"Unsafe asset destination: {relative}")
    return target


def prepare_asset(asset: dict[str, Any], timeout: float) -> None:
    target = checked_target(asset["path"])
    if target.exists():
        if sha256(target) != asset["sha256"]:
            raise ValueError(f"Checksum mismatch in existing {target}; move it aside before retrying")
        print(f"Verified {target.relative_to(ROOT)}")
    else:
        if not asset["url"].startswith("https://"):
            raise ValueError("Assets must use HTTPS")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            downloaded = Path(temporary) / "download"
            print(f"Downloading {target.relative_to(ROOT)}", flush=True)
            with urllib.request.urlopen(asset["url"], timeout=timeout) as response, downloaded.open("wb") as output:
                shutil.copyfileobj(response, output)
            if sha256(downloaded) != asset["sha256"]:
                raise ValueError(f"Downloaded checksum mismatch: {target.name}")
            downloaded.replace(target)
    if asset.get("extract_members"):
        with tarfile.open(target, "r:bz2") as archive:
            for member_name in asset["extract_members"]:
                destination = checked_target(str((target.parent / member_name).relative_to(ROOT)))
                member = archive.getmember(member_name)
                if not member.isfile():
                    raise ValueError("Only regular model/license files may be extracted")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("Missing archive member")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source:
                    if destination.exists():
                        if hashlib.sha256(source.read()).hexdigest() != sha256(destination):
                            raise ValueError(f"Existing extracted asset differs: {destination}")
                    else:
                        with destination.open("wb") as output:
                            shutil.copyfileobj(source, output)
    if asset.get("extract_zip_members"):
        with zipfile.ZipFile(target) as archive:
            for member_name in asset["extract_zip_members"]:
                destination = checked_target(str((target.parent / member_name).relative_to(ROOT)))
                member = archive.getinfo(member_name)
                if member.is_dir() or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("Only allowlisted regular runtime/license files may be extracted")
                with archive.open(member) as source:
                    if destination.exists():
                        if hashlib.sha256(source.read()).hexdigest() != sha256(destination):
                            raise ValueError(f"Existing extracted runtime differs: {destination}")
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("wb") as output:
                            shutil.copyfileobj(source, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-stt", action="store_true")
    parser.add_argument("--samples", action="store_true")
    parser.add_argument("--native", action="store_true", help="Download pinned optional Windows CPU runtime and Nemotron weights; never executes them")
    parser.add_argument("--embedding-candidates", action="store_true",
                        help="Download optional TitaNet-Large, WeSpeaker and CAM++ CPU embeddings")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/diarization_assets.json",
                        help="Checksum-pinned asset manifest")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    groups = {"diarization"} | ({"stt"} if args.with_stt else set()) | ({"samples"} if args.samples else set())
    if args.embedding_candidates:
        groups.add("embedding_candidates")
    if args.native:
        groups.add("native")
    for asset in manifest["assets"]:
        if asset["group"] in groups:
            prepare_asset(asset, float(manifest["timeout_seconds"]))
    print("Ready. No inference uploads audio to an external service.")


if __name__ == "__main__":
    main()
