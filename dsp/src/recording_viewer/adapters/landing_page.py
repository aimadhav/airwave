"""Initialize an upload-only page without recordings, models or fake transcripts."""

import json
from pathlib import Path
import shutil

from recording_viewer.exceptions import RecordingViewerError


def create_upload_landing(directory: Path, assets: Path, local_app_url: str) -> None:
    """Create missing landing assets, never replace an existing recording page."""
    if (directory / "index.html").exists():
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"local_app_url": local_app_url}).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        html = (assets / "welcome.html").read_text(encoding="utf-8").replace("__RECORDING_DATA__", payload)
        for asset in ("viewer.css", "upload.css", "upload.js"):
            if not (directory / asset).exists():
                shutil.copyfile(assets / asset, directory / asset)
        with (directory / "index.html").open("x", encoding="utf-8") as output:
            output.write(html)
    except OSError as error:
        raise RecordingViewerError("Could not initialize the local upload page; existing recordings were not replaced") from error
