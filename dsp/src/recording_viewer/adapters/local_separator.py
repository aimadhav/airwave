"""Timeout-isolated separator; trusted paths only, private temporary artifacts."""

import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from recording_viewer.exceptions import RecoveryError
from recording_viewer.models.recovery import SeparationOutput
from recording_viewer.recovery_config import RecoveryConfig
from recording_viewer.services.source_gain import shared_peak_gain


class LocalSpeechSeparator:
    """Run heavy optional dependencies outside the working DSP/STT runtime."""

    def __init__(self, root: Path, config: RecoveryConfig, scratch: Path) -> None:
        self.root, self.config, self.scratch = root.resolve(), config, scratch
        if not (root / config.separator_python).is_file() or not (root / config.separator_directory / "manifest.json").is_file():
            raise RecoveryError("Separator assets missing; run scripts/setup_overlap_separator.py")
        scratch.mkdir(parents=True, exist_ok=True)

    def separate(self, samples: np.ndarray, sample_rate: int) -> SeparationOutput:
        """A worker timeout/nonzero exit cannot corrupt the baseline transcript."""
        try:
            with tempfile.TemporaryDirectory(prefix="separate-", dir=self.scratch) as temporary:
                folder = Path(temporary)
                np.save(folder / "input.npy", samples, allow_pickle=False)
                environment = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
                # Do not inject the DSP runtime's binary wheels into PyTorch's runtime.
                environment.pop("PYTHONPATH", None)
                result = subprocess.run([str(self.root / self.config.separator_python),
                    str(self.root / "scripts/separate_overlap_worker.py"), "--input", str(folder / "input.npy"),
                    "--output", str(folder / "output.npz"), "--sample-rate", str(sample_rate),
                    "--model", str(self.root / self.config.separator_directory),
                    "--model-rate", str(self.config.separator_sample_rate), "--threads", str(self.config.separator_threads)],
                    cwd=self.root, env=environment, check=True, capture_output=True, text=True,
                    timeout=self.config.separator_timeout_seconds)
                with np.load(folder / "output.npz", allow_pickle=False) as output:
                    tracks = tuple(output[key].copy() for key in ("track_0", "track_1"))
                if self.config.separator_gain_mode == "shared_peak":
                    tracks, gain, peak = shared_peak_gain(tracks, self.config.separator_peak_limit)
                    return SeparationOutput(tracks, sample_rate, gain, peak)
                return SeparationOutput(tracks, sample_rate, 1.0,
                                        max(float(np.max(np.abs(t))) for t in tracks))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            detail = error.stderr[-1500:] if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
            raise RecoveryError(f"Local separator failed: {detail}") from error
