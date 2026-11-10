"""Typed immutable experiment boundary, independent of model runtimes."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
from audio_benchmarks.exceptions import BenchmarkConfigurationError


@dataclass(frozen=True)
class SuiteConfig:
    root: Path
    document: dict[str, Any]

    @classmethod
    def load(cls, path: Path, root: Path) -> "SuiteConfig":
        """Validate all selected paths and bounds before any inference is launched."""
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc["schema_version"]!=1 or not doc["datasets"] or not doc["models"]:
                raise ValueError("Missing suite inputs")
            import re
            for value in [doc["run_id"],*(d["id"] for d in doc["datasets"]),*(m["id"] for m in doc["models"])]:
                if not re.fullmatch(r"[a-zA-Z0-9_-]+",value):
                    raise ValueError("Identifiers must be simple names")
            for group in (doc["datasets"],doc["models"]):
                if len({d["id"] for d in group})!=len(group):
                    raise ValueError("Duplicate identifiers")
            paths = [doc["output_directory"],doc["recognition_config"],doc["recovery_config"]]
            for dataset in doc["datasets"]:
                paths += [dataset["audio"],dataset["reference"]]
                if dataset.get("transcript"):
                    paths.append(dataset["transcript"])
            paths += [m["config"] for m in doc["models"]]
            if any(not (root/p).resolve().is_relative_to(root.resolve()) for p in paths):
                raise ValueError("All suite paths must stay in the workspace")
            if not (root/doc["output_directory"]).resolve().is_relative_to((root/"data").resolve()):
                raise ValueError("Generated benchmark outputs must stay under data")
            conditions={"clean","quiet","noise_15db","noise_5db","stereo_48k"}
            if not set(doc["conditions"])<=conditions or not all(m["provider"] in {"local_model","single_speaker"} for m in doc["models"]):
                raise ValueError("Unsupported condition or provider")
            values=[*doc["collars_seconds"],doc["quiet_gain"],doc["resample_rate"],doc["runtime_timeout_seconds"],
                *doc["noise_snr_db"].values(),doc["separation"]["clip_seconds"],doc["separation"]["guard_seconds"]]
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
                raise ValueError("Non-finite configuration")
            if (any(v<0 for v in doc["collars_seconds"]) or not 0<doc["quiet_gain"]<=1
                    or doc["runtime_timeout_seconds"]<=0 or doc["separation"]["clip_seconds"]<=0
                    or doc["separation"]["max_pairs"]<1 or doc["separation"]["guard_seconds"]<0):
                raise ValueError("Invalid experiment bounds")
            if not all(isinstance(v,bool) for v in doc["score_overlap_modes"]):
                raise ValueError("Overlap modes must be booleans")
            return cls(root.resolve(),doc)
        except (OSError,KeyError,TypeError,ValueError) as error:
            raise BenchmarkConfigurationError(f"Invalid benchmark suite: {error}") from error

    @property
    def output(self) -> Path:
        return self.root/self.document["output_directory"]
