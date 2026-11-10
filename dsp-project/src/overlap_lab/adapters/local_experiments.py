"""Read-only capability discovery and isolated local inference composition."""

from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess
from overlap_lab.config import LabConfig
from overlap_lab.exceptions import LabError
from overlap_lab.models import ExperimentRequest, LabJob
from overlap_lab.interfaces import StudyRepository


class LocalExperimentProcessor:
    def __init__(self, root: Path, current_folder: Path, uploads: Path, repository: StudyRepository, config: LabConfig) -> None:
        self.root, self.current_folder, self.uploads, self.repository, self.config = root, current_folder, uploads, repository, config

    def capabilities(self) -> list[dict[str, object]]:
        """Report availability without importing models or downloading anything."""
        policy = json.loads((self.root / self.config.recovery_config).read_text(encoding="utf-8"))
        runtime = (self.root / self.config.python_executable).is_file()
        separator_ready = ((self.root / policy["separator_python"]).is_file()
                           and (self.root / policy["separator_directory"] / "manifest.json").is_file()
                           and (self.root / policy["embedding_model"]).is_file())
        variants = []
        for separator in self.config.separators:
            for model in self.config.models:
                weights = all((self.root / model.directory / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json"))
                reason = ("Separator adapter not connected" if separator.provider != "local_sepformer" else
                          "Recognition model not installed" if not weights else
                          "Separation assets not installed" if not separator_ready else
                          "Processing runtime not installed" if not runtime else "Installed · runs locally")
                variants.append({"id": f"{separator.id}--{model.id}", "separator": separator.label,
                    "model": model.label, "model_id": model.id, "separator_id": separator.id,
                    "ready": reason == "Installed · runs locally", "reason": reason})
        return variants

    def process(self, job_id: str, request: ExperimentRequest) -> dict[str, object]:
        folder = self.root / self.config.storage_directory / "runs" / job_id
        folder.mkdir(parents=True, exist_ok=False)
        source_folder = self.current_folder if request.recording_id == "current" else self.uploads / request.recording_id / "result"
        values = {"request": asdict(request), "study": self.repository.study(request.recording_id),
                  "recording_folder": str(source_folder.resolve()), "lab_config": asdict(self.config)}
        config_path = folder / "input.json"
        config_path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
        try:
            with (folder / "processing.log").open("w", encoding="utf-8") as log:
                subprocess.run([str(self.root / self.config.python_executable), str(self.root / "scripts/run_overlap_lab.py"),
                    "--input", str(config_path), "--output", str(folder / "result.json")], cwd=self.root,
                    stdout=log, stderr=subprocess.STDOUT, timeout=self.config.job_timeout_seconds, check=True, shell=False)
            return json.loads((folder / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise LabError("Local comparison worker failed") from error

    def load(self, job_id: str) -> LabJob | None:
        """Recover completed jobs after restart; never expose private input/log files."""
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            return None
        folder = self.root / self.config.storage_directory / "runs" / job_id
        try:
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            recording_id = result["recording_id"]
            if result["source_hash"] != self.repository.study(recording_id)["source_hash"]:
                return None
            return LabJob(job_id, recording_id, "completed", "Comparison ready", result)
        except (OSError, ValueError, KeyError, TypeError, LabError):
            return None

    def history(self, recording_id: str) -> list[LabJob]:
        root = self.root / self.config.storage_directory / "runs"
        if not root.is_dir():
            return []
        jobs: list[LabJob] = []
        for folder in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if folder.is_dir() and re.fullmatch(r"[a-f0-9]{32}", folder.name):
                job = self.load(folder.name)
                if job and job.recording_id == recording_id:
                    jobs.append(job)
                    if len(jobs) >= self.config.max_history:
                        break
        return jobs
