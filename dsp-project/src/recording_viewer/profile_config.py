"""Explicit local export paths; biometric profile data stays outside public assets."""

from dataclasses import dataclass
import json
from pathlib import Path
from recording_viewer.exceptions import VoiceProfileError


@dataclass(frozen=True)
class VoiceProfileExportConfig:
    """Enrollment parameters themselves come from the existing recovery policy."""

    recording_directory: str
    output_directory: str
    recovery_config: str
    preprocessing_version: str

    @classmethod
    def load(cls, path: Path) -> "VoiceProfileExportConfig":
        """Reject unknown/missing fields before any model execution or output write."""
        try:
            result = cls(**json.loads(path.read_text(encoding="utf-8")))
            if any(not isinstance(v, str) or not v.strip() for v in vars(result).values()):
                raise VoiceProfileError("Profile export configuration requires explicit paths and preprocessing version")
            return result
        except (OSError, TypeError, ValueError) as error:
            raise VoiceProfileError("Invalid voice-profile export configuration") from error
