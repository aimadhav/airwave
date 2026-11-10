"""Immutable local JSON storage; profile vectors must never be published by default."""

import json
import os
from pathlib import Path
import re
import tempfile

from recording_viewer.exceptions import VoiceProfileError
from recording_viewer.models.voice_profiles import VoiceProfileSet


class JsonVoiceProfileStore:
    """Atomic no-overwrite publication of one validated private profile document."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()

    def path_for(self, key: str) -> Path:
        """Never interpolate unvalidated identifiers into file paths."""
        if not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{64}", key):
            raise VoiceProfileError("Invalid voice-profile key")
        return self.directory / f"{key}.json"

    def save(self, profiles: VoiceProfileSet) -> str:
        """Publish a complete file atomically; never overwrite prior profiles."""
        key = profiles.context.key
        destination = self.path_for(key)
        encoded = json.dumps(profiles.to_document(), indent=2, ensure_ascii=False, allow_nan=False)
        temporary: Path | None = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                    prefix=".voice-profile-", suffix=".tmp", dir=self.directory, delete=False) as staging:
                temporary = Path(staging.name)
                staging.write(encoded + "\n")
                staging.flush()
                os.fsync(staging.fileno())
            os.link(temporary, destination)
            return key
        except FileExistsError as error:
            raise VoiceProfileError("Voice profiles already exist; existing data was not replaced") from error
        except (OSError, ValueError) as error:
            raise VoiceProfileError("Cannot save local voice profiles") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, key: str) -> VoiceProfileSet:
        """Reject malformed content or a document stored under the wrong key."""
        try:
            result = VoiceProfileSet.from_document(json.loads(self.path_for(key).read_text(encoding="utf-8")))
            if result.context.key != key:
                raise VoiceProfileError("Stored voice-profile context does not match its key")
            return result
        except (OSError, TypeError, ValueError) as error:
            raise VoiceProfileError("Cannot read valid local voice profiles") from error


class MemoryVoiceProfileStore:
    """Deterministic mock repository for tests and other consumers."""

    def __init__(self) -> None:
        self._documents: dict[str, dict] = {}

    def save(self, profiles: VoiceProfileSet) -> str:
        """Preserve the same no-overwrite contract as the real adapter."""
        key = profiles.context.key
        if key in self._documents:
            raise VoiceProfileError("Voice profiles already exist")
        self._documents[key] = profiles.to_document()
        return key

    def load(self, key: str) -> VoiceProfileSet:
        """Round-trip the DTO instead of leaking internal mutable test state."""
        if key not in self._documents:
            raise VoiceProfileError("Voice profiles not found")
        return VoiceProfileSet.from_document(self._documents[key])
