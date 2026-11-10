"""Persistence boundary independent of the file system or model runtime."""

from typing import Protocol
from recording_viewer.models.voice_profiles import VoiceProfileSet


class VoiceProfileRepository(Protocol):
    """Profile sets are immutable; callers must explicitly select a saved key."""

    def save(self, profiles: VoiceProfileSet) -> str:
        """Persist without replacing another set; return the context key."""
        ...

    def load(self, key: str) -> VoiceProfileSet:
        """Validate and load the selected profile set or raise a typed error."""
        ...
