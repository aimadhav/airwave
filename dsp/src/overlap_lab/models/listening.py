"""Human listening labels, deliberately independent of transcript references."""

from dataclasses import dataclass
from typing import Literal
from overlap_lab.exceptions import LabInputError
from overlap_lab.models import ClipSpec


ListeningVerdict = Literal["one_speaker", "two_overlapping", "two_sequential", "no_speech", "other_voices", "unsure"]


@dataclass(frozen=True)
class ListeningReview:
    """Describe audible activity in the core, not in the playback context.

    Empty heard_speakers with one_speaker means one voice with unknown identity;
    it does not mean silence. No text, WER, or model-training flag is implied.
    """

    clip: ClipSpec
    verdict: ListeningVerdict
    heard_speakers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, str) or self.verdict not in {"one_speaker", "two_overlapping", "two_sequential", "no_speech", "other_voices", "unsure"}:
            raise LabInputError("Choose a valid listening label")
        if not isinstance(self.heard_speakers, tuple) or any(not isinstance(s, str) for s in self.heard_speakers) or len(set(self.heard_speakers)) != len(self.heard_speakers):
            raise LabInputError("Audible speaker IDs must be distinct strings")
        if not set(self.heard_speakers).issubset(self.clip.speakers):
            raise LabInputError("Audible speaker IDs must belong to the candidate; use other voices otherwise")
        if self.verdict == "one_speaker" and len(self.heard_speakers) > 1:
            raise LabInputError("One voice cannot have two speaker IDs")
        if self.verdict in {"two_overlapping", "two_sequential"} and set(self.heard_speakers) != set(self.clip.speakers):
            raise LabInputError("A two-voice label must identify both candidate speakers")
        if self.verdict in {"no_speech", "other_voices", "unsure"} and self.heard_speakers:
            raise LabInputError("Do not guess speaker IDs for this label")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "ListeningReview":
        """Validate the narrow JSON contract at the HTTP boundary."""
        try:
            if not isinstance(values, dict):
                raise LabInputError("Expected a listening-review object")
            speakers = values.get("heard_speakers", [])
            if not isinstance(speakers, (list, tuple)) or not isinstance(values["verdict"], str):
                raise LabInputError("Invalid listening label")
            return cls(ClipSpec.from_mapping(values["clip"]), values["verdict"], tuple(speakers))
        except (KeyError, TypeError, ValueError) as error:
            raise LabInputError("Invalid listening review") from error
