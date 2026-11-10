from ai_layer.common.config_loader import SpeakerVerificationConfig
from ai_layer.models.speaker_models import IdentityCue, SpeakerProfile


class SpeakerConfidenceScorer:
    def __init__(self, config: SpeakerVerificationConfig) -> None:
        self._config = config

    def compute_confidence(
        self,
        profile: SpeakerProfile,
        cues: list[IdentityCue],
    ) -> float:
        relevant = [
            cue
            for cue in cues
            if cue.detected_name.lower() == profile.canonical_name.lower()
            or cue.detected_name.lower() in {alias.lower() for alias in profile.aliases}
        ]
        boost = len(relevant) * self._config.boost_per_confirmation
        confidence = min(1.0, profile.base_confidence + boost)
        return confidence if confidence >= self._config.min_confidence_threshold else confidence

    def meets_threshold(self, confidence: float) -> bool:
        return confidence >= self._config.min_confidence_threshold
