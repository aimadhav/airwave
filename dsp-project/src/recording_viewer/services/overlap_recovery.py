"""Conservative orchestration depends only on provider protocols and DTOs."""

import numpy as np
from recording_viewer.exceptions import RecoveryError
from recording_viewer.interfaces.recovery import ArrayTranscriber, SpeechSeparator, VoiceEmbedder
from recording_viewer.models import TranscriptWord, VoiceRegion
from recording_viewer.models.recovery import AttributedWord, RecoveryResult, RecoveryWindow, WindowReport
from recording_viewer.recovery_config import RecoveryConfig
from recording_viewer.services.recovery_quality import duplicate_text, match_tracks, signal_checks, unit
from recording_viewer.services.recovery_timeline import plan_windows, reference_spans


class OverlapRecovery:
    """Only promote two independently transcribed, confidently matched tracks.

    A third simultaneous speaker, missing reference, low confidence or provider
    failure is a visible unresolved result, never a guessed second transcript.
    """

    def __init__(self, config: RecoveryConfig, separator: SpeechSeparator, embedder: VoiceEmbedder,
                 transcriber: ArrayTranscriber) -> None:
        self.config = config
        self.separator = separator
        self.embedder = embedder
        self.transcriber = transcriber

    def recover(self, samples: np.ndarray, sample_rate: int, regions: list[VoiceRegion],
                *, windows: list[RecoveryWindow] | None = None) -> RecoveryResult:
        """Validate session, enroll clean references and audit every overlap."""
        if sample_rate != 16000 or samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
            raise RecoveryError("Recovery requires finite 16 kHz mono audio")
        duration = len(samples) / sample_rate
        planned = plan_windows(regions, duration, self.config)
        windows = planned if windows is None else windows
        if any(w.crop_end > duration or w.end - w.start > self.config.max_core_seconds for w in windows):
            raise RecoveryError("Requested recovery windows exceed duration limits")
        if not self.config.enabled:
            return RecoveryResult(tuple(WindowReport(w, "skipped", "disabled", {}) for w in windows), {})
        profiles: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        warnings: list[str] = []
        for speaker, spans in reference_spans(regions, duration, self.config).items():
            vectors: list[np.ndarray] = []
            for start, end in spans:
                try:
                    vectors.append(unit(self.embedder.embed(samples[round(start * sample_rate):round(end * sample_rate)], sample_rate)))
                except RecoveryError:
                    warnings.append(f"Reference embedding failed for {speaker}")
            counts[speaker] = len(vectors)
            if len(vectors) >= self.config.minimum_references:
                profiles[speaker] = unit(np.mean(vectors, axis=0))
        reports: list[WindowReport] = []
        attempted = 0
        for window in windows:
            reason = ("unsupported_simultaneous_speaker_count" if len(window.speakers) != 2 else
                      "overlap_too_short" if window.end - window.start < self.config.minimum_overlap_seconds else
                      "insufficient_clean_references" if any(s not in profiles for s in window.speakers) else
                      "window_budget_exceeded" if attempted >= self.config.max_windows else "")
            if reason:
                reports.append(WindowReport(window, "skipped", reason, {}))
                continue
            try:
                attempted += 1
                reports.append(self._recover_window(samples, sample_rate, window, profiles))
            except RecoveryError as error:
                reports.append(WindowReport(window, "failed", str(error), {}))
        return RecoveryResult(tuple(reports), counts, tuple(sorted(set(warnings))))

    def _recover_window(self, samples: np.ndarray, sample_rate: int, window: RecoveryWindow,
                        profiles: dict[str, np.ndarray]) -> WindowReport:
        """Provider offsets are restored using the actual sample-aligned crop."""
        offset = round(window.crop_start * sample_rate)
        crop_start = offset / sample_rate
        crop = samples[offset:round(window.crop_end * sample_rate)]
        output = self.separator.separate(crop, sample_rate)
        if output.sample_rate != sample_rate:
            return WindowReport(window, "rejected", "wrong_source_sample_rate", {})
        if len(output.tracks) != 2 or any(t.ndim != 1 or len(t) != len(crop) for t in output.tracks):
            return WindowReport(window, "rejected", "invalid_source_shape", {})
        # Signal checks use the core. Identity uses the separated track's allowed
        # context too: subsecond embeddings are unstable. Planner excludes third
        # speakers from context; the per-track cosine/margin still has to pass.
        left, right = round((window.start - crop_start) * sample_rate), round((window.end - crop_start) * sample_rate)
        reason, metrics = signal_checks(crop[left:right], tuple(t[left:right] for t in output.tracks), self.config)
        if any(len(t) != len(crop) for t in output.tracks):
            reason = "invalid_source_shape"
        if reason:
            return WindowReport(window, "rejected", reason, metrics)
        if len(crop) / sample_rate < self.config.minimum_embedding_seconds:
            return WindowReport(window, "skipped", "insufficient_voice_for_matching", metrics)
        assignment, scores = match_tracks([self.embedder.embed(t, sample_rate) for t in output.tracks],
                                           window.speakers, profiles, self.config)
        metrics.update(scores)
        if assignment is None:
            return WindowReport(window, "rejected", "ambiguous_speaker_match", metrics)
        recognized: list[list[TranscriptWord]] = []
        for track in output.tracks:
            clip_words = self.transcriber.transcribe_samples(track, sample_rate)
            if any(word.end > len(crop) / sample_rate for word in clip_words):
                return WindowReport(window, "rejected", "recognition_outside_crop", metrics)
            owned = [word for word in clip_words if window.start <= crop_start + (word.start + word.end) / 2 < window.end]
            if not owned:
                return WindowReport(window, "rejected", "no_secondary_words_recovered", metrics)
            probabilities = [word.probability for word in owned]
            if any(p is None for p in probabilities):
                return WindowReport(window, "rejected", "missing_recognition_confidence", metrics)
            if min(probabilities) < self.config.minimum_word_probability or np.mean(probabilities) < self.config.minimum_mean_probability:
                return WindowReport(window, "rejected", "low_recognition_confidence", metrics)
            recognized.append(owned)
        if duplicate_text("".join(w.text for w in recognized[0]), "".join(w.text for w in recognized[1]), self.config):
            return WindowReport(window, "rejected", "duplicated_transcripts", metrics)
        attributed = tuple(AttributedWord(speaker,
            TranscriptWord(word.start + crop_start, word.end + crop_start, word.text, word.probability),
            window.id, metrics[f"track_{i}_cosine"])
            for i, (speaker, words) in enumerate(zip(assignment, recognized)) for word in words)
        return WindowReport(window, "accepted", "quality_gates_passed_not_human_verified", metrics, attributed, assignment)
