import re
from ai_layer.common.config_loader import SpeakerVerificationConfig
from ai_layer.common.llm_provider import LLMProvider
from ai_layer.models.base_models import TimeRange
from ai_layer.models.speaker_models import SpeakerProfile, VerifiedTranscript, VerifiedTurn
from ai_layer.models.transcription_models import STTTranscriptOutput, StitchedTurn
from ai_layer.speaker_verification.confidence_scorer import SpeakerConfidenceScorer
from ai_layer.speaker_verification.cue_extractor import IdentityCueExtractor
from ai_layer.speaker_verification.identity_resolver import IdentityResolver
from ai_layer.transcription.transcript_parser import TranscriptParser


class SpeakerVerificationService:
    def __init__(self, llm_provider: LLMProvider, config: SpeakerVerificationConfig) -> None:
        self._cue_extractor = IdentityCueExtractor(llm_provider)
        self._identity_resolver = IdentityResolver()
        self._confidence_scorer = SpeakerConfidenceScorer(config)
        self._parser = TranscriptParser()

    def verify_speakers(
        self,
        transcript: STTTranscriptOutput,
        registered_profiles: list[SpeakerProfile],
    ) -> VerifiedTranscript:
        # Step 1: Expand any turn that contains embedded dialogue or multiple inline speakers
        processed_turns: list[StitchedTurn] = []
        turn_counter = 1

        for turn in transcript.turns:
            inline_turns = self._parser._parse_inline_or_formatted_dialogue(turn.text)
            if inline_turns and len(inline_turns) > 1:
                total_duration = max(1.0, turn.time_range.end_time_seconds - turn.time_range.start_time_seconds)
                total_words = sum(max(len(st.text.split()), 1) for st in inline_turns)
                cur_start = turn.time_range.start_time_seconds

                for st in inline_turns:
                    sub_words = max(len(st.text.split()), 1)
                    sub_duration = (sub_words / total_words) * total_duration
                    sub_end = min(turn.time_range.end_time_seconds, cur_start + sub_duration)
                    clean_name = st.speaker_name.strip()
                    norm_id = re.sub(r"[^a-zA-Z0-9_]", "", clean_name.lower().replace(" ", "_"))
                    spk_id = f"spk_{norm_id}" if norm_id else f"spk_{turn_counter}"

                    processed_turns.append(
                        StitchedTurn(
                            turn_id=f"turn_{turn_counter}",
                            speaker_id=spk_id,
                            text=st.text,
                            time_range=TimeRange(
                                start_time_seconds=round(cur_start, 2),
                                end_time_seconds=round(sub_end, 2),
                            ),
                            source_utterance_ids=getattr(turn, "source_utterance_ids", []),
                        )
                    )
                    cur_start = sub_end
                    turn_counter += 1
            else:
                processed_turns.append(
                    StitchedTurn(
                        turn_id=f"turn_{turn_counter}",
                        speaker_id=turn.speaker_id,
                        text=turn.text,
                        time_range=turn.time_range,
                        source_utterance_ids=getattr(turn, "source_utterance_ids", []),
                    )
                )
                turn_counter += 1

        # Step 2: Extract identity cues and resolve identities
        # Acoustic cluster labels are not identities. With no registered profiles,
        # keep them anonymous even if noisy STT says "I am very ..." or a name.
        identity_turns = [turn for turn in processed_turns
                          if registered_profiles or not turn.speaker_id.startswith("SPEAKER_")]
        cues = self._cue_extractor.extract_cues_from_turns(identity_turns) if identity_turns else []
        preliminary_ids = list({t.speaker_id for t in processed_turns})
        speaker_names, _, reassignments = self._identity_resolver.resolve_identities(
            cues,
            registered_profiles,
            preliminary_ids,
        )

        for profile in registered_profiles:
            self._confidence_scorer.compute_confidence(profile, cues)

        verified_turns = []
        for turn in processed_turns:
            resolved_name = speaker_names.get(turn.speaker_id)
            if not resolved_name or resolved_name == turn.speaker_id:
                if turn.speaker_id.startswith("spk_"):
                    resolved_name = turn.speaker_id[4:].replace("_", " ").title()
                else:
                    resolved_name = turn.speaker_id

            verified_turns.append(
                VerifiedTurn(
                    turn_id=turn.turn_id,
                    speaker_id=turn.speaker_id,
                    speaker_name=resolved_name,
                    text=turn.text,
                    time_range=turn.time_range,
                )
            )

        return VerifiedTranscript(
            session_id=transcript.session_id,
            turns=verified_turns,
            reassignments=reassignments,
        )
