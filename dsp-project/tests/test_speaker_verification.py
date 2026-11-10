from ai_layer.common.config_loader import load_config
from ai_layer.common.llm_provider import MockLLMProvider
from ai_layer.speaker_verification.confidence_scorer import SpeakerConfidenceScorer
from ai_layer.speaker_verification.cue_extractor import IdentityCueExtractor
from ai_layer.speaker_verification.identity_resolver import IdentityResolver
from ai_layer.speaker_verification.service import SpeakerVerificationService
from ai_layer.transcription.service import STTTranscriptionService
from ai_layer.common.stt_provider import MockSTTProvider
from tests.conftest import sample_clips, sample_speakers


def _verified_transcript():
    config = load_config("configs/ai_layer_config.yaml")
    transcript = STTTranscriptionService(
        MockSTTProvider(), config.transcription
    ).process_session_clips("session_001", sample_clips())
    service = SpeakerVerificationService(MockLLMProvider(), config.speaker_verification)
    return service.verify_speakers(transcript, sample_speakers())


def test_cue_extractor():
    config = load_config("configs/ai_layer_config.yaml")
    transcript = STTTranscriptionService(
        MockSTTProvider(), config.transcription
    ).process_session_clips("session_001", sample_clips())
    cues = IdentityCueExtractor(MockLLMProvider()).extract_cues_from_turns(transcript.turns)
    assert any(cue.detected_name == "Sarah" for cue in cues)


def test_identity_resolver():
    config = load_config("configs/ai_layer_config.yaml")
    transcript = STTTranscriptionService(
        MockSTTProvider(), config.transcription
    ).process_session_clips("session_001", sample_clips())
    cues = IdentityCueExtractor(MockLLMProvider()).extract_cues_from_turns(transcript.turns)
    names, _, reassignments = IdentityResolver().resolve_identities(
        cues,
        sample_speakers(),
        list({turn.speaker_id for turn in transcript.turns}),
    )
    assert names["Speaker_2"] == "Sarah"
    assert reassignments


def test_confidence_scorer():
    config = load_config("configs/ai_layer_config.yaml")
    transcript = STTTranscriptionService(
        MockSTTProvider(), config.transcription
    ).process_session_clips("session_001", sample_clips())
    cues = IdentityCueExtractor(MockLLMProvider()).extract_cues_from_turns(transcript.turns)
    scorer = SpeakerConfidenceScorer(config.speaker_verification)
    confidence = scorer.compute_confidence(sample_speakers()[0], cues)
    assert confidence >= sample_speakers()[0].base_confidence


def test_speaker_verification_service():
    verified = _verified_transcript()
    assert verified.session_id == "session_001"
    assert any(turn.speaker_name == "Sarah" for turn in verified.turns)
