"""Offline demonstrations must not invent evidence or speaker identities."""

from ai_layer.common.llm_provider import MockLLMProvider


def test_mock_does_not_invent_missing_insights() -> None:
    provider = MockLLMProvider()
    prompt = "structured conversation intelligence\n[turn_1] SPEAKER_00: Hello everyone."
    assert provider.generate_structured_json(prompt) == {
        "facts": [], "decisions": [], "action_items": [],
    }
    assert provider.generate_structured_json(
        "Extract identity cues\n[turn_1] SPEAKER_00: Hello everyone."
    ) == {"cues": []}


def test_demo_insights_cite_actual_turns() -> None:
    prompt = (
        "structured conversation intelligence\n"
        "[turn_1] SPEAKER_00: Welcome everyone.\n"
        "[turn_2] SPEAKER_01: The migration is complete and the tests are passing.\n"
        "[turn_3] SPEAKER_02: I will finish the security review by Friday.\n"
        "[turn_4] SPEAKER_00: We agreed to target the October 15 release."
    )
    result = MockLLMProvider().generate_structured_json(prompt)
    assert result["facts"] == [{"statement": "The migration is complete and the tests are passing.", "turn_id": "turn_2"}]
    assert result["decisions"][0]["turn_id"] == "turn_4"
    assert result["decisions"][0]["agreed_by"] == ["SPEAKER_00"]
    assert result["action_items"][0]["turn_id"] == "turn_3"
    assert result["action_items"][0]["assignee_name"] == "SPEAKER_02"
    assert result["action_items"][0]["deadline"] == "Friday"


def test_mock_introduction_requires_supplied_name() -> None:
    result = MockLLMProvider().generate_structured_json(
        "Extract identity cues\n[turn_7] Speaker_9: I'm Priya from engineering."
    )
    assert result["cues"][0]["detected_name"] == "Priya"
    assert result["cues"][0]["source_turn_id"] == "turn_7"
    assert result["cues"][0]["target_speaker_id"] == "Speaker_9"


def test_dialogue_words_do_not_select_mock_instruction_handler() -> None:
    result = MockLLMProvider().generate_structured_json(
        "Extract facts\n[turn_1] Speaker_1: We discussed facts, decisions and action items."
    )
    assert result == {"facts": []}
def test_unregistered_acoustic_speakers_stay_anonymous() -> None:
    from ai_layer.common.config_loader import SpeakerVerificationConfig
    from ai_layer.common.llm_provider import MockLLMProvider
    from ai_layer.models.base_models import TimeRange
    from ai_layer.models.transcription_models import STTTranscriptOutput, StitchedTurn
    from ai_layer.speaker_verification.service import SpeakerVerificationService
    transcript = STTTranscriptOutput("real", turns=[
        StitchedTurn("a", "SPEAKER_00", "I am very interested in generative AI.", TimeRange(0, 2)),
        StitchedTurn("b", "SPEAKER_01", "I'm Alex and I can explain it.", TimeRange(2, 4)),
    ])
    verified = SpeakerVerificationService(MockLLMProvider(), SpeakerVerificationConfig(0.7, 0.15)).verify_speakers(transcript, [])
    assert [turn.speaker_name for turn in verified.turns] == ["SPEAKER_00", "SPEAKER_01"]
