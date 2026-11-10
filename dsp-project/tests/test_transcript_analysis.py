"""Unit tests for TranscriptParser and the decoupled Main Analyzer pipeline."""

from ai_layer.common.config_loader import load_config
from ai_layer.models.base_models import SessionMetadata
from ai_layer.models.speaker_models import SpeakerProfile, VerifiedTranscript, VerifiedTurn
from ai_layer.models.transcription_models import STTTranscriptOutput
from ai_layer.pipeline_orchestrator import AILayerOrchestrator
from ai_layer.transcription.transcript_parser import TranscriptParser
from tests.conftest import orchestrator, sample_clips, sample_session_metadata, sample_speakers


def test_transcript_parser_standard_dialogue():
    text = """
    Sarah: Let's review the Q4 release roadmap.
    Alex: I've completed the database migration and verified all tests.
    David: I will finish the API security checklist by Friday 5 PM.
    """
    parser = TranscriptParser()
    profiles = [
        SpeakerProfile(speaker_id="spk_sarah", canonical_name="Sarah", aliases=["Sara"]),
        SpeakerProfile(speaker_id="spk_alex", canonical_name="Alex"),
    ]
    verified = parser.parse_to_verified_transcript(text, session_id="test_sess", known_speakers=profiles)

    assert len(verified.turns) == 3
    assert verified.turns[0].speaker_id == "spk_sarah"
    assert verified.turns[0].speaker_name == "Sarah"
    assert "roadmap" in verified.turns[0].text
    assert verified.turns[1].speaker_id == "spk_alex"
    assert verified.turns[2].speaker_id == "spk_david"
    # Ensure sequential timestamps are generated
    assert verified.turns[0].time_range.start_time_seconds == 0.0
    assert verified.turns[1].time_range.start_time_seconds >= verified.turns[0].time_range.end_time_seconds


def test_transcript_parser_timestamps():
    text = """
    [00:01:10 - 00:01:25] Sarah: We decided to reschedule the launch date to October 15.
    [00:01:28 - 00:01:40] Alex: Agreed, and David can prepare the slide deck.
    """
    parser = TranscriptParser()
    verified = parser.parse_to_verified_transcript(text, session_id="sess_ts")

    assert len(verified.turns) == 2
    assert verified.turns[0].time_range.start_time_seconds == 70.0
    assert verified.turns[0].time_range.end_time_seconds == 85.0
    assert verified.turns[1].time_range.start_time_seconds == 88.0
    assert verified.turns[1].time_range.end_time_seconds == 100.0


def test_transcript_parser_subtitles_and_json():
    # Test JSON array
    json_text = '[{"speaker": "Sarah", "text": "Hello world", "start": 0.0, "end": 4.0}]'
    parser = TranscriptParser()
    res_json = parser.parse_to_verified_transcript(json_text, session_id="sess_json")
    assert len(res_json.turns) == 1
    assert res_json.turns[0].speaker_name == "Sarah"
    assert res_json.turns[0].time_range.end_time_seconds == 4.0

    # Test WebVTT / SRT style
    srt_text = """
    1
    00:00:02,000 --> 00:00:06,000
    Sarah: Welcome to the project sync.

    2
    00:00:07,000 --> 00:00:11,000
    Alex: Thanks Sarah, glad to be here.
    """
    res_srt = parser.parse_to_verified_transcript(srt_text, session_id="sess_srt")
    assert len(res_srt.turns) == 2
    assert res_srt.turns[0].speaker_name == "Sarah"
    assert res_srt.turns[0].time_range.start_time_seconds == 2.0
    assert res_srt.turns[0].time_range.end_time_seconds == 6.0


def test_transcript_parser_plain_text():
    raw_notes = "We agreed that the product launch will be on October 1st. Sarah will send the invites."
    parser = TranscriptParser()
    verified = parser.parse_to_verified_transcript(raw_notes, session_id="sess_plain")
    assert len(verified.turns) >= 1
    assert "October 1st" in verified.turns[0].text


def test_orchestrator_analyze_transcript_direct():
    orch = orchestrator()
    metadata = SessionMetadata(
        session_id="session_direct_test",
        session_title="Direct Transcript Test",
        recorded_at="2026-09-17T12:00:00",
        participant_ids=["Speaker_1", "Speaker_2"],
        project_id="project_launch",
    )

    transcript_text = """
    Sarah: The client renewed the annual contract on Monday.
    Alex: The team decided to reschedule the launch date to October 15.
    David: I will prepare the slide deck by Friday.
    """

    result = orch.analyze_transcript(
        transcript_input=transcript_text,
        session_metadata=metadata,
        known_speakers=sample_speakers(),
    )

    assert len(result.verified_transcript.turns) == 3
    assert result.analysis_report.facts
    assert result.analysis_report.decisions
    assert result.analysis_report.action_items
    assert result.session_summary.overview
    assert result.project_summary is not None


def test_orchestrator_stt_and_analyzer_separation():
    orch = orchestrator()
    metadata = sample_session_metadata()
    clips = sample_clips()
    speakers = sample_speakers()

    # Step 1: Run STT Layer alone
    stt_output = orch.transcribe_audio_session(metadata.session_id, clips)
    assert isinstance(stt_output, STTTranscriptOutput)
    assert len(stt_output.turns) >= 4

    # Step 2: Run Main Analyzer alone on STT output
    analyzer_result = orch.analyze_transcript(
        transcript_input=stt_output,
        session_metadata=metadata,
        known_speakers=speakers,
    )
    assert analyzer_result.verified_transcript.turns
    assert analyzer_result.analysis_report.facts
    assert analyzer_result.session_summary.overview


def test_transcript_parser_inline_multispeaker_with_preamble():
    raw_text = (
        "Sure. Here is a **single conversation among four people**, where everyone talks to each other naturally: "
        "**Rahul:** Hey guys, today we were learning about classical feature extraction methods, and I found RANSAC and Hough Transform quite interesting. "
        "**Aman:** Yes, RANSAC is useful when our data contains outliers because it randomly selects points, fits a model, and chooses the model with the maximum number of inliers. "
        "**Priya:** Exactly, and Hough Transform is also used for detecting geometric structures like lines; instead of directly fitting a line, it uses voting in parameter space. "
        "**Neha:** That makes sense, and after detecting these structures, we can extract image features and keypoints for applications like object recognition, localization, and motion tracking. "
        "**Rahul:** We also learned about SIFT, which makes features more robust to changes in scale and rotation. "
        "**Aman:** Right, and the SIFT descriptor represents a keypoint using a 128-dimensional vector. "
        "**Priya:** So basically, filtering helps clean the image, RANSAC and Hough help extract geometric information, and SIFT helps us find and match important features. "
        "**Neha:** Yes, and all four of us can use these techniques to understand how a robot processes images and extracts useful information from them."
    )
    parser = TranscriptParser()
    verified = parser.parse_to_verified_transcript(raw_text, session_id="sess_multi_inline")

    # Must produce 8 distinct turns across the 4 participants
    assert len(verified.turns) == 8
    expected_speakers = ["Rahul", "Aman", "Priya", "Neha", "Rahul", "Aman", "Priya", "Neha"]
    for turn, expected_name in zip(verified.turns, expected_speakers):
        assert turn.speaker_name == expected_name
        assert turn.speaker_id == f"spk_{expected_name.lower()}"
        assert turn.time_range.start_time_seconds < turn.time_range.end_time_seconds

    # Participants should contain all 4 speakers
    participants = {t.speaker_name for t in verified.turns}
    assert participants == {"Rahul", "Aman", "Priya", "Neha"}


def test_speaker_verification_embedded_dialogue():
    from ai_layer.common.config_loader import SpeakerVerificationConfig
    from ai_layer.common.llm_provider import MockLLMProvider
    from ai_layer.models.base_models import TimeRange
    from ai_layer.models.transcription_models import StitchedTurn
    from ai_layer.speaker_verification.service import SpeakerVerificationService

    service = SpeakerVerificationService(
        llm_provider=MockLLMProvider(),
        config=SpeakerVerificationConfig(min_confidence_threshold=0.7, boost_per_confirmation=0.15),
    )

    # Simulate an STT output with a single turn containing dialogue between two speakers
    stt_output = STTTranscriptOutput(
        session_id="sess_embedded",
        turns=[
            StitchedTurn(
                turn_id="turn_1",
                speaker_id="Speaker_1",
                text="Rahul: Let's discuss feature matching. Aman: I agree, SIFT is very robust.",
                time_range=TimeRange(start_time_seconds=0.0, end_time_seconds=20.0),
            )
        ],
    )

    verified = service.verify_speakers(stt_output, registered_profiles=[])
    assert len(verified.turns) == 2
    assert verified.turns[0].speaker_name == "Rahul"
    assert verified.turns[1].speaker_name == "Aman"
    assert verified.turns[0].time_range.start_time_seconds == 0.0
    assert verified.turns[1].time_range.end_time_seconds <= 20.0


def test_transcript_parser_preamble_stripping():
    text = """
    Sure, here is the conversation:
    Rahul: First point discussed.
    Aman: Second point agreed.
    """
    parser = TranscriptParser()
    verified = parser.parse_to_verified_transcript(text, session_id="sess_preamble")

    assert len(verified.turns) == 2
    assert verified.turns[0].speaker_name == "Rahul"
    assert verified.turns[1].speaker_name == "Aman"

