"""Unit tests for progressive saving and concurrent fault-tolerant extraction."""

from pathlib import Path
from unittest.mock import MagicMock

from ai_layer.common.config_loader import load_config
from ai_layer.common.llm_provider import MockLLMProvider
from ai_layer.conversation_analysis.service import ConversationAnalysisService
from ai_layer.models.analysis_models import ActionItem, ExtractedDecision, ExtractedFact, UtteranceCitation
from ai_layer.models.base_models import SessionMetadata, TimeRange
from ai_layer.models.speaker_models import VerifiedTranscript, VerifiedTurn
from ai_layer.models.summary_models import SessionSummary
from ai_layer.pipeline_orchestrator import AILayerOrchestrator
from ai_layer.storage.sqlite_repository import SQLiteRepository


def _dummy_turns() -> list[VerifiedTurn]:
    return [
        VerifiedTurn(
            turn_id="turn_1",
            speaker_id="spk_sarah",
            speaker_name="Sarah",
            text="We agreed to reschedule the launch to October 15.",
            time_range=TimeRange(0.0, 3.0),
        ),
        VerifiedTurn(
            turn_id="turn_2",
            speaker_id="spk_david",
            speaker_name="David",
            text="I will prepare the slide deck by Friday.",
            time_range=TimeRange(3.5, 6.0),
        ),
        VerifiedTurn(
            turn_id="turn_3",
            speaker_id="spk_alex",
            speaker_name="Alex",
            text="The client renewed the annual contract on Monday.",
            time_range=TimeRange(6.5, 9.0),
        ),
    ]


def test_sqlite_progressive_saving(tmp_path: Path):
    db_file = str(tmp_path / "test_progressive.db")
    repo = SQLiteRepository(db_path=db_file)

    session_id = "sess_prog_001"
    metadata = SessionMetadata(
        session_id=session_id,
        session_title="Progressive Sync",
        recorded_at="2026-09-18T10:00:00Z",
        participant_ids=["spk_1", "spk_2"],
    )

    # 1. Save metadata initially as 'processing'
    repo.save_session_metadata(
        workspace_id="ws_test",
        session_metadata=metadata,
        status="processing",
        progress={"stage": "started", "percent": 15},
    )

    status_info = repo.load_session_status(session_id)
    assert status_info["status"] == "processing"
    assert status_info["progress"]["stage"] == "started"

    # 2. Save transcript
    transcript = VerifiedTranscript(session_id=session_id, turns=_dummy_turns())
    repo.save_transcript(session_id, transcript)

    loaded_trans = repo.load_verified_transcript(session_id)
    assert loaded_trans is not None
    assert len(loaded_trans.turns) == 3

    # 3. Save action items first (before facts or decisions)
    task = ActionItem(
        task_id="task_1",
        task_description="Prepare slide deck",
        assignee_name="David",
        deadline="Friday",
        citation=UtteranceCitation(
            turn_id="turn_2",
            speaker_name="David",
            time_range=TimeRange(3.5, 6.0),
            spoken_text="I will prepare the slide deck by Friday.",
        ),
    )
    repo.save_action_items(session_id, [task])

    # Check report has tasks and empty facts/decisions
    report = repo.load_analysis_report(session_id)
    assert report is not None
    assert len(report.action_items) == 1
    assert len(report.facts) == 0
    assert len(report.decisions) == 0

    # 4. Save decisions next without clobbering action items
    decision = ExtractedDecision(
        decision_id="dec_1",
        decision="Reschedule launch to October 15",
        agreed_by=["Sarah"],
        citation=UtteranceCitation(
            turn_id="turn_1",
            speaker_name="Sarah",
            time_range=TimeRange(0.0, 3.0),
            spoken_text="Let's reschedule the launch to October 15.",
        ),
    )
    repo.save_decisions(session_id, [decision])

    report = repo.load_analysis_report(session_id)
    assert len(report.action_items) == 1
    assert len(report.decisions) == 1
    assert len(report.facts) == 0

    # 5. Save facts next without clobbering decisions or action items
    fact = ExtractedFact(
        fact_id="fact_1",
        statement="The client renewed contract on Monday.",
        citation=UtteranceCitation(
            turn_id="turn_3",
            speaker_name="Alex",
            time_range=TimeRange(6.5, 9.0),
            spoken_text="The client renewed the annual contract on Monday.",
        ),
    )
    repo.save_facts(session_id, [fact])

    report = repo.load_analysis_report(session_id)
    assert len(report.action_items) == 1
    assert len(report.decisions) == 1
    assert len(report.facts) == 1

    # 6. Save executive session summary
    summary = SessionSummary(
        session_id=session_id,
        title="Progressive Sync",
        overview="Team aligned on October 15 launch date.",
        discussion_highlights=["Launch date rescheduled", "Slides due Friday"],
        decisions_summary=["Reschedule launch to October 15"],
        action_items_summary=["David: Prepare slide deck"],
    )
    repo.save_session_summary(session_id, summary)

    loaded_summary = repo.load_session_summary(session_id)
    assert loaded_summary is not None
    assert loaded_summary.overview == "Team aligned on October 15 launch date."
    assert len(loaded_summary.discussion_highlights) == 2

    # 7. Update status to completed
    repo.update_session_status(session_id, "completed", {"stage": "done", "percent": 100})
    final_status = repo.load_session_status(session_id)
    assert final_status["status"] == "completed"


def test_concurrent_extractors_with_callbacks():
    config = load_config("configs/ai_layer_config.yaml")
    service = ConversationAnalysisService(MockLLMProvider(), config.conversation_analysis)

    verified = VerifiedTranscript(session_id="test_sess", turns=_dummy_turns())
    metadata = SessionMetadata(
        session_id="test_sess",
        session_title="Sync",
        recorded_at="2026-09-18T10:00:00Z",
        participant_ids=[],
    )

    received_facts = []
    received_decisions = []
    received_tasks = []

    def on_facts(f):
        received_facts.extend(f)

    def on_decisions(d):
        received_decisions.extend(d)

    def on_tasks(t):
        received_tasks.extend(t)

    report = service.analyze_conversation(
        verified_transcript=verified,
        session_metadata=metadata,
        on_facts_ready=on_facts,
        on_decisions_ready=on_decisions,
        on_action_items_ready=on_tasks,
        concurrent_extractors=True,
    )

    assert len(report.facts) > 0
    assert len(report.decisions) > 0
    assert len(report.action_items) > 0

    assert len(received_facts) > 0
    assert len(received_decisions) > 0
    assert len(received_tasks) > 0


def test_extractor_fault_isolation():
    """Verify that if one extractor raises an exception, the others still succeed and are saved."""
    config = load_config("configs/ai_layer_config.yaml")
    service = ConversationAnalysisService(MockLLMProvider(), config.conversation_analysis)

    # Inject a failing fact extractor
    failing_extractor = MagicMock()
    failing_extractor.extract_facts.side_effect = RuntimeError("Simulated API rate limit / network failure on facts")
    service._fact_extractor = failing_extractor

    verified = VerifiedTranscript(session_id="test_sess_fail", turns=_dummy_turns())
    metadata = SessionMetadata(
        session_id="test_sess_fail",
        session_title="Fault Isolation Test",
        recorded_at="2026-09-18T10:00:00Z",
        participant_ids=[],
    )

    received_tasks = []
    received_decisions = []

    report = service.analyze_conversation(
        verified_transcript=verified,
        session_metadata=metadata,
        on_action_items_ready=lambda t: received_tasks.extend(t),
        on_decisions_ready=lambda d: received_decisions.extend(d),
        concurrent_extractors=True,
    )

    # Facts failed, but decisions and tasks succeeded!
    assert len(report.action_items) > 0
    assert len(report.decisions) > 0
    assert len(received_tasks) > 0
    assert len(received_decisions) > 0


def test_orchestrator_progressive_pipeline(tmp_path: Path):
    db_file = str(tmp_path / "test_orch_prog.db")
    repo = SQLiteRepository(db_path=db_file)

    config = load_config("configs/ai_layer_config.yaml")
    orch = AILayerOrchestrator(config=config, repository=repo)

    session_id = "sess_orch_001"
    metadata = SessionMetadata(
        session_id=session_id,
        session_title="Orchestrator Progressive Test",
        recorded_at="2026-09-18T10:00:00Z",
        participant_ids=["spk_sarah", "spk_alex"],
    )

    transcript_text = """
    Sarah: We agreed to reschedule the launch date to October 15.
    Alex: I will prepare the slide deck by Friday.
    """

    result = orch.analyze_transcript(
        transcript_input=transcript_text,
        session_metadata=metadata,
        workspace_id="ws_test",
    )

    assert result.session_summary.overview
    assert len(result.analysis_report.facts) > 0 or len(result.analysis_report.decisions) > 0

    # Verify everything persisted in SQLite
    status_info = repo.load_session_status(session_id)
    assert status_info["status"] == "completed"

    stored_trans = repo.load_verified_transcript(session_id)
    assert stored_trans is not None
    assert len(stored_trans.turns) >= 2

    stored_report = repo.load_analysis_report(session_id)
    assert stored_report is not None

    stored_summary = repo.load_session_summary(session_id)
    assert stored_summary is not None
    assert len(stored_summary.overview) > 0
