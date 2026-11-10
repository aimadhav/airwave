from ai_layer.common.llm_provider import LLMProvider
from ai_layer.models.analysis_models import AnalysisReport
from ai_layer.models.base_models import SessionMetadata
from ai_layer.models.speaker_models import VerifiedTranscript
from ai_layer.models.summary_models import SessionSummary


class SessionSummarizer:
    def __init__(self, llm_provider: LLMProvider, max_highlights: int) -> None:
        self._llm_provider = llm_provider
        self._max_highlights = max_highlights

    def generate_overview(
        self,
        verified_transcript: VerifiedTranscript,
        session_metadata: SessionMetadata,
    ) -> tuple[str, list[str]]:
        """Generates the narrative overview and discussion highlights using the LLM."""
        highlights = [
            turn.text for turn in verified_transcript.turns[: self._max_highlights]
        ]
        overview_prompt = (
            "Summarize this meeting session.\n"
            f"Title: {session_metadata.session_title}\n"
            f"Participants: {', '.join(session_metadata.participant_ids)}\n"
            f"Transcript excerpt: {' '.join(highlights)}"
        )
        overview = self._llm_provider.generate_text(overview_prompt)
        return overview, highlights

    def assemble_session_summary(
        self,
        overview: str,
        discussion_highlights: list[str],
        analysis_report: AnalysisReport,
        session_metadata: SessionMetadata,
    ) -> SessionSummary:
        """Assembles the final SessionSummary dataclass from overview and insights."""
        return SessionSummary(
            session_id=session_metadata.session_id,
            title=session_metadata.session_title,
            overview=overview,
            discussion_highlights=discussion_highlights,
            decisions_summary=[decision.decision for decision in analysis_report.decisions],
            action_items_summary=[
                f"{task.assignee_name}: {task.task_description}"
                + (f" (due {task.deadline})" if task.deadline else "")
                for task in analysis_report.action_items
            ],
        )

    def summarize_session(
        self,
        verified_transcript: VerifiedTranscript,
        analysis_report: AnalysisReport,
        session_metadata: SessionMetadata,
    ) -> SessionSummary:
        overview, highlights = self.generate_overview(verified_transcript, session_metadata)
        return self.assemble_session_summary(
            overview=overview,
            discussion_highlights=highlights,
            analysis_report=analysis_report,
            session_metadata=session_metadata,
        )
