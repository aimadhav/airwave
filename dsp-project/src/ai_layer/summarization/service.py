from ai_layer.common.config_loader import SummarizationConfig
from ai_layer.common.llm_provider import LLMProvider
from ai_layer.models.analysis_models import AnalysisReport
from ai_layer.models.base_models import SessionMetadata
from ai_layer.models.speaker_models import VerifiedTranscript
from ai_layer.models.summary_models import ProjectSummary, SessionSummary
from ai_layer.summarization.project_summarizer import ProjectSummarizer
from ai_layer.summarization.session_summarizer import SessionSummarizer


class SummarizationService:
    def __init__(self, llm_provider: LLMProvider, config: SummarizationConfig) -> None:
        self._session_summarizer = SessionSummarizer(llm_provider, config.max_highlights)
        self._project_summarizer = ProjectSummarizer(llm_provider)
        self._include_action_items = config.include_action_items

    def generate_session_overview(
        self,
        verified_transcript: VerifiedTranscript,
        session_metadata: SessionMetadata,
    ) -> tuple[str, list[str]]:
        return self._session_summarizer.generate_overview(
            verified_transcript, session_metadata
        )

    def assemble_session_summary(
        self,
        overview: str,
        discussion_highlights: list[str],
        analysis_report: AnalysisReport,
        session_metadata: SessionMetadata,
    ) -> SessionSummary:
        summary = self._session_summarizer.assemble_session_summary(
            overview=overview,
            discussion_highlights=discussion_highlights,
            analysis_report=analysis_report,
            session_metadata=session_metadata,
        )
        if not self._include_action_items:
            return SessionSummary(
                session_id=summary.session_id,
                title=summary.title,
                overview=summary.overview,
                discussion_highlights=summary.discussion_highlights,
                decisions_summary=summary.decisions_summary,
                action_items_summary=[],
            )
        return summary

    def summarize_session(
        self,
        verified_transcript: VerifiedTranscript,
        analysis_report: AnalysisReport,
        session_metadata: SessionMetadata,
    ) -> SessionSummary:
        summary = self._session_summarizer.summarize_session(
            verified_transcript,
            analysis_report,
            session_metadata,
        )
        if not self._include_action_items:
            return SessionSummary(
                session_id=summary.session_id,
                title=summary.title,
                overview=summary.overview,
                discussion_highlights=summary.discussion_highlights,
                decisions_summary=summary.decisions_summary,
                action_items_summary=[],
            )
        return summary

    def update_project_summary(
        self,
        project_id: str,
        project_title: str,
        session_summary: SessionSummary,
        existing_summary: ProjectSummary | None = None,
    ) -> ProjectSummary:
        return self._project_summarizer.update_project_summary(
            project_id,
            project_title,
            session_summary,
            existing_summary,
        )
