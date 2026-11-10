"""Main Conversation Analyzer Service.

Decouples post-transcription conversation intelligence from the STT audio layer.
Coordinates:
  1. Conversation Analysis & Extraction (Facts, Decisions, Action Items)
  2. Knowledge Memory Graph Integration & Long-Term Persistence
  3. Session & Project Summarization

Can be invoked with any verified transcript, whether originating from audio STT
or directly pasted/uploaded by a user.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ai_layer.conversation_analysis.service import ConversationAnalysisService
from ai_layer.knowledge_memory.service import KnowledgeMemoryService
from ai_layer.models.analysis_models import AnalysisReport
from ai_layer.models.base_models import SessionMetadata
from ai_layer.models.knowledge_models import KnowledgeStore
from ai_layer.models.speaker_models import VerifiedTranscript
from ai_layer.models.summary_models import ProjectSummary, SessionSummary
from ai_layer.summarization.service import SummarizationService


@dataclass
class SessionProcessingResult:
    """Consolidated result produced by the Main Analyzer."""
    verified_transcript: VerifiedTranscript
    analysis_report: AnalysisReport
    knowledge_store: KnowledgeStore
    session_summary: SessionSummary
    project_summary: ProjectSummary | None = None


class MainAnalyzerService:
    """Unified service responsible for all conversational insight extraction and summarization."""

    def __init__(
        self,
        conversation_analysis_service: ConversationAnalysisService,
        knowledge_memory_service: KnowledgeMemoryService,
        summarization_service: SummarizationService,
    ) -> None:
        self._conversation_analysis_service = conversation_analysis_service
        self._knowledge_memory_service = knowledge_memory_service
        self._summarization_service = summarization_service

    def analyze(
        self,
        verified_transcript: VerifiedTranscript,
        session_metadata: SessionMetadata,
        knowledge_store: KnowledgeStore,
        existing_project_summary: ProjectSummary | None = None,
        on_facts_ready: Any = None,
        on_decisions_ready: Any = None,
        on_action_items_ready: Any = None,
        on_summary_ready: Any = None,
        on_progress: Any = None,
    ) -> SessionProcessingResult:
        """Runs conversation analysis, knowledge memory integration, and summarization with live streaming callbacks."""
        if on_progress:
            on_progress("analysis_started", {"turns_count": len(verified_transcript.turns)})

        # 1. Concurrently run conversation analysis and overview summarization
        with ThreadPoolExecutor(max_workers=2) as executor:
            analysis_future = executor.submit(
                self._conversation_analysis_service.analyze_conversation,
                verified_transcript,
                session_metadata,
                on_facts_ready=on_facts_ready,
                on_decisions_ready=on_decisions_ready,
                on_action_items_ready=on_action_items_ready,
            )

            def run_overview() -> tuple[str, list[str]]:
                try:
                    ov, hl = self._summarization_service.generate_session_overview(
                        verified_transcript,
                        session_metadata,
                    )
                    return ov, hl
                except Exception as exc:
                    print(f"[!] Warning: Session overview generation failed ({exc}); using fallback highlights.")
                    fallback_hl = [turn.text for turn in verified_transcript.turns[:6]]
                    fallback_ov = f"Session notes and discussion for {session_metadata.session_title}."
                    return fallback_ov, fallback_hl

            overview_future = executor.submit(run_overview)

            analysis_report = analysis_future.result()
            overview, highlights = overview_future.result()

        # 2. Assemble executive session summary
        session_summary = self._summarization_service.assemble_session_summary(
            overview=overview,
            discussion_highlights=highlights,
            analysis_report=analysis_report,
            session_metadata=session_metadata,
        )
        if on_summary_ready:
            try:
                on_summary_ready(session_summary)
            except Exception as exc:
                print(f"[!] Warning: on_summary_ready callback error: {exc}")

        if on_progress:
            on_progress("summary_ready", {"overview": overview, "highlights_count": len(highlights)})

        # 3. Integrate insights into the knowledge memory graph
        if on_progress:
            on_progress("integrating_memory", {})

        try:
            updated_knowledge_store = self._knowledge_memory_service.integrate_session_insights(
                analysis_report,
                knowledge_store,
            )
        except Exception as exc:
            print(f"[!] Warning: Knowledge memory integration error: {exc}; preserving current store.")
            updated_knowledge_store = knowledge_store

        # 4. Optionally roll up into project-level summary
        project_summary = None
        if session_metadata.project_id:
            try:
                project_summary = self._summarization_service.update_project_summary(
                    project_id=session_metadata.project_id,
                    project_title=session_metadata.session_title,
                    session_summary=session_summary,
                    existing_summary=existing_project_summary,
                )
            except Exception as exc:
                print(f"[!] Warning: Project summary update error: {exc}")

        if on_progress:
            on_progress("analysis_completed", {})

        return SessionProcessingResult(
            verified_transcript=verified_transcript,
            analysis_report=analysis_report,
            knowledge_store=updated_knowledge_store,
            session_summary=session_summary,
            project_summary=project_summary,
        )
