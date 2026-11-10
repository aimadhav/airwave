from typing import Any

from ai_layer.common.config_loader import AILayerConfig, load_config
from ai_layer.common.llm_provider import LLMProvider
from ai_layer.common.provider_factory import create_llm_provider, create_stt_provider
from ai_layer.common.stt_provider import STTProvider
from ai_layer.conversation_analysis.main_analyzer import (
    MainAnalyzerService,
    SessionProcessingResult,
)
from ai_layer.conversation_analysis.service import ConversationAnalysisService
from ai_layer.knowledge_memory.service import KnowledgeMemoryService
from ai_layer.models.analysis_models import (
    ActionItem,
    AnalysisReport,
    ExtractedDecision,
    ExtractedFact,
)
from datetime import datetime, timezone
from ai_layer.models.base_models import SessionMetadata
from ai_layer.models.knowledge_models import KnowledgePayload, KnowledgeStore
from ai_layer.models.query_models import (
    QueryAnswer,
    SearchRequest,
    SearchResponse,
    UserQuery,
)
from ai_layer.models.speaker_models import SpeakerProfile, VerifiedTranscript
from ai_layer.models.summary_models import ProjectSummary, SessionSummary
from ai_layer.models.transcription_models import (
    AudioClip,
    AudioClipPayload,
    STTTranscriptOutput,
)
from ai_layer.query_answering.service import UserQueryService
from ai_layer.speaker_verification.service import SpeakerVerificationService
from ai_layer.storage.factory import create_repository
from ai_layer.storage.sqlite_repository import SQLiteRepository
from ai_layer.summarization.service import SummarizationService
from ai_layer.transcription.service import STTTranscriptionService
from ai_layer.transcription.transcript_parser import TranscriptParser


class AILayerOrchestrator:
    """Orchestrates speech-to-text transcription and downstream conversation analysis.

    Supports two primary modes of operation:
      1. Audio Mode: Takes segmented audio clips -> STT -> Speaker Verification -> Main Analyzer
      2. Direct Transcript Mode: Takes pasted/raw text or transcript object -> Main Analyzer
    """

    def __init__(
        self,
        config: AILayerConfig,
        llm_provider: LLMProvider | None = None,
        stt_provider: STTProvider | None = None,
        knowledge_store: KnowledgeStore | None = None,
        repository: SQLiteRepository | None = None,
    ) -> None:
        self._config = config
        llm = llm_provider or create_llm_provider(config)
        stt = stt_provider or create_stt_provider(config)

        # STT Transcription Layer
        self._transcription_service = STTTranscriptionService(stt, config.transcription)

        # Speaker Verification Sublayer
        self._speaker_verification_service = SpeakerVerificationService(
            llm, config.speaker_verification
        )

        # Conversation Analysis & Summarization Sublayers
        self._conversation_analysis_service = ConversationAnalysisService(
            llm, config.conversation_analysis
        )
        self._knowledge_memory_service = KnowledgeMemoryService(llm, config.knowledge_memory)
        self._summarization_service = SummarizationService(llm, config.summarization)

        # Main Analyzer Service (Decoupled Analysis Stack)
        self._main_analyzer = MainAnalyzerService(
            conversation_analysis_service=self._conversation_analysis_service,
            knowledge_memory_service=self._knowledge_memory_service,
            summarization_service=self._summarization_service,
        )

        # Query Answering & Utilities
        self._query_service = UserQueryService(llm, config.query_answering)
        self._transcript_parser = TranscriptParser()
        self._knowledge_store = knowledge_store or KnowledgeStore()
        self._repository = repository

    @classmethod
    def from_config_file(
        cls,
        config_file_path: str = "configs/ai_layer_config.yaml",
        llm_provider: LLMProvider | None = None,
        stt_provider: STTProvider | None = None,
        repository: SQLiteRepository | None = None,
        enable_storage: bool = False,
    ) -> "AILayerOrchestrator":
        config = load_config(config_file_path)
        repo = repository
        if repo is None and enable_storage:
            repo = create_repository(config)
        return cls(config, llm_provider, stt_provider, repository=repo)

    # -------------------------------------------------------------------------
    # Layer 1: STT Transcription Layer
    # -------------------------------------------------------------------------
    def transcribe_audio_session(
        self,
        session_id: str,
        clips: list[AudioClip],
    ) -> STTTranscriptOutput:
        """Runs the STT layer exclusively: processes audio clips into stitched turns."""
        return self._transcription_service.process_session_clips(
            session_id=session_id,
            clips=clips,
        )

    # -------------------------------------------------------------------------
    # Layer 2: Speaker Verification Layer
    # -------------------------------------------------------------------------
    def verify_speakers(
        self,
        transcript: STTTranscriptOutput,
        known_speakers: list[SpeakerProfile],
    ) -> VerifiedTranscript:
        """Verifies and resolves speaker identities based on conversational cues."""
        return self._speaker_verification_service.verify_speakers(
            transcript=transcript,
            registered_profiles=known_speakers,
        )

    # -------------------------------------------------------------------------
    # Main Analyzer Layer: Operates directly on transcripts
    # -------------------------------------------------------------------------
    def analyze_transcript(
        self,
        transcript_input: str | STTTranscriptOutput | VerifiedTranscript,
        session_metadata: SessionMetadata,
        known_speakers: list[SpeakerProfile] | None = None,
        existing_project_summary: ProjectSummary | None = None,
        workspace_id: str | None = None,
        audio_path: str | None = None,
        on_progress: Any = None,
    ) -> SessionProcessingResult:
        """Main Analyzer: extracts insights, updates memory, and generates summaries with progressive saving.

        Accepts:
          - A raw pasted string transcript (parsed automatically via TranscriptParser)
          - An STTTranscriptOutput (verified via SpeakerVerificationService)
          - An already verified VerifiedTranscript
        """
        session_id = session_metadata.session_id
        ws_id = workspace_id or "default"

        # Step 1: Initialize session record immediately in repository if configured
        if self._repository:
            try:
                self._repository.save_session_metadata(
                    workspace_id=ws_id,
                    session_metadata=session_metadata,
                    audio_path=audio_path,
                    status="processing",
                    progress={"stage": "parsing_transcript", "percent": 15},
                )
            except Exception as exc:
                print(f"[!] Warning: Initial session save warning: {exc}")

        if isinstance(transcript_input, VerifiedTranscript):
            verified_transcript = transcript_input
        elif isinstance(transcript_input, STTTranscriptOutput):
            if on_progress:
                on_progress("verifying_speakers", {"turns_count": len(transcript_input.turns)})
            verified_transcript = self.verify_speakers(
                transcript=transcript_input,
                known_speakers=known_speakers or [],
            )
        elif isinstance(transcript_input, str):
            if on_progress:
                on_progress("parsing_transcript", {})
            verified_transcript = self._transcript_parser.parse_to_verified_transcript(
                text=transcript_input,
                session_id=session_metadata.session_id,
                known_speakers=known_speakers,
            )
        else:
            raise TypeError(
                f"Unsupported transcript input type: {type(transcript_input).__name__}. "
                "Expected str, STTTranscriptOutput, or VerifiedTranscript."
            )

        # Step 2: Ensure participants and speaker profiles are synchronized
        discovered_speakers = list(known_speakers or [])
        extracted_names = list(dict.fromkeys(turn.speaker_name for turn in verified_transcript.turns if turn.speaker_name))
        for name in extracted_names:
            if not any(k.canonical_name.lower() == name.lower() for k in discovered_speakers):
                normalized_id = "".join(c for c in name.lower().replace(" ", "_") if c.isalnum() or c == "_")
                spk_id = f"spk_{normalized_id}" if normalized_id else f"spk_{len(discovered_speakers)+1}"
                discovered_speakers.append(
                    SpeakerProfile(
                        speaker_id=spk_id,
                        canonical_name=name,
                        aliases=[name],
                        base_confidence=0.9,
                    )
                )

        if self._repository:
            try:
                self._repository.upsert_speakers(discovered_speakers)
            except Exception as exc:
                print(f"[!] Warning: Speaker upsert error: {exc}")

        if extracted_names:
            if not session_metadata.participant_ids or session_metadata.participant_ids == ["Speaker_1"]:
                session_metadata.participant_ids = extracted_names
            else:
                for p in extracted_names:
                    if p not in session_metadata.participant_ids:
                        session_metadata.participant_ids.append(p)

        # Step 3: Immediately persist transcript turns & reassignments
        if self._repository:
            try:
                self._repository.save_transcript(session_id, verified_transcript)
                self._repository.update_session_status(
                    session_id=session_id,
                    status="processing",
                    progress_data={"stage": "extracting_insights", "percent": 40, "transcript_saved": True},
                )
            except Exception as exc:
                print(f"[!] Warning: Progressive transcript save error: {exc}")

        if on_progress:
            on_progress("transcript_ready", {"turns_count": len(verified_transcript.turns)})

        # Step 3: Set up progressive callbacks for immediate persistence
        def _on_facts_ready(facts: list[ExtractedFact]) -> None:
            if self._repository:
                try:
                    self._repository.save_facts(session_id, facts)
                except Exception as exc:
                    print(f"[!] Warning: Progressive facts save error: {exc}")
            if on_progress:
                on_progress("facts_ready", {"count": len(facts)})

        def _on_decisions_ready(decisions: list[ExtractedDecision]) -> None:
            if self._repository:
                try:
                    self._repository.save_decisions(session_id, decisions)
                except Exception as exc:
                    print(f"[!] Warning: Progressive decisions save error: {exc}")
            if on_progress:
                on_progress("decisions_ready", {"count": len(decisions)})

        def _on_action_items_ready(action_items: list[ActionItem]) -> None:
            if self._repository:
                try:
                    self._repository.save_action_items(session_id, action_items)
                except Exception as exc:
                    print(f"[!] Warning: Progressive action items save error: {exc}")
            if on_progress:
                on_progress("action_items_ready", {"count": len(action_items)})

        def _on_summary_ready(summary: SessionSummary) -> None:
            if self._repository:
                try:
                    self._repository.save_session_summary(session_id, summary)
                except Exception as exc:
                    print(f"[!] Warning: Progressive summary save error: {exc}")
            if on_progress:
                on_progress("summary_ready", {"overview": summary.overview})

        # Step 4: Run Main Analyzer
        if self._repository and not self._knowledge_store.records:
            try:
                loaded_ks = self._repository.load_knowledge_store(ws_id)
                if loaded_ks and loaded_ks.records:
                    self._knowledge_store = loaded_ks
            except Exception as exc:
                print(f"[!] Warning: Knowledge store loading error: {exc}")

        result = self._main_analyzer.analyze(
            verified_transcript=verified_transcript,
            session_metadata=session_metadata,
            knowledge_store=self._knowledge_store,
            existing_project_summary=existing_project_summary,
            on_facts_ready=_on_facts_ready,
            on_decisions_ready=_on_decisions_ready,
            on_action_items_ready=_on_action_items_ready,
            on_summary_ready=_on_summary_ready,
            on_progress=on_progress,
        )
        self._knowledge_store = result.knowledge_store

        # Step 5: Finalize status & knowledge store in repository
        if self._repository:
            try:
                self._repository.save_session_result(
                    workspace_id=ws_id,
                    session_metadata=session_metadata,
                    result=result,
                    audio_path=audio_path,
                )
                self._repository.update_session_status(
                    session_id=session_id,
                    status="completed",
                    progress_data={"stage": "completed", "percent": 100},
                )
            except Exception as exc:
                print(f"[!] Warning: Final session save error: {exc}")

        return result

    # -------------------------------------------------------------------------
    # End-to-End Convenience Pipeline (STT Layer + Main Analyzer)
    # -------------------------------------------------------------------------
    def process_audio_session(
        self,
        session_metadata: SessionMetadata,
        clips: list[AudioClip],
        known_speakers: list[SpeakerProfile],
        existing_project_summary: ProjectSummary | None = None,
        workspace_id: str | None = None,
        audio_path: str | None = None,
        on_progress: Any = None,
    ) -> SessionProcessingResult:
        """End-to-end processing: transcribes audio clips then runs the Main Analyzer."""
        if on_progress:
            on_progress("transcribing_audio", {"clips_count": len(clips)})

        if self._repository:
            try:
                self._repository.save_session_metadata(
                    workspace_id=workspace_id or "default",
                    session_metadata=session_metadata,
                    audio_path=audio_path,
                    status="processing",
                    progress={"stage": "transcribing_audio", "percent": 20},
                )
            except Exception as exc:
                print(f"[!] Warning: Audio session init save error: {exc}")

        stt_transcript = self.transcribe_audio_session(
            session_id=session_metadata.session_id,
            clips=clips,
        )
        return self.analyze_transcript(
            transcript_input=stt_transcript,
            session_metadata=session_metadata,
            known_speakers=known_speakers,
            existing_project_summary=existing_project_summary,
            workspace_id=workspace_id,
            audio_path=audio_path,
            on_progress=on_progress,
        )

    # -------------------------------------------------------------------------
    # RAG Query & Memory Access
    # -------------------------------------------------------------------------
    def query_session_memory(
        self,
        query_text: str,
        transcripts: list[VerifiedTranscript],
        reports: list[AnalysisReport],
        target_session_id: str | None = None,
    ) -> QueryAnswer:
        return self._query_service.answer_query(
            UserQuery(query_text=query_text, target_session_id=target_session_id),
            transcripts,
            reports,
        )

    @property
    def knowledge_store(self) -> KnowledgeStore:
        return self._knowledge_store

    @property
    def repository(self) -> SQLiteRepository | None:
        return self._repository

    @property
    def main_analyzer(self) -> MainAnalyzerService:
        return self._main_analyzer

    def save_session_result(
        self,
        workspace_id: str,
        session_metadata: SessionMetadata,
        result: SessionProcessingResult,
        audio_path: str | None = None,
        async_embeddings: bool = False,
    ) -> None:
        if self._repository is None:
            raise RuntimeError("Repository is not configured on this orchestrator.")
        self._repository.save_session_result(
            workspace_id=workspace_id,
            session_metadata=session_metadata,
            result=result,
            audio_path=audio_path,
            async_embeddings=async_embeddings,
        )

    def query_workspace(
        self,
        query_text: str,
        workspace_id: str,
        target_session_id: str | None = None,
    ) -> QueryAnswer:
        if self._repository is None:
            raise RuntimeError("Repository is not configured on this orchestrator.")

        # If a specific target session is requested, first search within that session
        if target_session_id:
            transcript = self._repository.load_verified_transcript(target_session_id)
            report = self._repository.load_analysis_report(target_session_id)
            if transcript or report:
                ans = self.query_session_memory(
                    query_text=query_text,
                    transcripts=[transcript] if transcript else [],
                    reports=[report] if report else [],
                    target_session_id=target_session_id,
                )
                if ans.is_found:
                    meta = self._repository.load_session_metadata(target_session_id)
                    stitle = meta["title"] if meta and meta.get("title") else target_session_id
                    for c in ans.citations:
                        if not c.session_title:
                            c.session_title = stitle
                    return ans

        # If not found in target session or searching entire workspace:
        all_sids = self._repository.list_session_ids(workspace_id)
        transcripts: list[VerifiedTranscript] = []
        reports: list[AnalysisReport] = []
        session_title_map: dict[str, str] = {}
        for sid in all_sids:
            t = self._repository.load_verified_transcript(sid)
            if t:
                transcripts.append(t)
            r = self._repository.load_analysis_report(sid)
            if r:
                reports.append(r)
            meta = self._repository.load_session_metadata(sid)
            if meta and meta.get("title"):
                session_title_map[sid] = meta["title"]

        ans = self.query_session_memory(
            query_text=query_text,
            transcripts=transcripts,
            reports=reports,
            target_session_id=None,
        )

        for c in ans.citations:
            if c.session_id and not c.session_title:
                c.session_title = session_title_map.get(c.session_id, c.session_id)

        # If user asked with a target_session_id but result was found across workspace, note it gracefully
        if target_session_id and ans.is_found and ans.citations:
            cited_sessions = {c.session_id for c in ans.citations if c.session_id}
            if target_session_id not in cited_sessions:
                primary_title = ans.citations[0].session_title or ans.citations[0].session_id
                ans.answer_text = f"(Found in '{primary_title}'): {ans.answer_text}"

        return ans

    # -------------------------------------------------------------------------
    # Canonical Inter-Layer Contract APIs
    # -------------------------------------------------------------------------

    def process_dsp_payload(
        self,
        payload: AudioClipPayload,
        session_title: str | None = None,
        known_speakers: list[SpeakerProfile] | None = None,
        workspace_id: str = "default",
        project_id: str | None = None,
        on_progress: Any = None,
    ) -> SessionProcessingResult:
        """Processes Contract 1 AudioClipPayload received directly from the DSP Layer."""
        metadata = SessionMetadata(
            session_id=payload.session_id,
            session_title=session_title or f"Session {payload.session_id}",
            recorded_at=datetime.now(timezone.utc).isoformat(),
            participant_ids=list(
                {s.preliminary_speaker_id for s in payload.segments if s.preliminary_speaker_id}
            ),
            project_id=project_id,
        )
        audio_clips = payload.to_audio_clips()
        return self.process_audio_session(
            session_metadata=metadata,
            clips=audio_clips,
            known_speakers=known_speakers or [],
            workspace_id=workspace_id,
            audio_path=payload.source_audio_path,
            on_progress=on_progress,
        )

    def export_knowledge_payload(
        self,
        session_id: str,
        workspace_id: str = "default",
    ) -> KnowledgePayload:
        """Exports Canonical Contract 2 KnowledgePayload for the Database Layer."""
        if self._repository is None:
            raise RuntimeError("Repository is not configured on this orchestrator.")
        return self._repository.export_knowledge_payload(
            session_id=session_id,
            workspace_id=workspace_id,
        )

    def search_contract_rag(
        self,
        request: SearchRequest,
    ) -> SearchResponse:
        """Executes Canonical Contract 3 hybrid search with provenance and ranking scores."""
        if self._repository is None:
            return SearchResponse()
        return self._repository.search_contract_rag(request)
