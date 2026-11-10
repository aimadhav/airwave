from datetime import datetime, timezone
import json
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ai_layer.common.embedding_provider import EmbeddingProvider
from ai_layer.models.analysis_models import (
    ActionItem,
    AnalysisReport,
    ExtractedDecision,
    ExtractedFact,
    UtteranceCitation,
)
from ai_layer.models.base_models import SessionMetadata, TimeRange
from ai_layer.models.knowledge_models import (
    ChangeLogEntry,
    ChangeLogPayload,
    ContentUnitPayload,
    EdgePayload,
    KnowledgePayload,
    KnowledgeRecord,
    KnowledgeRelationship,
    KnowledgeStore,
    UtterancePayload,
)
from ai_layer.models.query_models import (
    ConnectedEdge,
    ContractRetrievedResult,
    SearchRequest,
    SearchResponse,
    SourceUtteranceInfo,
)
from ai_layer.models.speaker_models import (
    SpeakerProfile,
    SpeakerReassignment,
    VerifiedTranscript,
    VerifiedTurn,
)
from ai_layer.models.summary_models import SessionSummary


class SQLiteRepository:
    """Production SQLite storage repository for AI Layer records, transcripts, and embeddings."""

    def __init__(self, db_path: str, embedding_provider: EmbeddingProvider | None = None) -> None:
        self._db_path = db_path
        self._embedding_provider = embedding_provider
        self._write_lock = threading.Lock()

        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS speakers (
                    speaker_id TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    aliases TEXT NOT NULL,
                    base_confidence REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    participant_ids TEXT NOT NULL,
                    project_id TEXT,
                    audio_path TEXT,
                    status TEXT DEFAULT 'completed',
                    progress_json TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    session_id TEXT PRIMARY KEY,
                    turns_json TEXT NOT NULL,
                    reassignments_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_reports (
                    session_id TEXT PRIMARY KEY,
                    facts_json TEXT NOT NULL,
                    decisions_json TEXT NOT NULL,
                    action_items_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    overview TEXT NOT NULL,
                    highlights_json TEXT NOT NULL,
                    decisions_summary_json TEXT NOT NULL,
                    action_items_summary_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_stores (
                    workspace_id TEXT PRIMARY KEY,
                    records_json TEXT NOT NULL,
                    change_log_json TEXT NOT NULL,
                    relationships_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS content_units (
                    unit_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    unit_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
            """)
            # Migrations for existing databases
            migrations = [
                "ALTER TABLE sessions ADD COLUMN participant_ids TEXT DEFAULT '[]'",
                "ALTER TABLE sessions ADD COLUMN status TEXT DEFAULT 'completed'",
                "ALTER TABLE sessions ADD COLUMN progress_json TEXT DEFAULT '{}'",
                "ALTER TABLE session_summaries ADD COLUMN highlights_json TEXT DEFAULT '[]'",
                "ALTER TABLE session_summaries ADD COLUMN decisions_summary_json TEXT DEFAULT '[]'",
                "ALTER TABLE session_summaries ADD COLUMN action_items_summary_json TEXT DEFAULT '[]'",
                "ALTER TABLE speakers ADD COLUMN canonical_name TEXT DEFAULT ''",
                "ALTER TABLE speakers ADD COLUMN aliases TEXT DEFAULT '[]'",
                "ALTER TABLE speakers ADD COLUMN base_confidence REAL DEFAULT 0.85",
            ]
            for m in migrations:
                try:
                    conn.execute(m)
                except sqlite3.OperationalError:
                    pass

            # Backfill legacy speaker columns if present
            try:
                conn.execute(
                    "UPDATE speakers SET canonical_name = display_name WHERE (canonical_name = '' OR canonical_name IS NULL) AND display_name IS NOT NULL"
                )
            except sqlite3.OperationalError:
                pass

            # Backfill legacy summary columns if present
            try:
                conn.execute(
                    "UPDATE session_summaries SET highlights_json = discussion_highlights WHERE (highlights_json = '[]' OR highlights_json IS NULL) AND discussion_highlights IS NOT NULL"
                )
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute(
                    "UPDATE session_summaries SET decisions_summary_json = decisions_summary WHERE (decisions_summary_json = '[]' OR decisions_summary_json IS NULL) AND decisions_summary IS NOT NULL"
                )
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute(
                    "UPDATE session_summaries SET action_items_summary_json = action_items_summary WHERE (action_items_summary_json = '[]' OR action_items_summary_json IS NULL) AND action_items_summary IS NOT NULL"
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()

    def upsert_speakers(self, speakers: list[SpeakerProfile]) -> None:
        with self._get_connection() as conn:
            for s in speakers:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO speakers (speaker_id, canonical_name, aliases, base_confidence)
                    VALUES (?, ?, ?, ?)
                    """,
                    (s.speaker_id, s.canonical_name, json.dumps(s.aliases), s.base_confidence),
                )
            conn.commit()

    def save_session_metadata(
        self,
        workspace_id: str,
        session_metadata: SessionMetadata,
        audio_path: str | None = None,
        status: str = "processing",
        progress: dict[str, Any] | None = None,
    ) -> None:
        """Persists initial session metadata and status immediately when ingestion starts."""
        with self._write_lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    session_id, workspace_id, title, recorded_at, participant_ids, project_id, audio_path, status, progress_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_metadata.session_id,
                    workspace_id,
                    session_metadata.session_title,
                    session_metadata.recorded_at,
                    json.dumps(session_metadata.participant_ids),
                    session_metadata.project_id,
                    audio_path,
                    status,
                    json.dumps(progress or {}),
                ),
            )
            conn.commit()

    def update_session_status(
        self,
        session_id: str,
        status: str,
        progress_data: dict[str, Any] | None = None,
    ) -> None:
        """Updates session status and live progress milestones."""
        with self._write_lock, self._get_connection() as conn:
            if progress_data is not None:
                conn.execute(
                    "UPDATE sessions SET status = ?, progress_json = ? WHERE session_id = ?",
                    (status, json.dumps(progress_data), session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET status = ? WHERE session_id = ?",
                    (status, session_id),
                )
            conn.commit()

    def load_session_status(self, session_id: str) -> dict[str, Any]:
        """Loads the current processing status and progress for a session."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT status, progress_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return {"status": "not_found", "progress": {}}
            progress = {}
            try:
                progress = json.loads(row["progress_json"] or "{}")
            except Exception:
                pass
            return {"status": row["status"] or "completed", "progress": progress}

    def load_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Loads session metadata including title, recorded_at, participants, status, etc."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT session_id, workspace_id, title, recorded_at, participant_ids, project_id, audio_path, status, progress_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            try:
                participants = json.loads(row["participant_ids"] or "[]")
            except Exception:
                participants = []
            try:
                progress = json.loads(row["progress_json"] or "{}")
            except Exception:
                progress = {}
            return {
                "session_id": row["session_id"],
                "workspace_id": row["workspace_id"],
                "title": row["title"],
                "recorded_at": row["recorded_at"],
                "participant_ids": participants,
                "project_id": row["project_id"],
                "audio_path": row["audio_path"],
                "status": row["status"] or "completed",
                "progress": progress,
            }

    def list_sessions_summary(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        """Lists sessions with metadata, ordered newest first (recorded_at DESC)."""
        with self._get_connection() as conn:
            if not workspace_id or workspace_id in ("all", "*", "default"):
                cursor = conn.execute(
                    "SELECT session_id, workspace_id, title, recorded_at, status, progress_json FROM sessions ORDER BY recorded_at DESC"
                )
            else:
                cursor = conn.execute(
                    "SELECT session_id, workspace_id, title, recorded_at, status, progress_json FROM sessions WHERE workspace_id = ? OR workspace_id IN ('ws_engineering', 'ws_default', 'default') ORDER BY recorded_at DESC",
                    (workspace_id,),
                )
            rows = cursor.fetchall()
            out = []
            for r in rows:
                try:
                    prog = json.loads(r["progress_json"] or "{}")
                except Exception:
                    prog = {}
                out.append({
                    "session_id": r["session_id"],
                    "workspace_id": r["workspace_id"],
                    "title": r["title"] or r["session_id"],
                    "recorded_at": r["recorded_at"],
                    "status": r["status"] or "completed",
                    "progress": prog,
                })
            return out

    def save_transcript(self, session_id: str, verified_transcript: VerifiedTranscript) -> None:
        """Persists verified transcript turns & reassignments immediately once ready."""
        turns_data = [
            {
                "turn_id": t.turn_id,
                "speaker_id": t.speaker_id,
                "speaker_name": t.speaker_name,
                "text": t.text,
                "start": t.time_range.start_time_seconds,
                "end": t.time_range.end_time_seconds,
            }
            for t in verified_transcript.turns
        ]
        reassignments_data = [
            {
                "preliminary_speaker_id": r.preliminary_speaker_id,
                "resolved_speaker_id": r.resolved_speaker_id,
                "resolved_name": r.resolved_name,
                "rationale": r.rationale,
            }
            for r in verified_transcript.reassignments
        ]
        with self._write_lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO transcripts (session_id, turns_json, reassignments_json)
                VALUES (?, ?, ?)
                """,
                (session_id, json.dumps(turns_data), json.dumps(reassignments_data)),
            )
            conn.commit()

    def save_facts(self, session_id: str, facts: list[ExtractedFact]) -> None:
        """Atomically saves/updates facts in analysis_reports without clobbering decisions or action items."""
        facts_data = [
            {
                "fact_id": f.fact_id,
                "statement": f.statement,
                "citation": {
                    "turn_id": f.citation.turn_id,
                    "speaker_name": f.citation.speaker_name,
                    "start": f.citation.time_range.start_time_seconds,
                    "end": f.citation.time_range.end_time_seconds,
                    "spoken_text": f.citation.spoken_text,
                }
                if f.citation
                else None,
            }
            for f in facts
        ]
        with self._write_lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT decisions_json, action_items_json FROM analysis_reports WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            decisions_json = row["decisions_json"] if row else "[]"
            action_items_json = row["action_items_json"] if row else "[]"

            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_reports (session_id, facts_json, decisions_json, action_items_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, json.dumps(facts_data), decisions_json, action_items_json),
            )
            conn.commit()

    def save_decisions(self, session_id: str, decisions: list[ExtractedDecision]) -> None:
        """Atomically saves/updates decisions in analysis_reports without clobbering facts or action items."""
        decisions_data = [
            {
                "decision_id": d.decision_id,
                "decision": d.decision,
                "agreed_by": d.agreed_by,
                "citation": {
                    "turn_id": d.citation.turn_id,
                    "speaker_name": d.citation.speaker_name,
                    "start": d.citation.time_range.start_time_seconds,
                    "end": d.citation.time_range.end_time_seconds,
                    "spoken_text": d.citation.spoken_text,
                }
                if d.citation
                else None,
            }
            for d in decisions
        ]
        with self._write_lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT facts_json, action_items_json FROM analysis_reports WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            facts_json = row["facts_json"] if row else "[]"
            action_items_json = row["action_items_json"] if row else "[]"

            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_reports (session_id, facts_json, decisions_json, action_items_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, facts_json, json.dumps(decisions_data), action_items_json),
            )
            conn.commit()

    def save_action_items(self, session_id: str, action_items: list[ActionItem]) -> None:
        """Atomically saves/updates action items in analysis_reports without clobbering facts or decisions."""
        tasks_data = [
            {
                "task_id": a.task_id,
                "task_description": a.task_description,
                "assignee_name": a.assignee_name,
                "deadline": a.deadline,
                "citation": {
                    "turn_id": a.citation.turn_id,
                    "speaker_name": a.citation.speaker_name,
                    "start": a.citation.time_range.start_time_seconds,
                    "end": a.citation.time_range.end_time_seconds,
                    "spoken_text": a.citation.spoken_text,
                }
                if a.citation
                else None,
            }
            for a in action_items
        ]
        with self._write_lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT facts_json, decisions_json FROM analysis_reports WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            facts_json = row["facts_json"] if row else "[]"
            decisions_json = row["decisions_json"] if row else "[]"

            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_reports (session_id, facts_json, decisions_json, action_items_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, facts_json, decisions_json, json.dumps(tasks_data)),
            )
            conn.commit()

    def save_session_summary(self, session_id: str, summary: SessionSummary) -> None:
        """Persists executive summary and discussion highlights."""
        with self._write_lock, self._get_connection() as conn:
            cols = {col[1] for col in conn.execute("PRAGMA table_info(session_summaries)").fetchall()}
            if "title" in cols:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO session_summaries (
                        session_id, title, overview, highlights_json, decisions_summary_json, action_items_summary_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        summary.title or "",
                        summary.overview,
                        json.dumps(summary.discussion_highlights),
                        json.dumps(summary.decisions_summary),
                        json.dumps(summary.action_items_summary),
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO session_summaries (
                        session_id, overview, highlights_json, decisions_summary_json, action_items_summary_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        summary.overview,
                        json.dumps(summary.discussion_highlights),
                        json.dumps(summary.decisions_summary),
                        json.dumps(summary.action_items_summary),
                    ),
                )
            conn.commit()

    def load_session_summary(self, session_id: str) -> SessionSummary | None:
        """Loads executive session summary and highlights."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT overview, highlights_json, decisions_summary_json, action_items_summary_json FROM session_summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            return SessionSummary(
                session_id=session_id,
                title="",
                overview=row["overview"],
                discussion_highlights=json.loads(row["highlights_json"] or "[]"),
                decisions_summary=json.loads(row["decisions_summary_json"] or "[]"),
                action_items_summary=json.loads(row["action_items_summary_json"] or "[]"),
            )

    def save_session_result(
        self,
        workspace_id: str,
        session_metadata: SessionMetadata,
        result: Any,
        audio_path: str | None = None,
        async_embeddings: bool = False,
    ) -> None:
        with self._write_lock, self._get_connection() as conn:
            # 1. Upsert session
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (session_id, workspace_id, title, recorded_at, participant_ids, project_id, audio_path, status, progress_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', '{"completed": true}')
                """,
                (
                    session_metadata.session_id,
                    workspace_id,
                    session_metadata.session_title,
                    session_metadata.recorded_at,
                    json.dumps(session_metadata.participant_ids),
                    session_metadata.project_id,
                    audio_path,
                ),
            )

            # 2. Save transcript
            turns_data = [
                {
                    "turn_id": t.turn_id,
                    "speaker_id": t.speaker_id,
                    "speaker_name": t.speaker_name,
                    "text": t.text,
                    "start": t.time_range.start_time_seconds,
                    "end": t.time_range.end_time_seconds,
                }
                for t in result.verified_transcript.turns
            ]
            reassignments_data = [
                {
                    "preliminary_speaker_id": r.preliminary_speaker_id,
                    "resolved_speaker_id": r.resolved_speaker_id,
                    "resolved_name": r.resolved_name,
                    "rationale": r.rationale,
                }
                for r in result.verified_transcript.reassignments
            ]
            conn.execute(
                """
                INSERT OR REPLACE INTO transcripts (session_id, turns_json, reassignments_json)
                VALUES (?, ?, ?)
                """,
                (session_metadata.session_id, json.dumps(turns_data), json.dumps(reassignments_data)),
            )

            # 3. Save analysis report
            facts_data = [
                {
                    "fact_id": f.fact_id,
                    "statement": f.statement,
                    "citation": {
                        "turn_id": f.citation.turn_id,
                        "speaker_name": f.citation.speaker_name,
                        "start": f.citation.time_range.start_time_seconds,
                        "end": f.citation.time_range.end_time_seconds,
                        "spoken_text": f.citation.spoken_text,
                    },
                }
                for f in result.analysis_report.facts
            ]
            decisions_data = [
                {
                    "decision_id": d.decision_id,
                    "decision": d.decision,
                    "agreed_by": d.agreed_by,
                    "citation": {
                        "turn_id": d.citation.turn_id,
                        "speaker_name": d.citation.speaker_name,
                        "start": d.citation.time_range.start_time_seconds,
                        "end": d.citation.time_range.end_time_seconds,
                        "spoken_text": d.citation.spoken_text,
                    }
                    if d.citation
                    else None,
                }
                for d in result.analysis_report.decisions
            ]
            tasks_data = [
                {
                    "task_id": a.task_id,
                    "task_description": a.task_description,
                    "assignee_name": a.assignee_name,
                    "deadline": a.deadline,
                    "citation": {
                        "turn_id": a.citation.turn_id,
                        "speaker_name": a.citation.speaker_name,
                        "start": a.citation.time_range.start_time_seconds,
                        "end": a.citation.time_range.end_time_seconds,
                        "spoken_text": a.citation.spoken_text,
                    }
                    if a.citation
                    else None,
                }
                for a in result.analysis_report.action_items
            ]
            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_reports (session_id, facts_json, decisions_json, action_items_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_metadata.session_id, json.dumps(facts_data), json.dumps(decisions_data), json.dumps(tasks_data)),
            )

            # 4. Save knowledge store
            ks = result.knowledge_store
            records_data = [
                {
                    "record_id": r.record_id,
                    "record_type": r.record_type,
                    "statement": r.statement,
                    "session_id": r.session_id,
                    "source_turn_id": r.source_turn_id,
                    "status": r.status,
                    "related_record_ids": r.related_record_ids,
                }
                for r in ks.records
            ]
            logs_data = [
                {
                    "entry_id": l.entry_id,
                    "record_id": l.record_id,
                    "action": l.action,
                    "session_id": l.session_id,
                    "details": l.details,
                }
                for l in ks.change_log
            ]
            rels_data = [
                {
                    "relationship_id": rel.relationship_id,
                    "source_record_id": rel.source_record_id,
                    "target_record_id": rel.target_record_id,
                    "relationship_type": rel.relationship_type,
                    "rationale": rel.rationale,
                }
                for rel in ks.relationships
            ]
            conn.execute(
                """
                INSERT OR REPLACE INTO knowledge_stores (workspace_id, records_json, change_log_json, relationships_json)
                VALUES (?, ?, ?, ?)
                """,
                (workspace_id, json.dumps(records_data), json.dumps(logs_data), json.dumps(rels_data)),
            )

            # 5. Save session summary if present
            if getattr(result, "session_summary", None):
                summary = result.session_summary
                summary_cols = {c[1] for c in conn.execute("PRAGMA table_info(session_summaries)").fetchall()}
                if "title" in summary_cols:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO session_summaries (
                            session_id, title, overview, highlights_json, decisions_summary_json, action_items_summary_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_metadata.session_id,
                            summary.title or session_metadata.session_title,
                            summary.overview,
                            json.dumps(summary.discussion_highlights),
                            json.dumps(summary.decisions_summary),
                            json.dumps(summary.action_items_summary),
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO session_summaries (
                            session_id, overview, highlights_json, decisions_summary_json, action_items_summary_json
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            session_metadata.session_id,
                            summary.overview,
                            json.dumps(summary.discussion_highlights),
                            json.dumps(summary.decisions_summary),
                            json.dumps(summary.action_items_summary),
                        ),
                    )

            conn.commit()

        # 5. Embed and save content units (optional semantic index)
        if self._embedding_provider:
            turns = list(result.verified_transcript.turns)
            facts = list(result.analysis_report.facts)
            if async_embeddings:
                worker = threading.Thread(
                    target=self._embed_and_save_content_units,
                    args=(workspace_id, session_metadata.session_id, turns, facts),
                    daemon=True,
                )
                worker.start()
            else:
                self._embed_and_save_content_units(
                    workspace_id, session_metadata.session_id, turns, facts
                )

    def _embed_and_save_content_units(
        self,
        workspace_id: str,
        session_id: str,
        turns: list[Any],
        facts: list[Any],
    ) -> None:
        if not self._embedding_provider:
            return

        try:
            turn_items = [
                (f"unit_{session_id}_turn_{idx}", "turn", t.text)
                for idx, t in enumerate(turns)
                if getattr(t, "text", None) and t.text.strip()
            ]
            fact_items = [
                (f"unit_{session_id}_{f.fact_id}", "fact", f.statement)
                for f in facts
                if getattr(f, "statement", None) and f.statement.strip()
            ]
            all_items = turn_items + fact_items
            if not all_items:
                return

            all_texts = [item[2] for item in all_items]

            if hasattr(self._embedding_provider, "embed_texts"):
                embeddings = self._embedding_provider.embed_texts(all_texts)
            else:
                embeddings = [self._embedding_provider.embed_text(t) for t in all_texts]

            with self._get_connection() as conn:
                cols = {c[1] for c in conn.execute("PRAGMA table_info(content_units)").fetchall()}
                for (unit_id, unit_type, content), emb in zip(all_items, embeddings):
                    if emb:
                        if "content_id" in cols:
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO content_units (
                                    content_id, workspace_id, session_id, type, content, embedding, last_reaffirmed_at
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    unit_id,
                                    workspace_id,
                                    session_id,
                                    unit_type,
                                    content,
                                    json.dumps(emb),
                                    datetime.now(timezone.utc).isoformat(),
                                ),
                            )
                        else:
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO content_units (unit_id, workspace_id, session_id, unit_type, content, embedding_json)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (unit_id, workspace_id, session_id, unit_type, content, json.dumps(emb)),
                            )
                conn.commit()
        except Exception as exc:
            print(f"[!] Warning: Semantic embedding generation skipped ({exc}); continuing session save.")

    def list_session_ids(self, workspace_id: str | None = None) -> list[str]:
        with self._get_connection() as conn:
            if not workspace_id or workspace_id in ("all", "*", "default"):
                cursor = conn.execute(
                    "SELECT session_id FROM sessions ORDER BY recorded_at DESC"
                )
            else:
                cursor = conn.execute(
                    "SELECT session_id FROM sessions WHERE workspace_id = ? OR workspace_id IN ('ws_engineering', 'ws_default', 'default') ORDER BY recorded_at DESC",
                    (workspace_id,),
                )
            return [row["session_id"] for row in cursor.fetchall()]

    def load_verified_transcript(self, session_id: str) -> VerifiedTranscript | None:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT turns_json, reassignments_json FROM transcripts WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            turns_data = json.loads(row["turns_json"])
            reassignments_data = json.loads(row["reassignments_json"])

            turns = [
                VerifiedTurn(
                    turn_id=t["turn_id"],
                    speaker_id=t["speaker_id"],
                    speaker_name=t["speaker_name"],
                    text=t["text"],
                    time_range=TimeRange(float(t["start"]), float(t["end"])),
                )
                for t in turns_data
            ]
            reassignments = [
                SpeakerReassignment(
                    preliminary_speaker_id=r["preliminary_speaker_id"],
                    resolved_speaker_id=r["resolved_speaker_id"],
                    resolved_name=r["resolved_name"],
                    rationale=r.get("rationale", ""),
                )
                for r in reassignments_data
            ]
            return VerifiedTranscript(session_id=session_id, turns=turns, reassignments=reassignments)

    def load_analysis_report(self, session_id: str) -> AnalysisReport | None:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT facts_json, decisions_json, action_items_json FROM analysis_reports WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            facts = [
                ExtractedFact(
                    fact_id=f["fact_id"],
                    statement=f["statement"],
                    citation=UtteranceCitation(
                        turn_id=f["citation"]["turn_id"],
                        speaker_name=f["citation"]["speaker_name"],
                        time_range=TimeRange(
                            float(f["citation"]["start"]), float(f["citation"]["end"])
                        ),
                        spoken_text=f["citation"]["spoken_text"],
                    ),
                )
                for f in json.loads(row["facts_json"])
            ]
            decisions = [
                ExtractedDecision(
                    decision_id=d["decision_id"],
                    decision=d["decision"],
                    agreed_by=d.get("agreed_by", []),
                    citation=UtteranceCitation(
                        turn_id=d["citation"]["turn_id"],
                        speaker_name=d["citation"]["speaker_name"],
                        time_range=TimeRange(
                            float(d["citation"]["start"]), float(d["citation"]["end"])
                        ),
                        spoken_text=d["citation"]["spoken_text"],
                    )
                    if d.get("citation")
                    else None,
                )
                for d in json.loads(row["decisions_json"])
            ]
            action_items = [
                ActionItem(
                    task_id=a["task_id"],
                    task_description=a["task_description"],
                    assignee_name=a["assignee_name"],
                    deadline=a.get("deadline"),
                    citation=UtteranceCitation(
                        turn_id=a["citation"]["turn_id"],
                        speaker_name=a["citation"]["speaker_name"],
                        time_range=TimeRange(
                            float(a["citation"]["start"]), float(a["citation"]["end"])
                        ),
                        spoken_text=a["citation"]["spoken_text"],
                    )
                    if a.get("citation")
                    else None,
                )
                for a in json.loads(row["action_items_json"])
            ]
            return AnalysisReport(
                session_id=session_id,
                facts=facts,
                decisions=decisions,
                action_items=action_items,
            )

    def load_knowledge_store(self, workspace_id: str) -> KnowledgeStore:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT records_json, change_log_json, relationships_json FROM knowledge_stores WHERE workspace_id = ?",
                (workspace_id,),
            )
            row = cursor.fetchone()
            if not row:
                return KnowledgeStore()

            records = [
                KnowledgeRecord(
                    record_id=r["record_id"],
                    record_type=r["record_type"],
                    statement=r["statement"],
                    session_id=r["session_id"],
                    source_turn_id=r["source_turn_id"],
                    status=r["status"],
                    related_record_ids=r.get("related_record_ids", []),
                )
                for r in json.loads(row["records_json"])
            ]
            change_log = [
                ChangeLogEntry(
                    entry_id=l["entry_id"],
                    record_id=l["record_id"],
                    action=l["action"],
                    session_id=l["session_id"],
                    details=l.get("details", ""),
                )
                for l in json.loads(row["change_log_json"])
            ]
            relationships = [
                KnowledgeRelationship(
                    relationship_id=rel["relationship_id"],
                    source_record_id=rel["source_record_id"],
                    target_record_id=rel["target_record_id"],
                    relationship_type=rel["relationship_type"],
                    rationale=rel.get("rationale", ""),
                )
                for rel in json.loads(row["relationships_json"])
            ]
            return KnowledgeStore(records=records, change_log=change_log, relationships=relationships)

    def search_content_units(
        self,
        workspace_id: str,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict]:
        if not self._embedding_provider:
            return []

        query_emb = self._embedding_provider.embed_text(query_text)
        with self._get_connection() as conn:
            cols = {c[1] for c in conn.execute("PRAGMA table_info(content_units)").fetchall()}
            if "content_id" in cols:
                cursor = conn.execute(
                    "SELECT content_id AS unit_id, session_id, type AS unit_type, content, embedding AS embedding_json FROM content_units WHERE workspace_id = ?",
                    (workspace_id,),
                )
            else:
                cursor = conn.execute(
                    "SELECT unit_id, session_id, unit_type, content, embedding_json FROM content_units WHERE workspace_id = ?",
                    (workspace_id,),
                )
            rows = cursor.fetchall()

        scored: list[tuple[float, dict]] = []
        for r in rows:
            emb = json.loads(r["embedding_json"])
            dot = sum(a * b for a, b in zip(query_emb, emb))
            norm_a = math.sqrt(sum(a * a for a in query_emb)) or 1.0
            norm_b = math.sqrt(sum(b * b for b in emb)) or 1.0
            sim = dot / (norm_a * norm_b)
            scored.append((
                sim,
                {
                    "unit_id": r["unit_id"],
                    "session_id": r["session_id"],
                    "unit_type": r["unit_type"],
                    "content": r["content"],
                    "similarity": sim,
                },
            ))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def export_knowledge_payload(
        self, session_id: str, workspace_id: str = "default"
    ) -> KnowledgePayload:
        """Exports the canonical Contract 2 KnowledgePayload for the Database Layer."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT session_id, workspace_id, project_id, audio_path, recorded_at FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            s_row = cur.fetchone()
            session_data = {
                "session_id": session_id,
                "workspace_id": s_row["workspace_id"] if s_row else workspace_id,
                "project_id": s_row["project_id"] if s_row else None,
                "audio_path": s_row["audio_path"] if s_row else "",
                "recorded_at": s_row["recorded_at"] if s_row else datetime.now(timezone.utc).isoformat(),
            }

            utterances: list[UtterancePayload] = []
            cur = conn.execute("SELECT turns_json FROM transcripts WHERE session_id = ?", (session_id,))
            t_row = cur.fetchone()
            if t_row and t_row["turns_json"]:
                turns = json.loads(t_row["turns_json"])
                for t in turns:
                    utterances.append(
                        UtterancePayload(
                            utterance_id=t.get("turn_id", f"utt_{len(utterances)+1:03d}"),
                            session_id=session_id,
                            speaker_id=t.get("speaker_id", "SPK_UNKNOWN"),
                            start_ts=float(t.get("start_time", 0.0)),
                            end_ts=float(t.get("end_time", 0.0)),
                            raw_text=t.get("spoken_text", ""),
                            identity_evidence={
                                "signals": [
                                    {"type": "speaker_name", "value": t.get("speaker_name", "")},
                                    {"type": "confidence", "value": float(t.get("confidence", 1.0))},
                                ]
                            },
                        )
                    )

            content_units: list[ContentUnitPayload] = []
            cur = conn.execute(
                "SELECT unit_id, workspace_id, session_id, unit_type, content, embedding_json FROM content_units WHERE session_id = ?",
                (session_id,),
            )
            for cu in cur.fetchall():
                content_units.append(
                    ContentUnitPayload(
                        content_id=cu["unit_id"],
                        workspace_id=cu["workspace_id"],
                        project_id=session_data.get("project_id"),
                        type=cu["unit_type"],
                        content=cu["content"],
                        embedding=json.loads(cu["embedding_json"]) if cu["embedding_json"] else [],
                        status="active",
                        decay_rate=0.01,
                        last_reaffirmed_at=session_data["recorded_at"],
                        source_utterance_ids=[],
                        source_content_ids=[],
                    )
                )

            edges: list[EdgePayload] = []
            change_logs: list[ChangeLogPayload] = []
            cur = conn.execute(
                "SELECT records_json, change_log_json, relationships_json FROM knowledge_stores WHERE workspace_id = ?",
                (workspace_id,),
            )
            ks_row = cur.fetchone()
            if ks_row:
                if ks_row["relationships_json"]:
                    for r in json.loads(ks_row["relationships_json"]):
                        edges.append(
                            EdgePayload(
                                edge_id=r.get("relationship_id", f"edge_{len(edges)+1:03d}"),
                                from_node=r.get("source_record_id", ""),
                                to_node=r.get("target_record_id", ""),
                                relation_type=r.get("relationship_type", "reinforces"),
                                confidence=1.0,
                                reasoning=r.get("rationale", ""),
                            )
                        )
                if ks_row["change_log_json"]:
                    for c in json.loads(ks_row["change_log_json"]):
                        if c.get("session_id") == session_id:
                            change_logs.append(
                                ChangeLogPayload(
                                    log_id=c.get("entry_id", f"log_{len(change_logs)+1:03d}"),
                                    entity_type="content_unit",
                                    entity_id=c.get("record_id", ""),
                                    field="status",
                                    old_value=None,
                                    new_value=c.get("action", "created"),
                                    confidence=1.0,
                                    reasoning=c.get("details", ""),
                                )
                            )

        return KnowledgePayload(
            session=session_data,
            utterances=utterances,
            content_units=content_units,
            edges=edges,
            change_logs=change_logs,
        )

    def search_contract_rag(self, request: SearchRequest) -> SearchResponse:
        """Executes hybrid RAG search matching Canonical Contract 3."""
        if not self._embedding_provider and not request.query_embedding:
            return SearchResponse()

        query_emb = (
            request.query_embedding
            if request.query_embedding
            else self._embedding_provider.embed_text(request.query_text)  # type: ignore
        )

        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT unit_id, session_id, workspace_id, unit_type, content, embedding_json FROM content_units WHERE workspace_id = ?",
                (request.workspace_id,),
            )
            rows = cur.fetchall()

            turns_by_session: dict[str, list[dict]] = {}
            for r in rows:
                sid = r["session_id"]
                if sid not in turns_by_session:
                    t_cur = conn.execute("SELECT turns_json FROM transcripts WHERE session_id = ?", (sid,))
                    t_row = t_cur.fetchone()
                    turns_by_session[sid] = json.loads(t_row["turns_json"]) if t_row and t_row["turns_json"] else []

            relationships: list[dict] = []
            if request.walk_edges:
                ks_cur = conn.execute(
                    "SELECT relationships_json FROM knowledge_stores WHERE workspace_id = ?",
                    (request.workspace_id,),
                )
                ks_row = ks_cur.fetchone()
                if ks_row and ks_row["relationships_json"]:
                    relationships = json.loads(ks_row["relationships_json"])

        scored: list[ContractRetrievedResult] = []
        for r in rows:
            emb = json.loads(r["embedding_json"])
            dot = sum(a * b for a, b in zip(query_emb, emb))
            norm_a = math.sqrt(sum(a * a for a in query_emb)) or 1.0
            norm_b = math.sqrt(sum(b * b for b in emb)) or 1.0
            sim = dot / (norm_a * norm_b)

            recency = 1.0
            status_factor = 1.0
            rank = sim * recency * status_factor

            source_utts: list[SourceUtteranceInfo] = []
            for t in turns_by_session.get(r["session_id"], [])[:2]:
                source_utts.append(
                    SourceUtteranceInfo(
                        utterance_id=t.get("turn_id", ""),
                        speaker_name=t.get("speaker_name", "Speaker"),
                        start_ts=float(t.get("start_time", 0.0)),
                        end_ts=float(t.get("end_time", 0.0)),
                        raw_text=t.get("spoken_text", ""),
                    )
                )

            connected: list[ConnectedEdge] = []
            if request.walk_edges:
                for rel in relationships:
                    if rel.get("source_record_id") == r["unit_id"]:
                        connected.append(
                            ConnectedEdge(
                                relation_type=rel.get("relationship_type", "relates_to"),
                                connected_content_id=rel.get("target_record_id", ""),
                                content=rel.get("rationale", ""),
                            )
                        )

            scored.append(
                ContractRetrievedResult(
                    content_id=r["unit_id"],
                    type=r["unit_type"],
                    content=r["content"],
                    similarity_score=round(sim, 4),
                    recency_score=round(recency, 4),
                    combined_rank_score=round(rank, 4),
                    source_utterances=source_utts,
                    connected_edges=connected,
                )
            )

        scored.sort(key=lambda x: x.combined_rank_score, reverse=True)
        return SearchResponse(results=scored[: request.top_k])
