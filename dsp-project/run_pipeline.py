"""End-to-End Runner for the Audio File AI Layer.

Demonstrates the complete multi-speaker pipeline:
  1. STT Transcription & Turn Stitching
  2. Speaker Identity Verification & Dialogue Cue Resolution
  3. Conversation Analysis & Structured Insight Extraction
  4. Knowledge Memory Graph Integration & Versioning
  5. Session & Project Summarization
  6. SQLite Database Persistence (Relational + Semantic Embeddings)
  7. RAG Natural Language Query Answering with Verbatim Citations
"""

import os
import sys
from pathlib import Path

# Auto-switch to project .venv if executed with system python without activated venv
if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
    _project_root = Path(__file__).resolve().parent
    _venv_py = _project_root / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if _venv_py.exists() and Path(sys.executable).resolve() != _venv_py.resolve():
        import subprocess
        sys.exit(subprocess.call([str(_venv_py)] + sys.argv))

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_layer.common.config_loader import load_config
from ai_layer.models.base_models import SessionMetadata, TimeRange
from ai_layer.models.speaker_models import SpeakerProfile
from ai_layer.models.transcription_models import AudioClip
from ai_layer.pipeline_orchestrator import AILayerOrchestrator


def build_sample_meeting() -> tuple[SessionMetadata, list[AudioClip], list[SpeakerProfile]]:
    """Builds a realistic multi-speaker engineering sync session."""
    session_metadata = SessionMetadata(
        session_id="session_sync_2026",
        session_title="Q4 Product & Engineering Sync",
        recorded_at="2026-09-17T14:30:00Z",
        participant_ids=["Speaker_1", "Speaker_2", "Speaker_3"],
        project_id="project_q4_launch",
    )

    known_speakers = [
        SpeakerProfile(
            speaker_id="speaker_sarah",
            canonical_name="Sarah",
            aliases=["Sara", "Sarah Chen"],
            base_confidence=0.6,
        ),
        SpeakerProfile(
            speaker_id="speaker_alex",
            canonical_name="Alex",
            aliases=["Alexander"],
            base_confidence=0.7,
        ),
        SpeakerProfile(
            speaker_id="speaker_david",
            canonical_name="David",
            aliases=["Dave"],
            base_confidence=0.65,
        ),
    ]

    # Simulated audio clips from Layer 1 (Speech Processing / Diarization)
    clips = [
        AudioClip(
            clip_id="clip_001",
            audio_path="clips/clip_001.wav",
            preliminary_speaker_id="Speaker_1",
            time_range=TimeRange(0.0, 3.2),
        ),
        AudioClip(
            clip_id="clip_002",
            audio_path="clips/clip_002.wav",
            preliminary_speaker_id="Speaker_2",
            time_range=TimeRange(3.8, 8.5),
        ),
        AudioClip(
            clip_id="clip_003",
            audio_path="clips/clip_003.wav",
            preliminary_speaker_id="Speaker_2",
            time_range=TimeRange(9.0, 13.0),
        ),
        AudioClip(
            clip_id="clip_004",
            audio_path="clips/clip_004.wav",
            preliminary_speaker_id="Speaker_3",
            time_range=TimeRange(14.0, 18.2),
        ),
        AudioClip(
            clip_id="clip_005",
            audio_path="clips/clip_005.wav",
            preliminary_speaker_id="Speaker_1",
            time_range=TimeRange(19.0, 24.5),
        ),
    ]

    return session_metadata, clips, known_speakers


def main() -> None:
    print("=" * 80)
    print("  AUDIO FILE PLATFORM - AI LAYER PIPELINE RUNNER")
    print("=" * 80)

    config_path = "configs/ai_layer_config.yaml"
    config = load_config(config_path)

    print(f"[*] Configuration Loaded: {config_path}")
    print(f"    - Provider Mode:     {config.providers.mode.upper()}")
    print(f"    - LLM Model:         {config.llm.model_name}")
    print(f"    - STT Model:         {config.stt.model_name}")
    print(f"    - Database Backend:  {config.database.backend} ({config.database.sqlite_path})")
    print(f"    - Confidence Thresh: {config.speaker_verification.min_confidence_threshold}")
    print()

    # Initialize the Orchestrator with SQLite database persistence
    orchestrator = AILayerOrchestrator.from_config_file(
        config_path,
        enable_storage=True,
    )

    workspace_id = "ws_engineering"
    session_metadata, clips, known_speakers = build_sample_meeting()

    print(f"[*] Ingesting Session: '{session_metadata.session_title}' ({session_metadata.session_id})")
    print(f"    - Clips received: {len(clips)}")
    print(f"    - Known Profiles: {[s.canonical_name for s in known_speakers]}")
    print()

    # Step 1-5: Process Audio Session through Sublayers 1 to 5
    print("[1/5] Running Speech Transcription & Turn Stitching...")
    print("[2/5] Running Dialogue-Cue Speaker Verification...")
    print("[3/5] Extracting Facts, Decisions & Action Items...")
    print("[4/5] Integrating Knowledge Graph & Updating Memory Store...")
    print("[5/5] Synthesizing Session and Project Summaries...")

    result = orchestrator.process_audio_session(
        session_metadata=session_metadata,
        clips=clips,
        known_speakers=known_speakers,
    )

    # Persist to SQLite
    orchestrator.save_session_result(
        workspace_id=workspace_id,
        session_metadata=session_metadata,
        result=result,
        audio_path="audio/meeting_2026_09_17.wav",
    )
    print(f"\n[+] Processing Complete & Persisted to SQLite: {config.database.sqlite_path}")

    # Display Verified Transcript
    print("\n" + "-" * 80)
    print("--- 1. VERIFIED TRANSCRIPT (Dialogue stitched & speakers identified) ---")
    print("-" * 80)
    for turn in result.verified_transcript.turns:
        ts = f"[{turn.time_range.start_time_seconds:.1f}s - {turn.time_range.end_time_seconds:.1f}s]"
        print(f"  {ts} {turn.speaker_name} ({turn.speaker_id}): {turn.text}")

    if result.verified_transcript.reassignments:
        print("\n  Speaker Reassignments / Confirmations:")
        for r in result.verified_transcript.reassignments:
            print(f"    * {r.preliminary_speaker_id} -> {r.resolved_name} ({r.resolved_speaker_id}): {r.rationale}")

    # Display Extracted Insights
    print("\n" + "-" * 80)
    print("--- 2. EXTRACTED INSIGHTS & ACTION ITEMS ---")
    print("-" * 80)
    print("  Facts:")
    for fact in result.analysis_report.facts:
        print(f"    * {fact.statement}")
        print(f"      [Citation: {fact.citation.speaker_name} @ {fact.citation.time_range.start_time_seconds:.1f}s]")

    print("\n  Decisions:")
    for dec in result.analysis_report.decisions:
        print(f"    * {dec.decision}")
        print(f"      [Citation: {dec.citation.speaker_name} @ {dec.citation.time_range.start_time_seconds:.1f}s]")

    print("\n  Action Items:")
    for task in result.analysis_report.action_items:
        deadline_str = f" (Deadline: {task.deadline})" if task.deadline else ""
        print(f"    * [{task.assignee_name}] {task.task_description}{deadline_str}")
        print(f"      [Citation: {task.citation.speaker_name} @ {task.citation.time_range.start_time_seconds:.1f}s]")

    # Display Session Summary
    print("\n" + "-" * 80)
    print("--- 3. EXECUTIVE SESSION SUMMARY ---")
    print("-" * 80)
    print(f"  Title:    {result.session_summary.title}")
    print(f"  Overview: {result.session_summary.overview}")
    if result.session_summary.discussion_highlights:
        print("  Highlights:")
        for h in result.session_summary.discussion_highlights:
            print(f"    - {h}")

    if result.project_summary:
        print("\n" + "-" * 80)
        print("--- 4. PROJECT-LEVEL STATUS ROLLUP ---")
        print("-" * 80)
        print(f"  Project ID: {result.project_summary.project_id}")
        print(f"  Status:     {result.project_summary.status_overview}")
        print(f"  Milestones: {result.project_summary.latest_milestones}")

    # Step 6: RAG Query Answering with Citations
    print("\n" + "-" * 80)
    print("--- 5. RAG USER QUERY ANSWERING (Tested against Database Records) ---")
    print("-" * 80)

    sample_queries = [
        "What did we decide about the launch date?",
        "What tasks did Sarah commit to?",
        "What was the marketing budget discussed?",
    ]

    for q in sample_queries:
        print(f"\n[?] Question: \"{q}\"")
        answer = orchestrator.query_workspace(
            query_text=q,
            workspace_id=workspace_id,
        )
        print(f"    Answer: {answer.answer_text}")
        if answer.citations:
            print("    Citations:")
            for c in answer.citations:
                print(f"      - {c.speaker_name} [{c.timestamp_str}]: \"{c.verbatim_quote}\"")

    # =========================================================================
    # EXTRA MODE: Direct Transcript Ingestion (No Audio STT Required)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  EXTRA MODE: DIRECT PASTED TRANSCRIPT INGESTION (STT BYPASS)")
    print("=" * 80)

    pasted_transcript = """
    Sarah: Hi everyone, let's do a quick sync on the mobile release.
    Alex: The backend authentication fix has landed on staging.
    David: I will finish the smoke testing by tomorrow 3 PM.
    Sarah: Excellent. We will proceed with the mobile app rollout on Thursday.
    """

    direct_metadata = SessionMetadata(
        session_id="session_direct_transcript_001",
        session_title="Mobile App Release Sync (Pasted Transcript)",
        recorded_at="2026-09-17T16:00:00Z",
        participant_ids=["Speaker_1", "Speaker_2", "Speaker_3"],
        project_id="project_mobile_v2",
    )

    print("[*] Feeding raw pasted transcript directly into the Main Analyzer...")
    direct_result = orchestrator.analyze_transcript(
        transcript_input=pasted_transcript,
        session_metadata=direct_metadata,
        known_speakers=known_speakers,
    )

    orchestrator.save_session_result(
        workspace_id=workspace_id,
        session_metadata=direct_metadata,
        result=direct_result,
        audio_path=None,
    )

    print(f"[+] Direct transcript analyzed and persisted successfully!")
    print(f"    - Turns: {len(direct_result.verified_transcript.turns)}")
    print(f"    - Facts: {len(direct_result.analysis_report.facts)}")
    print(f"    - Decisions: {len(direct_result.analysis_report.decisions)}")
    print(f"    - Action Items: {len(direct_result.analysis_report.action_items)}")
    print(f"    - Summary: {direct_result.session_summary.overview}")

    print("\n" + "=" * 80)
    print("  AI LAYER PIPELINE RUN SUCCEEDED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
