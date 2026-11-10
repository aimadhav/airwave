#!/usr/bin/env python3
"""Audio File Platform - Production Command Line Interface (CLI).

Commands:
  generate-sample   Synthesize a multi-speaker conversation WAV file
  ingest            Ingest, slice, transcribe, analyze, and store an audio file
  analyze-transcript Directly ingest and analyze a text transcript (no audio required)
  query             Ask natural language questions against conversation memory
  list-sessions     List all recorded sessions in the knowledge database
  show-session      Inspect full transcript, insights, and summary of a session
"""

import argparse
import os
import sys
import uuid
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
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
if (Path(__file__).parent / ".runtime-deps").is_dir():
    sys.path.insert(0, str(Path(__file__).parent / ".runtime-deps"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import yaml
from dsp import DSPConfig, DSPPipeline
from dsp.exceptions import DSPError
from ai_layer.common.local_stt import LocalSTTConfig, LocalWhisperSTTProvider, LocalSTTError
from ai_layer.audio_ingestion.sample_generator import (
    generate_real_speech_wav,
    generate_synthetic_acoustic_wav,
)
from ai_layer.common.config_loader import load_config
from ai_layer.models.base_models import SessionMetadata
from ai_layer.models.speaker_models import SpeakerProfile
from ai_layer.pipeline_orchestrator import AILayerOrchestrator
from ai_layer.storage.factory import create_repository


def cmd_generate_sample(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    print(f"[*] Generating multi-speaker sample audio: {output_path}")

    if args.use_tts:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[-] OPENAI_API_KEY not found in environment. Falling back to synthetic tone synthesis.")
            path = generate_synthetic_acoustic_wav(output_path)
        else:
            print("[+] Using OpenAI TTS for high-fidelity human speech voices...")
            path = generate_real_speech_wav(output_path, api_key=api_key)
    else:
        path = generate_synthetic_acoustic_wav(output_path)

    size_kb = path.stat().st_size / 1024
    print(f"[+] Audio successfully generated: {path} ({size_kb:.1f} KB)")
    print(f"    Run: py -3.14 cli.py ingest \"{path}\"")


def cmd_ingest(args: argparse.Namespace) -> None:
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"[!] Error: Audio file does not exist: {audio_path}")
        sys.exit(1)

    config_path = args.config
    config = load_config(config_path)

    # Allow CLI override for mode
    if args.mode:
        os.environ["AI_LAYER_MODE"] = args.mode
        config = load_config(config_path)

    workspace_id = args.workspace
    session_id = args.session_id or f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_title = args.title or audio_path.stem.replace("_", " ").title()

    print("=" * 80)
    print(f"  INGESTING AUDIO SESSION: '{session_title}'")
    print("=" * 80)
    print(f"[*] Audio File:      {audio_path}")
    print(f"[*] Session ID:      {session_id}")
    print(f"[*] Workspace:       {workspace_id}")
    print(f"[*] Provider Mode:   {config.providers.mode.upper()}")
    print(f"[*] Database:        {config.database.sqlite_path}")
    print()

    # Step 1: Segmentation
    print("[1/4] Normalizing audio and constructing the speaker timeline...")
    values = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    dsp_config = DSPConfig.from_mapping(values.get("dsp", {}))
    dsp_config = replace(dsp_config, random_audio_provider=args.dsp_provider or dsp_config.random_audio_provider,
                         local_model=replace(dsp_config.local_model, num_speakers=args.num_speakers,
                                             backend=args.dsp_backend or dsp_config.local_model.backend))
    dsp_result = DSPPipeline(dsp_config).process_audio_file(audio_path, "data/dsp_sessions", session_id)
    manifest = dsp_result.payload.to_dict()
    (Path("data/dsp_sessions") / session_id / "payload.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (Path("data/dsp_sessions") / session_id / "dsp_config.json").write_text(json.dumps(asdict(dsp_config), indent=2), encoding="utf-8")
    print(f"[+] {len(dsp_result.payload.segments)} clips; {len({s.preliminary_speaker_id for s in dsp_result.payload.segments})} anonymous speaker labels ({dsp_config.random_audio_provider}).")
    if dsp_config.random_audio_provider == "local_model":
        print(f"[*] Local diarization backend: {dsp_config.local_model.backend}; VAD is diagnostic, not a hard speech gate.")

    # Step 2: Initialize Orchestrator with SQLite storage
    print("[2/4] Initializing AI Layer pipeline...")
    stt = None
    if dsp_config.random_audio_provider == "local_model" and config.providers.mode == "mock":
        stt = LocalWhisperSTTProvider(LocalSTTConfig(**values.get("local_stt", {})))
        print("[*] Real offline Whisper STT; analysis uses deterministic offline heuristics.")
    orchestrator = AILayerOrchestrator(config, stt_provider=stt, repository=create_repository(config))

    # Step 3: End-to-End Processing
    print("[3/4] Transcribing clips, verifying speakers, and extracting insights...")
    result = orchestrator.process_dsp_payload(dsp_result.payload, session_title=session_title,
                                             workspace_id=workspace_id, project_id=args.project)

    # Step 4: Save to Database
    print("[4/4] Persisting knowledge units and relationships to SQLite...")
    # process_dsp_payload() already persists through the configured repository.

    print("\n" + "=" * 80)
    print("  INGESTION & ANALYSIS COMPLETED")
    print("=" * 80)

    print("\n--- VERIFIED TRANSCRIPT ---")
    for turn in result.verified_transcript.turns:
        ts = f"[{turn.time_range.start_time_seconds:.1f}s - {turn.time_range.end_time_seconds:.1f}s]"
        print(f"  {ts} {turn.speaker_name} ({turn.speaker_id}): {turn.text}")

    print("\n--- EXTRACTED FACTS & DECISIONS ---")
    for fact in result.analysis_report.facts:
        print(f"  * [Fact] {fact.statement}")
    for dec in result.analysis_report.decisions:
        print(f"  * [Decision] {dec.decision}")

    print("\n--- ACTION ITEMS ---")
    if result.analysis_report.action_items:
        for task in result.analysis_report.action_items:
            dl = f" (Deadline: {task.deadline})" if task.deadline else ""
            print(f"  * [{task.assignee_name}] {task.task_description}{dl}")
    else:
        print("  (None detected)")

    print("\n--- EXECUTIVE SUMMARY ---")
    print(f"  Overview: {result.session_summary.overview}")
    if result.session_summary.discussion_highlights:
        print("  Highlights:")
        for h in result.session_summary.discussion_highlights:
            print(f"    - {h}")

    print(f"\n[+] Stored in SQLite database: {config.database.sqlite_path}")
    print(f"[+] Query with: py -3.14 cli.py query \"Your question here\"")


def cmd_analyze_transcript(args: argparse.Namespace) -> None:
    transcript_text = ""
    if args.text:
        transcript_text = args.text
    elif args.transcript_file:
        if args.transcript_file == "-":
            transcript_text = sys.stdin.read()
        else:
            file_path = Path(args.transcript_file)
            if not file_path.exists():
                print(f"[!] Error: Transcript file does not exist: {file_path}")
                sys.exit(1)
            transcript_text = file_path.read_text(encoding="utf-8")
    else:
        print("[!] Error: Please provide a transcript file path or use --text \"...\" (or '-' for stdin)")
        sys.exit(1)

    if not transcript_text.strip():
        print("[!] Error: Provided transcript is empty.")
        sys.exit(1)

    config_path = args.config
    config = load_config(config_path)

    if args.mode:
        os.environ["AI_LAYER_MODE"] = args.mode
        config = load_config(config_path)

    workspace_id = args.workspace
    session_id = args.session_id or f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_title = args.title or "Direct Transcript Analysis"

    print("=" * 80)
    print(f"  ANALYZING CONVERSATION TRANSCRIPT: '{session_title}'")
    print("=" * 80)
    print(f"[*] Input Mode:      Direct Transcript Ingestion (No Audio / STT Bypass)")
    print(f"[*] Session ID:      {session_id}")
    print(f"[*] Workspace:       {workspace_id}")
    print(f"[*] Provider Mode:   {config.providers.mode.upper()}")
    print(f"[*] Database:        {config.database.sqlite_path}")
    print()

    known_speakers = [
        SpeakerProfile(speaker_id="spk_sarah", canonical_name="Sarah", aliases=["Sara", "Sarah Chen"]),
        SpeakerProfile(speaker_id="spk_alex", canonical_name="Alex", aliases=["Alexander"]),
        SpeakerProfile(speaker_id="spk_david", canonical_name="David", aliases=["Dave"]),
    ]

    session_metadata = SessionMetadata(
        session_id=session_id,
        session_title=session_title,
        recorded_at=datetime.now(UTC).isoformat(),
        participant_ids=["Speaker_1", "Speaker_2", "Speaker_3"],
        project_id=args.project,
    )

    print("[1/3] Initializing AI Layer pipeline...")
    orchestrator = AILayerOrchestrator.from_config_file(config_path, enable_storage=True)

    print("[2/3] Parsing transcript turns and running Main Analyzer (facts, decisions, tasks, summary)...")
    result = orchestrator.analyze_transcript(
        transcript_input=transcript_text,
        session_metadata=session_metadata,
        known_speakers=known_speakers,
    )

    print("[3/3] Persisting knowledge units and relationships to SQLite...")
    orchestrator.save_session_result(
        workspace_id=workspace_id,
        session_metadata=session_metadata,
        result=result,
        audio_path=None,
    )

    print("\n" + "=" * 80)
    print("  TRANSCRIPT ANALYSIS COMPLETED")
    print("=" * 80)

    print("\n--- VERIFIED TRANSCRIPT ---")
    for turn in result.verified_transcript.turns:
        ts = f"[{turn.time_range.start_time_seconds:.1f}s - {turn.time_range.end_time_seconds:.1f}s]"
        print(f"  {ts} {turn.speaker_name} ({turn.speaker_id}): {turn.text}")

    print("\n--- EXTRACTED FACTS & DECISIONS ---")
    for fact in result.analysis_report.facts:
        print(f"  * [Fact] {fact.statement}")
    for dec in result.analysis_report.decisions:
        print(f"  * [Decision] {dec.decision}")

    print("\n--- ACTION ITEMS ---")
    if result.analysis_report.action_items:
        for task in result.analysis_report.action_items:
            dl = f" (Deadline: {task.deadline})" if task.deadline else ""
            print(f"  * [{task.assignee_name}] {task.task_description}{dl}")
    else:
        print("  (None detected)")

    print("\n--- EXECUTIVE SUMMARY ---")
    print(f"  Overview: {result.session_summary.overview}")
    if result.session_summary.discussion_highlights:
        print("  Highlights:")
        for h in result.session_summary.discussion_highlights:
            print(f"    - {h}")

    print(f"\n[+] Stored in SQLite database: {config.database.sqlite_path}")
    print(f"[+] Query with: py -3.14 cli.py query \"Your question here\"")


def cmd_query(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    orchestrator = AILayerOrchestrator.from_config_file(args.config, enable_storage=True)

    question = args.question
    workspace_id = args.workspace
    target_session = args.session

    print(f"[*] Querying Knowledge Base: \"{question}\"")
    answer = orchestrator.query_workspace(
        query_text=question,
        workspace_id=workspace_id,
        target_session_id=target_session,
    )

    print("\n" + "-" * 80)
    print(f"ANSWER: {answer.answer_text}")
    print("-" * 80)
    if answer.citations:
        print("CITATIONS & EVIDENCE:")
        for c in answer.citations:
            print(f"  - {c.speaker_name} [{c.timestamp_str}]: \"{c.verbatim_quote}\"")
    else:
        print("  (No direct citations found)")


def cmd_list_sessions(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    repo = create_repository(config)
    workspace_id = args.workspace

    sessions = repo.list_session_ids(workspace_id)
    print("=" * 80)
    print(f"  RECORDED SESSIONS (Workspace: {workspace_id})")
    print("=" * 80)
    if not sessions:
        print("  No sessions found in database.")
        return

    for sid in sessions:
        transcript = repo.load_verified_transcript(sid)
        report = repo.load_analysis_report(sid)
        turn_count = len(transcript.turns) if transcript else 0
        fact_count = len(report.facts) if report else 0
        task_count = len(report.action_items) if report else 0
        print(f"  * {sid}")
        print(f"    Turns: {turn_count} | Facts: {fact_count} | Action Items: {task_count}")


def cmd_show_session(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    repo = create_repository(config)
    sid = args.session_id

    transcript = repo.load_verified_transcript(sid)
    if not transcript:
        print(f"[!] Session not found: {sid}")
        return

    report = repo.load_analysis_report(sid)
    print("=" * 80)
    print(f"  SESSION DETAILS: {sid}")
    print("=" * 80)

    print("\n--- TRANSCRIPT ---")
    for turn in transcript.turns:
        ts = f"[{turn.time_range.start_time_seconds:.1f}s - {turn.time_range.end_time_seconds:.1f}s]"
        print(f"  {ts} {turn.speaker_name}: {turn.text}")

    print("\n--- EXTRACTED INSIGHTS ---")
    for fact in report.facts:
        print(f"  * [Fact] {fact.statement}")
    for dec in report.decisions:
        print(f"  * [Decision] {dec.decision}")
    for task in report.action_items:
        dl = f" (Deadline: {task.deadline})" if task.deadline else ""
        print(f"  * [Task] {task.assignee_name}: {task.task_description}{dl}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio File Platform - Speech Intelligence & Knowledge Base CLI"
    )
    parser.add_argument(
        "--config",
        default="configs/ai_layer_config.yaml",
        help="Path to ai_layer_config.yaml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate-sample
    p_gen = subparsers.add_parser("generate-sample", help="Generate a sample multi-speaker audio WAV file")
    p_gen.add_argument("--output", default="sample_meeting.wav", help="Output file path")
    p_gen.add_argument("--use-tts", action="store_true", help="Use OpenAI TTS for real voice synthesis if key is present")
    p_gen.set_defaults(func=cmd_generate_sample)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest and analyze an audio recording")
    p_ingest.add_argument("audio_file", help="Path to input .wav audio file")
    p_ingest.add_argument("--title", help="Title of the conversation/meeting")
    p_ingest.add_argument("--project", help="Associated project ID")
    p_ingest.add_argument("--workspace", default="ws_engineering", help="Target workspace ID")
    p_ingest.add_argument("--session-id", help="Explicit session ID")
    p_ingest.add_argument("--mode", choices=["mock", "openai", "gemini"], help="Override provider mode")
    p_ingest.add_argument("--dsp-provider", choices=["local_model", "single_speaker", "mock"], help="Acoustic speaker provider; local_model recognizes voice groups")
    p_ingest.add_argument("--dsp-backend", choices=["sherpa", "nemo_native"], help="Optional local neural runtime; native models must be installed explicitly")
    p_ingest.add_argument("--num-speakers", type=int, default=-1, help="Known count, or -1 for automatic clustering")
    p_ingest.set_defaults(func=cmd_ingest)

    # analyze-transcript
    p_analyze = subparsers.add_parser("analyze-transcript", help="Directly ingest and analyze a text transcript (no audio required)")
    p_analyze.add_argument("transcript_file", nargs="?", help="Path to transcript text file (or '-' for stdin)")
    p_analyze.add_argument("--text", help="Raw transcript string to analyze directly")
    p_analyze.add_argument("--title", help="Title of the conversation/meeting")
    p_analyze.add_argument("--project", help="Associated project ID")
    p_analyze.add_argument("--workspace", default="ws_engineering", help="Target workspace ID")
    p_analyze.add_argument("--session-id", help="Explicit session ID")
    p_analyze.add_argument("--mode", choices=["mock", "openai", "gemini"], help="Override provider mode")
    p_analyze.set_defaults(func=cmd_analyze_transcript)

    # query
    p_query = subparsers.add_parser("query", help="Ask questions against conversation records")
    p_query.add_argument("question", help="The natural language question to ask")
    p_query.add_argument("--workspace", default="ws_engineering", help="Workspace ID to search")
    p_query.add_argument("--session", help="Restrict query to a specific session ID")
    p_query.set_defaults(func=cmd_query)

    # list-sessions
    p_list = subparsers.add_parser("list-sessions", help="List all processed sessions")
    p_list.add_argument("--workspace", default="ws_engineering", help="Workspace ID")
    p_list.set_defaults(func=cmd_list_sessions)

    # show-session
    p_show = subparsers.add_parser("show-session", help="Show transcript and insights of a session")
    p_show.add_argument("session_id", help="Session ID to display")
    p_show.set_defaults(func=cmd_show_session)

    args = parser.parse_args()
    try:
        args.func(args)
    except (DSPError, LocalSTTError) as error:
        parser.exit(1, f"Processing error: {error}\n")


if __name__ == "__main__":
    main()
