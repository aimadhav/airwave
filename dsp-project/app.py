#!/usr/bin/env python3
"""Audio File Platform - Interactive Web Application Server.

Runs a lightweight REST API & web dashboard for browsing conversations,
uploading/ingesting recordings, viewing live verified transcripts,
exploring extracted action items & decisions, and asking AI questions with citations.
"""

import json
import os
import re
import sys
import threading
from dataclasses import replace
import importlib.util
from time import perf_counter
import urllib.parse
from datetime import UTC, datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

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
    sys.path.append(str(Path(__file__).parent / ".runtime-deps"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import base64
import urllib.request

from ai_layer.audio_ingestion.sample_generator import (
    generate_real_speech_wav,
    generate_synthetic_acoustic_wav,
)
from ai_layer.common.config_loader import load_config
from ai_layer.common.provider_factory import create_stt_provider
from ai_layer.common.stt_provider import MultiSpeakerDemoSTTProvider
from ai_layer.common.local_stt import LocalSTTConfig, LocalWhisperSTTProvider, LocalSTTError
from ai_layer.models.base_models import SessionMetadata, TimeRange
from ai_layer.models.speaker_models import SpeakerProfile
from ai_layer.models.transcription_models import AudioClip
from ai_layer.pipeline_orchestrator import AILayerOrchestrator
from ai_layer.storage.factory import create_repository
from ai_layer.transcription.transcript_parser import TranscriptParser
from dsp import DSPConfig, DSPPipeline
from dsp.diarization import MockDiarizer
from dsp.exceptions import DSPError
import yaml

CONFIG_PATH = "configs/ai_layer_config.yaml"
WORKSPACE_ID = "ws_engineering"


def load_dsp_config() -> DSPConfig:
    """Load acoustic configuration separately from the AI layer."""
    values = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    return DSPConfig.from_mapping(values.get("dsp", {}))


def load_local_stt_config() -> LocalSTTConfig:
    values = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    return LocalSTTConfig(**values.get("local_stt", {}))


def local_models_available() -> bool:
    """A lightweight availability probe, not an inference accuracy claim."""
    try:
        load_dsp_config().local_model.validate_assets()
        load_local_stt_config().validate_assets()
        return importlib.util.find_spec("sherpa_onnx") is not None
    except (DSPError, LocalSTTError):
        return False


class AudioFileAppHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path in {"/", "/index.html"}:
            return self._serve_file(Path("web/index.html"), "text/html")
        if path in {"/dsp-explainer", "/dsp-explainer.html"}:
            return self._serve_file(Path("web/dsp-explainer.html"), "text/html")
        if path.startswith("/web/"):
            root = Path("web").resolve()
            target = (root / urllib.parse.unquote(path.removeprefix("/web/"))).resolve()
            if target.is_relative_to(root) and target.is_file():
                content_type = "text/css" if target.suffix == ".css" else "application/javascript"
                return self._serve_file(target, content_type)

        if path == "/api/status":
            return self._handle_get_status()
        if path == "/api/sessions":
            return self._handle_get_sessions()
        if path == "/api/memory-graph":
            return self._handle_get_memory_graph()
        if path.startswith("/api/audio/"):
            return self._handle_get_clip(path.removeprefix("/api/audio/"))
        if path.startswith("/api/sessions/"):
            session_id = path.replace("/api/sessions/", "").strip()
            return self._handle_get_session_details(session_id)

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        url = urllib.parse.urlparse(self.path)
        path = url.path

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length < 0:
                raise ValueError("Negative content length")
        except ValueError:
            self._send_json({"error": "Invalid Content-Length"}, status=400)
            return
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Request body must be valid JSON"}, status=400)
            return
        if not isinstance(payload, dict):
            self._send_json({"error": "Request body must be a JSON object"}, status=400)
            return

        if path == "/api/generate-sample":
            return self._handle_generate_sample()
        if path == "/api/demo-multispeaker":
            return self._handle_multispeaker_demo(payload)
        if path == "/api/demo-real-audio":
            return self._handle_real_audio_demo(payload)
        if path == "/api/ingest":
            return self._handle_ingest(payload)
        if path == "/api/upload":
            return self._handle_upload(payload)
        if path == "/api/fetch-audio":
            return self._handle_fetch_audio(payload)
        if path == "/api/query":
            return self._handle_query(payload)
        if path == "/api/set-mode":
            return self._handle_set_mode(payload)
        if path == "/api/analyze-transcript":
            return self._handle_analyze_transcript(payload)

        self.send_error(404, "Endpoint Not Found")

    def _handle_set_mode(self, payload: dict) -> None:
        new_mode = str(payload.get("mode", "gemini")).lower()
        if new_mode in {"gemini", "openai", "mock"}:
            os.environ["AI_LAYER_MODE"] = new_mode
        self._handle_get_status()

    def _serve_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.exists():
            self.send_error(404, "File Not Found")
            return
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()
        with file_path.open("rb") as handle:
            self.wfile.write(handle.read())

    def _send_json(self, data: dict, status: int = 200) -> None:
        raw = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _handle_get_status(self) -> None:
        config = load_config(CONFIG_PATH)
        repo = create_repository(config)
        sessions = repo.list_session_ids(WORKSPACE_ID)
        self._send_json({
            "mode": config.providers.mode,
            "llm_model": config.llm.model_name,
            "stt_model": config.stt.model_name,
            "database": config.database.sqlite_path,
            "sessions_count": len(sessions),
            "local_models_available": local_models_available(),
        })

    def _handle_get_sessions(self) -> None:
        config = load_config(CONFIG_PATH)
        repo = create_repository(config)
        session_summaries = repo.list_sessions_summary(WORKSPACE_ID)
        out = []
        for s in session_summaries:
            sid = s["session_id"]
            transcript = repo.load_verified_transcript(sid)
            report = repo.load_analysis_report(sid)
            out.append({
                "session_id": sid,
                "title": s["title"],
                "recorded_at": s["recorded_at"],
                "status": s["status"],
                "progress": s["progress"],
                "turn_count": len(transcript.turns) if transcript else 0,
                "fact_count": len(report.facts) if report else 0,
                "task_count": len(report.action_items) if report else 0,
            })
        self._send_json({"sessions": out})

    def _handle_get_clip(self, reference: str) -> None:
        """Serve only WAV clips listed in a validated session manifest."""
        parts = reference.split("/")
        if len(parts) != 2 or not all(re.fullmatch(r"[A-Za-z0-9_-]+", p) for p in parts):
            self.send_error(404, "Clip not found")
            return
        session_id, segment_id = parts
        manifest = self._load_dsp_manifest(session_id)
        clip = next((s for s in manifest.get("segments", []) if s["segment_id"] == segment_id), None)
        if clip:
            root = (Path("data/dsp_sessions") / session_id / "clips").resolve()
            target = (root / clip["filename"]).resolve()
            if target.is_relative_to(root) and target.suffix == ".wav" and target.is_file():
                return self._serve_file(target, "audio/wav")
        self.send_error(404, "Clip not found")

    @staticmethod
    def _load_dsp_manifest(session_id: str) -> dict[str, Any]:
        """Read optional presentation metadata without exposing filesystem paths."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            return {}
        path = Path("data/dsp_sessions") / session_id / "web_manifest.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _handle_get_session_details(self, session_id: str) -> None:
        config = load_config(CONFIG_PATH)
        repo = create_repository(config)
        meta = repo.load_session_metadata(session_id)
        if not meta:
            self._send_json({"error": "Session not found"}, status=404)
            return
        transcript = repo.load_verified_transcript(session_id)
        report = repo.load_analysis_report(session_id)
        stored_summary = repo.load_session_summary(session_id)
        status_info = repo.load_session_status(session_id)

        session_title = (meta.get("title") if meta else None) or session_id.replace("sess_", "Session ").replace("_", " ")
        recorded_at = (meta.get("recorded_at") if meta else None) or datetime.now(UTC).isoformat()
        current_status = (meta.get("status") if meta else None) or status_info.get("status", "completed")

        turns = []
        if transcript:
            for t in transcript.turns:
                turns.append({
                    "turn_id": t.turn_id,
                    "speaker_name": t.speaker_name,
                    "speaker": t.speaker_name,
                    "speaker_id": t.speaker_id,
                    "start_ts": t.time_range.start_time_seconds,
                    "end_ts": t.time_range.end_time_seconds,
                    "text": t.text,
                })

        facts = []
        decisions = []
        action_items = []
        if report:
            for f in report.facts:
                facts.append({"fact_id": f.fact_id, "statement": f.statement,
                              "source_turn_id": f.citation.turn_id})
            for d in report.decisions:
                decisions.append({
                    "decision_id": d.decision_id,
                    "decision": d.decision,
                    "content": d.decision,
                    "agreed_by": d.agreed_by,
                })
            for a in report.action_items:
                action_items.append({
                    "task_id": a.task_id,
                    "task_description": a.task_description,
                    "task": a.task_description,
                    "assignee_name": a.assignee_name,
                    "assignee": a.assignee_name,
                    "deadline": a.deadline,
                    "due_date": a.deadline,
                })

        if stored_summary:
            summary_dict = {
                "overview": stored_summary.overview,
                "discussion_highlights": stored_summary.discussion_highlights,
            }
        elif current_status == "processing":
            summary_dict = {
                "overview": "Synthesizing executive summary in background...",
                "discussion_highlights": [t["text"] for t in turns[:6]] if turns else ["Processing transcript..."],
            }
        else:
            summary_dict = {
                "overview": f"Session intelligence for {session_title}.",
                "discussion_highlights": [t["text"] for t in turns[:6]] if turns else [],
            }

        self._send_json({
            "session_id": session_id,
            "title": session_title,
            "status": current_status,
            "progress": status_info.get("progress", {}),
            "recorded_at": recorded_at,
            "transcript": turns,
            "facts": facts,
            "decisions": decisions,
            "action_items": action_items,
            "summary": summary_dict,
            "dsp": self._load_dsp_manifest(session_id),
        })

    def _handle_get_memory_graph(self) -> None:
        config = load_config(CONFIG_PATH)
        repo = create_repository(config)
        ks = repo.load_knowledge_store(WORKSPACE_ID)
        records_out = [
            {
                "record_id": r.record_id,
                "record_type": r.record_type,
                "statement": r.statement,
                "session_id": r.session_id,
                "status": r.status,
                "related_record_ids": r.related_record_ids,
            }
            for r in ks.records
        ]
        log_out = [
            {
                "entry_id": l.entry_id,
                "record_id": l.record_id,
                "action": l.action,
                "session_id": l.session_id,
                "details": l.details,
            }
            for l in ks.change_log
        ]
        rels_out = [
            {
                "relationship_id": rel.relationship_id,
                "source_record_id": rel.source_record_id,
                "target_record_id": rel.target_record_id,
                "relationship_type": rel.relationship_type,
                "rationale": rel.rationale,
            }
            for rel in ks.relationships
        ]
        self._send_json({
            "workspace_id": WORKSPACE_ID,
            "records": records_out,
            "change_log": log_out,
            "relationships": rels_out,
            "relations": rels_out,
            "total_records": len(records_out),
            "total_relations": len(rels_out),
        })

    def _handle_generate_sample(self) -> None:
        output_file = Path("sample_meeting.wav")
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            path = generate_real_speech_wav(output_file, api_key=api_key)
        else:
            path = generate_synthetic_acoustic_wav(output_file)

        self._send_json({
            "status": "success",
            "file_path": str(path),
            "size_bytes": path.stat().st_size,
        })

    def _handle_multispeaker_demo(self, payload: dict) -> None:
        """Generate and process the offline three-speaker demo end to end."""

        output_file = Path("data/demo/multispeaker_demo.wav")
        path = generate_synthetic_acoustic_wav(output_file, pause_sec=0.8)
        result = self._ingest_audio_file(
            audio_path=path,
            session_title=payload.get("title") or "DSP Multi-Speaker Demo",
            mode="mock",
            sync=True,
            dsp_provider="mock",
            mock_speaker_count=3,
            demo_stt=True,
        )
        self._send_json(result)

    def _handle_real_audio_demo(self, payload: dict) -> None:
        """Run a downloaded conversation with automatic acoustic speaker clustering."""
        demo = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8")).get("real_audio_demo", {})
        path = Path(demo.get("audio_path", ""))
        if not path.is_file() or not local_models_available():
            self._send_json({"error": "Real demo assets are missing. Install requirements-diarization.txt and run python scripts/setup_diarization.py --with-stt --samples."}, status=400)
            return
        self._send_json(self._ingest_audio_file(path, session_title=demo.get("title", "Real conversation"),
                                             mode="mock", dsp_provider="local_model", sync=False))

    def _start_audio_request(self, audio_path: Path, payload: dict, title: str | None = None) -> None:
        """Validate processing choices and translate domain errors at the HTTP boundary."""
        try:
            count = payload.get("num_speakers", -1)
            if not isinstance(count, int) or isinstance(count, bool):
                raise ValueError("num_speakers must be an integer, or -1 for automatic")
            result = self._ingest_audio_file(
                audio_path, session_title=payload.get("title") or title,
                mode=payload.get("mode"), sync=bool(payload.get("sync", False)),
                dsp_provider=payload.get("dsp_provider"),
                num_speakers=count,
            )
            self._send_json(result)
        except (DSPError, LocalSTTError, ValueError, TypeError) as error:
            self._send_json({"error": str(error)}, status=400)

    def _ingest_audio_file(
        self,
        audio_path: Path,
        session_title: str | None = None,
        mode: str | None = None,
        sync: bool = False,
        dsp_provider: str | None = None,
        mock_speaker_count: int = 2,
        demo_stt: bool = False,
        num_speakers: int = -1,
    ) -> dict:
        config = load_config(CONFIG_PATH)
        if mode:
            if mode not in {"mock", "openai", "gemini"}:
                raise ValueError("Unknown AI provider")
            config = replace(config, providers=replace(config.providers, mode=mode))
        dsp_config = load_dsp_config()
        if dsp_provider:
            if dsp_provider not in {"mock", "single_speaker", "local_model"}:
                raise ValueError("Unknown DSP provider")
            dsp_config = replace(dsp_config, random_audio_provider=dsp_provider)
        dsp_config = replace(dsp_config, local_model=replace(dsp_config.local_model, num_speakers=num_speakers))
        if dsp_config.random_audio_provider == "local_model":
            dsp_config.local_model.validate_assets()
            if config.providers.mode == "mock":
                load_local_stt_config().validate_assets()
        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        title = session_title or audio_path.stem.replace("_", " ").title()

        session_metadata = SessionMetadata(
            session_id=session_id,
            session_title=title,
            recorded_at=datetime.now(UTC).isoformat(),
            participant_ids=[],
        )

        repo = create_repository(config)
        # 1. Immediately initialize session record in database
        repo.save_session_metadata(
            workspace_id=WORKSPACE_ID,
            session_metadata=session_metadata,
            audio_path=str(audio_path),
            status="processing",
            progress={"stage": "audio_received", "percent": 10},
        )

        known_speakers = [
            SpeakerProfile(speaker_id="spk_sarah", canonical_name="Sarah", aliases=["Sara", "Sarah Chen"]),
            SpeakerProfile(speaker_id="spk_alex", canonical_name="Alex", aliases=["Alexander"]),
            SpeakerProfile(speaker_id="spk_david", canonical_name="David", aliases=["Dave"]),
        ]
        dsp_metrics: dict[str, object] = {"segments": 0, "speaker_ids": []}

        def _run_audio_pipeline() -> Any:
            if audio_path.suffix.lower() in {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg", ".flac"}:
                started = perf_counter()
                repo.update_session_status(session_id, "processing", {"stage": "neural_diarization" if dsp_config.random_audio_provider == "local_model" else "voice_activity_detection", "percent": 15})
                diarizer = None
                if dsp_config.random_audio_provider == "mock":
                    diarizer = MockDiarizer(speaker_count=mock_speaker_count)

                dsp_result = DSPPipeline(
                    config=dsp_config,
                    diarization_provider=diarizer,
                ).process_audio_file(
                    source=audio_path,
                    output_dir=Path("data/dsp_sessions"),
                    session_id=session_id,
                )
                dsp_metrics["segments"] = len(dsp_result.payload.segments)
                dsp_metrics["speaker_ids"] = sorted(
                    {segment.preliminary_speaker_id for segment in dsp_result.payload.segments}
                )
                manifest = {
                    "fixture": demo_stt,
                    "analysis_mode": config.providers.mode,
                    "stt_provider": "scripted_fixture" if demo_stt else ("local_whisper" if dsp_config.random_audio_provider == "local_model" and config.providers.mode == "mock" else config.providers.mode),
                    "speaker_provider": dsp_config.random_audio_provider,
                    "speaker_count_mode": "automatic" if num_speakers == -1 else "provided",
                    "cluster_threshold": dsp_config.local_model.cluster_threshold if dsp_config.random_audio_provider == "local_model" else None,
                    "dsp_elapsed_seconds": perf_counter() - started,
                    "sample_rate": dsp_result.payload.sample_rate,
                    "channels": dsp_result.payload.channels,
                    "duration_seconds": dsp_result.payload.duration_seconds,
                    "warnings": dsp_result.payload.warnings,
                    "segments": [
                        {
                            "segment_id": s.segment_id,
                            "filename": Path(s.clip_path).name,
                            "start_time": s.start_time,
                            "end_time": s.end_time,
                            "speaker_id": s.preliminary_speaker_id,
                            "speaker_source": s.speaker_source,
                            "is_overlap": s.is_overlap,
                        } for s in dsp_result.payload.segments
                    ],
                }
                manifest_path = Path("data/dsp_sessions") / session_id / "web_manifest.json"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                stt_provider = MultiSpeakerDemoSTTProvider() if demo_stt else None
                if not demo_stt and dsp_config.random_audio_provider == "local_model" and config.providers.mode == "mock":
                    stt_provider = LocalWhisperSTTProvider(load_local_stt_config())
                orchestrator = AILayerOrchestrator(
                    config,
                    stt_provider=stt_provider,
                    repository=repo,
                )
                result = orchestrator.process_dsp_payload(
                    payload=dsp_result.payload, session_title=title, workspace_id=WORKSPACE_ID,
                )
                manifest["pipeline_elapsed_seconds"] = perf_counter() - started
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                return result

            clips: list[AudioClip] = []
            stt = create_stt_provider(config)
            segments = stt.transcribe_file_segments(str(audio_path))
            for idx, seg in enumerate(segments, start=1):
                clip_id = f"clip_{session_id}_{idx:03d}"
                spk = seg.get("speaker_id") or "SPK_UNKNOWN"
                clips.append(
                    AudioClip(
                        clip_id=clip_id,
                        audio_path=str(audio_path),
                        preliminary_speaker_id=spk,
                        time_range=TimeRange(
                            start_time_seconds=float(seg.get("start", 0.0)),
                            end_time_seconds=float(seg.get("end", 0.0)),
                        ),
                    )
                )

            orchestrator = AILayerOrchestrator.from_config_file(CONFIG_PATH, enable_storage=True)
            return orchestrator.process_audio_session(
                session_metadata=session_metadata,
                clips=clips,
                known_speakers=known_speakers,
                workspace_id=WORKSPACE_ID,
                audio_path=str(audio_path),
            )

        if sync:
            result = _run_audio_pipeline()
            return {
                "status": "success",
                "session_id": session_id,
                "title": title,
                "turns_count": len(result.verified_transcript.turns),
                "facts_count": len(result.analysis_report.facts),
                "decisions_count": len(result.analysis_report.decisions),
                "tasks_count": len(result.analysis_report.action_items),
                "dsp_segments": dsp_metrics["segments"],
                "speaker_ids": dsp_metrics["speaker_ids"],
            }

        def _bg_audio_worker():
            try:
                _run_audio_pipeline()
            except Exception as exc:
                print(f"[!] Background audio processing failed for {session_id}: {exc}")
                repo.update_session_status(session_id, "failed", {"error": str(exc)})

        worker_thread = threading.Thread(target=_bg_audio_worker, daemon=True)
        worker_thread.start()

        return {
            "status": "processing",
            "session_id": session_id,
            "title": title,
        }

    def _handle_ingest(self, payload: dict) -> None:
        audio_file = payload.get("file_path", "sample_meeting.wav")
        audio_path = Path(audio_file)
        if not audio_path.is_file():
            self._send_json({"error": "Audio file not found. Upload a file or generate a sample first."}, status=400)
            return

        self._start_audio_request(audio_path, payload)

    def _handle_upload(self, payload: dict) -> None:
        audio_base64 = payload.get("audio_base64")
        if not audio_base64:
            self._send_json({"error": "Missing audio_base64 in payload"}, status=400)
            return

        filename = payload.get("filename", f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm")
        safe_name = Path(filename).name
        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        target_path = uploads_dir / safe_name

        try:
            if "," in audio_base64:
                audio_base64 = audio_base64.split(",", 1)[1]
            raw_bytes = base64.b64decode(audio_base64)
            with target_path.open("wb") as handle:
                handle.write(raw_bytes)
        except Exception as exc:
            self._send_json({"error": f"Failed to decode audio: {exc}"}, status=400)
            return

        self._start_audio_request(target_path, payload, f"Uploaded {safe_name}")

    def _handle_fetch_audio(self, payload: dict) -> None:
        url = payload.get("url")
        if not url:
            self._send_json({"error": "Missing url in payload"}, status=400)
            return

        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        ext = ".mp3"
        if ".wav" in url.lower():
            ext = ".wav"
        elif ".m4a" in url.lower():
            ext = ".m4a"
        elif ".ogg" in url.lower():
            ext = ".ogg"

        filename = f"fetched_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        target_path = uploads_dir / filename

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AudioFilePlatform/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response, target_path.open("wb") as out_file:
                out_file.write(response.read())
        except Exception as exc:
            self._send_json({"error": f"Failed to download audio: {exc}"}, status=400)
            return

        self._start_audio_request(target_path, payload, f"Fetched Audio: {target_path.name}")

    def _handle_analyze_transcript(self, payload: dict) -> None:
        transcript_text = payload.get("transcript_text", "").strip()
        if not transcript_text:
            self._send_json({"error": "Missing transcript_text in payload"}, status=400)
            return

        session_title = payload.get("title") or "Pasted Transcript Analysis"
        mode = payload.get("mode")
        if mode:
            os.environ["AI_LAYER_MODE"] = mode

        session_id = payload.get("session_id") or f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        workspace_id = payload.get("workspace_id", WORKSPACE_ID)

        known_speakers = [
            SpeakerProfile(speaker_id="spk_sarah", canonical_name="Sarah", aliases=["Sara", "Sarah Chen"]),
            SpeakerProfile(speaker_id="spk_alex", canonical_name="Alex", aliases=["Alexander"]),
            SpeakerProfile(speaker_id="spk_david", canonical_name="David", aliases=["Dave"]),
        ]

        # 1. Immediately parse transcript into structured VerifiedTurn objects
        parser = TranscriptParser()
        verified_transcript = parser.parse_to_verified_transcript(
            text=transcript_text,
            session_id=session_id,
            known_speakers=known_speakers,
        )

        # Collect dynamically discovered speakers from the parsed transcript
        for turn in verified_transcript.turns:
            spk_name = turn.speaker_name
            if spk_name and not any(k.canonical_name.lower() == spk_name.lower() for k in known_speakers):
                normalized_id = re.sub(r"[^a-zA-Z0-9_]", "", spk_name.lower().replace(" ", "_"))
                spk_id = f"spk_{normalized_id}" if normalized_id else f"spk_{len(known_speakers)+1}"
                known_speakers.append(
                    SpeakerProfile(
                        speaker_id=spk_id,
                        canonical_name=spk_name,
                        aliases=[spk_name],
                        base_confidence=0.9,
                    )
                )

        participants = list(dict.fromkeys(turn.speaker_name for turn in verified_transcript.turns if turn.speaker_name))
        session_metadata = SessionMetadata(
            session_id=session_id,
            session_title=session_title,
            recorded_at=datetime.now(UTC).isoformat(),
            participant_ids=participants,
        )

        config = load_config(CONFIG_PATH)
        repo = create_repository(config)
        try:
            repo.upsert_speakers(known_speakers)
        except Exception as exc:
            print(f"[!] Warning: Speaker upsert failed: {exc}")

        # 2. Immediately persist initial session record AND parsed transcript turns in database
        repo.save_session_metadata(
            workspace_id=workspace_id,
            session_metadata=session_metadata,
            audio_path=None,
            status="processing",
            progress={
                "stage": "transcript_parsed",
                "percent": 30,
                "turns_count": len(verified_transcript.turns),
            },
        )
        repo.save_transcript(session_id, verified_transcript)

        if bool(payload.get("sync", False)):
            orchestrator = AILayerOrchestrator.from_config_file(CONFIG_PATH, enable_storage=True)
            res = orchestrator.analyze_transcript(
                transcript_input=verified_transcript,
                session_metadata=session_metadata,
                known_speakers=known_speakers,
                workspace_id=workspace_id,
            )
            self._send_json({
                "status": "success",
                "session_id": session_id,
                "title": session_title,
                "turn_count": len(verified_transcript.turns),
                "facts_count": len(res.analysis_report.facts),
                "decisions_count": len(res.analysis_report.decisions),
                "tasks_count": len(res.analysis_report.action_items),
            })
            return

        def _bg_transcript_worker():
            try:
                orchestrator = AILayerOrchestrator.from_config_file(CONFIG_PATH, enable_storage=True)
                orchestrator.analyze_transcript(
                    transcript_input=verified_transcript,
                    session_metadata=session_metadata,
                    known_speakers=known_speakers,
                    workspace_id=workspace_id,
                )
            except Exception as exc:
                print(f"[!] Background transcript analysis failed for {session_id}: {exc}")
                repo.update_session_status(session_id, "failed", {"error": str(exc)})

        worker_thread = threading.Thread(target=_bg_transcript_worker, daemon=True)
        worker_thread.start()

        self._send_json({
            "status": "processing",
            "session_id": session_id,
            "title": session_title,
            "turn_count": len(verified_transcript.turns),
        })

    def _handle_query(self, payload: dict) -> None:
        query_text = payload.get("query", "")
        session_id = payload.get("session_id")
        scope = payload.get("scope", "current")
        if scope == "workspace" or session_id in ("all", "*", "", None):
            session_id = None

        mode = payload.get("mode")
        if mode:
            os.environ["AI_LAYER_MODE"] = mode

        try:
            orchestrator = AILayerOrchestrator.from_config_file(CONFIG_PATH, enable_storage=True)
            answer = orchestrator.query_workspace(
                query_text=query_text,
                workspace_id=WORKSPACE_ID,
                target_session_id=session_id,
            )
        except Exception as exc:
            print(f"[!] Warning: Live query generation exception: {exc}")
            # Graceful degradation: Answer via local retrieval & grounding
            old_mode = os.environ.get("AI_LAYER_MODE")
            os.environ["AI_LAYER_MODE"] = "mock"
            try:
                fallback_orchestrator = AILayerOrchestrator.from_config_file(CONFIG_PATH, enable_storage=True)
                answer = fallback_orchestrator.query_workspace(
                    query_text=query_text,
                    workspace_id=WORKSPACE_ID,
                    target_session_id=session_id,
                )
            finally:
                if old_mode:
                    os.environ["AI_LAYER_MODE"] = old_mode
                else:
                    os.environ.pop("AI_LAYER_MODE", None)

        citations = []
        for c in answer.citations:
            citations.append({
                "speaker_name": c.speaker_name,
                "speaker": c.speaker_name,
                "timestamp_str": c.timestamp_str,
                "verbatim_quote": c.verbatim_quote,
                "quote": c.verbatim_quote,
                "session_id": getattr(c, "session_id", ""),
                "session_title": getattr(c, "session_title", ""),
            })

        self._send_json({
            "answer_text": answer.answer_text,
            "answer": answer.answer_text,
            "citations": citations,
            "is_found": answer.is_found,
        })


def run_server(port: int = 5000) -> None:
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, AudioFileAppHandler)
    print("=" * 80)
    print(f"  AUDIO FILE PLATFORM - WEB APPLICATION SERVER RUNNING")
    print("=" * 80)
    print(f"[*] Dashboard URL:  http://127.0.0.1:{port}")
    print(f"[*] Database Path:  data/ai_layer.db")
    print(f"[*] Press Ctrl+C to stop.")
    print("=" * 80)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Audio File Web Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    args, _ = parser.parse_known_args()
    run_server(args.port)
