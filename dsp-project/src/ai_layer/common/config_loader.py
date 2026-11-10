import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> bool:
        return False


@dataclass(frozen=True)
class ProvidersConfig:
    mode: str


@dataclass(frozen=True)
class LLMConfig:
    model_name: str
    temperature: float
    timeout_seconds: float
    max_retries: int
    api_key_env_var: str


@dataclass(frozen=True)
class STTConfig:
    model_name: str
    language: str
    device: str


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    api_key_env_var: str
    dimensions: int


@dataclass(frozen=True)
class DatabaseConfig:
    backend: str
    sqlite_path: str
    connection_url_env_var: str


@dataclass(frozen=True)
class TranscriptionConfig:
    max_pause_seconds: float
    allow_overlap: bool


@dataclass(frozen=True)
class SpeakerVerificationConfig:
    min_confidence_threshold: float
    boost_per_confirmation: float


@dataclass(frozen=True)
class ConversationAnalysisConfig:
    enable_normalization: bool
    max_facts_per_session: int


@dataclass(frozen=True)
class KnowledgeMemoryConfig:
    similarity_threshold: float
    enable_contradiction_flagging: bool


@dataclass(frozen=True)
class SummarizationConfig:
    max_highlights: int
    include_action_items: bool


@dataclass(frozen=True)
class QueryAnsweringConfig:
    retrieval_top_k: int
    min_relevance_score: float
    fallback_message: str


@dataclass(frozen=True)
class AILayerConfig:
    providers: ProvidersConfig
    llm: LLMConfig
    stt: STTConfig
    embedding: EmbeddingConfig
    database: DatabaseConfig
    transcription: TranscriptionConfig
    speaker_verification: SpeakerVerificationConfig
    conversation_analysis: ConversationAnalysisConfig
    knowledge_memory: KnowledgeMemoryConfig
    summarization: SummarizationConfig
    query_answering: QueryAnsweringConfig


def _require_section(data: dict[str, Any], key: str) -> dict[str, Any]:
    section = data.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid config section: {key}")
    return section


def _parse_yaml_fallback(text: str) -> dict[str, Any]:
    """Lightweight pure-python parser for simple nested YAML configuration without PyYAML."""
    result: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            if not raw_line.startswith(" ") and not raw_line.startswith("\t"):
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.split("#")[0].strip()
                if not val:
                    current_section = key
                    result[current_section] = {}
                else:
                    result[key] = _parse_scalar(val)
            elif current_section is not None:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.split("#")[0].strip()
                result[current_section][key] = _parse_scalar(val)
    return result


def _parse_scalar(val: str) -> Any:
    val = val.strip().strip('"').strip("'")
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() in {"none", "null"}:
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def load_config(config_file_path: str = "configs/ai_layer_config.yaml") -> AILayerConfig:
    load_dotenv()
    path = Path(config_file_path)
    if not path.exists():
        # Check if json alternative exists
        json_alt = path.with_suffix(".json")
        if json_alt.exists():
            path = json_alt
        else:
            raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle) or {}
    elif yaml is not None:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    else:
        json_path = path.with_suffix(".json")
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle) or {}
        else:
            with path.open("r", encoding="utf-8") as handle:
                raw = _parse_yaml_fallback(handle.read())

    providers = _require_section(raw, "providers")
    llm = _require_section(raw, "llm")
    stt = _require_section(raw, "stt")
    embedding = _require_section(raw, "embedding")
    database = _require_section(raw, "database")
    transcription = _require_section(raw, "transcription")
    speaker_verification = _require_section(raw, "speaker_verification")
    conversation_analysis = _require_section(raw, "conversation_analysis")
    knowledge_memory = _require_section(raw, "knowledge_memory")
    summarization = _require_section(raw, "summarization")
    query_answering = _require_section(raw, "query_answering")

    mode = os.environ.get("AI_LAYER_MODE", str(providers.get("mode", "mock"))).lower()

    llm_model = str(llm["model_name"])
    stt_model = str(stt["model_name"])
    embed_model = str(embedding["model_name"])
    embed_dim = int(embedding["dimensions"])
    llm_key_env = str(llm["api_key_env_var"])
    embed_key_env = str(embedding["api_key_env_var"])

    if mode == "gemini":
        if "gpt" in llm_model.lower() or "gemini-3" in llm_model.lower() or not llm_model:
            llm_model = os.environ.get("GEMINI_LLM_MODEL", "gemini-flash-latest")
        if "whisper" in stt_model.lower() or "gemini-3" in stt_model.lower() or not stt_model:
            stt_model = os.environ.get("GEMINI_STT_MODEL", "gemini-2.0-flash")
        if "text-embedding-3" in embed_model.lower() or "gemini-embedding-001" in embed_model.lower() or not embed_model:
            embed_model = os.environ.get("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
            embed_dim = 768
        if llm_key_env == "OPENAI_API_KEY":
            llm_key_env = "GEMINI_API_KEY"
        if embed_key_env == "OPENAI_API_KEY":
            embed_key_env = "GEMINI_API_KEY"

    return AILayerConfig(
        providers=ProvidersConfig(mode=mode),
        llm=LLMConfig(
            model_name=llm_model,
            temperature=float(llm["temperature"]),
            timeout_seconds=float(llm["timeout_seconds"]),
            max_retries=int(llm["max_retries"]),
            api_key_env_var=llm_key_env,
        ),
        stt=STTConfig(
            model_name=stt_model,
            language=str(stt["language"]),
            device=str(stt["device"]),
        ),
        embedding=EmbeddingConfig(
            model_name=embed_model,
            api_key_env_var=embed_key_env,
            dimensions=embed_dim,
        ),
        database=DatabaseConfig(
            backend=str(database["backend"]),
            sqlite_path=str(database["sqlite_path"]),
            connection_url_env_var=str(database["connection_url_env_var"]),
        ),
        transcription=TranscriptionConfig(
            max_pause_seconds=float(transcription["max_pause_seconds"]),
            allow_overlap=bool(transcription["allow_overlap"]),
        ),
        speaker_verification=SpeakerVerificationConfig(
            min_confidence_threshold=float(speaker_verification["min_confidence_threshold"]),
            boost_per_confirmation=float(speaker_verification["boost_per_confirmation"]),
        ),
        conversation_analysis=ConversationAnalysisConfig(
            enable_normalization=bool(conversation_analysis["enable_normalization"]),
            max_facts_per_session=int(conversation_analysis["max_facts_per_session"]),
        ),
        knowledge_memory=KnowledgeMemoryConfig(
            similarity_threshold=float(knowledge_memory["similarity_threshold"]),
            enable_contradiction_flagging=bool(knowledge_memory["enable_contradiction_flagging"]),
        ),
        summarization=SummarizationConfig(
            max_highlights=int(summarization["max_highlights"]),
            include_action_items=bool(summarization["include_action_items"]),
        ),
        query_answering=QueryAnsweringConfig(
            retrieval_top_k=int(query_answering["retrieval_top_k"]),
            min_relevance_score=float(query_answering["min_relevance_score"]),
            fallback_message=str(query_answering["fallback_message"]),
        ),
    )
