import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_layer.common.config_loader import (
    AILayerConfig,
    ConversationAnalysisConfig,
    DatabaseConfig,
    EmbeddingConfig,
    KnowledgeMemoryConfig,
    LLMConfig,
    ProvidersConfig,
    QueryAnsweringConfig,
    SpeakerVerificationConfig,
    STTConfig,
    SummarizationConfig,
    TranscriptionConfig,
)
from ai_layer.common.exceptions import AILayerError, TranscriptionError
from ai_layer.common.gemini_embedding_provider import GeminiEmbeddingProvider
from ai_layer.common.gemini_llm_provider import GeminiLLMProvider
from ai_layer.common.gemini_stt_provider import GeminiSTTProvider
from ai_layer.common.provider_factory import (
    create_embedding_provider,
    create_llm_provider,
    create_stt_provider,
)
from ai_layer.models.base_models import TimeRange
from ai_layer.models.transcription_models import AudioClip


@pytest.fixture
def gemini_config() -> AILayerConfig:
    return AILayerConfig(
        providers=ProvidersConfig(mode="gemini"),
        llm=LLMConfig(
            model_name="gemini-3.6-flash",
            temperature=0.1,
            timeout_seconds=30.0,
            max_retries=3,
            api_key_env_var="GEMINI_API_KEY",
        ),
        stt=STTConfig(
            model_name="gemini-3.6-flash",
            language="en",
            device="cpu",
        ),
        embedding=EmbeddingConfig(
            model_name="gemini-embedding-001",
            api_key_env_var="GEMINI_API_KEY",
            dimensions=768,
        ),
        database=DatabaseConfig(
            backend="sqlite",
            sqlite_path="data/test.db",
            connection_url_env_var="DATABASE_URL",
        ),
        transcription=TranscriptionConfig(
            max_pause_seconds=1.5,
            allow_overlap=True,
        ),
        speaker_verification=SpeakerVerificationConfig(
            min_confidence_threshold=0.7,
            boost_per_confirmation=0.15,
        ),
        conversation_analysis=ConversationAnalysisConfig(
            enable_normalization=True,
            max_facts_per_session=50,
        ),
        knowledge_memory=KnowledgeMemoryConfig(
            similarity_threshold=0.6,
            enable_contradiction_flagging=True,
        ),
        summarization=SummarizationConfig(
            max_highlights=10,
            include_action_items=True,
        ),
        query_answering=QueryAnsweringConfig(
            retrieval_top_k=5,
            min_relevance_score=0.5,
            fallback_message="No info",
        ),
    )


def test_gemini_llm_missing_api_key(gemini_config: AILayerConfig, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(AILayerError, match="Missing API key"):
        GeminiLLMProvider(gemini_config.llm)


def test_gemini_llm_generation_with_mock_sdk(gemini_config: AILayerConfig, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"status": "ok", "message": "hello world"}'
    mock_client.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_google = MagicMock()
    mock_google.genai = mock_genai

    with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
        provider = GeminiLLMProvider(gemini_config.llm)
        text = provider.generate_text("test prompt")
        assert "hello world" in text

        payload = provider.generate_structured_json("test json prompt")
        assert payload == {"status": "ok", "message": "hello world"}


def test_gemini_stt_missing_api_key(gemini_config: AILayerConfig, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(TranscriptionError, match="Missing API key"):
        GeminiSTTProvider(gemini_config.stt)


def test_gemini_stt_transcribe_clip(
    gemini_config: AILayerConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    dummy_wav = tmp_path / "test.wav"
    dummy_wav.write_bytes(b"RIFFdummywavcontent")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello team, this is our project update."
    mock_client.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_genai_types = MagicMock()
    mock_genai_types.Part.from_bytes.return_value = "audio_part"
    mock_genai.types = mock_genai_types
    mock_google = MagicMock()
    mock_google.genai = mock_genai

    with patch.dict(
        sys.modules,
        {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_genai_types,
        },
    ):
        provider = GeminiSTTProvider(gemini_config.stt)
        clip = AudioClip(
            clip_id="clip_001",
            audio_path=str(dummy_wav),
            preliminary_speaker_id="Speaker_1",
            time_range=TimeRange(0.0, 3.0),
        )
        text = provider.transcribe_clip(clip)
        assert text == "Hello team, this is our project update."


def test_gemini_stt_transcribe_file_segments(
    gemini_config: AILayerConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    dummy_wav = tmp_path / "test_session.wav"
    dummy_wav.write_bytes(b"RIFFdummywavcontent")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        '[{"start": 0.0, "end": 2.5, "text": "Turn 1"}, '
        '{"start": 3.0, "end": 6.0, "text": "Turn 2"}]'
    )
    mock_client.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_genai_types = MagicMock()
    mock_genai_types.Part.from_bytes.return_value = "audio_part"
    mock_genai.types = mock_genai_types
    mock_google = MagicMock()
    mock_google.genai = mock_genai

    with patch.dict(
        sys.modules,
        {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_genai_types,
        },
    ):
        provider = GeminiSTTProvider(gemini_config.stt)
        segments = provider.transcribe_file_segments(dummy_wav)
        assert len(segments) == 2
        assert segments[0]["text"] == "Turn 1"
        assert segments[1]["start"] == 3.0


def test_gemini_embedding_provider(gemini_config: AILayerConfig, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3, 0.4]
    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding]
    mock_client.models.embed_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_google = MagicMock()
    mock_google.genai = mock_genai

    with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
        provider = GeminiEmbeddingProvider(gemini_config.embedding)
        vec = provider.embed_text("sample phrase")
        assert vec == [0.1, 0.2, 0.3, 0.4]


def test_provider_factory_gemini_mode(gemini_config: AILayerConfig, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    mock_genai = MagicMock()
    mock_google = MagicMock()
    mock_google.genai = mock_genai
    with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
        llm = create_llm_provider(gemini_config)
        assert isinstance(llm, GeminiLLMProvider)

        stt = create_stt_provider(gemini_config)
        assert isinstance(stt, GeminiSTTProvider)

        embed = create_embedding_provider(gemini_config)
        assert isinstance(embed, GeminiEmbeddingProvider)
