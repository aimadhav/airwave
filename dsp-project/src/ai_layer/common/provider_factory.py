from ai_layer.common.config_loader import AILayerConfig
from ai_layer.common.embedding_provider import (
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from ai_layer.common.gemini_llm_provider import GeminiLLMProvider
from ai_layer.common.gemini_stt_provider import GeminiSTTProvider
from ai_layer.common.llm_provider import LLMProvider, MockLLMProvider
from ai_layer.common.openai_llm_provider import OpenAILLMProvider
from ai_layer.common.openai_stt_provider import OpenAISTTProvider
from ai_layer.common.stt_provider import MockSTTProvider, STTProvider


def create_llm_provider(config: AILayerConfig) -> LLMProvider:
    if config.providers.mode == "openai":
        return OpenAILLMProvider(config.llm)
    elif config.providers.mode == "gemini":
        return GeminiLLMProvider(config.llm)
    return MockLLMProvider()


def create_stt_provider(config: AILayerConfig) -> STTProvider:
    if config.providers.mode == "openai":
        return OpenAISTTProvider(config.stt)
    elif config.providers.mode == "gemini":
        return GeminiSTTProvider(config.stt)
    return MockSTTProvider()


def create_embedding_provider(config: AILayerConfig) -> EmbeddingProvider:
    if config.providers.mode == "openai":
        return OpenAIEmbeddingProvider(config.embedding)
    elif config.providers.mode == "gemini":
        return GeminiEmbeddingProvider(config.embedding)
    return MockEmbeddingProvider(config.embedding.dimensions)
