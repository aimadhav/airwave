import hashlib
import math
import os
import re
from typing import Protocol

from ai_layer.common.config_loader import EmbeddingConfig
from ai_layer.common.exceptions import AILayerError


class EmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class MockEmbeddingProvider:
    """Deterministic pseudo-embeddings for offline tests."""

    def __init__(self, dimensions: int = 64) -> None:
        self._dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vector = [0.0] * self._dimensions
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class OpenAIEmbeddingProvider:
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._api_key = os.environ.get(config.api_key_env_var, "")
        if not self._api_key:
            raise AILayerError(
                f"Missing API key: set {config.api_key_env_var} for OpenAI embeddings."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise AILayerError(
                "OpenAI embeddings require the openai package. Install with: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self._api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._config.model_name,
            input=texts,
        )
        return [list(item.embedding) for item in response.data]

    def embed_text(self, text: str) -> list[float]:
        results = self.embed_texts([text])
        return results[0] if results else []


from ai_layer.common.gemini_embedding_provider import GeminiEmbeddingProvider  # noqa: E402

__all__ = [
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "GeminiEmbeddingProvider",
]
