import os

from ai_layer.common.config_loader import EmbeddingConfig
from ai_layer.common.exceptions import AILayerError


class GeminiEmbeddingProvider:
    """Google Gemini text embedding provider using gemini-embedding-001."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._api_key = (
            os.environ.get(config.api_key_env_var, "")
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        if not self._api_key:
            raise AILayerError(
                f"Missing API key: set {config.api_key_env_var} or GEMINI_API_KEY / GOOGLE_API_KEY for Gemini embeddings."
            )

        self._sdk_type = None
        self._client = None

        try:
            from google import genai

            self._sdk_type = "genai"
            self._client = genai.Client(api_key=self._api_key)
        except ImportError:
            try:
                import google.generativeai as genai_legacy

                self._sdk_type = "generativeai"
                genai_legacy.configure(api_key=self._api_key)
            except ImportError as exc:
                raise AILayerError(
                    "Gemini embeddings require google-genai or google-generativeai package. "
                    "Install with: pip install google-genai"
                ) from exc

        self._active_model: str | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Determine candidate models to try
        if self._active_model:
            candidates = [self._active_model]
        else:
            candidates = []
            if self._config.model_name and self._config.model_name != "gemini-embedding-001":
                candidates.append(self._config.model_name)
            for m in ["text-embedding-004", "embedding-001"]:
                if m not in candidates:
                    candidates.append(m)

        last_error = None
        for model_name in candidates:
            try:
                if self._sdk_type == "genai":
                    response = self._client.models.embed_content(
                        model=model_name,
                        contents=texts,
                    )
                    if hasattr(response, "embeddings") and response.embeddings:
                        self._active_model = model_name
                        return [list(e.values) for e in response.embeddings]
                elif self._sdk_type == "generativeai":
                    import google.generativeai as genai_legacy

                    target_model = model_name if model_name.startswith("models/") else f"models/{model_name}"
                    result = genai_legacy.embed_content(
                        model=target_model,
                        content=texts,
                    )
                    embeddings = result.get("embedding", [])
                    if embeddings:
                        self._active_model = model_name
                        if isinstance(embeddings[0], list):
                            return [list(e) for e in embeddings]
                        return [list(embeddings)]
            except Exception as exc:
                last_error = exc
                continue

        if last_error:
            raise AILayerError(f"Gemini embedding generation failed: {last_error}") from last_error

        return []

    def embed_text(self, text: str) -> list[float]:
        results = self.embed_texts([text])
        return results[0] if results else []
