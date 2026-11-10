import json
import os
import re
from typing import Any

from ai_layer.common.config_loader import LLMConfig
from ai_layer.common.exceptions import AILayerError


class OpenAILLMProvider:
    """OpenAI chat-completions provider for text and structured JSON."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._api_key = os.environ.get(config.api_key_env_var, "")
        if not self._api_key:
            raise AILayerError(
                f"Missing API key: set {config.api_key_env_var} for OpenAI LLM provider."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise AILayerError(
                "OpenAI provider requires the openai package. Install with: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self._api_key, timeout=config.timeout_seconds, max_retries=config.max_retries)

    def generate_text(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._config.model_name,
            temperature=self._config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def generate_structured_json(self, prompt: str) -> dict[str, Any]:
        json_prompt = (
            f"{prompt}\n\n"
            "Respond with valid JSON only. Do not include markdown fences or commentary."
        )
        raw = self.generate_text(json_prompt)
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise AILayerError(f"OpenAI returned invalid JSON: {raw[:200]}") from exc
        if not isinstance(payload, dict):
            raise AILayerError("OpenAI JSON response must be an object.")
        return payload
