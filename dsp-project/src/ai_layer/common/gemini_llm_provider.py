import json
import os
import re
from typing import Any

from ai_layer.common.config_loader import LLMConfig
from ai_layer.common.exceptions import AILayerError


class GeminiLLMProvider:
    """Google Gemini provider for text completion and structured JSON generation."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._api_key = (
            os.environ.get(config.api_key_env_var, "")
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        if not self._api_key:
            raise AILayerError(
                f"Missing API key: set {config.api_key_env_var} or GEMINI_API_KEY / GOOGLE_API_KEY for Gemini LLM provider."
            )

        self._sdk_type = None
        self._client = None
        self._model = None

        # Try modern google-genai SDK first
        try:
            from google import genai

            self._sdk_type = "genai"
            self._client = genai.Client(api_key=self._api_key)
        except ImportError:
            # Fall back to legacy google-generativeai SDK
            try:
                import google.generativeai as genai_legacy

                self._sdk_type = "generativeai"
                genai_legacy.configure(api_key=self._api_key)
                self._model = genai_legacy.GenerativeModel(model_name=self._config.model_name)
            except ImportError as exc:
                raise AILayerError(
                    "Gemini provider requires google-genai or google-generativeai package. "
                    "Install with: pip install google-genai"
                ) from exc

    def generate_text(self, prompt: str) -> str:
        model_candidates = []
        if getattr(self, "_active_model", None):
            model_candidates.append(self._active_model)
        else:
            if self._config.model_name and "gemini-3" not in self._config.model_name.lower():
                model_candidates.append(self._config.model_name)
            for m in ["gemini-flash-latest", "gemini-flash-lite-latest"]:
                if m not in model_candidates:
                    model_candidates.append(m)

        retries = max(1, self._config.max_retries)
        last_error = None

        for model_name in model_candidates:
            for attempt in range(retries):
                try:
                    if self._sdk_type == "genai":
                        response = self._client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                        )
                        text = getattr(response, "text", "") or ""
                        self._active_model = model_name
                        return text.strip()
                    elif self._sdk_type == "generativeai":
                        import google.generativeai as genai_legacy

                        target_model = model_name if model_name.startswith("models/") else f"models/{model_name}"
                        model_obj = genai_legacy.GenerativeModel(model_name=target_model)
                        response = model_obj.generate_content(prompt)
                        text = getattr(response, "text", "") or ""
                        self._active_model = model_name
                        return text.strip()
                except Exception as exc:
                    last_error = exc
                    err_msg = str(exc)
                    is_not_found = "404" in err_msg or "not found" in err_msg.lower()
                    if is_not_found:
                        # Skip retrying this model and advance to next candidate
                        break

                    is_rate_limit = (
                        "429" in err_msg
                        or "RESOURCE_EXHAUSTED" in err_msg
                        or "Quota exceeded" in err_msg
                    )
                    if is_rate_limit and attempt < retries - 1:
                        delay_match = re.search(r"retry\s+in\s+([\d\.]+)\s*s", err_msg, re.IGNORECASE)
                        if delay_match:
                            sleep_sec = min(float(delay_match.group(1)) + 1.0, 35.0)
                        else:
                            sleep_sec = min(2.0 * (2 ** attempt), 20.0)
                        import time
                        time.sleep(sleep_sec)
                        continue
                    break

        if last_error:
            raise AILayerError(f"Gemini LLM generation failed: {last_error}") from last_error
        return ""

    def generate_structured_json(self, prompt: str) -> dict[str, Any]:
        json_prompt = (
            f"{prompt}\n\n"
            "Respond with valid JSON only. Do not include markdown fences, preambles, or commentary."
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
            raise AILayerError(f"Gemini returned invalid JSON: {raw[:200]}") from exc
        if not isinstance(payload, dict):
            raise AILayerError("Gemini JSON response must be an object.")
        return payload
