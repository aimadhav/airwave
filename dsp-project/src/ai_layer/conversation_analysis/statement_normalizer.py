from ai_layer.common.llm_provider import LLMProvider


class StatementNormalizer:
    def __init__(self, llm_provider: LLMProvider, enabled: bool = True) -> None:
        self._llm_provider = llm_provider
        self._enabled = enabled

    def normalize_to_self_contained(
        self,
        raw_statement: str,
        speaker_name: str,
        context_window: str,
    ) -> str:
        if not self._enabled:
            return raw_statement

        prompt = (
            "Normalize this statement into a self-contained sentence with explicit entities.\n"
            f"Speaker: {speaker_name}\n"
            f"Context: {context_window}\n"
            f"Statement: {raw_statement}\n"
            'Return JSON: {"normalized_statement": "..."}'
        )
        payload = self._llm_provider.generate_structured_json(prompt)
        normalized = payload.get("normalized_statement")
        return normalized if isinstance(normalized, str) and normalized.strip() else raw_statement
