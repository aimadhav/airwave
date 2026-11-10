from ai_layer.common.llm_provider import LLMProvider
from ai_layer.models.speaker_models import VerifiedTurn


class FactExtractor:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def extract_facts(self, turns: list[VerifiedTurn]) -> list[dict[str, str]]:
        if not turns:
            return []

        dialogue = "\n".join(
            f"[{turn.turn_id}] {turn.speaker_name}: {turn.text}" for turn in turns
        )
        prompt = (
            "Extract factual statements from this verified transcript.\n"
            f"{dialogue}\n"
            'Return JSON: {"facts": [{"statement": "...", "turn_id": "..."}]}'
        )
        payload = self._llm_provider.generate_structured_json(prompt)
        facts = payload.get("facts", [])
        return [
            {"statement": item["statement"], "turn_id": item["turn_id"]}
            for item in facts
            if item.get("statement") and item.get("turn_id")
        ]
