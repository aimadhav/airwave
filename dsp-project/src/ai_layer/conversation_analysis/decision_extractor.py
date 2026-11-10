from ai_layer.common.llm_provider import LLMProvider
from ai_layer.models.speaker_models import VerifiedTurn


class DecisionExtractor:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def extract_decisions(self, turns: list[VerifiedTurn]) -> list[dict[str, object]]:
        if not turns:
            return []

        dialogue = "\n".join(
            f"[{turn.turn_id}] {turn.speaker_name}: {turn.text}" for turn in turns
        )
        prompt = (
            "Extract decisions and agreed conclusions from this verified transcript.\n"
            f"{dialogue}\n"
            'Return JSON: {"decisions": [{"decision": "...", "agreed_by": ["..."], "turn_id": "..."}]}'
        )
        payload = self._llm_provider.generate_structured_json(prompt)
        decisions = payload.get("decisions", [])
        return [
            {
                "decision": item["decision"],
                "agreed_by": list(item.get("agreed_by", [])),
                "turn_id": item["turn_id"],
            }
            for item in decisions
            if item.get("decision") and item.get("turn_id")
        ]
