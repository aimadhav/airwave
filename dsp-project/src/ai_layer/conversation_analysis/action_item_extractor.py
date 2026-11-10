from ai_layer.common.llm_provider import LLMProvider
from ai_layer.models.speaker_models import VerifiedTurn


class ActionItemExtractor:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def extract_action_items(self, turns: list[VerifiedTurn]) -> list[dict[str, str | None]]:
        if not turns:
            return []

        dialogue = "\n".join(
            f"[{turn.turn_id}] {turn.speaker_name}: {turn.text}" for turn in turns
        )
        prompt = (
            "Extract action items, assignees, and deadlines from this verified transcript.\n"
            f"{dialogue}\n"
            'Return JSON: {"action_items": [{"task_description": "...", '
            '"assignee_name": "...", "deadline": "...", "turn_id": "..."}]}'
        )
        payload = self._llm_provider.generate_structured_json(prompt)
        action_items = payload.get("action_items", [])
        return [
            {
                "task_description": item["task_description"],
                "assignee_name": item.get("assignee_name", "Unassigned"),
                "deadline": item.get("deadline"),
                "turn_id": item["turn_id"],
            }
            for item in action_items
            if item.get("task_description") and item.get("turn_id")
        ]
