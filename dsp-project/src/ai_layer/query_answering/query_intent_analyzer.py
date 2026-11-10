import re
from ai_layer.common.llm_provider import LLMProvider
from ai_layer.models.query_models import QueryIntent, QueryIntentCategory, UserQuery


class QueryIntentAnalyzer:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def analyze_intent(self, user_query: UserQuery) -> QueryIntent:
        prompt = (
            "Analyze query intent and extract entities.\n"
            f"Query: {user_query.query_text}\n"
            'Return JSON: {"intent_category": "decision|task|fact|general", '
            '"target_person": "...", "target_topic": "..."}'
        )
        try:
            payload = self._llm_provider.generate_structured_json(prompt)
        except Exception:
            payload = {}

        category = payload.get("intent_category") if isinstance(payload, dict) else None
        if category not in {"decision", "task", "fact", "general"}:
            category = self._heuristic_category(user_query.query_text)

        target_person = payload.get("target_person") if isinstance(payload, dict) else None
        if not target_person:
            target_person = self._heuristic_person(user_query.query_text)

        target_topic = payload.get("target_topic") if isinstance(payload, dict) else None
        if not target_topic:
            target_topic = self._heuristic_topic(user_query.query_text)

        return QueryIntent(
            intent_category=category,  # type: ignore[arg-type]
            target_person=target_person,
            target_topic=target_topic,
        )

    @staticmethod
    def _heuristic_category(query_text: str) -> QueryIntentCategory:
        lowered = query_text.lower()
        if "decide" in lowered or "decision" in lowered or "agree" in lowered or "launch" in lowered:
            return "decision"
        if "task" in lowered or "commit" in lowered or "action" in lowered or "assign" in lowered or "who will" in lowered or "deadline" in lowered:
            return "task"
        if "fact" in lowered or "when" in lowered or "what happened" in lowered or "what is" in lowered or "how does" in lowered:
            return "fact"
        return "general"

    @staticmethod
    def _heuristic_person(query_text: str) -> str | None:
        person_patterns = [
            r"\bwhat did\s+([A-Z][a-zA-Z]+)\s+(?:say|think|mention|suggest|commit|do|decide|agree|work on)",
            r"\bopinion of\s+([A-Z][a-zA-Z]+)\b",
            r"\baccording to\s+([A-Z][a-zA-Z]+)\b",
            r"\bassigned to\s+([A-Z][a-zA-Z]+)\b",
            r"\bwhat will\s+([A-Z][a-zA-Z]+)\s+(?:do|work on|commit|deliver|prepare|handle)",
            r"\btasks? for\s+([A-Z][a-zA-Z]+)\b",
            r"\b([A-Z][a-zA-Z]+)'s\s+(?:task|action|opinion|decision|role|view|commitment)",
        ]
        for pat in person_patterns:
            m = re.search(pat, query_text, re.IGNORECASE)
            if m:
                candidate = m.group(1).capitalize()
                if candidate.lower() not in {"what", "who", "when", "where", "how", "the", "this", "that", "we", "they", "our", "all", "is"}:
                    return candidate
        return None

    @staticmethod
    def _heuristic_topic(query_text: str) -> str | None:
        cleaned = re.sub(
            r"^(?:what did we decide about|what was decided about|what did \w+ say about|what was discussed about|what did \w+ mention about|what are the tasks for|what is|tell me about|what about|who is assigned to|can you explain)\s+",
            "",
            query_text.strip(),
            flags=re.IGNORECASE,
        ).rstrip("?.,! ")
        if cleaned and len(cleaned) >= 2:
            return cleaned
        return None
