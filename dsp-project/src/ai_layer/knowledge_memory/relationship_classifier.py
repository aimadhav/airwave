import re

from ai_layer.common.config_loader import KnowledgeMemoryConfig
from ai_layer.common.llm_provider import LLMProvider
from ai_layer.models.knowledge_models import KnowledgeRecord, RelationshipType


class RelationshipClassifier:
    def __init__(self, llm_provider: LLMProvider, config: KnowledgeMemoryConfig) -> None:
        self._llm_provider = llm_provider
        self._config = config

    def classify(
        self,
        new_statement: str,
        existing_record: KnowledgeRecord | None,
    ) -> tuple[RelationshipType, str]:
        if existing_record is None:
            return "novel", "No prior matching record found."

        similarity = self._token_similarity(new_statement, existing_record.statement)
        if similarity < self._config.similarity_threshold:
            return "novel", "Statement topic does not match existing records."

        lowered_new = new_statement.lower()
        lowered_old = existing_record.statement.lower()
        if lowered_new == lowered_old:
            return "reinforce", "New statement confirms an existing record."

        if self._looks_like_update(lowered_new, lowered_old):
            return "supersede", "New statement updates a prior record."

        if self._config.enable_contradiction_flagging and self._looks_contradictory(
            lowered_new, lowered_old
        ):
            return "contradict", "New statement conflicts with an existing record."

        return "elaborate", "New statement adds detail to an existing record."

    @staticmethod
    def _token_similarity(left: str, right: str) -> float:
        left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
        right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = left_tokens.intersection(right_tokens)
        return len(overlap) / max(len(left_tokens), len(right_tokens))

    @staticmethod
    def _looks_like_update(new_text: str, old_text: str) -> bool:
        update_markers = ("changed", "updated", "reschedule", "switch", "instead", "now")
        return any(marker in new_text for marker in update_markers) and any(
            token in old_text for token in re.findall(r"[a-z0-9]+", new_text)
        )

    @staticmethod
    def _looks_contradictory(new_text: str, old_text: str) -> bool:
        negations = ("not", "never", "no longer", "cancel")
        return any(marker in new_text for marker in negations) and RelationshipClassifier._token_similarity(
            new_text, old_text
        ) >= 0.4
