from ai_layer.common.config_loader import KnowledgeMemoryConfig
from ai_layer.common.llm_provider import LLMProvider
from ai_layer.knowledge_memory.record_merger import RecordMerger
from ai_layer.knowledge_memory.relationship_classifier import RelationshipClassifier
from ai_layer.models.analysis_models import AnalysisReport
from ai_layer.models.knowledge_models import KnowledgeRecord, KnowledgeStore, RelationshipType


class KnowledgeMemoryService:
    def __init__(self, llm_provider: LLMProvider, config: KnowledgeMemoryConfig) -> None:
        self._classifier = RelationshipClassifier(llm_provider, config)
        self._merger = RecordMerger()

    def integrate_session_insights(
        self,
        report: AnalysisReport,
        store: KnowledgeStore | None = None,
    ) -> KnowledgeStore:
        working_store = store or KnowledgeStore()
        relationships: list[tuple[str, KnowledgeRecord | None, RelationshipType, str]] = []

        for statement in self._statements_from_report(report):
            existing = self._find_best_match(statement, working_store.records)
            relationship_type, rationale = self._classifier.classify(statement, existing)
            relationships.append((statement, existing, relationship_type, rationale))

        return self._merger.merge_report(report, working_store, relationships)

    @staticmethod
    def _statements_from_report(report: AnalysisReport) -> list[str]:
        return (
            [fact.statement for fact in report.facts]
            + [decision.decision for decision in report.decisions]
            + [task.task_description for task in report.action_items]
        )

    @staticmethod
    def _find_best_match(
        statement: str,
        records: list[KnowledgeRecord],
    ) -> KnowledgeRecord | None:
        if not records:
            return None

        def score(record: KnowledgeRecord) -> float:
            left = set(statement.lower().split())
            right = set(record.statement.lower().split())
            if not left or not right:
                return 0.0
            return len(left.intersection(right)) / max(len(left), len(right))

        ranked = sorted(records, key=score, reverse=True)
        best = ranked[0]
        return best if score(best) > 0 else None
