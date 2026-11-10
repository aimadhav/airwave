from ai_layer.models.analysis_models import ActionItem, AnalysisReport, ExtractedDecision, ExtractedFact
from ai_layer.models.knowledge_models import (
    ChangeLogEntry,
    KnowledgeRecord,
    KnowledgeRelationship,
    KnowledgeStore,
    RelationshipType,
)


class RecordMerger:
    def merge_report(
        self,
        report: AnalysisReport,
        store: KnowledgeStore,
        relationships: list[tuple[str, KnowledgeRecord | None, RelationshipType, str]],
    ) -> KnowledgeStore:
        next_index = len(store.records) + 1

        for record_type, items in (
            ("fact", report.facts),
            ("decision", report.decisions),
            ("action_item", report.action_items),
        ):
            for item in items:
                statement, turn_id = self._extract_statement_and_turn(item, record_type)
                relationship = next(
                    (
                        rel
                        for rel in relationships
                        if rel[0] == statement
                    ),
                    (statement, None, "novel", "No prior matching record found."),
                )
                _, existing, relationship_type, rationale = relationship

                record_id = f"record_{next_index}"
                next_index += 1
                new_record = KnowledgeRecord(
                    record_id=record_id,
                    record_type=record_type,  # type: ignore[arg-type]
                    statement=statement,
                    session_id=report.session_id,
                    source_turn_id=turn_id,
                    status="active",
                    related_record_ids=[existing.record_id] if existing else [],
                )

                if existing and relationship_type == "supersede":
                    store.records = [
                        KnowledgeRecord(
                            record_id=record.record_id,
                            record_type=record.record_type,
                            statement=record.statement,
                            session_id=record.session_id,
                            source_turn_id=record.source_turn_id,
                            status="superseded",
                            related_record_ids=record.related_record_ids,
                        )
                        if record.record_id == existing.record_id
                        else record
                        for record in store.records
                    ]

                if existing and relationship_type == "contradict":
                    new_record = KnowledgeRecord(
                        record_id=new_record.record_id,
                        record_type=new_record.record_type,
                        statement=new_record.statement,
                        session_id=new_record.session_id,
                        source_turn_id=new_record.source_turn_id,
                        status="flagged",
                        related_record_ids=[existing.record_id],
                    )

                store.records.append(new_record)
                store.change_log.append(
                    ChangeLogEntry(
                        entry_id=f"log_{len(store.change_log) + 1}",
                        record_id=record_id,
                        action=relationship_type,
                        session_id=report.session_id,
                        details=rationale,
                    )
                )

                if existing:
                    store.relationships.append(
                        KnowledgeRelationship(
                            relationship_id=f"rel_{len(store.relationships) + 1}",
                            source_record_id=record_id,
                            target_record_id=existing.record_id,
                            relationship_type=relationship_type,
                            rationale=rationale,
                        )
                    )

        return store

    @staticmethod
    def _extract_statement_and_turn(
        item: ExtractedFact | ExtractedDecision | ActionItem,
        record_type: str,
    ) -> tuple[str, str]:
        if record_type == "fact":
            fact = item  # type: ignore[assignment]
            return fact.statement, fact.citation.turn_id
        if record_type == "decision":
            decision = item  # type: ignore[assignment]
            return decision.decision, decision.citation.turn_id
        action = item  # type: ignore[assignment]
        return action.task_description, action.citation.turn_id
