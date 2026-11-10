from ai_layer.models.knowledge_models import ChangeLogEntry, KnowledgeStore


class ChangeLogger:
    @staticmethod
    def get_session_changes(store: KnowledgeStore, session_id: str) -> list[ChangeLogEntry]:
        return [entry for entry in store.change_log if entry.session_id == session_id]

    @staticmethod
    def append_entry(
        store: KnowledgeStore,
        record_id: str,
        action: str,
        session_id: str,
        details: str,
    ) -> ChangeLogEntry:
        entry = ChangeLogEntry(
            entry_id=f"log_{len(store.change_log) + 1}",
            record_id=record_id,
            action=action,
            session_id=session_id,
            details=details,
        )
        store.change_log.append(entry)
        return entry
