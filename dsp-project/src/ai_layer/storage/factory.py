from ai_layer.common.config_loader import AILayerConfig
from ai_layer.common.provider_factory import create_embedding_provider
from ai_layer.storage.sqlite_repository import SQLiteRepository


def create_repository(config: AILayerConfig) -> SQLiteRepository:
    embedder = create_embedding_provider(config)
    return SQLiteRepository(
        db_path=config.database.sqlite_path,
        embedding_provider=embedder,
    )
