from ai_layer.common.config_loader import load_config


def test_config_loader_reads_yaml():
    config = load_config("configs/ai_layer_config.yaml")

    assert config.llm.model_name == "gpt-4o"
    assert config.stt.model_name in {"whisper-1", "whisper-large-v3"}
    assert config.transcription.max_pause_seconds == 1.5
    assert config.speaker_verification.min_confidence_threshold == 0.70
    assert config.query_answering.retrieval_top_k == 5
