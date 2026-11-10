# AI Layer Implementation Plan: Sublayers 1 to 3 & 6 (Decoupled from Database)

## 1. Overview & Core Philosophy

This document outlines the technical implementation plan for the **AI Layer** of the speech processing platform, covering:
- **Sublayer 1**: STT Transcription Service (`transcription/`)
- **Sublayer 2**: Speaker Identity Verification Service (`speaker_verification/`)
- **Sublayer 3**: Conversation Analysis & Insight Extraction Service (`conversation_analysis/`)
- **Sublayer 6**: RAG & User Query Answering Service (`query_answering/`)

### Core Architectural Directives:
1. **Actual Human-Editable Configuration File (`configs/ai_layer_config.yaml`)**: All runtime parameters (model names, prompt templates, temperature, max pause seconds, confidence thresholds, top_k retrieval limits) are stored in an external `.yaml` configuration file. You can change any parameter without touching a single line of Python code. A simple `config_loader.py` parses this file and validates values into typed dataclasses.
2. **Domain-Driven Clean Folder Naming**: Numerical prefixes are omitted (`transcription`, `speaker_verification`, `conversation_analysis`, `query_answering`) for a clean, idiomatic project structure.
3. **Unified Models Architecture (`models/`)**: Eliminated scattered `models.py` files across subfolders. All domain data contracts are centralized under `src/ai_layer/models/`, preventing circular dependencies, eliminating duplicate filenames, and making data flow contracts transparent across the entire pipeline.
4. **Strict Single Responsibility (SRP)**: Each file performs exactly one task (e.g., sorting clips, transcribing, stitching turns, extracting cues, linking citations).
5. **Pure Database Decoupling**: Operates entirely on typed in-memory data models (`dataclasses`). Data is ingested and passed through clean protocol interfaces without coupling to SQL, ORMs, or storage engines.

---

## 2. Directory & File Structure

```text
configs/
└── ai_layer_config.yaml                # The actual configuration file (YAML) for all adjustable parameters

src/ai_layer/
├── __init__.py
│
├── models/                             # Centralized domain models (Zero scattering, zero circular imports)
│   ├── __init__.py                     # Re-exports all models for easy top-level imports
│   ├── base_models.py                  # Single responsibility: Shared primitives (TimeRange, SessionMetadata)
│   ├── transcription_models.py         # Single responsibility: AudioClip, RawUtterance, StitchedTurn, TranscriptOutput
│   ├── speaker_models.py               # Single responsibility: SpeakerProfile, IdentityCue, SpeakerReassignment, VerifiedTranscript
│   ├── analysis_models.py              # Single responsibility: ExtractedFact, ExtractedDecision, ActionItem, AnalysisReport
│   └── query_models.py                 # Single responsibility: UserQuery, QueryIntent, RetrievedSnippet, QueryAnswer
│
├── common/                             # Shared utilities, interfaces & adapters
│   ├── __init__.py
│   ├── config_loader.py                # Single responsibility: Loads & parses ai_layer_config.yaml into typed dataclasses
│   ├── llm_provider.py                 # Single responsibility: Abstract protocol & client wrapper for LLM requests
│   ├── stt_provider.py                 # Single responsibility: Abstract protocol & client wrapper for STT inference
│   └── exceptions.py                   # Single responsibility: Domain exception hierarchy for AI pipeline failures
│
├── transcription/                      # Sublayer 1: STT Transcription
│   ├── __init__.py
│   ├── clip_sorter.py                  # Single responsibility: Chronological ordering & timestamp validation
│   ├── audio_transcriber.py            # Single responsibility: Calling STT provider on individual audio clips
│   ├── turn_stitcher.py                # Single responsibility: Merging adjacent same-speaker utterances
│   └── service.py                      # Single responsibility: Orchestrating transcription flow (Sort -> Transcribe -> Stitch)
│
├── speaker_verification/               # Sublayer 2: Speaker Identity Verification
│   ├── __init__.py
│   ├── cue_extractor.py                # Single responsibility: Scanning dialogue for self-introductions & vocatives via LLM
│   ├── identity_resolver.py            # Single responsibility: Mapping cues to known profiles & resolving ID conflicts
│   ├── confidence_scorer.py            # Single responsibility: Calculating confidence scores for speaker mappings
│   └── service.py                      # Single responsibility: Orchestrating verification flow (Extract Cues -> Resolve -> Score)
│
├── conversation_analysis/              # Sublayer 3: Conversation Analysis & Insight Extraction
│   ├── __init__.py
│   ├── fact_extractor.py               # Single responsibility: Extracting factual statements from transcript
│   ├── decision_extractor.py           # Single responsibility: Extracting agreed choices and conclusions
│   ├── action_item_extractor.py        # Single responsibility: Extracting action items, assignees, and deadlines
│   ├── statement_normalizer.py         # Single responsibility: Rewriting pronouns into standalone explicit entities
│   ├── citation_linker.py              # Single responsibility: Mapping extracted items to source utterance timestamps
│   └── service.py                      # Single responsibility: Orchestrating analysis flow (Extract -> Normalize -> Link Citations)
│
├── query_answering/                    # Sublayer 6: RAG & User Query Answering
│   ├── __init__.py
│   ├── query_intent_analyzer.py        # Single responsibility: Deconstructing query intent, entities, and time filters
│   ├── in_memory_retriever.py          # Single responsibility: Filtering & ranking transcript and insight records in memory
│   ├── citation_resolver.py            # Single responsibility: Matching candidate facts to exact speaker/timestamp citations
│   ├── answer_synthesizer.py           # Single responsibility: Composing grounded answer with citations or fallback
│   └── service.py                      # Single responsibility: Orchestrating query flow (Analyze Intent -> Retrieve -> Synthesize)
│
└── pipeline_orchestrator.py            # Single responsibility: End-to-end pipeline coordination (1 -> 2 -> 3 -> 6)
```

---

## 3. Configuration Specification (`configs/ai_layer_config.yaml`)

The actual configuration file lives in `configs/ai_layer_config.yaml`. No Python code needs to be modified to change parameters:

```yaml
# configs/ai_layer_config.yaml

llm:
  model_name: "gpt-4o"
  temperature: 0.1
  timeout_seconds: 30.0
  max_retries: 3
  api_key_env_var: "LLM_API_KEY"

stt:
  model_name: "whisper-large-v3"
  language: "en"
  device: "cpu"

transcription:
  max_pause_seconds: 1.5
  allow_overlap: true

speaker_verification:
  min_confidence_threshold: 0.70
  boost_per_confirmation: 0.15

conversation_analysis:
  enable_normalization: true
  max_facts_per_session: 50

query_answering:
  retrieval_top_k: 5
  min_relevance_score: 0.5
  fallback_message: "The provided conversation records do not contain this information."
```

### Python Loader (`src/ai_layer/common/config_loader.py`)
- **Single Responsibility**: Read `ai_layer_config.yaml`, overlay any environment variables (from `.env` or system environment), and return a typed `AILayerConfig` object.
- **Functions**:
  - `load_config(config_file_path: str = "configs/ai_layer_config.yaml") -> AILayerConfig`

---

## 4. Centralized Domain Models (`src/ai_layer/models/`)

All data contracts are centralized under `models/`, eliminating duplicate file names (`models.py`) and avoiding circular dependencies.

#### 1. `models/base_models.py`
- **Single Responsibility**: Primitive value objects shared across multiple sublayers.
- **Classes**:
  - `TimeRange`: `start_time_seconds: float`, `end_time_seconds: float`, `duration_seconds -> float`
  - `SessionMetadata`: `session_id: str`, `session_title: str`, `recorded_at: str`, `participant_ids: list[str]`

#### 2. `models/transcription_models.py`
- **Single Responsibility**: Data structures representing raw audio inputs, individual utterances, and stitched dialogue turns.
- **Classes**:
  - `AudioClip`: `clip_id: str`, `audio_path: str`, `preliminary_speaker_id: str`, `time_range: TimeRange`
  - `RawUtterance`: `utterance_id: str`, `speaker_id: str`, `text: str`, `time_range: TimeRange`, `audio_path: str`
  - `StitchedTurn`: `turn_id: str`, `speaker_id: str`, `text: str`, `time_range: TimeRange`, `source_utterance_ids: list[str]`
  - `STTTranscriptOutput`: `session_id: str`, `turns: list[StitchedTurn]`, `raw_utterances: list[RawUtterance]`

#### 3. `models/speaker_models.py`
- **Single Responsibility**: Data structures for speaker profiles, dialogue identity cues, and verified transcripts.
- **Classes**:
  - `SpeakerProfile`: `speaker_id: str`, `canonical_name: str`, `aliases: list[str]`, `base_confidence: float`
  - `IdentityCue`: `cue_type: Literal['self_intro', 'direct_address', 'third_party']`, `detected_name: str`, `target_speaker_id: str`, `source_turn_id: str`, `confidence: float`
  - `SpeakerReassignment`: `preliminary_speaker_id: str`, `resolved_speaker_id: str`, `resolved_name: str`, `rationale: str`
  - `VerifiedTurn`: `turn_id: str`, `speaker_id: str`, `speaker_name: str`, `text: str`, `time_range: TimeRange`
  - `VerifiedTranscript`: `session_id: str`, `turns: list[VerifiedTurn]`, `reassignments: list[SpeakerReassignment]`

#### 4. `models/analysis_models.py`
- **Single Responsibility**: Data structures for extracted insights, tasks, decisions, and citation evidence.
- **Classes**:
  - `UtteranceCitation`: `turn_id: str`, `speaker_name: str`, `time_range: TimeRange`, `spoken_text: str`
  - `ExtractedFact`: `fact_id: str`, `statement: str`, `citation: UtteranceCitation`
  - `ExtractedDecision`: `decision_id: str`, `decision: str`, `agreed_by: list[str]`, `citation: UtteranceCitation`
  - `ActionItem`: `task_id: str`, `task_description: str`, `assignee_name: str`, `deadline: str | None`, `citation: UtteranceCitation`
  - `AnalysisReport`: `session_id: str`, `facts: list[ExtractedFact]`, `decisions: list[ExtractedDecision]`, `action_items: list[ActionItem]`

#### 5. `models/query_models.py`
- **Single Responsibility**: Data structures for queries, search intents, retrieved contexts, citations, and syntheses.
- **Classes**:
  - `UserQuery`: `query_text: str`, `target_session_id: str | None`
  - `QueryIntent`: `intent_category: Literal['decision', 'task', 'fact', 'general']`, `target_person: str | None`, `target_topic: str | None`
  - `RetrievedSnippet`: `snippet_id: str`, `content: str`, `speaker_name: str`, `time_range: TimeRange`, `relevance_score: float`
  - `AnswerCitation`: `speaker_name: str`, `timestamp_str: str`, `verbatim_quote: str`
  - `QueryAnswer`: `answer_text: str`, `citations: list[AnswerCitation]`, `is_found: bool`

#### 6. `models/__init__.py`
- **Single Responsibility**: Re-exports all models for easy top-level imports:
  `from ai_layer.models import AudioClip, VerifiedTranscript, ExtractedFact, QueryAnswer`

---

## 5. Sublayer Service Breakdown

### 5.1. Common Services & Interfaces (`src/ai_layer/common/`)
- `config_loader.py`: `load_config(path: str) -> AILayerConfig`.
- `llm_provider.py`: Protocol `LLMProvider` (`generate_text`, `generate_structured_json`) + `MockLLMProvider`.
- `stt_provider.py`: Protocol `STTProvider` (`transcribe_clip`) + `MockSTTProvider`.
- `exceptions.py`: Hierarchy: `AILayerError`, `TranscriptionError`, `SpeakerVerificationError`, `AnalysisExtractionError`, `QueryAnsweringError`.

### 5.2. Transcription (`src/ai_layer/transcription/`)
- `clip_sorter.py`: `ClipSorter.sort_by_start_time(clips)`, `validate_timestamps(clips)`.
- `audio_transcriber.py`: `AudioTranscriber.transcribe_single_clip(clip)`, `transcribe_batch(clips)` using `STTProvider`.
- `turn_stitcher.py`: `TurnStitcher.stitch_consecutive_turns(utterances)` using `max_pause_seconds` from config.
- `service.py`: `STTTranscriptionService.process_session_clips(session_id, clips)`.

### 5.3. Speaker Verification (`src/ai_layer/speaker_verification/`)
- `cue_extractor.py`: `IdentityCueExtractor.extract_cues_from_turns(turns)` using `LLMProvider`.
- `identity_resolver.py`: `IdentityResolver.resolve_identities(cues, known_profiles, preliminary_speaker_ids)`.
- `confidence_scorer.py`: `SpeakerConfidenceScorer.compute_confidence(profile, cues)` using `min_confidence_threshold` from config.
- `service.py`: `SpeakerVerificationService.verify_speakers(transcript, registered_profiles)`.

### 5.4. Conversation Analysis (`src/ai_layer/conversation_analysis/`)
- `fact_extractor.py`: `FactExtractor.extract_facts(turns)`.
- `decision_extractor.py`: `DecisionExtractor.extract_decisions(turns)`.
- `action_item_extractor.py`: `ActionItemExtractor.extract_action_items(turns)`.
- `statement_normalizer.py`: `StatementNormalizer.normalize_to_self_contained(raw_statement, speaker_name, context_window)`.
- `citation_linker.py`: `CitationLinker.link_citation(turn)`, `find_matching_turn(turns, turn_id)`.
- `service.py`: `ConversationAnalysisService.analyze_conversation(verified_transcript, session_metadata)`.

### 5.5. Query Answering (`src/ai_layer/query_answering/`)
- `query_intent_analyzer.py`: `QueryIntentAnalyzer.analyze_intent(user_query)`.
- `in_memory_retriever.py`: `InMemoryRetriever.retrieve_relevant_turns(intent, transcripts)`, `retrieve_relevant_insights(intent, reports)` using `retrieval_top_k` from config.
- `citation_resolver.py`: `CitationResolver.resolve_citations(snippets)`.
- `answer_synthesizer.py`: `AnswerSynthesizer.synthesize_answer(query, snippets, citations)` using `fallback_message` from config.
- `service.py`: `UserQueryService.answer_query(query, transcripts, reports)`.

### 5.6. Pipeline Orchestrator (`src/ai_layer/pipeline_orchestrator.py`)
- `AILayerOrchestrator`:
  - `process_audio_session(session_metadata, clips, known_speakers) -> tuple[VerifiedTranscript, AnalysisReport]`
  - `query_session_memory(query_text, transcripts, reports) -> QueryAnswer`

---

## 6. End-to-End Data Flow

```text
ai_layer_config.yaml (Human-editable parameters)
               │
               ▼ (loaded by common/config_loader.py)
[AudioClips from Voice Separation]
               │
               ▼
┌─────────────────────────────────────────┐
│ transcription/                          │
│  - ClipSorter (sort & validate)         │
│  - AudioTranscriber (STT inference)     │
│  - TurnStitcher (merge consecutive)     │
└──────────────────┬──────────────────────┘
                   │ STTTranscriptOutput (from models/transcription_models.py)
                   ▼
┌─────────────────────────────────────────┐
│ speaker_verification/                   │
│  - IdentityCueExtractor (scan text)     │
│  - IdentityResolver (resolve IDs)       │
│  - SpeakerConfidenceScorer (score)      │
└──────────────────┬──────────────────────┘
                   │ VerifiedTranscript (from models/speaker_models.py)
                   ▼
┌─────────────────────────────────────────┐
│ conversation_analysis/                  │
│  - Fact / Decision / Action Extractors  │
│  - StatementNormalizer (clear pronouns) │
│  - CitationLinker (link timestamps)     │
└──────────────────┬──────────────────────┘
                   │ VerifiedTranscript + AnalysisReport (from models/analysis_models.py)
                   ▼ (Pure In-Memory Context)
┌─────────────────────────────────────────┐
│ query_answering/                        │
│  - QueryIntentAnalyzer (parse question) │
│  - InMemoryRetriever (search memory)    │
│  - CitationResolver (link evidence)     │
│  - AnswerSynthesizer (generate answer)  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
       QueryAnswer with Citations (from models/query_models.py)
```

---

## 7. Verification & Testing Strategy

Each component is validated with targeted unit tests using the deterministic mock providers and configuration fixtures:

1. **Config Validation**:
   - `test_config_loader_reads_yaml`: Ensures `ai_layer_config.yaml` is parsed correctly.
   - `test_config_env_override`: Ensures environment variables can override YAML values.

2. **Transcription Tests (`tests/test_transcription.py`)**:
   - `test_clip_sorter`: Verifies chronological sorting and timestamp validation.
   - `test_audio_transcriber`: Verifies batch audio clip transcription into `RawUtterance`.
   - `test_turn_stitcher`: Verifies merging contiguous speaker turns respecting `max_pause_seconds`.
   - `test_transcription_service`: End-to-end clip-to-transcript flow.

3. **Speaker Verification Tests (`tests/test_speaker_verification.py`)**:
   - `test_cue_extractor`: Detects self-introductions ("Hi, I'm Sarah") and vocatives ("Alex, what do you think?").
   - `test_identity_resolver`: Reassigns preliminary speaker IDs (e.g., `Speaker_2` -> `Sarah`).
   - `test_confidence_scorer`: Verifies confidence rating adjustments against `min_confidence_threshold`.
   - `test_speaker_verification_service`: End-to-end verified transcript generation.

4. **Conversation Analysis Tests (`tests/test_conversation_analysis.py`)**:
   - `test_fact_extractor`: Extracts factual statements.
   - `test_decision_extractor`: Extracts finalized decisions.
   - `test_action_item_extractor`: Extracts tasks, assignees, and deadlines.
   - `test_statement_normalizer`: Normalizes context-dependent pronouns into entity names.
   - `test_citation_linker`: Verifies exact timestamp links back to source dialogue turns.

5. **Query Answering Tests (`tests/test_query_answering.py`)**:
   - `test_query_intent_analyzer`: Deconstructs query intent and extracts target entities.
   - `test_in_memory_retriever`: Ranks relevant turns and facts from memory pools.
   - `test_citation_resolver`: Resolves quotes and timestamp formatting.
   - `test_answer_synthesizer`: Generates grounded answers with citations.
   - `test_answer_synthesizer_fallback`: Issues explicit configured fallback message when evidence is missing.

6. **Pipeline Orchestrator Integration (`tests/test_pipeline_orchestrator.py`)**:
   - Runs mock audio clips through `transcription` -> `speaker_verification` -> `conversation_analysis`, and asserts successful query answering via `query_answering`.
