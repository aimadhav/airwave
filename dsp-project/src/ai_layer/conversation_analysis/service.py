from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ai_layer.common.config_loader import ConversationAnalysisConfig
from ai_layer.common.llm_provider import LLMProvider
from ai_layer.conversation_analysis.action_item_extractor import ActionItemExtractor
from ai_layer.conversation_analysis.citation_linker import CitationLinker
from ai_layer.conversation_analysis.decision_extractor import DecisionExtractor
from ai_layer.conversation_analysis.fact_extractor import FactExtractor
from ai_layer.conversation_analysis.statement_normalizer import StatementNormalizer
from ai_layer.models.analysis_models import (
    ActionItem,
    AnalysisReport,
    ExtractedDecision,
    ExtractedFact,
)
from ai_layer.models.base_models import SessionMetadata
from ai_layer.models.speaker_models import VerifiedTranscript, VerifiedTurn


class ConversationAnalysisService:
    def __init__(self, llm_provider: LLMProvider, config: ConversationAnalysisConfig) -> None:
        self._config = config
        self._llm_provider = llm_provider
        self._fact_extractor = FactExtractor(llm_provider)
        self._decision_extractor = DecisionExtractor(llm_provider)
        self._action_item_extractor = ActionItemExtractor(llm_provider)
        self._normalizer = StatementNormalizer(llm_provider, config.enable_normalization)
        self._citation_linker = CitationLinker()

    def _extract_all_insights(self, turns: list[VerifiedTurn]) -> dict[str, Any]:
        """Extracts facts, decisions, and action items in a single LLM call."""
        if not turns:
            return {"facts": [], "decisions": [], "action_items": []}

        dialogue = "\n".join(
            f"[{turn.turn_id}] {turn.speaker_name}: {turn.text}" for turn in turns
        )
        prompt = (
            "Analyze this verified conversation transcript and extract structured conversation intelligence in ONE pass.\n\n"
            f"Transcript:\n{dialogue}\n\n"
            "Extract the following three sections:\n"
            "1. Facts: Verified factual statements, status updates, or constraints mentioned.\n"
            "2. Decisions: Explicit agreements, choices, or consensus reached by participants.\n"
            "3. Action Items: Specific tasks, commitments, or assignments, including the assigned person and deadline if mentioned.\n\n"
            "CRITICAL FORMAT RULES:\n"
            "- Make all statements and task descriptions self-contained (explicitly use person names instead of vague pronouns like he/she/they).\n"
            "- Set turn_id to the exact turn where the statement/decision/task was spoken (e.g. 'turn_1').\n\n"
            "Return valid JSON:\n"
            "{\n"
            '  "facts": [{"statement": "...", "turn_id": "turn_1"}],\n'
            '  "decisions": [{"decision": "...", "agreed_by": ["..."], "turn_id": "turn_2"}],\n'
            '  "action_items": [{"task_description": "...", "assignee_name": "...", "deadline": "..." or null, "turn_id": "turn_3"}]\n'
            "}"
        )
        try:
            payload = self._llm_provider.generate_structured_json(prompt)
            if isinstance(payload, dict) and any(k in payload for k in ("facts", "decisions", "action_items")):
                return payload
        except Exception:
            pass
        return {}

    def _build_facts(
        self,
        raw_facts: list[dict[str, Any]],
        turns: list[VerifiedTurn],
        session_id: str,
    ) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        for index, item in enumerate(raw_facts, start=1):
            turn = self._citation_linker.find_matching_turn(turns, str(item.get("turn_id", ""))) or (turns[0] if turns else None)
            statement = str(item.get("statement", "")).strip()
            if not statement or turn is None:
                continue
            facts.append(
                ExtractedFact(
                    fact_id=f"fact_{session_id}_{index}",
                    statement=statement,
                    citation=self._citation_linker.link_citation(turn),
                )
            )
        return facts

    def _build_decisions(
        self,
        raw_decisions: list[dict[str, Any]],
        turns: list[VerifiedTurn],
        session_id: str,
    ) -> list[ExtractedDecision]:
        decisions: list[ExtractedDecision] = []
        for index, item in enumerate(raw_decisions, start=1):
            turn = self._citation_linker.find_matching_turn(turns, str(item.get("turn_id", ""))) or (turns[0] if turns else None)
            decision_text = str(item.get("decision", "")).strip()
            if not decision_text or turn is None:
                continue
            decisions.append(
                ExtractedDecision(
                    decision_id=f"decision_{session_id}_{index}",
                    decision=decision_text,
                    agreed_by=[str(name) for name in item.get("agreed_by", [])],
                    citation=self._citation_linker.link_citation(turn),
                )
            )
        return decisions

    def _build_action_items(
        self,
        raw_action_items: list[dict[str, Any]],
        turns: list[VerifiedTurn],
        session_id: str,
    ) -> list[ActionItem]:
        action_items: list[ActionItem] = []
        for index, item in enumerate(raw_action_items, start=1):
            turn = self._citation_linker.find_matching_turn(turns, str(item.get("turn_id", ""))) or (turns[0] if turns else None)
            task_description = str(item.get("task_description", "")).strip()
            if not task_description or turn is None:
                continue
            action_items.append(
                ActionItem(
                    task_id=f"task_{session_id}_{index}",
                    task_description=task_description,
                    assignee_name=str(item.get("assignee_name") or (turn.speaker_name if turn else "Unassigned")),
                    deadline=item.get("deadline") if item.get("deadline") else None,
                    citation=self._citation_linker.link_citation(turn),
                )
            )
        return action_items

    def analyze_conversation(
        self,
        verified_transcript: VerifiedTranscript,
        session_metadata: SessionMetadata,
        on_facts_ready: Callable[[list[ExtractedFact]], None] | None = None,
        on_decisions_ready: Callable[[list[ExtractedDecision]], None] | None = None,
        on_action_items_ready: Callable[[list[ActionItem]], None] | None = None,
        concurrent_extractors: bool = True,
    ) -> AnalysisReport:
        turns = verified_transcript.turns
        if not turns:
            return AnalysisReport(session_id=verified_transcript.session_id)

        session_id = session_metadata.session_id

        if concurrent_extractors:
            # Run tasks, decisions, and facts concurrently with streaming callbacks
            facts: list[ExtractedFact] = []
            decisions: list[ExtractedDecision] = []
            action_items: list[ActionItem] = []

            def run_tasks() -> list[ActionItem]:
                nonlocal action_items
                try:
                    raw = self._action_item_extractor.extract_action_items(turns)
                    built = self._build_action_items(raw, turns, session_id)
                    action_items = built
                    if on_action_items_ready:
                        on_action_items_ready(built)
                    return built
                except Exception as exc:
                    print(f"[!] Warning: Action item extraction error: {exc}")
                    return []

            def run_decisions() -> list[ExtractedDecision]:
                nonlocal decisions
                try:
                    raw = self._decision_extractor.extract_decisions(turns)
                    built = self._build_decisions(raw, turns, session_id)
                    decisions = built
                    if on_decisions_ready:
                        on_decisions_ready(built)
                    return built
                except Exception as exc:
                    print(f"[!] Warning: Decision extraction error: {exc}")
                    return []

            def run_facts() -> list[ExtractedFact]:
                nonlocal facts
                try:
                    raw = self._fact_extractor.extract_facts(turns)[: self._config.max_facts_per_session]
                    built = self._build_facts(raw, turns, session_id)
                    facts = built
                    if on_facts_ready:
                        on_facts_ready(built)
                    return built
                except Exception as exc:
                    print(f"[!] Warning: Fact extraction error: {exc}")
                    return []

            with ThreadPoolExecutor(max_workers=3) as executor:
                t_future = executor.submit(run_tasks)
                d_future = executor.submit(run_decisions)
                f_future = executor.submit(run_facts)

                t_future.result()
                d_future.result()
                f_future.result()

            # If all 3 returned empty (e.g. offline mock or unexpected output), try unified as fallback
            if not facts and not decisions and not action_items:
                raw_insights = self._extract_all_insights(turns)
                if raw_insights:
                    facts = self._build_facts(raw_insights.get("facts", []), turns, session_id)[: self._config.max_facts_per_session]
                    decisions = self._build_decisions(raw_insights.get("decisions", []), turns, session_id)
                    action_items = self._build_action_items(raw_insights.get("action_items", []), turns, session_id)

            # If still empty (e.g. LLM API failed or error occurred), run heuristic rule-based extractor
            if not facts and not decisions and not action_items:
                facts, decisions, action_items = self._heuristic_extract_insights(turns, session_id)

            if on_action_items_ready and action_items:
                on_action_items_ready(action_items)
            if on_decisions_ready and decisions:
                on_decisions_ready(decisions)
            if on_facts_ready and facts:
                on_facts_ready(facts)

            return AnalysisReport(
                session_id=verified_transcript.session_id,
                facts=facts,
                decisions=decisions,
                action_items=action_items,
            )

        # Non-concurrent sequential path
        raw_insights = self._extract_all_insights(turns)
        raw_facts = raw_insights.get("facts", [])[: self._config.max_facts_per_session]
        raw_decisions = raw_insights.get("decisions", [])
        raw_action_items = raw_insights.get("action_items", [])

        if not raw_facts and not raw_decisions and not raw_action_items:
            raw_facts = self._fact_extractor.extract_facts(turns)[: self._config.max_facts_per_session]
            raw_decisions = self._decision_extractor.extract_decisions(turns)
            raw_action_items = self._action_item_extractor.extract_action_items(turns)

        facts = self._build_facts(raw_facts, turns, session_id)
        decisions = self._build_decisions(raw_decisions, turns, session_id)
        action_items = self._build_action_items(raw_action_items, turns, session_id)

        if not facts and not decisions and not action_items:
            facts, decisions, action_items = self._heuristic_extract_insights(turns, session_id)

        if on_action_items_ready and action_items:
            on_action_items_ready(action_items)
        if on_decisions_ready and decisions:
            on_decisions_ready(decisions)
        if on_facts_ready and facts:
            on_facts_ready(facts)

        return AnalysisReport(
            session_id=verified_transcript.session_id,
            facts=facts,
            decisions=decisions,
            action_items=action_items,
        )

    def _heuristic_extract_insights(
        self, turns: list[VerifiedTurn], session_id: str
    ) -> tuple[list[ExtractedFact], list[ExtractedDecision], list[ActionItem]]:
        """Heuristic rule-based fallback when LLM extractors are unavailable or offline."""
        action_items: list[ActionItem] = []
        decisions: list[ExtractedDecision] = []
        facts: list[ExtractedFact] = []

        import re

        action_triggers = re.compile(
            r"\b(will|shall|going to|need to|let me|action item|todo|prepare|review|finalize|deploy|coordinate|send|update|schedule)\b",
            re.IGNORECASE,
        )
        decision_triggers = re.compile(
            r"\b(decided|agreed|target|let's|lets|approved|consensus|plan is to|confirmed|commit to|reschedule)\b",
            re.IGNORECASE,
        )
        fact_triggers = re.compile(
            r"\b(completed|finished|verified|renewed|signed|contract|milestone|status|fixed|launched|deployed|report)\b",
            re.IGNORECASE,
        )
        deadline_regex = re.compile(
            r"\b(by\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next\s+week|eod|[\d:]+\s*(?:am|pm)?))\b",
            re.IGNORECASE,
        )

        for turn in turns:
            text = turn.text.strip()
            if not text:
                continue

            # Check action items
            if action_triggers.search(text):
                dl_match = deadline_regex.search(text)
                deadline_str = dl_match.group(1).strip() if dl_match else None
                action_items.append(
                    ActionItem(
                        task_id=f"task_{session_id}_{len(action_items) + 1}",
                        task_description=text,
                        assignee_name=turn.speaker_name or "Unassigned",
                        deadline=deadline_str,
                        citation=self._citation_linker.link_citation(turn),
                    )
                )

            # Check decisions
            if decision_triggers.search(text):
                decisions.append(
                    ExtractedDecision(
                        decision_id=f"decision_{session_id}_{len(decisions) + 1}",
                        decision=text,
                        agreed_by=[turn.speaker_name] if turn.speaker_name else ["Team"],
                        citation=self._citation_linker.link_citation(turn),
                    )
                )

            # Check facts
            if fact_triggers.search(text):
                facts.append(
                    ExtractedFact(
                        fact_id=f"fact_{session_id}_{len(facts) + 1}",
                        statement=text,
                        citation=self._citation_linker.link_citation(turn),
                    )
                )

        return facts, decisions, action_items
