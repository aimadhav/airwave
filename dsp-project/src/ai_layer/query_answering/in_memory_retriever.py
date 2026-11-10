import re

from ai_layer.common.config_loader import QueryAnsweringConfig
from ai_layer.models.analysis_models import AnalysisReport
from ai_layer.models.query_models import QueryIntent, RetrievedSnippet
from ai_layer.models.speaker_models import VerifiedTranscript

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "tell", "show", "say", "said",
    "saying", "regard", "regarding", "discuss", "discussed", "mention", "mentioned"
}


META_TOPIC_WORDS = {
    "task", "tasks", "action", "actions", "item", "items",
    "decision", "decisions", "fact", "facts", "thing", "things", "topic", "topics"
}


class InMemoryRetriever:
    def __init__(self, config: QueryAnsweringConfig) -> None:
        self._config = config

    def retrieve_relevant_turns(
        self,
        intent: QueryIntent,
        transcripts: list[VerifiedTranscript],
        query_text: str = "",
    ) -> list[RetrievedSnippet]:
        snippets: list[RetrievedSnippet] = []
        for transcript in transcripts:
            for turn in transcript.turns:
                score = self._score_text(
                    intent=intent,
                    text=turn.text,
                    speaker_name=turn.speaker_name,
                    query_text=query_text,
                    item_type="turn",
                )
                if score >= self._config.min_relevance_score:
                    snippets.append(
                        RetrievedSnippet(
                            snippet_id=f"{transcript.session_id}_{turn.turn_id}",
                            content=turn.text,
                            speaker_name=turn.speaker_name,
                            time_range=turn.time_range,
                            relevance_score=score,
                            session_id=transcript.session_id,
                        )
                    )
        snippets.sort(key=lambda item: item.relevance_score, reverse=True)
        return snippets[: self._config.retrieval_top_k]

    def retrieve_relevant_insights(
        self,
        intent: QueryIntent,
        reports: list[AnalysisReport],
        query_text: str = "",
    ) -> list[RetrievedSnippet]:
        snippets: list[RetrievedSnippet] = []
        for report in reports:
            if intent.intent_category in {"fact", "general"}:
                snippets.extend(self._insights_from_facts(report, intent, query_text))
            if intent.intent_category in {"decision", "general"}:
                snippets.extend(self._insights_from_decisions(report, intent, query_text))
            if intent.intent_category in {"task", "general"}:
                snippets.extend(self._insights_from_tasks(report, intent, query_text))

        filtered = [
            snippet
            for snippet in snippets
            if snippet.relevance_score >= self._config.min_relevance_score
        ]
        filtered.sort(key=lambda item: item.relevance_score, reverse=True)
        return filtered[: self._config.retrieval_top_k]

    def _insights_from_facts(
        self,
        report: AnalysisReport,
        intent: QueryIntent,
        query_text: str = "",
    ) -> list[RetrievedSnippet]:
        snippets: list[RetrievedSnippet] = []
        for fact in report.facts:
            score = self._score_text(
                intent=intent,
                text=fact.statement,
                speaker_name=fact.citation.speaker_name,
                query_text=query_text,
                item_type="fact",
            )
            snippets.append(
                RetrievedSnippet(
                    snippet_id=fact.fact_id,
                    content=fact.statement,
                    speaker_name=fact.citation.speaker_name,
                    time_range=fact.citation.time_range,
                    relevance_score=score,
                    session_id=report.session_id,
                )
            )
        return snippets

    def _insights_from_decisions(
        self,
        report: AnalysisReport,
        intent: QueryIntent,
        query_text: str = "",
    ) -> list[RetrievedSnippet]:
        snippets: list[RetrievedSnippet] = []
        for decision in report.decisions:
            speaker_str = decision.citation.speaker_name
            if decision.agreed_by:
                speaker_str = f"{speaker_str}, {', '.join(decision.agreed_by)}"
            score = self._score_text(
                intent=intent,
                text=decision.decision,
                speaker_name=speaker_str,
                query_text=query_text,
                item_type="decision",
            )
            snippets.append(
                RetrievedSnippet(
                    snippet_id=decision.decision_id,
                    content=decision.decision,
                    speaker_name=decision.citation.speaker_name,
                    time_range=decision.citation.time_range,
                    relevance_score=score,
                    session_id=report.session_id,
                )
            )
        return snippets

    def _insights_from_tasks(
        self,
        report: AnalysisReport,
        intent: QueryIntent,
        query_text: str = "",
    ) -> list[RetrievedSnippet]:
        snippets: list[RetrievedSnippet] = []
        for task in report.action_items:
            content = f"{task.assignee_name}: {task.task_description}"
            if task.deadline:
                content += f" (Deadline: {task.deadline})"
            score = self._score_text(
                intent=intent,
                text=content,
                speaker_name=f"{task.citation.speaker_name} (Assignee: {task.assignee_name})",
                query_text=query_text,
                item_type="task",
            )
            snippets.append(
                RetrievedSnippet(
                    snippet_id=task.task_id,
                    content=content,
                    speaker_name=task.assignee_name or task.citation.speaker_name,
                    time_range=task.citation.time_range,
                    relevance_score=score,
                    session_id=report.session_id,
                )
            )
        return snippets

    @staticmethod
    def _score_text(
        intent: QueryIntent,
        text: str,
        speaker_name: str,
        query_text: str = "",
        item_type: str = "turn",
    ) -> float:
        score = 0.0
        lowered_text = text.lower()
        lowered_speaker = speaker_name.lower()

        # 1. Content/Keyword Match
        query_tokens: set[str] = set()
        if query_text:
            raw_tokens = re.findall(r"[a-z0-9]+", query_text.lower())
            query_tokens.update(t for t in raw_tokens if t not in STOP_WORDS and len(t) >= 2)
        if intent.target_topic:
            topic_tokens = re.findall(r"[a-z0-9]+", intent.target_topic.lower())
            query_tokens.update(t for t in topic_tokens if t not in STOP_WORDS and len(t) >= 2)

        # Do not count target_person as a general keyword token
        if intent.target_person:
            p_tokens = set(re.findall(r"[a-z0-9]+", intent.target_person.lower()))
            query_tokens.difference_update(p_tokens)

        matched_tokens = 0
        if query_tokens:
            matched_tokens = sum(
                1 for token in query_tokens if token in lowered_text or token in lowered_speaker
            )
            overlap_ratio = matched_tokens / len(query_tokens)
            score += min(0.60, matched_tokens * 0.25 + overlap_ratio * 0.35)

            # Topic boost if exact target topic appears verbatim
            if intent.target_topic:
                clean_topic = intent.target_topic.lower().strip()
                if clean_topic and (clean_topic in lowered_text or clean_topic in lowered_speaker):
                    score += 0.35

        # If specific non-meta topic tokens were searched for, but this snippet has ZERO matches,
        # it is not relevant to the topic regardless of who spoke it.
        specific_tokens = query_tokens.difference(META_TOPIC_WORDS)
        if specific_tokens and not any(t in lowered_text or t in lowered_speaker for t in specific_tokens):
            return 0.0

        # 2. Speaker / Target Person Match
        if intent.target_person:
            target_p = intent.target_person.lower()
            if target_p in lowered_speaker:
                score += 0.35
            elif target_p in lowered_text:
                score += 0.20
            else:
                if query_text and any(q in query_text.lower() for q in ("what did", "say about", "opinion of")):
                    score -= 0.15

        # 3. Intent Category Match
        if intent.intent_category == item_type:
            score += 0.30

        if intent.intent_category == "decision":
            if any(marker in lowered_text for marker in ("decid", "reschedule", "launch", "agreed", "agrees")):
                score += 0.20
        elif intent.intent_category == "task":
            if any(marker in lowered_text for marker in ("will", "prepare", "commit", "action", "deadline", "task", "assign")):
                score += 0.20
        elif intent.intent_category == "fact":
            if any(marker in lowered_text for marker in ("verified", "confirmed", "found", "completed")):
                score += 0.20

        return min(1.0, max(0.0, score))

