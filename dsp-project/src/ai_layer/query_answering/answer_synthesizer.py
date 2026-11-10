from ai_layer.common.config_loader import QueryAnsweringConfig
from ai_layer.common.llm_provider import LLMProvider
from ai_layer.models.query_models import AnswerCitation, QueryAnswer, RetrievedSnippet, UserQuery


class AnswerSynthesizer:
    def __init__(self, llm_provider: LLMProvider, config: QueryAnsweringConfig) -> None:
        self._llm_provider = llm_provider
        self._config = config

    def synthesize_answer(
        self,
        query: UserQuery,
        snippets: list[RetrievedSnippet],
        citations: list[AnswerCitation],
    ) -> QueryAnswer:
        if not snippets:
            return QueryAnswer(
                answer_text=self._config.fallback_message,
                citations=[],
                is_found=False,
            )

        evidence = "\n".join(
            f"- {snippet.speaker_name}: {snippet.content}" for snippet in snippets
        )
        prompt = (
            "Answer the user query using only the provided evidence.\n"
            f"Query: {query.query_text}\n"
            f"Evidence:\n{evidence}"
        )
        answer_text = self._llm_provider.generate_text(prompt)
        return QueryAnswer(
            answer_text=answer_text,
            citations=citations,
            is_found=True,
        )
