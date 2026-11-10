from ai_layer.models.query_models import AnswerCitation, RetrievedSnippet


class CitationResolver:
    @staticmethod
    def resolve_citations(snippets: list[RetrievedSnippet]) -> list[AnswerCitation]:
        citations: list[AnswerCitation] = []
        for snippet in snippets:
            citations.append(
                AnswerCitation(
                    speaker_name=snippet.speaker_name,
                    timestamp_str=CitationResolver.format_timestamp(snippet.time_range.start_time_seconds),
                    verbatim_quote=snippet.content,
                    session_id=snippet.session_id,
                    session_title=snippet.session_title,
                )
            )
        return citations

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{remaining:02d}"
        return f"{minutes:02d}:{remaining:02d}"
