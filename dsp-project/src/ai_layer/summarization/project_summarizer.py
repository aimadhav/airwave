from ai_layer.common.llm_provider import LLMProvider
from ai_layer.models.summary_models import ProjectSummary, SessionSummary


class ProjectSummarizer:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def update_project_summary(
        self,
        project_id: str,
        project_title: str,
        session_summary: SessionSummary,
        existing_summary: ProjectSummary | None = None,
    ) -> ProjectSummary:
        prompt = (
            "Update the project status overview using the latest session summary.\n"
            f"Project: {project_title}\n"
            f"Latest session overview: {session_summary.overview}\n"
            f"Decisions: {session_summary.decisions_summary}\n"
            f"Action items: {session_summary.action_items_summary}"
        )
        status_overview = self._llm_provider.generate_text(prompt)

        session_ids = []
        if existing_summary:
            session_ids = list(existing_summary.session_ids)
        if session_summary.session_id not in session_ids:
            session_ids.append(session_summary.session_id)

        open_action_items = session_summary.action_items_summary
        if existing_summary:
            open_action_items = list(existing_summary.open_action_items) + open_action_items

        return ProjectSummary(
            project_id=project_id,
            title=project_title,
            status_overview=status_overview,
            latest_milestones=session_summary.decisions_summary,
            open_action_items=open_action_items,
            session_ids=session_ids,
        )
