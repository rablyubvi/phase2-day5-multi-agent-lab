"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.
        """
        query = state.request.query.strip()
        research = (state.research_notes or "").strip()
        sources = list(state.sources)

        lines = [f"Question: {query}"]
        lines.append(f"Core claim: {research}" if research else "Core claim: insufficient research notes.")

        if sources:
            lines.append("Source comparison:")
            snippets = []
            for index, source in enumerate(sources[: state.request.max_sources], start=1):
                snippet = source.snippet.strip()
                snippets.append(snippet)
                weakness = []
                if not source.url:
                    weakness.append("no URL")
                if len(snippet.split()) < 8:
                    weakness.append("thin snippet")
                if source.metadata.get("kind") == "generated":
                    weakness.append("generated placeholder")
                status = "strong" if not weakness else f"weak ({', '.join(weakness)})"
                lines.append(f"- {index}. {source.title}: {status}")

            unique_snippets = {snippet for snippet in snippets if snippet}
            if len(unique_snippets) > 1:
                lines.append("Viewpoints: sources emphasize different angles or tradeoffs.")
            else:
                lines.append("Viewpoints: sources are broadly aligned.")
        else:
            lines.append("Source comparison: no sources available.")

        weak_signals: list[str] = []
        for source in sources:
            if not source.url:
                weak_signals.append(source.title)
            elif len(source.snippet.split()) < 8:
                weak_signals.append(source.title)
            elif source.metadata.get("kind") == "generated":
                weak_signals.append(source.title)

        if weak_signals:
            lines.append(f"Weak evidence: {', '.join(weak_signals[:3])}.")
        else:
            lines.append("Weak evidence: none obvious from available notes.")

        state.analysis_notes = "\n".join(lines)
        return state
