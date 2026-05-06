"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.
        """
        query = state.request.query.strip()
        audience = state.request.audience.strip()
        research = (state.research_notes or "").strip()
        analysis = (state.analysis_notes or "").strip()
        sources = list(state.sources[: state.request.max_sources])

        lines = [f"Question: {query}"]
        lines.append(f"Audience: {audience}")

        if research:
            lines.append("")
            lines.append("Research synthesis:")
            lines.append(research)

        if analysis:
            lines.append("")
            lines.append("Analysis synthesis:")
            lines.append(analysis)

        lines.append("")
        lines.append(f"Final answer: {self._compose_summary(research, analysis, audience)}")

        if sources:
            lines.append("")
            lines.append("Source references:")
            for index, source in enumerate(sources, start=1):
                ref = f"[{index}] {source.title}"
                if source.url:
                    ref = f"{ref} - {source.url}"
                lines.append(ref)
        else:
            lines.append("")
            lines.append("Source references: none available.")

        final_answer = "\n".join(lines)
        state.final_answer = final_answer
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=final_answer,
                metadata={
                    "query": query,
                    "audience": audience,
                    "source_count": len(sources),
                    "has_research": bool(research),
                    "has_analysis": bool(analysis),
                },
            )
        )
        state.add_trace_event(
            "writer",
            {
                "query": query,
                "source_count": len(sources),
                "has_research": bool(research),
                "has_analysis": bool(analysis),
            },
        )
        return state

    def _compose_summary(self, research: str, analysis: str, audience: str) -> str:
        if research and analysis:
            return f"{research} {analysis} Tailor the conclusion for {audience}."
        if research:
            return f"{research} Tailor the conclusion for {audience}."
        if analysis:
            return f"{analysis} Tailor the conclusion for {audience}."
        return f"Insufficient evidence to synthesize a strong answer for {audience}."
