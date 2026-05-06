"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""

        final_answer = (state.final_answer or "").strip()
        sources = list(state.sources)
        findings: list[str] = []

        if not final_answer:
            findings.append("No final answer available for review.")
        else:
            if sources:
                cited_sources = 0
                for source in sources[: state.request.max_sources]:
                    title = source.title.strip()
                    url = (source.url or "").strip()
                    if (title and title in final_answer) or (url and url in final_answer):
                        cited_sources += 1

                if cited_sources == 0:
                    findings.append("Citation coverage is weak: no source titles or URLs are referenced.")
                elif cited_sources < len(sources):
                    findings.append(
                        f"Citation coverage is partial: {cited_sources}/{len(sources)} sources are referenced."
                    )
            else:
                findings.append("No sources available, so claims cannot be fact-checked.")

            if state.research_notes and "insufficient" in state.research_notes.lower():
                findings.append("Hallucination risk: answer was produced after weak research notes.")
            if state.analysis_notes and "weak evidence" in state.analysis_notes.lower():
                findings.append("Hallucination risk: analysis notes still indicate weak evidence.")

        if not findings:
            findings.append("No obvious issues detected.")

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content="\n".join(findings),
                metadata={
                    "findings": findings,
                    "has_final_answer": bool(final_answer),
                    "source_count": len(sources),
                },
            )
        )
        state.add_trace_event(
            "critic",
            {
                "findings": findings,
                "source_count": len(sources),
                "has_final_answer": bool(final_answer),
            },
        )
        state.errors.extend(f for f in findings if f != "No obvious issues detected.")
        return state
