"""Shared state for the multi-agent workflow.

Students should extend this file when adding new agents, outputs, or evaluation metrics.
"""

from typing import Any

from pydantic import BaseModel, Field

from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery, SourceDocument


class ResearchState(BaseModel):
    """Single source of truth passed through the workflow."""

    request: ResearchQuery
    iteration: int = 0
    route_history: list[str] = Field(default_factory=list)

    sources: list[SourceDocument] = Field(default_factory=list)
    research_notes: str | None = None
    analysis_notes: str | None = None
    final_answer: str | None = None
    critic_notes: str | None = None

    agent_results: list[AgentResult] = Field(default_factory=list)
    agent_outputs: dict[AgentName, str] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def record_route(self, route: str) -> None:
        self.route_history.append(route)
        self.iteration += 1

    def add_trace_event(self, name: str, payload: dict[str, Any]) -> None:
        self.trace.append({"name": name, "payload": payload})

    def record_agent_result(
        self,
        agent: AgentName,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        result = AgentResult(agent=agent, content=content, metadata=metadata or {})
        self.agent_results.append(result)
        self.agent_outputs[agent] = content

        if agent == AgentName.RESEARCHER:
            self.research_notes = content
        elif agent == AgentName.ANALYST:
            self.analysis_notes = content
        elif agent == AgentName.WRITER:
            self.final_answer = content
        elif agent == AgentName.CRITIC:
            self.critic_notes = content

        return result
