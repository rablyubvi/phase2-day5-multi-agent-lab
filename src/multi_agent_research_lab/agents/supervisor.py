"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"
    max_iterations = 6

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.
        """
        if state.iteration >= self.max_iterations or state.final_answer:
            route = "done"
        elif not state.research_notes:
            route = "researcher"
        elif not state.analysis_notes:
            route = "analyst"
        elif not state.final_answer:
            route = "writer"
        else:
            route = "done"

        state.record_route(route)
        state.add_trace_event(
            "route",
            {
                "next": route,
                "iteration": state.iteration,
                "query": state.request.query,
            },
        )
        return state
