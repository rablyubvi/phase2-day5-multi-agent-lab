"""LangGraph workflow skeleton."""

from __future__ import annotations

from typing import Any, Callable

from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.core.state import ResearchState

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - optional dependency fallback
    END = "__end__"
    START = "__start__"

    class _CompiledGraph:
        def __init__(self, workflow: "_FallbackGraph") -> None:
            self._workflow = workflow

        def invoke(self, state: ResearchState) -> ResearchState:
            current = self._workflow.entry_point
            while current != END:
                updates = self._workflow.nodes[current](state)
                if updates:
                    state = state.model_copy(update=updates)
                if current in self._workflow.conditional_edges:
                    router, mapping = self._workflow.conditional_edges[current]
                    current = mapping[router(state)]
                else:
                    current = self._workflow.edges.get(current, END)
            return state

    class _FallbackGraph:
        def __init__(self, _: type[ResearchState]) -> None:
            self.nodes: dict[str, Callable[[ResearchState], dict[str, Any]]] = {}
            self.edges: dict[str, str] = {}
            self.conditional_edges: dict[str, tuple[Callable[[ResearchState], str], dict[str, str]]] = {}
            self.entry_point = END

        def add_node(self, name: str, func: Callable[[ResearchState], dict[str, Any]]) -> None:
            self.nodes[name] = func

        def add_edge(self, source: str, target: str) -> None:
            if source == START:
                self.entry_point = target
            else:
                self.edges[source] = target

        def add_conditional_edges(
            self,
            source: str,
            router: Callable[[ResearchState], str],
            mapping: dict[str, str],
        ) -> None:
            self.conditional_edges[source] = (router, mapping)

        def compile(self) -> _CompiledGraph:
            workflow = self

            class _Runtime:
                def invoke(self, state: ResearchState) -> ResearchState:
                    current = workflow.entry_point
                    while current != END:
                        updates = workflow.nodes[current](state)
                        if updates:
                            state = state.model_copy(update=updates)
                        if current in workflow.conditional_edges:
                            router, mapping = workflow.conditional_edges[current]
                            current = mapping[router(state)]
                        else:
                            current = workflow.edges.get(current, END)
                    return state

            return _Runtime()

    StateGraph = _FallbackGraph


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> object:
        """Create a LangGraph graph."""

        def supervisor(state: ResearchState) -> dict[str, Any]:
            if state.iteration >= 6 or state.final_answer:
                route = "done"
            elif not state.research_notes:
                route = "researcher"
            elif not state.analysis_notes:
                route = "analyst"
            else:
                route = "writer"

            next_iteration = state.iteration + 1
            return {
                "iteration": next_iteration,
                "route_history": [*state.route_history, route],
                "trace": [
                    *state.trace,
                    {
                        "name": "route",
                        "payload": {
                            "next": route,
                            "iteration": next_iteration,
                            "query": state.request.query,
                        },
                    },
                ],
            }

        def researcher(state: ResearchState) -> dict[str, Any]:
            sources = list(state.sources[: state.request.max_sources])
            if not sources:
                sources = [
                    SourceDocument(
                        title="Initial background source",
                        url=None,
                        snippet=state.request.query,
                        metadata={"kind": "generated"},
                    )
                ]
            return {
                "research_notes": (
                    f"Key research points for '{state.request.query}': "
                    f"survey core concepts, compare tradeoffs, and collect up to {state.request.max_sources} sources."
                ),
                "sources": sources,
                "trace": [
                    *state.trace,
                    {"name": "researcher", "payload": {"query": state.request.query}},
                ],
            }

        def analyst(state: ResearchState) -> dict[str, Any]:
            base = state.research_notes or state.request.query
            return {
                "analysis_notes": (
                    f"Analysis: synthesize evidence from research and identify decision points around {base}."
                ),
                "trace": [
                    *state.trace,
                    {"name": "analyst", "payload": {"has_research": bool(state.research_notes)}},
                ],
            }

        def writer(state: ResearchState) -> dict[str, Any]:
            research = state.research_notes or state.request.query
            analysis = state.analysis_notes or "no analysis notes available"
            return {
                "final_answer": (
                    f"{research}\n\n{analysis}\n\nFinal answer: concise synthesis for {state.request.audience}."
                ),
                "trace": [
                    *state.trace,
                    {"name": "writer", "payload": {"has_analysis": bool(state.analysis_notes)}},
                ],
            }

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", supervisor)
        graph.add_node("researcher", researcher)
        graph.add_node("analyst", analyst)
        graph.add_node("writer", writer)
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state.route_history[-1],
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        return graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        compiled = self.build().compile()
        result = compiled.invoke(state)
        return result if isinstance(result, ResearchState) else ResearchState.model_validate(result)
