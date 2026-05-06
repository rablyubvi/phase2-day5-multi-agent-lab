"""Command-line entrypoint for the lab starter."""

from typing import Annotated
from time import perf_counter

import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _estimate_cost_usd(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None

    rates: dict[str, tuple[float, float]] = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (5.00, 15.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1": (2.00, 8.00),
    }
    prompt_rate, completion_rate = rates.get(model, (0.0, 0.0))
    return ((input_tokens * prompt_rate) + (output_tokens * completion_rate)) / 1_000_000


def _quality_score(answer: str, source_count: int = 0, analysis_count: int = 0) -> float:
    score = 0.0
    if answer.strip():
        score += 2.0
    if len(answer) >= 200:
        score += 2.0
    if len(answer) >= 500:
        score += 1.5
    if source_count:
        score += min(3.0, source_count * 0.5)
    if analysis_count:
        score += min(2.5, analysis_count * 0.75)
    return min(score, 10.0)


def _print_result(title: str, answer: str, metrics: BenchmarkMetrics) -> None:
    console.print(Panel.fit(answer, title=title))
    console.print(Pretty(metrics.model_dump(), expand_all=False))


def _run_single_agent(query: str) -> tuple[str, BenchmarkMetrics]:
    settings = get_settings()
    client = LLMClient()
    request = ResearchQuery(query=query)
    started = perf_counter()
    with trace_span("single-agent.run", {"query": request.query, "audience": request.audience}) as span:
        response = client.complete(
            "You are a precise research assistant. Give a clear, direct answer with caveats.",
            (
                f"Research this query for {request.audience}: {request.query}\n"
                "Write a concise summary with key findings, tradeoffs, and a short conclusion."
            ),
        )
    latency = perf_counter() - started
    cost = _estimate_cost_usd(settings.openai_model, response.input_tokens, response.output_tokens)
    quality = _quality_score(response.content)
    span["attributes"]["metrics"] = {
        "latency_seconds": latency,
        "estimated_cost_usd": cost,
        "quality_score": quality,
    }
    metrics = BenchmarkMetrics(
        run_name="single-agent",
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        notes="Estimated quality is heuristic.",
    )
    return response.content, metrics


def _run_multi_agent(query: str) -> tuple[str, BenchmarkMetrics]:
    settings = get_settings()
    client = LLMClient()
    request = ResearchQuery(query=query)
    started = perf_counter()
    with trace_span("multi-agent.run", {"query": request.query, "audience": request.audience}) as span:
        with trace_span("multi-agent.research"):
            research = client.complete(
                "You are the Researcher. Extract sources, facts, and open questions.",
                (
                    f"Query: {request.query}\n"
                    f"Audience: {request.audience}\n"
                    "Return concise research notes with bullets and source references if available."
                ),
            )
        with trace_span("multi-agent.analysis"):
            analysis = client.complete(
                "You are the Analyst. Critique the research notes and synthesize implications.",
                (
                    f"Query: {request.query}\n\nResearch notes:\n{research.content}\n"
                    "Return a short analysis that organizes the strongest claims and caveats."
                ),
            )
        with trace_span("multi-agent.writer"):
            final = client.complete(
                "You are the Writer. Produce the final answer from research and analysis.",
                (
                    f"Query: {request.query}\n"
                    f"Audience: {request.audience}\n\nResearch notes:\n{research.content}\n\n"
                    f"Analysis:\n{analysis.content}\n"
                    "Write the final answer in a polished, readable form."
                ),
            )

    latency = perf_counter() - started
    total_input_tokens = sum(
        token for token in (research.input_tokens, analysis.input_tokens, final.input_tokens) if token is not None
    )
    total_output_tokens = sum(
        token for token in (research.output_tokens, analysis.output_tokens, final.output_tokens) if token is not None
    )
    cost = _estimate_cost_usd(settings.openai_model, total_input_tokens, total_output_tokens)
    quality = _quality_score(final.content, source_count=1, analysis_count=1)
    span["attributes"]["metrics"] = {
        "latency_seconds": latency,
        "estimated_cost_usd": cost,
        "quality_score": quality,
    }
    metrics = BenchmarkMetrics(
        run_name="multi-agent",
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        notes="Estimated quality is heuristic.",
    )
    return final.content, metrics


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline."""

    _init()
    try:
        answer, metrics = _run_single_agent(query)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Error", style="red"))
        raise typer.Exit(code=2) from exc
    _print_result("Single-Agent Baseline", answer, metrics)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    try:
        answer, metrics = _run_multi_agent(query)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Error", style="red"))
        raise typer.Exit(code=2) from exc
    _print_result("Multi-Agent Run", answer, metrics)


if __name__ == "__main__":
    app()
