"""Benchmark skeleton for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def _estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    rates: dict[str, tuple[float, float]] = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (5.00, 15.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1": (2.00, 8.00),
    }
    prompt_rate, completion_rate = rates.get(model, (0.0, 0.0))
    if prompt_rate == 0.0 and completion_rate == 0.0:
        return None
    return ((input_tokens * prompt_rate) + (output_tokens * completion_rate)) / 1_000_000


def _quality_score(state: ResearchState) -> float:
    answer = state.final_answer or ""
    score = 0.0
    if answer.strip():
        score += 2.0
    if len(answer) >= 200:
        score += 2.0
    if len(answer) >= 500:
        score += 1.5
    if state.sources:
        score += min(3.0, len(state.sources) * 0.5)
    if state.analysis_notes:
        score += 1.5
    if state.critic_notes:
        score += 1.0
    return min(score, 10.0)


def _citation_coverage(state: ResearchState) -> float:
    source_count = len(state.sources)
    if source_count == 0:
        return 0.0
    text = " ".join(filter(None, [state.research_notes, state.analysis_notes, state.final_answer, state.critic_notes])).lower()
    cited = 0
    for source in state.sources:
        title = source.title.lower().strip()
        url = (source.url or "").lower().strip()
        if (title and title in text) or (url and url in text):
            cited += 1
    return cited / source_count


def _error_rate(state: ResearchState) -> float:
    denominator = max(1, state.iteration)
    return min(1.0, len(state.errors) / denominator)


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and derive benchmark metrics."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    settings = get_settings()
    input_tokens = _estimate_tokens(query)
    input_tokens += _estimate_tokens(state.research_notes)
    input_tokens += _estimate_tokens(state.analysis_notes)
    output_tokens = _estimate_tokens(state.final_answer)
    output_tokens += _estimate_tokens(state.critic_notes)
    cost = _estimate_cost_usd(settings.openai_model, input_tokens, output_tokens)
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=_quality_score(state),
        citation_coverage=_citation_coverage(state),
        error_rate=_error_rate(state),
    )
    return state, metrics
