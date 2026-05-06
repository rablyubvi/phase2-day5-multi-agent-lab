"""Researcher agent skeleton."""

from collections import Counter
from typing import Iterable

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.
        """
        query = state.request.query.strip()
        max_sources = state.request.max_sources

        sources = self._filter_sources(self._search(query, max_sources), query, max_sources)
        citations = self._format_citations(sources)
        notes = self._build_notes(query, sources, citations)

        state.sources = sources
        state.research_notes = notes
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=notes,
                metadata={
                    "query": query,
                    "source_count": len(sources),
                    "citations": citations,
                },
            )
        )
        state.add_trace_event(
            "researcher",
            {
                "query": query,
                "source_count": len(sources),
                "citations": citations,
            },
        )
        return state

    def _search(self, query: str, max_sources: int) -> list[SourceDocument]:
        keywords = self._keywords(query)
        query_label = query[:80].strip() or "research topic"

        candidates: list[SourceDocument] = [
            SourceDocument(
                title=f"{query_label}: overview",
                url=None,
                snippet=f"High-level overview of {query}. Focus on the core concepts, definitions, and background context.",
                metadata={"kind": "generated", "angle": "overview"},
            ),
            SourceDocument(
                title=f"{query_label}: tradeoffs",
                url=None,
                snippet=f"Compare the main tradeoffs, constraints, and implementation choices for {query}.",
                metadata={"kind": "generated", "angle": "tradeoffs"},
            ),
            SourceDocument(
                title=f"{query_label}: risks",
                url=None,
                snippet=f"Common risks and limitations to watch when evaluating {query}.",
                metadata={"kind": "generated", "angle": "risks"},
            ),
        ]

        if keywords:
            for keyword in keywords[: max_sources]:
                candidates.append(
                    SourceDocument(
                        title=f"{keyword.title()} background",
                        url=None,
                        snippet=f"{keyword.title()} is relevant to {query}. This note captures a likely supporting angle.",
                        metadata={"kind": "generated", "keyword": keyword},
                    )
                )

        return candidates

    def _filter_sources(
        self,
        sources: Iterable[SourceDocument],
        query: str,
        max_sources: int,
    ) -> list[SourceDocument]:
        keywords = self._keywords(query)
        seen: set[tuple[str, str]] = set()
        scored: list[tuple[int, SourceDocument]] = []

        for source in sources:
            title = source.title.strip()
            url = (source.url or "").strip()
            snippet = source.snippet.strip()
            key = (title.lower(), url.lower())
            if not title or key in seen:
                continue
            seen.add(key)

            text = f"{title} {snippet}".lower()
            score = 0
            score += sum(1 for keyword in keywords if keyword in text)
            score += 2 if url else 1
            score += min(len(snippet.split()) // 8, 3)
            if source.metadata.get("kind") == "generated":
                score += 1

            if score > 0:
                scored.append((score, source))

        scored.sort(key=lambda item: (-item[0], item[1].title.lower()))
        filtered = [source for _, source in scored[:max_sources]]

        if not filtered:
            filtered = [
                SourceDocument(
                    title="Generated background source",
                    url=None,
                    snippet=query,
                    metadata={"kind": "generated"},
                )
            ]

        return filtered

    def _format_citations(self, sources: list[SourceDocument]) -> list[str]:
        citations: list[str] = []
        for index, source in enumerate(sources, start=1):
            citation = f"[{index}] {source.title}"
            if source.url:
                citation = f"{citation} - {source.url}"
            citations.append(citation)
        return citations

    def _build_notes(self, query: str, sources: list[SourceDocument], citations: list[str]) -> str:
        lines = [f"Question: {query}"]
        lines.append(f"Sources reviewed: {len(sources)}")

        if sources:
            lines.append("Key points:")
            for index, source in enumerate(sources, start=1):
                snippet = source.snippet.strip().rstrip(".")
                lines.append(f"- [{index}] {snippet}.")
            lines.append("Citations:")
            lines.extend(citations)
        else:
            lines.append("Key points: insufficient source coverage.")

        return "\n".join(lines)

    @staticmethod
    def _keywords(query: str) -> list[str]:
        words = []
        counts = Counter()
        for raw in query.lower().split():
            token = "".join(ch for ch in raw if ch.isalnum())
            if len(token) < 4:
                continue
            if token not in counts:
                words.append(token)
            counts[token] += 1
        return words
