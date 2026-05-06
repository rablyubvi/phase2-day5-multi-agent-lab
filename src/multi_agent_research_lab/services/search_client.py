"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client with local fallback."""

    def __init__(self, timeout_seconds: int | None = None, docs_root: Path | None = None) -> None:
        settings = get_settings()
        self._timeout_seconds = timeout_seconds or settings.timeout_seconds
        self._docs_root = docs_root or Path(__file__).resolve().parents[3]

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        query = query.strip()
        if not query:
            return []

        for provider in (self._search_tavily, self._search_bing, self._search_serpapi):
            results = provider(query, max_results)
            if results:
                return results[:max_results]

        results = self._search_internal_docs(query, max_results)
        if results:
            return results[:max_results]

        return self._search_local_mock(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        api_key = get_settings().tavily_api_key
        if not api_key:
            return []

        payload = self._post_json(
            "https://api.tavily.com/search",
            {
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
        )
        results = payload.get("results")
        return self._normalize_results(results, "tavily")

    def _search_bing(self, query: str, max_results: int) -> list[SourceDocument]:
        api_key = os.getenv("BING_SEARCH_API_KEY") or os.getenv("BING_API_KEY")
        endpoint = os.getenv("BING_SEARCH_ENDPOINT") or "https://api.bing.microsoft.com/v7.0/search"
        if not api_key:
            return []

        url = f"{endpoint.rstrip('/')}?{urllib.parse.urlencode({'q': query, 'count': max_results})}"
        payload = self._request_json(
            url,
            headers={"Ocp-Apim-Subscription-Key": api_key},
        )
        web_pages = payload.get("webPages", {})
        results = web_pages.get("value", []) if isinstance(web_pages, dict) else []
        return self._normalize_results(results, "bing")

    def _search_serpapi(self, query: str, max_results: int) -> list[SourceDocument]:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return []

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": str(max_results),
        }
        url = f"https://serpapi.com/search.json?{urllib.parse.urlencode(params)}"
        payload = self._request_json(url)
        results = payload.get("organic_results", [])
        return self._normalize_results(results, "serpapi")

    def _search_internal_docs(self, query: str, max_results: int) -> list[SourceDocument]:
        candidates: list[tuple[int, SourceDocument]] = []
        keywords = self._keywords(query)
        for path in self._candidate_doc_paths():
            text = self._read_text(path)
            if not text:
                continue
            title = self._title_for_path(path)
            score = self._score_text(text, keywords)
            if score <= 0:
                continue
            snippet = self._extract_snippet(text, keywords)
            candidates.append(
                (
                    score,
                    SourceDocument(
                        title=title,
                        url=str(path),
                        snippet=snippet,
                        metadata={"kind": "internal-doc", "path": str(path)},
                    ),
                )
            )

        candidates.sort(key=lambda item: (-item[0], item[1].title.lower()))
        return [source for _, source in candidates[:max_results]]

    def _search_local_mock(self, query: str, max_results: int) -> list[SourceDocument]:
        label = query[:80] or "research topic"
        return [
            SourceDocument(
                title=f"{label}: overview",
                url=None,
                snippet=f"High-level overview of {query}. Focus on core definitions and background context.",
                metadata={"kind": "mock", "angle": "overview"},
            ),
            SourceDocument(
                title=f"{label}: tradeoffs",
                url=None,
                snippet=f"Compare the main tradeoffs, constraints, and implementation choices for {query}.",
                metadata={"kind": "mock", "angle": "tradeoffs"},
            ),
            SourceDocument(
                title=f"{label}: risks",
                url=None,
                snippet=f"Identify common risks, limitations, and open questions when evaluating {query}.",
                metadata={"kind": "mock", "angle": "risks"},
            ),
        ][:max_results]

    def _normalize_results(self, results: Any, kind: str) -> list[SourceDocument]:
        if not isinstance(results, list):
            return []

        normalized: list[SourceDocument] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = self._first_string(item, "title", "name") or "Untitled result"
            url = self._first_string(item, "url", "link", "webpage_url")
            snippet = self._first_string(item, "snippet", "content", "description", "body") or ""
            if not snippet and url:
                snippet = url
            normalized.append(
                SourceDocument(
                    title=title.strip(),
                    url=url.strip() if url else None,
                    snippet=snippet.strip(),
                    metadata={"kind": kind, "raw": {k: v for k, v in item.items() if isinstance(k, str)}},
                )
            )
        return [source for source in normalized if source.title and source.snippet]

    def _post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request_json(request)

    def _request_json(
        self,
        url: str | urllib.request.Request,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if isinstance(url, str):
            request = urllib.request.Request(url=url, headers=headers or {}, method="GET")
        else:
            request = url
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _candidate_doc_paths(self) -> list[Path]:
        roots = [self._docs_root, self._docs_root / "docs"]
        candidates: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            if not root.exists():
                continue
            for pattern in ("README.md", "docs/*.md", "docs/**/*.md"):
                for path in root.glob(pattern):
                    key = str(path.resolve()).lower()
                    if key in seen or not path.is_file():
                        continue
                    seen.add(key)
                    candidates.append(path)
        return candidates

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""

    def _score_text(self, text: str, keywords: list[str]) -> int:
        if not keywords:
            return 0
        haystack = text.lower()
        return sum(haystack.count(keyword) for keyword in keywords)

    def _extract_snippet(self, text: str, keywords: list[str], limit: int = 220) -> str:
        if not text:
            return ""
        lower = text.lower()
        for keyword in keywords:
            index = lower.find(keyword)
            if index >= 0:
                start = max(0, index - 80)
                end = min(len(text), index + 140)
                return " ".join(text[start:end].split())[:limit]
        return " ".join(text.split())[:limit]

    @staticmethod
    def _title_for_path(path: Path) -> str:
        if path.name.lower() == "readme.md":
            return "Repository README"
        return path.stem.replace("_", " ").replace("-", " ").title()

    @staticmethod
    def _keywords(query: str) -> list[str]:
        tokens: list[str] = []
        seen: set[str] = set()
        for raw in query.lower().split():
            token = "".join(ch for ch in raw if ch.isalnum())
            if len(token) < 3 or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens

    @staticmethod
    def _first_string(item: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None
