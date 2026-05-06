"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self, timeout_seconds: int | None = None) -> None:
        settings = get_settings()
        self._timeout_seconds = timeout_seconds or settings.timeout_seconds
        self._model = settings.openai_model

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Keep retry, timeout, and token logging here rather than inside agents.
        """
        return self._complete_with_retry(system_prompt, user_prompt)

    @retry(
        retry=retry_if_exception_type((urllib.error.URLError, TimeoutError, RuntimeError)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _complete_with_retry(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url, headers, body = self._build_request(system_prompt, user_prompt)
        payload = self._request_json(method="POST", url=url, headers=headers, body=body)

        content = self._extract_content(payload)
        usage = payload.get("usage") or {}
        return LLMResponse(
            content=content,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    def _build_request(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, dict[str, str], dict[str, object]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        if azure_endpoint and azure_key:
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or self._model
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
            url = (
                f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
                f"?api-version={api_version}"
            )
            return (
                url,
                {"api-key": azure_key, "Content-Type": "application/json"},
                {"messages": messages, "temperature": 0.2},
            )

        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY") or get_settings().openai_api_key
        if not api_key:
            raise StudentTodoError("Missing LLM API key")
        return (
            f"{base_url.rstrip('/')}/chat/completions",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            {"model": self._model, "messages": messages, "temperature": 0.2},
        )

    def _request_json(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict[str, object],
    ) -> dict[str, object]:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM request failed: {exc.code} {detail}") from exc

    def _extract_content(self, payload: dict[str, object]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM response missing choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError("LLM response choice is invalid")
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        text = first.get("text")
        if isinstance(text, str):
            return text
        raise RuntimeError("LLM response missing content")
