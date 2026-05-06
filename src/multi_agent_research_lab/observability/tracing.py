"""Tracing hooks with optional LangSmith/Langfuse provider binding."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
from os import environ
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from multi_agent_research_lab.core.config import get_settings

try:  # pragma: no cover - optional dependency
    from langfuse import Langfuse
except ImportError:  # pragma: no cover - optional dependency
    Langfuse = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from langsmith import Client as LangSmithClient
except ImportError:  # pragma: no cover - optional dependency
    LangSmithClient = None  # type: ignore[assignment]

_LANGFUSE_CLIENT: Any | None = None
_LANGSMITH_CLIENT: Any | None = None


def _trace_jsonl_path() -> Path:
    raw = environ.get("TRACE_JSONL_PATH", "reports/traces/trace.jsonl")
    return Path(raw)


def _write_json_trace(event: dict[str, Any]) -> None:
    path = _trace_jsonl_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")


def _configure_provider_env() -> str | None:
    settings = get_settings()
    if settings.langsmith_api_key:
        environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
        environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
        return "langsmith"

    if environ.get("LANGFUSE_PUBLIC_KEY") and environ.get("LANGFUSE_SECRET_KEY"):
        return "langfuse"

    return None


def _get_langfuse_client() -> Any | None:
    global _LANGFUSE_CLIENT
    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT
    if Langfuse is None:
        return None
    public_key = environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    host = environ.get("LANGFUSE_HOST")
    _LANGFUSE_CLIENT = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    return _LANGFUSE_CLIENT


def _get_langsmith_client() -> Any | None:
    global _LANGSMITH_CLIENT
    if _LANGSMITH_CLIENT is not None:
        return _LANGSMITH_CLIENT
    if LangSmithClient is None:
        return None
    api_key = environ.get("LANGSMITH_API_KEY")
    if not api_key:
        return None
    _LANGSMITH_CLIENT = LangSmithClient(api_key=api_key)
    return _LANGSMITH_CLIENT


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton."""

    started = perf_counter()
    provider = _configure_provider_env()
    payload = attributes or {}
    langfuse_span: Any | None = None
    langsmith_client: Any | None = None
    langsmith_run_id: UUID | None = None
    span_error: str | None = None
    if provider == "langfuse":
        client = _get_langfuse_client()
        if client is not None:
            langfuse_span = client.span(name=name, input=payload)
    elif provider == "langsmith":
        langsmith_client = _get_langsmith_client()
        if langsmith_client is not None and hasattr(langsmith_client, "create_run"):
            langsmith_run_id = uuid4()
            langsmith_client.create_run(
                id=langsmith_run_id,
                name=name,
                run_type="chain",
                project_name=get_settings().langsmith_project,
                inputs=payload,
                start_time=_utc_now(),
            )
    span: dict[str, Any] = {
        "name": name,
        "attributes": payload,
        "duration_seconds": None,
        "provider": provider,
    }
    try:
        yield span
    except Exception as exc:
        span_error = str(exc)
        if langfuse_span is not None:
            langfuse_span.update(
                level="ERROR",
                status_message=str(exc),
            )
        raise
    finally:
        duration = perf_counter() - started
        span["duration_seconds"] = duration
        _write_json_trace(span)
        if langsmith_client is not None and langsmith_run_id is not None and hasattr(langsmith_client, "update_run"):
            langsmith_client.update_run(
                run_id=langsmith_run_id,
                outputs={
                    "duration_seconds": duration,
                    "provider": provider,
                    "attributes": span.get("attributes", {}),
                },
                error=span_error,
                end_time=_utc_now(),
            )
        if langfuse_span is not None:
            langfuse_span.update(
                output={
                    "duration_seconds": duration,
                    "provider": provider,
                    "attributes": span.get("attributes", {}),
                }
            )
            client = _get_langfuse_client()
            if client is not None:
                client.flush()
