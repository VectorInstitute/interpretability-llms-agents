"""Lightweight wrappers around Langfuse v4 observations for the MEP pipeline.

All helpers accept ``None`` as the client/trace and become no-ops, so the
rest of the codebase can call them unconditionally.
"""

import contextlib
from contextlib import contextmanager
from typing import Optional


def _normalize_usage(usage: dict) -> dict:
    """Map provider usage dicts (OpenAI/Gemini) to Langfuse v4 usage_details keys."""
    normalized: dict = {}
    # OpenAI keys
    if "prompt_tokens" in usage:
        normalized["input"] = usage["prompt_tokens"]
    elif "input" in usage:
        normalized["input"] = usage["input"]
    if "completion_tokens" in usage:
        normalized["output"] = usage["completion_tokens"]
    elif "output" in usage:
        normalized["output"] = usage["output"]
    if "total_tokens" in usage:
        normalized["total"] = usage["total_tokens"]
    elif "total" in usage:
        normalized["total"] = usage["total"]
    return normalized or usage


class _TraceHandle:
    """Thin wrapper yielded by sample_trace; exposes a stable interface across callers.

    Attributes
    ----------
    id : str | None
        The Langfuse trace ID, usable for attaching scores after the run.
    """

    def __init__(self, span: object, trace_id: Optional[str]) -> None:
        self._span = span
        self.id = trace_id

    def update(self, **kwargs: object) -> None:
        """Update the root trace span (e.g. set output after the run)."""
        if self._span is not None:
            with contextlib.suppress(Exception):
                self._span.update(**kwargs)  # type: ignore[union-attr]

    def score_trace(self, name: str, value: float) -> None:
        """Attach a numeric score to the root trace."""
        if self._span is not None:
            with contextlib.suppress(Exception):
                self._span.score_trace(name=name, value=value)  # type: ignore[union-attr]


@contextmanager
def sample_trace(
    client: object,
    sample_id: str,
    question: str,
    expected_output: str,
    question_type: str,
    config_name: str,
    run_id: str,
    project_name: str = "chartqapro-eval",
):  # type: ignore[return]
    """Open a Langfuse trace for one sample; yield a _TraceHandle (or None)."""
    del project_name  # kept for API compatibility; Langfuse v4 uses project from SDK config
    if client is None:
        yield None
        return

    from langfuse import propagate_attributes

    with client.start_as_current_observation(  # type: ignore[union-attr]
        name=f"chartqapro/{sample_id}",
        as_type="span",
        input={"question": question, "expected_output": expected_output},
        metadata={
            "run_id": run_id,
            "config": config_name,
            "question_type": question_type,
        },
    ) as span:
        with propagate_attributes(session_id=run_id):
            trace_id = client.get_current_trace_id()  # type: ignore[union-attr]
            handle = _TraceHandle(span=span, trace_id=trace_id)
            try:
                yield handle
            finally:
                with contextlib.suppress(Exception):
                    client.flush()  # type: ignore[union-attr]


def open_llm_span(
    trace: object,
    name: str,
    input_data: dict,
    model: str,
    metadata: Optional[dict] = None,
    parent_span_id: Optional[str] = None,
) -> object:
    """Create a Langfuse generation on the trace span (or return None).

    ``parent_span_id`` is accepted for API compatibility but is unused in v4 —
    nesting is handled by calling ``start_observation`` on the parent span.
    """
    del parent_span_id  # kept for API compatibility; v4 uses contextual nesting
    if trace is None:
        return None
    span = getattr(trace, "_span", None)
    if span is None:
        return None
    with contextlib.suppress(Exception):
        return span.start_observation(  # type: ignore[union-attr]
            name=name,
            as_type="generation",
            input=input_data,
            model=model,
            metadata=metadata or {},
        )
    return None


def close_span(
    span: object,
    output: Optional[dict] = None,
    usage: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    """End a Langfuse generation (no-op if span is None)."""
    if span is None:
        return
    with contextlib.suppress(Exception):
        update_kwargs: dict = {}
        if output is not None:
            update_kwargs["output"] = output
        if usage:
            update_kwargs["usage_details"] = _normalize_usage(usage)
        if error:
            update_kwargs["level"] = "ERROR"
            update_kwargs["status_message"] = error
        if update_kwargs:
            span.update(**update_kwargs)  # type: ignore[union-attr]
        span.end()  # type: ignore[union-attr]


def log_trace_scores(trace: object, scores: dict) -> None:
    """Log a dict of {metric_name: float} as scores on the trace."""
    if trace is None:
        return
    for name, value in scores.items():
        if isinstance(value, (int, float)):
            with contextlib.suppress(Exception):
                if hasattr(trace, "score_trace"):
                    trace.score_trace(name=name, value=float(value))  # type: ignore[union-attr]
