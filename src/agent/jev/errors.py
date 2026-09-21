"""Error taxonomy for the Jev API.

The server distinguishes two failure shapes, and they mean different things:

    422 -> pydantic validation, with field-level ``loc`` paths. Our bug.
           Never retryable; the same request will always fail.
    400 -> {"detail": {"error_type": ..., "message": ...}}
           ``max_tokens_exceeded``  -> shed state and retry once
           ``api_usage_error``      -> our bug (bad model name, too many choices)

Collapsing these into one exception class loses the distinction that decides
whether to retry, so they stay separate.
"""

from __future__ import annotations

from typing import Any


class JevError(Exception):
    """Base for every Jev failure."""


class JevValidationError(JevError):
    """422. The request was malformed. Not retryable."""

    def __init__(self, detail: list[dict[str, Any]]) -> None:
        self.detail = detail
        fields = ", ".join(".".join(str(p) for p in d.get("loc", [])) for d in detail)
        super().__init__(f"Jev rejected the request; invalid fields: {fields}")


class JevUsageError(JevError):
    """400 api_usage_error. Bad model name, too many choices. Not retryable."""


class JevTooLargeError(JevError):
    """400 max_tokens_exceeded. Shed state and retry once."""


class JevTransportError(JevError):
    """Network failure or 5xx. Retryable with backoff."""


class JevAuthError(JevError):
    """401/403. Jev needs no auth today, so this means that changed.

    Deliberately its own class: it should page a human rather than be
    swallowed by a generic retry loop.
    """


class CassetteMiss(JevError):
    """Replay mode was asked for a request that was never recorded.

    Never falls back to the network -- that would make the offline suite
    silently non-deterministic and able to spend money.
    """

    def __init__(self, key: str, closest: str | None = None) -> None:
        self.key = key
        self.closest = closest
        message = f"no recorded response for request {key[:16]}..."
        if closest:
            message += f" (closest recorded: {closest[:16]}...)"
        message += "\nRe-record with: uv run pytest -m live"
        super().__init__(message)


def classify_http_error(status: int, body: Any) -> JevError:
    """Map an HTTP failure onto the taxonomy above."""
    if status in (401, 403):
        return JevAuthError(
            f"Jev returned {status}. It required no authentication when this "
            "was built, so either that changed or TYPESAFE_API_KEY is wrong."
        )

    if status == 422:
        detail = body.get("detail") if isinstance(body, dict) else None
        return JevValidationError(detail if isinstance(detail, list) else [])

    if status == 400:
        detail = body.get("detail") if isinstance(body, dict) else None
        # Two observed shapes: a dict with error_type, or a bare string
        # (as returned for the 255-choice limit).
        if isinstance(detail, dict):
            error_type = detail.get("error_type")
            message = detail.get("message", "")
            if error_type == "max_tokens_exceeded":
                return JevTooLargeError(
                    "request exceeded Jev's 32,768 input-token ceiling "
                    "(the cap covers question criteria, not just state)"
                )
            return JevUsageError(f"{error_type}: {message}")
        return JevUsageError(str(detail))

    if status >= 500:
        return JevTransportError(f"Jev returned {status}")

    return JevError(f"unexpected status {status}: {body!r}")
