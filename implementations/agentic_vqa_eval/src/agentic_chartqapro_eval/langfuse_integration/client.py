"""Langfuse client singleton with graceful degradation.

Returns None when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set or
langfuse is not installed, so every caller can guard with ``if client:``.
"""

import os


_client = None
_initialised = False


def get_client():
    """Return a configured langfuse.Langfuse() instance, or None if unavailable."""
    global _client, _initialised  # noqa: PLW0603
    if _initialised:
        return _client

    _initialised = True

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        return None

    try:
        from langfuse import Langfuse

        kwargs: dict = {"public_key": public_key, "secret_key": secret_key}
        # Accept LANGFUSE_HOST or LANGFUSE_BASE_URL (both are common)
        host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL", "")
        if host:
            kwargs["host"] = host

        _client = Langfuse(**kwargs)
    except Exception as exc:
        print(f"[langfuse] client init failed: {exc}")
        _client = None

    return _client


def reset_client() -> None:
    """Force re-initialisation on next call (useful for tests)."""
    global _client, _initialised  # noqa: PLW0603
    _client = None
    _initialised = False
