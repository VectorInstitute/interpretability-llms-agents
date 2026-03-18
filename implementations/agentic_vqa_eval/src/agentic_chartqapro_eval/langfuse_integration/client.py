"""Langfuse client singleton with graceful degradation.

Returns None when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set or
langfuse is not installed, so every caller can guard with ``if client:``.
"""

import os
from contextlib import suppress

from dotenv import load_dotenv
from langfuse import Langfuse


_client = None
_initialised = False


def get_client():
    """
    Initialize and return a globally cached Langfuse client.

    Retrieves configuration from environment variables and configures
    the SDK for local or cloud usage.

    Returns
    -------
    Langfuse or None
        An active client, or None if configuration is missing or invalid.
    """
    global _client, _initialised  # noqa: PLW0603
    if _initialised:
        return _client

    _initialised = True

    # Load environment variables from .env file
    with suppress(Exception):
        load_dotenv()

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        return None

    try:
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
    """
    Clear the cached client and reset initialization state.

    Returns
    -------
    None
    """
    global _client, _initialised  # noqa: PLW0603
    _client = None
    _initialised = False
