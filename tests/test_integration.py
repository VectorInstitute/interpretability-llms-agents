"""Integration tests for Langfuse and Gemini API key validation."""

import os

import httpx
import pytest
from dotenv import load_dotenv
from langfuse import Langfuse


load_dotenv(verbose=True)


@pytest.mark.integration_test
def test_langfuse_auth() -> None:
    """Test that Langfuse API keys are valid and authentication succeeds."""
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    langfuse_client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )

    assert langfuse_client.auth_check(), "Langfuse authentication failed. Check your API keys."


@pytest.mark.integration_test
def test_gemini_auth() -> None:
    """Test that GEMINI_API_KEY is valid by making a minimal API call."""
    api_key = os.environ["GEMINI_API_KEY"]

    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
        json={"contents": [{"parts": [{"text": "Say hello."}]}]},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    assert data.get("candidates"), "Gemini API returned no candidates. Check your API key."
