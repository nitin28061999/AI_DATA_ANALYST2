import os

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_GEMINI_LIVE_TESTS") != "1",
    reason=(
        "Live Gemini tests are disabled by default. "
        "Set RUN_GEMINI_LIVE_TESTS=1 to enable them."
    ),
)
def test_gemini_connection():
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        pytest.skip("GEMINI_API_KEY is not configured.")

    client = genai.Client(
        api_key=api_key
    )

    response = client.models.generate_content(
        model=os.getenv(
            "GEMINI_MODEL",
            "gemini-3.6-flash",
        ),
        contents="Say hello in one sentence.",
    )

    assert response.text