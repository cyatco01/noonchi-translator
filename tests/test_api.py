"""
API integration tests for the two-step translation flow.

Uses httpx.AsyncClient + LifespanManager so no live server is needed.
The ClaudeTranslationAgent constructor does not make API calls, so the app
starts normally with a fake key. Only translate() is mocked to avoid hitting
the real Claude API during tests.

Run: pytest tests/test_api.py -v
"""

import os
# Must be set before any backend imports so config.py reads a non-empty key
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "api"))

import pytest
import httpx
from unittest.mock import AsyncMock, patch
from asgi_lifespan import LifespanManager

import app as app_module
from models.schemas import TranslationResponse, RelationshipType, FormalityToken


@pytest.fixture(scope="module")
async def client():
    """Start the app once per test module with proper lifespan handling."""
    async with LifespanManager(app_module.app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app),
            base_url="http://test",
        ) as c:
            yield c


def _fake_translation(
    text: str = "Do you want to eat?",
    formality: FormalityToken = FormalityToken.FORMAL,
    translated: str = "드시겠습니까?",
) -> TranslationResponse:
    return TranslationResponse(
        original_text=text,
        conditioned_input=f"{formality.as_token()} {text}",
        translated_text=translated,
        formality_token=formality,
        relationship=RelationshipType.BOSS,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Set-context — structured path (no API call, pure FormalityResolver)
# ---------------------------------------------------------------------------

async def test_set_context_boss_workplace_resolves_formal(client):
    response = await client.post("/api/set-context", json={
        "relationship": "boss",
        "age_differential": -10,
        "setting": "workplace",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["formality_token"] == "formal"
    assert data["conditioning_input_prefix"] == "<formal>"
    assert "session_id" in data


async def test_set_context_colleague_resolves_polite(client):
    response = await client.post("/api/set-context", json={
        "relationship": "colleague",
        "age_differential": 0,
        "setting": "workplace",
    })
    assert response.status_code == 200
    assert response.json()["formality_token"] == "polite"


async def test_set_context_friend_intimate_resolves_casual(client):
    response = await client.post("/api/set-context", json={
        "relationship": "friend",
        "age_differential": 0,
        "setting": "intimate",
    })
    assert response.status_code == 200
    assert response.json()["formality_token"] == "casual"


async def test_formality_override_takes_precedence(client):
    """formality_override should override the inferred token."""
    response = await client.post("/api/set-context", json={
        "relationship": "boss",
        "age_differential": -10,
        "setting": "workplace",
        "formality_override": "casual",
    })
    assert response.status_code == 200
    assert response.json()["formality_token"] == "casual"


async def test_set_context_missing_fields_returns_422(client):
    """Providing neither situation nor structured fields is rejected."""
    response = await client.post("/api/set-context", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Translate
# ---------------------------------------------------------------------------

async def test_invalid_session_returns_404(client):
    response = await client.post("/api/translate", json={
        "session_id": "00000000-0000-0000-0000-000000000000",
        "text": "Hello",
    })
    assert response.status_code == 404


async def test_two_step_flow(client):
    """Full flow: set context → translate. translate() is mocked."""
    ctx = await client.post("/api/set-context", json={
        "relationship": "boss",
        "age_differential": -10,
        "setting": "workplace",
    })
    assert ctx.status_code == 200
    session_id = ctx.json()["session_id"]

    mock_result = _fake_translation("Do you want to eat?", FormalityToken.FORMAL, "드시겠습니까?")
    with patch.object(
        app_module.translation_agent, "translate", new=AsyncMock(return_value=mock_result)
    ):
        tr = await client.post("/api/translate", json={
            "session_id": session_id,
            "text": "Do you want to eat?",
        })

    assert tr.status_code == 200
    data = tr.json()
    assert data["translated_text"] == "드시겠습니까?"
    assert data["formality_token"] == "formal"
    assert data["conditioned_input"] == "<formal> Do you want to eat?"


async def test_multiple_translations_reuse_session(client):
    """Session stays valid across multiple translate calls."""
    ctx = await client.post("/api/set-context", json={
        "relationship": "friend",
        "age_differential": 0,
        "setting": "intimate",
    })
    session_id = ctx.json()["session_id"]

    phrases = ["See you later.", "Are you hungry?", "Let's go."]
    for phrase in phrases:
        mock_result = _fake_translation(phrase, FormalityToken.CASUAL, "번역")
        with patch.object(
            app_module.translation_agent, "translate", new=AsyncMock(return_value=mock_result)
        ):
            tr = await client.post("/api/translate", json={"session_id": session_id, "text": phrase})
        assert tr.status_code == 200
