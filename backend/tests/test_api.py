from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import litellm
import pytest
from fastapi.testclient import TestClient

from app.config import KeyEntry, Settings
from app.db import Database
from app.llm import LLMGateway
from app.main import create_app


def _make_response(
    text: str,
    prompt_tokens: int = 25,
    completion_tokens: int = 12,
    cache_hit: bool = False,
):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        _hidden_params={"cache_hit": cache_hit},
    )


def _build_test_client(
    tmp_path,
    enable_failure_simulation: bool = True,
    keys: tuple[KeyEntry, ...] | None = None,
) -> tuple[TestClient, Database]:
    db_file = tmp_path / "test_app.db"
    settings = Settings(
        keys=keys
        or (
            KeyEntry(key_id="key_1", api_key="secret-test-key-1", model="gemini/gemini-2.0-flash"),
            KeyEntry(key_id="key_2", api_key="secret-test-key-2", model="gemini/gemini-2.0-flash"),
        ),
        database_url=f"sqlite:///{db_file}",
        allowed_origins=("http://localhost:3000",),
        enable_failure_simulation=enable_failure_simulation,
        cache_type="local",
        disk_cache_dir=str(tmp_path / "cache"),
        default_system_prompt="You are a test assistant.",
        request_timeout_seconds=10.0,
    )
    db = Database(settings.database_url)
    db.init_schema()
    gateway = LLMGateway(settings)
    app = create_app(settings=settings, db=db, gateway=gateway)
    return TestClient(app), db


def test_health_endpoint(tmp_path):
    client, _ = _build_test_client(tmp_path)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {
        "status": "ok",
        "keys_configured": 2,
        "cache_enabled": True,
    }


def test_chat_and_cache_hit_across_sessions(tmp_path):
    client, _ = _build_test_client(tmp_path)

    memory_cache: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[str, int, int]] = {}

    def fake_completion(**kwargs):
        cache_opt = kwargs.get("cache") or {}
        no_cache = bool(cache_opt.get("no-cache"))
        key = (
            kwargs["model"],
            tuple((m["role"], m["content"]) for m in kwargs["messages"]),
        )
        if not no_cache and key in memory_cache:
            reply, pt, ct = memory_cache[key]
            return _make_response(reply, pt, ct, cache_hit=True)

        time.sleep(0.015)
        reply = f"Answer for: {kwargs['messages'][-1]['content']}"
        pt = sum(len(m["content"]) for m in kwargs["messages"])
        ct = 9
        if not cache_opt.get("no-store"):
            memory_cache[key] = (reply, pt, ct)
        return _make_response(reply, pt, ct, cache_hit=False)

    with patch("app.llm.litellm.completion", side_effect=fake_completion):
        r1 = client.post(
            "/chat",
            json={"session_id": "s-1", "message": "What is the capital of France?"},
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["session_id"] == "s-1"
        assert d1["reply"] == "Answer for: What is the capital of France?"
        assert d1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert d1["metadata"]["key_id"] == "key_1"
        assert d1["metadata"]["fallback_attempts"] == 0
        assert d1["metadata"]["input_tokens"] > 0
        assert d1["metadata"]["output_tokens"] == 9
        assert isinstance(d1["metadata"]["latency_ms"], int)

        # Identical first message in a new session must hit cache
        r2 = client.post(
            "/chat",
            json={"session_id": "s-2", "message": "What is the capital of France?"},
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["session_id"] == "s-2"
        assert d2["reply"] == d1["reply"]
        assert d2["metadata"]["resolution_source"] == "LLM_CACHE"
        assert d2["metadata"]["key_id"] is None
        assert d2["metadata"]["fallback_attempts"] == 0
        assert d2["metadata"]["input_tokens"] == d1["metadata"]["input_tokens"]
        assert d2["metadata"]["output_tokens"] == d1["metadata"]["output_tokens"]
        assert d2["metadata"]["latency_ms"] < d1["metadata"]["latency_ms"]

        # Different message must never come from cache
        r3 = client.post(
            "/chat",
            json={"session_id": "s-3", "message": "What is the capital of Germany?"},
        )
        assert r3.status_code == 200
        assert r3.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_multiturn_history_and_system_prompt(tmp_path):
    client, _ = _build_test_client(tmp_path)
    seen_messages: list[list[dict[str, str]]] = []

    def fake_completion(**kwargs):
        seen_messages.append(list(kwargs["messages"]))
        return _make_response(f"Reply #{len(seen_messages)}", 20, 5)

    with patch("app.llm.litellm.completion", side_effect=fake_completion):
        r1 = client.post(
            "/chat",
            json={
                "session_id": "sess-alpha",
                "message": "My name is Ada.",
                "system_prompt": "Speak like a pirate.",
            },
        )
        assert r1.status_code == 200

        r2 = client.post(
            "/chat",
            json={
                "session_id": "sess-alpha",
                "message": "What is my name?",
            },
        )
        assert r2.status_code == 200

    # Turn 1 had system prompt + user1
    assert seen_messages[0] == [
        {"role": "system", "content": "Speak like a pirate."},
        {"role": "user", "content": "My name is Ada."},
    ]
    # Turn 2 kept session's system prompt + history + user2
    assert seen_messages[1] == [
        {"role": "system", "content": "Speak like a pirate."},
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Reply #1"},
        {"role": "user", "content": "What is my name?"},
    ]

    hist = client.get("/sessions/sess-alpha/messages")
    assert hist.status_code == 200
    body = hist.json()
    assert body["session_id"] == "sess-alpha"
    assert len(body["messages"]) == 4
    # No system prompt in stored messages response
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user", "assistant"]
    assert "metadata" not in body["messages"][0]
    assert body["messages"][1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert body["messages"][1]["metadata"]["input_tokens"] == 20
    assert body["messages"][1]["metadata"]["output_tokens"] == 5

    # Unknown session returns 404
    missing = client.get("/sessions/does-not-exist/messages")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "session not found"}


def test_automatic_failover_on_429_and_5xx(tmp_path):
    client, _ = _build_test_client(tmp_path)
    calls: list[str] = []

    def fake_completion(**kwargs):
        calls.append(kwargs["api_key"])
        if kwargs["api_key"] == "secret-test-key-1":
            raise litellm.RateLimitError(
                message="Rate limit exceeded",
                llm_provider="gemini",
                model=kwargs["model"],
            )
        return _make_response("Recovered with key 2", 30, 7)

    with patch("app.llm.litellm.completion", side_effect=fake_completion):
        res = client.post("/chat", json={"session_id": "s-failover", "message": "Hello"})

    assert res.status_code == 200
    data = res.json()
    assert data["reply"] == "Recovered with key 2"
    assert data["metadata"]["key_id"] == "key_2"
    assert data["metadata"]["fallback_attempts"] == 1
    assert data["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert calls == ["secret-test-key-1", "secret-test-key-2"]


def test_simulate_failure_header_behavior(tmp_path):
    client, _ = _build_test_client(tmp_path, enable_failure_simulation=True)

    def fake_completion(**kwargs):
        # Ensure secret-test-key-1 is never even called when key_1=429 or key_1=500
        assert kwargs["api_key"] == "secret-test-key-2"
        return _make_response("Key 2 answered", 18, 4)

    with patch("app.llm.litellm.completion", side_effect=fake_completion):
        r429 = client.post(
            "/chat",
            json={"session_id": "s-sim-1", "message": "Test 429"},
            headers={"X-Simulate-Failure": "key_1=429"},
        )
        assert r429.status_code == 200
        assert r429.json()["metadata"]["key_id"] == "key_2"
        assert r429.json()["metadata"]["fallback_attempts"] == 1

        r500 = client.post(
            "/chat",
            json={"session_id": "s-sim-2", "message": "Test 500"},
            headers={"X-Simulate-Failure": "key_1=500"},
        )
        assert r500.status_code == 200
        assert r500.json()["metadata"]["key_id"] == "key_2"
        assert r500.json()["metadata"]["fallback_attempts"] == 1

    # All keys fail -> 503 and nothing added to history
    r_all = client.post(
        "/chat",
        json={"session_id": "s-sim-all", "message": "Should fail"},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert r_all.status_code == 503
    assert r_all.json() == {"detail": "all providers unavailable"}
    assert client.get("/sessions/s-sim-all/messages").status_code == 404


def test_simulate_failure_ignored_when_disabled(tmp_path):
    client, _ = _build_test_client(tmp_path, enable_failure_simulation=False)

    with patch("app.llm.litellm.completion", return_value=_make_response("Key 1 answered")):
        res = client.post(
            "/chat",
            json={"session_id": "s-disabled", "message": "Hello"},
            headers={"X-Simulate-Failure": "*=429"},
        )
    assert res.status_code == 200
    assert res.json()["metadata"]["key_id"] == "key_1"
    assert res.json()["metadata"]["fallback_attempts"] == 0


def test_validation_422_and_no_history_polluted(tmp_path):
    client, _ = _build_test_client(tmp_path)

    # Empty message after trimming
    r1 = client.post("/chat", json={"session_id": "s-val", "message": "   \n  "})
    assert r1.status_code == 422
    assert "detail" in r1.json()

    # Over 4000 chars
    r2 = client.post("/chat", json={"session_id": "s-val", "message": "x" * 4001})
    assert r2.status_code == 422
    assert "detail" in r2.json()

    # Missing session_id
    r3 = client.post("/chat", json={"message": "Hello"})
    assert r3.status_code == 422
    assert "detail" in r3.json()

    # Malformed JSON
    r4 = client.post(
        "/chat",
        content="{bad json",
        headers={"Content-Type": "application/json"},
    )
    assert r4.status_code == 422
    assert "detail" in r4.json()

    assert client.get("/sessions/s-val/messages").status_code == 404


def test_real_litellm_local_cache_integration(tmp_path):
    """Verify real litellm.Cache(type='local') hits without missing optional deps."""
    client, _ = _build_test_client(
        tmp_path,
        keys=(
            KeyEntry(key_id="key_1", api_key="k1", model="gpt-4o-mini"),
            KeyEntry(key_id="key_2", api_key="k2", model="gpt-4o-mini"),
        ),
    )

    real_completion = litellm.completion

    def mock_provider_completion(**kwargs):
        kwargs["mock_response"] = "Paris"
        return real_completion(**kwargs)

    with patch("app.llm.litellm.completion", side_effect=mock_provider_completion):
        r1 = client.post("/chat", json={"session_id": "real-c1", "message": "Capital of France?"})
        assert r1.status_code == 200
        assert r1.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert r1.json()["metadata"]["key_id"] == "key_1"

        r2 = client.post("/chat", json={"session_id": "real-c2", "message": "Capital of France?"})
        assert r2.status_code == 200
        assert r2.json()["reply"] == "Paris"
        assert r2.json()["metadata"]["resolution_source"] == "LLM_CACHE"
        assert r2.json()["metadata"]["key_id"] is None
        assert r2.json()["metadata"]["fallback_attempts"] == 0
        assert r2.json()["metadata"]["input_tokens"] == r1.json()["metadata"]["input_tokens"]
        assert r2.json()["metadata"]["output_tokens"] == r1.json()["metadata"]["output_tokens"]

