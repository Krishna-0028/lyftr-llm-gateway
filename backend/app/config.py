from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KeyEntry:
    key_id: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Settings:
    keys: tuple[KeyEntry, ...]
    database_url: str
    allowed_origins: tuple[str, ...]
    enable_failure_simulation: bool
    cache_type: str
    disk_cache_dir: str
    default_system_prompt: str
    request_timeout_seconds: float


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_settings() -> Settings:
    raw_keys = _parse_csv(os.getenv("LLM_API_KEYS"))
    default_model = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash").strip() or "gemini/gemini-2.0-flash"
    raw_models = _parse_csv(os.getenv("LLM_MODELS"))

    entries: list[KeyEntry] = []
    for idx, api_key in enumerate(raw_keys, start=1):
        if raw_models and idx - 1 < len(raw_models):
            model = raw_models[idx - 1]
        elif len(raw_models) == 1:
            model = raw_models[0]
        else:
            model = default_model
        entries.append(KeyEntry(key_id=f"key_{idx}", api_key=api_key, model=model))

    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/app.db").strip() or "sqlite:///./data/app.db"
    origins = _parse_csv(os.getenv("ALLOWED_ORIGINS")) or [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    cache_type = os.getenv("CACHE_TYPE", "local").strip().lower() or "local"
    if cache_type not in {"local", "disk"}:
        cache_type = "local"

    disk_cache_dir = os.getenv("DISK_CACHE_DIR", "/data/litellm_cache").strip() or "/data/litellm_cache"
    default_system_prompt = (
        os.getenv("DEFAULT_SYSTEM_PROMPT", "You are a helpful, concise AI assistant.").strip()
        or "You are a helpful, concise AI assistant."
    )
    timeout_raw = os.getenv("LLM_TIMEOUT_SECONDS", "30").strip()
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 30.0

    return Settings(
        keys=tuple(entries),
        database_url=database_url,
        allowed_origins=tuple(origins),
        enable_failure_simulation=_parse_bool(os.getenv("ENABLE_FAILURE_SIMULATION"), default=False),
        cache_type=cache_type,
        disk_cache_dir=disk_cache_dir,
        default_system_prompt=default_system_prompt,
        request_timeout_seconds=timeout,
    )
