from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from litellm.caching.caching import Cache

from app.config import KeyEntry, Settings

logger = logging.getLogger(__name__)

VALID_SIMULATED_STATUSES = {429, 500, 502, 503}


class AllProvidersUnavailableError(Exception):
    """Raised when every configured API key fails."""


class SimulatedProviderError(Exception):
    """Raised when X-Simulate-Failure triggers a synthetic provider status."""

    def __init__(self, key_id: str, status_code: int) -> None:
        self.key_id = key_id
        self.status_code = status_code
        super().__init__(f"Simulated HTTP {status_code} for {key_id}")


@dataclass(frozen=True)
class LLMResult:
    reply: str
    resolution_source: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


def parse_simulate_failure_header(header_value: Optional[str]) -> dict[str, int]:
    """Parse X-Simulate-Failure header into {key_id_or_star: status_code}."""
    if not header_value:
        return {}

    parsed: dict[str, int] = {}
    for pair in header_value.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        target, status_str = pair.split("=", 1)
        target = target.strip()
        status_str = status_str.strip()
        if not target:
            continue
        try:
            status_code = int(status_str)
        except ValueError:
            continue
        if status_code in VALID_SIMULATED_STATUSES:
            parsed[target] = status_code
    return parsed


def _extract_reply_and_usage(response: Any) -> tuple[str, int, int]:
    choices = getattr(response, "choices", None)
    if not choices and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        raise ValueError("Empty choices in LLM response")

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty assistant reply in LLM response")

    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    if isinstance(usage, dict):
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
    elif usage is not None:
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    else:
        input_tokens = 0
        output_tokens = 0

    return content, input_tokens, output_tokens


def _is_cache_hit(response: Any) -> bool:
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        return bool(hidden.get("cache_hit"))
    if isinstance(response, dict):
        hidden = response.get("_hidden_params")
        if isinstance(hidden, dict):
            return bool(hidden.get("cache_hit"))
    return False


class LLMGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache_enabled = False
        self._configure_litellm()

    def _configure_litellm(self) -> None:
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        litellm.drop_params = True

        if self.settings.cache_type == "disk":
            os.makedirs(self.settings.disk_cache_dir, exist_ok=True)
            litellm.cache = Cache(type="disk", disk_cache_dir=self.settings.disk_cache_dir)
        else:
            litellm.cache = Cache(type="local")
        self.cache_enabled = True

    def generate_reply(
        self,
        messages: list[dict[str, str]],
        simulate_failure_header: Optional[str] = None,
    ) -> LLMResult:
        if not self.settings.keys:
            raise AllProvidersUnavailableError("No API keys configured")

        simulated_failures: dict[str, int] = {}
        if self.settings.enable_failure_simulation:
            simulated_failures = parse_simulate_failure_header(simulate_failure_header)

        bypass_cache = bool(simulated_failures)
        start_time = time.perf_counter()
        fallback_attempts = 0

        for entry in self.settings.keys:
            simulated_status = simulated_failures.get("*") or simulated_failures.get(entry.key_id)
            if simulated_status is not None:
                logger.warning(
                    "Simulated provider error status=%s on key_id=%s",
                    simulated_status,
                    entry.key_id,
                )
                fallback_attempts += 1
                continue

            try:
                call_kwargs: dict[str, Any] = {
                    "model": entry.model,
                    "messages": messages,
                    "api_key": entry.api_key,
                    "timeout": self.settings.request_timeout_seconds,
                    "num_retries": 0,
                }
                if bypass_cache:
                    call_kwargs["cache"] = {"no-cache": True, "no-store": True}

                response = litellm.completion(**call_kwargs)
                reply_text, input_tokens, output_tokens = _extract_reply_and_usage(response)
                latency_ms = max(0, int(round((time.perf_counter() - start_time) * 1000)))

                cache_hit = (not bypass_cache) and _is_cache_hit(response)
                if cache_hit:
                    return LLMResult(
                        reply=reply_text,
                        resolution_source="LLM_CACHE",
                        model=entry.model,
                        latency_ms=latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        key_id=None,
                        fallback_attempts=0,
                    )

                return LLMResult(
                    reply=reply_text,
                    resolution_source="LIVE_INFERENCE",
                    model=entry.model,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=entry.key_id,
                    fallback_attempts=fallback_attempts,
                )
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                logger.warning(
                    "Provider call failed on key_id=%s model=%s status=%s error=%s",
                    entry.key_id,
                    entry.model,
                    status_code,
                    type(exc).__name__,
                )
                fallback_attempts += 1
                continue

        raise AllProvidersUnavailableError("All configured API keys failed")
