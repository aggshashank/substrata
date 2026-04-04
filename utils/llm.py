"""Substrata LLM wrapper — all LLM calls go through here via litellm."""

import time
from typing import Any

import litellm
import requests
from loguru import logger

from config import get_settings


def call_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Call the configured LLM via litellm with retry logic.

    Returns empty string on failure — never raises.
    """
    settings = get_settings()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(3):
        try:
            start = time.monotonic()
            response = litellm.completion(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_base=settings.OLLAMA_BASE_URL,
            )
            latency = time.monotonic() - start
            result: str = response.choices[0].message.content or ""
            logger.info(
                "LLM call: prompt_chars={} response_chars={} latency={:.2f}s",
                len(prompt),
                len(result),
                latency,
            )
            return result
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning("LLM attempt {}/{} failed: {}. Retrying in {}s", attempt + 1, 3, exc, wait)
            if attempt < 2:
                time.sleep(wait)

    logger.error("All LLM retries exhausted for prompt (first 80 chars): {}", prompt[:80])
    return ""


def check_health() -> dict[str, bool]:
    """Check Ollama availability and model presence."""
    settings = get_settings()
    result: dict[str, bool] = {
        "ollama_running": False,
        "llm_model_available": False,
        "embed_model_available": False,
    }
    try:
        resp = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return result
        result["ollama_running"] = True
        models: list[str] = [m["name"] for m in resp.json().get("models", [])]

        def _name(model_str: str) -> str:
            return model_str.removeprefix("ollama/")

        llm_name = _name(settings.LLM_MODEL)
        embed_name = _name(settings.EMBED_MODEL)
        result["llm_model_available"] = any(llm_name in m for m in models)
        result["embed_model_available"] = any(embed_name in m for m in models)
    except Exception as exc:
        logger.warning("Health check failed: {}", exc)
    return result


def parse_bullet_list(text: str) -> list[str]:
    """Extract bullet/numbered list items from LLM output."""
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        for prefix in ("- ", "* "):
            if stripped.startswith(prefix):
                items.append(stripped[len(prefix):].strip())
                break
        else:
            # Numbered list: "1. " "10. " etc.
            parts = stripped.split(". ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                items.append(parts[1].strip())
    return [item for item in items if item]
