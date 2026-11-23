# app/services/llm.py
#
# Provider abstraction (spec section 31): application code depends only on
# LLMService's generate_json/generate_text methods. Which vendor backs them
# is chosen entirely by the LLM_PROVIDER env var, with no other code changes.
from abc import ABC, abstractmethod
from typing import Dict, Any
import json
import re
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_json(self, prompt: str, system: str = "") -> Dict[str, Any]: ...

    @abstractmethod
    async def generate_text(self, prompt: str, temperature: float = 0.7) -> str: ...


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction for providers without a native JSON-mode guarantee"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    async def generate_json(self, prompt: str, system: str = "") -> Dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    async def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content or ""


class AnthropicProvider(BaseLLMProvider):
    def __init__(self):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model

    async def generate_json(self, prompt: str, system: str = "") -> Dict[str, Any]:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.3,
            system=system or "You are a helpful assistant. Respond with valid JSON only, no prose, no markdown fences.",
            messages=[{"role": "user", "content": prompt}]
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return _extract_json(text)

    async def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return "".join(block.text for block in response.content if block.type == "text")


class GeminiProvider(BaseLLMProvider):
    # Gemini regularly answers 503 ("high demand") / 429 (rate limit). Retry with
    # backoff on the primary model, then fall back to lighter models.
    FALLBACK_MODELS = ("gemini-2.5-flash-lite", "gemini-2.0-flash")
    RETRIES_PER_MODEL = 3

    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    @staticmethod
    def _is_transient(e: Exception) -> bool:
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        msg = str(e)
        return code in (429, 500, 502, 503, 504) or any(
            t in msg for t in ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "high demand")
        )

    async def _call(self, contents, build_config, parse):
        import asyncio
        models = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]
        last: Exception = RuntimeError("no models tried")
        for model in models:
            for attempt in range(self.RETRIES_PER_MODEL):
                try:
                    response = await self.client.aio.models.generate_content(
                        model=model, contents=contents, config=build_config()
                    )
                    return parse(response.text or "")
                except Exception as e:
                    last = e
                    if not self._is_transient(e) and not isinstance(e, ValueError):
                        raise
                    logger.warning(f"Gemini {model} attempt {attempt + 1} failed: {str(e)[:150]}")
                    await asyncio.sleep(1.5 * (2 ** attempt))
        raise last

    async def generate_json(self, prompt: str, system: str = "") -> Dict[str, Any]:
        from google.genai import types
        return await self._call(
            prompt,
            lambda: types.GenerateContentConfig(
                system_instruction=system or "You are a helpful assistant. Respond with valid JSON only, no prose, no markdown fences.",
                temperature=0.3,
                response_mime_type="application/json",
            ),
            _extract_json,
        )

    async def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        from google.genai import types
        return await self._call(
            prompt,
            lambda: types.GenerateContentConfig(temperature=temperature),
            lambda t: t,
        )


def _build_provider() -> BaseLLMProvider:
    if settings.llm_provider == "anthropic":
        return AnthropicProvider()
    if settings.llm_provider == "gemini":
        return GeminiProvider()
    return OpenAIProvider()


class LLMService:
    """Facade used by all agents; delegates to whichever provider is configured."""

    def __init__(self):
        self._provider = _build_provider()

    async def generate_json(self, prompt: str, system: str = "") -> Dict[str, Any]:
        try:
            return await self._provider.generate_json(prompt, system)
        except Exception as e:
            # Log AND re-raise (was previously swallowed into `{}`), which made every
            # downstream failure - including a missing/invalid API key - look like "the
            # LLM found nothing", e.g. sample-paper extraction reporting the misleading
            # "No questions could be extracted from this paper" instead of the real
            # cause. Callers that want a soft-fail already wrap this in their own
            # try/except (see app/agents/evaluation.py, app/agents/question_generator.py);
            # callers that don't (blueprint_pipeline, resource_pipeline) have an outer
            # except that now records the real error message against the record's
            # status_message.
            logger.error(f"LLM JSON generation failed: {e}")
            raise

    async def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            return await self._provider.generate_text(prompt, temperature)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
