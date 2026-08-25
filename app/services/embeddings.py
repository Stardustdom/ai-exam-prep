# app/services/embeddings.py
#
# Provider abstraction, mirroring app.services.llm: application code depends
# only on EmbeddingService's embed_text/embed_batch methods. Which vendor
# backs them is chosen by settings.embedding_provider, independent of
# llm_provider — not every LLM provider has a first-party embedding model
# (Anthropic doesn't; OpenAI and Gemini both do).
from abc import ABC, abstractmethod
from typing import List
from app.config.settings import settings
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential, before_sleep_log
import logging

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Both providers' SDKs raise a generic exception whose message carries
    the HTTP status rather than a dedicated exception type — string-match it."""
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate_limit" in msg.lower()


# Retries only actual rate-limit errors (with backoff, since the free tier's
# own error already tells you to "retry in 30s") and re-raises anything else
# or an exhausted retry immediately, so EmbeddingService's soft-fail-to-[]
# still applies to genuine failures — just not to "the batch would've
# succeeded 30 seconds later" ones, which is what silently produced 12,685
# NULL-embedding chunks the first time a large ingestion run hit this quota.
_retry_on_rate_limit = retry(
    retry=retry_if_exception(_is_rate_limit_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]: ...


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model

    @_retry_on_rate_limit
    async def embed_text(self, text: str) -> List[float]:
        response = await self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    @_retry_on_rate_limit
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(model=self.model, input=texts)
        return [data.embedding for data in response.data]


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    # Gemini's batchEmbedContents hard-rejects (400 INVALID_ARGUMENT) any
    # call with more than 100 texts — not a soft limit, so every caller must
    # respect it regardless of how big a list they happen to pass in.
    MAX_BATCH_SIZE = 100

    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_embedding_model
        # MRL truncation to match the existing pgvector column dimension
        # (EMBEDDING_DIMENSION in app.database.models) with no migration.
        self.output_dimensionality = settings.gemini_embedding_dimension

    async def embed_text(self, text: str) -> List[float]:
        result = await self._embed([text])
        return result[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if len(texts) <= self.MAX_BATCH_SIZE:
            return await self._embed(texts)
        results: List[List[float]] = []
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            results.extend(await self._embed(texts[i:i + self.MAX_BATCH_SIZE]))
        return results

    @_retry_on_rate_limit
    async def _embed(self, texts: List[str]) -> List[List[float]]:
        from google.genai import types
        response = await self.client.aio.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self.output_dimensionality),
        )
        return [e.values for e in response.embeddings]


def _build_provider() -> BaseEmbeddingProvider:
    if settings.embedding_provider == "gemini":
        return GeminiEmbeddingProvider()
    return OpenAIEmbeddingProvider()


class EmbeddingService:
    """Facade used by all agents/pipelines; delegates to whichever provider is configured.

    Soft-fails on error (returns [] / a list of []'s) rather than raising, because
    embeddings are an optional enrichment in most call sites (semantic search,
    exam/chapter matching) that already check truthiness before using the
    result and degrade gracefully without it — unlike LLMService.generate_json,
    where a missing result silently masquerades as "the LLM found nothing".
    """

    def __init__(self):
        self._provider = _build_provider()

    async def embed_text(self, text: str) -> List[float]:
        try:
            return await self._provider.embed_text(text)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            return await self._provider.embed_batch(texts)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [[] for _ in texts]
