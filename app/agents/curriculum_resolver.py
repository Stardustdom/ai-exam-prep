# app/agents/curriculum_resolver.py
from typing import Optional, Tuple, List, Dict, Any
import re
import math
import asyncio
from app.database.repositories import ChapterRepository, TopicRepository
from app.services.embeddings import EmbeddingService
from app.services.semantic_cache import SemanticCacheService
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)

_OVERALL_ALIASES = {"overall", "all", "everything", "any", "all topics", "full syllabus", "complete syllabus"}

# Process-level cache of chapter/topic NAME embeddings, keyed by exam_id.
# CurriculumResolverAgent is rebuilt fresh on every Telegram update (see
# app.bot.dependencies), so an instance attribute wouldn't survive between
# messages — this has to be a module-level dict to actually help. Without it,
# every single free-text resolution attempt (including a wrong guess like
# "hi") re-embeds the ENTIRE curriculum's chapter+topic names from scratch —
# 222 names for this exam, split into multiple sequential API calls, each one
# exposed to rate-limit retry/backoff — before it can even fail. Keyed and
# invalidated by the exact tuple of candidate names, so an admin adding/
# renaming chapters transparently invalidates it on the next lookup.
_candidate_embedding_cache: Dict[str, Tuple[Tuple[str, ...], List[List[float]]]] = {}


class CurriculumResolverAgent:
    """
    Agent 3: Curriculum Resolver
    Responsibilities: resolve a chapter/topic from user input (button click or
    free text) scoped to the currently selected exam, using exact match,
    normalized match, and semantic search over the AI-extracted curriculum
    (never a hardcoded list — see spec section 12).
    """

    def __init__(
        self,
        chapter_repo: ChapterRepository,
        topic_repo: TopicRepository,
        embedding_service: EmbeddingService,
        semantic_cache_service: SemanticCacheService
    ):
        self.chapter_repo = chapter_repo
        self.topic_repo = topic_repo
        self.embedding_service = embedding_service
        self.semantic_cache_service = semantic_cache_service

    async def get_options(self, exam_id: str) -> List[Dict[str, Any]]:
        """All chapters and topics available for an exam, for menu display and fallback listing"""
        chapters = await self.chapter_repo.get_by_exam(exam_id)
        topics = await self.topic_repo.get_by_exam(exam_id)
        options = [{"id": str(c.id), "name": c.name, "type": "chapter"} for c in chapters]
        options += [{"id": str(t.id), "name": t.name, "type": "topic"} for t in topics]
        return options

    async def resolve_chapter(
        self,
        exam_id: str,
        user_input: str
    ) -> Tuple[Optional[str], Optional[str], Optional[float], str]:
        """
        Resolve a chapter/topic from user input.
        Returns: (chapter_or_topic_id, name, confidence, method)
        `chapter_or_topic_id` is the literal string "overall" for the whole-syllabus case.
        """
        if not user_input or not user_input.strip():
            return None, None, 0.0, "empty"

        normalized_input = self._normalize(user_input)

        # "Overall" is a fixed sentinel, not something derived from the corpus
        if normalized_input in _OVERALL_ALIASES:
            return "overall", "Overall", 1.0, "overall"

        options = await self.get_options(exam_id)
        if not options:
            return None, None, 0.0, "no_curriculum"

        # Step 1: semantic cache (scoped to this exam so the same phrase can
        # resolve differently in different exams)
        cached = await self.semantic_cache_service.get_resolved_entity(
            query=normalized_input,
            entity_type="chapter",
            additional_filter={"exam_id": exam_id}
        )
        if cached:
            match = next((o for o in options if o["id"] == cached["entity_id"]), None)
            if match:
                return match["id"], match["name"], cached["confidence"], "cache"

        # Step 2: exact match (case-insensitive)
        for option in options:
            if self._normalize(option["name"]) == normalized_input:
                await self._cache(normalized_input, exam_id, option["id"], 1.0)
                return option["id"], option["name"], 1.0, "exact"

        # Step 3: normalized substring match (handles "laws of motion" vs "Laws Of Motion ")
        substring_matches = [
            o for o in options
            if normalized_input in self._normalize(o["name"]) or self._normalize(o["name"]) in normalized_input
        ]
        if len(substring_matches) == 1:
            match = substring_matches[0]
            await self._cache(normalized_input, exam_id, match["id"], 0.95)
            return match["id"], match["name"], 0.95, "normalized"

        # Step 4: semantic match via embeddings over candidate names. Bounded
        # with a timeout — this runs inline in a live chat turn, unlike
        # ingestion's embedding calls which are a background Celery task
        # that can reasonably wait out a full retry/backoff cycle (up to
        # ~135s per 100-item batch, and a cold cache here needs 3 of those
        # plus one more for the query). Blocking someone's Telegram message
        # for minutes is worse than just falling back to the exam_graph's
        # difflib-based "closest matches" suggestions a few seconds sooner.
        try:
            semantic_match = await asyncio.wait_for(
                self._semantic_match(exam_id, normalized_input, options), timeout=12.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Semantic chapter match timed out for exam {exam_id}")
            semantic_match = None
        if semantic_match:
            option, confidence = semantic_match
            if confidence >= settings.chapter_match_confidence_threshold:
                await self._cache(normalized_input, exam_id, option["id"], confidence)
                return option["id"], option["name"], confidence, "semantic"
            else:
                return None, None, confidence, "semantic_low"

        return None, None, 0.0, "none"

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    async def _semantic_match(
        self,
        exam_id: str,
        normalized_input: str,
        options: List[Dict[str, Any]]
    ) -> Optional[Tuple[Dict[str, Any], float]]:
        try:
            candidate_names = tuple(o["name"] for o in options)

            cached = _candidate_embedding_cache.get(exam_id)
            if cached and cached[0] == candidate_names:
                candidate_embeddings = cached[1]
            else:
                candidate_embeddings = await self.embedding_service.embed_batch(list(candidate_names))
                if candidate_embeddings and all(candidate_embeddings):
                    _candidate_embedding_cache[exam_id] = (candidate_names, candidate_embeddings)

            query_vec = await self.embedding_service.embed_text(normalized_input)
            if not query_vec:
                return None

            best_option, best_score = None, -1.0
            for option, vec in zip(options, candidate_embeddings):
                if not vec:
                    continue
                score = self._cosine_similarity(query_vec, vec)
                if score > best_score:
                    best_option, best_score = option, score

            if best_option:
                return best_option, best_score
        except Exception as e:
            logger.error(f"Semantic chapter match failed: {e}")
        return None

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def _cache(self, normalized_input: str, exam_id: str, entity_id: str, confidence: float) -> None:
        await self.semantic_cache_service.cache_resolved_entity(
            query=normalized_input,
            entity_type="chapter",
            entity_id=entity_id,
            confidence=confidence,
            additional_filter={"exam_id": exam_id}
        )
