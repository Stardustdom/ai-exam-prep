from typing import Optional, Dict, Any
import hashlib
from datetime import datetime, timedelta
from app.database.repositories import SemanticCacheRepository
from app.services.embeddings import EmbeddingService
import logging

logger = logging.getLogger(__name__)

class SemanticCacheService:
    """
    Caches (normalized query -> resolved entity) mappings so repeated free-text
    input (e.g. "jee", "joint entrance examination", "JEE exam") doesn't
    re-invoke embeddings/LLM matching every time. Entries are scoped by
    entity_type plus any additional_filter (e.g. exam_id when resolving a
    chapter) so the same text can resolve differently in different contexts.
    """

    def __init__(
        self,
        cache_repo: SemanticCacheRepository,
        embedding_service: EmbeddingService
    ):
        self.cache_repo = cache_repo
        self.embedding_service = embedding_service

    async def get_resolved_entity(
        self,
        query: str,
        entity_type: str,
        additional_filter: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Get resolved entity from cache"""
        try:
            query_hash = self._hash_query(query, entity_type, additional_filter)
            cache_entry = await self.cache_repo.get_by_query_hash(query_hash)

            if cache_entry:
                # Check if expired
                if cache_entry.expires_at and cache_entry.expires_at < datetime.utcnow():
                    return None

                return {
                    "entity_id": cache_entry.resolved_entity_id,
                    "entity_type": cache_entry.resolved_entity_type,
                    "confidence": cache_entry.confidence,
                    "metadata": cache_entry.extra_data
                }

        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")

        return None

    async def cache_resolved_entity(
        self,
        query: str,
        entity_type: str,
        entity_id: str,
        confidence: float,
        additional_filter: Optional[Dict] = None,
        ttl_hours: int = 24
    ):
        """Cache resolved entity"""
        try:
            query_hash = self._hash_query(query, entity_type, additional_filter)

            cache_data = {
                "query_hash": query_hash,
                "normalized_query": query.lower().strip(),
                "resolved_entity_type": entity_type,
                "resolved_entity_id": entity_id,
                "confidence": confidence,
                "extra_data": additional_filter,
                "expires_at": datetime.utcnow() + timedelta(hours=ttl_hours)
            }

            # Check if exists
            existing = await self.cache_repo.get_by_query_hash(query_hash)
            if existing:
                # Update existing
                for key, value in cache_data.items():
                    setattr(existing, key, value)
                await self.cache_repo.session.commit()
            else:
                # Create new
                await self.cache_repo.create(cache_data)

        except Exception as e:
            logger.error(f"Cache storage failed: {e}")

    async def invalidate_entity(self, entity_type: str, entity_id: str) -> None:
        """
        Remove cached mappings pointing at a specific entity. Call this when
        the admin deletes, deactivates, or materially edits an exam/chapter/topic
        so stale cache entries can't keep resolving user input to it.
        """
        try:
            await self.cache_repo.delete_by_entity(entity_type, entity_id)
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")

    def _hash_query(
        self,
        query: str,
        entity_type: str,
        additional_filter: Optional[Dict] = None
    ) -> str:
        """Hash a query for cache key, scoped by entity type and any filter (e.g. exam_id)"""
        normalized = query.lower().strip()
        scope = ""
        if additional_filter:
            scope = "|" + "|".join(f"{k}={v}" for k, v in sorted(additional_filter.items()))
        key = f"{entity_type}|{normalized}{scope}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()