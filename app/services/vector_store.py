from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.database import AsyncSessionLocal
import logging

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        self.session = None

    async def _get_session(self):
        if not self.session:
            self.session = AsyncSessionLocal()
        return self.session

    # Columns allowed as equality filters, mapped to their SQL type cast.
    # Only keys in this allow-list are ever interpolated into the query
    # *as column names*; values are always passed as bound parameters.
    _FILTERABLE_COLUMNS = {
        "exam_id": "uuid",
        "subject_id": "uuid",
        "chapter_id": "uuid",
        "topic_id": "uuid",
    }

    async def search(
        self,
        query_embedding: List[float],
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search the vector store"""
        try:
            session = await self._get_session()
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

            # Build filter conditions using only allow-listed column names;
            # values are always bound parameters, never string-interpolated.
            params: Dict[str, Any] = {"embedding": embedding_str, "limit": limit}
            conditions = []
            if filter:
                for key, value in filter.items():
                    if value is None or key not in self._FILTERABLE_COLUMNS:
                        continue
                    cast = self._FILTERABLE_COLUMNS[key]
                    conditions.append(f"{key} = CAST(:{key} AS {cast})")
                    params[key] = str(value)
            filter_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

            query = text(f"""
                SELECT
                    id,
                    document_id,
                    chunk_index,
                    content,
                    extra_data,
                    1 - (embedding <=> CAST(:embedding AS vector)) as similarity
                FROM document_chunks
                {filter_clause}
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
            """)

            result = await session.execute(query, params)

            results = []
            for row in result:
                # NULL for any chunk whose embedding failed to generate (e.g.
                # transient provider error during ingestion) — skip it rather
                # than letting one bad row's `None >= float` comparison crash
                # the whole search and silently return zero results.
                if row.similarity is not None and row.similarity >= min_score:
                    results.append({
                        "id": str(row.id),
                        "document_id": str(row.document_id),
                        "chunk_index": row.chunk_index,
                        "content": row.content,
                        "metadata": row.extra_data,
                        "similarity": row.similarity
                    })

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []