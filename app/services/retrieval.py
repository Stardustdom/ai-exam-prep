from typing import List, Dict, Any
from app.services.vector_store import VectorStoreService
from app.services.embeddings import EmbeddingService
import logging

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(
        self,
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    async def retrieve_relevant_chunks(
        self,
        query: str,
        exam_id: str,
        chapter_id: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query"""
        try:
            # Get query embedding
            embedding = await self.embedding_service.embed_text(query)
            
            # Build filter
            filter_dict = {"exam_id": exam_id}
            if chapter_id:
                filter_dict["chapter_id"] = chapter_id
            
            # Search
            results = await self.vector_store.search(
                query_embedding=embedding,
                filter=filter_dict,
                limit=limit
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []