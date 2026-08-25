# app/agents/retrieval.py
from typing import List, Dict, Any, Optional
from app.database.repositories import DocumentChunkRepository
from app.services.vector_store import VectorStoreService
from app.services.embeddings import EmbeddingService
import logging

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Agent 4: Retrieval Agent
    Responsibilities: Query vector database, retrieve relevant source chunks, filter by metadata
    """
    
    def __init__(
        self,
        chunk_repo: DocumentChunkRepository,
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService
    ):
        self.chunk_repo = chunk_repo
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
    async def retrieve_chunks(
        self,
        exam_id: str,
        query: str,
        chapter_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        limit: int = 20,
        min_relevance_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks based on query and metadata filters
        """
        try:
            # Build metadata filter
            metadata_filter = {"exam_id": exam_id}
            if chapter_id:
                metadata_filter["chapter_id"] = chapter_id
            if topic_id:
                metadata_filter["topic_id"] = topic_id
            
            # Get query embedding
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Search vector store
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                filter=metadata_filter,
                limit=limit,
                min_score=min_relevance_score
            )
            
            # Enrich with full chunk data
            enriched_results = []
            for result in results:
                chunk_id = result.get("id")
                if chunk_id:
                    chunk = await self.chunk_repo.get_by_id(chunk_id)
                    if chunk:
                        enriched_results.append({
                            **result,
                            "content": chunk.content,
                            "document_id": str(chunk.document_id),
                            "metadata": chunk.extra_data
                        })
            
            logger.info(f"Retrieved {len(enriched_results)} chunks for query: {query[:50]}...")
            return enriched_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    async def retrieve_for_question_generation(
        self,
        exam_id: str,
        chapter_id: str,
        query: str,
        num_chunks: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Specialized retrieval for question generation
        Retrieves more chunks and ensures variety
        """
        # Retrieve chunks from vector store
        chunks = await self.retrieve_chunks(
            exam_id=exam_id,
            query=query,
            chapter_id=chapter_id,
            limit=num_chunks,
            min_relevance_score=0.4
        )
        
        # If not enough chunks, try without chapter filter
        if len(chunks) < 10:
            logger.warning(f"Not enough chunks for chapter {chapter_id}, expanding search")
            additional_chunks = await self.retrieve_chunks(
                exam_id=exam_id,
                query=query,
                limit=num_chunks - len(chunks),
                min_relevance_score=0.3
            )
            chunks.extend(additional_chunks)
        
        # Deduplicate by content
        seen_content = set()
        unique_chunks = []
        for chunk in chunks:
            content_hash = chunk.get("content_hash") or hash(chunk.get("content", ""))
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)
        
        logger.info(f"Retrieved {len(unique_chunks)} unique chunks for question generation")
        return unique_chunks