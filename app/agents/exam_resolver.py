# app/agents/exam_resolver.py
from typing import Optional, Tuple, List, Dict, Any
import re
from app.agents.state import ExamSessionState
from app.database.repositories import ExamRepository, SemanticCacheRepository
from app.services.embeddings import EmbeddingService
from app.services.semantic_cache import SemanticCacheService
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class ExamResolverAgent:
    """
    Agent 2: Exam Resolver
    Responsibilities: Exact matching, alias matching, semantic matching, confidence scoring
    """
    
    def __init__(
        self,
        exam_repo: ExamRepository,
        semantic_cache_repo: SemanticCacheRepository,
        embedding_service: EmbeddingService,
        semantic_cache_service: SemanticCacheService
    ):
        self.exam_repo = exam_repo
        self.semantic_cache_repo = semantic_cache_repo
        self.embedding_service = embedding_service
        self.semantic_cache_service = semantic_cache_service
        
    async def resolve_exam(
        self,
        state: ExamSessionState,
        user_input: str
    ) -> Tuple[Optional[str], Optional[str], Optional[float], str]:
        """
        Resolve exam from user input using multiple strategies
        
        Returns: (exam_id, exam_name, confidence, method)
        """
        if not user_input or not user_input.strip():
            return None, None, 0.0, "empty"
            
        normalized_input = self._normalize_input(user_input)
        
        # Step 1: Check semantic cache
        cached_result = await self.semantic_cache_service.get_resolved_entity(
            query=normalized_input,
            entity_type="exam"
        )
        
        if cached_result:
            logger.info(f"Exam resolved from cache: {cached_result}")
            exam = await self.exam_repo.get_by_id(cached_result["entity_id"])
            if exam and exam.is_active:
                return str(exam.id), exam.name, cached_result["confidence"], "cache"
        
        # Step 2: Exact match on short name or name
        exact_match = await self._exact_match(normalized_input)
        if exact_match:
            exam = await self.exam_repo.get_by_id(exact_match)
            if exam and exam.is_active:
                await self.semantic_cache_service.cache_resolved_entity(
                    query=normalized_input,
                    entity_type="exam",
                    entity_id=str(exam.id),
                    confidence=1.0
                )
                return str(exam.id), exam.name, 1.0, "exact"
        
        # Step 3: Alias match
        alias_match = await self._alias_match(normalized_input)
        if alias_match:
            exam = await self.exam_repo.get_by_id(alias_match)
            if exam and exam.is_active:
                confidence = 0.95
                await self.semantic_cache_service.cache_resolved_entity(
                    query=normalized_input,
                    entity_type="exam",
                    entity_id=str(exam.id),
                    confidence=confidence
                )
                return str(exam.id), exam.name, confidence, "alias"
        
        # Step 4: Semantic search
        semantic_match = await self._semantic_match(normalized_input)
        if semantic_match:
            exam_id, confidence = semantic_match
            if confidence >= settings.exam_match_confidence_threshold:
                exam = await self.exam_repo.get_by_id(exam_id)
                if exam and exam.is_active:
                    await self.semantic_cache_service.cache_resolved_entity(
                        query=normalized_input,
                        entity_type="exam",
                        entity_id=str(exam.id),
                        confidence=confidence
                    )
                    return str(exam.id), exam.name, confidence, "semantic"
            else:
                # Low confidence - need user to confirm
                return None, None, confidence, "semantic_low"
        
        # No match found
        return None, None, 0.0, "none"
    
    async def get_all_active_exams(self) -> List[Dict[str, Any]]:
        """Get all active exams for menu display"""
        exams = await self.exam_repo.get_active_exams()
        return [
            {
                "id": str(exam.id),
                "name": exam.name,
                "short_name": exam.short_name,
                "description": exam.description
            }
            for exam in exams
        ]
    
    def _normalize_input(self, text: str) -> str:
        """Normalize user input"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common stop words
        stop_words = {"exam", "test", "of", "the", "for", "and", "or"}
        words = text.split()
        words = [w for w in words if w not in stop_words]
        text = " ".join(words)
        
        return text
    
    async def _exact_match(self, normalized_input: str) -> Optional[str]:
        """Try exact match on exam name or short name"""
        # Try matching short name
        exam = await self.exam_repo.get_by_short_name(normalized_input.upper())
        if exam:
            return str(exam.id)
        
        # Try matching name
        exam = await self.exam_repo.get_by_name(normalized_input)
        if exam:
            return str(exam.id)
        
        return None
    
    async def _alias_match(self, normalized_input: str) -> Optional[str]:
        """Try matching against aliases"""
        alias = await self.exam_repo.get_alias(normalized_input)
        if alias:
            return str(alias.exam_id)
        return None
    
    async def _semantic_match(self, normalized_input: str) -> Optional[Tuple[str, float]]:
        """Use embeddings for semantic matching"""
        try:
            # Get embedding for input
            embedding = await self.embedding_service.embed_text(normalized_input)
            
            # Search for similar exams
            matches = await self.exam_repo.search_semantic(embedding, limit=3)
            
            if matches:
                # Check confidence threshold
                best_match = matches[0]
                confidence = best_match.get("similarity", 0.0)
                return best_match["id"], confidence
            
        except Exception as e:
            logger.error(f"Semantic match failed: {e}")
            
        return None