# app/ingestion/resource_pipeline.py
#
# UPLOAD -> FILE STORAGE -> PARSE -> CLEAN -> CHUNK -> EMBED -> STORE ->
# CURRICULUM EXTRACTION (spec section 2). Runs as a FastAPI BackgroundTask
# (app.services.workers), one resource at a time, updating Resource.status
# at each stage so the admin UI can show live processing state.
import hashlib
from typing import List
from app.database import AsyncSessionLocal
from app.database.models import ProcessingStatus, Document, DocumentChunk
from app.database.repositories import (
    ResourceRepository, DocumentChunkRepository, SubjectRepository,
    ChapterRepository, TopicRepository
)
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.ingestion.parsers import parse_document, UnsupportedFileType
from app.ingestion.chunking import chunk_text
from app.ingestion.curriculum_extraction import extract_curriculum, assign_chunks_to_curriculum
import logging

logger = logging.getLogger(__name__)

EMBEDDING_BATCH_SIZE = 64


async def process_resource(resource_id: str) -> None:
    async with AsyncSessionLocal() as session:
        resource_repo = ResourceRepository(session)
        resource = await resource_repo.get_by_id(resource_id)
        if not resource:
            logger.error(f"process_resource: resource {resource_id} not found")
            return

        try:
            await resource_repo.update_status(resource_id, ProcessingStatus.EXTRACTING_TEXT)
            text, page_count = parse_document(resource.file_path)
            if not text.strip():
                raise ValueError("No extractable text found in document")

            await resource_repo.update_status(resource_id, ProcessingStatus.CHUNKING)
            chunks = chunk_text(text)
            if not chunks:
                raise ValueError("Document produced no chunks after cleaning")

            document = Document(
                resource_id=resource.id,
                title=resource.filename,
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                page_count=page_count
            )
            session.add(document)
            await session.commit()
            await session.refresh(document)

            await resource_repo.update_status(resource_id, ProcessingStatus.EMBEDDING)
            embedding_service = EmbeddingService()
            db_chunks: List[DocumentChunk] = []
            for batch_start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[batch_start:batch_start + EMBEDDING_BATCH_SIZE]
                embeddings = await embedding_service.embed_batch([c.content for c in batch])
                for chunk, embedding in zip(batch, embeddings):
                    db_chunk = DocumentChunk(
                        document_id=document.id,
                        chunk_index=chunk.index,
                        content=chunk.content,
                        token_count=chunk.token_count,
                        embedding=embedding or None,
                        exam_id=resource.exam_id
                    )
                    session.add(db_chunk)
                    db_chunks.append(db_chunk)
            await session.commit()

            await resource_repo.update_status(resource_id, ProcessingStatus.EXTRACTING_STRUCTURE)
            subject_repo = SubjectRepository(session)
            chapter_repo = ChapterRepository(session)
            topic_repo = TopicRepository(session)
            llm_service = LLMService()

            from app.database.repositories import ExamRepository
            exam_row = await ExamRepository(session).get_by_id(str(resource.exam_id))
            exam_name = exam_row.name if exam_row else "this exam"

            chunk_sample = [{"id": str(c.id), "content": c.content} for c in db_chunks]
            await extract_curriculum(
                exam_id=str(resource.exam_id),
                exam_name=exam_name,
                chunks=chunk_sample,
                subject_repo=subject_repo,
                chapter_repo=chapter_repo,
                topic_repo=topic_repo,
                llm_service=llm_service
            )
            await assign_chunks_to_curriculum(
                exam_id=str(resource.exam_id),
                db_chunks=db_chunks,
                chapter_repo=chapter_repo,
                embedding_service=embedding_service
            )
            await session.commit()

            await resource_repo.update_status(resource_id, ProcessingStatus.COMPLETED)
            from datetime import datetime
            resource.processed_at = datetime.utcnow()
            await session.commit()
            logger.info(f"Resource {resource_id} processed: {len(db_chunks)} chunks, {page_count} pages")

        except UnsupportedFileType as e:
            logger.error(f"Resource {resource_id} unsupported file type: {e}")
            await resource_repo.update_status(resource_id, ProcessingStatus.FAILED, str(e))
        except Exception as e:
            logger.error(f"Resource {resource_id} processing failed: {e}")
            await resource_repo.update_status(resource_id, ProcessingStatus.FAILED, str(e))
