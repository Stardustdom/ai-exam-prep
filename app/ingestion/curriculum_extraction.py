# app/ingestion/curriculum_extraction.py
#
# Derives the subject -> chapter -> topic hierarchy from the uploaded corpus
# (spec section 6) — never a hardcoded structure. Two passes:
#   1. An LLM proposes a hierarchy from a sample of chunks (with a source
#      chunk id attached to each leaf, for explainability/admin drill-down).
#   2. Every chunk in the resource is assigned to its nearest topic/chapter
#      by embedding-similarity (cheap, deterministic) rather than one LLM
#      call per chunk.
from typing import List, Dict, Any
import math
from app.database.models import DocumentChunk
from app.database.repositories import SubjectRepository, ChapterRepository, TopicRepository
from app.services.llm import LLMService
from app.services.embeddings import EmbeddingService
import logging

logger = logging.getLogger(__name__)

MAX_SAMPLE_CHUNKS = 40
ASSIGNMENT_SIMILARITY_THRESHOLD = 0.3


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def extract_curriculum(
    exam_id: str,
    exam_name: str,
    chunks: List[Dict[str, Any]],
    subject_repo: SubjectRepository,
    chapter_repo: ChapterRepository,
    topic_repo: TopicRepository,
    llm_service: LLMService
) -> Dict[str, Any]:
    """
    chunks: [{"id": chunk_id, "content": str}, ...] for ONE resource (or a
    representative sample of it). Upserts Subject/Chapter/Topic rows (shared
    across the whole exam, not per-resource) and returns the proposed
    hierarchy with source chunk references for admin inspection.
    """
    if not chunks:
        return {"subjects": []}

    sample = chunks[:MAX_SAMPLE_CHUNKS]
    context = "\n\n".join(f"[chunk {c['id']}] {c['content'][:800]}" for c in sample)

    prompt = f"""
You are analyzing study material for the exam "{exam_name}". Identify the
educational hierarchy (subjects, chapters, topics) actually present in this
text. Do NOT invent a generic textbook structure — base it only on what the
excerpts below actually cover. If the material only supports one or two
levels, omit the rest (topics are optional).

Excerpts:
{context}

Return JSON exactly in this shape:
{{
  "subjects": [
    {{
      "name": "Subject name",
      "chapters": [
        {{
          "name": "Chapter name",
          "topics": ["Topic name", "..."],
          "source_chunk_ids": ["chunk-id-1", "chunk-id-2"]
        }}
      ]
    }}
  ]
}}
"""
    result = await llm_service.generate_json(
        prompt,
        system="You extract curriculum structure from educational text and respond with strict JSON only."
    )

    subjects_payload = result.get("subjects", []) if isinstance(result, dict) else []
    stored_hierarchy = []

    for subject_data in subjects_payload:
        subject_name = (subject_data.get("name") or "").strip()
        if not subject_name:
            continue
        subject = await subject_repo.get_or_create(exam_id, subject_name)

        chapters_out = []
        for chapter_data in subject_data.get("chapters", []):
            chapter_name = (chapter_data.get("name") or "").strip()
            if not chapter_name:
                continue
            chapter = await chapter_repo.get_or_create(str(subject.id), chapter_name)

            source_refs = [{"chunk_id": cid} for cid in chapter_data.get("source_chunk_ids", [])]
            if chapter.source_references:
                existing_ids = {r.get("chunk_id") for r in chapter.source_references}
                source_refs = chapter.source_references + [r for r in source_refs if r["chunk_id"] not in existing_ids]
            chapter.source_references = source_refs

            topics_out = []
            for topic_name in chapter_data.get("topics", []):
                topic_name = (topic_name or "").strip()
                if not topic_name:
                    continue
                topic = await topic_repo.get_or_create(str(chapter.id), topic_name)
                topics_out.append({"id": str(topic.id), "name": topic.name})

            chapters_out.append({"id": str(chapter.id), "name": chapter.name, "topics": topics_out})

        stored_hierarchy.append({"id": str(subject.id), "name": subject.name, "chapters": chapters_out})

    await chapter_repo.session.commit()
    return {"subjects": stored_hierarchy}


async def assign_chunks_to_curriculum(
    exam_id: str,
    db_chunks: List[DocumentChunk],
    chapter_repo: ChapterRepository,
    embedding_service: EmbeddingService
) -> None:
    """
    Backfills DocumentChunk.subject_id/chapter_id/topic_id by nearest-neighbor
    similarity between each chunk's embedding and each chapter's/topic's name
    embedding. Chunks below the similarity threshold are left exam-scoped
    only (still retrievable, just not attributed to a specific chapter).
    """
    chapters = await chapter_repo.get_by_exam(exam_id)
    if not chapters or not db_chunks:
        return

    candidates = []  # (subject_id, chapter_id, topic_id, name)
    for chapter in chapters:
        candidates.append((str(chapter.subject_id), str(chapter.id), None, chapter.name))
        for topic in (chapter.topics or []):
            candidates.append((str(chapter.subject_id), str(chapter.id), str(topic.id), topic.name))

    if not candidates:
        return

    candidate_embeddings = await embedding_service.embed_batch([c[3] for c in candidates])

    for chunk in db_chunks:
        if not chunk.embedding:
            continue
        best_idx, best_score = None, -1.0
        for i, cand_vec in enumerate(candidate_embeddings):
            if not cand_vec:
                continue
            score = _cosine(list(chunk.embedding), cand_vec)
            if score > best_score:
                best_idx, best_score = i, score

        if best_idx is not None and best_score >= ASSIGNMENT_SIMILARITY_THRESHOLD:
            subject_id, chapter_id, topic_id, _ = candidates[best_idx]
            chunk.subject_id, chunk.chapter_id, chunk.topic_id = subject_id, chapter_id, topic_id
