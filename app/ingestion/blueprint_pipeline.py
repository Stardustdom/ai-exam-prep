# app/ingestion/blueprint_pipeline.py
#
# SAMPLE PAPER -> PARSE -> QUESTION EXTRACTION -> PATTERN ANALYSIS ->
# EXAM BLUEPRINT (spec section 7). Question *extraction* uses the LLM
# (unstructured text -> structured questions); blueprint *aggregation* is
# deterministic arithmetic over the extracted rows, not another LLM guess,
# so re-running it is stable and auditable.
from typing import List, Dict, Any
from collections import Counter
from app.database import AsyncSessionLocal
from app.database.models import ProcessingStatus
from app.database.repositories import (
    SamplePaperRepository, SampleQuestionRepository, BlueprintRepository, ExamRepository
)
from app.services.llm import LLMService
from app.ingestion.parsers import parse_document, UnsupportedFileType
import logging

logger = logging.getLogger(__name__)

QUESTION_EXTRACTION_PROMPT = """
You are analyzing a past/sample exam paper for "{exam_name}". Extract every
question you can find. For each question determine, as best as the text
supports:
- question_text
- question_type (e.g. "MCQ", "numerical", "theory")
- options (list of option strings; omit/empty if not multiple choice)
- correct_answer (only if the answer key is present in the text; else null)
- difficulty ("easy", "medium", or "hard" — your best judgment from complexity)
- subject, chapter, topic (best guess from the question content; null if unclear)
- marks (positive marks for a correct answer, if stated; else null)
- negative_marks (marks deducted for a wrong answer, if stated; else null)

Text of the paper:
{content}

Return JSON exactly as: {{"questions": [ {{...one object per field above...}} ]}}
Only include questions you can actually find in the text — do not invent any.
"""


async def extract_sample_questions(sample_paper_id: str) -> None:
    async with AsyncSessionLocal() as session:
        paper_repo = SamplePaperRepository(session)
        question_repo = SampleQuestionRepository(session)
        exam_repo = ExamRepository(session)

        paper = await paper_repo.get_by_id(sample_paper_id)
        if not paper:
            logger.error(f"extract_sample_questions: sample paper {sample_paper_id} not found")
            return

        try:
            await paper_repo.update_status(sample_paper_id, ProcessingStatus.EXTRACTING_TEXT)
            text, _ = parse_document(paper.file_path)
            if not text.strip():
                raise ValueError("No extractable text found in sample paper")

            exam = await exam_repo.get_by_id(str(paper.exam_id))
            exam_name = exam.name if exam else "this exam"

            await paper_repo.update_status(sample_paper_id, ProcessingStatus.EXTRACTING_STRUCTURE)
            llm_service = LLMService()
            # Sample papers are bounded documents (a handful of pages), so
            # unlike study resources this can go to the LLM as one prompt.
            result = await llm_service.generate_json(
                QUESTION_EXTRACTION_PROMPT.format(exam_name=exam_name, content=text[:24000]),
                system="You extract exam questions from raw text and respond with strict JSON only."
            )
            questions = result.get("questions", []) if isinstance(result, dict) else []

            stored = 0
            for q in questions:
                question_text = (q.get("question_text") or "").strip()
                if not question_text:
                    continue
                await question_repo.create({
                    "sample_paper_id": paper.id,
                    "question_text": question_text,
                    "question_type": q.get("question_type") or "MCQ",
                    "options": q.get("options") or None,
                    "correct_answer": q.get("correct_answer"),
                    "difficulty": q.get("difficulty"),
                    "marks": q.get("marks"),
                    "negative_marks": q.get("negative_marks"),
                    "subject": q.get("subject"),
                    "chapter": q.get("chapter"),
                    "topic": q.get("topic"),
                })
                stored += 1

            if stored == 0:
                raise ValueError("No questions could be extracted from this paper")

            from datetime import datetime
            paper.processed_at = datetime.utcnow()
            await paper_repo.update_status(sample_paper_id, ProcessingStatus.COMPLETED)
            logger.info(f"Sample paper {sample_paper_id}: extracted {stored} questions")

        except UnsupportedFileType as e:
            await paper_repo.update_status(sample_paper_id, ProcessingStatus.FAILED, str(e))
        except Exception as e:
            logger.error(f"Sample paper {sample_paper_id} extraction failed: {e}")
            await paper_repo.update_status(sample_paper_id, ProcessingStatus.FAILED, str(e))

    # Regenerate the exam-wide blueprint from all sample papers now that this one has questions
    await generate_blueprint(str(paper.exam_id))


def _distribution(values: List[str]) -> Dict[str, float]:
    values = [v for v in values if v]
    if not values:
        return {}
    counts = Counter(values)
    total = sum(counts.values())
    return {k: round(v / total, 4) for k, v in counts.items()}


async def generate_blueprint(exam_id: str) -> Dict[str, Any]:
    """Aggregate every extracted SampleQuestion for this exam into a structured blueprint (spec section 7)."""
    async with AsyncSessionLocal() as session:
        question_repo = SampleQuestionRepository(session)
        blueprint_repo = BlueprintRepository(session)

        questions = await question_repo.get_by_exam(exam_id)
        if not questions:
            logger.warning(f"generate_blueprint: no sample questions for exam {exam_id} yet")
            return {}

        option_counts = [len(q.options) for q in questions if q.options]
        typical_options = Counter(option_counts).most_common(1)[0][0] if option_counts else 4

        marks = [q.marks for q in questions if q.marks is not None]
        negative_marks = [q.negative_marks for q in questions if q.negative_marks is not None]

        blueprint_data = {
            "question_type": Counter(q.question_type for q in questions).most_common(1)[0][0],
            "options": typical_options,
            "difficulty": _distribution([q.difficulty for q in questions]),
            "subject_distribution": _distribution([q.subject for q in questions]),
            "chapter_distribution": _distribution([q.chapter for q in questions]),
            "topic_distribution": _distribution([q.topic for q in questions]),
            "marking_scheme": {
                "average_marks_per_question": round(sum(marks) / len(marks), 2) if marks else None,
                "negative_marking": bool(negative_marks),
                "average_negative_marks": round(sum(negative_marks) / len(negative_marks), 2) if negative_marks else 0,
            },
            "sample_size": len(questions),
        }

        previous = await blueprint_repo.get_active_by_exam(exam_id)
        next_version = (previous.version + 1) if previous else 1
        if previous:
            previous.is_active = False

        source_paper_ids = list({str(q.sample_paper_id) for q in questions})
        blueprint = await blueprint_repo.create({
            "exam_id": exam_id,
            "version": next_version,
            "blueprint_data": blueprint_data,
            "is_active": True,
            "source_sample_papers": source_paper_ids,
        })
        await session.commit()
        logger.info(f"Blueprint v{next_version} generated for exam {exam_id} from {len(questions)} questions")
        return blueprint_data
