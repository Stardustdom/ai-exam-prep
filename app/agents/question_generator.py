# app/agents/question_generator.py
from typing import List, Dict, Any, Optional
import json
from app.agents.state import ExamSessionState
from app.services.llm import LLMService
from app.services.embeddings import EmbeddingService
from app.database.repositories import QuizRepository
import logging

logger = logging.getLogger(__name__)


class QuestionGeneratorAgent:
    """
    Agent 6: Question Generator
    Responsibilities: Generate questions following blueprint, create MCQs with distractors
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        quiz_repo: QuizRepository
    ):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.quiz_repo = quiz_repo
        
    async def generate_questions(
        self,
        exam_id: str,
        blueprint: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
        chapter_name: str,
        question_count: int,
        difficulty_distribution: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generate questions based on blueprint and retrieved chunks
        """
        # Prepare context from chunks
        context = self._prepare_context(retrieved_chunks)
        
        # Determine question distribution
        distribution = self._calculate_distribution(
            question_count,
            difficulty_distribution
        )
        
        # Generate questions in batches
        all_questions = []
        for difficulty, count in distribution.items():
            if count > 0:
                questions = await self._generate_batch(
                    exam_id=exam_id,
                    blueprint=blueprint,
                    context=context,
                    chapter_name=chapter_name,
                    count=count,
                    difficulty=difficulty
                )
                all_questions.extend(questions)
        
        # Shuffle questions
        import random
        random.shuffle(all_questions)
        
        logger.info(f"Generated {len(all_questions)} questions")
        return all_questions
    
    async def _generate_batch(
        self,
        exam_id: str,
        blueprint: Dict[str, Any],
        context: str,
        chapter_name: str,
        count: int,
        difficulty: str
    ) -> List[Dict[str, Any]]:
        """Generate a batch of questions with same difficulty"""
        
        prompt = self._build_generation_prompt(
            blueprint=blueprint,
            context=context,
            chapter_name=chapter_name,
            count=count,
            difficulty=difficulty
        )
        
        try:
            response = await self.llm_service.generate_json(prompt)
            
            # Parse and validate questions
            questions = response.get("questions", [])
            validated_questions = []
            
            for q in questions:
                if self._validate_question(q):
                    # Add metadata
                    q["difficulty"] = difficulty
                    q["chapter"] = chapter_name
                    q["exam_id"] = exam_id
                    
                    # Generate question fingerprint
                    q["fingerprint"] = self._generate_fingerprint(q)
                    
                    validated_questions.append(q)
            
            return validated_questions
            
        except Exception as e:
            logger.error(f"Failed to generate questions: {e}")
            return []
    
    def _build_generation_prompt(
        self,
        blueprint: Dict[str, Any],
        context: str,
        chapter_name: str,
        count: int,
        difficulty: str
    ) -> str:
        """Build the prompt for question generation"""
        
        question_type = blueprint.get("question_type", "MCQ")
        options_count = blueprint.get("options", 4)
        
        prompt = f"""
You are an expert exam question generator. Generate {count} {question_type} questions about "{chapter_name}" with {difficulty} difficulty.

**IMPORTANT RULES:**
1. Questions MUST be based ONLY on the provided study material context
2. Do NOT generate questions about topics not covered in the context
3. Each question must have exactly one correct answer
4. Generate {options_count} options for each question
5. Questions should test understanding, not just memorization
6. Follow the exam style from the blueprint

**Context (study material):**
{context}

**Exam Blueprint:**
{json.dumps(blueprint, indent=2)}

**Question Format:**
Return a JSON array of questions with this structure:
{{
    "questions": [
        {{
            "question_text": "The question text",
            "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
            "correct_answer": "A) option1",
            "explanation": "Explanation of the correct answer",
            "source_reference": "Reference to the source material (e.g., chapter, section)"
        }}
    ]
}}

Generate {count} questions of {difficulty} difficulty level about {chapter_name}.
"""
        return prompt
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved chunks"""
        contexts = []
        for i, chunk in enumerate(chunks[:20], 1):  # Limit to 20 chunks
            content = chunk.get("content", "")
            if content:
                contexts.append(f"[Source {i}] {content}")
        
        return "\n\n".join(contexts)
    
    def _calculate_distribution(
        self,
        total_count: int,
        difficulty_distribution: Dict[str, float]
    ) -> Dict[str, int]:
        """Calculate how many questions per difficulty level"""
        distribution = {}
        remaining = total_count
        
        # Sort difficulties by percentage
        sorted_difficulties = sorted(
            difficulty_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for difficulty, percentage in sorted_difficulties:
            count = int(total_count * percentage)
            distribution[difficulty] = count
            remaining -= count
        
        # Distribute remaining questions to first difficulty
        if remaining > 0 and sorted_difficulties:
            first_difficulty = sorted_difficulties[0][0]
            distribution[first_difficulty] = distribution.get(first_difficulty, 0) + remaining
        
        return distribution
    
    def _validate_question(self, question: Dict[str, Any]) -> bool:
        """Validate a generated question"""
        # Check required fields
        required_fields = ["question_text", "options", "correct_answer"]
        for field in required_fields:
            if field not in question or not question[field]:
                return False
        
        # Check options
        if not isinstance(question["options"], list) or len(question["options"]) < 2:
            return False
        
        # Check that correct answer is in options
        if question["correct_answer"] not in question["options"]:
            # Try to match without prefix
            correct = question["correct_answer"].strip()
            for opt in question["options"]:
                if correct in opt or opt in correct:
                    question["correct_answer"] = opt
                    break
            else:
                return False
        
        # Check for duplicates in options
        if len(set(question["options"])) != len(question["options"]):
            return False
        
        return True
    
    def _generate_fingerprint(self, question: Dict[str, Any]) -> str:
        """Generate a fingerprint for deduplication"""
        import hashlib
        text = question.get("question_text", "")
        # Normalize text
        text = text.lower().strip()
        # Remove extra whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        # Hash
        return hashlib.md5(text.encode('utf-8')).hexdigest()