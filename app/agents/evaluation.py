# app/agents/evaluation.py
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.database.repositories import QuizRepository, BlueprintRepository
from app.services.llm import LLMService
from app.services.retrieval import RetrievalService
import logging

logger = logging.getLogger(__name__)


class EvaluationAgent:
    """
    Agent 9: Evaluation Agent
    Responsibilities: Score quiz, calculate metrics, generate performance analysis.
    MCQ correctness and score are computed deterministically from the stored
    correct_answer and the exam's marking scheme (spec section 25) — the LLM
    is used only for the qualitative strong/weak-area narrative afterward.
    """

    def __init__(
        self,
        quiz_repo: QuizRepository,
        llm_service: LLMService,
        retrieval_service: RetrievalService,
        blueprint_repo: Optional[BlueprintRepository] = None
    ):
        self.quiz_repo = quiz_repo
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.blueprint_repo = blueprint_repo
        
    async def evaluate_quiz(
        self,
        quiz_id: str
    ) -> Dict[str, Any]:
        """Evaluate a completed quiz"""
        try:
            # Get quiz data
            quiz = await self.quiz_repo.get_by_id(quiz_id)
            if not quiz:
                raise ValueError(f"Quiz {quiz_id} not found")
            
            questions = quiz.generated_questions
            user_answers = quiz.user_answers or []

            # Convert answers to dict
            answers_dict = {
                item["question_index"]: item["answer"]
                for item in user_answers
            }

            # Marking scheme from the active exam blueprint, if available (spec section 25)
            marks_per_question = 1.0
            negative_marks_per_wrong = 0.0
            if self.blueprint_repo:
                blueprint = await self.blueprint_repo.get_active_by_exam(str(quiz.exam_id))
                if blueprint:
                    marking = blueprint.blueprint_data.get("marking_scheme", {})
                    marks_per_question = marking.get("average_marks_per_question") or 1.0
                    if marking.get("negative_marking"):
                        negative_marks_per_wrong = marking.get("average_negative_marks") or 0.0

            # Evaluate each question
            results = []
            correct_count = 0
            incorrect_count = 0
            unanswered_count = 0

            for idx, question in enumerate(questions):
                selected_answer = answers_dict.get(idx)
                correct_answer = question.get("correct_answer", "")

                is_correct = False
                if selected_answer:
                    # Deterministic comparison
                    is_correct = selected_answer.strip() == correct_answer.strip()
                    if is_correct:
                        correct_count += 1
                    else:
                        incorrect_count += 1
                else:
                    unanswered_count += 1

                results.append({
                    "question_index": idx,
                    "selected_answer": selected_answer,
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                    "question": question
                })

            # Calculate score using the blueprint's marking scheme (positive marks per
            # correct answer, minus negative marks per wrong answer when configured;
            # unanswered questions are never penalized)
            total_questions = len(questions)
            attempted = correct_count + incorrect_count
            score = round(correct_count * marks_per_question - incorrect_count * negative_marks_per_wrong, 2)

            # Calculate accuracy
            accuracy = (correct_count / attempted * 100) if attempted > 0 else 0
            
            # Calculate time taken
            time_taken = None
            if quiz.start_time and quiz.submitted_at:
                time_taken = int((quiz.submitted_at - quiz.start_time).total_seconds())
            
            # Generate performance analysis
            performance = await self._generate_performance_analysis(
                results=results,
                exam_id=quiz.exam_id,
                chapter_id=quiz.chapter_id
            )
            
            evaluation = {
                "total_questions": total_questions,
                "attempted": attempted,
                "correct": correct_count,
                "incorrect": incorrect_count,
                "unanswered": unanswered_count,
                "score": score,
                "accuracy": accuracy,
                "time_taken_seconds": time_taken,
                "results": results,
                "performance": performance,
                "evaluated_at": datetime.utcnow().isoformat()
            }
            
            # Update quiz with evaluation
            await self.quiz_repo.update(
                quiz_id,
                {
                    "status": "evaluated",
                    "evaluation": evaluation,
                    "score": score,
                    "time_taken_seconds": time_taken
                }
            )
            
            logger.info(f"Quiz {quiz_id} evaluated: {correct_count}/{total_questions} correct")
            return evaluation
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    async def _generate_performance_analysis(
        self,
        results: List[Dict[str, Any]],
        exam_id: str,
        chapter_id: str
    ) -> Dict[str, Any]:
        """Generate performance analysis using LLM"""
        
        # Prepare results summary
        results_summary = []
        for r in results:
            results_summary.append({
                "question": r["question"]["question_text"][:100],
                "correct": r["is_correct"],
                "difficulty": r["question"].get("difficulty", "medium")
            })
        
        # Get the user's performance by difficulty
        difficulty_performance = {}
        for r in results:
            difficulty = r["question"].get("difficulty", "medium")
            if difficulty not in difficulty_performance:
                difficulty_performance[difficulty] = {"correct": 0, "total": 0}
            difficulty_performance[difficulty]["total"] += 1
            if r["is_correct"]:
                difficulty_performance[difficulty]["correct"] += 1
        
        # Generate analysis with LLM
        prompt = f"""
Analyze the following exam performance and provide insights.

Results: {results_summary}

Difficulty Performance: {difficulty_performance}

Provide a JSON analysis with:
1. strong_areas: List of topics/concepts the user performed well on
2. weak_areas: List of topics/concepts needing improvement
3. overall_assessment: Brief summary of performance
4. recommendations: Study recommendations

Focus on identifying patterns in correct vs incorrect answers.
"""
        
        try:
            analysis = await self.llm_service.generate_json(prompt)
            return analysis
        except Exception as e:
            logger.error(f"Failed to generate performance analysis: {e}")
            return {
                "strong_areas": [],
                "weak_areas": [],
                "overall_assessment": "Performance analysis unavailable",
                "recommendations": []
            }