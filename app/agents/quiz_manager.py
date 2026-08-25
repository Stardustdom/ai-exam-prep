# app/agents/quiz_manager.py
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import uuid
from app.agents.state import ExamSessionState, SessionStep
from app.database.repositories import QuizRepository
from app.services.telegram import TelegramService
import logging

logger = logging.getLogger(__name__)


class QuizManagerAgent:
    """
    Agent 8: Quiz Manager
    Responsibilities: Create quiz, store questions, track state, handle submission
    """
    
    def __init__(
        self,
        quiz_repo: QuizRepository,
        telegram_service: TelegramService
    ):
        self.quiz_repo = quiz_repo
        self.telegram_service = telegram_service
        
    async def create_quiz(
        self,
        state: ExamSessionState,
        questions: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Create a new quiz"""
        try:
            # "overall" is the UI sentinel for "no specific chapter" (see ExamGraph)
            # and must never be written to the chapter_id UUID foreign key.
            # ExamSessionState uses "" (not None) as its "unset" value — see
            # ExamSessionState.error's docstring — so both are checked here.
            real_chapter_id = None if state.chapter_id in (None, "", "overall") else uuid.UUID(state.chapter_id)
            real_topic_id = uuid.UUID(state.topic_id) if state.topic_id else None

            quiz_data = {
                "user_id": uuid.UUID(state.user_id),
                "exam_id": uuid.UUID(state.exam_id),
                "blueprint_version": state.context.get("blueprint_version", 1),
                "chapter_id": real_chapter_id,
                "topic_id": real_topic_id,
                "question_count": len(questions),
                "duration_minutes": state.duration_minutes,
                "status": "generated",
                "generated_questions": questions,
                "user_answers": [],
                "extra_data": {
                    "chapter_name": state.chapter_name,
                    "topic_name": state.topic_name,
                    "exam_name": state.exam_name
                }
            }
            
            quiz = await self.quiz_repo.create(quiz_data)
            quiz_id = str(quiz.id)
            
            state.quiz_id = quiz_id
            state.questions = questions
            state.current_question_index = 0
            
            logger.info(f"Quiz created: {quiz_id} with {len(questions)} questions")
            return quiz_id
            
        except Exception as e:
            logger.error(f"Failed to create quiz: {e}")
            return None
    
    async def start_quiz(self, quiz_id: str) -> bool:
        """Start a quiz (set start time)"""
        try:
            await self.quiz_repo.update(
                quiz_id,
                {
                    "status": "started",
                    "start_time": datetime.utcnow()
                }
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start quiz: {e}")
            return False
    
    async def submit_quiz(
        self,
        quiz_id: str,
        answers: Dict[int, str]
    ) -> bool:
        """Submit a quiz with answers"""
        try:
            # Format answers
            formatted_answers = [
                {"question_index": idx, "answer": answer}
                for idx, answer in answers.items()
            ]
            
            await self.quiz_repo.update(
                quiz_id,
                {
                    "status": "submitted",
                    "user_answers": formatted_answers,
                    "submitted_at": datetime.utcnow()
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit quiz: {e}")
            return False
    
    async def get_quiz(self, quiz_id: str) -> Optional[Dict[str, Any]]:
        """Get quiz data"""
        try:
            quiz = await self.quiz_repo.get_by_id(quiz_id)
            if quiz:
                return {
                    "id": str(quiz.id),
                    "questions": quiz.generated_questions,
                    "answers": quiz.user_answers or [],
                    "status": quiz.status,
                    "start_time": quiz.start_time,
                    "submitted_at": quiz.submitted_at,
                    "duration_minutes": quiz.duration_minutes
                }
        except Exception as e:
            logger.error(f"Failed to get quiz: {e}")
        return None
    
    async def get_next_question(
        self,
        state: ExamSessionState
    ) -> Optional[Dict[str, Any]]:
        """Get the next unanswered question"""
        if not state.questions:
            return None
            
        # Find first unanswered question
        for idx, question in enumerate(state.questions):
            if idx not in state.answers:
                state.current_question_index = idx
                return question
                
        # All questions answered
        return None
    
    async def get_question_at_index(
        self,
        state: ExamSessionState,
        index: int
    ) -> Optional[Dict[str, Any]]:
        """Get question at specific index"""
        if 0 <= index < len(state.questions):
            state.current_question_index = index
            return state.questions[index]
        return None
    
    async def answer_question(
        self,
        state: ExamSessionState,
        question_index: int,
        answer: str
    ) -> None:
        """Record an answer, both in the (checkpointed) graph state and the DB row.
        The DB write matters even before final submission: the scheduled expiry
        worker (app.workers.expiry) force-submits quizzes past their deadline by
        reading straight from the `quizzes` table, independent of whether the
        user's LangGraph thread is ever resumed again."""
        state.answers[question_index] = answer

        if state.quiz_id:
            formatted_answers = [{"question_index": idx, "answer": ans} for idx, ans in state.answers.items()]
            await self.quiz_repo.update(state.quiz_id, {"user_answers": formatted_answers})

        logger.debug(f"Recorded answer for question {question_index}")
    
    async def check_expiry(
        self,
        state: ExamSessionState
    ) -> bool:
        """Check if quiz has expired"""
        if not state.expiry_time:
            return False
            
        return datetime.utcnow() >= state.expiry_time