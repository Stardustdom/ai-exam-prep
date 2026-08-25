# app/workers/expiry.py
#
# Server-side quiz expiry (spec section 23). Runs on a schedule (see
# app.services.workers beat_schedule), independent of the Telegram
# conversation: an inactive user, a delayed webhook, or a server restart
# must never let a quiz run past its duration. This is a plain DB sweep —
# it does NOT go through the LangGraph thread, because the whole point is
# that it must work even if that thread is never resumed again. The graph's
# `_handle_quiz_node` separately re-checks quiz.status on every resume so a
# thread that *does* get resumed after this sweep ran picks up the
# already-submitted/evaluated result instead of re-prompting a stale question.
from datetime import datetime, timedelta
from app.database import AsyncSessionLocal
from app.database.repositories import QuizRepository, UserRepository, ChatSessionRepository, BlueprintRepository
from app.services.telegram import TelegramService
from app.agents.evaluation import EvaluationAgent
from app.services.llm import LLMService
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStoreService
from app.services.embeddings import EmbeddingService
import logging

logger = logging.getLogger(__name__)


async def sweep_expired_quizzes() -> int:
    """Force-submits and evaluates every in-progress quiz past its deadline. Returns count handled."""
    async with AsyncSessionLocal() as session:
        quiz_repo = QuizRepository(session)
        user_repo = UserRepository(session)
        chat_session_repo = ChatSessionRepository(session)
        telegram_service = TelegramService()
        evaluation_agent = EvaluationAgent(
            quiz_repo, LLMService(), RetrievalService(VectorStoreService(), EmbeddingService()),
            blueprint_repo=BlueprintRepository(session)
        )

        now = datetime.utcnow()
        in_progress = await quiz_repo.get_by_status("started")
        expired = [
            q for q in in_progress
            if q.start_time and now >= q.start_time + timedelta(minutes=q.duration_minutes)
        ]

        for quiz in expired:
            try:
                await quiz_repo.update(str(quiz.id), {
                    "status": "submitted",
                    "submitted_at": quiz.start_time + timedelta(minutes=quiz.duration_minutes)
                })
                evaluation = await evaluation_agent.evaluate_quiz(str(quiz.id))
                logger.info(f"Auto-submitted expired quiz {quiz.id}: {evaluation.get('correct')}/{evaluation.get('total_questions')} correct")

                chat_session = await chat_session_repo.get_by_user(str(quiz.user_id))
                if chat_session:
                    await telegram_service.send_message(
                        chat_id=int(chat_session.telegram_chat_id),
                        text=(
                            "⏰ Time's up! Your test was submitted automatically.\n\n"
                            f"Score: {evaluation.get('score', 0)} | "
                            f"Correct: {evaluation.get('correct', 0)}/{evaluation.get('total_questions', 0)} | "
                            f"Accuracy: {evaluation.get('accuracy', 0):.1f}%\n\n"
                            "Send any message to see full results."
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to auto-submit expired quiz {quiz.id}: {e}")

        return len(expired)
