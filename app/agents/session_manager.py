# app/agents/session_manager.py
from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, END
from app.agents.state import ExamSessionState, SessionStep
from app.database.repositories import UserRepository, ChatSessionRepository
from app.services.telegram import TelegramService
import logging

logger = logging.getLogger(__name__)


class SessionManagerAgent:
    """
    Agent 1: Session Manager
    Responsibilities: Create/retrieve user session, maintain state, persist state
    """
    
    def __init__(
        self,
        user_repo: UserRepository,
        chat_session_repo: ChatSessionRepository,
        telegram_service: TelegramService
    ):
        self.user_repo = user_repo
        self.chat_session_repo = chat_session_repo
        self.telegram_service = telegram_service
        
    async def initialize_session(
        self,
        state: ExamSessionState,
        telegram_user_id: str,
        telegram_chat_id: str,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        reset: bool = False
    ) -> ExamSessionState:
        """Initialize or retrieve user session.

        `reset=True` (a /start while mid-conversation) must skip restoring the
        OLD saved exam/chapter/quiz_id/current_step onto the caller's fresh
        state — otherwise the caller's brand-new graph.start() call still
        fast-forwards straight back through every already-answered step
        (exam set -> skip, chapter set -> skip, quiz_id set -> regenerate a
        quiz...) and lands right back on the same stuck prompt, which is
        exactly what silently defeated the *previous* attempt at a real
        /start reset. The stale row itself is also cleared so a later normal
        (non-reset) turn doesn't resurrect it either.
        """
        try:
            # Get or create user
            user = await self.user_repo.get_or_create(
                telegram_user_id=telegram_user_id,
                username=username,
                first_name=first_name,
                last_name=last_name
            )

            state.user_id = str(user.id)
            state.telegram_chat_id = telegram_chat_id

            # Get or create chat session
            chat_session = await self.chat_session_repo.get_or_create(
                user_id=str(user.id),
                telegram_chat_id=telegram_chat_id
            )

            state.session_id = str(chat_session.id)

            if reset:
                await self.chat_session_repo.update_state(
                    session_id=state.session_id, current_step=SessionStep.START.value, state_data={}
                )
                return state

            state.current_step = SessionStep(chat_session.current_step)

            # Restore state data if available
            if chat_session.state_data:
                for key, value in chat_session.state_data.items():
                    if hasattr(state, key):
                        setattr(state, key, value)

            logger.info(f"Session initialized for user {telegram_user_id}, step: {state.current_step}")
            return state

        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")
            state.error = str(e)
            return state
    
    async def persist_state(self, state: ExamSessionState) -> None:
        """Persist current state to database"""
        try:
            if not state.session_id:
                return
                
            state_data = {
                "current_step": state.current_step.value,
                "exam_id": state.exam_id,
                "exam_name": state.exam_name,
                "question_count": state.question_count,
                "chapter_id": state.chapter_id,
                "chapter_name": state.chapter_name,
                "topic_id": state.topic_id,
                "topic_name": state.topic_name,
                "duration_minutes": state.duration_minutes,
                "quiz_id": state.quiz_id,
                "current_question_index": state.current_question_index,
                "answers": state.answers,
                "start_time": state.start_time.isoformat() if state.start_time else None,
                "expiry_time": state.expiry_time.isoformat() if state.expiry_time else None,
            }
            
            await self.chat_session_repo.update_state(
                session_id=state.session_id,
                current_step=state.current_step.value,
                state_data=state_data
            )
            
            logger.debug(f"State persisted for session {state.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")
    
    async def transition_step(
        self,
        state: ExamSessionState,
        new_step: SessionStep
    ) -> ExamSessionState:
        """Transition to a new step"""
        state.previous_step = state.current_step
        state.current_step = new_step
        await self.persist_state(state)
        return state