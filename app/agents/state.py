# app/agents/state.py
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class SessionStep(str, Enum):
    START = "start"
    SELECT_EXAM = "select_exam"
    SELECT_QUESTION_COUNT = "select_question_count"
    SELECT_CHAPTER = "select_chapter"
    SELECT_DURATION = "select_duration"
    GENERATING_QUIZ = "generating_quiz"
    QUIZ_READY = "quiz_ready"
    QUIZ_IN_PROGRESS = "quiz_in_progress"
    QUIZ_SUBMITTED = "quiz_submitted"
    EVALUATING = "evaluating"
    RESULTS = "results"


class ExamSessionState(BaseModel):
    """Strongly typed state for the exam session"""
    
    # User info
    user_id: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Current step
    current_step: SessionStep = SessionStep.START
    previous_step: Optional[SessionStep] = None
    
    # Exam selection
    exam_id: Optional[str] = None
    exam_name: Optional[str] = None
    exam_short_name: Optional[str] = None
    exam_confidence: Optional[float] = None
    exam_matched_method: Optional[str] = None  # exact, alias, semantic
    
    # Quiz parameters
    question_count: Optional[int] = None
    chapter_id: Optional[str] = None
    chapter_name: Optional[str] = None
    topic_id: Optional[str] = None
    topic_name: Optional[str] = None
    duration_minutes: Optional[int] = None
    
    # Quiz data
    quiz_id: Optional[str] = None
    questions: Optional[List[Dict[str, Any]]] = None
    current_question_index: int = 0
    answers: Dict[int, str] = Field(default_factory=dict)
    
    # Timing
    start_time: Optional[datetime] = None
    expiry_time: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    
    # Status
    status: str = "active"
    
    # Results
    score: Optional[float] = None
    evaluation: Optional[Dict[str, Any]] = None
    # "" (never None) means "no error". LangGraph checkpoints each Pydantic
    # field as an independent channel, and — confirmed by direct
    # reproduction against this project's real checkpointer/Postgres setup —
    # a node returning None for a field that previously held a real value
    # does NOT propagate as a clear; the old value silently survives into
    # the NEXT node's view of state. Every node that sets state.error = ""
    # (never None) after consuming it, and every read uses `or` against a
    # falsy value, so "" behaves identically to None everywhere it's
    # checked — the only thing that changes is that clearing an old error
    # actually sticks. Getting this wrong is exactly what caused a stale
    # error from a retried step to render as the *next* step's prompt.
    error: str = ""
    retry_count: int = 0
    
    # Context data (for debugging/observability)
    context: Dict[str, Any] = Field(default_factory=dict)
    llm_calls: int = 0
    retrieval_calls: int = 0
    
    class Config:
        arbitrary_types_allowed = True