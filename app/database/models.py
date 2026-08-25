# app/database/models.py
#
# NOTE: no column here is named `metadata` — SQLAlchemy's declarative API
# reserves that attribute name for `Base.metadata` (the schema MetaData
# collection), and a column with that name raises InvalidRequestError at
# class-definition time. Flexible/free-form JSON columns are named
# `extra_data` instead.
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    ForeignKey, Text, JSON, Enum, Index, UniqueConstraint,
    BigInteger, Table
)
from sqlalchemy.orm import relationship, Mapped, mapped_column, declarative_base
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from pgvector.sqlalchemy import Vector
import uuid
import enum

Base = declarative_base()

EMBEDDING_DIMENSION = 1536  # matches text-embedding-3-small; keep in sync with settings.vector_dimension


class ProcessingStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    EXTRACTING_TEXT = "extracting_text"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    EXTRACTING_STRUCTURE = "extracting_structure"
    COMPLETED = "completed"
    FAILED = "failed"


class Exam(Base):
    __tablename__ = "exams"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    short_name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    subjects: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    # Embedding of name + short_name + description + aliases, used for semantic
    # exam resolution (Telegram free-text input -> exam). Recomputed whenever
    # the admin creates/edits the exam or its aliases.
    embedding: Mapped[Optional[Any]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    aliases = relationship("ExamAlias", back_populates="exam", cascade="all, delete-orphan")
    resources = relationship("Resource", back_populates="exam", cascade="all, delete-orphan")
    subjects_relation = relationship("Subject", back_populates="exam", cascade="all, delete-orphan")
    sample_papers = relationship("SamplePaper", back_populates="exam", cascade="all, delete-orphan")
    blueprints = relationship("ExamBlueprint", back_populates="exam", cascade="all, delete-orphan")
    quizzes = relationship("Quiz", back_populates="exam")
    
    __table_args__ = (
        Index("idx_exams_short_name", "short_name"),
        Index("idx_exams_is_active", "is_active"),
    )


class ExamAlias(Base):
    __tablename__ = "exam_aliases"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exam_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("exams.id", ondelete="CASCADE"))
    alias: Mapped[str] = mapped_column(String(100), nullable=False)
    is_semantic: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    exam = relationship("Exam", back_populates="aliases")
    
    __table_args__ = (
        Index("idx_exam_aliases_alias", "alias"),
        UniqueConstraint("exam_id", "alias", name="uq_exam_alias"),
    )


class Resource(Base):
    __tablename__ = "resources"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exam_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("exams.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger)
    mime_type: Mapped[str] = mapped_column(String(100))
    checksum: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[ProcessingStatus] = mapped_column(Enum(ProcessingStatus), default=ProcessingStatus.UPLOADED)
    status_message: Mapped[Optional[str]] = mapped_column(Text)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    exam = relationship("Exam", back_populates="resources")
    documents = relationship("Document", back_populates="resource", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_resources_exam_id", "exam_id"),
        Index("idx_resources_status", "status"),
        Index("idx_resources_checksum", "checksum"),
    )


class Document(Base):
    __tablename__ = "documents"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("resources.id", ondelete="CASCADE"))
    title: Mapped[Optional[str]] = mapped_column(String(255))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    resource = relationship("Resource", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    embedding: Mapped[Optional[Any]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=True)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    page_number: Mapped[Optional[int]] = mapped_column(Integer)

    # Denormalized curriculum/exam references, populated at chunking/structure-extraction
    # time, so retrieval can filter chunks by exam/subject/chapter/topic without joining
    # through document -> resource -> exam on every query (see spec section 18).
    exam_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=True), ForeignKey("exams.id", ondelete="CASCADE"), index=True)
    subject_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=True), ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True)
    chapter_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=True), ForeignKey("chapters.id", ondelete="SET NULL"), nullable=True)
    topic_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=True), ForeignKey("topics.id", ondelete="SET NULL"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunks_document_id", "document_id"),
        Index("idx_chunks_chunk_index", "chunk_index"),
        Index("idx_chunks_exam_id", "exam_id"),
        Index("idx_chunks_chapter_id", "chapter_id"),
        Index("idx_chunks_topic_id", "topic_id"),
    )


class Subject(Base):
    __tablename__ = "subjects"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exam_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("exams.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    order: Mapped[int] = mapped_column(Integer, default=0)
    source_references: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    exam = relationship("Exam", back_populates="subjects_relation")
    chapters = relationship("Chapter", back_populates="subject", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_subjects_exam_id", "exam_id"),
        Index("idx_subjects_name", "name"),
        UniqueConstraint("exam_id", "name", name="uq_subject_exam"),
    )


class Chapter(Base):
    __tablename__ = "chapters"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("subjects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    order: Mapped[int] = mapped_column(Integer, default=0)
    source_references: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    subject = relationship("Subject", back_populates="chapters")
    topics = relationship("Topic", back_populates="chapter", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_chapters_subject_id", "subject_id"),
        Index("idx_chapters_name", "name"),
        UniqueConstraint("subject_id", "name", name="uq_chapter_subject"),
    )


class Topic(Base):
    __tablename__ = "topics"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chapter_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("chapters.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    order: Mapped[int] = mapped_column(Integer, default=0)
    source_references: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    chapter = relationship("Chapter", back_populates="topics")
    
    __table_args__ = (
        Index("idx_topics_chapter_id", "chapter_id"),
        Index("idx_topics_name", "name"),
        UniqueConstraint("chapter_id", "name", name="uq_topic_chapter"),
    )


class SamplePaper(Base):
    __tablename__ = "sample_papers"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exam_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("exams.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger)
    mime_type: Mapped[str] = mapped_column(String(100))
    checksum: Mapped[Optional[str]] = mapped_column(String(64))
    year: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[ProcessingStatus] = mapped_column(Enum(ProcessingStatus), default=ProcessingStatus.UPLOADED)
    status_message: Mapped[Optional[str]] = mapped_column(Text)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    exam = relationship("Exam", back_populates="sample_papers")
    questions = relationship("SampleQuestion", back_populates="sample_paper", cascade="all, delete-orphan")


class SampleQuestion(Base):
    __tablename__ = "sample_questions"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sample_paper_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("sample_papers.id", ondelete="CASCADE"))
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[str] = mapped_column(String(50), nullable=False)
    options: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    correct_answer: Mapped[Optional[str]] = mapped_column(Text)
    difficulty: Mapped[Optional[str]] = mapped_column(String(20))
    marks: Mapped[Optional[float]] = mapped_column(Float)
    negative_marks: Mapped[Optional[float]] = mapped_column(Float)
    subject: Mapped[Optional[str]] = mapped_column(String(100))
    chapter: Mapped[Optional[str]] = mapped_column(String(100))
    topic: Mapped[Optional[str]] = mapped_column(String(100))
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    sample_paper = relationship("SamplePaper", back_populates="questions")


class ExamBlueprint(Base):
    __tablename__ = "exam_blueprints"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exam_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("exams.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer, default=1)
    blueprint_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    source_sample_papers: Mapped[List[str]] = mapped_column(ARRAY(UUID(as_uuid=True)))
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    exam = relationship("Exam", back_populates="blueprints")
    
    __table_args__ = (
        Index("idx_blueprints_exam_id", "exam_id"),
        Index("idx_blueprints_is_active", "is_active"),
    )


class User(Base):
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    telegram_user_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    username: Mapped[Optional[str]] = mapped_column(String(100))
    first_name: Mapped[Optional[str]] = mapped_column(String(100))
    last_name: Mapped[Optional[str]] = mapped_column(String(100))
    language_code: Mapped[Optional[str]] = mapped_column(String(10))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    total_quizzes: Mapped[int] = mapped_column(Integer, default=0)
    total_questions_answered: Mapped[int] = mapped_column(Integer, default=0)
    total_correct_answers: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    quizzes = relationship("Quiz", back_populates="user")
    
    __table_args__ = (
        Index("idx_users_telegram_user_id", "telegram_user_id"),
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    telegram_chat_id: Mapped[str] = mapped_column(String(100), nullable=False)
    current_step: Mapped[str] = mapped_column(String(50), default="start")
    state_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_message_id: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    
    __table_args__ = (
        Index("idx_chat_sessions_user_id", "user_id"),
        Index("idx_chat_sessions_telegram_chat_id", "telegram_chat_id"),
        Index("idx_chat_sessions_current_step", "current_step"),
        UniqueConstraint("user_id", "telegram_chat_id", name="uq_user_chat"),
    )


class Quiz(Base):
    __tablename__ = "quizzes"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    exam_id: Mapped[str] = mapped_column(UUID(as_uuid=True), ForeignKey("exams.id", ondelete="CASCADE"))
    blueprint_version: Mapped[int] = mapped_column(Integer)
    chapter_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=True), ForeignKey("chapters.id"))
    topic_id: Mapped[Optional[str]] = mapped_column(UUID(as_uuid=True), ForeignKey("topics.id"))
    question_count: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="generated")  # generated, started, submitted, evaluated, expired
    generated_questions: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)
    user_answers: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    evaluation: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    score: Mapped[Optional[float]] = mapped_column(Float)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    submitted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    time_taken_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="quizzes")
    exam = relationship("Exam", back_populates="quizzes")
    
    __table_args__ = (
        Index("idx_quizzes_user_id", "user_id"),
        Index("idx_quizzes_exam_id", "exam_id"),
        Index("idx_quizzes_status", "status"),
        Index("idx_quizzes_start_time", "start_time"),
    )


class SemanticCache(Base):
    __tablename__ = "semantic_cache"
    
    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    normalized_query: Mapped[str] = mapped_column(String(255), nullable=False)
    query_embedding: Mapped[Optional[Any]] = mapped_column(JSON)
    resolved_entity_type: Mapped[str] = mapped_column(String(50), nullable=False)  # exam, chapter, topic
    resolved_entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_semantic_cache_query_hash", "query_hash"),
        Index("idx_semantic_cache_entity_type", "resolved_entity_type"),
        Index("idx_semantic_cache_expires_at", "expires_at"),
    )