# app/bot/dependencies.py
#
# Builds one fully-wired set of repositories/services/agents/graph per
# Telegram update, sharing a DB session for the duration of that update and
# the process-wide LangGraph checkpointer (which is what actually persists
# conversation state between updates).
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.repositories import (
    ExamRepository, SemanticCacheRepository, ChapterRepository, TopicRepository,
    DocumentChunkRepository, QuizRepository, UserRepository, ChatSessionRepository,
    BlueprintRepository, GroupSubscriptionRepository
)
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.semantic_cache import SemanticCacheService
from app.services.vector_store import VectorStoreService
from app.services.telegram import TelegramService
from app.agents.session_manager import SessionManagerAgent
from app.agents.exam_resolver import ExamResolverAgent
from app.agents.curriculum_resolver import CurriculumResolverAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.question_generator import QuestionGeneratorAgent
from app.agents.quiz_manager import QuizManagerAgent
from app.agents.evaluation import EvaluationAgent
from app.graph.exam_graph import ExamGraph
from app.graph.checkpointer import get_checkpointer


@dataclass
class BotContext:
    db: AsyncSession
    exam_repo: ExamRepository
    chapter_repo: ChapterRepository
    topic_repo: TopicRepository
    user_repo: UserRepository
    chat_session_repo: ChatSessionRepository
    quiz_repo: QuizRepository
    group_repo: GroupSubscriptionRepository
    telegram_service: TelegramService
    session_manager: SessionManagerAgent
    exam_resolver: ExamResolverAgent
    curriculum_resolver: CurriculumResolverAgent
    graph: ExamGraph


def build_context(db: AsyncSession) -> BotContext:
    exam_repo = ExamRepository(db)
    chapter_repo = ChapterRepository(db)
    topic_repo = TopicRepository(db)
    user_repo = UserRepository(db)
    chat_session_repo = ChatSessionRepository(db)
    chunk_repo = DocumentChunkRepository(db)
    quiz_repo = QuizRepository(db)
    cache_repo = SemanticCacheRepository(db)
    blueprint_repo = BlueprintRepository(db)
    group_repo = GroupSubscriptionRepository(db)

    embedding_service = EmbeddingService()
    llm_service = LLMService()
    telegram_service = TelegramService()
    vector_store = VectorStoreService()
    cache_service = SemanticCacheService(cache_repo, embedding_service)

    from app.services.retrieval import RetrievalService

    session_manager = SessionManagerAgent(user_repo, chat_session_repo, telegram_service)
    exam_resolver = ExamResolverAgent(exam_repo, cache_repo, embedding_service, cache_service, llm_service)
    curriculum_resolver = CurriculumResolverAgent(chapter_repo, topic_repo, embedding_service, cache_service, llm_service)
    retrieval_agent = RetrievalAgent(chunk_repo, vector_store, embedding_service)
    question_generator = QuestionGeneratorAgent(llm_service, embedding_service, quiz_repo)
    quiz_manager = QuizManagerAgent(quiz_repo, telegram_service)
    evaluation_agent = EvaluationAgent(
        quiz_repo, llm_service, RetrievalService(vector_store, embedding_service), blueprint_repo
    )

    graph = ExamGraph(
        session_manager=session_manager,
        exam_resolver=exam_resolver,
        curriculum_resolver=curriculum_resolver,
        retrieval_agent=retrieval_agent,
        question_generator=question_generator,
        quiz_manager=quiz_manager,
        evaluation_agent=evaluation_agent,
        checkpointer=get_checkpointer()
    )

    return BotContext(
        db=db, exam_repo=exam_repo, chapter_repo=chapter_repo, topic_repo=topic_repo,
        user_repo=user_repo, chat_session_repo=chat_session_repo, quiz_repo=quiz_repo,
        group_repo=group_repo,
        telegram_service=telegram_service,
        session_manager=session_manager, exam_resolver=exam_resolver,
        curriculum_resolver=curriculum_resolver, graph=graph
    )
