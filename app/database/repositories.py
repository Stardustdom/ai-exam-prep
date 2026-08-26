from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import text
import uuid
from app.database.models import (
    Exam, ExamAlias, Resource, Document, DocumentChunk,
    Subject, Chapter, Topic, SamplePaper, SampleQuestion,
    ExamBlueprint, User, ChatSession, Quiz, SemanticCache,
    ProcessingStatus, GroupSubscription
)

class BaseRepository:
    model = None

    def __init__(self, session: AsyncSession):
        self.session = session

    async def delete(self, entity_id: str) -> bool:
        """Generic delete-by-id, relying on the subclass's get_by_id and model's FK cascades"""
        entity = await self.get_by_id(entity_id)
        if not entity:
            return False
        await self.session.delete(entity)
        await self.session.commit()
        return True

class ExamRepository(BaseRepository):
    async def create(self, data: Dict[str, Any]) -> Exam:
        exam = Exam(**data)
        self.session.add(exam)
        await self.session.commit()
        await self.session.refresh(exam)
        return exam

    async def get_by_id(self, exam_id: str) -> Optional[Exam]:
        result = await self.session.execute(
            select(Exam).where(Exam.id == uuid.UUID(exam_id))
        )
        return result.scalar_one_or_none()

    async def get_by_short_name(self, short_name: str) -> Optional[Exam]:
        result = await self.session.execute(
            select(Exam).where(Exam.short_name == short_name.upper())
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Optional[Exam]:
        result = await self.session.execute(
            select(Exam).where(Exam.name.ilike(name))
        )
        return result.scalar_one_or_none()

    async def get_alias(self, alias: str) -> Optional[ExamAlias]:
        result = await self.session.execute(
            select(ExamAlias).where(ExamAlias.alias.ilike(alias))
        )
        return result.scalar_one_or_none()

    async def get_aliases_for_exam(self, exam_id: str) -> List[ExamAlias]:
        """Explicit query rather than the `exam.aliases` relationship — the latter
        lazy-loads, which raises MissingGreenlet if accessed on an object fetched
        without eager loading under the async ORM."""
        result = await self.session.execute(
            select(ExamAlias).where(ExamAlias.exam_id == uuid.UUID(exam_id))
        )
        return result.scalars().all()

    async def get_active_exams(self) -> List[Exam]:
        result = await self.session.execute(
            select(Exam).where(Exam.is_active == True)
        )
        return result.scalars().all()

    async def get_all(self) -> List[Exam]:
        result = await self.session.execute(select(Exam))
        return result.scalars().all()

    async def update(self, exam_id: str, data: Dict[str, Any]) -> Optional[Exam]:
        exam = await self.get_by_id(exam_id)
        if exam:
            for key, value in data.items():
                setattr(exam, key, value)
            await self.session.commit()
            await self.session.refresh(exam)
        return exam

    async def search_semantic(self, embedding: List[float], limit: int = 5) -> List[Dict]:
        # pgvector cosine-distance similarity search over active exams only.
        # Bound parameters throughout; embedding is passed as a pgvector literal,
        # never string-interpolated into the query.
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        query = text("""
            SELECT id, name, short_name, 1 - (embedding <=> CAST(:embedding AS vector)) as similarity
            FROM exams
            WHERE embedding IS NOT NULL AND is_active = true
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """)
        result = await self.session.execute(query, {"embedding": embedding_str, "limit": limit})
        return [dict(row._mapping) for row in result]

    async def update_embedding(self, exam_id: str, embedding: List[float]) -> None:
        exam = await self.get_by_id(exam_id)
        if exam:
            exam.embedding = embedding
            await self.session.commit()

class ResourceRepository(BaseRepository):
    async def create(self, data: Dict[str, Any]) -> Resource:
        resource = Resource(**data)
        self.session.add(resource)
        await self.session.commit()
        await self.session.refresh(resource)
        return resource

    async def get_by_id(self, resource_id: str) -> Optional[Resource]:
        result = await self.session.execute(
            select(Resource).where(Resource.id == uuid.UUID(resource_id))
        )
        return result.scalar_one_or_none()

    async def get_by_exam(self, exam_id: str) -> List[Resource]:
        result = await self.session.execute(
            select(Resource).where(Resource.exam_id == uuid.UUID(exam_id))
        )
        return result.scalars().all()

    async def get_all(self) -> List[Resource]:
        result = await self.session.execute(select(Resource))
        return result.scalars().all()

    async def update_status(self, resource_id: str, status: ProcessingStatus, message: str = None):
        resource = await self.get_by_id(resource_id)
        if resource:
            resource.status = status
            if message:
                resource.status_message = message
            await self.session.commit()

class DocumentRepository(BaseRepository):
    async def create(self, data: Dict[str, Any]) -> Document:
        document = Document(**data)
        self.session.add(document)
        await self.session.commit()
        await self.session.refresh(document)
        return document

class DocumentChunkRepository(BaseRepository):
    async def create(self, data: Dict[str, Any]) -> DocumentChunk:
        chunk = DocumentChunk(**data)
        self.session.add(chunk)
        await self.session.commit()
        await self.session.refresh(chunk)
        return chunk

    async def get_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        result = await self.session.execute(
            select(DocumentChunk).where(DocumentChunk.id == uuid.UUID(chunk_id))
        )
        return result.scalar_one_or_none()

    async def get_by_document(self, document_id: str) -> List[DocumentChunk]:
        result = await self.session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == uuid.UUID(document_id))
            .order_by(DocumentChunk.chunk_index)
        )
        return result.scalars().all()

class SubjectRepository(BaseRepository):
    async def create(self, data: Dict[str, Any]) -> Subject:
        subject = Subject(**data)
        self.session.add(subject)
        await self.session.commit()
        await self.session.refresh(subject)
        return subject

    async def get_by_id(self, subject_id: str) -> Optional[Subject]:
        result = await self.session.execute(
            select(Subject).where(Subject.id == uuid.UUID(subject_id))
        )
        return result.scalar_one_or_none()

    async def get_by_exam(self, exam_id: str) -> List[Subject]:
        result = await self.session.execute(
            select(Subject).where(Subject.exam_id == uuid.UUID(exam_id))
            .order_by(Subject.order)
        )
        return result.scalars().all()

    async def get_all(self) -> List[Subject]:
        result = await self.session.execute(select(Subject))
        return result.scalars().all()

    async def get_or_create(self, exam_id: str, name: str) -> Subject:
        result = await self.session.execute(
            select(Subject).where(
                and_(Subject.exam_id == uuid.UUID(exam_id), Subject.name.ilike(name))
            )
        )
        subject = result.scalar_one_or_none()
        if subject:
            return subject
        return await self.create({"exam_id": uuid.UUID(exam_id), "name": name})

class ChapterRepository(BaseRepository):
    async def create(self, data: Dict[str, Any]) -> Chapter:
        chapter = Chapter(**data)
        self.session.add(chapter)
        await self.session.commit()
        await self.session.refresh(chapter)
        return chapter

    async def get_by_id(self, chapter_id: str) -> Optional[Chapter]:
        result = await self.session.execute(
            select(Chapter).where(Chapter.id == uuid.UUID(chapter_id))
        )
        return result.scalar_one_or_none()

    async def get_by_subject(self, subject_id: str) -> List[Chapter]:
        result = await self.session.execute(
            select(Chapter).where(Chapter.subject_id == uuid.UUID(subject_id))
            .order_by(Chapter.order)
        )
        return result.scalars().all()

    async def get_by_exam(self, exam_id: str) -> List[Chapter]:
        result = await self.session.execute(
            select(Chapter)
            .join(Subject)
            .where(Subject.exam_id == uuid.UUID(exam_id))
            .order_by(Chapter.order)
            # eager-load: callers (e.g. curriculum chunk assignment) read
            # chapter.topics, which would otherwise lazy-load and raise
            # MissingGreenlet under the async ORM.
            .options(selectinload(Chapter.topics))
        )
        return result.scalars().all()

    async def get_or_create(self, subject_id: str, name: str) -> Chapter:
        result = await self.session.execute(
            select(Chapter).where(
                and_(Chapter.subject_id == uuid.UUID(subject_id), Chapter.name.ilike(name))
            )
        )
        chapter = result.scalar_one_or_none()
        if chapter:
            return chapter
        return await self.create({"subject_id": uuid.UUID(subject_id), "name": name})

    async def get_all(self) -> List[Chapter]:
        result = await self.session.execute(select(Chapter))
        return result.scalars().all()

class TopicRepository(BaseRepository):
    async def create(self, data: Dict[str, Any]) -> Topic:
        topic = Topic(**data)
        self.session.add(topic)
        await self.session.commit()
        await self.session.refresh(topic)
        return topic

    async def get_by_id(self, topic_id: str) -> Optional[Topic]:
        result = await self.session.execute(
            select(Topic).where(Topic.id == uuid.UUID(topic_id))
        )
        return result.scalar_one_or_none()

    async def get_by_chapter(self, chapter_id: str) -> List[Topic]:
        result = await self.session.execute(
            select(Topic).where(Topic.chapter_id == uuid.UUID(chapter_id))
            .order_by(Topic.order)
        )
        return result.scalars().all()

    async def get_by_exam(self, exam_id: str) -> List[Topic]:
        result = await self.session.execute(
            select(Topic)
            .join(Chapter, Topic.chapter_id == Chapter.id)
            .join(Subject, Chapter.subject_id == Subject.id)
            .where(Subject.exam_id == uuid.UUID(exam_id))
            .order_by(Topic.order)
        )
        return result.scalars().all()

    async def get_or_create(self, chapter_id: str, name: str) -> Topic:
        result = await self.session.execute(
            select(Topic).where(
                and_(Topic.chapter_id == uuid.UUID(chapter_id), Topic.name.ilike(name))
            )
        )
        topic = result.scalar_one_or_none()
        if topic:
            return topic
        return await self.create({"chapter_id": uuid.UUID(chapter_id), "name": name})

    async def get_all(self) -> List[Topic]:
        result = await self.session.execute(select(Topic))
        return result.scalars().all()

class SamplePaperRepository(BaseRepository):
    async def create(self, data: Dict[str, Any]) -> SamplePaper:
        paper = SamplePaper(**data)
        self.session.add(paper)
        await self.session.commit()
        await self.session.refresh(paper)
        return paper

    async def get_by_id(self, paper_id: str) -> Optional[SamplePaper]:
        result = await self.session.execute(
            select(SamplePaper).where(SamplePaper.id == uuid.UUID(paper_id))
        )
        return result.scalar_one_or_none()

    async def get_by_exam(self, exam_id: str) -> List[SamplePaper]:
        result = await self.session.execute(
            select(SamplePaper).where(SamplePaper.exam_id == uuid.UUID(exam_id))
        )
        return result.scalars().all()

    async def update_status(self, paper_id: str, status: ProcessingStatus, message: str = None):
        paper = await self.get_by_id(paper_id)
        if paper:
            paper.status = status
            if message:
                paper.status_message = message
            await self.session.commit()

    async def get_all(self) -> List[SamplePaper]:
        result = await self.session.execute(select(SamplePaper))
        return result.scalars().all()

class SampleQuestionRepository(BaseRepository):
    async def create(self, data: Dict[str, Any]) -> SampleQuestion:
        question = SampleQuestion(**data)
        self.session.add(question)
        await self.session.commit()
        await self.session.refresh(question)
        return question

    async def get_by_exam(self, exam_id: str) -> List[SampleQuestion]:
        result = await self.session.execute(
            select(SampleQuestion)
            .join(SamplePaper, SampleQuestion.sample_paper_id == SamplePaper.id)
            .where(SamplePaper.exam_id == uuid.UUID(exam_id))
        )
        return result.scalars().all()

class BlueprintRepository(BaseRepository):
    async def create(self, data: Dict[str, Any]) -> ExamBlueprint:
        blueprint = ExamBlueprint(**data)
        self.session.add(blueprint)
        await self.session.commit()
        await self.session.refresh(blueprint)
        return blueprint

    async def get_by_id(self, blueprint_id: str) -> Optional[ExamBlueprint]:
        result = await self.session.execute(
            select(ExamBlueprint).where(ExamBlueprint.id == uuid.UUID(blueprint_id))
        )
        return result.scalar_one_or_none()

    async def get_active_by_exam(self, exam_id: str) -> Optional[ExamBlueprint]:
        result = await self.session.execute(
            select(ExamBlueprint)
            .where(
                and_(
                    ExamBlueprint.exam_id == uuid.UUID(exam_id),
                    ExamBlueprint.is_active == True
                )
            )
            .order_by(ExamBlueprint.version.desc())
        )
        return result.scalar_one_or_none()

class UserRepository(BaseRepository):
    async def get_or_create(self, telegram_user_id: str, **kwargs) -> User:
        result = await self.session.execute(
            select(User).where(User.telegram_user_id == telegram_user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            user = User(
                telegram_user_id=telegram_user_id,
                username=kwargs.get('username'),
                first_name=kwargs.get('first_name'),
                last_name=kwargs.get('last_name'),
                language_code=kwargs.get('language_code')
            )
            self.session.add(user)
            await self.session.commit()
            await self.session.refresh(user)

        return user

    async def get_all(self) -> List[User]:
        result = await self.session.execute(select(User))
        return result.scalars().all()

class ChatSessionRepository(BaseRepository):
    async def get_or_create(self, user_id: str, telegram_chat_id: str) -> ChatSession:
        # str() first: tolerates callers passing an already-UUID value (e.g. a
        # model's .id straight from asyncpg) as well as a plain string, since
        # uuid.UUID() only accepts str for its hex argument.
        user_uuid = uuid.UUID(str(user_id))
        result = await self.session.execute(
            select(ChatSession).where(
                and_(
                    ChatSession.user_id == user_uuid,
                    ChatSession.telegram_chat_id == telegram_chat_id
                )
            )
        )
        session = result.scalar_one_or_none()

        if not session:
            session = ChatSession(
                user_id=user_uuid,
                telegram_chat_id=telegram_chat_id,
                current_step="start"
            )
            self.session.add(session)
            await self.session.commit()
            await self.session.refresh(session)
        
        return session

    async def get_by_id(self, session_id: str) -> Optional[ChatSession]:
        result = await self.session.execute(
            select(ChatSession).where(ChatSession.id == uuid.UUID(str(session_id)))
        )
        return result.scalar_one_or_none()

    async def update_state(self, session_id: str, current_step: str, state_data: Dict):
        result = await self.session.execute(
            select(ChatSession).where(ChatSession.id == uuid.UUID(session_id))
        )
        session = result.scalar_one_or_none()
        if session:
            session.current_step = current_step
            session.state_data = state_data
            await self.session.commit()

    async def get_by_user(self, user_id: str) -> Optional[ChatSession]:
        result = await self.session.execute(
            select(ChatSession).where(ChatSession.user_id == uuid.UUID(user_id))
        )
        return result.scalar_one_or_none()

class QuizRepository(BaseRepository):
    async def create(self, data: Dict[str, Any]) -> Quiz:
        quiz = Quiz(**data)
        self.session.add(quiz)
        await self.session.commit()
        await self.session.refresh(quiz)
        return quiz

    async def get_by_id(self, quiz_id: str) -> Optional[Quiz]:
        result = await self.session.execute(
            select(Quiz).where(Quiz.id == uuid.UUID(quiz_id))
        )
        return result.scalar_one_or_none()

    async def get_by_user(self, user_id: str) -> List[Quiz]:
        result = await self.session.execute(
            select(Quiz).where(Quiz.user_id == uuid.UUID(user_id))
            .order_by(Quiz.created_at.desc())
        )
        return result.scalars().all()

    async def get_all(self) -> List[Quiz]:
        result = await self.session.execute(select(Quiz))
        return result.scalars().all()

    async def get_by_status(self, status_value: str) -> List[Quiz]:
        result = await self.session.execute(select(Quiz).where(Quiz.status == status_value))
        return result.scalars().all()

    async def update(self, quiz_id: str, data: Dict[str, Any]) -> Optional[Quiz]:
        quiz = await self.get_by_id(quiz_id)
        if quiz:
            for key, value in data.items():
                setattr(quiz, key, value)
            await self.session.commit()
            await self.session.refresh(quiz)
        return quiz

class SemanticCacheRepository(BaseRepository):
    async def get_by_query_hash(self, query_hash: str) -> Optional[SemanticCache]:
        result = await self.session.execute(
            select(SemanticCache).where(SemanticCache.query_hash == query_hash)
        )
        return result.scalar_one_or_none()

    async def create(self, data: Dict[str, Any]) -> SemanticCache:
        cache = SemanticCache(**data)
        self.session.add(cache)
        await self.session.commit()
        await self.session.refresh(cache)
        return cache

    async def delete_by_entity(self, entity_type: str, entity_id: str) -> None:
        """Delete all cache rows resolving to a given entity (invalidation on admin edit/delete)."""
        result = await self.session.execute(
            select(SemanticCache).where(
                and_(
                    SemanticCache.resolved_entity_type == entity_type,
                    SemanticCache.resolved_entity_id == entity_id
                )
            )
        )
        for row in result.scalars().all():
            await self.session.delete(row)
        await self.session.commit()


class GroupSubscriptionRepository(BaseRepository):
    async def get_by_telegram_group_id(self, telegram_group_id: str) -> Optional[GroupSubscription]:
        result = await self.session.execute(
            select(GroupSubscription).where(GroupSubscription.telegram_group_id == str(telegram_group_id))
        )
        return result.scalar_one_or_none()

    async def get_or_create(self, telegram_group_id: str, **defaults) -> GroupSubscription:
        """FR-1.3: if the bot is removed and re-added, resume the existing
        row (re-activating it) rather than re-onboarding from scratch."""
        existing = await self.get_by_telegram_group_id(telegram_group_id)
        if existing:
            if not existing.is_active:
                existing.is_active = True
                await self.session.commit()
                await self.session.refresh(existing)
            return existing

        group = GroupSubscription(
            telegram_group_id=str(telegram_group_id),
            send_times=defaults.get("send_times", ["09:00", "18:00"]),
            content_types_enabled=defaults.get("content_types_enabled", ["notes", "daily10", "popquiz", "flashcards"]),
            rate_limit_per_day=defaults.get("rate_limit_per_day", 2),
        )
        self.session.add(group)
        await self.session.commit()
        await self.session.refresh(group)
        return group

    async def deactivate(self, telegram_group_id: str) -> None:
        """Bot removed from the group — stop posting, but keep the row (and
        its history) so FR-1.3's resume-not-reonboard works if re-added."""
        group = await self.get_by_telegram_group_id(telegram_group_id)
        if group:
            group.is_active = False
            await self.session.commit()

    async def get_active(self) -> List[GroupSubscription]:
        result = await self.session.execute(
            select(GroupSubscription).where(GroupSubscription.is_active == True)
        )
        return result.scalars().all()

    async def set_exam(self, telegram_group_id: str, exam_id: str) -> Optional[GroupSubscription]:
        group = await self.get_by_telegram_group_id(telegram_group_id)
        if group:
            group.exam_id = uuid.UUID(str(exam_id))
            await self.session.commit()
            await self.session.refresh(group)
        return group