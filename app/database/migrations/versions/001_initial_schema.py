"""Initial schema

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON, ARRAY
from pgvector.sqlalchemy import Vector
import uuid

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIMENSION = 1536


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # `processingstatus` is shared by two tables below (resources, sample_papers).
    # No explicit CREATE TYPE here — the first column definition that uses it
    # (resources.status, `create_type` left at its default) creates it; the
    # second (sample_papers.status) passes create_type=False to reuse it
    # rather than attempt to create it again. (A prior version of this
    # migration issued a redundant explicit `CREATE TYPE` here AND left
    # create_type=False on the first column too, which — confirmed against a
    # real fresh Postgres, not just local dev's create_all() fallback which
    # never exercises this path — raises DuplicateObject instead of being a
    # harmless no-op.)

    op.create_table(
        'users',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('telegram_user_id', sa.String(100), nullable=False, unique=True),
        sa.Column('username', sa.String(100)),
        sa.Column('first_name', sa.String(100)),
        sa.Column('last_name', sa.String(100)),
        sa.Column('language_code', sa.String(10)),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('total_quizzes', sa.Integer, default=0),
        sa.Column('total_questions_answered', sa.Integer, default=0),
        sa.Column('total_correct_answers', sa.Integer, default=0),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now())
    )

    op.create_table(
        'exams',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('short_name', sa.String(50), nullable=False, unique=True),
        sa.Column('description', sa.Text),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('subjects', ARRAY(sa.String)),
        sa.Column('extra_data', JSON),
        sa.Column('embedding', Vector(EMBEDDING_DIMENSION), nullable=True),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now())
    )

    op.create_table(
        'exam_aliases',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('exam_id', UUID, sa.ForeignKey('exams.id', ondelete='CASCADE')),
        sa.Column('alias', sa.String(100), nullable=False),
        sa.Column('is_semantic', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.UniqueConstraint('exam_id', 'alias', name='uq_exam_alias')
    )

    op.create_table(
        'resources',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('exam_id', UUID, sa.ForeignKey('exams.id', ondelete='CASCADE')),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('file_path', sa.String(512), nullable=False),
        sa.Column('file_size', sa.BigInteger),
        sa.Column('mime_type', sa.String(100)),
        sa.Column('checksum', sa.String(64)),
        sa.Column('version', sa.Integer, default=1),
        sa.Column('status', sa.Enum('uploaded', 'queued', 'processing', 'extracting_text', 'chunking', 'embedding', 'extracting_structure', 'completed', 'failed', name='processingstatus')),
        sa.Column('status_message', sa.Text),
        sa.Column('extra_data', JSON),
        sa.Column('processed_at', sa.DateTime),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now())
    )

    op.create_table(
        'subjects',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('exam_id', UUID, sa.ForeignKey('exams.id', ondelete='CASCADE')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('order', sa.Integer, default=0),
        sa.Column('source_references', JSON),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint('exam_id', 'name', name='uq_subject_exam')
    )

    op.create_table(
        'chapters',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('subject_id', UUID, sa.ForeignKey('subjects.id', ondelete='CASCADE')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('order', sa.Integer, default=0),
        sa.Column('source_references', JSON),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint('subject_id', 'name', name='uq_chapter_subject')
    )

    op.create_table(
        'topics',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('chapter_id', UUID, sa.ForeignKey('chapters.id', ondelete='CASCADE')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('order', sa.Integer, default=0),
        sa.Column('source_references', JSON),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint('chapter_id', 'name', name='uq_topic_chapter')
    )

    op.create_table(
        'documents',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('resource_id', UUID, sa.ForeignKey('resources.id', ondelete='CASCADE')),
        sa.Column('title', sa.String(255)),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('page_count', sa.Integer),
        sa.Column('extra_data', JSON),
        sa.Column('created_at', sa.DateTime, default=sa.func.now())
    )

    op.create_table(
        'document_chunks',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('document_id', UUID, sa.ForeignKey('documents.id', ondelete='CASCADE')),
        sa.Column('chunk_index', sa.Integer, nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('token_count', sa.Integer),
        sa.Column('embedding', Vector(EMBEDDING_DIMENSION), nullable=True),
        sa.Column('extra_data', JSON),
        sa.Column('page_number', sa.Integer),
        sa.Column('exam_id', UUID, sa.ForeignKey('exams.id', ondelete='CASCADE')),
        sa.Column('subject_id', UUID, sa.ForeignKey('subjects.id', ondelete='SET NULL'), nullable=True),
        sa.Column('chapter_id', UUID, sa.ForeignKey('chapters.id', ondelete='SET NULL'), nullable=True),
        sa.Column('topic_id', UUID, sa.ForeignKey('topics.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime, default=sa.func.now())
    )

    op.create_table(
        'sample_papers',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('exam_id', UUID, sa.ForeignKey('exams.id', ondelete='CASCADE')),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('file_path', sa.String(512), nullable=False),
        sa.Column('file_size', sa.BigInteger),
        sa.Column('mime_type', sa.String(100)),
        sa.Column('checksum', sa.String(64)),
        sa.Column('year', sa.Integer),
        sa.Column('status', sa.Enum('uploaded', 'queued', 'processing', 'extracting_text', 'chunking', 'embedding', 'extracting_structure', 'completed', 'failed', name='processingstatus', create_type=False)),
        sa.Column('status_message', sa.Text),
        sa.Column('extra_data', JSON),
        sa.Column('processed_at', sa.DateTime),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now())
    )

    op.create_table(
        'sample_questions',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('sample_paper_id', UUID, sa.ForeignKey('sample_papers.id', ondelete='CASCADE')),
        sa.Column('question_text', sa.Text, nullable=False),
        sa.Column('question_type', sa.String(50), nullable=False),
        sa.Column('options', ARRAY(sa.Text)),
        sa.Column('correct_answer', sa.Text),
        sa.Column('difficulty', sa.String(20)),
        sa.Column('marks', sa.Float),
        sa.Column('negative_marks', sa.Float),
        sa.Column('subject', sa.String(100)),
        sa.Column('chapter', sa.String(100)),
        sa.Column('topic', sa.String(100)),
        sa.Column('extra_data', JSON),
        sa.Column('created_at', sa.DateTime, default=sa.func.now())
    )

    op.create_table(
        'exam_blueprints',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('exam_id', UUID, sa.ForeignKey('exams.id', ondelete='CASCADE')),
        sa.Column('version', sa.Integer, default=1),
        sa.Column('blueprint_data', JSON, nullable=False),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('source_sample_papers', ARRAY(UUID)),
        sa.Column('generated_at', sa.DateTime, default=sa.func.now()),
        sa.Column('created_at', sa.DateTime, default=sa.func.now())
    )

    op.create_table(
        'chat_sessions',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', UUID, sa.ForeignKey('users.id', ondelete='CASCADE')),
        sa.Column('telegram_chat_id', sa.String(100), nullable=False),
        sa.Column('current_step', sa.String(50), default='start'),
        sa.Column('state_data', JSON),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('last_message_id', sa.String(100)),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint('user_id', 'telegram_chat_id', name='uq_user_chat')
    )

    op.create_table(
        'quizzes',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', UUID, sa.ForeignKey('users.id', ondelete='CASCADE')),
        sa.Column('exam_id', UUID, sa.ForeignKey('exams.id', ondelete='CASCADE')),
        sa.Column('blueprint_version', sa.Integer),
        sa.Column('chapter_id', UUID, sa.ForeignKey('chapters.id')),
        sa.Column('topic_id', UUID, sa.ForeignKey('topics.id')),
        sa.Column('question_count', sa.Integer, nullable=False),
        sa.Column('duration_minutes', sa.Integer, nullable=False),
        sa.Column('status', sa.String(20), default='generated'),
        sa.Column('generated_questions', JSON, nullable=False),
        sa.Column('user_answers', JSON),
        sa.Column('evaluation', JSON),
        sa.Column('score', sa.Float),
        sa.Column('start_time', sa.DateTime),
        sa.Column('end_time', sa.DateTime),
        sa.Column('submitted_at', sa.DateTime),
        sa.Column('time_taken_seconds', sa.Integer),
        sa.Column('extra_data', JSON),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now())
    )

    op.create_table(
        'semantic_cache',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('query_hash', sa.String(64), unique=True, nullable=False),
        sa.Column('normalized_query', sa.String(255), nullable=False),
        sa.Column('query_embedding', JSON),
        sa.Column('resolved_entity_type', sa.String(50), nullable=False),
        sa.Column('resolved_entity_id', sa.String(100), nullable=False),
        sa.Column('confidence', sa.Float, nullable=False),
        sa.Column('extra_data', JSON),
        sa.Column('expires_at', sa.DateTime),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now())
    )

    op.create_index('idx_exams_short_name', 'exams', ['short_name'])
    op.create_index('idx_exams_is_active', 'exams', ['is_active'])
    op.create_index('idx_exam_aliases_alias', 'exam_aliases', ['alias'])
    op.create_index('idx_resources_exam_id', 'resources', ['exam_id'])
    op.create_index('idx_resources_status', 'resources', ['status'])
    op.create_index('idx_resources_checksum', 'resources', ['checksum'])
    op.create_index('idx_chunks_document_id', 'document_chunks', ['document_id'])
    op.create_index('idx_chunks_chunk_index', 'document_chunks', ['chunk_index'])
    op.create_index('idx_chunks_exam_id', 'document_chunks', ['exam_id'])
    op.create_index('idx_chunks_chapter_id', 'document_chunks', ['chapter_id'])
    op.create_index('idx_chunks_topic_id', 'document_chunks', ['topic_id'])
    op.create_index('idx_subjects_exam_id', 'subjects', ['exam_id'])
    op.create_index('idx_subjects_name', 'subjects', ['name'])
    op.create_index('idx_chapters_subject_id', 'chapters', ['subject_id'])
    op.create_index('idx_chapters_name', 'chapters', ['name'])
    op.create_index('idx_topics_chapter_id', 'topics', ['chapter_id'])
    op.create_index('idx_topics_name', 'topics', ['name'])
    op.create_index('idx_blueprints_exam_id', 'exam_blueprints', ['exam_id'])
    op.create_index('idx_blueprints_is_active', 'exam_blueprints', ['is_active'])
    op.create_index('idx_users_telegram_user_id', 'users', ['telegram_user_id'])
    op.create_index('idx_chat_sessions_user_id', 'chat_sessions', ['user_id'])
    op.create_index('idx_chat_sessions_telegram_chat_id', 'chat_sessions', ['telegram_chat_id'])
    op.create_index('idx_chat_sessions_current_step', 'chat_sessions', ['current_step'])
    op.create_index('idx_quizzes_user_id', 'quizzes', ['user_id'])
    op.create_index('idx_quizzes_exam_id', 'quizzes', ['exam_id'])
    op.create_index('idx_quizzes_status', 'quizzes', ['status'])
    op.create_index('idx_quizzes_start_time', 'quizzes', ['start_time'])
    op.create_index('idx_semantic_cache_query_hash', 'semantic_cache', ['query_hash'])
    op.create_index('idx_semantic_cache_entity_type', 'semantic_cache', ['resolved_entity_type'])
    op.create_index('idx_semantic_cache_expires_at', 'semantic_cache', ['expires_at'])

    # Approximate nearest-neighbor indexes for vector similarity search (spec section 18)
    op.execute("CREATE INDEX idx_exams_embedding ON exams USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
    op.execute("CREATE INDEX idx_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")


def downgrade() -> None:
    op.drop_table('semantic_cache')
    op.drop_table('quizzes')
    op.drop_table('chat_sessions')
    op.drop_table('exam_blueprints')
    op.drop_table('sample_questions')
    op.drop_table('sample_papers')
    op.drop_table('document_chunks')
    op.drop_table('documents')
    op.drop_table('topics')
    op.drop_table('chapters')
    op.drop_table('subjects')
    op.drop_table('resources')
    op.drop_table('exam_aliases')
    op.drop_table('exams')
    op.drop_table('users')
    op.execute("DROP TYPE processingstatus")
