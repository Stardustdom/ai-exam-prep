"""Add pop_quizzes + pop_quiz_answers (Gen-Z engagement layer, FR-4)

Revision ID: 003
Revises: 002
Create Date: 2026-08-26 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid

revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'pop_quizzes',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('telegram_group_id', sa.String(100), nullable=False),
        sa.Column('telegram_message_id', sa.String(100), nullable=True),
        sa.Column('exam_id', UUID, sa.ForeignKey('exams.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chapter_name', sa.String(255), nullable=True),
        sa.Column('question_text', sa.Text, nullable=False),
        sa.Column('options', ARRAY(sa.Text), nullable=False),
        sa.Column('correct_answer', sa.Text, nullable=False),
        sa.Column('explanation', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, default=sa.func.now())
    )
    op.create_index('idx_pop_quizzes_telegram_group_id', 'pop_quizzes', ['telegram_group_id'])

    op.create_table(
        'pop_quiz_answers',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('pop_quiz_id', UUID, sa.ForeignKey('pop_quizzes.id', ondelete='CASCADE'), nullable=False),
        sa.Column('telegram_user_id', sa.String(100), nullable=False),
        sa.Column('answer', sa.Text, nullable=False),
        sa.Column('is_correct', sa.Boolean, nullable=False),
        sa.Column('answered_at', sa.DateTime, default=sa.func.now()),
        sa.UniqueConstraint('pop_quiz_id', 'telegram_user_id', name='uq_pop_quiz_answer_once')
    )
    op.create_index('idx_pop_quiz_answers_pop_quiz_id', 'pop_quiz_answers', ['pop_quiz_id'])


def downgrade() -> None:
    op.drop_index('idx_pop_quiz_answers_pop_quiz_id', table_name='pop_quiz_answers')
    op.drop_table('pop_quiz_answers')
    op.drop_index('idx_pop_quizzes_telegram_group_id', table_name='pop_quizzes')
    op.drop_table('pop_quizzes')
