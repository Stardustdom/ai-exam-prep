"""Add name_embeddings (persisted chapter/topic name embeddings)

Revision ID: 004
Revises: 003
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
import uuid

revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'name_embeddings',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('exam_id', UUID, sa.ForeignKey('exams.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('model', sa.String(100), nullable=False),
        sa.Column('embedding', sa.JSON, nullable=False),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.UniqueConstraint('exam_id', 'name', 'model', name='uq_name_embedding'),
    )
    op.create_index('idx_name_embeddings_exam_id', 'name_embeddings', ['exam_id'])


def downgrade() -> None:
    op.drop_index('idx_name_embeddings_exam_id', table_name='name_embeddings')
    op.drop_table('name_embeddings')
