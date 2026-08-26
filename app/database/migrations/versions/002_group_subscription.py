"""Add group_subscriptions (Gen-Z engagement layer, FR-1)

Revision ID: 002
Revises: 001
Create Date: 2026-08-25 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid

revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'group_subscriptions',
        sa.Column('id', UUID, primary_key=True, default=uuid.uuid4),
        sa.Column('telegram_group_id', sa.String(100), nullable=False, unique=True),
        sa.Column('exam_id', UUID, sa.ForeignKey('exams.id', ondelete='SET NULL'), nullable=True),
        sa.Column('subjects_enabled', ARRAY(sa.String)),
        sa.Column('send_times', ARRAY(sa.String), nullable=False, server_default='{}'),
        sa.Column('timezone', sa.String(50), nullable=False, server_default='Asia/Kolkata'),
        sa.Column('content_types_enabled', ARRAY(sa.String), nullable=False, server_default='{}'),
        sa.Column('rate_limit_per_day', sa.Integer, nullable=False, server_default='2'),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column('added_via_referral_group_id', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now())
    )
    op.create_index('idx_group_subscriptions_telegram_group_id', 'group_subscriptions', ['telegram_group_id'])
    op.create_index('idx_group_subscriptions_is_active', 'group_subscriptions', ['is_active'])


def downgrade() -> None:
    op.drop_index('idx_group_subscriptions_is_active', table_name='group_subscriptions')
    op.drop_index('idx_group_subscriptions_telegram_group_id', table_name='group_subscriptions')
    op.drop_table('group_subscriptions')
