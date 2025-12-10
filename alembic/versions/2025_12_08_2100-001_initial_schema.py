"""Initial schema with sources and messages tables

Revision ID: 001
Revises:
Create Date: 2025-12-08 21:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial sources and messages tables."""
    # Create sources table
    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_sources_id"), "sources", ["id"], unique=False)
    op.create_index(op.f("ix_sources_type"), "sources", ["type"], unique=False)

    # Create messages table
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("author", sa.String(length=255), nullable=True),
        sa.Column("channel", sa.String(length=255), nullable=True),
        sa.Column("thread_id", sa.String(length=255), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("message_metadata", sa.JSON(), nullable=True),
        sa.Column("embedding_id", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["sources.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_messages_id"), "messages", ["id"], unique=False)
    op.create_index(op.f("ix_messages_source_id"), "messages", ["source_id"], unique=False)
    op.create_index(op.f("ix_messages_external_id"), "messages", ["external_id"], unique=False)
    op.create_index(op.f("ix_messages_author"), "messages", ["author"], unique=False)
    op.create_index(op.f("ix_messages_channel"), "messages", ["channel"], unique=False)
    op.create_index(op.f("ix_messages_thread_id"), "messages", ["thread_id"], unique=False)
    op.create_index(op.f("ix_messages_timestamp"), "messages", ["timestamp"], unique=False)
    op.create_index(op.f("ix_messages_embedding_id"), "messages", ["embedding_id"], unique=False)


def downgrade() -> None:
    """Drop messages and sources tables."""
    op.drop_index(op.f("ix_messages_embedding_id"), table_name="messages")
    op.drop_index(op.f("ix_messages_timestamp"), table_name="messages")
    op.drop_index(op.f("ix_messages_thread_id"), table_name="messages")
    op.drop_index(op.f("ix_messages_channel"), table_name="messages")
    op.drop_index(op.f("ix_messages_author"), table_name="messages")
    op.drop_index(op.f("ix_messages_external_id"), table_name="messages")
    op.drop_index(op.f("ix_messages_source_id"), table_name="messages")
    op.drop_index(op.f("ix_messages_id"), table_name="messages")
    op.drop_table("messages")
    op.drop_index(op.f("ix_sources_type"), table_name="sources")
    op.drop_index(op.f("ix_sources_id"), table_name="sources")
    op.drop_table("sources")
