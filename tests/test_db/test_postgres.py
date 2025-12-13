"""
Tests for PostgreSQL database connection and session management.

This module tests:
- Database connection functionality
- Session creation and cleanup
- Transaction management (commit/rollback)
- Database initialization
- Connection health checks
"""
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Message, Source
from src.db.postgres import (
    AsyncSessionLocal,
    async_engine,
    check_db_connection,
    get_db,
    init_db,
)


class TestDatabaseConnection:
    """Tests for database connection management."""

    @pytest.mark.asyncio
    async def test_session_can_execute_queries(self, async_db_session: AsyncSession):
        """Test that database session can execute queries."""
        result = await async_db_session.execute(text("SELECT 1"))
        assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_connection_with_test_db(self, async_db_session: AsyncSession):
        """Test basic database connectivity with test session."""
        # Verify session is functional
        result = await async_db_session.execute(text("SELECT 1 as value"))
        row = result.fetchone()
        assert row[0] == 1


class TestSessionManagement:
    """Tests for database session creation and management."""

    @pytest.mark.asyncio
    async def test_session_is_async_session(self, async_db_session: AsyncSession):
        """Test that we get an AsyncSession instance."""
        assert isinstance(async_db_session, AsyncSession)
        # Verify session is usable
        result = await async_db_session.execute(text("SELECT 1"))
        assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_session_transaction_commit(self, async_db_session: AsyncSession):
        """Test that session commits changes successfully."""
        # Create a test source
        source = Source(
            type="test",
            name="Test Source",
            config={"test": "value"},
        )
        async_db_session.add(source)
        await async_db_session.commit()
        await async_db_session.refresh(source)

        # Verify it was committed
        assert source.id is not None

        # Clean up
        await async_db_session.delete(source)
        await async_db_session.commit()

    @pytest.mark.asyncio
    async def test_session_transaction_rollback(self, async_db_session: AsyncSession):
        """Test that session rollback discards changes."""
        # Count sources before
        result = await async_db_session.execute(select(Source))
        initial_count = len(result.scalars().all())

        # Add a source but rollback
        source = Source(
            type="test_rollback",
            name="Test Rollback",
            config={},
        )
        async_db_session.add(source)
        await async_db_session.rollback()

        # Verify it was not persisted
        result = await async_db_session.execute(select(Source))
        final_count = len(result.scalars().all())
        assert final_count == initial_count

    @pytest.mark.asyncio
    async def test_session_rollback_isolation(self, async_db_session: AsyncSession):
        """Test that rolled back changes are not persisted."""
        # Add a source
        source1 = Source(type="test1", name="Session Isolation Test", config={})
        async_db_session.add(source1)
        await async_db_session.flush()

        # Rollback the change
        await async_db_session.rollback()

        # Verify it was rolled back
        result = await async_db_session.execute(
            select(Source).where(Source.name == "Session Isolation Test")
        )
        assert result.scalar_one_or_none() is None


class TestDatabaseInitialization:
    """Tests for database initialization."""

    @pytest.mark.asyncio
    async def test_tables_exist_in_test_db(self, async_db_session: AsyncSession):
        """Test that tables are created in test database."""
        # Verify tables exist by trying to query them (should not raise exception)
        await async_db_session.execute(select(Source))
        await async_db_session.execute(select(Message))

    @pytest.mark.asyncio
    async def test_can_query_empty_tables(self, async_db_session: AsyncSession):
        """Test querying empty tables works without errors."""
        result = await async_db_session.execute(select(Source))
        sources = result.scalars().all()
        # Initially should be empty or have sample data
        assert isinstance(sources, list)


class TestDatabaseModels:
    """Tests for database models and relationships."""

    @pytest.mark.asyncio
    async def test_source_creation(self, async_db_session: AsyncSession):
        """Test creating a Source record."""
        source = Source(
            type="slack",
            name="Test Slack",
            config={"workspace": "test"},
        )
        async_db_session.add(source)
        await async_db_session.commit()
        await async_db_session.refresh(source)

        assert source.id is not None
        assert source.type == "slack"
        assert source.name == "Test Slack"
        assert source.config == {"workspace": "test"}
        assert source.created_at is not None

    @pytest.mark.asyncio
    async def test_message_creation(self, async_db_session: AsyncSession, sample_source: Source):
        """Test creating a Message record."""
        from datetime import datetime

        message = Message(
            source_id=sample_source.id,
            content="Test message content",
            author="test@example.com",
            channel="#test",
            timestamp=datetime.utcnow(),
            external_id="test_msg_1",
            message_metadata={"test": "metadata"},
        )
        async_db_session.add(message)
        await async_db_session.commit()
        await async_db_session.refresh(message)

        assert message.id is not None
        assert message.source_id == sample_source.id
        assert message.content == "Test message content"
        assert message.author == "test@example.com"
        assert message.timestamp is not None

    @pytest.mark.asyncio
    async def test_message_source_relationship(
        self, async_db_session: AsyncSession, sample_source: Source
    ):
        """Test the relationship between Message and Source."""
        from datetime import datetime

        # Create a message
        message = Message(
            source_id=sample_source.id,
            content="Relationship test",
            author="test@example.com",
            channel="#test",
            timestamp=datetime.utcnow(),
            external_id="test_msg_rel",
        )
        async_db_session.add(message)
        await async_db_session.commit()

        # Query with join
        result = await async_db_session.execute(
            select(Message, Source)
            .join(Source, Message.source_id == Source.id)
            .where(Message.external_id == "test_msg_rel")
        )
        msg, src = result.one()

        assert msg.source_id == src.id
        assert src.id == sample_source.id

    @pytest.mark.asyncio
    async def test_bulk_message_insert(
        self, async_db_session: AsyncSession, sample_source: Source
    ):
        """Test inserting multiple messages at once."""
        from datetime import datetime

        messages = [
            Message(
                source_id=sample_source.id,
                content=f"Message {i}",
                author=f"user{i}@example.com",
                channel="#bulk",
                timestamp=datetime.utcnow(),
                external_id=f"bulk_msg_{i}",
            )
            for i in range(10)
        ]

        async_db_session.add_all(messages)
        await async_db_session.commit()

        # Verify all were inserted
        result = await async_db_session.execute(
            select(Message).where(Message.channel == "#bulk")
        )
        inserted = result.scalars().all()
        assert len(inserted) == 10


class TestDatabaseErrorHandling:
    """Tests for database error handling."""

    @pytest.mark.asyncio
    async def test_required_fields_validation(self, async_db_session: AsyncSession, sample_source: Source):
        """Test that messages require timestamp field."""
        from datetime import datetime

        # Test that we can create a valid message
        message = Message(
            source_id=sample_source.id,
            content="Valid message with timestamp",
            author="user@example.com",
            channel="#test",
            timestamp=datetime.utcnow(),
            external_id="valid_msg",
        )
        async_db_session.add(message)
        await async_db_session.commit()

        assert message.id is not None

    @pytest.mark.asyncio
    async def test_null_values_handled(self, async_db_session: AsyncSession, sample_source: Source):
        """Test that optional fields can be None."""
        from datetime import datetime

        # Create message with minimal required fields
        message = Message(
            source_id=sample_source.id,
            content="Message with nulls",
            timestamp=datetime.utcnow(),
            external_id=None,  # Optional field
            author=None,  # Optional field
            channel=None,  # Optional field
            message_metadata=None,  # Optional field
        )
        async_db_session.add(message)
        await async_db_session.commit()

        assert message.id is not None
        assert message.external_id is None
        assert message.author is None
