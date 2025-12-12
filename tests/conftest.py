"""
Pytest configuration and shared fixtures for tests.
"""
from datetime import datetime, timedelta
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Message, Source
from src.main import create_app


@pytest.fixture
def app():
    """Create a FastAPI app instance for testing."""
    return create_app()


@pytest.fixture
def client(app):
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from src.config import Settings

    return Settings(
        environment="test",
        anthropic_api_key="test-key",
        postgres_url="postgresql://test:test@localhost:5432/test_coordination",
        redis_url="redis://localhost:6379/1",
    )


@pytest.fixture
async def async_db_engine():
    """Create an in-memory async SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def async_db_session(async_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async database session for testing."""
    async_session_maker = sessionmaker(
        async_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def sample_source(async_db_session: AsyncSession) -> Source:
    """Create a sample source for testing."""
    source = Source(
        type="slack",
        name="Engineering Slack",
        config={"workspace": "test-workspace"},
    )
    async_db_session.add(source)
    await async_db_session.commit()
    await async_db_session.refresh(source)
    return source


@pytest.fixture
async def sample_messages(async_db_session: AsyncSession, sample_source: Source) -> list[Message]:
    """Create sample messages for testing."""
    now = datetime.utcnow()

    messages = [
        Message(
            source_id=sample_source.id,
            content="OAuth2 implementation discussion in platform team",
            author="alice@demo.com",
            channel="#platform",
            timestamp=now - timedelta(hours=2),
            external_id="slack_msg_1",
            message_metadata={"reactions": ["thumbsup"]},
        ),
        Message(
            source_id=sample_source.id,
            content="Starting OAuth integration with Auth0",
            author="bob@demo.com",
            channel="#engineering",
            timestamp=now - timedelta(hours=1),
            external_id="slack_msg_2",
            message_metadata={},
        ),
        Message(
            source_id=sample_source.id,
            content="Bug fix for login page responsiveness",
            author="charlie@demo.com",
            channel="#frontend",
            timestamp=now,
            external_id="slack_msg_3",
            message_metadata={},
        ),
    ]

    for msg in messages:
        async_db_session.add(msg)

    await async_db_session.commit()

    for msg in messages:
        await async_db_session.refresh(msg)

    return messages


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore for testing."""
    mock_store = MagicMock()
    mock_store.check_connection.return_value = True
    mock_store.get_collection_count.return_value = 10
    mock_store.collection_name = "test_collection"

    # Mock search results
    mock_store.search.return_value = [
        ("msg_1", "OAuth2 implementation discussion", 0.92, {"message_id": 1}),
        ("msg_2", "OAuth integration with Auth0", 0.87, {"message_id": 2}),
    ]

    return mock_store


@pytest.fixture
def mock_search_service(mock_vector_store):
    """Create a mock SearchService for testing."""
    from src.services.search_service import SearchService

    service = SearchService(mock_vector_store)
    return service


# TODO: Add more fixtures as needed:
# - Mock external API clients (Slack, GitHub, etc.)
# - Mock LLM responses
