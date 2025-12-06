"""
Pytest configuration and shared fixtures for tests.
"""
import pytest
from fastapi.testclient import TestClient

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


# TODO: Add more fixtures as needed:
# - Mock database connections
# - Mock external API clients (Slack, GitHub, etc.)
# - Sample data fixtures
# - Mock LLM responses
