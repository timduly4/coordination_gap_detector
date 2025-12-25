"""
Tests for Claude API client wrapper.

These tests use mocked Anthropic API calls to ensure reliability
without requiring actual API credits.
"""

import json
from unittest.mock import Mock, patch, MagicMock

import pytest
from anthropic import RateLimitError, APIError
from pydantic import BaseModel

from src.models.llm import ClaudeClient, ClaudeResponse


class SampleSchema(BaseModel):
    """Sample schema for testing structured output."""

    name: str
    score: float
    tags: list[str]


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    mock_response = Mock()
    mock_response.content = [Mock(text='{"name": "test", "score": 0.9, "tags": ["a", "b"]}')]
    mock_response.model = "claude-sonnet-4-5-20250929"
    mock_response.stop_reason = "end_turn"
    mock_response.usage = Mock(input_tokens=100, output_tokens=50)
    return mock_response


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response):
    """Create a mock Anthropic client."""
    with patch("src.models.llm.Anthropic") as mock_anthropic:
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_client
        yield mock_anthropic


@pytest.fixture
def claude_client(mock_anthropic_client):
    """Create a ClaudeClient with mocked API."""
    # Reset global token counter and cost tracker for test isolation
    import src.utils.token_utils as token_utils
    token_utils._token_counter = None
    token_utils._cost_tracker = None

    return ClaudeClient(api_key="test_key")


def test_claude_client_initialization():
    """Test ClaudeClient initializes correctly."""
    with patch("src.models.llm.Anthropic"):
        client = ClaudeClient(
            api_key="test_key",
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            temperature=0.5,
        )

        assert client.api_key == "test_key"
        assert client.model == "claude-sonnet-4-5-20250929"
        assert client.max_tokens == 2048
        assert client.temperature == 0.5


def test_claude_client_requires_api_key():
    """Test ClaudeClient raises error without API key."""
    with patch("src.models.llm.Anthropic"):
        with patch("src.models.llm.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None

            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not found"):
                ClaudeClient()


def test_complete_basic(claude_client, mock_anthropic_response):
    """Test basic completion call."""
    response = claude_client.complete(prompt="Test prompt")

    assert isinstance(response, ClaudeResponse)
    assert response.content == '{"name": "test", "score": 0.9, "tags": ["a", "b"]}'
    assert response.model == "claude-sonnet-4-5-20250929"
    assert response.input_tokens == 100
    assert response.output_tokens == 50


def test_complete_with_system_prompt(claude_client):
    """Test completion with system prompt."""
    response = claude_client.complete(
        prompt="User prompt", system_prompt="System instructions"
    )

    assert isinstance(response, ClaudeResponse)

    # Verify system prompt was passed to API
    claude_client.client.messages.create.assert_called_once()
    call_kwargs = claude_client.client.messages.create.call_args.kwargs
    assert "system" in call_kwargs
    assert call_kwargs["system"] == "System instructions"


def test_complete_with_custom_params(claude_client):
    """Test completion with custom parameters."""
    response = claude_client.complete(
        prompt="Test", max_tokens=1000, temperature=0.8
    )

    assert isinstance(response, ClaudeResponse)

    # Verify custom params were used
    call_kwargs = claude_client.client.messages.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 1000
    assert call_kwargs["temperature"] == 0.8


def test_complete_json_basic(claude_client):
    """Test JSON completion parsing."""
    result = claude_client.complete_json(prompt="Generate JSON")

    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["score"] == 0.9
    assert result["tags"] == ["a", "b"]


def test_complete_json_with_schema(claude_client):
    """Test JSON completion with schema validation."""
    result = claude_client.complete_json(
        prompt="Generate JSON", schema=SampleSchema
    )

    assert isinstance(result, dict)
    assert "name" in result
    assert "score" in result
    assert "tags" in result


def test_complete_json_invalid_json(claude_client):
    """Test JSON completion handles invalid JSON."""
    # Mock response with invalid JSON
    claude_client.client.messages.create.return_value.content = [
        Mock(text="Not valid JSON")
    ]

    with pytest.raises(ValueError, match="Invalid JSON"):
        claude_client.complete_json(prompt="Test")


def test_complete_json_schema_validation_failure(claude_client):
    """Test JSON completion with schema validation failure."""
    # Mock response with JSON that doesn't match schema
    claude_client.client.messages.create.return_value.content = [
        Mock(text='{"wrong": "fields"}')
    ]

    with pytest.raises(ValueError, match="validation failed"):
        claude_client.complete_json(prompt="Test", schema=SampleSchema)


def test_parse_json(claude_client):
    """Test JSON parsing from response."""
    response = ClaudeResponse(
        content='{"key": "value"}',
        model="claude-sonnet-4-5-20250929",
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
    )

    result = claude_client.parse_json(response)
    assert result == {"key": "value"}


def test_parse_json_invalid(claude_client):
    """Test JSON parsing with invalid JSON."""
    response = ClaudeResponse(
        content="Not JSON",
        model="claude-sonnet-4-5-20250929",
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
    )

    with pytest.raises(ValueError, match="Invalid JSON"):
        claude_client.parse_json(response)


def test_retry_on_rate_limit(claude_client):
    """Test retry logic on rate limit error."""
    # Create properly mocked RateLimitError
    mock_response = Mock()
    mock_response.status_code = 429
    rate_limit_error = RateLimitError(
        "Rate limited",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}}
    )

    # First call raises RateLimitError, second succeeds
    claude_client.client.messages.create.side_effect = [
        rate_limit_error,
        claude_client.client.messages.create.return_value,
    ]

    response = claude_client.complete(prompt="Test")
    assert isinstance(response, ClaudeResponse)

    # Verify it was called twice (first fail, then succeed)
    assert claude_client.client.messages.create.call_count == 2


def test_quota_check_before_call(claude_client):
    """Test quota is checked before API call."""
    # Set usage to exceed quota
    claude_client.cost_tracker.daily_usage = claude_client.daily_quota + 1000

    with pytest.raises(ValueError, match="quota exceeded"):
        claude_client.complete(prompt="Test")


def test_usage_tracking(claude_client):
    """Test usage and cost tracking."""
    # Reset quota to ensure clean state
    claude_client.cost_tracker.daily_usage = 0
    initial_usage = claude_client.cost_tracker.daily_usage

    response = claude_client.complete(prompt="Test")

    # Verify usage was tracked
    assert claude_client.cost_tracker.daily_usage > initial_usage
    assert response.input_tokens == 100
    assert response.output_tokens == 50
    assert response.cost_usd > 0


def test_get_usage_stats(claude_client):
    """Test getting usage statistics."""
    # Reset quota to ensure clean state
    claude_client.cost_tracker.daily_usage = 0

    # Make a call to generate usage
    claude_client.complete(prompt="Test")

    stats = claude_client.get_usage_stats(days=1)

    assert "total_calls" in stats
    assert "total_input_tokens" in stats
    assert "total_output_tokens" in stats
    assert "total_cost" in stats
    assert stats["total_calls"] >= 1


def test_get_daily_usage(claude_client):
    """Test getting daily usage stats."""
    # Reset quota to ensure clean state
    claude_client.cost_tracker.daily_usage = 0

    # Make a call to generate usage
    claude_client.complete(prompt="Test")

    daily = claude_client.get_daily_usage()

    assert "daily_usage" in daily
    assert "daily_quota" in daily
    assert "quota_remaining" in daily
    assert daily["daily_usage"] > 0


def test_estimate_cost(claude_client):
    """Test cost estimation."""
    estimate = claude_client.estimate_cost(prompt="Short prompt", max_output_tokens=500)

    assert "input_tokens" in estimate
    assert "estimated_output_tokens" in estimate
    assert "total_estimated_cost_usd" in estimate
    assert estimate["estimated_output_tokens"] == 500
    assert estimate["total_estimated_cost_usd"] > 0


def test_api_error_handling(claude_client):
    """Test handling of general API errors."""
    # Reset quota to ensure clean state
    claude_client.cost_tracker.daily_usage = 0

    # Mock a generic exception that won't trigger retry
    claude_client.client.messages.create.side_effect = ValueError("API Error")

    with pytest.raises(ValueError, match="API Error"):
        claude_client.complete(prompt="Test")


def test_unexpected_error_handling(claude_client):
    """Test handling of unexpected errors."""
    # Reset quota to ensure it doesn't interfere
    claude_client.cost_tracker.daily_usage = 0

    claude_client.client.messages.create.side_effect = Exception("Unexpected error")

    with pytest.raises(Exception, match="Unexpected error"):
        claude_client.complete(prompt="Test")


def test_cost_tracking_accumulation(claude_client):
    """Test that costs accumulate correctly over multiple calls."""
    # Reset quota to ensure clean state
    claude_client.cost_tracker.daily_usage = 0
    claude_client.cost_tracker.usage_history = []

    # Make multiple calls
    claude_client.complete(prompt="Call 1")
    claude_client.complete(prompt="Call 2")
    claude_client.complete(prompt="Call 3")

    stats = claude_client.get_usage_stats(days=1)

    assert stats["total_calls"] == 3
    assert stats["total_input_tokens"] == 300  # 100 per call
    assert stats["total_output_tokens"] == 150  # 50 per call


def test_warning_on_near_quota(claude_client):
    """Test that completion works when approaching quota."""
    # Set usage to 85% of quota
    claude_client.cost_tracker.daily_usage = int(claude_client.daily_quota * 0.85)

    # Check that we're near limit before the call
    stats_before = claude_client.get_daily_usage()
    assert stats_before["is_near_limit"] is True

    # Should still complete successfully
    response = claude_client.complete(prompt="Test")
    assert isinstance(response, ClaudeResponse)


@pytest.mark.parametrize(
    "prompt,expected_min_tokens",
    [
        ("Short", 1),
        ("This is a longer prompt with more words", 5),
        ("A" * 1000, 100),  # Long prompt
    ],
)
def test_token_counting(claude_client, prompt, expected_min_tokens):
    """Test token counting for various prompt lengths."""
    tokens = claude_client.token_counter.count_tokens(prompt)
    assert tokens >= expected_min_tokens
