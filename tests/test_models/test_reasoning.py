"""
Tests for multi-step reasoning chains.

These tests verify the gap detection reasoning logic with mocked LLM calls.
"""

from unittest.mock import Mock, patch

import pytest

from src.models.reasoning import (
    GapReasoningChain,
    GapVerificationResult,
    InsightResult,
    RecommendationResult,
)
from src.models.llm import ClaudeClient


@pytest.fixture
def mock_claude_client():
    """Create a mock Claude client."""
    client = Mock(spec=ClaudeClient)
    return client


@pytest.fixture
def reasoning_chain(mock_claude_client):
    """Create a GapReasoningChain with mock client."""
    return GapReasoningChain(client=mock_claude_client)


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {
            "id": "msg1",
            "content": "Starting OAuth2 implementation",
            "author": "alice@company.com",
            "channel": "#platform",
            "timestamp": "2024-12-01T09:00:00Z",
        },
        {
            "id": "msg2",
            "content": "We're building OAuth support",
            "author": "bob@company.com",
            "channel": "#auth",
            "timestamp": "2024-12-01T14:00:00Z",
        },
    ]


@pytest.fixture
def sample_teams():
    """Sample team list."""
    return ["platform-team", "auth-team"]


def test_verify_duplicate_work_positive(reasoning_chain, mock_claude_client, sample_messages, sample_teams):
    """Test duplicate work verification with positive result."""
    # Mock LLM response
    mock_claude_client.complete_json.return_value = {
        "is_duplicate": True,
        "confidence": 0.89,
        "reasoning": "Both teams implementing OAuth2 independently",
        "evidence": ["OAuth2 mentioned in both channels"],
        "recommendation": "Connect alice and bob immediately",
        "overlap_ratio": 0.95,
    }

    result = reasoning_chain.verify_duplicate_work(
        messages=sample_messages,
        teams=sample_teams,
        topic="OAuth2 integration",
        start_date="2024-12-01",
        end_date="2024-12-15",
        overlap_days=14,
    )

    assert isinstance(result, GapVerificationResult)
    assert result.is_duplicate is True
    assert result.confidence == 0.89
    assert "OAuth2" in result.reasoning
    assert len(result.evidence) > 0
    assert result.overlap_ratio == 0.95


def test_verify_duplicate_work_negative(reasoning_chain, mock_claude_client, sample_messages, sample_teams):
    """Test duplicate work verification with negative result."""
    # Mock LLM response indicating no duplication
    mock_claude_client.complete_json.return_value = {
        "is_duplicate": False,
        "confidence": 0.25,
        "reasoning": "Different scopes - platform doing gateway, auth doing service auth",
        "evidence": [],
        "recommendation": "No action needed - complementary work",
        "overlap_ratio": 0.1,
    }

    result = reasoning_chain.verify_duplicate_work(
        messages=sample_messages,
        teams=sample_teams,
        topic="Authentication",
        start_date="2024-12-01",
        end_date="2024-12-15",
        overlap_days=14,
    )

    assert isinstance(result, GapVerificationResult)
    assert result.is_duplicate is False
    assert result.confidence == 0.25
    assert result.overlap_ratio == 0.1


def test_verify_duplicate_work_error_handling(reasoning_chain, mock_claude_client, sample_messages, sample_teams):
    """Test error handling in duplicate work verification."""
    # Mock LLM error
    mock_claude_client.complete_json.side_effect = Exception("API Error")

    result = reasoning_chain.verify_duplicate_work(
        messages=sample_messages,
        teams=sample_teams,
        topic="OAuth2",
        start_date="2024-12-01",
        end_date="2024-12-15",
        overlap_days=14,
    )

    # Should return conservative result on error
    assert isinstance(result, GapVerificationResult)
    assert result.is_duplicate is False
    assert result.confidence == 0.0
    assert "failed" in result.reasoning.lower()


def test_generate_insights(reasoning_chain, mock_claude_client):
    """Test insight generation."""
    # Mock LLM response
    mock_claude_client.complete_json.return_value = {
        "summary": "Two teams independently implementing OAuth2",
        "root_cause": "Lack of cross-team communication",
        "impact": "~80 hours of duplicate engineering effort",
        "immediate_actions": [
            "Connect team leads",
            "Review implementations for consolidation",
        ],
        "preventive_measures": [
            "Weekly architecture sync",
            "Shared roadmap visibility",
        ],
    }

    evidence = [
        {"source": "slack", "content": "Starting OAuth2..."},
        {"source": "slack", "content": "Building OAuth..."},
    ]

    result = reasoning_chain.generate_insights(
        gap_type="DUPLICATE_WORK",
        teams=["platform-team", "auth-team"],
        topic="OAuth2 integration",
        evidence=evidence,
    )

    assert isinstance(result, InsightResult)
    assert "OAuth2" in result.summary
    assert len(result.immediate_actions) > 0
    assert len(result.preventive_measures) > 0


def test_generate_insights_error_handling(reasoning_chain, mock_claude_client):
    """Test error handling in insight generation."""
    # Mock LLM error
    mock_claude_client.complete_json.side_effect = Exception("API Error")

    result = reasoning_chain.generate_insights(
        gap_type="DUPLICATE_WORK",
        teams=["team1"],
        topic="test",
        evidence=[],
    )

    # Should return basic result on error
    assert isinstance(result, InsightResult)
    assert "unable" in result.summary.lower() or "error" in result.summary.lower()
    assert "Manual review" in result.immediate_actions[0]


def test_generate_recommendations(reasoning_chain, mock_claude_client):
    """Test recommendation generation."""
    # Mock LLM response
    mock_claude_client.complete_json.return_value = {
        "immediate_actions": [
            {
                "action": "Schedule sync meeting",
                "owner": "Engineering Manager",
                "urgency": "critical",
            }
        ],
        "short_term_fixes": [
            {
                "action": "Consolidate implementations",
                "timeline": "This week",
                "expected_impact": "Prevent wasted effort",
            }
        ],
        "long_term_improvements": [
            {
                "action": "Implement architecture review process",
                "rationale": "Prevent future duplication",
                "effort": "medium",
            }
        ],
        "talking_points": [
            "Duplicate OAuth2 work detected",
            "Estimated 80 hours wasted",
            "Need better coordination",
        ],
    }

    evidence = [{"source": "slack", "content": "test"}]

    result = reasoning_chain.generate_recommendations(
        gap_type="DUPLICATE_WORK",
        teams=["team1", "team2"],
        topic="OAuth2",
        impact_score=0.89,
        evidence=evidence,
    )

    assert isinstance(result, RecommendationResult)
    assert len(result.immediate_actions) > 0
    assert len(result.short_term_fixes) > 0
    assert len(result.long_term_improvements) > 0
    assert len(result.talking_points) > 0


def test_generate_recommendations_error_handling(reasoning_chain, mock_claude_client):
    """Test error handling in recommendation generation."""
    # Mock LLM error
    mock_claude_client.complete_json.side_effect = Exception("API Error")

    result = reasoning_chain.generate_recommendations(
        gap_type="DUPLICATE_WORK",
        teams=["team1"],
        topic="test",
        impact_score=0.8,
        evidence=[],
    )

    # Should return basic recommendations on error
    assert isinstance(result, RecommendationResult)
    assert len(result.immediate_actions) > 0
    assert "Manual" in result.immediate_actions[0]["action"]


def test_analyze_gap_full_verified_high_confidence(
    reasoning_chain, mock_claude_client, sample_messages, sample_teams
):
    """Test full gap analysis with verified high-confidence gap."""
    # Mock verification response (high confidence)
    mock_claude_client.complete_json.side_effect = [
        # Verification
        {
            "is_duplicate": True,
            "confidence": 0.85,
            "reasoning": "Clear duplication",
            "evidence": ["key evidence"],
            "recommendation": "Act now",
            "overlap_ratio": 0.9,
        },
        # Insights
        {
            "summary": "Duplicate OAuth work",
            "root_cause": "Poor communication",
            "impact": "High",
            "immediate_actions": ["Connect teams"],
            "preventive_measures": ["Better planning"],
        },
        # Recommendations
        {
            "immediate_actions": [{"action": "Meet", "owner": "Manager", "urgency": "high"}],
            "short_term_fixes": [{"action": "Fix", "timeline": "Week", "expected_impact": "Good"}],
            "long_term_improvements": [{"action": "Improve", "rationale": "Prevention", "effort": "medium"}],
            "talking_points": ["Point 1"],
        },
    ]

    result = reasoning_chain.analyze_gap_full(
        messages=sample_messages,
        teams=sample_teams,
        topic="OAuth2",
        start_date="2024-12-01",
        end_date="2024-12-15",
        overlap_days=14,
    )

    assert "verification" in result
    assert "insights" in result
    assert "recommendations" in result

    # All should be populated for high-confidence gap
    assert result["verification"]["is_duplicate"] is True
    assert result["verification"]["confidence"] == 0.85
    assert result["insights"] is not None
    assert result["recommendations"] is not None


def test_analyze_gap_full_verified_medium_confidence(
    reasoning_chain, mock_claude_client, sample_messages, sample_teams
):
    """Test full gap analysis with medium-confidence gap."""
    # Mock verification response (medium confidence - insights but no recommendations)
    mock_claude_client.complete_json.side_effect = [
        # Verification
        {
            "is_duplicate": True,
            "confidence": 0.65,
            "reasoning": "Possible duplication",
            "evidence": ["some evidence"],
            "recommendation": "Investigate",
            "overlap_ratio": 0.6,
        },
        # Insights
        {
            "summary": "Possible duplicate work",
            "root_cause": "Unclear",
            "impact": "Medium",
            "immediate_actions": ["Review"],
            "preventive_measures": ["Monitor"],
        },
    ]

    result = reasoning_chain.analyze_gap_full(
        messages=sample_messages,
        teams=sample_teams,
        topic="OAuth2",
        start_date="2024-12-01",
        end_date="2024-12-15",
        overlap_days=14,
    )

    assert result["verification"]["is_duplicate"] is True
    assert result["verification"]["confidence"] == 0.65
    assert result["insights"] is not None
    # Recommendations should be None (confidence < 0.7)
    assert result["recommendations"] is None


def test_analyze_gap_full_not_verified(
    reasoning_chain, mock_claude_client, sample_messages, sample_teams
):
    """Test full gap analysis when gap is not verified."""
    # Mock verification response (not a gap)
    mock_claude_client.complete_json.return_value = {
        "is_duplicate": False,
        "confidence": 0.2,
        "reasoning": "Different scopes",
        "evidence": [],
        "recommendation": "No action",
        "overlap_ratio": 0.1,
    }

    result = reasoning_chain.analyze_gap_full(
        messages=sample_messages,
        teams=sample_teams,
        topic="Authentication",
        start_date="2024-12-01",
        end_date="2024-12-15",
        overlap_days=14,
    )

    assert result["verification"]["is_duplicate"] is False
    # Should not generate insights or recommendations for non-gaps
    assert result["insights"] is None
    assert result["recommendations"] is None


def test_batch_verify_gaps(reasoning_chain, mock_claude_client):
    """Test batch verification of multiple gaps."""
    # Mock responses for multiple gaps
    mock_claude_client.complete_json.side_effect = [
        # Gap 1 - verified
        {
            "is_duplicate": True,
            "confidence": 0.9,
            "reasoning": "Clear dup",
            "evidence": ["e1"],
            "recommendation": "Fix",
            "overlap_ratio": 0.95,
        },
        # Gap 2 - not verified
        {
            "is_duplicate": False,
            "confidence": 0.3,
            "reasoning": "Different",
            "evidence": [],
            "recommendation": "None",
            "overlap_ratio": 0.2,
        },
        # Gap 3 - verified
        {
            "is_duplicate": True,
            "confidence": 0.85,
            "reasoning": "Duplicate",
            "evidence": ["e2"],
            "recommendation": "Act",
            "overlap_ratio": 0.9,
        },
    ]

    potential_gaps = [
        {
            "messages": [{"content": "msg1"}],
            "teams": ["team1", "team2"],
            "topic": "OAuth",
            "start_date": "2024-12-01",
            "end_date": "2024-12-15",
            "overlap_days": 14,
        },
        {
            "messages": [{"content": "msg2"}],
            "teams": ["team3", "team4"],
            "topic": "API",
            "start_date": "2024-12-01",
            "end_date": "2024-12-15",
            "overlap_days": 10,
        },
        {
            "messages": [{"content": "msg3"}],
            "teams": ["team5", "team6"],
            "topic": "Auth",
            "start_date": "2024-12-01",
            "end_date": "2024-12-15",
            "overlap_days": 12,
        },
    ]

    results = reasoning_chain.batch_verify_gaps(potential_gaps)

    assert len(results) == 3
    assert results[0].is_duplicate is True
    assert results[1].is_duplicate is False
    assert results[2].is_duplicate is True


def test_batch_verify_gaps_with_errors(reasoning_chain, mock_claude_client):
    """Test batch verification handles individual errors gracefully."""
    # First succeeds, second fails, third succeeds
    mock_claude_client.complete_json.side_effect = [
        {
            "is_duplicate": True,
            "confidence": 0.9,
            "reasoning": "Dup",
            "evidence": ["e1"],
            "recommendation": "Fix",
            "overlap_ratio": 0.9,
        },
        Exception("API Error"),
        {
            "is_duplicate": True,
            "confidence": 0.85,
            "reasoning": "Dup",
            "evidence": ["e2"],
            "recommendation": "Fix",
            "overlap_ratio": 0.85,
        },
    ]

    potential_gaps = [
        {
            "messages": [{"content": "msg1"}],
            "teams": ["team1"],
            "topic": "OAuth",
            "start_date": "2024-12-01",
            "end_date": "2024-12-15",
            "overlap_days": 14,
        },
        {
            "messages": [{"content": "msg2"}],
            "teams": ["team2"],
            "topic": "API",
            "start_date": "2024-12-01",
            "end_date": "2024-12-15",
            "overlap_days": 10,
        },
        {
            "messages": [{"content": "msg3"}],
            "teams": ["team3"],
            "topic": "Auth",
            "start_date": "2024-12-01",
            "end_date": "2024-12-15",
            "overlap_days": 12,
        },
    ]

    results = reasoning_chain.batch_verify_gaps(potential_gaps)

    assert len(results) == 3
    assert results[0].is_duplicate is True
    # Second should have error result
    assert results[1].is_duplicate is False
    assert results[1].confidence == 0.0
    assert results[2].is_duplicate is True
