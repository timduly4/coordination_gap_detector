"""
Pytest fixtures for gap detection scenarios.

Provides easy access to scenario data for integration testing.
"""
import pytest
from typing import List, Dict, Any

from src.ingestion.slack.mock_client import MockSlackClient, MockMessage


@pytest.fixture(scope="session")
def mock_slack_client():
    """Session-scoped MockSlackClient instance."""
    return MockSlackClient()


@pytest.fixture
def oauth_duplication_scenario(mock_slack_client) -> Dict[str, Any]:
    """
    OAuth duplication scenario (positive case).

    Expected: Should detect as duplicate work with HIGH impact.

    Returns:
        Dict with:
            - messages: List of MockMessage objects
            - expected_detect: True
            - expected_impact_tier: "HIGH"
            - expected_teams: ["platform-team", "auth-team"]
            - expected_topic: "OAuth2 integration"
    """
    messages = mock_slack_client.get_scenario_messages("oauth_duplication")
    return {
        "scenario_id": "oauth_duplication_001",
        "name": "OAuth Integration Duplication",
        "messages": messages,
        "expected_detect": True,
        "expected_impact_tier": "HIGH",
        "expected_impact_score_min": 0.80,
        "expected_confidence_min": 0.75,
        "expected_teams": ["platform-team", "auth-team"],
        "expected_topic": "OAuth2 integration",
        "expected_overlap_days_min": 10,
        "expected_evidence_count_min": 10,
    }


@pytest.fixture
def api_redesign_scenario(mock_slack_client) -> Dict[str, Any]:
    """
    API redesign duplication scenario (positive case).

    Expected: Should detect as duplicate work with HIGH impact.

    Returns:
        Dict with scenario metadata and expected detection results.
    """
    messages = mock_slack_client.get_scenario_messages("api_redesign_duplication")
    return {
        "scenario_id": "api_redesign_001",
        "name": "API Redesign Duplication",
        "messages": messages,
        "expected_detect": True,
        "expected_impact_tier": "HIGH",
        "expected_impact_score_min": 0.75,
        "expected_confidence_min": 0.70,
        "expected_teams": ["platform-team", "backend-team"],
        "expected_topic": "REST API redesign",
        "expected_overlap_days_min": 15,
        "expected_evidence_count_min": 8,
    }


@pytest.fixture
def auth_migration_scenario(mock_slack_client) -> Dict[str, Any]:
    """
    Auth migration duplication scenario (positive case).

    Expected: Should detect as duplicate work with MEDIUM-HIGH impact.

    Returns:
        Dict with scenario metadata and expected detection results.
    """
    messages = mock_slack_client.get_scenario_messages("auth_migration_duplication")
    return {
        "scenario_id": "auth_migration_001",
        "name": "Auth Migration Duplication",
        "messages": messages,
        "expected_detect": True,
        "expected_impact_tier": "MEDIUM_HIGH",
        "expected_impact_score_min": 0.70,
        "expected_confidence_min": 0.70,
        "expected_teams": ["security-team", "platform-team"],
        "expected_topic": "JWT authentication migration",
        "expected_overlap_days_min": 8,
        "expected_evidence_count_min": 6,
    }


@pytest.fixture
def similar_topics_different_scope_scenario(mock_slack_client) -> Dict[str, Any]:
    """
    Similar topics but different scope edge case (negative case).

    Expected: Should NOT detect as duplicate work.

    Returns:
        Dict with scenario metadata and expected detection results.
    """
    messages = mock_slack_client.get_scenario_messages("similar_topics_different_scope")
    return {
        "scenario_id": "edge_case_different_scope_001",
        "name": "Similar Topics, Different Scope",
        "messages": messages,
        "expected_detect": False,
        "reason": "Different problem domains - user auth vs service-to-service auth",
        "expected_teams": ["frontend-team", "platform-team"],
    }


@pytest.fixture
def sequential_work_scenario(mock_slack_client) -> Dict[str, Any]:
    """
    Sequential work with no temporal overlap edge case (negative case).

    Expected: Should NOT detect as duplicate work.

    Returns:
        Dict with scenario metadata and expected detection results.
    """
    messages = mock_slack_client.get_scenario_messages("sequential_work")
    return {
        "scenario_id": "edge_case_sequential_001",
        "name": "Sequential Work, No Overlap",
        "messages": messages,
        "expected_detect": False,
        "reason": "No temporal overlap - teams worked 60+ days apart",
        "expected_teams": ["web-team", "mobile-team"],
    }


@pytest.fixture
def intentional_collaboration_scenario(mock_slack_client) -> Dict[str, Any]:
    """
    Intentional collaboration scenario (negative example).

    Expected: Should NOT detect as duplicate work.

    Returns:
        Dict with scenario metadata and expected detection results.
    """
    messages = mock_slack_client.get_scenario_messages("intentional_collaboration")
    return {
        "scenario_id": "collaboration_001",
        "name": "Intentional Collaboration",
        "messages": messages,
        "expected_detect": False,
        "reason": "Explicit coordination and cross-references present",
        "expected_teams": ["platform-team", "auth-team"],
        "collaboration_indicators": [
            "cross-references",
            "division of labor",
            "shared meetings",
            "coordination mentions",
        ],
    }


@pytest.fixture
def all_positive_scenarios(
    oauth_duplication_scenario,
    api_redesign_scenario,
    auth_migration_scenario,
) -> List[Dict[str, Any]]:
    """All positive scenarios that should detect as duplicate work."""
    return [
        oauth_duplication_scenario,
        api_redesign_scenario,
        auth_migration_scenario,
    ]


@pytest.fixture
def all_negative_scenarios(
    similar_topics_different_scope_scenario,
    sequential_work_scenario,
    intentional_collaboration_scenario,
) -> List[Dict[str, Any]]:
    """All negative scenarios that should NOT detect as duplicate work."""
    return [
        similar_topics_different_scope_scenario,
        sequential_work_scenario,
        intentional_collaboration_scenario,
    ]


@pytest.fixture
def all_gap_scenarios(
    all_positive_scenarios,
    all_negative_scenarios,
) -> Dict[str, List[Dict[str, Any]]]:
    """All gap detection scenarios organized by type."""
    return {
        "positive": all_positive_scenarios,
        "negative": all_negative_scenarios,
    }


# Scenario metadata helpers

@pytest.fixture
def scenario_expectations() -> Dict[str, Dict[str, Any]]:
    """
    Expected detection results for all scenarios.

    Useful for parameterized testing and validation.
    """
    return {
        "oauth_duplication": {
            "should_detect": True,
            "min_impact_score": 0.80,
            "impact_tier": "HIGH",
            "min_confidence": 0.75,
            "teams": ["platform-team", "auth-team"],
            "min_overlap_days": 10,
        },
        "api_redesign_duplication": {
            "should_detect": True,
            "min_impact_score": 0.75,
            "impact_tier": "HIGH",
            "min_confidence": 0.70,
            "teams": ["platform-team", "backend-team"],
            "min_overlap_days": 15,
        },
        "auth_migration_duplication": {
            "should_detect": True,
            "min_impact_score": 0.70,
            "impact_tier": "MEDIUM_HIGH",
            "min_confidence": 0.70,
            "teams": ["security-team", "platform-team"],
            "min_overlap_days": 8,
        },
        "similar_topics_different_scope": {
            "should_detect": False,
            "reason": "Different scopes",
        },
        "sequential_work": {
            "should_detect": False,
            "reason": "No temporal overlap",
        },
        "intentional_collaboration": {
            "should_detect": False,
            "reason": "Explicit collaboration",
        },
    }
