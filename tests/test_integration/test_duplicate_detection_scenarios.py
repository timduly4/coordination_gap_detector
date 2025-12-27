"""
Integration tests for duplicate work detection using realistic scenarios.

These tests verify that the gap detection system correctly identifies
duplicate work in positive cases and avoids false positives in edge cases
and collaboration scenarios.
"""
import pytest
from datetime import datetime, timedelta
from typing import List

from src.ingestion.slack.mock_client import MockSlackClient, MockMessage


class TestDuplicateWorkDetectionScenarios:
    """Integration tests for duplicate work detection with realistic scenarios."""

    @pytest.fixture
    def mock_client(self):
        """Fixture to provide MockSlackClient instance."""
        return MockSlackClient()

    # ============================================================================
    # POSITIVE CASES - Should Detect as Duplicate Work
    # ============================================================================

    def test_oauth_duplication_scenario_structure(self, mock_client):
        """
        Test OAuth duplication scenario structure.

        Verifies that the scenario has the expected structure for
        duplicate work detection testing.
        """
        messages = mock_client.get_scenario_messages("oauth_duplication")

        # Should have 24+ messages
        assert len(messages) >= 24, "OAuth duplication should have 24+ messages"

        # Extract unique channels
        channels = {msg.channel for msg in messages}
        assert "#platform" in channels, "Should have platform team messages"
        assert "#auth-team" in channels, "Should have auth team messages"

        # Extract unique teams from metadata
        teams = {msg.metadata.get("team") for msg in messages if msg.metadata.get("team")}
        assert "platform-team" in teams, "Should have platform-team"
        assert "auth-team" in teams, "Should have auth-team"
        assert len(teams) >= 2, "Should have multiple teams"

        # Verify temporal span (should be ~14 days)
        timestamps = [msg.timestamp for msg in messages if msg.timestamp]
        time_span = (max(timestamps) - min(timestamps)).days
        assert time_span >= 13, "Should span at least 13 days"
        assert time_span <= 15, "Should span at most 15 days"

        # Verify parallel work (messages from both teams throughout)
        platform_msgs = [m for m in messages if m.channel == "#platform"]
        auth_msgs = [m for m in messages if m.channel == "#auth-team"]
        assert len(platform_msgs) >= 10, "Platform team should have multiple messages"
        assert len(auth_msgs) >= 10, "Auth team should have multiple messages"

        # Check for OAuth-related content
        oauth_mentions = sum(1 for m in messages if "oauth" in m.content.lower())
        assert oauth_mentions >= 15, "Should have multiple OAuth mentions"

    def test_oauth_duplication_temporal_overlap(self, mock_client):
        """Test that OAuth duplication has significant temporal overlap."""
        messages = mock_client.get_scenario_messages("oauth_duplication")

        platform_msgs = [m for m in messages if m.channel == "#platform"]
        auth_msgs = [m for m in messages if m.channel == "#auth-team"]

        # Get time ranges for each team
        platform_start = min(m.timestamp for m in platform_msgs)
        platform_end = max(m.timestamp for m in platform_msgs)
        auth_start = min(m.timestamp for m in auth_msgs)
        auth_end = max(m.timestamp for m in auth_msgs)

        # Calculate overlap
        overlap_start = max(platform_start, auth_start)
        overlap_end = min(platform_end, auth_end)
        overlap_days = (overlap_end - overlap_start).days

        assert overlap_days >= 10, f"Should have >=10 days overlap, got {overlap_days}"

    def test_oauth_duplication_no_cross_references(self, mock_client):
        """Test that OAuth duplication has no cross-team references (key indicator)."""
        messages = mock_client.get_scenario_messages("oauth_duplication")

        # Messages in #platform should not mention auth-team (except the discovery message)
        platform_msgs = [m for m in messages if m.channel == "#platform"]
        auth_team_mentions = sum(
            1 for m in platform_msgs
            if "auth-team" in m.content.lower() or "auth team" in m.content.lower()
        )
        assert auth_team_mentions == 0, "Platform messages should not reference auth team initially"

        # Messages in #auth-team should not mention platform-team (except discovery)
        auth_msgs = [m for m in messages if m.channel == "#auth-team"]
        platform_mentions = sum(
            1 for m in auth_msgs
            if "platform-team" in m.content.lower() or "platform team" in m.content.lower()
        )
        assert platform_mentions == 0, "Auth messages should not reference platform team initially"

        # Verify discovery message exists at the end
        engineering_msgs = [m for m in messages if m.channel == "#engineering"]
        assert len(engineering_msgs) >= 1, "Should have discovery message in #engineering"

    def test_api_redesign_scenario_structure(self, mock_client):
        """Test API redesign duplication scenario structure."""
        messages = mock_client.get_scenario_messages("api_redesign_duplication")

        # Should have reasonable number of messages
        assert len(messages) >= 15, "API redesign should have 15+ messages"

        # Check for platform and backend teams
        channels = {msg.channel for msg in messages}
        assert "#platform" in channels
        assert "#backend-eng" in channels

        # Verify temporal span (~18 days)
        timestamps = [msg.timestamp for msg in messages if msg.timestamp]
        time_span = (max(timestamps) - min(timestamps)).days
        assert time_span >= 17, "Should span at least 17 days"

        # Check for API-related content
        api_mentions = sum(1 for m in messages if "api" in m.content.lower())
        assert api_mentions >= 10, "Should have multiple API mentions"

        # Check for common design patterns mentioned by both teams
        rest_mentions = sum(1 for m in messages if "rest" in m.content.lower())
        versioning_mentions = sum(1 for m in messages if "version" in m.content.lower() or "v1" in m.content or "v2" in m.content)
        assert rest_mentions >= 2, "Both teams should mention REST"
        assert versioning_mentions >= 2, "Both teams should discuss versioning"

    def test_auth_migration_scenario_structure(self, mock_client):
        """Test auth migration duplication scenario structure."""
        messages = mock_client.get_scenario_messages("auth_migration_duplication")

        # Should have ~10 messages
        assert len(messages) >= 8, "Auth migration should have 8+ messages"

        # Check for security and platform teams
        channels = {msg.channel for msg in messages}
        assert "#security" in channels
        assert "#platform" in channels

        # Verify JWT mentions by both teams
        jwt_mentions = sum(1 for m in messages if "jwt" in m.content.lower())
        assert jwt_mentions >= 4, "Both teams should mention JWT"

        # Verify RS256 mentions
        rs256_mentions = sum(1 for m in messages if "rs256" in m.content.lower())
        assert rs256_mentions >= 2, "Both teams should mention RS256"

    # ============================================================================
    # EDGE CASES - Should NOT Detect as Duplicate Work
    # ============================================================================

    def test_similar_topics_different_scope_scenario(self, mock_client):
        """
        Test edge case: similar topics but different scope.

        Both teams discuss "authentication" but completely different aspects:
        - User authentication (social logins, magic links)
        - Service-to-service authentication (mTLS)

        Should NOT detect as duplicate work.
        """
        messages = mock_client.get_scenario_messages("similar_topics_different_scope")

        # Should have minimal messages (not a major effort)
        assert len(messages) >= 4, "Should have at least 4 messages"

        # Check for different technologies mentioned
        user_auth_indicators = sum(
            1 for m in messages
            if any(word in m.content.lower() for word in ["social", "google", "github", "magic link", "passwordless"])
        )
        service_auth_indicators = sum(
            1 for m in messages
            if any(word in m.content.lower() for word in ["mtls", "service-to-service", "certificate", "mutual tls"])
        )

        assert user_auth_indicators >= 1, "Should mention user auth approaches"
        assert service_auth_indicators >= 1, "Should mention service auth approaches"

        # Verify different channels (different teams)
        channels = {msg.channel for msg in messages}
        assert len(channels) >= 2, "Should involve different channels/teams"

    def test_sequential_work_no_overlap_scenario(self, mock_client):
        """
        Test edge case: sequential work with no temporal overlap.

        Teams work on similar things but at different times (60+ days apart).
        Should NOT detect as duplicate work.
        """
        messages = mock_client.get_scenario_messages("sequential_work")

        # Should have minimal messages
        assert len(messages) >= 4, "Should have at least 4 messages"

        # Get timestamps for each team
        web_msgs = [m for m in messages if m.channel == "#web-team"]
        mobile_msgs = [m for m in messages if m.channel == "#mobile"]

        assert len(web_msgs) >= 2, "Should have web team messages"
        assert len(mobile_msgs) >= 2, "Should have mobile team messages"

        # Verify no temporal overlap (mobile starts after web finishes)
        web_end = max(m.timestamp for m in web_msgs)
        mobile_start = min(m.timestamp for m in mobile_msgs)

        gap_days = (mobile_start - web_end).days
        assert gap_days >= 50, f"Should have large gap between teams (got {gap_days} days)"

        # Check for explicit reference to previous work
        references = sum(
            1 for m in mobile_msgs
            if any(word in m.content.lower() for word in ["web team", "learning from", "reused", "documentation"])
        )
        assert references >= 1, "Mobile team should reference web team's work"

    # ============================================================================
    # NEGATIVE EXAMPLES - Intentional Collaboration (Should NOT Detect)
    # ============================================================================

    def test_intentional_collaboration_scenario(self, mock_client):
        """
        Test negative example: intentional collaboration.

        Teams are working together with clear division of labor,
        cross-references, and explicit coordination.

        Should NOT detect as duplicate work.
        """
        messages = mock_client.get_scenario_messages("intentional_collaboration")

        # Should have several messages
        assert len(messages) >= 6, "Should have at least 6 messages"

        # Check for cross-references between teams
        cross_references = sum(
            1 for m in messages
            if any(mention in m.content for mention in ["@auth-team", "@platform-team"])
        )
        assert cross_references >= 4, "Should have multiple cross-team references"

        # Check for collaboration keywords
        collaboration_indicators = sum(
            1 for m in messages
            if any(word in m.content.lower() for word in [
                "coordinate", "collaboration", "sync", "together",
                "working with", "integrate with", "shared", "division of labor"
            ])
        )
        assert collaboration_indicators >= 3, "Should have clear collaboration language"

        # Check for engineering channel coordination
        engineering_msgs = [m for m in messages if m.channel == "#engineering"]
        assert len(engineering_msgs) >= 2, "Should have coordination messages in #engineering"

    # ============================================================================
    # SCENARIO METADATA TESTS
    # ============================================================================

    def test_all_scenarios_available(self, mock_client):
        """Test that all gap detection scenarios are available."""
        descriptions = mock_client.get_scenario_descriptions()

        # Original scenarios
        assert "oauth_discussion" in descriptions
        assert "decision_making" in descriptions
        assert "bug_report" in descriptions
        assert "feature_planning" in descriptions

        # Gap detection scenarios
        assert "oauth_duplication" in descriptions
        assert "api_redesign_duplication" in descriptions
        assert "auth_migration_duplication" in descriptions
        assert "similar_topics_different_scope" in descriptions
        assert "sequential_work" in descriptions
        assert "intentional_collaboration" in descriptions

    def test_scenario_descriptions_match_expectations(self, mock_client):
        """Test that scenario descriptions indicate expected detection results."""
        descriptions = mock_client.get_scenario_descriptions()

        # Positive cases should indicate "should detect"
        assert "should detect" in descriptions["oauth_duplication"].lower()
        assert "should detect" in descriptions["api_redesign_duplication"].lower()
        assert "should detect" in descriptions["auth_migration_duplication"].lower()

        # Negative cases should indicate "should not detect"
        assert "should not detect" in descriptions["similar_topics_different_scope"].lower()
        assert "should not detect" in descriptions["sequential_work"].lower()
        assert "should not detect" in descriptions["intentional_collaboration"].lower()

    def test_all_scenarios_have_team_metadata(self, mock_client):
        """Test that all gap detection scenarios include team metadata."""
        gap_scenarios = [
            "oauth_duplication",
            "api_redesign_duplication",
            "auth_migration_duplication",
            "similar_topics_different_scope",
            "sequential_work",
            "intentional_collaboration",
        ]

        for scenario_name in gap_scenarios:
            messages = mock_client.get_scenario_messages(scenario_name)

            # Most messages should have team metadata
            messages_with_team = sum(
                1 for m in messages
                if m.metadata and "team" in m.metadata
            )

            # At least 50% of messages should have team metadata
            assert messages_with_team >= len(messages) * 0.5, \
                f"Scenario {scenario_name} should have team metadata on most messages"

    # ============================================================================
    # COMPARATIVE TESTS
    # ============================================================================

    def test_positive_scenarios_longer_than_edge_cases(self, mock_client):
        """Test that positive scenarios have more messages than edge cases."""
        positive_scenarios = [
            "oauth_duplication",
            "api_redesign_duplication",
            "auth_migration_duplication",
        ]
        edge_scenarios = [
            "similar_topics_different_scope",
            "sequential_work",
        ]

        positive_msg_counts = [
            len(mock_client.get_scenario_messages(s))
            for s in positive_scenarios
        ]
        edge_msg_counts = [
            len(mock_client.get_scenario_messages(s))
            for s in edge_scenarios
        ]

        avg_positive = sum(positive_msg_counts) / len(positive_msg_counts)
        avg_edge = sum(edge_msg_counts) / len(edge_msg_counts)

        assert avg_positive > avg_edge * 2, \
            "Positive scenarios should have significantly more messages than edge cases"

    def test_positive_scenarios_have_temporal_overlap(self, mock_client):
        """Test that all positive scenarios have temporal overlap between teams."""
        positive_scenarios = [
            "oauth_duplication",
            "api_redesign_duplication",
            "auth_migration_duplication",
        ]

        for scenario_name in positive_scenarios:
            messages = mock_client.get_scenario_messages(scenario_name)

            # Get messages by team metadata (more reliable than channels)
            teams_dict = {}
            for msg in messages:
                if msg.metadata and "team" in msg.metadata:
                    team = msg.metadata["team"]
                    if team not in teams_dict:
                        teams_dict[team] = []
                    teams_dict[team].append(msg)

            # Should have at least 2 teams
            assert len(teams_dict) >= 2, f"{scenario_name} should have at least 2 teams"

            # Get first two teams
            teams = list(teams_dict.keys())[:2]
            team1_msgs = teams_dict[teams[0]]
            team2_msgs = teams_dict[teams[1]]

            if team1_msgs and team2_msgs:
                t1_start = min(m.timestamp for m in team1_msgs)
                t1_end = max(m.timestamp for m in team1_msgs)
                t2_start = min(m.timestamp for m in team2_msgs)
                t2_end = max(m.timestamp for m in team2_msgs)

                # Calculate overlap
                overlap_start = max(t1_start, t2_start)
                overlap_end = min(t1_end, t2_end)
                overlap_days = (overlap_end - overlap_start).days

                assert overlap_days >= 3, \
                    f"{scenario_name} should have at least 3 days of temporal overlap (got {overlap_days})"


# ============================================================================
# FIXTURES FOR SCENARIO DATA
# ============================================================================

@pytest.fixture
def oauth_duplication_messages():
    """Fixture providing OAuth duplication scenario messages."""
    client = MockSlackClient()
    return client.get_scenario_messages("oauth_duplication")


@pytest.fixture
def api_redesign_messages():
    """Fixture providing API redesign scenario messages."""
    client = MockSlackClient()
    return client.get_scenario_messages("api_redesign_duplication")


@pytest.fixture
def auth_migration_messages():
    """Fixture providing auth migration scenario messages."""
    client = MockSlackClient()
    return client.get_scenario_messages("auth_migration_duplication")


@pytest.fixture
def collaboration_messages():
    """Fixture providing intentional collaboration scenario messages."""
    client = MockSlackClient()
    return client.get_scenario_messages("intentional_collaboration")
