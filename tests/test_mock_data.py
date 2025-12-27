"""
Tests for mock data generation.
"""
import pytest
from datetime import datetime, timedelta

from src.ingestion.slack.mock_client import MockSlackClient, MockMessage


class TestMockMessage:
    """Tests for MockMessage dataclass."""

    def test_mock_message_creation(self):
        """Test basic mock message creation."""
        msg = MockMessage(
            content="Test message",
            author="test@example.com",
            channel="#test",
            timestamp=datetime.utcnow(),
        )

        assert msg.content == "Test message"
        assert msg.author == "test@example.com"
        assert msg.channel == "#test"
        assert msg.thread_id is None
        assert msg.external_id != ""  # Should be auto-generated

    def test_mock_message_with_thread(self):
        """Test mock message with thread_id."""
        msg = MockMessage(
            content="Reply message",
            author="test@example.com",
            channel="#test",
            timestamp=datetime.utcnow(),
            thread_id="thread_123",
        )

        assert msg.thread_id == "thread_123"

    def test_mock_message_with_metadata(self):
        """Test mock message with metadata."""
        metadata = {
            "reactions": [{"emoji": "thumbsup", "count": 3}],
            "mentions": ["user@example.com"],
        }

        msg = MockMessage(
            content="Test message",
            author="test@example.com",
            channel="#test",
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )

        assert msg.metadata == metadata
        assert "reactions" in msg.metadata
        assert msg.metadata["reactions"][0]["emoji"] == "thumbsup"

    def test_external_id_generation(self):
        """Test that external_id is auto-generated from timestamp."""
        timestamp = datetime.utcnow()
        msg = MockMessage(
            content="Test message",
            author="test@example.com",
            channel="#test",
            timestamp=timestamp,
        )

        assert msg.external_id.startswith("msg_")
        assert str(timestamp.timestamp()) in msg.external_id


class TestMockSlackClient:
    """Tests for MockSlackClient."""

    def test_client_initialization(self):
        """Test that client initializes with all scenarios."""
        client = MockSlackClient()

        # Original scenarios
        original_scenarios = [
            "oauth_discussion",
            "decision_making",
            "bug_report",
            "feature_planning",
        ]

        # Gap detection scenarios (added in Milestone 3G)
        gap_detection_scenarios = [
            "oauth_duplication",
            "api_redesign_duplication",
            "auth_migration_duplication",
            "similar_topics_different_scope",
            "sequential_work",
            "intentional_collaboration",
        ]

        all_scenarios = original_scenarios + gap_detection_scenarios

        assert set(client.scenarios.keys()) == set(all_scenarios)
        assert len(client.scenarios) == 10  # 4 original + 6 gap detection

    def test_get_all_messages(self):
        """Test getting all messages from all scenarios."""
        client = MockSlackClient()
        messages = client.get_all_messages()

        assert len(messages) > 0
        assert all(isinstance(msg, MockMessage) for msg in messages)

        # Should have messages from all scenarios
        # Original: OAuth: 8, Decision: 5, Bug: 5, Feature: 6 = 24 total
        # Gap detection: oauth_dup: 24, api_redesign: 18, auth_mig: 10,
        #                different_scope: 4, sequential: 4, collab: 6 = 66 total
        # Grand total: 24 + 66 = 90 messages (approximately, may vary by 1-2)
        assert len(messages) >= 90
        assert len(messages) <= 95  # Allow some variance

    def test_get_scenario_messages(self):
        """Test getting messages from specific scenarios."""
        client = MockSlackClient()

        # Test each scenario
        oauth_msgs = client.get_scenario_messages("oauth_discussion")
        assert len(oauth_msgs) == 8
        assert all(msg.channel in ["#platform", "#auth-team"] for msg in oauth_msgs)

        decision_msgs = client.get_scenario_messages("decision_making")
        assert len(decision_msgs) == 5
        assert all(msg.channel in ["#backend-eng", "#data-team"] for msg in decision_msgs)

        bug_msgs = client.get_scenario_messages("bug_report")
        assert len(bug_msgs) == 5
        assert all(msg.channel in ["#incidents", "#backend-eng"] for msg in bug_msgs)

        feature_msgs = client.get_scenario_messages("feature_planning")
        assert len(feature_msgs) == 6
        assert all(
            msg.channel in ["#product", "#engineering", "#frontend"] for msg in feature_msgs
        )

    def test_get_unknown_scenario(self):
        """Test that getting unknown scenario raises error."""
        client = MockSlackClient()

        with pytest.raises(ValueError, match="Unknown scenario"):
            client.get_scenario_messages("nonexistent_scenario")

    def test_scenario_descriptions(self):
        """Test getting scenario descriptions."""
        client = MockSlackClient()
        descriptions = client.get_scenario_descriptions()

        # Should have 10 scenarios total (4 original + 6 gap detection)
        assert len(descriptions) == 10

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

        # Check that descriptions are meaningful
        assert len(descriptions["oauth_discussion"]) > 10
        assert len(descriptions["oauth_duplication"]) > 10


class TestOAuthScenario:
    """Tests specific to OAuth discussion scenario."""

    def test_oauth_scenario_structure(self):
        """Test OAuth scenario has correct structure."""
        client = MockSlackClient()
        messages = client.get_scenario_messages("oauth_discussion")

        # Check message count
        assert len(messages) == 8

        # Check channels
        channels = set(msg.channel for msg in messages)
        assert channels == {"#platform", "#auth-team"}

        # Check threading
        thread_ids = set(msg.thread_id for msg in messages if msg.thread_id)
        assert len(thread_ids) == 2  # Two separate threads

        # Check authors (multiple participants)
        authors = set(msg.author for msg in messages)
        assert len(authors) >= 4  # At least 4 different participants

    def test_oauth_scenario_temporal_pattern(self):
        """Test OAuth scenario has realistic temporal patterns."""
        client = MockSlackClient()
        messages = client.get_scenario_messages("oauth_discussion")

        # Messages should be ordered by timestamp
        timestamps = [msg.timestamp for msg in messages]
        assert timestamps == sorted(timestamps)

        # Should span multiple days
        time_span = messages[-1].timestamp - messages[0].timestamp
        assert time_span.days >= 2

    def test_oauth_scenario_metadata(self):
        """Test OAuth scenario has rich metadata."""
        client = MockSlackClient()
        messages = client.get_scenario_messages("oauth_discussion")

        # At least some messages should have reactions
        messages_with_reactions = [
            msg for msg in messages if msg.metadata and "reactions" in msg.metadata
        ]
        assert len(messages_with_reactions) > 0

        # At least some messages should have mentions
        messages_with_mentions = [
            msg for msg in messages if msg.metadata and "mentions" in msg.metadata
        ]
        assert len(messages_with_mentions) > 0


class TestDecisionScenario:
    """Tests specific to decision-making scenario."""

    def test_decision_scenario_shows_missing_stakeholder(self):
        """Test decision scenario demonstrates missing stakeholder pattern."""
        client = MockSlackClient()
        messages = client.get_scenario_messages("decision_making")

        # Decision made in #backend-eng
        decision_messages = [msg for msg in messages if msg.channel == "#backend-eng"]
        assert len(decision_messages) >= 4

        # Later reaction from #data-team shows they were missing
        data_team_messages = [msg for msg in messages if msg.channel == "#data-team"]
        assert len(data_team_messages) >= 1

        # Data team message should come after decision
        last_decision_time = max(msg.timestamp for msg in decision_messages)
        data_team_time = data_team_messages[0].timestamp
        assert data_team_time > last_decision_time


class TestBugReportScenario:
    """Tests specific to bug report scenario."""

    def test_bug_report_has_incident_metadata(self):
        """Test bug report scenario includes incident metadata."""
        client = MockSlackClient()
        messages = client.get_scenario_messages("bug_report")

        # First message should be marked as incident
        first_message = messages[0]
        assert first_message.metadata.get("incident") is True
        assert first_message.metadata.get("severity") == "high"

        # Should have a resolution marker
        resolved_messages = [
            msg for msg in messages if msg.metadata.get("resolved") is True
        ]
        assert len(resolved_messages) > 0


class TestFeaturePlanningScenario:
    """Tests specific to feature planning scenario."""

    def test_feature_planning_cross_team(self):
        """Test feature planning involves multiple teams."""
        client = MockSlackClient()
        messages = client.get_scenario_messages("feature_planning")

        # Should span multiple channels (different teams)
        channels = set(msg.channel for msg in messages)
        assert len(channels) >= 3  # At least 3 different channels

        # Should have multiple participants
        authors = set(msg.author for msg in messages)
        assert len(authors) >= 4

    def test_feature_planning_has_coordination_signals(self):
        """Test feature planning shows coordination signals."""
        client = MockSlackClient()
        messages = client.get_scenario_messages("feature_planning")

        # Should have meeting scheduling
        meeting_messages = [
            msg for msg in messages if msg.metadata.get("meeting_scheduled")
        ]
        assert len(meeting_messages) > 0

        # Should have progress updates
        progress_messages = [
            msg for msg in messages if msg.metadata.get("progress_update")
        ]
        assert len(progress_messages) > 0
