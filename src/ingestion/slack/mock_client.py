"""
Mock Slack client for generating realistic conversation data.

This module provides mock Slack conversation scenarios for development and testing.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List
from dataclasses import dataclass, field


@dataclass
class MockMessage:
    """Represents a mock Slack message."""

    content: str
    author: str
    channel: str
    timestamp: datetime
    thread_id: str | None = None
    external_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate external_id if not provided."""
        if not self.external_id:
            self.external_id = f"msg_{self.timestamp.timestamp()}"


class MockSlackClient:
    """
    Mock Slack client that generates realistic conversation scenarios.

    Scenarios include:
    1. OAuth Implementation Discussion - Technical thread with 8 messages
    2. Team Decision Thread - Decision-making with stakeholders
    3. Bug Report Discussion - Issue identification and resolution
    4. Feature Planning - Cross-team feature discussion
    """

    def __init__(self):
        """Initialize the mock client with predefined scenarios."""
        self.scenarios = {
            "oauth_discussion": self._generate_oauth_scenario,
            "decision_making": self._generate_decision_scenario,
            "bug_report": self._generate_bug_report_scenario,
            "feature_planning": self._generate_feature_planning_scenario,
        }

    def get_all_messages(self) -> List[MockMessage]:
        """Get all messages from all scenarios."""
        messages = []
        for scenario_name, generator in self.scenarios.items():
            messages.extend(generator())
        return messages

    def get_scenario_messages(self, scenario_name: str) -> List[MockMessage]:
        """Get messages from a specific scenario."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        return self.scenarios[scenario_name]()

    def _generate_oauth_scenario(self) -> List[MockMessage]:
        """
        Generate OAuth implementation discussion scenario.

        Scenario: Two teams (Platform and Auth) independently start working
        on OAuth2 implementation, creating a potential coordination gap.
        """
        base_time = datetime.utcnow() - timedelta(days=7)
        thread_id = "oauth_thread_001"

        messages = [
            MockMessage(
                content="Hey team, I'm starting work on OAuth2 integration for our API. "
                        "Planning to use the authorization code flow with PKCE. "
                        "Will have a PR ready by end of week.",
                author="alice@company.com",
                channel="#platform",
                timestamp=base_time,
                thread_id=thread_id,
                metadata={
                    "reactions": [{"emoji": "thumbsup", "count": 3}],
                    "mentions": []
                }
            ),
            MockMessage(
                content="@alice sounds good! Are you handling both the authorization "
                        "server and client library, or just the client side?",
                author="bob@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(minutes=15),
                thread_id=thread_id,
                metadata={
                    "mentions": ["alice@company.com"]
                }
            ),
            MockMessage(
                content="Just the client side for now. We'll integrate with an external "
                        "OAuth provider like Auth0 or Okta. Focusing on the token "
                        "management and refresh logic.",
                author="alice@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(minutes=30),
                thread_id=thread_id,
                metadata={
                    "reactions": [{"emoji": "eyes", "count": 2}]
                }
            ),
            MockMessage(
                content="Starting OAuth2 implementation today. We need this for the "
                        "mobile app authentication. Going with the standard authorization "
                        "code flow.",
                author="charlie@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(hours=4),
                thread_id="oauth_thread_002",
                metadata={
                    "reactions": [{"emoji": "rocket", "count": 2}]
                }
            ),
            MockMessage(
                content="@charlie are we building the full OAuth server or just the client?",
                author="diana@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(hours=4, minutes=20),
                thread_id="oauth_thread_002",
                metadata={
                    "mentions": ["charlie@company.com"]
                }
            ),
            MockMessage(
                content="Full stack - both server and client. We need to own the "
                        "authentication flow end-to-end for security compliance.",
                author="charlie@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(hours=4, minutes=35),
                thread_id="oauth_thread_002",
                metadata={
                    "reactions": [{"emoji": "lock", "count": 1}]
                }
            ),
            MockMessage(
                content="Good progress on OAuth2! Got the authorization endpoint working. "
                        "Token generation and validation next.",
                author="alice@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=2),
                thread_id=thread_id,
                metadata={
                    "reactions": [{"emoji": "fire", "count": 4}],
                    "attachments": [{"type": "code", "language": "python"}]
                }
            ),
            MockMessage(
                content="OAuth2 server implementation about 60% done. Authorization and "
                        "token endpoints are working. Need to add refresh token logic.",
                author="charlie@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(days=2, hours=3),
                thread_id="oauth_thread_002",
                metadata={
                    "reactions": [{"emoji": "muscle", "count": 3}]
                }
            ),
        ]

        return messages

    def _generate_decision_scenario(self) -> List[MockMessage]:
        """
        Generate team decision scenario.

        Scenario: Important architectural decision made without key stakeholders.
        """
        base_time = datetime.utcnow() - timedelta(days=5)
        thread_id = "decision_thread_001"

        messages = [
            MockMessage(
                content="We need to decide on the database for the new analytics service. "
                        "I'm leaning toward PostgreSQL for consistency with our other services.",
                author="evan@company.com",
                channel="#backend-eng",
                timestamp=base_time,
                thread_id=thread_id,
                metadata={"mentions": []}
            ),
            MockMessage(
                content="PostgreSQL makes sense, but have we considered the write volume? "
                        "Analytics can be pretty write-heavy.",
                author="fiona@company.com",
                channel="#backend-eng",
                timestamp=base_time + timedelta(minutes=20),
                thread_id=thread_id,
                metadata={"reactions": [{"emoji": "thinking_face", "count": 2}]}
            ),
            MockMessage(
                content="Good point. We're estimating ~10k writes/sec at peak. "
                        "PostgreSQL can handle that with proper indexing and partitioning.",
                author="evan@company.com",
                channel="#backend-eng",
                timestamp=base_time + timedelta(minutes=35),
                thread_id=thread_id,
                metadata={}
            ),
            MockMessage(
                content="Sounds good to me. Let's go with PostgreSQL. I'll create the "
                        "infrastructure tickets.",
                author="george@company.com",
                channel="#backend-eng",
                timestamp=base_time + timedelta(hours=1),
                thread_id=thread_id,
                metadata={
                    "reactions": [{"emoji": "thumbsup", "count": 5}],
                    "decision": True
                }
            ),
            MockMessage(
                content="Wait, I just saw the decision to use PostgreSQL for analytics. "
                        "Did anyone from the data team review this? We have specific "
                        "requirements for analytics queries that might be better served "
                        "by ClickHouse or TimescaleDB.",
                author="hannah@company.com",
                channel="#data-team",
                timestamp=base_time + timedelta(days=1),
                thread_id="decision_thread_002",
                metadata={
                    "reactions": [{"emoji": "eyes", "count": 3}],
                    "urgency": "high"
                }
            ),
        ]

        return messages

    def _generate_bug_report_scenario(self) -> List[MockMessage]:
        """
        Generate bug report and resolution scenario.

        Scenario: Bug reported, investigated, and resolved across multiple channels.
        """
        base_time = datetime.utcnow() - timedelta(days=3)
        thread_id = "bug_thread_001"

        messages = [
            MockMessage(
                content="🚨 Users are reporting 500 errors on the /api/users endpoint. "
                        "Started about 30 minutes ago. Checking logs now.",
                author="ian@company.com",
                channel="#incidents",
                timestamp=base_time,
                thread_id=thread_id,
                metadata={
                    "reactions": [{"emoji": "fire", "count": 5}],
                    "incident": True,
                    "severity": "high"
                }
            ),
            MockMessage(
                content="I see the errors in Sentry. Looks like a database connection pool "
                        "exhaustion. Connection count is at max (100).",
                author="jane@company.com",
                channel="#incidents",
                timestamp=base_time + timedelta(minutes=5),
                thread_id=thread_id,
                metadata={
                    "mentions": ["ian@company.com"],
                    "attachments": [{"type": "link", "url": "sentry.io/error/123"}]
                }
            ),
            MockMessage(
                content="Scaling up the connection pool to 200. Deploying now.",
                author="ian@company.com",
                channel="#incidents",
                timestamp=base_time + timedelta(minutes=10),
                thread_id=thread_id,
                metadata={"reactions": [{"emoji": "rocket", "count": 3}]}
            ),
            MockMessage(
                content="✅ Fix deployed. Error rate is back to normal. Will investigate "
                        "root cause - we shouldn't need that many connections.",
                author="ian@company.com",
                channel="#incidents",
                timestamp=base_time + timedelta(minutes=25),
                thread_id=thread_id,
                metadata={
                    "reactions": [{"emoji": "white_check_mark", "count": 6}],
                    "resolved": True
                }
            ),
            MockMessage(
                content="Found the root cause: recent change in user service is not "
                        "properly closing database connections. Opening a PR to fix.",
                author="jane@company.com",
                channel="#backend-eng",
                timestamp=base_time + timedelta(hours=2),
                thread_id="bug_thread_002",
                metadata={
                    "references": ["incidents/bug_thread_001"],
                    "attachments": [{"type": "pr", "url": "github.com/company/pr/456"}]
                }
            ),
        ]

        return messages

    def _generate_feature_planning_scenario(self) -> List[MockMessage]:
        """
        Generate cross-team feature planning scenario.

        Scenario: Multiple teams discussing a new feature that requires coordination.
        """
        base_time = datetime.utcnow() - timedelta(days=10)
        thread_id = "feature_thread_001"

        messages = [
            MockMessage(
                content="Proposal: Add real-time notifications for user mentions and replies. "
                        "This would significantly improve user engagement. Thoughts?",
                author="kate@company.com",
                channel="#product",
                timestamp=base_time,
                thread_id=thread_id,
                metadata={
                    "reactions": [{"emoji": "eyes", "count": 8}],
                    "proposal": True
                }
            ),
            MockMessage(
                content="Love this idea! From a backend perspective, we'd probably use "
                        "WebSockets for the real-time connection. Would need to think "
                        "about scaling and connection management.",
                author="liam@company.com",
                channel="#product",
                timestamp=base_time + timedelta(hours=1),
                thread_id=thread_id,
                metadata={
                    "mentions": ["kate@company.com"],
                    "reactions": [{"emoji": "thinking_face", "count": 3}]
                }
            ),
            MockMessage(
                content="On mobile, we'd need to handle background notifications. "
                        "Push notifications via FCM/APNS when app is backgrounded, "
                        "WebSocket when active.",
                author="maria@company.com",
                channel="#product",
                timestamp=base_time + timedelta(hours=2),
                thread_id=thread_id,
                metadata={
                    "reactions": [{"emoji": "iphone", "count": 2}]
                }
            ),
            MockMessage(
                content="Let's set up a meeting to align on the technical approach. "
                        "We need backend, mobile, and web teams all on the same page.",
                author="kate@company.com",
                channel="#product",
                timestamp=base_time + timedelta(hours=3),
                thread_id=thread_id,
                metadata={
                    "mentions": ["liam@company.com", "maria@company.com"],
                    "meeting_scheduled": True
                }
            ),
            MockMessage(
                content="Quick update: Backend team is starting work on the WebSocket "
                        "infrastructure for real-time notifications. Should be ready "
                        "for testing in 2 weeks.",
                author="liam@company.com",
                channel="#engineering",
                timestamp=base_time + timedelta(days=3),
                thread_id="feature_thread_002",
                metadata={
                    "reactions": [{"emoji": "rocket", "count": 5}],
                    "progress_update": True
                }
            ),
            MockMessage(
                content="Frontend team is also working on the notification UI components. "
                        "Should integrate nicely with the WebSocket backend.",
                author="nathan@company.com",
                channel="#frontend",
                timestamp=base_time + timedelta(days=3, hours=2),
                thread_id="feature_thread_003",
                metadata={
                    "reactions": [{"emoji": "art", "count": 3}]
                }
            ),
        ]

        return messages

    def get_scenario_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available scenarios."""
        return {
            "oauth_discussion": "OAuth implementation discussion - potential duplicate work",
            "decision_making": "Team decision without key stakeholders",
            "bug_report": "Bug report and resolution workflow",
            "feature_planning": "Cross-team feature planning coordination",
        }
