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

    # Scenarios that work with currently implemented DUPLICATE_WORK detector
    DUPLICATE_WORK_SCENARIOS = [
        "oauth_duplication",
        "api_redesign_duplication",
        "auth_migration_duplication",
        "similar_topics_different_scope",  # Negative example
        "sequential_work",  # Negative example
        "intentional_collaboration",  # Negative example
    ]

    def __init__(self):
        """Initialize the mock client with predefined scenarios."""
        self.scenarios = {
            "oauth_discussion": self._generate_oauth_scenario,
            "decision_making": self._generate_decision_scenario,
            "bug_report": self._generate_bug_report_scenario,
            "feature_planning": self._generate_feature_planning_scenario,
            # Gap detection scenarios
            "oauth_duplication": self._generate_oauth_duplication_scenario,
            "api_redesign_duplication": self._generate_api_redesign_scenario,
            "auth_migration_duplication": self._generate_auth_migration_scenario,
            "similar_topics_different_scope": self._generate_edge_case_different_scope,
            "sequential_work": self._generate_edge_case_sequential,
            "intentional_collaboration": self._generate_collaboration_scenario,
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

    def get_duplicate_work_scenarios(self) -> Dict[str, Any]:
        """
        Get only scenarios that work with the DUPLICATE_WORK detector.

        Returns scenarios designed for duplicate work detection, including
        both positive examples (should detect gaps) and negative examples
        (should NOT detect gaps).

        Returns:
            Dictionary mapping scenario names to generator functions
        """
        return {
            key: self.scenarios[key]
            for key in self.DUPLICATE_WORK_SCENARIOS
        }

    def get_duplicate_work_scenario_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for DUPLICATE_WORK scenarios only.

        Filters the full scenario description list to only include scenarios
        that work with the currently implemented DUPLICATE_WORK detector.

        Returns:
            Dictionary mapping scenario names to their descriptions
        """
        all_descriptions = self.get_scenario_descriptions()
        return {
            key: all_descriptions[key]
            for key in self.DUPLICATE_WORK_SCENARIOS
        }

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

    def _generate_oauth_duplication_scenario(self) -> List[MockMessage]:
        """
        Generate enhanced OAuth duplication scenario for gap detection.

        Scenario: Platform and Auth teams independently implementing OAuth2
        with 24+ messages showing clear parallel work over 14 days.

        Expected: Should detect as duplicate work with HIGH impact.
        """
        base_time = datetime.utcnow() - timedelta(days=15)
        platform_thread = "oauth_platform_001"
        auth_thread = "oauth_auth_001"

        messages = [
            # Day 1 - Both teams start independently
            MockMessage(
                content="Starting OAuth2 implementation for our API gateway today. "
                        "Planning to use authorization code flow with PKCE for security.",
                author="alice@company.com",
                channel="#platform",
                timestamp=base_time,
                thread_id=platform_thread,
                metadata={"team": "platform-team", "reactions": [{"emoji": "thumbsup", "count": 3}]}
            ),
            MockMessage(
                content="We're building OAuth support for the new auth service. "
                        "Going with standard authorization code flow.",
                author="bob@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(hours=4, minutes=20),
                thread_id=auth_thread,
                metadata={"team": "auth-team", "reactions": [{"emoji": "rocket", "count": 2}]}
            ),

            # Day 1 continued - Team discussions
            MockMessage(
                content="@alice are we handling both the authorization server and client library?",
                author="charlie@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(hours=1),
                thread_id=platform_thread,
                metadata={"team": "platform-team", "mentions": ["alice@company.com"]}
            ),
            MockMessage(
                content="Just the client side for now - we'll integrate with external providers "
                        "like Auth0 or Okta. Focus is on token management and refresh logic.",
                author="alice@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(hours=1, minutes=15),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),
            MockMessage(
                content="@bob full server implementation or just client integration?",
                author="diana@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(hours=4, minutes=45),
                thread_id=auth_thread,
                metadata={"team": "auth-team", "mentions": ["bob@company.com"]}
            ),
            MockMessage(
                content="Full OAuth server - we need to own the authentication flow "
                        "end-to-end for compliance. Building authorization and token endpoints.",
                author="bob@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(hours=5),
                thread_id=auth_thread,
                metadata={"team": "auth-team", "reactions": [{"emoji": "lock", "count": 2}]}
            ),

            # Day 3 - Design decisions
            MockMessage(
                content="Decided on authorization code flow for OAuth. Researched the spec "
                        "and best practices. Starting implementation of the token exchange.",
                author="alice@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=2, hours=10),
                thread_id=platform_thread,
                metadata={"team": "platform-team", "reactions": [{"emoji": "books", "count": 2}]}
            ),
            MockMessage(
                content="After reviewing OAuth 2.1 spec, going with authorization code flow. "
                        "It's the most secure option for our use case.",
                author="bob@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(days=2, hours=15),
                thread_id=auth_thread,
                metadata={"team": "auth-team"}
            ),

            # Day 4 - Technical implementation details
            MockMessage(
                content="Working on the token endpoint. Using JWT for access tokens with "
                        "1-hour expiry and refresh tokens with 30-day expiry.",
                author="charlie@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=3, hours=9),
                thread_id=platform_thread,
                metadata={"team": "platform-team", "reactions": [{"emoji": "code", "count": 3}]}
            ),
            MockMessage(
                content="Implementing token generation - using JWT with RS256 signing. "
                        "Access tokens expire in 1 hour, refresh tokens in 30 days.",
                author="diana@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(days=3, hours=14),
                thread_id=auth_thread,
                metadata={"team": "auth-team"}
            ),

            # Day 5 - Progress updates
            MockMessage(
                content="Good progress on OAuth2! Got the authorization endpoint working. "
                        "Token generation and validation next. About 40% complete.",
                author="alice@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=4, hours=11),
                thread_id=platform_thread,
                metadata={"team": "platform-team", "reactions": [{"emoji": "fire", "count": 4}]}
            ),
            MockMessage(
                content="OAuth2 server implementation about 50% done. Authorization and "
                        "token endpoints are working. Need to add refresh token rotation.",
                author="bob@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(days=4, hours=16),
                thread_id=auth_thread,
                metadata={"team": "auth-team", "reactions": [{"emoji": "muscle", "count": 3}]}
            ),

            # Day 7 - More technical details
            MockMessage(
                content="Implementing PKCE (Proof Key for Code Exchange) for added security. "
                        "This prevents authorization code interception attacks.",
                author="charlie@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=6, hours=10),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),
            MockMessage(
                content="Adding PKCE support to our OAuth implementation. It's required for "
                        "public clients and adds good security for confidential clients too.",
                author="diana@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(days=6, hours=13),
                thread_id=auth_thread,
                metadata={"team": "auth-team"}
            ),

            # Day 8 - Database and storage
            MockMessage(
                content="Setting up Postgres tables for OAuth tokens and authorization codes. "
                        "Using TTL-based cleanup for expired tokens.",
                author="alice@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=7, hours=9),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),
            MockMessage(
                content="Database schema ready for OAuth - tables for clients, tokens, "
                        "and auth codes. Implementing automatic cleanup for expired entries.",
                author="bob@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(days=7, hours=14),
                thread_id=auth_thread,
                metadata={"team": "auth-team"}
            ),

            # Day 10 - Testing
            MockMessage(
                content="Testing OAuth flow end-to-end. Authorization → token exchange → "
                        "refresh → revocation. All working smoothly. 70% complete.",
                author="charlie@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=9, hours=11),
                thread_id=platform_thread,
                metadata={"team": "platform-team", "reactions": [{"emoji": "white_check_mark", "count": 5}]}
            ),
            MockMessage(
                content="OAuth server passing all test cases. Authorization flow, token "
                        "refresh, and revocation all working. About 75% done overall.",
                author="diana@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(days=9, hours=15),
                thread_id=auth_thread,
                metadata={"team": "auth-team", "reactions": [{"emoji": "tada", "count": 4}]}
            ),

            # Day 11 - Scope and permissions
            MockMessage(
                content="Adding scope validation for OAuth tokens. Users can grant different "
                        "permission levels (read, write, admin) to client applications.",
                author="alice@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=10, hours=10),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),
            MockMessage(
                content="Implementing OAuth scopes - read, write, admin. Clients request "
                        "specific scopes and users approve during authorization.",
                author="bob@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(days=10, hours=16),
                thread_id=auth_thread,
                metadata={"team": "auth-team"}
            ),

            # Day 12 - Integration work
            MockMessage(
                content="Integrating OAuth with our API gateway. All endpoints now check "
                        "for valid OAuth tokens. Legacy API keys still supported for now.",
                author="charlie@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=11, hours=14),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),
            MockMessage(
                content="Wiring up OAuth to the user service. Authentication middleware "
                        "validates tokens and extracts user identity. Almost done!",
                author="diana@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(days=11, hours=17),
                thread_id=auth_thread,
                metadata={"team": "auth-team"}
            ),

            # Day 14 - Discovery moment
            MockMessage(
                content="Almost ready to deploy OAuth to staging! This will enable third-party "
                        "integrations with our API. Super excited about this!",
                author="alice@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=13, hours=10),
                thread_id=platform_thread,
                metadata={"team": "platform-team", "reactions": [{"emoji": "rocket", "count": 6}]}
            ),
            MockMessage(
                content="OAuth server is production-ready! Final security review tomorrow, "
                        "then we can roll out. This opens up our platform for integrations.",
                author="bob@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(days=13, hours=15),
                thread_id=auth_thread,
                metadata={"team": "auth-team", "reactions": [{"emoji": "fire", "count": 5}]}
            ),
            MockMessage(
                content="Wait... I just saw messages in both #platform and #auth-team about OAuth. "
                        "@alice @bob are both teams working on OAuth implementations? 🤔",
                author="eve@company.com",
                channel="#engineering",
                timestamp=base_time + timedelta(days=13, hours=18),
                thread_id="oauth_discovery_001",
                metadata={
                    "mentions": ["alice@company.com", "bob@company.com"],
                    "reactions": [{"emoji": "eyes", "count": 8}]
                }
            ),
        ]

        return messages

    def _generate_api_redesign_scenario(self) -> List[MockMessage]:
        """
        Generate API redesign duplication scenario.

        Scenario: Platform and Backend teams independently redesigning REST API
        structure, causing duplicate architectural work.

        Expected: Should detect as duplicate work with HIGH impact.
        """
        base_time = datetime.utcnow() - timedelta(days=21)
        platform_thread = "api_platform_001"
        backend_thread = "api_backend_001"

        messages = [
            # Platform team starts
            MockMessage(
                content="We need to redesign our REST API structure. Current design has "
                        "inconsistent naming and resource hierarchies. Starting spec work.",
                author="frank@company.com",
                channel="#platform",
                timestamp=base_time,
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),
            MockMessage(
                content="Looking at RESTful best practices. Planning to standardize on "
                        "plural resource names and proper HTTP verbs (GET/POST/PUT/DELETE).",
                author="frank@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(hours=3),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),

            # Backend team starts (same day, different time)
            MockMessage(
                content="Starting API restructuring project. Our endpoints are all over the place. "
                        "Need consistent naming, versioning, and resource patterns.",
                author="grace@company.com",
                channel="#backend-eng",
                timestamp=base_time + timedelta(hours=6),
                thread_id=backend_thread,
                metadata={"team": "backend-team"}
            ),
            MockMessage(
                content="Creating new API design guidelines. Standardizing on REST principles - "
                        "proper resource naming, HTTP methods, and status codes.",
                author="grace@company.com",
                channel="#backend-eng",
                timestamp=base_time + timedelta(hours=7),
                thread_id=backend_thread,
                metadata={"team": "backend-team"}
            ),

            # Day 3 - Both teams working on specs
            MockMessage(
                content="API redesign spec is coming together. Example: POST /api/v2/users "
                        "instead of /create_user. Much cleaner and more RESTful.",
                author="frank@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=2, hours=10),
                thread_id=platform_thread,
                metadata={"team": "platform-team", "reactions": [{"emoji": "sparkles", "count": 3}]}
            ),
            MockMessage(
                content="Draft API spec ready. New structure: POST /api/v2/users, "
                        "GET /api/v2/users/:id, etc. Following REST conventions.",
                author="grace@company.com",
                channel="#backend-eng",
                timestamp=base_time + timedelta(days=2, hours=14),
                thread_id=backend_thread,
                metadata={"team": "backend-team"}
            ),

            # Day 5 - Versioning strategies
            MockMessage(
                content="Implementing API versioning. Going with URL versioning (v1, v2) "
                        "rather than headers - simpler for clients to understand.",
                author="henry@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=4, hours=11),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),
            MockMessage(
                content="Decided on URL-based versioning (/v1, /v2) for the new API. "
                        "Easier for developers than header-based versioning.",
                author="grace@company.com",
                channel="#backend-eng",
                timestamp=base_time + timedelta(days=4, hours=15),
                thread_id=backend_thread,
                metadata={"team": "backend-team"}
            ),

            # Day 7 - Error handling
            MockMessage(
                content="Standardizing error responses. All errors return JSON with "
                        "'error', 'message', and 'details' fields. Proper HTTP status codes.",
                author="frank@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=6, hours=9),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),
            MockMessage(
                content="New error response format: {error: 'code', message: 'description', "
                        "details: {...}}. Consistent across all endpoints.",
                author="henry@company.com",
                channel="#backend-eng",
                timestamp=base_time + timedelta(days=6, hours=13),
                thread_id=backend_thread,
                metadata={"team": "backend-team"}
            ),

            # Day 10 - Pagination
            MockMessage(
                content="Adding pagination to all list endpoints. Using cursor-based "
                        "pagination with 'limit' and 'cursor' query parameters.",
                author="frank@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=9, hours=10),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),
            MockMessage(
                content="Implementing cursor pagination for list endpoints. More efficient "
                        "than offset-based for large datasets.",
                author="grace@company.com",
                channel="#backend-eng",
                timestamp=base_time + timedelta(days=9, hours=16),
                thread_id=backend_thread,
                metadata={"team": "backend-team"}
            ),

            # Day 12 - Documentation
            MockMessage(
                content="Writing OpenAPI specs for the new API design. Interactive docs "
                        "will be available at /api/docs. 60% complete overall.",
                author="henry@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=11, hours=14),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),
            MockMessage(
                content="Generating OpenAPI documentation. Setting up Swagger UI for "
                        "interactive API exploration. Making good progress.",
                author="henry@company.com",
                channel="#backend-eng",
                timestamp=base_time + timedelta(days=11, hours=17),
                thread_id=backend_thread,
                metadata={"team": "backend-team"}
            ),

            # Day 15 - Rate limiting
            MockMessage(
                content="Adding rate limiting to the new API. 1000 requests/hour for "
                        "authenticated users, 100/hour for anonymous.",
                author="frank@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=14, hours=11),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),
            MockMessage(
                content="Implementing rate limits: 1000 req/hr for auth users, "
                        "100 req/hr for public. Using Redis for tracking.",
                author="grace@company.com",
                channel="#backend-eng",
                timestamp=base_time + timedelta(days=14, hours=15),
                thread_id=backend_thread,
                metadata={"team": "backend-team"}
            ),

            # Day 18 - Testing
            MockMessage(
                content="API redesign testing complete. All endpoints follow new conventions. "
                        "Ready for internal review. Great work team!",
                author="frank@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=17, hours=16),
                thread_id=platform_thread,
                metadata={"team": "platform-team", "reactions": [{"emoji": "tada", "count": 5}]}
            ),
            MockMessage(
                content="New API structure is ready for review. Comprehensive test coverage. "
                        "This is going to make our API so much better!",
                author="grace@company.com",
                channel="#backend-eng",
                timestamp=base_time + timedelta(days=17, hours=18),
                thread_id=backend_thread,
                metadata={"team": "backend-team", "reactions": [{"emoji": "rocket", "count": 4}]}
            ),
        ]

        return messages

    def _generate_auth_migration_scenario(self) -> List[MockMessage]:
        """
        Generate auth migration duplication scenario.

        Scenario: Security and Platform teams both migrating from legacy
        auth to modern JWT-based system independently.

        Expected: Should detect as duplicate work with MEDIUM-HIGH impact.
        """
        base_time = datetime.utcnow() - timedelta(days=11)
        security_thread = "auth_security_001"
        platform_thread = "auth_platform_001"

        messages = [
            # Day 1
            MockMessage(
                content="Starting migration from legacy session-based auth to JWT tokens. "
                        "Current system is causing scaling issues with sticky sessions.",
                author="iris@company.com",
                channel="#security",
                timestamp=base_time,
                thread_id=security_thread,
                metadata={"team": "security-team"}
            ),
            MockMessage(
                content="Planning to migrate our authentication to JWT. Session-based "
                        "auth is becoming a bottleneck for horizontal scaling.",
                author="jack@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(hours=5),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),

            # Day 3
            MockMessage(
                content="Researching JWT best practices. Planning to use RS256 (asymmetric) "
                        "instead of HS256 for better security and key rotation.",
                author="iris@company.com",
                channel="#security",
                timestamp=base_time + timedelta(days=2, hours=10),
                thread_id=security_thread,
                metadata={"team": "security-team"}
            ),
            MockMessage(
                content="Going with RS256 for JWT signing. Allows us to verify tokens "
                        "without sharing secret keys across services.",
                author="jack@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=2, hours=14),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),

            # Day 5
            MockMessage(
                content="Implementing JWT generation and validation middleware. "
                        "Tokens contain user ID, roles, and expiry (15 min for access tokens).",
                author="iris@company.com",
                channel="#security",
                timestamp=base_time + timedelta(days=4, hours=9),
                thread_id=security_thread,
                metadata={"team": "security-team"}
            ),
            MockMessage(
                content="JWT middleware ready. Access tokens expire in 15 minutes, "
                        "refresh tokens in 7 days. Including user claims in payload.",
                author="jack@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=4, hours=15),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),

            # Day 7
            MockMessage(
                content="Building migration path: existing sessions will be gradually "
                        "replaced with JWTs. Supporting both during transition period.",
                author="kelly@company.com",
                channel="#security",
                timestamp=base_time + timedelta(days=6, hours=11),
                thread_id=security_thread,
                metadata={"team": "security-team"}
            ),
            MockMessage(
                content="Migration strategy: run both auth systems in parallel for 2 weeks, "
                        "then deprecate legacy sessions. Gradual cutover.",
                author="jack@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=6, hours=16),
                thread_id=platform_thread,
                metadata={"team": "platform-team"}
            ),

            # Day 9
            MockMessage(
                content="JWT auth system testing complete. Significantly better performance "
                        "than session-based. Ready for staged rollout.",
                author="iris@company.com",
                channel="#security",
                timestamp=base_time + timedelta(days=8, hours=14),
                thread_id=security_thread,
                metadata={"team": "security-team", "reactions": [{"emoji": "fire", "count": 4}]}
            ),
            MockMessage(
                content="Finished JWT implementation and testing. Performance is great - "
                        "no more sticky session issues. Ready to deploy!",
                author="kelly@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=8, hours=17),
                thread_id=platform_thread,
                metadata={"team": "platform-team", "reactions": [{"emoji": "rocket", "count": 3}]}
            ),
        ]

        return messages

    def _generate_edge_case_different_scope(self) -> List[MockMessage]:
        """
        Generate edge case: similar topics but different scope.

        Scenario: Two teams discussing "authentication" but completely
        different aspects (user auth vs service-to-service auth).

        Expected: Should NOT detect as duplicate work.
        """
        base_time = datetime.utcnow() - timedelta(days=8)

        messages = [
            # User authentication team
            MockMessage(
                content="Working on user authentication improvements. Adding support for "
                        "social logins (Google, GitHub) and magic links for passwordless auth.",
                author="laura@company.com",
                channel="#frontend",
                timestamp=base_time,
                thread_id="auth_user_001",
                metadata={"team": "frontend-team"}
            ),
            MockMessage(
                content="User auth improvements looking good. OAuth flows for Google and "
                        "GitHub are working. Magic link emails going out successfully.",
                author="laura@company.com",
                channel="#frontend",
                timestamp=base_time + timedelta(days=3),
                thread_id="auth_user_001",
                metadata={"team": "frontend-team"}
            ),

            # Service-to-service authentication team
            MockMessage(
                content="Implementing service-to-service authentication for our microservices. "
                        "Using mTLS (mutual TLS) for secure service communication.",
                author="mike@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(hours=2),
                thread_id="auth_service_001",
                metadata={"team": "platform-team"}
            ),
            MockMessage(
                content="mTLS setup complete for inter-service auth. Services now verify "
                        "each other's certificates. Much more secure than API keys.",
                author="mike@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=3, hours=4),
                thread_id="auth_service_001",
                metadata={"team": "platform-team"}
            ),
        ]

        return messages

    def _generate_edge_case_sequential(self) -> List[MockMessage]:
        """
        Generate edge case: sequential work (not parallel).

        Scenario: Two teams working on OAuth but at different times
        (no temporal overlap).

        Expected: Should NOT detect as duplicate work.
        """
        base_time = datetime.utcnow() - timedelta(days=90)

        messages = [
            # Team A works in November
            MockMessage(
                content="Starting OAuth implementation for our web app. Authorization "
                        "code flow with Auth0 integration.",
                author="nancy@company.com",
                channel="#web-team",
                timestamp=base_time,
                thread_id="oauth_web_001",
                metadata={"team": "web-team"}
            ),
            MockMessage(
                content="OAuth integration complete! Users can now sign in with Google "
                        "and GitHub. Great success! 🎉",
                author="nancy@company.com",
                channel="#web-team",
                timestamp=base_time + timedelta(days=10),
                thread_id="oauth_web_001",
                metadata={"team": "web-team", "reactions": [{"emoji": "tada", "count": 5}]}
            ),

            # Team B works 60 days later in January
            MockMessage(
                content="Starting OAuth for our mobile app. Learning from the web team's "
                        "implementation. Should be straightforward.",
                author="olivia@company.com",
                channel="#mobile",
                timestamp=base_time + timedelta(days=70),
                thread_id="oauth_mobile_001",
                metadata={"team": "mobile-team"}
            ),
            MockMessage(
                content="Mobile OAuth done! Reused a lot of the patterns from web team. "
                        "Thanks for the great documentation @nancy!",
                author="olivia@company.com",
                channel="#mobile",
                timestamp=base_time + timedelta(days=77),
                thread_id="oauth_mobile_001",
                metadata={
                    "team": "mobile-team",
                    "mentions": ["nancy@company.com"],
                    "reactions": [{"emoji": "heart", "count": 3}]
                }
            ),
        ]

        return messages

    def _generate_collaboration_scenario(self) -> List[MockMessage]:
        """
        Generate collaboration scenario (negative example).

        Scenario: Platform and Auth teams intentionally collaborating on OAuth,
        with clear division of labor and cross-references.

        Expected: Should NOT detect as duplicate work.
        """
        base_time = datetime.utcnow() - timedelta(days=12)
        collab_thread = "oauth_collab_001"

        messages = [
            # Initial coordination
            MockMessage(
                content="@auth-team and @platform-team: Let's coordinate on OAuth implementation. "
                        "Auth team can handle the server, Platform handles client integration. "
                        "Weekly syncs to stay aligned.",
                author="paul@company.com",
                channel="#engineering",
                timestamp=base_time,
                thread_id=collab_thread,
                metadata={
                    "mentions": ["auth-team", "platform-team"],
                    "reactions": [{"emoji": "handshake", "count": 8}]
                }
            ),

            # Platform team work with explicit references
            MockMessage(
                content="Platform team here - we're building the OAuth client library. "
                        "Will integrate with the server that @auth-team is building. "
                        "Following the spec we agreed on in yesterday's meeting.",
                author="quinn@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=1),
                thread_id="oauth_platform_collab",
                metadata={
                    "team": "platform-team",
                    "mentions": ["auth-team"]
                }
            ),

            # Auth team work with explicit references
            MockMessage(
                content="Auth team working on the OAuth server implementation. "
                        "@platform-team will handle client integration. "
                        "We're exposing the endpoints per our shared design doc.",
                author="rachel@company.com",
                channel="#auth-team",
                timestamp=base_time + timedelta(days=1, hours=2),
                thread_id="oauth_auth_collab",
                metadata={
                    "team": "auth-team",
                    "mentions": ["platform-team"]
                }
            ),

            # Coordination check-in
            MockMessage(
                content="Sync between Platform and Auth teams went great! Server endpoints "
                        "are ready for client integration. @quinn will start testing tomorrow.",
                author="paul@company.com",
                channel="#engineering",
                timestamp=base_time + timedelta(days=5),
                thread_id=collab_thread,
                metadata={
                    "mentions": ["quinn@company.com"],
                    "reactions": [{"emoji": "thumbsup", "count": 6}]
                }
            ),

            # Joint progress
            MockMessage(
                content="Great collaboration with @auth-team! Their server implementation "
                        "integrates perfectly with our client. OAuth flow working end-to-end!",
                author="quinn@company.com",
                channel="#platform",
                timestamp=base_time + timedelta(days=7),
                thread_id="oauth_platform_collab",
                metadata={
                    "team": "platform-team",
                    "mentions": ["auth-team"],
                    "reactions": [{"emoji": "tada", "count": 7}]
                }
            ),

            # Completion
            MockMessage(
                content="OAuth implementation complete! 🎉 Thanks to excellent collaboration "
                        "between @platform-team and @auth-team. Division of labor worked perfectly.",
                author="paul@company.com",
                channel="#engineering",
                timestamp=base_time + timedelta(days=10),
                thread_id=collab_thread,
                metadata={
                    "mentions": ["platform-team", "auth-team"],
                    "reactions": [{"emoji": "rocket", "count": 12}]
                }
            ),
        ]

        return messages

    def get_scenario_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available scenarios."""
        return {
            # Original scenarios
            "oauth_discussion": "OAuth implementation discussion - potential duplicate work",
            "decision_making": "Team decision without key stakeholders",
            "bug_report": "Bug report and resolution workflow",
            "feature_planning": "Cross-team feature planning coordination",
            # Gap detection scenarios
            "oauth_duplication": "OAuth duplication - Platform and Auth teams (HIGH impact, should detect)",
            "api_redesign_duplication": "API redesign - Platform and Backend teams (HIGH impact, should detect)",
            "auth_migration_duplication": "Auth migration - Security and Platform teams (MEDIUM-HIGH impact, should detect)",
            "similar_topics_different_scope": "Edge case - Similar topics, different scope (should NOT detect)",
            "sequential_work": "Edge case - Sequential work, no overlap (should NOT detect)",
            "intentional_collaboration": "Negative example - Intentional collaboration (should NOT detect)",
        }
