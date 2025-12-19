"""
Unit tests for feature extraction.
"""

from datetime import datetime, timedelta

import pytest

from src.ranking.feature_config import FeatureConfig, get_minimal_config
from src.ranking.features import FeatureExtractor


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return {
        "content": "OAuth implementation guide with code examples",
        "author": "alice@example.com",
        "channel": "engineering",
        "timestamp": datetime(2024, 1, 14, 10, 0, 0),
        "message_metadata": {
            "reply_count": 5,
            "participants": ["alice@example.com", "bob@example.com", "charlie@example.com"],
            "reactions": {"thumbsup": 3, "heart": 2},
            "thread_timestamps": [
                datetime(2024, 1, 14, 10, 0, 0),
                datetime(2024, 1, 14, 10, 30, 0),
                datetime(2024, 1, 14, 11, 0, 0),
            ],
            "author_seniority": 0.8,
            "channel_importance": 0.7,
            "team_influence": 0.6,
        }
    }


@pytest.fixture
def query_context():
    """Sample query context."""
    return {
        "reference_time": datetime(2024, 1, 15, 12, 0, 0),
        "recency_half_life": 30.0,
        "relevance_window_days": 90.0,
    }


class TestFeatureExtractor:
    """Test suite for FeatureExtractor class."""

    def test_initialization_default(self):
        """Test default initialization."""
        extractor = FeatureExtractor()

        assert extractor.config is not None
        assert len(extractor.config.enabled_features) > 0

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = FeatureConfig(
            enabled_features=["semantic_score", "bm25_score"]
        )
        extractor = FeatureExtractor(config=config)

        assert len(extractor.config.enabled_features) == 2

    def test_extract_returns_dict(self, sample_document, query_context):
        """Test that extract returns a dictionary."""
        extractor = FeatureExtractor()

        features = extractor.extract(
            query="OAuth implementation",
            document=sample_document,
            query_context=query_context,
            semantic_score=0.9,
            bm25_score=0.8
        )

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extract_includes_enabled_features(self, sample_document):
        """Test that only enabled features are extracted."""
        config = FeatureConfig(
            enabled_features=["semantic_score", "bm25_score", "exact_match"]
        )
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="OAuth",
            document=sample_document,
            semantic_score=0.9,
            bm25_score=0.8
        )

        assert "semantic_score" in features
        assert "bm25_score" in features
        assert "exact_match" in features
        assert "recency" not in features  # Not enabled

    def test_extract_with_minimal_config(self, sample_document):
        """Test extraction with minimal config."""
        config = get_minimal_config()
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="OAuth",
            document=sample_document,
            semantic_score=0.9,
            bm25_score=0.8
        )

        # Should have high-importance features only
        assert len(features) < 15
        assert "semantic_score" in features
        assert "bm25_score" in features


class TestQueryDocSimilarityFeatures:
    """Test suite for query-document similarity features."""

    def test_semantic_score(self, sample_document):
        """Test semantic score feature."""
        config = FeatureConfig(enabled_features=["semantic_score"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="test",
            document=sample_document,
            semantic_score=0.92
        )

        assert features["semantic_score"] == 0.92

    def test_bm25_score(self, sample_document):
        """Test BM25 score feature."""
        config = FeatureConfig(enabled_features=["bm25_score"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="test",
            document=sample_document,
            bm25_score=15.5
        )

        # Should be normalized
        assert "bm25_score" in features

    def test_exact_match_positive(self, sample_document):
        """Test exact match when query is in document."""
        config = FeatureConfig(enabled_features=["exact_match"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="OAuth implementation",
            document=sample_document
        )

        assert features["exact_match"] == 1.0

    def test_exact_match_negative(self, sample_document):
        """Test exact match when query is not in document."""
        config = FeatureConfig(enabled_features=["exact_match"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="database migration",
            document=sample_document
        )

        assert features["exact_match"] == 0.0

    def test_term_coverage_full(self, sample_document):
        """Test term coverage when all terms match."""
        config = FeatureConfig(enabled_features=["term_coverage"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="OAuth implementation",
            document=sample_document
        )

        # Both "oauth" and "implementation" are in document
        assert features["term_coverage"] == 1.0

    def test_term_coverage_partial(self, sample_document):
        """Test term coverage with partial match."""
        config = FeatureConfig(enabled_features=["term_coverage"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="OAuth database migration",
            document=sample_document
        )

        # Only "OAuth" matches (1 out of 3 terms)
        assert 0.3 < features["term_coverage"] < 0.4

    def test_term_coverage_none(self, sample_document):
        """Test term coverage with no matches."""
        config = FeatureConfig(enabled_features=["term_coverage"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="database schema migration",
            document=sample_document
        )

        assert features["term_coverage"] == 0.0

    def test_title_match(self):
        """Test title/channel match feature."""
        config = FeatureConfig(enabled_features=["title_match"])
        extractor = FeatureExtractor(config=config)

        document = {
            "content": "Some content here",
            "channel": "engineering-oauth-team",
            "timestamp": datetime(2024, 1, 15),
        }

        features = extractor.extract(
            query="OAuth engineering",
            document=document
        )

        # Both terms in channel name
        assert features["title_match"] == 1.0

    def test_entity_overlap(self, sample_document):
        """Test entity overlap feature."""
        config = FeatureConfig(enabled_features=["entity_overlap"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="OAuth and Auth0",  # Capitalized entities
            document=sample_document
        )

        # Should detect entity overlap
        assert "entity_overlap" in features


class TestTemporalFeatures:
    """Test suite for temporal features."""

    def test_recency_recent(self, query_context):
        """Test recency for recent message."""
        config = FeatureConfig(enabled_features=["recency"])
        extractor = FeatureExtractor(config=config)

        document = {
            "content": "test",
            "timestamp": datetime(2024, 1, 15, 11, 0, 0),  # 1 hour ago
        }

        features = extractor.extract(
            query="test",
            document=document,
            query_context=query_context
        )

        # Very recent
        assert features["recency"] > 0.95

    def test_recency_old(self, query_context):
        """Test recency for old message."""
        config = FeatureConfig(enabled_features=["recency"])
        extractor = FeatureExtractor(config=config)

        document = {
            "content": "test",
            "timestamp": datetime(2023, 1, 15),  # 1 year ago
        }

        features = extractor.extract(
            query="test",
            document=document,
            query_context=query_context
        )

        # Very old
        assert features["recency"] < 0.1

    def test_activity_burst(self, sample_document, query_context):
        """Test activity burst detection."""
        config = FeatureConfig(enabled_features=["activity_burst"])
        extractor = FeatureExtractor(config=config)

        # Add burst of recent activity
        sample_document["message_metadata"]["thread_timestamps"] = [
            datetime(2024, 1, 1),  # Historical
            datetime(2024, 1, 15, 11, 0),  # Recent burst
            datetime(2024, 1, 15, 11, 30),
            datetime(2024, 1, 15, 12, 0),
        ]

        features = extractor.extract(
            query="test",
            document=sample_document,
            query_context=query_context
        )

        assert "activity_burst" in features

    def test_temporal_relevance(self, query_context):
        """Test temporal relevance feature."""
        config = FeatureConfig(enabled_features=["temporal_relevance"])
        extractor = FeatureExtractor(config=config)

        document = {
            "content": "test",
            "timestamp": datetime(2024, 1, 10),  # 5 days ago
        }

        features = extractor.extract(
            query="test",
            document=document,
            query_context=query_context
        )

        # Within relevance window
        assert features["temporal_relevance"] > 0.9

    def test_edit_freshness_with_edit(self, query_context):
        """Test edit freshness with recent edit."""
        config = FeatureConfig(enabled_features=["edit_freshness"])
        extractor = FeatureExtractor(config=config)

        document = {
            "content": "test",
            "timestamp": datetime(2024, 1, 1),  # Old creation
            "message_metadata": {
                "edited_at": datetime(2024, 1, 14),  # Recent edit
            }
        }

        features = extractor.extract(
            query="test",
            document=document,
            query_context=query_context
        )

        # Recent edit = high freshness
        assert features["edit_freshness"] > 0.8

    def test_response_velocity(self, sample_document):
        """Test response velocity feature."""
        config = FeatureConfig(enabled_features=["response_velocity"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="test",
            document=sample_document
        )

        assert "response_velocity" in features
        assert 0.0 <= features["response_velocity"] <= 1.0


class TestEngagementFeatures:
    """Test suite for engagement features."""

    def test_thread_depth(self, sample_document):
        """Test thread depth feature."""
        config = FeatureConfig(enabled_features=["thread_depth"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="test",
            document=sample_document
        )

        # Should be log-normalized
        assert "thread_depth" in features
        assert features["thread_depth"] > 0.0

    def test_participant_count(self, sample_document):
        """Test participant count feature."""
        config = FeatureConfig(enabled_features=["participant_count"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="test",
            document=sample_document
        )

        # 3 participants
        assert "participant_count" in features
        assert features["participant_count"] > 0.0

    def test_reaction_count(self, sample_document):
        """Test reaction count feature."""
        config = FeatureConfig(enabled_features=["reaction_count"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="test",
            document=sample_document
        )

        # 3 + 2 = 5 reactions
        assert "reaction_count" in features
        assert features["reaction_count"] > 0.0

    def test_reaction_diversity(self, sample_document):
        """Test reaction diversity feature."""
        config = FeatureConfig(enabled_features=["reaction_diversity"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="test",
            document=sample_document
        )

        # 2 unique reaction types
        assert "reaction_diversity" in features

    def test_cross_team_engagement(self):
        """Test cross-team engagement feature."""
        config = FeatureConfig(enabled_features=["cross_team_engagement"])
        extractor = FeatureExtractor(config=config)

        document = {
            "content": "test",
            "timestamp": datetime(2024, 1, 15),
            "message_metadata": {
                "teams_involved": ["engineering", "product", "design"]
            }
        }

        features = extractor.extract(
            query="test",
            document=document
        )

        assert "cross_team_engagement" in features


class TestAuthorityFeatures:
    """Test suite for authority features."""

    def test_author_seniority(self, sample_document):
        """Test author seniority feature."""
        config = FeatureConfig(enabled_features=["author_seniority"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="test",
            document=sample_document
        )

        # Should be 0.8 from metadata
        assert features["author_seniority"] == 0.8

    def test_channel_importance(self, sample_document):
        """Test channel importance feature."""
        config = FeatureConfig(enabled_features=["channel_importance"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="test",
            document=sample_document
        )

        # Should be 0.7 from metadata
        assert features["channel_importance"] == 0.7

    def test_team_influence(self, sample_document):
        """Test team influence feature."""
        config = FeatureConfig(enabled_features=["team_influence"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="test",
            document=sample_document
        )

        # Should be 0.6 from metadata
        assert features["team_influence"] == 0.6

    def test_domain_expertise(self, sample_document):
        """Test domain expertise feature."""
        config = FeatureConfig(enabled_features=["domain_expertise"])
        extractor = FeatureExtractor(config=config)

        # Add domain expertise to metadata
        sample_document["message_metadata"]["domain_expertise"] = {
            "oauth": 0.9,
            "authentication": 0.85
        }

        features = extractor.extract(
            query="OAuth authentication",
            document=sample_document
        )

        # Should match high expertise
        assert features["domain_expertise"] >= 0.85


class TestContentFeatures:
    """Test suite for content features."""

    def test_message_length(self, sample_document):
        """Test message length feature."""
        config = FeatureConfig(enabled_features=["message_length"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="test",
            document=sample_document
        )

        # Should be log-normalized
        assert "message_length" in features
        assert features["message_length"] > 0.0

    def test_code_snippet_present_with_code(self):
        """Test code snippet detection with code present."""
        config = FeatureConfig(enabled_features=["code_snippet_present"])
        extractor = FeatureExtractor(config=config)

        document = {
            "content": "Here is the code: ```python\nprint('hello')\n```",
            "timestamp": datetime(2024, 1, 15),
        }

        features = extractor.extract(
            query="test",
            document=document
        )

        assert features["code_snippet_present"] == 1.0

    def test_code_snippet_present_without_code(self, sample_document):
        """Test code snippet detection without code."""
        config = FeatureConfig(enabled_features=["code_snippet_present"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="test",
            document=sample_document
        )

        # Sample document has no code blocks
        assert features["code_snippet_present"] == 0.0

    def test_link_count(self):
        """Test link count feature."""
        config = FeatureConfig(enabled_features=["link_count"])
        extractor = FeatureExtractor(config=config)

        document = {
            "content": "Check out https://example.com and https://docs.example.com/guide",
            "timestamp": datetime(2024, 1, 15),
        }

        features = extractor.extract(
            query="test",
            document=document
        )

        # Should detect 2 links and log-normalize
        assert "link_count" in features
        assert features["link_count"] > 0.0


class TestFeatureNormalization:
    """Test suite for feature normalization."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        config = FeatureConfig(
            enabled_features=["bm25_score"],
            normalization_stats={
                "bm25_score": {"min": 0.0, "max": 100.0, "mean": 50.0, "std": 20.0}
            }
        )
        extractor = FeatureExtractor(config=config)

        document = {"content": "test", "timestamp": datetime(2024, 1, 15)}

        features = extractor.extract(
            query="test",
            document=document,
            bm25_score=50.0
        )

        # 50.0 normalized to [0, 100] = 0.5
        assert features["bm25_score"] == 0.5

    def test_log_normalization(self):
        """Test log normalization for count features."""
        config = FeatureConfig(enabled_features=["thread_depth"])
        extractor = FeatureExtractor(config=config)

        document = {
            "content": "test",
            "timestamp": datetime(2024, 1, 15),
            "message_metadata": {"reply_count": 100}
        }

        features = extractor.extract(
            query="test",
            document=document
        )

        # Should be log-normalized
        assert 0.0 < features["thread_depth"] < 1.0

    def test_no_normalization(self, sample_document):
        """Test features with no normalization."""
        config = FeatureConfig(enabled_features=["semantic_score", "exact_match"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="OAuth",
            document=sample_document,
            semantic_score=0.95
        )

        # These should not be normalized (already in [0, 1])
        assert features["semantic_score"] == 0.95
        assert features["exact_match"] in [0.0, 1.0]


class TestFeatureExtractionEdgeCases:
    """Test suite for edge cases in feature extraction."""

    def test_missing_metadata(self):
        """Test extraction with missing metadata."""
        config = FeatureConfig(enabled_features=["reaction_count", "thread_depth"])
        extractor = FeatureExtractor(config=config)

        document = {
            "content": "test",
            "timestamp": datetime(2024, 1, 15),
            # No message_metadata
        }

        features = extractor.extract(
            query="test",
            document=document
        )

        # Should use defaults
        assert features["reaction_count"] == 0.0
        assert features["thread_depth"] == 0.0

    def test_missing_scores(self, sample_document):
        """Test extraction without pre-computed scores."""
        config = FeatureConfig(enabled_features=["semantic_score", "bm25_score"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="test",
            document=sample_document
            # No semantic_score or bm25_score provided
        )

        # Should default to 0.0
        assert features["semantic_score"] == 0.0
        assert features["bm25_score"] == 0.0

    def test_empty_query(self, sample_document):
        """Test extraction with empty query."""
        config = FeatureConfig(enabled_features=["term_coverage", "exact_match"])
        extractor = FeatureExtractor(config=config)

        features = extractor.extract(
            query="",
            document=sample_document
        )

        # Empty query should give 0 coverage
        assert features["term_coverage"] == 0.0
        assert features["exact_match"] == 0.0

    def test_empty_content(self):
        """Test extraction with empty content."""
        config = FeatureConfig(enabled_features=["message_length", "term_coverage"])
        extractor = FeatureExtractor(config=config)

        document = {
            "content": "",
            "timestamp": datetime(2024, 1, 15),
        }

        features = extractor.extract(
            query="test query",
            document=document
        )

        assert features["message_length"] == 0.0
        assert features["term_coverage"] == 0.0

    def test_feature_extraction_error_handling(self, sample_document):
        """Test that extraction errors are handled gracefully."""
        config = FeatureConfig(enabled_features=["semantic_score", "recency"])
        extractor = FeatureExtractor(config=config)

        # Document with invalid timestamp
        bad_document = sample_document.copy()
        bad_document["timestamp"] = "invalid"

        # Should not raise exception, use defaults
        features = extractor.extract(
            query="test",
            document=bad_document,
            semantic_score=0.9
        )

        # Semantic score should work
        assert features["semantic_score"] == 0.9


class TestStatisticsTracking:
    """Test suite for statistics tracking."""

    def test_track_stats_enabled(self, sample_document):
        """Test statistics tracking when enabled."""
        extractor = FeatureExtractor(track_stats=True)

        extractor.extract(
            query="test",
            document=sample_document,
            semantic_score=0.9,
            bm25_score=0.8
        )

        assert len(extractor.feature_stats) > 0

    def test_get_normalization_stats(self, sample_document):
        """Test getting normalization statistics."""
        extractor = FeatureExtractor(track_stats=True)

        # Extract features multiple times
        for i in range(10):
            extractor.extract(
                query="test",
                document=sample_document,
                semantic_score=0.8 + i * 0.01
            )

        stats = extractor.get_normalization_stats()

        assert "semantic_score" in stats
        assert "min" in stats["semantic_score"]
        assert "max" in stats["semantic_score"]
        assert "mean" in stats["semantic_score"]
        assert "std" in stats["semantic_score"]

    def test_stats_not_tracked_by_default(self, sample_document):
        """Test that stats are not tracked by default."""
        extractor = FeatureExtractor(track_stats=False)

        extractor.extract(
            query="test",
            document=sample_document
        )

        assert len(extractor.feature_stats) == 0
