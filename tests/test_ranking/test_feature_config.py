"""
Unit tests for feature configuration.
"""

import pytest

from src.ranking.feature_config import (
    FEATURE_DEFINITIONS,
    FeatureConfig,
    FeatureDefinition,
    FeatureType,
    NormalizationMethod,
    get_default_config,
    get_features_by_category,
    get_minimal_config,
)


class TestFeatureDefinition:
    """Test suite for FeatureDefinition dataclass."""

    def test_feature_definition_creation(self):
        """Test creating a feature definition."""
        fd = FeatureDefinition(
            name="test_feature",
            feature_type=FeatureType.QUERY_DOC_SIMILARITY,
            description="A test feature",
            normalization=NormalizationMethod.MINMAX,
            default_value=0.5,
            importance=0.8,
            enabled=True
        )

        assert fd.name == "test_feature"
        assert fd.feature_type == FeatureType.QUERY_DOC_SIMILARITY
        assert fd.description == "A test feature"
        assert fd.normalization == NormalizationMethod.MINMAX
        assert fd.default_value == 0.5
        assert fd.importance == 0.8
        assert fd.enabled is True

    def test_feature_definition_defaults(self):
        """Test default values for feature definition."""
        fd = FeatureDefinition(
            name="test",
            feature_type=FeatureType.TEMPORAL,
            description="Test"
        )

        assert fd.normalization == NormalizationMethod.MINMAX
        assert fd.default_value == 0.0
        assert fd.importance == 0.5
        assert fd.enabled is True


class TestFeatureDefinitions:
    """Test suite for FEATURE_DEFINITIONS constant."""

    def test_all_features_unique(self):
        """Test that all feature names are unique."""
        names = [fd.name for fd in FEATURE_DEFINITIONS]

        assert len(names) == len(set(names))

    def test_minimum_feature_count(self):
        """Test that we have at least 20 features defined."""
        assert len(FEATURE_DEFINITIONS) >= 20

    def test_features_by_category_count(self):
        """Test feature count by category."""
        by_type = {}
        for fd in FEATURE_DEFINITIONS:
            by_type[fd.feature_type] = by_type.get(fd.feature_type, 0) + 1

        # Query-doc similarity: 6 features
        assert by_type[FeatureType.QUERY_DOC_SIMILARITY] == 6

        # Temporal: 5 features
        assert by_type[FeatureType.TEMPORAL] == 5

        # Engagement: 6 features
        assert by_type[FeatureType.ENGAGEMENT] == 6

        # Authority: 4 features
        assert by_type[FeatureType.AUTHORITY] == 4

        # Content: 3 features
        assert by_type[FeatureType.CONTENT] == 3

    def test_all_features_have_descriptions(self):
        """Test that all features have descriptions."""
        for fd in FEATURE_DEFINITIONS:
            assert fd.description
            assert len(fd.description) > 10

    def test_importance_values_valid(self):
        """Test that all importance values are in [0, 1]."""
        for fd in FEATURE_DEFINITIONS:
            assert 0.0 <= fd.importance <= 1.0

    def test_high_importance_features_exist(self):
        """Test that there are high-importance features."""
        high_importance = [
            fd for fd in FEATURE_DEFINITIONS
            if fd.importance >= 0.7
        ]

        assert len(high_importance) >= 5

    def test_semantic_and_bm25_high_importance(self):
        """Test that semantic_score and bm25_score have high importance."""
        semantic = next(fd for fd in FEATURE_DEFINITIONS if fd.name == "semantic_score")
        bm25 = next(fd for fd in FEATURE_DEFINITIONS if fd.name == "bm25_score")

        assert semantic.importance >= 0.8
        assert bm25.importance >= 0.8


class TestFeatureConfig:
    """Test suite for FeatureConfig class."""

    def test_default_initialization(self):
        """Test default feature config initialization."""
        config = FeatureConfig()

        # Should auto-populate with all enabled features
        assert len(config.enabled_features) > 0
        assert len(config.feature_definitions) > 0

    def test_custom_enabled_features(self):
        """Test custom enabled features list."""
        config = FeatureConfig(
            enabled_features=["semantic_score", "bm25_score", "recency"]
        )

        assert len(config.enabled_features) == 3
        assert "semantic_score" in config.enabled_features
        assert "bm25_score" in config.enabled_features
        assert "recency" in config.enabled_features

    def test_get_feature_definition(self):
        """Test getting feature definition by name."""
        config = FeatureConfig()

        fd = config.get_feature_definition("semantic_score")

        assert fd is not None
        assert fd.name == "semantic_score"
        assert fd.feature_type == FeatureType.QUERY_DOC_SIMILARITY

    def test_get_nonexistent_feature(self):
        """Test getting nonexistent feature returns None."""
        config = FeatureConfig()

        fd = config.get_feature_definition("nonexistent_feature")

        assert fd is None

    def test_is_feature_enabled(self):
        """Test checking if feature is enabled."""
        config = FeatureConfig(
            enabled_features=["semantic_score", "bm25_score"]
        )

        assert config.is_feature_enabled("semantic_score") is True
        assert config.is_feature_enabled("bm25_score") is True
        assert config.is_feature_enabled("recency") is False

    def test_get_features_by_type(self):
        """Test getting features by type."""
        config = FeatureConfig()

        similarity_features = config.get_features_by_type(
            FeatureType.QUERY_DOC_SIMILARITY
        )

        assert len(similarity_features) == 6
        assert "semantic_score" in similarity_features
        assert "bm25_score" in similarity_features

    def test_get_feature_importance(self):
        """Test getting feature importance."""
        config = FeatureConfig()

        importance = config.get_feature_importance("semantic_score")

        assert importance > 0.0
        assert importance <= 1.0

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = FeatureConfig()

        config_dict = config.to_dict()

        assert "enabled_features" in config_dict
        assert "feature_count" in config_dict
        assert "features_by_type" in config_dict
        assert isinstance(config_dict["enabled_features"], list)
        assert config_dict["feature_count"] > 0


class TestConfigFactoryFunctions:
    """Test suite for config factory functions."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()

        assert isinstance(config, FeatureConfig)
        assert len(config.enabled_features) > 15  # Most features enabled

    def test_get_minimal_config(self):
        """Test getting minimal configuration."""
        config = get_minimal_config()

        assert isinstance(config, FeatureConfig)
        # Only high-importance features (>= 0.7)
        assert len(config.enabled_features) < len(get_default_config().enabled_features)

        # All enabled features should have high importance
        for feature_name in config.enabled_features:
            fd = config.get_feature_definition(feature_name)
            assert fd.importance >= 0.7

    def test_minimal_includes_semantic_and_bm25(self):
        """Test that minimal config includes core features."""
        config = get_minimal_config()

        assert "semantic_score" in config.enabled_features
        assert "bm25_score" in config.enabled_features

    def test_get_features_by_category(self):
        """Test getting features organized by category."""
        features = get_features_by_category()

        assert isinstance(features, dict)
        assert "query_doc_similarity" in features
        assert "temporal" in features
        assert "engagement" in features
        assert "authority" in features
        assert "content" in features

        # All categories should have features
        for category, feature_list in features.items():
            assert isinstance(feature_list, list)
            assert len(feature_list) > 0


class TestNormalizationMethods:
    """Test suite for normalization method enum."""

    def test_all_normalization_methods_exist(self):
        """Test that all normalization methods are defined."""
        methods = [
            NormalizationMethod.NONE,
            NormalizationMethod.MINMAX,
            NormalizationMethod.ZSCORE,
            NormalizationMethod.LOG,
            NormalizationMethod.SIGMOID,
        ]

        assert len(methods) == 5

    def test_features_use_appropriate_normalization(self):
        """Test that features use appropriate normalization."""
        # Temporal features should mostly use NONE (already normalized)
        temporal = [
            fd for fd in FEATURE_DEFINITIONS
            if fd.feature_type == FeatureType.TEMPORAL
        ]
        none_normalized = [
            fd for fd in temporal
            if fd.normalization == NormalizationMethod.NONE
        ]
        assert len(none_normalized) >= 3  # Most temporal features pre-normalized

        # Count features should use LOG
        count_features = [
            fd for fd in FEATURE_DEFINITIONS
            if "count" in fd.name or "depth" in fd.name
        ]
        log_normalized = [
            fd for fd in count_features
            if fd.normalization == NormalizationMethod.LOG
        ]
        assert len(log_normalized) >= 2


class TestFeatureTypes:
    """Test suite for feature type enum."""

    def test_all_feature_types_covered(self):
        """Test that all feature types are used."""
        types_used = set(fd.feature_type for fd in FEATURE_DEFINITIONS)

        assert FeatureType.QUERY_DOC_SIMILARITY in types_used
        assert FeatureType.TEMPORAL in types_used
        assert FeatureType.ENGAGEMENT in types_used
        assert FeatureType.AUTHORITY in types_used
        assert FeatureType.CONTENT in types_used

    def test_feature_type_values(self):
        """Test feature type enum values."""
        assert FeatureType.QUERY_DOC_SIMILARITY.value == "query_doc_similarity"
        assert FeatureType.TEMPORAL.value == "temporal"
        assert FeatureType.ENGAGEMENT.value == "engagement"
        assert FeatureType.AUTHORITY.value == "authority"
        assert FeatureType.CONTENT.value == "content"


class TestFeatureConfigEdgeCases:
    """Test suite for edge cases in feature configuration."""

    def test_empty_enabled_features(self):
        """Test config with empty enabled features gets auto-populated."""
        config = FeatureConfig(enabled_features=[])

        # Empty list triggers auto-population with default enabled features
        assert len(config.enabled_features) > 0
        assert len(config.feature_definitions) > 0

    def test_config_with_normalization_stats(self):
        """Test config with pre-computed normalization stats."""
        stats = {
            "semantic_score": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.2}
        }

        config = FeatureConfig(normalization_stats=stats)

        assert "semantic_score" in config.normalization_stats
        assert config.normalization_stats["semantic_score"]["min"] == 0.0

    def test_config_importance_for_nonexistent(self):
        """Test getting importance for nonexistent feature."""
        config = FeatureConfig()

        importance = config.get_feature_importance("nonexistent")

        assert importance == 0.0
