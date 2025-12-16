"""
Unit tests for hybrid search fusion strategies.
"""

import pytest

from src.search.hybrid_search import HybridSearchFusion, deduplicate_results


class TestHybridSearchFusion:
    """Test suite for HybridSearchFusion class."""

    @pytest.fixture
    def semantic_results(self):
        """Sample semantic search results."""
        return [
            {
                "message_id": 1,
                "content": "OAuth implementation guide",
                "score": 0.95,
                "semantic_score": 0.95,
            },
            {
                "message_id": 2,
                "content": "Authentication best practices",
                "score": 0.85,
                "semantic_score": 0.85,
            },
            {
                "message_id": 3,
                "content": "Database migration planning",
                "score": 0.70,
                "semantic_score": 0.70,
            },
        ]

    @pytest.fixture
    def keyword_results(self):
        """Sample keyword search results."""
        return [
            {
                "message_id": 2,
                "content": "Authentication best practices",
                "score": 12.5,
                "keyword_score": 12.5,
            },
            {
                "message_id": 1,
                "content": "OAuth implementation guide",
                "score": 10.2,
                "keyword_score": 10.2,
            },
            {
                "message_id": 4,
                "content": "Security tokens and OAuth",
                "score": 8.7,
                "keyword_score": 8.7,
            },
        ]

    def test_initialization_rrf(self):
        """Test RRF strategy initialization."""
        fusion = HybridSearchFusion(strategy="rrf")
        assert fusion.strategy == "rrf"
        assert fusion.rrf_k == 60

    def test_initialization_weighted(self):
        """Test weighted strategy initialization."""
        fusion = HybridSearchFusion(
            strategy="weighted",
            semantic_weight=0.7,
            keyword_weight=0.3
        )
        assert fusion.strategy == "weighted"
        assert fusion.semantic_weight == 0.7
        assert fusion.keyword_weight == 0.3

    def test_initialization_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unsupported fusion strategy"):
            HybridSearchFusion(strategy="invalid")

    def test_initialization_invalid_weights(self):
        """Test that weights not summing to 1.0 raises error."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            HybridSearchFusion(
                strategy="weighted",
                semantic_weight=0.6,
                keyword_weight=0.5
            )

    def test_rrf_fusion_basic(self, semantic_results, keyword_results):
        """Test basic RRF fusion."""
        fusion = HybridSearchFusion(strategy="rrf")
        fused = fusion.fuse(semantic_results, keyword_results)

        # Should have all unique documents
        assert len(fused) == 4  # messages 1, 2, 3, 4

        # Results should be sorted by RRF score
        scores = [r["score"] for r in fused]
        assert scores == sorted(scores, reverse=True)

        # Check that ranking details are included
        assert "ranking_details" in fused[0]
        assert fused[0]["ranking_details"]["fusion_method"] == "rrf"

    def test_rrf_top_ranked_documents(self, semantic_results, keyword_results):
        """Test that documents ranked high in both get highest RRF scores."""
        fusion = HybridSearchFusion(strategy="rrf", rrf_k=60)
        fused = fusion.fuse(semantic_results, keyword_results)

        # Messages 1 and 2 appear in both result sets
        # They should be ranked highest in fused results
        top_message_ids = [r["message_id"] for r in fused[:2]]
        assert 1 in top_message_ids
        assert 2 in top_message_ids

    def test_rrf_formula(self):
        """Test RRF score calculation formula."""
        fusion = HybridSearchFusion(strategy="rrf", rrf_k=60)

        semantic = [{"message_id": 1, "score": 0.9, "semantic_score": 0.9}]
        keyword = [{"message_id": 1, "score": 10.0, "keyword_score": 10.0}]

        fused = fusion.fuse(semantic, keyword)

        # RRF score = 1/(60+1) + 1/(60+1) = 2/61
        expected_score = 2.0 / 61
        assert abs(fused[0]["score"] - expected_score) < 0.0001

    def test_weighted_fusion_basic(self, semantic_results, keyword_results):
        """Test basic weighted fusion."""
        fusion = HybridSearchFusion(
            strategy="weighted",
            semantic_weight=0.7,
            keyword_weight=0.3
        )
        fused = fusion.fuse(semantic_results, keyword_results)

        # Should have all unique documents
        assert len(fused) == 4

        # Results should be sorted by weighted score
        scores = [r["score"] for r in fused]
        assert scores == sorted(scores, reverse=True)

        # Check ranking details
        assert fused[0]["ranking_details"]["fusion_method"] == "weighted"
        assert fused[0]["ranking_details"]["semantic_weight"] == 0.7
        assert fused[0]["ranking_details"]["keyword_weight"] == 0.3

    def test_weighted_fusion_semantic_emphasis(self):
        """Test weighted fusion emphasizing semantic scores."""
        semantic = [
            {"message_id": 1, "score": 1.0, "semantic_score": 1.0},
            {"message_id": 2, "score": 0.5, "semantic_score": 0.5},
        ]
        keyword = [
            {"message_id": 2, "score": 100.0, "keyword_score": 100.0},
            {"message_id": 1, "score": 1.0, "keyword_score": 1.0},
        ]

        # High semantic weight should favor message 1
        fusion = HybridSearchFusion(
            strategy="weighted",
            semantic_weight=0.9,
            keyword_weight=0.1
        )
        fused = fusion.fuse(semantic, keyword)

        assert fused[0]["message_id"] == 1

    def test_weighted_fusion_keyword_emphasis(self):
        """Test weighted fusion emphasizing keyword scores."""
        semantic = [
            {"message_id": 1, "score": 1.0, "semantic_score": 1.0},
            {"message_id": 2, "score": 0.5, "semantic_score": 0.5},
        ]
        keyword = [
            {"message_id": 2, "score": 100.0, "keyword_score": 100.0},
            {"message_id": 1, "score": 1.0, "keyword_score": 1.0},
        ]

        # High keyword weight should favor message 2
        fusion = HybridSearchFusion(
            strategy="weighted",
            semantic_weight=0.1,
            keyword_weight=0.9
        )
        fused = fusion.fuse(semantic, keyword)

        assert fused[0]["message_id"] == 2

    def test_fusion_with_empty_semantic(self, keyword_results):
        """Test fusion when semantic results are empty."""
        fusion = HybridSearchFusion(strategy="rrf")
        fused = fusion.fuse([], keyword_results)

        # Should return all keyword results
        assert len(fused) == len(keyword_results)

    def test_fusion_with_empty_keyword(self, semantic_results):
        """Test fusion when keyword results are empty."""
        fusion = HybridSearchFusion(strategy="rrf")
        fused = fusion.fuse(semantic_results, [])

        # Should return all semantic results
        assert len(fused) == len(semantic_results)

    def test_fusion_with_both_empty(self):
        """Test fusion when both result sets are empty."""
        fusion = HybridSearchFusion(strategy="rrf")
        fused = fusion.fuse([], [])

        assert len(fused) == 0

    def test_fusion_preserves_content(self, semantic_results, keyword_results):
        """Test that fusion preserves document content."""
        fusion = HybridSearchFusion(strategy="rrf")
        fused = fusion.fuse(semantic_results, keyword_results)

        # Check that original content is preserved
        for result in fused:
            assert "content" in result
            assert "message_id" in result


class TestDeduplication:
    """Test suite for result deduplication."""

    def test_deduplicate_by_message_id(self):
        """Test deduplication based on message_id."""
        results = [
            {"message_id": 1, "content": "test", "score": 0.9},
            {"message_id": 2, "content": "test2", "score": 0.8},
            {"message_id": 1, "content": "test duplicate", "score": 0.7},
        ]

        deduped = deduplicate_results(results)

        # Should have only 2 results (message 1 and 2)
        assert len(deduped) == 2

        # Should keep the one with higher score
        message_1 = [r for r in deduped if r["message_id"] == 1][0]
        assert message_1["score"] == 0.9

    def test_deduplicate_by_external_id(self):
        """Test deduplication based on external_id."""
        results = [
            {"external_id": "slack_123", "content": "test", "score": 0.9},
            {"external_id": "github_456", "content": "test2", "score": 0.8},
            {"external_id": "slack_123", "content": "duplicate", "score": 0.6},
        ]

        deduped = deduplicate_results(results)

        assert len(deduped) == 2

    def test_deduplicate_keeps_higher_score(self):
        """Test that deduplication keeps higher scoring duplicate."""
        results = [
            {"message_id": 1, "score": 0.5},
            {"message_id": 1, "score": 0.9},
            {"message_id": 1, "score": 0.7},
        ]

        deduped = deduplicate_results(results)

        assert len(deduped) == 1
        assert deduped[0]["score"] == 0.9

    def test_deduplicate_maintains_order(self):
        """Test that deduplication maintains score-based order."""
        results = [
            {"message_id": 1, "score": 0.9},
            {"message_id": 2, "score": 0.8},
            {"message_id": 3, "score": 0.7},
            {"message_id": 1, "score": 0.6},
        ]

        deduped = deduplicate_results(results)

        scores = [r["score"] for r in deduped]
        assert scores == sorted(scores, reverse=True)

    def test_deduplicate_no_duplicates(self):
        """Test deduplication when there are no duplicates."""
        results = [
            {"message_id": 1, "score": 0.9},
            {"message_id": 2, "score": 0.8},
            {"message_id": 3, "score": 0.7},
        ]

        deduped = deduplicate_results(results)

        assert len(deduped) == len(results)

    def test_deduplicate_empty_list(self):
        """Test deduplication with empty list."""
        deduped = deduplicate_results([])
        assert len(deduped) == 0


class TestScoreNormalization:
    """Test suite for score normalization."""

    def test_normalize_scores(self):
        """Test min-max normalization of scores."""
        fusion = HybridSearchFusion(strategy="weighted")

        results = [
            {"message_id": 1, "score": 10.0, "semantic_score": 10.0},
            {"message_id": 2, "score": 5.0, "semantic_score": 5.0},
            {"message_id": 3, "score": 0.0, "semantic_score": 0.0},
        ]

        normalized = fusion._normalize_scores(results, "semantic_score")

        # Check normalization range [0, 1]
        assert normalized[0]["normalized_score"] == 1.0  # Max
        assert normalized[1]["normalized_score"] == 0.5  # Mid
        assert normalized[2]["normalized_score"] == 0.0  # Min

    def test_normalize_scores_all_same(self):
        """Test normalization when all scores are identical."""
        fusion = HybridSearchFusion(strategy="weighted")

        results = [
            {"message_id": 1, "score": 5.0, "semantic_score": 5.0},
            {"message_id": 2, "score": 5.0, "semantic_score": 5.0},
        ]

        normalized = fusion._normalize_scores(results, "semantic_score")

        # All should be normalized to 1.0
        for result in normalized:
            assert result["normalized_score"] == 1.0

    def test_normalize_empty_results(self):
        """Test normalization with empty results."""
        fusion = HybridSearchFusion(strategy="weighted")
        normalized = fusion._normalize_scores([], "score")
        assert len(normalized) == 0
