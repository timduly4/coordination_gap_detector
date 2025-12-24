"""
Tests for semantic similarity computation.
"""

import numpy as np
import pytest

from src.analysis.similarity import (
    average_similarity,
    compute_cluster_cohesion,
    compute_cluster_separation,
    cosine_distance,
    cosine_similarity,
    max_similarity,
    pairwise_distance,
    pairwise_similarity,
)


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        """Test that identical vectors have similarity of 1.0."""
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        similarity = cosine_similarity(vector, vector)
        assert similarity == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity of 0.0."""
        vector1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vector2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        similarity = cosine_similarity(vector1, vector2)
        assert similarity == pytest.approx(0.0, abs=1e-5)

    def test_similar_vectors(self):
        """Test that similar vectors have high similarity."""
        vector1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vector2 = np.array([1.1, 2.1, 3.1], dtype=np.float32)
        similarity = cosine_similarity(vector1, vector2)
        assert 0.95 < similarity <= 1.0

    def test_different_shapes_raises_error(self):
        """Test that vectors with different shapes raise ValueError."""
        vector1 = np.array([1.0, 2.0], dtype=np.float32)
        vector2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="same shape"):
            cosine_similarity(vector1, vector2)

    def test_empty_vectors_raises_error(self):
        """Test that empty vectors raise ValueError."""
        vector1 = np.array([], dtype=np.float32)
        vector2 = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="cannot be empty"):
            cosine_similarity(vector1, vector2)

    def test_zero_vector(self):
        """Test that zero vectors return 0.0 similarity."""
        vector1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        vector2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        similarity = cosine_similarity(vector1, vector2)
        assert similarity == 0.0


class TestCosineDistance:
    """Tests for cosine distance computation."""

    def test_identical_vectors(self):
        """Test that identical vectors have distance of 0.0."""
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        distance = cosine_distance(vector, vector)
        assert distance == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have distance of 1.0."""
        vector1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vector2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        distance = cosine_distance(vector1, vector2)
        assert distance == pytest.approx(1.0, abs=1e-5)


class TestPairwiseSimilarity:
    """Tests for pairwise similarity matrix computation."""

    def test_single_embedding(self):
        """Test pairwise similarity with a single embedding."""
        embeddings = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        matrix = pairwise_similarity(embeddings)
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == pytest.approx(1.0)

    def test_multiple_embeddings(self):
        """Test pairwise similarity with multiple embeddings."""
        embeddings = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        ]
        matrix = pairwise_similarity(embeddings)

        assert matrix.shape == (3, 3)

        # Diagonal should be all 1.0
        assert matrix[0, 0] == pytest.approx(1.0)
        assert matrix[1, 1] == pytest.approx(1.0)
        assert matrix[2, 2] == pytest.approx(1.0)

        # Symmetric matrix
        assert matrix[0, 1] == pytest.approx(matrix[1, 0])
        assert matrix[0, 2] == pytest.approx(matrix[2, 0])

        # Same vectors should be similar
        assert matrix[0, 2] == pytest.approx(1.0)

        # Orthogonal vectors should be dissimilar
        assert matrix[0, 1] == pytest.approx(0.0, abs=1e-5)

    def test_empty_embeddings_raises_error(self):
        """Test that empty embeddings list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            pairwise_similarity([])

    def test_different_shapes_raises_error(self):
        """Test that embeddings with different shapes raise ValueError."""
        embeddings = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ]
        with pytest.raises(ValueError, match="same shape"):
            pairwise_similarity(embeddings)


class TestPairwiseDistance:
    """Tests for pairwise distance matrix computation."""

    def test_pairwise_distance_matrix(self):
        """Test that distance matrix is complement of similarity matrix."""
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
        dist_matrix = pairwise_distance(embeddings)
        sim_matrix = pairwise_similarity(embeddings)

        # Distance = 1 - Similarity
        np.testing.assert_array_almost_equal(dist_matrix, 1.0 - sim_matrix)


class TestAverageSimilarity:
    """Tests for average similarity computation."""

    def test_average_similarity_identical(self):
        """Test average similarity with identical vectors."""
        target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        references = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ]
        avg_sim = average_similarity(target, references)
        assert avg_sim == pytest.approx(1.0)

    def test_average_similarity_mixed(self):
        """Test average similarity with mixed vectors."""
        target = np.array([1.0, 0.0], dtype=np.float32)
        references = [
            np.array([1.0, 0.0], dtype=np.float32),  # Identical: sim = 1.0
            np.array([0.0, 1.0], dtype=np.float32),  # Orthogonal: sim = 0.0
        ]
        avg_sim = average_similarity(target, references)
        assert avg_sim == pytest.approx(0.5, abs=0.1)

    def test_empty_references_raises_error(self):
        """Test that empty references list raises ValueError."""
        target = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="cannot be empty"):
            average_similarity(target, [])


class TestMaxSimilarity:
    """Tests for maximum similarity computation."""

    def test_max_similarity_finds_most_similar(self):
        """Test that max_similarity finds the most similar vector."""
        target = np.array([1.0, 0.0], dtype=np.float32)
        references = [
            np.array([0.0, 1.0], dtype=np.float32),  # Orthogonal
            np.array([1.0, 0.0], dtype=np.float32),  # Identical
            np.array([0.5, 0.5], dtype=np.float32),  # Somewhat similar
        ]
        max_sim, max_idx = max_similarity(target, references)

        assert max_sim == pytest.approx(1.0)
        assert max_idx == 1

    def test_empty_references_raises_error(self):
        """Test that empty references list raises ValueError."""
        target = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="cannot be empty"):
            max_similarity(target, [])


class TestClusterCohesion:
    """Tests for cluster cohesion computation."""

    def test_cohesion_identical_vectors(self):
        """Test cohesion with identical vectors."""
        embeddings = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ]
        cohesion = compute_cluster_cohesion(embeddings)
        assert cohesion == pytest.approx(1.0)

    def test_cohesion_diverse_vectors(self):
        """Test cohesion with diverse vectors."""
        embeddings = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ]
        cohesion = compute_cluster_cohesion(embeddings)
        # Orthogonal vectors have low cohesion
        assert 0.0 <= cohesion < 0.3

    def test_cohesion_requires_minimum_size(self):
        """Test that cohesion requires at least 2 embeddings."""
        embeddings = [np.array([1.0, 2.0], dtype=np.float32)]
        with pytest.raises(ValueError, match="at least 2 embeddings"):
            compute_cluster_cohesion(embeddings)


class TestClusterSeparation:
    """Tests for cluster separation computation."""

    def test_separation_identical_clusters(self):
        """Test separation with identical clusters."""
        cluster1 = [np.array([1.0, 0.0], dtype=np.float32)]
        cluster2 = [np.array([1.0, 0.0], dtype=np.float32)]
        separation = compute_cluster_separation(cluster1, cluster2)
        # Identical vectors: distance = 0
        assert separation == pytest.approx(0.0)

    def test_separation_orthogonal_clusters(self):
        """Test separation with orthogonal clusters."""
        cluster1 = [np.array([1.0, 0.0], dtype=np.float32)]
        cluster2 = [np.array([0.0, 1.0], dtype=np.float32)]
        separation = compute_cluster_separation(cluster1, cluster2)
        # Orthogonal vectors: distance = 1
        assert separation == pytest.approx(1.0)

    def test_separation_requires_non_empty_clusters(self):
        """Test that separation requires non-empty clusters."""
        cluster1 = [np.array([1.0, 2.0], dtype=np.float32)]
        cluster2 = []
        with pytest.raises(ValueError, match="non-empty"):
            compute_cluster_separation(cluster1, cluster2)
