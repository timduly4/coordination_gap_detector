"""
Tests for semantic clustering.
"""

from datetime import datetime, timedelta
from typing import Any, List

import numpy as np
import pytest

from src.detection.clustering import (
    MessageCluster,
    SemanticClusterer,
    visualize_clusters,
)


class MockMessage:
    """Mock message for testing."""

    def __init__(
        self,
        id: int,
        content: str,
        timestamp: datetime,
        author: str = "user@test.com",
        channel: str = "#general",
    ):
        self.id = id
        self.content = content
        self.timestamp = timestamp
        self.author = author
        self.channel = channel


@pytest.fixture
def sample_embeddings() -> List[np.ndarray]:
    """Create sample embeddings for testing."""
    # Create 3 distinct clusters
    cluster1 = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.95, 0.05, 0.0], dtype=np.float32),
        np.array([0.9, 0.1, 0.0], dtype=np.float32),
    ]
    cluster2 = [
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.05, 0.95, 0.0], dtype=np.float32),
    ]
    cluster3 = [
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.05, 0.95], dtype=np.float32),
        np.array([0.05, 0.0, 0.95], dtype=np.float32),
    ]
    # Add noise point
    noise = [np.array([0.5, 0.5, 0.5], dtype=np.float32)]

    return cluster1 + cluster2 + cluster3 + noise


@pytest.fixture
def sample_messages() -> List[MockMessage]:
    """Create sample messages for testing."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    messages = []

    # Cluster 1: OAuth discussions
    for i in range(3):
        messages.append(
            MockMessage(
                id=i,
                content=f"OAuth implementation {i}",
                timestamp=base_time + timedelta(hours=i),
                author=f"user{i}@test.com",
                channel="#platform",
            )
        )

    # Cluster 2: API discussions
    for i in range(3, 5):
        messages.append(
            MockMessage(
                id=i,
                content=f"API design {i}",
                timestamp=base_time + timedelta(hours=i),
                author=f"user{i}@test.com",
                channel="#backend",
            )
        )

    # Cluster 3: Auth discussions
    for i in range(5, 8):
        messages.append(
            MockMessage(
                id=i,
                content=f"Auth migration {i}",
                timestamp=base_time + timedelta(hours=i),
                author=f"user{i}@test.com",
                channel="#security",
            )
        )

    # Noise
    messages.append(
        MockMessage(
            id=8,
            content="Random message",
            timestamp=base_time + timedelta(hours=8),
            author="random@test.com",
            channel="#random",
        )
    )

    return messages


class TestSemanticClusterer:
    """Tests for SemanticClusterer."""

    def test_initialization(self):
        """Test clusterer initialization."""
        clusterer = SemanticClusterer(
            similarity_threshold=0.85, min_cluster_size=2
        )

        assert clusterer.similarity_threshold == 0.85
        assert clusterer.min_cluster_size == 2
        assert clusterer.eps == pytest.approx(0.15)  # 1 - 0.85

    def test_cluster_empty_embeddings(self):
        """Test that clustering empty embeddings raises error."""
        clusterer = SemanticClusterer()

        with pytest.raises(ValueError, match="cannot be empty"):
            clusterer.cluster([])

    def test_cluster_finds_clusters(self, sample_embeddings):
        """Test that clustering finds expected clusters."""
        clusterer = SemanticClusterer(
            similarity_threshold=0.80, min_cluster_size=2
        )

        clusters = clusterer.cluster(sample_embeddings)

        # Should find 3 clusters (cluster1, cluster2, cluster3)
        assert len(clusters) == 3

        # Check cluster sizes
        cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
        assert cluster_sizes == [3, 3, 2]

    def test_cluster_with_high_threshold(self, sample_embeddings):
        """Test clustering with very high similarity threshold."""
        clusterer = SemanticClusterer(
            similarity_threshold=0.99, min_cluster_size=2
        )

        clusters = clusterer.cluster(sample_embeddings)

        # With very high threshold, should find same or fewer clusters
        # Our sample embeddings are already very similar within groups
        assert len(clusters) <= 3  # Should not exceed base clustering

    def test_cluster_with_temporal_filtering(self, sample_embeddings):
        """Test clustering with temporal time window."""
        # Create timestamps (some recent, some old)
        now = datetime.utcnow()
        timestamps = [
            now - timedelta(days=5),  # Recent
            now - timedelta(days=5),  # Recent
            now - timedelta(days=5),  # Recent
            now - timedelta(days=50),  # Old - filtered out
            now - timedelta(days=50),  # Old - filtered out
            now - timedelta(days=5),  # Recent
            now - timedelta(days=5),  # Recent
            now - timedelta(days=5),  # Recent
            now - timedelta(days=5),  # Recent
        ]

        clusterer = SemanticClusterer(
            similarity_threshold=0.80,
            min_cluster_size=2,
            time_window_days=30,
        )

        clusters = clusterer.cluster(sample_embeddings, timestamps=timestamps)

        # Should only cluster recent messages
        all_indices = [idx for cluster in clusters for idx in cluster]
        # Indices 3 and 4 should be filtered out (old messages)
        assert 3 not in all_indices
        assert 4 not in all_indices

    def test_get_cluster_quality_metrics(self, sample_embeddings):
        """Test cluster quality metrics computation."""
        clusterer = SemanticClusterer(
            similarity_threshold=0.80, min_cluster_size=2
        )

        clusters = clusterer.cluster(sample_embeddings)
        metrics = clusterer.get_cluster_quality_metrics(
            sample_embeddings, clusters
        )

        assert "silhouette_score" in metrics
        assert "avg_cluster_size" in metrics
        assert "num_clusters" in metrics
        assert "coverage" in metrics

        # Check metric ranges
        assert -1.0 <= metrics["silhouette_score"] <= 1.0
        assert metrics["avg_cluster_size"] >= 0
        assert metrics["num_clusters"] == len(clusters)
        assert 0.0 <= metrics["coverage"] <= 1.0

    def test_quality_metrics_empty_clusters(self):
        """Test quality metrics with empty clusters."""
        clusterer = SemanticClusterer()
        embeddings = [np.array([1.0, 0.0], dtype=np.float32)]

        metrics = clusterer.get_cluster_quality_metrics(embeddings, [])

        assert metrics["silhouette_score"] == 0.0
        assert metrics["avg_cluster_size"] == 0.0
        assert metrics["num_clusters"] == 0
        assert metrics["coverage"] == 0.0

    def test_create_message_clusters(
        self, sample_messages, sample_embeddings
    ):
        """Test creating MessageCluster objects."""
        clusterer = SemanticClusterer(
            similarity_threshold=0.80, min_cluster_size=2
        )

        cluster_indices = clusterer.cluster(sample_embeddings)
        message_clusters = clusterer.create_message_clusters(
            sample_messages, sample_embeddings, cluster_indices
        )

        assert len(message_clusters) > 0

        # Check first cluster
        cluster = message_clusters[0]
        assert isinstance(cluster, MessageCluster)
        assert cluster.size > 0
        assert len(cluster.message_ids) == cluster.size
        assert 0.0 <= cluster.avg_similarity <= 1.0
        assert cluster.participant_count is not None
        assert cluster.channels is not None
        assert cluster.start_time is not None
        assert cluster.end_time is not None

    def test_cluster_timespan_calculation(
        self, sample_messages, sample_embeddings
    ):
        """Test that cluster timespan is calculated correctly."""
        clusterer = SemanticClusterer(
            similarity_threshold=0.80, min_cluster_size=2
        )

        cluster_indices = clusterer.cluster(sample_embeddings)
        message_clusters = clusterer.create_message_clusters(
            sample_messages, sample_embeddings, cluster_indices
        )

        for cluster in message_clusters:
            if cluster.start_time and cluster.end_time:
                # Timespan should be non-negative
                assert cluster.timespan_days >= 0

                # Verify calculation
                expected_timespan = (
                    cluster.end_time - cluster.start_time
                ).total_seconds() / 86400
                assert cluster.timespan_days == pytest.approx(
                    expected_timespan, abs=0.01
                )

    def test_compute_cluster_statistics(self, sample_messages):
        """Test cluster statistics computation."""
        clusterer = SemanticClusterer()

        # Create a simple cluster
        cluster = MessageCluster(
            cluster_id="test_cluster",
            message_ids=[0, 1, 2],
            size=3,
            avg_similarity=0.9,
            timespan_days=2.0,
            participant_count=3,
            channels=["#platform"],
            cohesion_score=0.9,
        )

        stats = clusterer.compute_cluster_statistics(
            cluster, sample_messages[:3]
        )

        assert "temporal_density" in stats
        assert "avg_content_length" in stats
        assert "unique_authors" in stats
        assert "unique_channels" in stats
        assert "message_count" in stats
        assert "cohesion_score" in stats

        assert stats["message_count"] == 3
        assert stats["unique_authors"] == 3
        assert stats["unique_channels"] == 1


class TestVisualizeClusterts:
    """Tests for cluster visualization."""

    def test_visualize_empty_clusters(self):
        """Test visualization with no clusters."""
        result = visualize_clusters([])
        assert "No clusters found" in result

    def test_visualize_clusters(self, sample_messages, sample_embeddings):
        """Test cluster visualization."""
        clusterer = SemanticClusterer(
            similarity_threshold=0.80, min_cluster_size=2
        )

        cluster_indices = clusterer.cluster(sample_embeddings)
        message_clusters = clusterer.create_message_clusters(
            sample_messages, sample_embeddings, cluster_indices
        )

        result = visualize_clusters(message_clusters)

        assert "Found" in result
        assert "clusters" in result
        assert "Cluster" in result
        assert "Size:" in result
        assert "Similarity:" in result

    def test_visualize_limits_clusters(self, sample_messages, sample_embeddings):
        """Test that visualization limits to top_k clusters."""
        clusterer = SemanticClusterer(
            similarity_threshold=0.80, min_cluster_size=2
        )

        cluster_indices = clusterer.cluster(sample_embeddings)
        message_clusters = clusterer.create_message_clusters(
            sample_messages, sample_embeddings, cluster_indices
        )

        # Request only top 1 cluster
        result = visualize_clusters(message_clusters, top_k=1)

        # Should only show 1 cluster in detail
        cluster_count = result.count("Cluster 1:")
        assert cluster_count == 1
        # Should not show Cluster 2
        assert "Cluster 2:" not in result


class TestClusteringEdgeCases:
    """Tests for edge cases in clustering."""

    def test_single_embedding(self):
        """Test clustering with a single embedding."""
        clusterer = SemanticClusterer(min_cluster_size=1)
        embeddings = [np.array([1.0, 0.0], dtype=np.float32)]

        # Single embedding with min_cluster_size=1 can form a cluster
        clusters = clusterer.cluster(embeddings)
        assert len(clusters) <= 1

    def test_all_identical_embeddings(self):
        """Test clustering with all identical embeddings."""
        clusterer = SemanticClusterer(
            similarity_threshold=0.90, min_cluster_size=2
        )
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32) for _ in range(5)
        ]

        clusters = clusterer.cluster(embeddings)

        # All identical embeddings should form one cluster
        assert len(clusters) == 1
        assert len(clusters[0]) == 5

    def test_no_clusters_found(self):
        """Test when no clusters meet minimum size."""
        clusterer = SemanticClusterer(
            similarity_threshold=0.80, min_cluster_size=10
        )
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]

        clusters = clusterer.cluster(embeddings)

        # With min_cluster_size=10 and only 2 embeddings, no clusters
        assert len(clusters) == 0

    def test_mismatched_embeddings_timestamps_length(self):
        """Test error when embeddings and timestamps have different lengths."""
        clusterer = SemanticClusterer(time_window_days=30)
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
        timestamps = [datetime.utcnow()]  # Only 1 timestamp for 2 embeddings

        with pytest.raises(ValueError, match="same length"):
            clusterer.cluster(embeddings, timestamps=timestamps)
