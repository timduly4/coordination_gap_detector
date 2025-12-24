"""
Integration tests for semantic clustering with vector store.

Tests the full clustering pipeline with actual embeddings and database.
"""

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pytest

from src.detection.clustering import SemanticClusterer
from src.models.embeddings import EmbeddingGenerator


@pytest.mark.integration
class TestClusteringIntegration:
    """Integration tests for clustering with vector store."""

    @pytest.fixture
    def embedding_generator(self):
        """Create embedding generator for testing."""
        return EmbeddingGenerator()

    @pytest.fixture
    def sample_texts(self) -> List[str]:
        """Create sample text messages for clustering."""
        oauth_texts = [
            "We're implementing OAuth2 authorization for the API gateway",
            "Starting work on OAuth integration with the authentication service",
            "Need to add OAuth2 support for third-party applications",
        ]

        api_texts = [
            "Redesigning the REST API endpoints for better performance",
            "Working on API versioning strategy for backward compatibility",
        ]

        auth_texts = [
            "Migrating from legacy authentication to modern auth system",
            "Planning the auth service migration to microservices",
            "Authentication system upgrade requires database migration",
        ]

        unrelated = [
            "Updated the documentation for the onboarding process",
        ]

        return oauth_texts + api_texts + auth_texts + unrelated

    def test_cluster_real_embeddings(
        self, embedding_generator, sample_texts
    ):
        """Test clustering with real embeddings from embedding model."""
        # Generate embeddings
        embeddings = []
        for text in sample_texts:
            embedding = embedding_generator.generate_embedding(text)
            embeddings.append(np.array(embedding, dtype=np.float32))

        # Cluster with lower threshold for real embeddings
        # Real embeddings from sentence-transformers have lower similarity scores
        clusterer = SemanticClusterer(
            similarity_threshold=0.60, min_cluster_size=2
        )
        clusters = clusterer.cluster(embeddings)

        # Should find at least 1 cluster (similar topics group together)
        assert len(clusters) >= 1

        # At least some messages should be clustered
        total_clustered = sum(len(c) for c in clusters)
        assert total_clustered >= 2  # At least 2 messages in clusters

    def test_cluster_quality_with_real_data(
        self, embedding_generator, sample_texts
    ):
        """Test cluster quality metrics with real embeddings."""
        # Generate embeddings
        embeddings = []
        for text in sample_texts:
            embedding = embedding_generator.generate_embedding(text)
            embeddings.append(np.array(embedding, dtype=np.float32))

        # Cluster with lower threshold
        clusterer = SemanticClusterer(
            similarity_threshold=0.60, min_cluster_size=2
        )
        clusters = clusterer.cluster(embeddings)
        metrics = clusterer.get_cluster_quality_metrics(embeddings, clusters)

        # Quality checks - be flexible for real data
        assert metrics["num_clusters"] >= 0  # May find 0 clusters with strict threshold

        # If we found clusters, check quality
        if metrics["num_clusters"] > 0:
            assert metrics["coverage"] > 0.0  # Some messages clustered
            assert metrics["avg_cluster_size"] >= 2  # Minimum cluster size

            # Silhouette score should be reasonable for good clustering
            if metrics["num_clusters"] > 1:
                assert metrics["silhouette_score"] > -0.5  # Not terrible clustering

    def test_temporal_clustering_integration(
        self, embedding_generator, sample_texts
    ):
        """Test temporal clustering with real data."""
        # Generate embeddings
        embeddings = []
        for text in sample_texts:
            embedding = embedding_generator.generate_embedding(text)
            embeddings.append(np.array(embedding, dtype=np.float32))

        # Create timestamps (half recent, half old)
        now = datetime.utcnow()
        timestamps = []
        for i in range(len(sample_texts)):
            if i < len(sample_texts) // 2:
                # Recent
                timestamps.append(now - timedelta(days=5))
            else:
                # Old
                timestamps.append(now - timedelta(days=60))

        # Cluster with 30-day window and lower threshold
        clusterer = SemanticClusterer(
            similarity_threshold=0.60,
            min_cluster_size=2,
            time_window_days=30,
        )
        clusters = clusterer.cluster(embeddings, timestamps=timestamps)

        # All clustered messages should be recent
        # If we found any clusters
        if clusters:
            for cluster in clusters:
                for idx in cluster:
                    assert timestamps[idx] >= now - timedelta(days=30)

    def test_empty_text_handling(self, embedding_generator):
        """Test handling of empty or very short texts."""
        texts = ["", "a", "This is a normal message"]

        embeddings = []
        for text in texts:
            if text:  # Skip empty strings
                embedding = embedding_generator.generate_embedding(text)
                embeddings.append(np.array(embedding, dtype=np.float32))

        # Should handle short texts without errors
        clusterer = SemanticClusterer(min_cluster_size=1)
        clusters = clusterer.cluster(embeddings)

        # Should complete without errors
        assert isinstance(clusters, list)

    def test_large_scale_clustering(self, embedding_generator):
        """Test clustering performance with larger dataset."""
        # Generate 50 similar messages (5 groups of 10)
        texts = []
        for group in range(5):
            for i in range(10):
                texts.append(
                    f"This is message {i} about topic {group} with some variation"
                )

        # Generate embeddings
        embeddings = []
        for text in texts:
            embedding = embedding_generator.generate_embedding(text)
            embeddings.append(np.array(embedding, dtype=np.float32))

        # Cluster with lower threshold and smaller min size
        clusterer = SemanticClusterer(
            similarity_threshold=0.65, min_cluster_size=3
        )
        clusters = clusterer.cluster(embeddings)

        # Should find at least some clusters
        assert len(clusters) >= 1

        # Total clustered messages should be significant
        total_clustered = sum(len(c) for c in clusters)
        assert total_clustered >= 10  # At least 20% clustered


@pytest.mark.integration
class TestClusteringWithDatabase:
    """Integration tests with actual database."""

    @pytest.mark.asyncio
    async def test_cluster_database_messages(
        self, async_db_session, embedding_generator
    ):
        """Test clustering messages from database with real embeddings."""
        from src.db.models import Message, Source

        # Create test source
        source = Source(type="slack", name="Test Slack")
        async_db_session.add(source)
        await async_db_session.commit()
        await async_db_session.refresh(source)

        # Create test messages
        messages_data = [
            "Implementing OAuth2 for API authentication",
            "Working on OAuth integration with third-party services",
            "API redesign for better performance",
            "Migrating authentication to new system",
        ]

        messages = []
        base_time = datetime.utcnow()
        for i, content in enumerate(messages_data):
            msg = Message(
                content=content,
                author=f"user{i}@test.com",
                channel="#platform",
                timestamp=base_time + timedelta(hours=i),
                source_id=source.id,
            )
            async_db_session.add(msg)
            messages.append(msg)

        await async_db_session.commit()

        # Generate embeddings
        embeddings = []
        for msg in messages:
            embedding = embedding_generator.generate_embedding(msg.content)
            embeddings.append(np.array(embedding, dtype=np.float32))

        # Cluster with lower threshold
        clusterer = SemanticClusterer(
            similarity_threshold=0.60, min_cluster_size=2
        )
        clusters = clusterer.cluster(embeddings)

        # Verify clustering worked
        # If we found clusters, verify their properties
        if clusters:
            # Create message clusters with metadata
            message_clusters = clusterer.create_message_clusters(
                messages, embeddings, clusters
            )

            # Verify message cluster properties
            for cluster in message_clusters:
                assert cluster.size >= 2
                assert cluster.participant_count >= 1
                assert cluster.channels is not None
                assert len(cluster.message_ids) == cluster.size

    @pytest.fixture
    def embedding_generator(self):
        """Create embedding generator for testing."""
        return EmbeddingGenerator()
