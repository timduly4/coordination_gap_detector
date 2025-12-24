"""
Semantic clustering for coordination gap detection.

This module implements DBSCAN-based clustering to group similar messages
and identify coordination patterns across teams.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from src.analysis.similarity import (
    compute_cluster_cohesion,
    pairwise_distance,
)
from src.detection.patterns import ClusterBase, ClusteringStrategy


class MessageCluster(ClusterBase):
    """Extended cluster schema with message data and metadata."""

    label: Optional[str] = None  # Cluster label/topic
    participant_count: Optional[int] = None  # Unique authors
    channels: Optional[List[str]] = None  # Channels involved
    teams: Optional[List[str]] = None  # Teams identified
    cohesion_score: Optional[float] = None  # Cluster quality
    start_time: Optional[datetime] = None  # First message timestamp
    end_time: Optional[datetime] = None  # Last message timestamp


class SemanticClusterer(ClusteringStrategy):
    """
    DBSCAN-based semantic clustering for messages.

    Uses density-based clustering to group messages with similar embeddings,
    with temporal awareness and quality metrics.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 2,
        time_window_days: Optional[int] = None,
        metric: str = "cosine",
    ):
        """
        Initialize semantic clusterer.

        Args:
            similarity_threshold: Minimum similarity for clustering (0-1)
                                Higher = more strict clustering
            min_cluster_size: Minimum messages per cluster
            time_window_days: Optional time window for temporal clustering
            metric: Distance metric ('cosine', 'euclidean')
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.time_window_days = time_window_days
        self.metric = metric

        # Convert similarity threshold to distance threshold
        # For cosine: distance = 1 - similarity
        self.eps = 1.0 - similarity_threshold

    def cluster(
        self,
        embeddings: List[NDArray[np.float32]],
        timestamps: Optional[List[datetime]] = None,
        **kwargs: Any,
    ) -> List[List[int]]:
        """
        Cluster embeddings using DBSCAN.

        Args:
            embeddings: List of embedding vectors
            timestamps: Optional timestamps for temporal clustering
            **kwargs: Additional parameters

        Returns:
            List of clusters, where each cluster is a list of message indices

        Raises:
            ValueError: If embeddings list is empty
        """
        if not embeddings:
            raise ValueError("Embeddings list cannot be empty")

        # Apply temporal filtering if specified
        if self.time_window_days and timestamps:
            embeddings, timestamps, original_indices = self._filter_by_time_window(
                embeddings, timestamps
            )
            if not embeddings:
                return []
        else:
            original_indices = list(range(len(embeddings)))

        # Convert embeddings to numpy array
        X = np.array(embeddings, dtype=np.float32)

        # Run DBSCAN
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_cluster_size,
            metric=self.metric,
        )
        labels = dbscan.fit_predict(X)

        # Group indices by cluster label
        clusters_dict: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            # Skip noise points (label = -1)
            if label == -1:
                continue

            if label not in clusters_dict:
                clusters_dict[label] = []

            # Map back to original indices
            clusters_dict[label].append(original_indices[idx])

        # Convert to list of clusters
        clusters = list(clusters_dict.values())

        return clusters

    def _filter_by_time_window(
        self,
        embeddings: List[NDArray[np.float32]],
        timestamps: List[datetime],
    ) -> Tuple[List[NDArray[np.float32]], List[datetime], List[int]]:
        """
        Filter messages to recent time window.

        Args:
            embeddings: List of embedding vectors
            timestamps: List of timestamps

        Returns:
            Tuple of (filtered_embeddings, filtered_timestamps, original_indices)
        """
        if len(embeddings) != len(timestamps):
            raise ValueError("Embeddings and timestamps must have same length")

        cutoff_time = datetime.utcnow() - timedelta(days=self.time_window_days)

        filtered_embeddings = []
        filtered_timestamps = []
        original_indices = []

        for idx, (emb, ts) in enumerate(zip(embeddings, timestamps)):
            if ts >= cutoff_time:
                filtered_embeddings.append(emb)
                filtered_timestamps.append(ts)
                original_indices.append(idx)

        return filtered_embeddings, filtered_timestamps, original_indices

    def get_cluster_quality_metrics(
        self,
        embeddings: List[NDArray[np.float32]],
        clusters: List[List[int]],
    ) -> Dict[str, float]:
        """
        Compute quality metrics for clustering.

        Args:
            embeddings: Original embedding vectors
            clusters: Clustering result

        Returns:
            Dictionary with quality metrics:
                - silhouette_score: Overall clustering quality (-1 to 1)
                - avg_cluster_size: Average messages per cluster
                - num_clusters: Number of clusters found
                - coverage: Fraction of messages clustered (vs noise)
        """
        if not clusters or not embeddings:
            return {
                "silhouette_score": 0.0,
                "avg_cluster_size": 0.0,
                "num_clusters": 0,
                "coverage": 0.0,
            }

        # Create cluster labels array
        labels = np.full(len(embeddings), -1, dtype=int)
        for cluster_id, cluster_indices in enumerate(clusters):
            for idx in cluster_indices:
                labels[idx] = cluster_id

        # Count clustered messages
        clustered_count = sum(len(cluster) for cluster in clusters)
        coverage = clustered_count / len(embeddings) if embeddings else 0.0

        # Compute silhouette score (only if we have multiple clusters and enough samples)
        silhouette = 0.0
        if len(clusters) > 1 and clustered_count >= 2:
            try:
                # Get only clustered embeddings
                clustered_embeddings = []
                clustered_labels = []
                for idx, label in enumerate(labels):
                    if label != -1:
                        clustered_embeddings.append(embeddings[idx])
                        clustered_labels.append(label)

                if len(clustered_embeddings) >= 2:
                    X = np.array(clustered_embeddings, dtype=np.float32)
                    silhouette = silhouette_score(
                        X, clustered_labels, metric=self.metric
                    )
            except Exception:
                # Silhouette score can fail in edge cases
                silhouette = 0.0

        # Average cluster size
        avg_size = clustered_count / len(clusters) if clusters else 0.0

        return {
            "silhouette_score": float(silhouette),
            "avg_cluster_size": float(avg_size),
            "num_clusters": len(clusters),
            "coverage": float(coverage),
        }

    def create_message_clusters(
        self,
        messages: List[Any],
        embeddings: List[NDArray[np.float32]],
        cluster_indices: List[List[int]],
    ) -> List[MessageCluster]:
        """
        Create MessageCluster objects from clustering results.

        Args:
            messages: Original messages (must have id, timestamp, author, channel)
            embeddings: Message embeddings
            cluster_indices: Cluster assignments from cluster()

        Returns:
            List of MessageCluster objects with metadata
        """
        clusters = []

        for cluster_idx, indices in enumerate(cluster_indices):
            if not indices:
                continue

            # Get messages in cluster
            cluster_messages = [messages[i] for i in indices]
            cluster_embeddings = [embeddings[i] for i in indices]

            # Extract metadata
            message_ids = [msg.id for msg in cluster_messages]
            timestamps = [msg.timestamp for msg in cluster_messages]
            authors = set(msg.author for msg in cluster_messages if msg.author)
            channels = set(msg.channel for msg in cluster_messages if msg.channel)

            # Compute timespan
            if timestamps:
                start_time = min(timestamps)
                end_time = max(timestamps)
                timespan_days = (end_time - start_time).total_seconds() / 86400
            else:
                start_time = None
                end_time = None
                timespan_days = None

            # Compute average similarity (cohesion)
            try:
                avg_similarity = compute_cluster_cohesion(cluster_embeddings)
                cohesion_score = avg_similarity
            except Exception:
                avg_similarity = 0.0
                cohesion_score = 0.0

            # Create cluster
            cluster = MessageCluster(
                cluster_id=f"cluster_{uuid4().hex[:8]}",
                message_ids=message_ids,
                size=len(indices),
                avg_similarity=avg_similarity,
                timespan_days=timespan_days,
                participant_count=len(authors),
                channels=list(channels) if channels else None,
                cohesion_score=cohesion_score,
                start_time=start_time,
                end_time=end_time,
            )

            clusters.append(cluster)

        return clusters

    def compute_cluster_statistics(
        self, cluster: MessageCluster, messages: List[Any]
    ) -> Dict[str, Any]:
        """
        Compute detailed statistics for a cluster.

        Args:
            cluster: MessageCluster object
            messages: Original messages

        Returns:
            Dictionary with cluster statistics
        """
        cluster_messages = [
            msg for msg in messages if msg.id in cluster.message_ids
        ]

        if not cluster_messages:
            return {}

        # Temporal distribution
        timestamps = [msg.timestamp for msg in cluster_messages]
        temporal_density = len(timestamps) / max(
            cluster.timespan_days, 0.01
        )  # messages per day

        # Content statistics
        avg_content_length = np.mean(
            [len(msg.content) for msg in cluster_messages]
        )

        return {
            "temporal_density": float(temporal_density),
            "avg_content_length": float(avg_content_length),
            "unique_authors": cluster.participant_count,
            "unique_channels": len(cluster.channels) if cluster.channels else 0,
            "message_count": cluster.size,
            "cohesion_score": cluster.cohesion_score,
        }


def visualize_clusters(
    clusters: List[MessageCluster],
    top_k: int = 10,
) -> str:
    """
    Create a text visualization of clustering results.

    Args:
        clusters: List of MessageCluster objects
        top_k: Number of top clusters to show

    Returns:
        Formatted string with cluster summary
    """
    if not clusters:
        return "No clusters found."

    # Sort by size
    sorted_clusters = sorted(clusters, key=lambda c: c.size, reverse=True)[:top_k]

    lines = [f"Found {len(clusters)} clusters\n"]
    lines.append("=" * 60)

    for i, cluster in enumerate(sorted_clusters, 1):
        lines.append(f"\nCluster {i}: {cluster.cluster_id}")
        lines.append(f"  Size: {cluster.size} messages")
        lines.append(f"  Similarity: {cluster.avg_similarity:.3f}")

        if cluster.timespan_days is not None:
            lines.append(f"  Timespan: {cluster.timespan_days:.1f} days")

        if cluster.participant_count:
            lines.append(f"  Participants: {cluster.participant_count}")

        if cluster.channels:
            channels_str = ", ".join(cluster.channels[:3])
            if len(cluster.channels) > 3:
                channels_str += f" (+{len(cluster.channels) - 3} more)"
            lines.append(f"  Channels: {channels_str}")

        if cluster.cohesion_score:
            lines.append(f"  Cohesion: {cluster.cohesion_score:.3f}")

    return "\n".join(lines)
