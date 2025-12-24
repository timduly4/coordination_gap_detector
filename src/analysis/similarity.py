"""
Semantic similarity computation for message clustering.

This module provides functions for computing semantic similarity between
messages using cosine similarity on embeddings.
"""

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(
    vector1: NDArray[np.float32], vector2: NDArray[np.float32]
) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vector1: First embedding vector
        vector2: Second embedding vector

    Returns:
        Cosine similarity score between 0 and 1 (higher = more similar)

    Raises:
        ValueError: If vectors have different dimensions or are empty
    """
    if vector1.shape != vector2.shape:
        raise ValueError(
            f"Vectors must have same shape. Got {vector1.shape} and {vector2.shape}"
        )

    if len(vector1) == 0:
        raise ValueError("Vectors cannot be empty")

    # Compute dot product
    dot_product = np.dot(vector1, vector2)

    # Compute magnitudes
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Handle zero vectors
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Compute cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    # Clip to [0, 1] range (embeddings should be normalized)
    # Cosine similarity is in [-1, 1], but for text embeddings we expect [0, 1]
    return float(np.clip(similarity, 0.0, 1.0))


def cosine_distance(
    vector1: NDArray[np.float32], vector2: NDArray[np.float32]
) -> float:
    """
    Compute cosine distance between two vectors.

    Cosine distance = 1 - cosine similarity

    Args:
        vector1: First embedding vector
        vector2: Second embedding vector

    Returns:
        Cosine distance between 0 and 1 (lower = more similar)
    """
    return 1.0 - cosine_similarity(vector1, vector2)


def pairwise_similarity(
    embeddings: List[NDArray[np.float32]],
) -> NDArray[np.float32]:
    """
    Compute pairwise cosine similarity matrix for a list of embeddings.

    Args:
        embeddings: List of embedding vectors

    Returns:
        Square matrix of pairwise similarities (n x n)

    Raises:
        ValueError: If embeddings list is empty or vectors have different shapes
    """
    if not embeddings:
        raise ValueError("Embeddings list cannot be empty")

    n = len(embeddings)

    # Check all embeddings have same shape
    first_shape = embeddings[0].shape
    if not all(emb.shape == first_shape for emb in embeddings):
        raise ValueError("All embeddings must have the same shape")

    # Initialize similarity matrix
    similarity_matrix = np.zeros((n, n), dtype=np.float32)

    # Compute pairwise similarities
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric matrix

    return similarity_matrix


def pairwise_distance(
    embeddings: List[NDArray[np.float32]],
) -> NDArray[np.float32]:
    """
    Compute pairwise cosine distance matrix for a list of embeddings.

    Args:
        embeddings: List of embedding vectors

    Returns:
        Square matrix of pairwise distances (n x n)
    """
    return 1.0 - pairwise_similarity(embeddings)


def average_similarity(
    target_embedding: NDArray[np.float32],
    reference_embeddings: List[NDArray[np.float32]],
) -> float:
    """
    Compute average similarity between a target embedding and a list of reference embeddings.

    Useful for measuring how similar a message is to a cluster of messages.

    Args:
        target_embedding: Target embedding vector
        reference_embeddings: List of reference embedding vectors

    Returns:
        Average cosine similarity (0 to 1)

    Raises:
        ValueError: If reference embeddings list is empty
    """
    if not reference_embeddings:
        raise ValueError("Reference embeddings list cannot be empty")

    similarities = [
        cosine_similarity(target_embedding, ref_emb)
        for ref_emb in reference_embeddings
    ]

    return float(np.mean(similarities))


def max_similarity(
    target_embedding: NDArray[np.float32],
    reference_embeddings: List[NDArray[np.float32]],
) -> Tuple[float, int]:
    """
    Find maximum similarity between a target embedding and a list of reference embeddings.

    Args:
        target_embedding: Target embedding vector
        reference_embeddings: List of reference embedding vectors

    Returns:
        Tuple of (max_similarity, index_of_most_similar)

    Raises:
        ValueError: If reference embeddings list is empty
    """
    if not reference_embeddings:
        raise ValueError("Reference embeddings list cannot be empty")

    similarities = [
        cosine_similarity(target_embedding, ref_emb)
        for ref_emb in reference_embeddings
    ]

    max_sim = max(similarities)
    max_idx = similarities.index(max_sim)

    return max_sim, max_idx


def compute_cluster_cohesion(
    cluster_embeddings: List[NDArray[np.float32]],
) -> float:
    """
    Compute cohesion score for a cluster (average intra-cluster similarity).

    Args:
        cluster_embeddings: List of embeddings in the cluster

    Returns:
        Average pairwise similarity within cluster (0 to 1)

    Raises:
        ValueError: If cluster has fewer than 2 embeddings
    """
    if len(cluster_embeddings) < 2:
        raise ValueError("Cluster must have at least 2 embeddings to compute cohesion")

    similarity_matrix = pairwise_similarity(cluster_embeddings)

    # Get upper triangle (excluding diagonal) to avoid counting each pair twice
    n = len(cluster_embeddings)
    upper_triangle_indices = np.triu_indices(n, k=1)
    pairwise_sims = similarity_matrix[upper_triangle_indices]

    return float(np.mean(pairwise_sims))


def compute_cluster_separation(
    cluster1_embeddings: List[NDArray[np.float32]],
    cluster2_embeddings: List[NDArray[np.float32]],
) -> float:
    """
    Compute separation score between two clusters (average inter-cluster distance).

    Args:
        cluster1_embeddings: Embeddings from first cluster
        cluster2_embeddings: Embeddings from second cluster

    Returns:
        Average cosine distance between clusters (0 to 1, higher = more separated)

    Raises:
        ValueError: If either cluster is empty
    """
    if not cluster1_embeddings or not cluster2_embeddings:
        raise ValueError("Both clusters must be non-empty")

    distances = []
    for emb1 in cluster1_embeddings:
        for emb2 in cluster2_embeddings:
            distances.append(cosine_distance(emb1, emb2))

    return float(np.mean(distances))
