"""
Temporal utilities for ranking features.

Provides time-based calculations for recency, activity bursts,
and temporal relevance scoring.
"""

import math
from datetime import datetime, timedelta
from typing import List, Optional, Tuple


def calculate_recency_score(
    message_time: datetime,
    reference_time: Optional[datetime] = None,
    half_life_days: float = 30.0
) -> float:
    """
    Calculate exponential decay recency score.

    Uses exponential decay: score = e^(-λt) where λ = ln(2)/half_life

    Args:
        message_time: When the message was created
        reference_time: Reference time (default: now)
        half_life_days: Days for score to decay to 0.5

    Returns:
        Recency score in [0, 1] range (1 = just now, 0.5 = half_life ago)

    Example:
        >>> now = datetime(2024, 1, 15, 12, 0, 0)
        >>> recent = datetime(2024, 1, 15, 11, 0, 0)  # 1 hour ago
        >>> old = datetime(2024, 1, 1, 12, 0, 0)      # 14 days ago
        >>> calculate_recency_score(recent, now, half_life_days=7.0)
        0.99  # Very recent
        >>> calculate_recency_score(old, now, half_life_days=7.0)
        0.25  # 2 half-lives old
    """
    if reference_time is None:
        reference_time = datetime.utcnow()

    # Calculate age in days
    age_seconds = (reference_time - message_time).total_seconds()
    age_days = age_seconds / 86400.0

    # Exponential decay: e^(-λt) where λ = ln(2)/half_life
    decay_constant = math.log(2) / half_life_days
    score = math.exp(-decay_constant * age_days)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def detect_activity_burst(
    timestamps: List[datetime],
    reference_time: Optional[datetime] = None,
    window_hours: float = 24.0,
    baseline_multiplier: float = 3.0
) -> float:
    """
    Detect recent activity bursts compared to historical baseline.

    A burst is detected when recent activity significantly exceeds
    the historical average rate.

    Args:
        timestamps: List of activity timestamps (sorted or unsorted)
        reference_time: Reference time (default: now)
        window_hours: Recent window to check for bursts
        baseline_multiplier: Threshold for burst detection

    Returns:
        Burst score in [0, 1] range:
        - 0.0: No recent activity
        - 0.5: Normal activity rate
        - 1.0: Strong burst (>= baseline_multiplier × average)

    Example:
        >>> timestamps = [
        ...     datetime(2024, 1, 1),  # 14 days ago
        ...     datetime(2024, 1, 14, 10, 0),  # Recent burst
        ...     datetime(2024, 1, 14, 11, 0),
        ...     datetime(2024, 1, 14, 12, 0),
        ... ]
        >>> detect_activity_burst(timestamps, datetime(2024, 1, 15))
        0.85  # High burst score
    """
    if not timestamps:
        return 0.0

    if reference_time is None:
        reference_time = datetime.utcnow()

    # Split into recent and historical
    recent_cutoff = reference_time - timedelta(hours=window_hours)

    recent_count = sum(1 for t in timestamps if t >= recent_cutoff)
    total_count = len(timestamps)

    if total_count == 0:
        return 0.0

    # Calculate rates (events per hour)
    recent_rate = recent_count / window_hours

    # Historical rate (excluding recent window)
    if timestamps:
        earliest = min(timestamps)
        total_hours = (reference_time - earliest).total_seconds() / 3600.0

        if total_hours > window_hours:
            historical_hours = total_hours - window_hours
            historical_count = total_count - recent_count
            historical_rate = historical_count / historical_hours if historical_hours > 0 else 0.0
        else:
            # Not enough history, use total rate
            historical_rate = total_count / total_hours if total_hours > 0 else 0.0
    else:
        historical_rate = 0.0

    # Avoid division by zero
    if historical_rate == 0:
        return 1.0 if recent_count > 0 else 0.0

    # Calculate burst ratio
    burst_ratio = recent_rate / historical_rate

    # Normalize to [0, 1] range
    # 0.0 = no activity, 0.5 = baseline, 1.0 = baseline_multiplier × baseline
    if burst_ratio < 1.0:
        # Below baseline
        score = 0.5 * burst_ratio
    else:
        # Above baseline
        normalized = (burst_ratio - 1.0) / (baseline_multiplier - 1.0)
        score = 0.5 + 0.5 * min(1.0, normalized)

    return max(0.0, min(1.0, score))


def calculate_response_velocity(
    thread_timestamps: List[datetime],
    window_hours: float = 24.0
) -> float:
    """
    Calculate response velocity in a conversation thread.

    Measures how quickly responses are happening in a thread,
    indicating active discussion.

    Args:
        thread_timestamps: Timestamps of messages in thread (chronological)
        window_hours: Time window to measure velocity

    Returns:
        Velocity score in [0, 1] range:
        - 0.0: No responses
        - 0.5: ~1 response per hour
        - 1.0: Very active (>= 10 responses per hour)

    Example:
        >>> timestamps = [
        ...     datetime(2024, 1, 15, 10, 0),
        ...     datetime(2024, 1, 15, 10, 15),
        ...     datetime(2024, 1, 15, 10, 30),
        ...     datetime(2024, 1, 15, 10, 45),
        ... ]
        >>> calculate_response_velocity(timestamps)
        0.75  # 4 messages in 45 minutes = high velocity
    """
    if len(thread_timestamps) < 2:
        return 0.0

    # Sort timestamps
    sorted_times = sorted(thread_timestamps)

    # Calculate time span
    time_span = (sorted_times[-1] - sorted_times[0]).total_seconds() / 3600.0

    if time_span == 0:
        return 1.0  # All messages at once = very high velocity

    # Responses per hour (excluding first message)
    response_count = len(sorted_times) - 1
    responses_per_hour = response_count / time_span

    # Normalize to [0, 1] with 10 responses/hour as max
    max_velocity = 10.0
    score = responses_per_hour / max_velocity

    return max(0.0, min(1.0, score))


def calculate_temporal_relevance(
    message_time: datetime,
    query_context_time: Optional[datetime] = None,
    relevance_window_days: float = 90.0
) -> float:
    """
    Calculate temporal relevance based on query context.

    Some queries have temporal context (e.g., "recent OAuth discussion").
    This scores messages based on their temporal alignment.

    Args:
        message_time: When the message was created
        query_context_time: Temporal context of query (default: now)
        relevance_window_days: Days around context time considered relevant

    Returns:
        Temporal relevance score in [0, 1] range:
        - 1.0: Within relevance window
        - 0.5: At window boundary
        - 0.0: Far outside window

    Example:
        >>> context = datetime(2024, 1, 15)
        >>> within = datetime(2024, 1, 10)  # 5 days before
        >>> boundary = datetime(2023, 10, 15)  # 90 days before
        >>> far = datetime(2023, 1, 1)  # 1 year before
        >>> calculate_temporal_relevance(within, context, 90)
        1.0  # Within window
        >>> calculate_temporal_relevance(boundary, context, 90)
        0.5  # At boundary
    """
    if query_context_time is None:
        query_context_time = datetime.utcnow()

    # Calculate distance from context time
    distance_days = abs((message_time - query_context_time).total_seconds()) / 86400.0

    # Score based on distance from context
    if distance_days <= relevance_window_days / 2:
        # Within core relevance window
        score = 1.0
    elif distance_days <= relevance_window_days:
        # In outer relevance window (linear decay)
        normalized = (distance_days - relevance_window_days / 2) / (relevance_window_days / 2)
        score = 1.0 - 0.5 * normalized
    else:
        # Outside window (exponential decay)
        excess_days = distance_days - relevance_window_days
        decay_rate = math.log(2) / relevance_window_days
        score = 0.5 * math.exp(-decay_rate * excess_days)

    return max(0.0, min(1.0, score))


def calculate_edit_freshness(
    created_time: datetime,
    edited_time: Optional[datetime] = None,
    reference_time: Optional[datetime] = None,
    half_life_days: float = 7.0
) -> float:
    """
    Calculate freshness score based on most recent edit.

    Edited content may be more relevant than old content.

    Args:
        created_time: Original creation time
        edited_time: Last edit time (None if never edited)
        reference_time: Reference time (default: now)
        half_life_days: Decay half-life for edited content

    Returns:
        Freshness score in [0, 1] range based on most recent timestamp

    Example:
        >>> created = datetime(2024, 1, 1)
        >>> edited = datetime(2024, 1, 14)
        >>> now = datetime(2024, 1, 15)
        >>> calculate_edit_freshness(created, edited, now, half_life_days=7)
        0.91  # Recent edit = high freshness
        >>> calculate_edit_freshness(created, None, now, half_life_days=7)
        0.25  # Old creation, no edits = low freshness
    """
    if reference_time is None:
        reference_time = datetime.utcnow()

    # Use most recent time
    most_recent = edited_time if edited_time else created_time

    # Calculate recency of most recent change
    return calculate_recency_score(most_recent, reference_time, half_life_days)


def time_window_overlap(
    start1: datetime,
    end1: datetime,
    start2: datetime,
    end2: datetime
) -> Tuple[bool, float]:
    """
    Calculate overlap between two time windows.

    Args:
        start1, end1: First time window
        start2, end2: Second time window

    Returns:
        Tuple of (has_overlap, overlap_ratio)
        - has_overlap: Whether windows overlap
        - overlap_ratio: Fraction of shorter window that overlaps [0, 1]

    Example:
        >>> start1 = datetime(2024, 1, 1)
        >>> end1 = datetime(2024, 1, 10)
        >>> start2 = datetime(2024, 1, 5)
        >>> end2 = datetime(2024, 1, 15)
        >>> time_window_overlap(start1, end1, start2, end2)
        (True, 0.55)  # 5 days overlap out of 9 days (shorter window)
    """
    # Find overlap window
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_start >= overlap_end:
        return (False, 0.0)

    # Calculate overlap duration
    overlap_duration = (overlap_end - overlap_start).total_seconds()

    # Calculate window durations
    window1_duration = (end1 - start1).total_seconds()
    window2_duration = (end2 - start2).total_seconds()

    # Use shorter window as denominator
    shorter_duration = min(window1_duration, window2_duration)

    if shorter_duration == 0:
        return (True, 1.0) if overlap_duration > 0 else (False, 0.0)

    overlap_ratio = overlap_duration / shorter_duration

    return (True, max(0.0, min(1.0, overlap_ratio)))
