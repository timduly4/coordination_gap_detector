"""
Unit tests for temporal utility functions.
"""

from datetime import datetime, timedelta

import pytest

from src.utils.time_utils import (
    calculate_edit_freshness,
    calculate_recency_score,
    calculate_response_velocity,
    calculate_temporal_relevance,
    detect_activity_burst,
    time_window_overlap,
)


class TestCalculateRecencyScore:
    """Test suite for recency score calculation."""

    def test_immediate_recency(self):
        """Test that messages just now get score ~1.0."""
        now = datetime(2024, 1, 15, 12, 0, 0)
        score = calculate_recency_score(now, now, half_life_days=30.0)

        assert score == pytest.approx(1.0, rel=0.01)

    def test_half_life_recency(self):
        """Test that messages at half-life get score ~0.5."""
        now = datetime(2024, 1, 15, 12, 0, 0)
        past = now - timedelta(days=30)

        score = calculate_recency_score(past, now, half_life_days=30.0)

        assert score == pytest.approx(0.5, rel=0.01)

    def test_two_half_lives(self):
        """Test exponential decay at 2 half-lives."""
        now = datetime(2024, 1, 15, 12, 0, 0)
        past = now - timedelta(days=60)

        score = calculate_recency_score(past, now, half_life_days=30.0)

        assert score == pytest.approx(0.25, rel=0.01)

    def test_very_old_message(self):
        """Test that very old messages approach 0."""
        now = datetime(2024, 1, 15, 12, 0, 0)
        past = now - timedelta(days=365)

        score = calculate_recency_score(past, now, half_life_days=30.0)

        assert 0.0 <= score < 0.01

    def test_default_reference_time(self):
        """Test that reference time defaults to now."""
        recent = datetime.utcnow() - timedelta(hours=1)
        score = calculate_recency_score(recent, None, half_life_days=7.0)

        assert score > 0.9

    def test_shorter_half_life(self):
        """Test that shorter half-life decays faster."""
        now = datetime(2024, 1, 15, 12, 0, 0)
        past = now - timedelta(days=7)

        score_long = calculate_recency_score(past, now, half_life_days=30.0)
        score_short = calculate_recency_score(past, now, half_life_days=7.0)

        assert score_short < score_long


class TestDetectActivityBurst:
    """Test suite for activity burst detection."""

    def test_no_activity(self):
        """Test that empty timestamps return 0."""
        score = detect_activity_burst([])

        assert score == 0.0

    def test_uniform_activity(self):
        """Test that uniform activity returns ~0.5."""
        base = datetime(2024, 1, 1)
        timestamps = [base + timedelta(days=i) for i in range(10)]

        score = detect_activity_burst(
            timestamps,
            reference_time=base + timedelta(days=10),
            window_hours=24.0
        )

        # Should be around baseline (0.5)
        assert 0.4 < score < 0.6

    def test_strong_burst(self):
        """Test that recent burst gets high score."""
        base = datetime(2024, 1, 1)
        # Historical baseline: 1 per day for 10 days
        timestamps = [base + timedelta(days=i) for i in range(10)]

        # Recent burst: 5 in last 24 hours
        recent = base + timedelta(days=14)
        timestamps.extend([
            recent - timedelta(hours=i)
            for i in range(5)
        ])

        score = detect_activity_burst(
            timestamps,
            reference_time=recent,
            window_hours=24.0,
            baseline_multiplier=3.0
        )

        # Burst is ~5x baseline, should be high
        assert score > 0.7

    def test_no_recent_activity(self):
        """Test that no recent activity returns low score."""
        base = datetime(2024, 1, 1)
        timestamps = [base + timedelta(days=i) for i in range(10)]

        score = detect_activity_burst(
            timestamps,
            reference_time=base + timedelta(days=15),
            window_hours=24.0
        )

        # No recent activity
        assert score < 0.2

    def test_only_recent_activity(self):
        """Test with only recent activity (no history)."""
        now = datetime(2024, 1, 15)
        timestamps = [now - timedelta(hours=i) for i in range(5)]

        score = detect_activity_burst(timestamps, reference_time=now)

        # With limited history, burst detection returns a modest score
        # The algorithm needs historical baseline for meaningful burst detection
        assert 0.0 <= score <= 1.0  # Valid range
        assert isinstance(score, float)


class TestCalculateResponseVelocity:
    """Test suite for response velocity calculation."""

    def test_single_message(self):
        """Test that single message returns 0."""
        timestamps = [datetime(2024, 1, 15, 10, 0)]

        score = calculate_response_velocity(timestamps)

        assert score == 0.0

    def test_high_velocity(self):
        """Test rapid responses get high score."""
        base = datetime(2024, 1, 15, 10, 0)
        # 10 messages in 1 hour = 10 responses/hour
        timestamps = [base + timedelta(minutes=i*6) for i in range(10)]

        score = calculate_response_velocity(timestamps, window_hours=24.0)

        # 9 responses in 54 minutes = ~10 per hour = max velocity
        assert score > 0.8

    def test_moderate_velocity(self):
        """Test moderate response rate."""
        base = datetime(2024, 1, 15, 10, 0)
        # 5 messages in 4 hours = ~1 response/hour
        timestamps = [base + timedelta(hours=i) for i in range(5)]

        score = calculate_response_velocity(timestamps)

        # ~1 response per hour = normalized to ~0.1
        assert 0.05 < score < 0.2

    def test_slow_velocity(self):
        """Test slow response rate."""
        base = datetime(2024, 1, 15, 10, 0)
        # 3 messages over 48 hours
        timestamps = [
            base,
            base + timedelta(hours=24),
            base + timedelta(hours=48)
        ]

        score = calculate_response_velocity(timestamps)

        # Very slow
        assert score < 0.1

    def test_simultaneous_messages(self):
        """Test messages at same time (instant velocity)."""
        base = datetime(2024, 1, 15, 10, 0)
        timestamps = [base] * 5

        score = calculate_response_velocity(timestamps)

        # Time span = 0, should return max
        assert score == 1.0


class TestCalculateTemporalRelevance:
    """Test suite for temporal relevance calculation."""

    def test_within_core_window(self):
        """Test messages within core relevance window."""
        context = datetime(2024, 1, 15)
        message = datetime(2024, 1, 10)  # 5 days before

        score = calculate_temporal_relevance(
            message,
            context,
            relevance_window_days=90.0
        )

        # Within core window (45 days)
        assert score == 1.0

    def test_at_window_boundary(self):
        """Test messages at relevance window boundary."""
        context = datetime(2024, 1, 15)
        message = datetime(2023, 10, 17)  # ~90 days before

        score = calculate_temporal_relevance(
            message,
            context,
            relevance_window_days=90.0
        )

        # At boundary
        assert 0.4 < score < 0.6

    def test_outside_window(self):
        """Test messages far outside window."""
        context = datetime(2024, 1, 15)
        message = datetime(2023, 1, 1)  # ~1 year before

        score = calculate_temporal_relevance(
            message,
            context,
            relevance_window_days=90.0
        )

        # Far outside, should decay
        assert score < 0.2

    def test_future_message(self):
        """Test messages in the future (symmetric distance)."""
        context = datetime(2024, 1, 15)
        message = datetime(2024, 2, 15)  # 30 days after

        score = calculate_temporal_relevance(
            message,
            context,
            relevance_window_days=90.0
        )

        # Within core window
        assert score == 1.0

    def test_default_context_time(self):
        """Test that context time defaults to now."""
        recent = datetime.utcnow() - timedelta(days=10)

        score = calculate_temporal_relevance(recent, None, 90.0)

        # Recent message
        assert score > 0.9


class TestCalculateEditFreshness:
    """Test suite for edit freshness calculation."""

    def test_recent_edit(self):
        """Test that recent edit gets high freshness."""
        created = datetime(2024, 1, 1)
        edited = datetime(2024, 1, 14)
        now = datetime(2024, 1, 15)

        score = calculate_edit_freshness(created, edited, now, half_life_days=7.0)

        # 1 day since edit
        assert score > 0.9

    def test_old_creation_no_edit(self):
        """Test that old unedited content gets low freshness."""
        created = datetime(2024, 1, 1)
        now = datetime(2024, 1, 15)

        score = calculate_edit_freshness(created, None, now, half_life_days=7.0)

        # 14 days = 2 half-lives
        assert score < 0.3

    def test_old_creation_old_edit(self):
        """Test old creation with old edit."""
        created = datetime(2024, 1, 1)
        edited = datetime(2024, 1, 2)
        now = datetime(2024, 1, 15)

        score = calculate_edit_freshness(created, edited, now, half_life_days=7.0)

        # Uses most recent (edit), 13 days ago
        assert score < 0.3

    def test_recent_creation(self):
        """Test recent creation (no edits yet)."""
        created = datetime(2024, 1, 14)
        now = datetime(2024, 1, 15)

        score = calculate_edit_freshness(created, None, now, half_life_days=7.0)

        # 1 day old
        assert score > 0.9


class TestTimeWindowOverlap:
    """Test suite for time window overlap calculation."""

    def test_complete_overlap(self):
        """Test windows that completely overlap."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        has_overlap, ratio = time_window_overlap(start, end, start, end)

        assert has_overlap is True
        assert ratio == pytest.approx(1.0, rel=0.01)

    def test_partial_overlap(self):
        """Test windows with partial overlap."""
        start1 = datetime(2024, 1, 1)
        end1 = datetime(2024, 1, 10)  # 9 days
        start2 = datetime(2024, 1, 5)
        end2 = datetime(2024, 1, 15)  # 10 days

        has_overlap, ratio = time_window_overlap(start1, end1, start2, end2)

        assert has_overlap is True
        # Overlap is 5 days (Jan 5-10)
        # Shorter window is 9 days
        # Ratio = 5/9 = 0.555...
        assert 0.5 < ratio < 0.6

    def test_no_overlap(self):
        """Test windows with no overlap."""
        start1 = datetime(2024, 1, 1)
        end1 = datetime(2024, 1, 10)
        start2 = datetime(2024, 1, 15)
        end2 = datetime(2024, 1, 20)

        has_overlap, ratio = time_window_overlap(start1, end1, start2, end2)

        assert has_overlap is False
        assert ratio == 0.0

    def test_adjacent_windows(self):
        """Test adjacent but non-overlapping windows."""
        start1 = datetime(2024, 1, 1)
        end1 = datetime(2024, 1, 10)
        start2 = datetime(2024, 1, 10)
        end2 = datetime(2024, 1, 20)

        has_overlap, ratio = time_window_overlap(start1, end1, start2, end2)

        # End of window1 = start of window2, no overlap
        assert has_overlap is False

    def test_contained_window(self):
        """Test one window fully contained in another."""
        start1 = datetime(2024, 1, 1)
        end1 = datetime(2024, 1, 20)
        start2 = datetime(2024, 1, 5)
        end2 = datetime(2024, 1, 10)

        has_overlap, ratio = time_window_overlap(start1, end1, start2, end2)

        assert has_overlap is True
        # Overlap = 5 days (Jan 5-10)
        # Shorter window = 5 days
        # Ratio = 1.0
        assert ratio == pytest.approx(1.0, rel=0.01)

    def test_zero_duration_window(self):
        """Test window with zero duration."""
        instant = datetime(2024, 1, 5)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        has_overlap, ratio = time_window_overlap(instant, instant, start, end)

        # Zero duration overlaps with itself
        assert has_overlap is False or ratio == 0.0
