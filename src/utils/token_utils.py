"""
Token counting and cost tracking utilities for Claude API interactions.

This module provides utilities for counting tokens, estimating costs,
and tracking API usage against quotas.
"""

from datetime import datetime, timedelta
from typing import Optional

import tiktoken


class TokenCounter:
    """Token counter for Claude API calls using tiktoken."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        """
        Initialize token counter.

        Args:
            model: Claude model name for token counting
        """
        self.model = model
        # Use cl100k_base encoding for Claude models (same as GPT-4)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_message_tokens(self, messages: list[dict]) -> int:
        """
        Count tokens in a list of messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Total number of tokens
        """
        total = 0
        for message in messages:
            # Count tokens for role
            total += len(self.encoding.encode(message.get("role", "")))
            # Count tokens for content
            total += len(self.encoding.encode(message.get("content", "")))
            # Add overhead per message (approximately 4 tokens per message)
            total += 4

        # Add overhead for request formatting
        total += 3

        return total

    def estimate_prompt_tokens(self, prompt: str, system_prompt: Optional[str] = None) -> int:
        """
        Estimate tokens for a prompt including system prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Estimated token count
        """
        total = self.count_tokens(prompt)
        if system_prompt:
            total += self.count_tokens(system_prompt)
        # Add overhead for formatting
        total += 10
        return total


class CostTracker:
    """Track API costs and quota usage."""

    # Pricing per 1M tokens (as of December 2024)
    # These are example prices - update with actual pricing
    PRICING = {
        "claude-sonnet-4-5-20250929": {
            "input": 3.00,  # $3 per 1M input tokens
            "output": 15.00,  # $15 per 1M output tokens
        },
        "claude-opus-4-5-20251101": {
            "input": 15.00,  # $15 per 1M input tokens
            "output": 75.00,  # $75 per 1M output tokens
        },
        "claude-3-5-haiku-20241022": {
            "input": 0.80,  # $0.80 per 1M input tokens
            "output": 4.00,  # $4 per 1M output tokens
        },
    }

    def __init__(self, model: str = "claude-sonnet-4-5-20250929", daily_quota: int = 1000000):
        """
        Initialize cost tracker.

        Args:
            model: Claude model name
            daily_quota: Daily token quota
        """
        self.model = model
        self.daily_quota = daily_quota
        self.usage_history: list[dict] = []
        self.current_day: Optional[datetime] = None
        self.daily_usage = 0

    def add_usage(self, input_tokens: int, output_tokens: int) -> dict:
        """
        Add API usage and calculate cost.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used

        Returns:
            Dictionary with cost details
        """
        now = datetime.now()

        # Reset daily usage if it's a new day
        if self.current_day is None or now.date() > self.current_day.date():
            self.current_day = now
            self.daily_usage = 0

        # Calculate cost
        pricing = self.PRICING.get(self.model, self.PRICING["claude-sonnet-4-5-20250929"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        # Update daily usage
        total_tokens = input_tokens + output_tokens
        self.daily_usage += total_tokens

        # Record usage
        usage_record = {
            "timestamp": now,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "model": self.model,
        }
        self.usage_history.append(usage_record)

        return usage_record

    def get_daily_usage(self) -> dict:
        """
        Get current daily usage statistics.

        Returns:
            Dictionary with daily usage stats
        """
        quota_used_pct = (self.daily_usage / self.daily_quota) * 100 if self.daily_quota > 0 else 0
        quota_remaining = max(0, self.daily_quota - self.daily_usage)

        return {
            "daily_usage": self.daily_usage,
            "daily_quota": self.daily_quota,
            "quota_remaining": quota_remaining,
            "quota_used_percent": quota_used_pct,
            "is_near_limit": quota_used_pct >= 80,
            "is_over_limit": self.daily_usage >= self.daily_quota,
        }

    def get_total_cost(self, days: int = 1) -> float:
        """
        Get total cost for the last N days.

        Args:
            days: Number of days to calculate cost for

        Returns:
            Total cost in dollars
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_usage = [u for u in self.usage_history if u["timestamp"] >= cutoff]
        return sum(u["total_cost"] for u in recent_usage)

    def get_usage_summary(self, days: int = 1) -> dict:
        """
        Get usage summary for the last N days.

        Args:
            days: Number of days to summarize

        Returns:
            Dictionary with usage summary
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_usage = [u for u in self.usage_history if u["timestamp"] >= cutoff]

        if not recent_usage:
            return {
                "days": days,
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_tokens_per_call": 0,
            }

        total_input = sum(u["input_tokens"] for u in recent_usage)
        total_output = sum(u["output_tokens"] for u in recent_usage)
        total_tokens = sum(u["total_tokens"] for u in recent_usage)
        total_cost = sum(u["total_cost"] for u in recent_usage)

        return {
            "days": days,
            "total_calls": len(recent_usage),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_tokens_per_call": total_tokens / len(recent_usage),
        }

    def check_quota(self) -> tuple[bool, str]:
        """
        Check if there's remaining quota.

        Returns:
            Tuple of (has_quota, message)
        """
        stats = self.get_daily_usage()

        if stats["is_over_limit"]:
            return False, f"Daily quota exceeded: {self.daily_usage}/{self.daily_quota} tokens"

        if stats["is_near_limit"]:
            return (
                True,
                f"Warning: {stats['quota_used_percent']:.1f}% of daily quota used "
                f"({self.daily_usage}/{self.daily_quota} tokens)",
            )

        return True, f"Quota OK: {stats['quota_remaining']} tokens remaining"


# Global instances
_token_counter: Optional[TokenCounter] = None
_cost_tracker: Optional[CostTracker] = None


def get_token_counter(model: str = "claude-sonnet-4-5-20250929") -> TokenCounter:
    """
    Get global token counter instance.

    Args:
        model: Claude model name

    Returns:
        TokenCounter instance
    """
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter(model)
    return _token_counter


def get_cost_tracker(
    model: str = "claude-sonnet-4-5-20250929", daily_quota: int = 1000000
) -> CostTracker:
    """
    Get global cost tracker instance.

    Args:
        model: Claude model name
        daily_quota: Daily token quota

    Returns:
        CostTracker instance
    """
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker(model, daily_quota)
    return _cost_tracker
