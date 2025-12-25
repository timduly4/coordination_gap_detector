"""
Claude API client wrapper with retry logic, rate limiting, and structured output.

This module provides a production-ready wrapper around the Anthropic Claude API
with features like:
- Exponential backoff retry logic
- Rate limiting and quota management
- Structured JSON output parsing
- Token counting and cost tracking
- Comprehensive error handling
"""

import json
import logging
import time
from typing import Any, Optional

from anthropic import Anthropic, APIError, RateLimitError, APITimeoutError
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.config import get_settings
from src.utils.token_utils import get_token_counter, get_cost_tracker

logger = logging.getLogger(__name__)


class ClaudeResponse(BaseModel):
    """Structured response from Claude API."""

    content: str
    model: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class ClaudeClient:
    """
    Production-ready Claude API client with reliability features.

    Features:
    - Automatic retries with exponential backoff
    - Rate limiting and quota tracking
    - Token counting and cost estimation
    - Structured JSON parsing
    - Comprehensive error handling
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        daily_quota_tokens: Optional[int] = None,
    ):
        """
        Initialize Claude API client.

        Args:
            api_key: Anthropic API key (defaults to settings)
            model: Claude model to use (defaults to settings)
            max_tokens: Maximum tokens for responses (defaults to settings)
            temperature: Temperature for completions (defaults to settings)
            daily_quota_tokens: Daily token quota (defaults to settings)
        """
        settings = get_settings()

        self.api_key = api_key or settings.anthropic_api_key
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or settings")

        self.model = model or settings.claude_model
        self.max_tokens = max_tokens or settings.claude_max_tokens
        self.temperature = temperature or settings.claude_temperature
        self.daily_quota = daily_quota_tokens or settings.claude_daily_quota_tokens

        # Initialize Anthropic client
        self.client = Anthropic(api_key=self.api_key)

        # Initialize token counter and cost tracker
        self.token_counter = get_token_counter(self.model)
        self.cost_tracker = get_cost_tracker(self.model, self.daily_quota)

        logger.info(f"Initialized ClaudeClient with model: {self.model}")

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> ClaudeResponse:
        """
        Generate a completion from Claude with retry logic.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Temperature for completion (overrides default)
            **kwargs: Additional arguments for Claude API

        Returns:
            ClaudeResponse with content and metadata

        Raises:
            ValueError: If quota exceeded or invalid parameters
            APIError: If API call fails after retries
        """
        # Check quota before making request
        has_quota, quota_msg = self.cost_tracker.check_quota()
        if not has_quota:
            raise ValueError(quota_msg)

        if "is_near_limit" in quota_msg:
            logger.warning(quota_msg)

        # Estimate input tokens
        input_token_estimate = self.token_counter.estimate_prompt_tokens(prompt, system_prompt)
        logger.info(f"Estimated input tokens: {input_token_estimate}")

        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        # Prepare API call parameters
        api_kwargs = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "messages": messages,
        }

        if system_prompt:
            api_kwargs["system"] = system_prompt

        # Add any additional kwargs
        api_kwargs.update(kwargs)

        try:
            # Make API call
            logger.debug(f"Calling Claude API with model: {self.model}")
            start_time = time.time()

            response = self.client.messages.create(**api_kwargs)

            elapsed = time.time() - start_time
            logger.info(f"Claude API call completed in {elapsed:.2f}s")

            # Extract response content
            content = response.content[0].text if response.content else ""

            # Track usage and cost
            usage_record = self.cost_tracker.add_usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            logger.info(
                f"Token usage - Input: {response.usage.input_tokens}, "
                f"Output: {response.usage.output_tokens}, "
                f"Cost: ${usage_record['total_cost']:.4f}"
            )

            return ClaudeResponse(
                content=content,
                model=response.model,
                stop_reason=response.stop_reason,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=usage_record["total_cost"],
            )

        except RateLimitError as e:
            logger.warning(f"Rate limit hit, will retry: {e}")
            raise
        except APITimeoutError as e:
            logger.warning(f"API timeout, will retry: {e}")
            raise
        except APIError as e:
            logger.error(f"Claude API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Claude API call: {e}")
            raise

    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Generate a JSON completion from Claude with structured parsing.

        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: Optional system prompt
            schema: Optional Pydantic model for validation
            **kwargs: Additional arguments for Claude API

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If response is not valid JSON or doesn't match schema
            APIError: If API call fails
        """
        response = self.complete(prompt, system_prompt, **kwargs)

        try:
            # Parse JSON from response
            parsed = json.loads(response.content)

            # Validate against schema if provided
            if schema:
                try:
                    validated = schema(**parsed)
                    return validated.model_dump()
                except ValidationError as e:
                    logger.error(f"JSON response doesn't match schema: {e}")
                    raise ValueError(f"Response validation failed: {e}")

            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Claude response: {e}")
            logger.debug(f"Response content: {response.content[:500]}")
            raise ValueError(f"Invalid JSON in response: {e}")

    def parse_json(self, response: ClaudeResponse) -> dict:
        """
        Parse JSON from a Claude response.

        Args:
            response: ClaudeResponse to parse

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If response is not valid JSON
        """
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            raise ValueError(f"Invalid JSON in response: {e}")

    def get_usage_stats(self, days: int = 1) -> dict:
        """
        Get API usage statistics.

        Args:
            days: Number of days to get stats for

        Returns:
            Dictionary with usage statistics
        """
        return self.cost_tracker.get_usage_summary(days)

    def get_daily_usage(self) -> dict:
        """
        Get current daily usage statistics.

        Returns:
            Dictionary with daily usage stats
        """
        return self.cost_tracker.get_daily_usage()

    def estimate_cost(self, prompt: str, max_output_tokens: int = 1000) -> dict:
        """
        Estimate cost for a prompt.

        Args:
            prompt: Prompt to estimate cost for
            max_output_tokens: Estimated output tokens

        Returns:
            Dictionary with cost estimate
        """
        input_tokens = self.token_counter.count_tokens(prompt)

        # Get pricing for current model
        pricing = self.cost_tracker.PRICING.get(
            self.model, self.cost_tracker.PRICING["claude-sonnet-4-5-20250929"]
        )

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (max_output_tokens / 1_000_000) * pricing["output"]

        return {
            "input_tokens": input_tokens,
            "estimated_output_tokens": max_output_tokens,
            "input_cost_usd": input_cost,
            "estimated_output_cost_usd": output_cost,
            "total_estimated_cost_usd": input_cost + output_cost,
        }

    async def complete_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> ClaudeResponse:
        """
        Async version of complete method.

        Currently wraps sync version. Future enhancement: implement true async.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for completion
            **kwargs: Additional arguments for Claude API

        Returns:
            ClaudeResponse with content and metadata
        """
        # For now, wrap sync version
        # TODO: Implement true async support with asyncio
        return self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )


class AsyncClaudeClient(ClaudeClient):
    """
    Async version of Claude client (placeholder for future async support).

    Note: Currently wraps sync client. Can be extended with asyncio
    when needed for high-throughput scenarios.
    """

    async def acomplete(self, *args: Any, **kwargs: Any) -> ClaudeResponse:
        """Async complete method (currently wraps sync)."""
        # TODO: Implement true async support with asyncio
        return self.complete(*args, **kwargs)

    async def acomplete_json(self, *args: Any, **kwargs: Any) -> dict:
        """Async complete_json method (currently wraps sync)."""
        # TODO: Implement true async support with asyncio
        return self.complete_json(*args, **kwargs)
