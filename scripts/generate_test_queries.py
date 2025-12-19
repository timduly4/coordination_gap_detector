#!/usr/bin/env python3
"""
Generate test queries from mock data for evaluation.

This script creates test queries based on existing mock conversation data.
It generates queries in different categories (factual, technical, temporal, etc.)
and outputs them to a JSONL file for offline evaluation.

Usage:
    # Generate test queries
    python scripts/generate_test_queries.py --output data/test_queries/queries.jsonl --count 50

    # Generate with specific categories
    python scripts/generate_test_queries.py --categories factual,technical --output queries.jsonl
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Message, Source
from src.db.postgres import async_engine, get_db


# Pre-defined test query templates based on mock scenarios
TEST_QUERY_TEMPLATES = {
    "factual": [
        {
            "query_id": "factual_1",
            "query_text": "OAuth implementation",
            "category": "factual",
            "expected_topics": ["oauth", "authentication", "auth0"]
        },
        {
            "query_id": "factual_2",
            "query_text": "Who is working on authentication",
            "category": "factual",
            "expected_topics": ["authentication", "oauth", "team"]
        },
        {
            "query_id": "factual_3",
            "query_text": "Database migration decisions",
            "category": "factual",
            "expected_topics": ["database", "migration", "decision"]
        },
        {
            "query_id": "factual_4",
            "query_text": "API endpoint design",
            "category": "factual",
            "expected_topics": ["api", "endpoint", "design"]
        },
        {
            "query_id": "factual_5",
            "query_text": "Testing strategy discussion",
            "category": "factual",
            "expected_topics": ["testing", "test", "qa"]
        },
    ],
    "technical": [
        {
            "query_id": "technical_1",
            "query_text": "OAuth implementation code",
            "category": "technical",
            "expected_topics": ["oauth", "code", "implementation"]
        },
        {
            "query_id": "technical_2",
            "query_text": "Authentication flow diagram",
            "category": "technical",
            "expected_topics": ["authentication", "flow", "diagram"]
        },
        {
            "query_id": "technical_3",
            "query_text": "Database schema changes",
            "category": "technical",
            "expected_topics": ["database", "schema", "migration"]
        },
        {
            "query_id": "technical_4",
            "query_text": "API rate limiting implementation",
            "category": "technical",
            "expected_topics": ["api", "rate", "limiting"]
        },
        {
            "query_id": "technical_5",
            "query_text": "Caching strategy for API",
            "category": "technical",
            "expected_topics": ["caching", "cache", "api"]
        },
    ],
    "temporal": [
        {
            "query_id": "temporal_1",
            "query_text": "Recent discussions about authentication",
            "category": "temporal",
            "expected_topics": ["authentication", "recent", "discussion"]
        },
        {
            "query_id": "temporal_2",
            "query_text": "Latest decisions on database",
            "category": "temporal",
            "expected_topics": ["database", "latest", "decision"]
        },
        {
            "query_id": "temporal_3",
            "query_text": "Recent bug reports",
            "category": "temporal",
            "expected_topics": ["bug", "recent", "report"]
        },
        {
            "query_id": "temporal_4",
            "query_text": "Today's incident discussions",
            "category": "temporal",
            "expected_topics": ["incident", "today", "discussion"]
        },
        {
            "query_id": "temporal_5",
            "query_text": "This week's feature planning",
            "category": "temporal",
            "expected_topics": ["feature", "planning", "week"]
        },
    ],
    "ambiguous": [
        {
            "query_id": "ambiguous_1",
            "query_text": "Login issues",
            "category": "ambiguous",
            "expected_topics": ["login", "issue", "authentication"]
        },
        {
            "query_id": "ambiguous_2",
            "query_text": "Performance problems",
            "category": "ambiguous",
            "expected_topics": ["performance", "problem", "slow"]
        },
        {
            "query_id": "ambiguous_3",
            "query_text": "API not working",
            "category": "ambiguous",
            "expected_topics": ["api", "error", "bug"]
        },
        {
            "query_id": "ambiguous_4",
            "query_text": "Database issue",
            "category": "ambiguous",
            "expected_topics": ["database", "issue", "error"]
        },
        {
            "query_id": "ambiguous_5",
            "query_text": "Need help with feature",
            "category": "ambiguous",
            "expected_topics": ["help", "feature", "support"]
        },
    ],
    "multi_source": [
        {
            "query_id": "multi_source_1",
            "query_text": "Authentication mentioned in Slack and GitHub",
            "category": "multi_source",
            "expected_topics": ["authentication", "slack", "github"]
        },
        {
            "query_id": "multi_source_2",
            "query_text": "Feature discussions across teams",
            "category": "multi_source",
            "expected_topics": ["feature", "discussion", "team"]
        },
        {
            "query_id": "multi_source_3",
            "query_text": "Cross-team collaboration on API",
            "category": "multi_source",
            "expected_topics": ["api", "collaboration", "team"]
        },
    ],
}


async def get_existing_messages(db: AsyncSession) -> List[Message]:
    """
    Fetch existing messages from the database.

    Args:
        db: Database session

    Returns:
        List of Message objects
    """
    result = await db.execute(
        select(Message).order_by(Message.timestamp.desc()).limit(100)
    )
    messages = result.scalars().all()
    return list(messages)


async def generate_queries_from_messages(
    messages: List[Message],
    max_queries: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate queries based on actual message content.

    Args:
        messages: List of messages to analyze
        max_queries: Maximum number of queries to generate

    Returns:
        List of query dictionaries
    """
    queries = []

    # Extract common terms and topics from messages
    for i, message in enumerate(messages[:max_queries]):
        # Create query based on message content
        # Simple extraction: take first 3-5 words
        words = message.content.split()[:5]
        query_text = " ".join(words)

        query = {
            "query_id": f"generated_{i+1}",
            "query_text": query_text,
            "category": "generated",
            "source_message_id": message.id,
            "channel": message.channel,
            "timestamp": message.timestamp.isoformat()
        }
        queries.append(query)

    return queries


async def generate_test_queries(
    db: AsyncSession,
    categories: List[str],
    count: int = 50
) -> List[Dict[str, Any]]:
    """
    Generate test queries for evaluation.

    Args:
        db: Database session
        categories: List of query categories to include
        count: Target number of queries

    Returns:
        List of test query dictionaries
    """
    all_queries = []

    # Add queries from templates
    for category in categories:
        if category in TEST_QUERY_TEMPLATES:
            queries = TEST_QUERY_TEMPLATES[category]
            all_queries.extend(queries)

    # If we need more queries, generate from actual messages
    if len(all_queries) < count:
        messages = await get_existing_messages(db)
        if messages:
            additional_count = count - len(all_queries)
            generated = await generate_queries_from_messages(
                messages,
                max_queries=additional_count
            )
            all_queries.extend(generated)

    # Limit to requested count
    all_queries = all_queries[:count]

    print(f"✓ Generated {len(all_queries)} test queries")
    print(f"  Categories: {', '.join(categories)}")

    return all_queries


async def save_queries_to_file(
    queries: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Save queries to JSONL file.

    Args:
        queries: List of query dictionaries
        output_path: Path to output file
    """
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for query in queries:
            f.write(json.dumps(query) + '\n')

    print(f"✓ Saved {len(queries)} queries to {output_path}")


async def main(args):
    """Main script execution."""
    print("=" * 60)
    print("Test Query Generation")
    print("=" * 60)

    # Determine categories
    if args.categories:
        categories = [c.strip() for c in args.categories.split(',')]
    else:
        # Use all categories by default
        categories = list(TEST_QUERY_TEMPLATES.keys())

    print(f"\nCategories: {', '.join(categories)}")
    print(f"Target count: {args.count}")
    print(f"Output: {args.output}")
    print()

    # Get database session
    async with async_engine.begin() as conn:
        await conn.run_sync(lambda _: None)  # Ensure connection works

    async for db in get_db():
        try:
            # Generate queries
            queries = await generate_test_queries(
                db=db,
                categories=categories,
                count=args.count
            )

            # Save to file
            output_path = Path(args.output)
            await save_queries_to_file(queries, output_path)

            print()
            print("=" * 60)
            print("Query Generation Complete!")
            print("=" * 60)
            print(f"\nTotal queries: {len(queries)}")
            print(f"Output file: {output_path}")
            print()
            print("Next steps:")
            print("1. Review the generated queries")
            print("2. Create relevance judgments (manually or semi-automatically)")
            print("3. Run evaluation with: python scripts/evaluate_ranking.py")

        finally:
            await db.close()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate test queries for ranking evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/test_queries/queries.jsonl",
        help="Output file path (default: data/test_queries/queries.jsonl)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of queries to generate (default: 50)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories (default: all)"
    )

    args = parser.parse_args()

    asyncio.run(main(args))
