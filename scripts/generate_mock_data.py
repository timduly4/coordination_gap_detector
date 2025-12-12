#!/usr/bin/env python3
"""
Generate and seed mock Slack conversation data into the database and vector store.

This script creates realistic Slack conversation scenarios and stores them
in PostgreSQL and ChromaDB (with embeddings) for development and testing.

Usage:
    # Generate all scenarios with embeddings
    python scripts/generate_mock_data.py --scenarios all

    # Generate specific scenarios
    python scripts/generate_mock_data.py --scenarios oauth_discussion decision_making

    # Clear existing data before generating
    python scripts/generate_mock_data.py --scenarios all --clear

    # Skip embeddings (faster, Postgres only)
    python scripts/generate_mock_data.py --scenarios all --skip-embeddings
"""
import argparse
import asyncio
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.db.models import Message, Source
from src.db.postgres import async_engine, get_db
from src.db.vector_store import get_vector_store
from src.ingestion.slack.mock_client import MockSlackClient, MockMessage


async def ensure_slack_source(db: AsyncSession) -> Source:
    """
    Ensure a Slack source exists in the database.

    Args:
        db: Database session

    Returns:
        The Slack source instance
    """
    # Check if Slack source already exists
    result = await db.execute(
        select(Source).where(Source.type == "slack", Source.name == "Mock Slack Workspace")
    )
    source = result.scalar_one_or_none()

    if not source:
        # Create new Slack source
        source = Source(
            type="slack",
            name="Mock Slack Workspace",
            config={
                "workspace_id": "mock_workspace_001",
                "workspace_name": "Demo Company",
                "channels": [
                    "#platform",
                    "#auth-team",
                    "#backend-eng",
                    "#data-team",
                    "#incidents",
                    "#product",
                    "#engineering",
                    "#frontend",
                ],
            },
        )
        db.add(source)
        await db.flush()  # Flush to get the ID without committing
        print(f"✓ Created Slack source: {source.name} (ID: {source.id})")
    else:
        print(f"✓ Using existing Slack source: {source.name} (ID: {source.id})")

    return source


async def clear_existing_data(db: AsyncSession, clear_embeddings: bool = True) -> None:
    """
    Clear existing mock data from the database and vector store.

    Args:
        db: Database session
        clear_embeddings: Whether to also clear embeddings from vector store
    """
    # Get Slack source
    result = await db.execute(
        select(Source).where(Source.type == "slack", Source.name == "Mock Slack Workspace")
    )
    source = result.scalar_one_or_none()

    if source:
        # Delete all messages from this source
        result = await db.execute(
            select(Message).where(Message.source_id == source.id)
        )
        messages = result.scalars().all()

        for message in messages:
            await db.delete(message)

        await db.commit()
        print(f"✓ Deleted {len(messages)} existing messages from Postgres")

    # Clear vector store
    if clear_embeddings:
        try:
            vector_store = get_vector_store()
            vector_store.clear_collection()
            print(f"✓ Cleared all embeddings from ChromaDB")
        except Exception as e:
            print(f"⚠️  Warning: Failed to clear vector store: {e}")


async def insert_mock_messages(
    db: AsyncSession, source: Source, mock_messages: List[MockMessage]
) -> int:
    """
    Insert mock messages into the database.

    Args:
        db: Database session
        source: The Slack source
        mock_messages: List of mock messages to insert

    Returns:
        Number of messages inserted
    """
    count = 0
    source_id = source.id  # Get ID before loop to avoid detachment issues

    for mock_msg in mock_messages:
        message = Message(
            source_id=source_id,
            content=mock_msg.content,
            external_id=mock_msg.external_id,
            author=mock_msg.author,
            channel=mock_msg.channel,
            thread_id=mock_msg.thread_id,
            timestamp=mock_msg.timestamp,
            message_metadata=mock_msg.metadata,
        )
        db.add(message)
        count += 1

    # Note: Commit is done by caller to avoid session detachment issues
    return count


def populate_vector_store(db_messages: List[Message]) -> int:
    """
    Populate the vector store with embeddings for messages.

    Args:
        db_messages: List of Message objects from the database

    Returns:
        Number of embeddings created
    """
    if not db_messages:
        return 0

    try:
        vector_store = get_vector_store()

        # Prepare data for batch insertion
        message_ids = []
        contents = []
        metadatas = []

        for msg in db_messages:
            message_ids.append(msg.id)
            contents.append(msg.content)
            metadatas.append({
                "source": "slack",
                "channel": msg.channel,
                "author": msg.author,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "external_id": msg.external_id,
            })

        # Batch insert into vector store
        embedding_ids = vector_store.insert_batch(message_ids, contents, metadatas)
        return len(embedding_ids)

    except Exception as e:
        print(f"⚠️  Warning: Failed to populate vector store: {e}")
        return 0


async def generate_mock_data(scenarios: List[str], clear: bool = False, skip_embeddings: bool = False) -> None:
    """
    Generate and insert mock data for specified scenarios.

    Args:
        scenarios: List of scenario names to generate ('all' for all scenarios)
        clear: Whether to clear existing data before generating
        skip_embeddings: Whether to skip generating embeddings (faster for testing)
    """
    settings = get_settings()
    print(f"Generating mock data for environment: {settings.environment}")
    print(f"Database: {settings.postgres_url}")
    print()

    # Initialize mock client
    client = MockSlackClient()
    available_scenarios = client.get_scenario_descriptions()

    # Validate scenarios
    if "all" in scenarios:
        scenarios = list(available_scenarios.keys())
    else:
        invalid = [s for s in scenarios if s not in available_scenarios]
        if invalid:
            print(f"❌ Unknown scenarios: {', '.join(invalid)}")
            print(f"Available scenarios: {', '.join(available_scenarios.keys())}")
            return

    print("Scenarios to generate:")
    for scenario in scenarios:
        print(f"  - {scenario}: {available_scenarios[scenario]}")
    print()

    # Get database session
    async with AsyncSession(async_engine) as db:
        # Clear existing data if requested
        if clear:
            print("Clearing existing data...")
            await clear_existing_data(db, clear_embeddings=not skip_embeddings)
            print()

        # Ensure Slack source exists
        source = await ensure_slack_source(db)
        source_id = source.id  # Save ID before commit to avoid detachment issues
        print()

        # Generate and insert messages for each scenario
        total_messages = 0
        for scenario in scenarios:
            print(f"Generating scenario: {scenario}...")
            messages = client.get_scenario_messages(scenario)
            count = await insert_mock_messages(db, source, messages)
            print(f"✓ Inserted {count} messages into Postgres")
            total_messages += count

        # Commit all messages at once
        await db.commit()
        print()

        # Populate vector store with embeddings
        total_embeddings = 0
        if not skip_embeddings:
            print("Generating embeddings for vector store...")
            # Fetch all messages that were just inserted
            result = await db.execute(
                select(Message).where(Message.source_id == source_id)
            )
            all_messages = result.scalars().all()

            total_embeddings = populate_vector_store(all_messages)
            print(f"✓ Created {total_embeddings} embeddings in ChromaDB")
            print()

        print(f"{'=' * 60}")
        print(f"✅ Successfully generated:")
        print(f"   Messages: {total_messages} (across {len(scenarios)} scenarios)")
        if not skip_embeddings:
            print(f"   Embeddings: {total_embeddings}")
        print(f"{'=' * 60}")
        print()
        print("Verify in database:")
        print('  docker compose exec postgres psql -U coordination_user -d coordination -c "SELECT COUNT(*) FROM messages;"')
        if not skip_embeddings:
            print()
            print("Test semantic search:")
            print('  python -c "from src.db.vector_store import get_vector_store; vs = get_vector_store(); print(vs.search(\\"OAuth\\", limit=3))"')


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate and seed mock Slack conversation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all scenarios with embeddings
  python scripts/generate_mock_data.py --scenarios all

  # Generate specific scenarios
  python scripts/generate_mock_data.py --scenarios oauth_discussion decision_making

  # Clear existing data before generating
  python scripts/generate_mock_data.py --scenarios all --clear

  # Skip embeddings for faster testing (Postgres only)
  python scripts/generate_mock_data.py --scenarios all --skip-embeddings

Available scenarios:
  - oauth_discussion: OAuth implementation discussion - potential duplicate work
  - decision_making: Team decision without key stakeholders
  - bug_report: Bug report and resolution workflow
  - feature_planning: Cross-team feature planning coordination
        """,
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        required=True,
        help="Scenarios to generate (use 'all' for all scenarios)",
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing mock data before generating new data",
    )

    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip generating embeddings (faster, Postgres only)",
    )

    args = parser.parse_args()

    # Run async generation
    asyncio.run(generate_mock_data(args.scenarios, args.clear, args.skip_embeddings))


if __name__ == "__main__":
    main()
