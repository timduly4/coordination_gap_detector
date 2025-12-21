#!/usr/bin/env python3
"""
Index messages from PostgreSQL into Elasticsearch for BM25 search.

This script reads messages from the database and indexes them into
Elasticsearch with proper BM25 scoring mappings.

Usage:
    python scripts/index_to_elasticsearch.py
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.elasticsearch import get_es_client
from src.db.models import Message
from src.db.postgres import async_engine


async def index_messages_to_elasticsearch():
    """Index all messages from PostgreSQL into Elasticsearch."""
    print("Indexing messages to Elasticsearch...")
    print()

    # Initialize Elasticsearch client
    es_client = get_es_client()

    # Check connection
    if not es_client.check_connection():
        print("❌ Failed to connect to Elasticsearch")
        return

    print("✓ Connected to Elasticsearch")
    print()

    # Create messages index
    print("Creating 'messages' index...")
    if es_client.create_messages_index("messages"):
        print("✓ Index created successfully")
    else:
        print("❌ Failed to create index")
        return

    print()

    # Fetch all messages from PostgreSQL
    print("Fetching messages from PostgreSQL...")
    async with AsyncSession(async_engine) as db:
        result = await db.execute(select(Message))
        messages = result.scalars().all()

    print(f"✓ Found {len(messages)} messages in database")
    print()

    # Prepare messages for bulk indexing
    es_documents = []
    for msg in messages:
        doc = {
            "message_id": str(msg.id),
            "content": msg.content,
            "source": "slack",  # All mock messages are from Slack
            "channel": msg.channel,
            "author": msg.author,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            "metadata": msg.message_metadata or {},
        }
        if msg.thread_id:
            doc["thread_id"] = msg.thread_id

        es_documents.append(doc)

    # Bulk index to Elasticsearch
    print(f"Bulk indexing {len(es_documents)} messages...")
    success_count, failed_count = es_client.bulk_index_messages("messages", es_documents)

    print(f"✓ Successfully indexed: {success_count}")
    if failed_count > 0:
        print(f"⚠️  Failed to index: {failed_count}")

    print()

    # Verify indexing
    doc_count = es_client.get_document_count("messages")
    print(f"✓ Total documents in 'messages' index: {doc_count}")
    print()

    # Test a simple search
    print("Testing BM25 search for 'OAuth'...")
    search_results = es_client.search_messages("messages", "OAuth", size=3)
    print(f"✓ Search returned {search_results['total']} results")

    if search_results['results']:
        print("\nTop result:")
        top_result = search_results['results'][0]
        print(f"  Score: {top_result['score']:.4f}")
        print(f"  Author: {top_result['source']['author']}")
        print(f"  Channel: {top_result['source']['channel']}")
        print(f"  Content: {top_result['source']['content'][:100]}...")

    print()
    print("=" * 60)
    print("✅ Elasticsearch indexing complete!")
    print("=" * 60)
    print()
    print("Now test the search API:")
    print("  curl -X POST http://localhost:8000/api/v1/search/ \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"query\": \"OAuth\", \"limit\": 3}'")


def main():
    """Main entry point."""
    asyncio.run(index_messages_to_elasticsearch())


if __name__ == "__main__":
    main()
