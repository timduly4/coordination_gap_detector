"""
FastAPI dependency injection for database sessions and services.
"""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.postgres import get_db as get_db_session
from src.db.vector_store import get_vector_store, VectorStore


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database session.

    Usage:
        @app.get("/items")
        async def read_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Message))
            messages = result.scalars().all()
            return messages
    """
    async for session in get_db_session():
        yield session


def get_vector_db() -> VectorStore:
    """
    FastAPI dependency for vector store (ChromaDB).

    Usage:
        @app.get("/search")
        async def search(
            query: str,
            vector_db: VectorStore = Depends(get_vector_db)
        ):
            results = vector_db.search(query)
            return results
    """
    return get_vector_store()
