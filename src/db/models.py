"""
SQLAlchemy database models for coordination gap detector.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from src.db.postgres import Base


class Source(Base):
    """
    Represents a data source (Slack, GitHub, Google Docs, etc.)
    """

    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(50), nullable=False, index=True)  # slack, github, google_docs, etc.
    name = Column(String(255), nullable=False)  # e.g., "Engineering Slack", "main-repo"
    config = Column(JSON, nullable=True)  # Source-specific configuration
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    messages = relationship("Message", back_populates="source", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Source(id={self.id}, type={self.type}, name={self.name})>"


class Message(Base):
    """
    Represents a message/document from any source.
    This is the core content that will be analyzed for coordination gaps.
    """

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)

    # Core content
    content = Column(Text, nullable=False)

    # Metadata
    external_id = Column(String(255), nullable=True, index=True)  # ID in source system
    author = Column(String(255), nullable=True, index=True)  # email or username
    channel = Column(String(255), nullable=True, index=True)  # channel, repo, doc name
    thread_id = Column(String(255), nullable=True, index=True)  # for threading

    # Timestamps
    timestamp = Column(DateTime, nullable=False, index=True)  # When message was created in source
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Additional metadata stored as JSON
    message_metadata = Column(JSON, nullable=True)  # reactions, mentions, attachments, etc.

    # Embedding reference (will be in ChromaDB)
    embedding_id = Column(String(255), nullable=True, index=True)  # Reference to vector store

    # Relationships
    source = relationship("Source", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, source_id={self.source_id}, author={self.author})>"
