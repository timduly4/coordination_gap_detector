"""
Pydantic schemas for API requests and responses.

This module defines the data models used for API communication,
validation, and serialization.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class MessageBase(BaseModel):
    """Base schema for message data."""

    content: str = Field(..., description="Message content", min_length=1)
    author: Optional[str] = Field(None, description="Message author (email or username)")
    channel: Optional[str] = Field(None, description="Channel, repository, or document name")
    timestamp: datetime = Field(..., description="When the message was created")
    external_id: Optional[str] = Field(None, description="ID in the source system")
    thread_id: Optional[str] = Field(None, description="Thread identifier for grouped messages")
    message_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata (reactions, mentions, etc.)"
    )


class MessageCreate(MessageBase):
    """Schema for creating a new message."""

    source_id: int = Field(..., description="ID of the source this message belongs to", gt=0)


class MessageResponse(MessageBase):
    """Schema for message response."""

    id: int = Field(..., description="Message ID")
    source_id: int = Field(..., description="Source ID")
    embedding_id: Optional[str] = Field(None, description="Reference to vector store embedding")
    created_at: datetime = Field(..., description="When the message was created in our system")
    updated_at: datetime = Field(..., description="When the message was last updated")

    class Config:
        from_attributes = True


class SearchRequest(BaseModel):
    """Schema for search requests."""

    query: str = Field(..., description="Search query text", min_length=1, max_length=1000)
    limit: int = Field(
        default=10, description="Maximum number of results to return", ge=1, le=100
    )
    threshold: float = Field(
        default=0.0,
        description="Minimum similarity threshold (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    ranking_strategy: Optional[str] = Field(
        default="auto",
        description="Search ranking strategy: 'auto' (detect intent), 'semantic', 'bm25', 'hybrid_rrf', 'hybrid_weighted'"
    )
    semantic_weight: Optional[float] = Field(
        default=0.7,
        description="Weight for semantic scores in hybrid_weighted strategy (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    keyword_weight: Optional[float] = Field(
        default=0.3,
        description="Weight for keyword scores in hybrid_weighted strategy (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    source_types: Optional[List[str]] = Field(
        None, description="Filter by source types (e.g., ['slack', 'github'])"
    )
    date_from: Optional[datetime] = Field(None, description="Filter messages from this date")
    date_to: Optional[datetime] = Field(None, description="Filter messages until this date")
    channels: Optional[List[str]] = Field(None, description="Filter by specific channels")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and clean the query string."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or only whitespace")
        return v

    @field_validator("ranking_strategy")
    @classmethod
    def validate_strategy(cls, v: Optional[str]) -> Optional[str]:
        """Validate ranking strategy."""
        if v is None:
            return "auto"

        valid_strategies = ["auto", "semantic", "bm25", "hybrid_rrf", "hybrid_weighted"]
        if v not in valid_strategies:
            raise ValueError(
                f"Invalid ranking strategy: {v}. Must be one of {valid_strategies}"
            )
        return v


class SearchResultItem(BaseModel):
    """Schema for a single search result."""

    content: str = Field(..., description="Message content")
    source: str = Field(..., description="Source type (slack, github, etc.)")
    channel: Optional[str] = Field(None, description="Channel or location")
    author: Optional[str] = Field(None, description="Message author")
    timestamp: datetime = Field(..., description="When the message was created")
    score: float = Field(..., description="Similarity score (0.0 to 1.0)", ge=0.0)
    message_id: int = Field(..., description="Database message ID")
    external_id: Optional[str] = Field(None, description="ID in source system")
    message_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    ranking_details: Optional[Dict[str, Any]] = Field(
        None, description="Detailed ranking information (semantic/keyword scores, ranks, fusion method)"
    )


class SearchResponse(BaseModel):
    """Schema for search response."""

    results: List[SearchResultItem] = Field(..., description="List of search results")
    total: int = Field(..., description="Total number of results", ge=0)
    query: str = Field(..., description="The search query that was executed")
    query_time_ms: int = Field(..., description="Query execution time in milliseconds", ge=0)
    threshold: float = Field(..., description="Similarity threshold used", ge=0.0, le=1.0)
    ranking_strategy: Optional[str] = Field(None, description="Ranking strategy used for this search")
    query_intent: Optional[str] = Field(None, description="Detected query intent (if auto strategy)")


class VectorStoreInsertRequest(BaseModel):
    """Schema for inserting documents into vector store."""

    message_id: int = Field(..., description="Database message ID", gt=0)
    content: str = Field(..., description="Content to embed and store", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata to store with embedding")


class VectorStoreInsertResponse(BaseModel):
    """Schema for vector store insert response."""

    message_id: int = Field(..., description="Database message ID")
    embedding_id: str = Field(..., description="ID of the stored embedding in vector store")
    success: bool = Field(..., description="Whether the insertion was successful")


class VectorStoreSearchRequest(BaseModel):
    """Schema for vector store search request."""

    query: str = Field(..., description="Query text to search for", min_length=1)
    limit: int = Field(default=10, description="Number of results to return", ge=1, le=100)
    threshold: float = Field(
        default=0.0, description="Minimum similarity score", ge=0.0, le=1.0
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadata filters to apply"
    )


class VectorStoreSearchResult(BaseModel):
    """Schema for a single vector store search result."""

    embedding_id: str = Field(..., description="Embedding ID in vector store")
    content: str = Field(..., description="Content text")
    score: float = Field(..., description="Similarity score", ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Associated metadata")


class VectorStoreSearchResponse(BaseModel):
    """Schema for vector store search response."""

    results: List[VectorStoreSearchResult] = Field(..., description="Search results")
    total: int = Field(..., description="Number of results returned", ge=0)
    query_time_ms: int = Field(..., description="Query time in milliseconds", ge=0)


class SourceBase(BaseModel):
    """Base schema for source data."""

    type: str = Field(..., description="Source type (slack, github, google_docs, etc.)")
    name: str = Field(..., description="Source name (e.g., 'Engineering Slack')")
    config: Optional[Dict[str, Any]] = Field(None, description="Source-specific configuration")


class SourceCreate(SourceBase):
    """Schema for creating a new source."""

    pass


class SourceResponse(SourceBase):
    """Schema for source response."""

    id: int = Field(..., description="Source ID")
    created_at: datetime = Field(..., description="When the source was created")
    updated_at: datetime = Field(..., description="When the source was last updated")

    class Config:
        from_attributes = True


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""

    status: str = Field(..., description="Overall system status")
    database: bool = Field(..., description="Database connection status")
    vector_store: bool = Field(..., description="Vector store connection status")
    vector_store_count: int = Field(..., description="Number of documents in vector store", ge=0)
    timestamp: datetime = Field(..., description="Health check timestamp")


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


# Evaluation Schemas


class EvaluationQueryItem(BaseModel):
    """Schema for a single test query in evaluation."""

    query_id: str = Field(..., description="Unique query identifier")
    query_text: str = Field(..., description="Query text", min_length=1)
    category: Optional[str] = Field(None, description="Query category (factual, technical, etc.)")


class EvaluationJudgmentItem(BaseModel):
    """Schema for a single relevance judgment."""

    query_id: str = Field(..., description="Query identifier")
    document_id: str = Field(..., description="Document/message ID")
    relevance: int = Field(..., description="Relevance score (0-3)", ge=0, le=3)
    query_text: Optional[str] = Field(None, description="Optional query text")


class EvaluateStrategyRequest(BaseModel):
    """Schema for evaluating ranking strategies."""

    test_queries: List[Dict[str, Any]] = Field(
        ...,
        description="List of test queries with query_id and query_text",
        min_length=1
    )
    strategies: List[str] = Field(
        default=["semantic", "bm25", "hybrid_rrf"],
        description="List of ranking strategies to evaluate"
    )
    k: int = Field(
        default=10,
        description="Cutoff for @k metrics",
        ge=1,
        le=100
    )


class EvaluationMetricsResponse(BaseModel):
    """Schema for evaluation metrics."""

    mrr: Optional[float] = Field(None, description="Mean Reciprocal Rank")
    ndcg: Optional[float] = Field(None, description="Normalized Discounted Cumulative Gain")
    precision: Optional[float] = Field(None, description="Precision at k")
    recall: Optional[float] = Field(None, description="Recall at k")
    f1: Optional[float] = Field(None, description="F1 score at k")


class EvaluationComparisonResponse(BaseModel):
    """Schema for strategy comparison results."""

    strategies: List[str] = Field(..., description="Strategies evaluated")
    num_queries: int = Field(..., description="Number of queries evaluated", ge=0)
    k: int = Field(..., description="Cutoff used for metrics", ge=1)
    results: Dict[str, Dict[str, Optional[float]]] = Field(
        ...,
        description="Evaluation results by metric name and strategy"
    )
    best_strategy: str = Field(..., description="Best performing strategy")
    improvement_over_baseline: float = Field(
        ...,
        description="Percentage improvement over baseline strategy"
    )


class AddJudgmentRequest(BaseModel):
    """Schema for adding a single judgment."""

    query_id: str = Field(..., description="Query identifier")
    document_id: str = Field(..., description="Document identifier (message ID)")
    relevance: int = Field(..., description="Relevance score (0-3)", ge=0, le=3)
    query_text: Optional[str] = Field(None, description="Optional query text")


class AddJudgmentResponse(BaseModel):
    """Schema for add judgment response."""

    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Response message")


class EvaluationStatisticsResponse(BaseModel):
    """Schema for evaluation statistics."""

    num_queries: int = Field(..., description="Number of queries with judgments", ge=0)
    num_judgments: int = Field(..., description="Total number of judgments", ge=0)
