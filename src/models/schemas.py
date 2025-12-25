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


# Clustering Schemas


class ClusterResponse(BaseModel):
    """Schema for cluster response."""

    cluster_id: str = Field(..., description="Unique cluster identifier")
    message_ids: List[int] = Field(..., description="Database message IDs in cluster")
    size: int = Field(..., description="Number of messages in cluster", ge=0)
    avg_similarity: float = Field(
        ..., description="Average intra-cluster similarity", ge=0.0, le=1.0
    )
    timespan_days: Optional[float] = Field(
        None, description="Timespan from first to last message in days"
    )
    participant_count: Optional[int] = Field(
        None, description="Number of unique authors", ge=0
    )
    channels: Optional[List[str]] = Field(None, description="Channels involved in cluster")
    teams: Optional[List[str]] = Field(None, description="Teams identified in cluster")
    cohesion_score: Optional[float] = Field(
        None, description="Cluster cohesion quality", ge=0.0, le=1.0
    )
    start_time: Optional[datetime] = Field(None, description="First message timestamp")
    end_time: Optional[datetime] = Field(None, description="Last message timestamp")
    label: Optional[str] = Field(None, description="Cluster topic label")
    created_at: datetime = Field(..., description="When cluster was created")


class ClusteringRequest(BaseModel):
    """Schema for clustering request."""

    similarity_threshold: float = Field(
        default=0.85,
        description="Minimum similarity for clustering (0-1)",
        ge=0.0,
        le=1.0,
    )
    min_cluster_size: int = Field(
        default=2, description="Minimum messages per cluster", ge=1, le=100
    )
    time_window_days: Optional[int] = Field(
        None, description="Time window for clustering (days)", ge=1, le=365
    )
    source_types: Optional[List[str]] = Field(
        None, description="Filter by source types"
    )
    channels: Optional[List[str]] = Field(None, description="Filter by channels")
    date_from: Optional[datetime] = Field(None, description="Filter messages from this date")
    date_to: Optional[datetime] = Field(None, description="Filter messages until this date")


class ClusteringResponse(BaseModel):
    """Schema for clustering response."""

    clusters: List[ClusterResponse] = Field(..., description="List of clusters found")
    total_clusters: int = Field(..., description="Number of clusters", ge=0)
    total_messages: int = Field(..., description="Total messages analyzed", ge=0)
    clustered_messages: int = Field(..., description="Messages in clusters", ge=0)
    coverage: float = Field(
        ..., description="Fraction of messages clustered", ge=0.0, le=1.0
    )
    quality_metrics: Dict[str, float] = Field(..., description="Clustering quality metrics")
    clustering_time_ms: int = Field(..., description="Time taken for clustering", ge=0)


# Gap Detection Schemas


class EvidenceItem(BaseModel):
    """Schema for a single piece of evidence supporting a gap."""

    source: str = Field(..., description="Source type (slack, github, etc.)")
    message_id: Optional[int] = Field(None, description="Database message ID")
    external_id: Optional[str] = Field(None, description="ID in source system")
    channel: Optional[str] = Field(None, description="Channel or location")
    author: Optional[str] = Field(None, description="Message author")
    content: str = Field(..., description="Evidence content/quote")
    timestamp: datetime = Field(..., description="When this evidence was created")
    relevance_score: float = Field(..., description="Relevance to gap (0-1)", ge=0.0, le=1.0)
    team: Optional[str] = Field(None, description="Team associated with this evidence")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class TemporalOverlap(BaseModel):
    """Schema for temporal overlap analysis."""

    start: datetime = Field(..., description="Start of overlap period")
    end: datetime = Field(..., description="End of overlap period")
    overlap_days: int = Field(..., description="Number of overlapping days", ge=0)
    team_timelines: Optional[Dict[str, Dict[str, str]]] = Field(
        None, description="Timeline for each team (start/end dates)"
    )


class LLMVerification(BaseModel):
    """Schema for LLM verification results."""

    is_duplicate: bool = Field(..., description="Whether this is duplicate work")
    confidence: float = Field(..., description="LLM confidence (0-1)", ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Explanation of the determination")
    evidence: List[str] = Field(..., description="Key quotes supporting the determination")
    recommendation: str = Field(..., description="Recommended action")
    overlap_ratio: float = Field(
        ..., description="Estimated overlap ratio (0-1)", ge=0.0, le=1.0
    )


class ImpactBreakdown(BaseModel):
    """Schema for impact scoring breakdown."""

    team_size_score: float = Field(..., description="Score based on team size", ge=0.0, le=1.0)
    time_investment_score: float = Field(
        ..., description="Score based on time invested", ge=0.0, le=1.0
    )
    project_criticality_score: float = Field(
        ..., description="Score based on project importance", ge=0.0, le=1.0
    )
    velocity_impact_score: float = Field(
        ..., description="Score based on velocity impact", ge=0.0, le=1.0
    )
    duplicate_effort_score: float = Field(
        ..., description="Score based on duplication amount", ge=0.0, le=1.0
    )


class CostEstimate(BaseModel):
    """Schema for cost estimation."""

    engineering_hours: float = Field(..., description="Estimated engineering hours wasted", ge=0.0)
    dollar_value: float = Field(..., description="Estimated dollar cost", ge=0.0)
    explanation: str = Field(..., description="Explanation of cost calculation")
    note: str = Field(
        default="This is organizational waste cost, not Claude API cost",
        description="Clarification note"
    )


class CoordinationGap(BaseModel):
    """Schema for a detected coordination gap."""

    id: str = Field(..., description="Unique gap identifier")
    type: str = Field(..., description="Gap type (DUPLICATE_WORK, etc.)")
    title: str = Field(..., description="Human-readable gap title")
    topic: str = Field(..., description="Main topic of the gap")
    teams_involved: List[str] = Field(..., description="Teams involved in the gap")
    impact_score: float = Field(..., description="Overall impact score (0-1)", ge=0.0, le=1.0)
    impact_tier: str = Field(..., description="Impact tier (CRITICAL, HIGH, MEDIUM, LOW)")
    confidence: float = Field(..., description="Detection confidence (0-1)", ge=0.0, le=1.0)

    # Evidence and verification
    evidence: List[EvidenceItem] = Field(..., description="Supporting evidence")
    temporal_overlap: Optional[TemporalOverlap] = Field(
        None, description="Temporal overlap analysis"
    )
    verification: Optional[LLMVerification] = Field(None, description="LLM verification results")

    # Impact details
    impact_breakdown: Optional[ImpactBreakdown] = Field(
        None, description="Detailed impact scoring"
    )
    estimated_cost: Optional[CostEstimate] = Field(None, description="Cost estimation")

    # Insights and recommendations
    insight: str = Field(..., description="AI-generated insight about the gap")
    recommendation: str = Field(..., description="Recommended action to address gap")

    # Metadata
    detected_at: datetime = Field(..., description="When the gap was detected")
    cluster_id: Optional[str] = Field(None, description="Source cluster ID")
    people_affected: Optional[int] = Field(None, description="Number of people affected", ge=0)
    timespan_days: Optional[int] = Field(None, description="Timespan of the gap", ge=0)
    messages_analyzed: Optional[int] = Field(None, description="Number of messages analyzed", ge=0)


class GapDetectionRequest(BaseModel):
    """Schema for gap detection request."""

    timeframe_days: int = Field(
        default=30, description="Timeframe to analyze (days)", ge=1, le=365
    )
    sources: List[str] = Field(
        default=["slack"], description="Source types to analyze"
    )
    gap_types: List[str] = Field(
        default=["duplicate_work"], description="Types of gaps to detect"
    )
    teams: Optional[List[str]] = Field(None, description="Filter by specific teams")
    min_impact_score: float = Field(
        default=0.0, description="Minimum impact score (0-1)", ge=0.0, le=1.0
    )
    include_evidence: bool = Field(default=True, description="Include evidence in response")
    channels: Optional[List[str]] = Field(None, description="Filter by specific channels")
    max_gaps: Optional[int] = Field(None, description="Maximum gaps to return", ge=1, le=100)


class GapDetectionMetadata(BaseModel):
    """Schema for gap detection metadata."""

    total_gaps: int = Field(..., description="Total gaps detected", ge=0)
    critical_gaps: int = Field(..., description="Critical impact gaps", ge=0)
    high_gaps: int = Field(..., description="High impact gaps", ge=0)
    medium_gaps: int = Field(..., description="Medium impact gaps", ge=0)
    low_gaps: int = Field(..., description="Low impact gaps", ge=0)
    detection_time_ms: int = Field(..., description="Detection time in milliseconds", ge=0)
    messages_analyzed: int = Field(..., description="Total messages analyzed", ge=0)
    clusters_found: int = Field(..., description="Total clusters found", ge=0)
    llm_calls: Optional[int] = Field(None, description="Number of LLM API calls made", ge=0)


class GapDetectionResponse(BaseModel):
    """Schema for gap detection response."""

    gaps: List[CoordinationGap] = Field(..., description="Detected coordination gaps")
    metadata: GapDetectionMetadata = Field(..., description="Detection metadata")


class GapListRequest(BaseModel):
    """Schema for listing gaps with filters."""

    gap_type: Optional[str] = Field(None, description="Filter by gap type")
    min_impact_score: float = Field(
        default=0.0, description="Minimum impact score", ge=0.0, le=1.0
    )
    teams: Optional[List[str]] = Field(None, description="Filter by teams")
    start_date: Optional[datetime] = Field(None, description="Filter from date")
    end_date: Optional[datetime] = Field(None, description="Filter to date")
    page: int = Field(default=1, description="Page number", ge=1)
    limit: int = Field(default=10, description="Results per page", ge=1, le=100)


class GapListResponse(BaseModel):
    """Schema for gap list response."""

    gaps: List[CoordinationGap] = Field(..., description="List of gaps")
    total: int = Field(..., description="Total number of gaps", ge=0)
    page: int = Field(..., description="Current page", ge=1)
    limit: int = Field(..., description="Results per page", ge=1)
    has_more: bool = Field(..., description="Whether there are more results")
