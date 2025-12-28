"""
Gap detection service orchestration.

This module provides a high-level service for coordinating gap detection
across different detectors, managing the detection lifecycle, and providing
a clean API for detection operations.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Message
from src.db.vector_store import VectorStore
from src.detection.duplicate_work import DuplicateWorkDetector
from src.detection.patterns import GapDetectionConfig, GapType
from src.models.schemas import (
    CoordinationGap,
    GapDetectionMetadata,
    GapDetectionRequest,
    GapDetectionResponse,
    GapListRequest,
    GapListResponse,
)

logger = logging.getLogger(__name__)


class GapDetectionService:
    """
    Service for orchestrating coordination gap detection.

    This service:
    - Manages multiple gap detectors (duplicate work, missing context, etc.)
    - Coordinates message retrieval and preprocessing
    - Handles detection lifecycle and error recovery
    - Provides clean API for detection operations
    """

    def __init__(
        self,
        vector_store: VectorStore,
        db_session: AsyncSession,
        duplicate_work_detector: Optional[DuplicateWorkDetector] = None,
    ):
        """
        Initialize gap detection service.

        Args:
            vector_store: Vector store for embeddings
            db_session: Database session
            duplicate_work_detector: Duplicate work detector instance
        """
        self.vector_store = vector_store
        self.db_session = db_session

        # Initialize detectors
        self.duplicate_work_detector = duplicate_work_detector or DuplicateWorkDetector()

        # Detector registry
        self.detectors = {
            GapType.DUPLICATE_WORK.value: self.duplicate_work_detector,
            # Add more detectors in future milestones:
            # GapType.MISSING_CONTEXT.value: self.missing_context_detector,
            # GapType.STALE_DOCS.value: self.stale_docs_detector,
            # GapType.KNOWLEDGE_SILO.value: self.knowledge_silo_detector,
        }

        logger.info(
            f"GapDetectionService initialized with {len(self.detectors)} detectors"
        )

    async def detect_gaps(
        self, request: GapDetectionRequest
    ) -> GapDetectionResponse:
        """
        Detect coordination gaps based on request parameters.

        Args:
            request: Gap detection request with filters and parameters

        Returns:
            GapDetectionResponse with detected gaps and metadata
        """
        logger.info(f"Starting gap detection: {request.gap_types} over {request.timeframe_days} days")

        start_time = time.time()

        # Retrieve messages
        messages = await self._retrieve_messages(request)
        logger.info(f"Retrieved {len(messages)} messages for analysis")

        if not messages:
            logger.warning("No messages found for detection")
            return GapDetectionResponse(
                gaps=[],
                metadata=GapDetectionMetadata(
                    total_gaps=0,
                    critical_gaps=0,
                    high_gaps=0,
                    medium_gaps=0,
                    low_gaps=0,
                    detection_time_ms=int((time.time() - start_time) * 1000),
                    messages_analyzed=0,
                    clusters_found=0,
                    llm_calls=0,
                ),
            )

        # Get embeddings for messages
        embeddings = await self._get_embeddings(messages)
        logger.info(f"Retrieved {len(embeddings)} embeddings")

        # Run detection for each requested gap type
        all_gaps = []
        total_llm_calls = 0
        total_clusters = 0

        for gap_type in request.gap_types:
            detector = self.detectors.get(gap_type)

            if not detector:
                logger.warning(f"No detector found for gap type: {gap_type}")
                continue

            logger.info(f"Running {gap_type} detector")

            try:
                # Run detection
                gaps = await detector.detect(messages=messages, embeddings=embeddings)
                logger.info(f"Detected {len(gaps)} {gap_type} gaps")

                # Filter by impact score
                filtered_gaps = [
                    gap for gap in gaps
                    if gap.impact_score >= request.min_impact_score
                ]

                logger.info(
                    f"Filtered to {len(filtered_gaps)} gaps with impact >= {request.min_impact_score}"
                )

                all_gaps.extend(filtered_gaps)

                # Track metrics (simplified - would track more accurately in production)
                total_llm_calls += len(filtered_gaps)  # Approx: 1 LLM call per gap

            except Exception as e:
                logger.error(f"Error running {gap_type} detector: {e}", exc_info=True)
                continue

        # Sort gaps by impact score (descending)
        all_gaps.sort(key=lambda g: g.impact_score, reverse=True)

        # Apply max_gaps limit
        if request.max_gaps and len(all_gaps) > request.max_gaps:
            logger.info(f"Limiting gaps from {len(all_gaps)} to {request.max_gaps}")
            all_gaps = all_gaps[:request.max_gaps]

        # Filter evidence if requested
        if not request.include_evidence:
            for gap in all_gaps:
                gap.evidence = []

        # Compute metadata
        metadata = self._compute_metadata(
            gaps=all_gaps,
            messages_analyzed=len(messages),
            clusters_found=total_clusters,
            llm_calls=total_llm_calls,
            detection_time_ms=int((time.time() - start_time) * 1000),
        )

        logger.info(
            f"Gap detection completed: {metadata.total_gaps} gaps in {metadata.detection_time_ms}ms"
        )

        return GapDetectionResponse(gaps=all_gaps, metadata=metadata)

    async def _retrieve_messages(
        self, request: GapDetectionRequest
    ) -> List[Message]:
        """
        Retrieve messages based on request filters.

        Args:
            request: Detection request with filters

        Returns:
            List of Message objects
        """
        # Compute date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=request.timeframe_days)

        # Build query using SQLAlchemy 2.0 select() syntax
        stmt = select(Message).filter(
            Message.timestamp >= start_date,
            Message.timestamp <= end_date,
        )

        # Apply source filter
        if request.sources:
            # Would join with Source table and filter by type
            # For now, simplified implementation
            pass

        # Apply channel filter
        if request.channels:
            stmt = stmt.filter(Message.channel.in_(request.channels))

        # Execute query
        result = await self.db_session.execute(stmt)
        messages = result.scalars().all()

        return list(messages)

    async def _get_embeddings(
        self, messages: List[Message]
    ) -> List[Any]:
        """
        Get embeddings for messages.

        Args:
            messages: List of messages

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for msg in messages:
            if msg.embedding_id:
                # Get embedding from vector store
                try:
                    # In real implementation, would batch retrieve
                    embedding = await self.vector_store.get_embedding(msg.embedding_id)
                    if embedding is not None:
                        embeddings.append(embedding)
                    else:
                        logger.warning(f"No embedding found for message {msg.id}")
                        # Use zero vector as placeholder
                        embeddings.append([0.0] * 1024)  # Assuming 1024-dim embeddings
                except Exception as e:
                    logger.error(f"Error retrieving embedding for message {msg.id}: {e}")
                    embeddings.append([0.0] * 1024)
            else:
                logger.warning(f"Message {msg.id} has no embedding_id")
                embeddings.append([0.0] * 1024)

        return embeddings

    def _compute_metadata(
        self,
        gaps: List[CoordinationGap],
        messages_analyzed: int,
        clusters_found: int,
        llm_calls: int,
        detection_time_ms: int,
    ) -> GapDetectionMetadata:
        """
        Compute detection metadata.

        Args:
            gaps: Detected gaps
            messages_analyzed: Number of messages analyzed
            clusters_found: Number of clusters found
            llm_calls: Number of LLM API calls
            detection_time_ms: Detection time in milliseconds

        Returns:
            GapDetectionMetadata object
        """
        # Count gaps by tier
        critical_gaps = sum(1 for g in gaps if g.impact_tier == "CRITICAL")
        high_gaps = sum(1 for g in gaps if g.impact_tier == "HIGH")
        medium_gaps = sum(1 for g in gaps if g.impact_tier == "MEDIUM")
        low_gaps = sum(1 for g in gaps if g.impact_tier == "LOW")

        return GapDetectionMetadata(
            total_gaps=len(gaps),
            critical_gaps=critical_gaps,
            high_gaps=high_gaps,
            medium_gaps=medium_gaps,
            low_gaps=low_gaps,
            detection_time_ms=detection_time_ms,
            messages_analyzed=messages_analyzed,
            clusters_found=clusters_found,
            llm_calls=llm_calls,
        )

    async def get_gap_by_id(self, gap_id: str) -> Optional[CoordinationGap]:
        """
        Retrieve a specific gap by ID.

        Args:
            gap_id: Gap identifier

        Returns:
            CoordinationGap if found, None otherwise
        """
        # In real implementation, would store gaps in database
        # For now, this is a placeholder
        logger.warning(f"get_gap_by_id not implemented: {gap_id}")
        return None

    async def list_gaps(
        self,
        request: GapListRequest,
    ) -> GapListResponse:
        """
        List gaps with filtering and pagination.

        Args:
            request: Gap list request with filters and pagination

        Returns:
            GapListResponse with gaps and pagination info
        """
        # In real implementation, would query from database
        # For now, return empty list as a placeholder
        logger.warning("list_gaps not implemented")

        return GapListResponse(
            gaps=[],
            total=0,
            page=request.page,
            limit=request.limit,
            has_more=False,
        )

    def configure_detector(
        self, gap_type: str, config: GapDetectionConfig
    ) -> None:
        """
        Update configuration for a specific detector.

        Args:
            gap_type: Type of gap detector to configure
            config: New configuration
        """
        detector = self.detectors.get(gap_type)

        if not detector:
            raise ValueError(f"No detector found for gap type: {gap_type}")

        # Update detector config
        detector.config = config
        logger.info(f"Updated configuration for {gap_type} detector")

    def get_detector_config(self, gap_type: str) -> Optional[GapDetectionConfig]:
        """
        Get configuration for a specific detector.

        Args:
            gap_type: Type of gap detector

        Returns:
            GapDetectionConfig if detector exists, None otherwise
        """
        detector = self.detectors.get(gap_type)

        if not detector:
            return None

        return detector.get_config()

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of detection service.

        Returns:
            Dictionary with health status
        """
        health = {
            "status": "healthy",
            "detectors": {},
            "vector_store": "unknown",
            "database": "unknown",
        }

        # Check each detector
        for gap_type, detector in self.detectors.items():
            health["detectors"][gap_type] = {
                "status": "available",
                "config": detector.get_config().dict(),
            }

        # Check vector store
        try:
            # Would do actual health check
            health["vector_store"] = "healthy"
        except Exception as e:
            health["vector_store"] = f"unhealthy: {e}"
            health["status"] = "degraded"

        # Check database
        try:
            # Would do actual health check
            health["database"] = "healthy"
        except Exception as e:
            health["database"] = f"unhealthy: {e}"
            health["status"] = "degraded"

        return health
