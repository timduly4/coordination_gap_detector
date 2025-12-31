# Coordination Gap Detector

AI-powered coordination gap detection system for identifying failures across enterprise communication channels. This platform automatically detects when teams are duplicating work, missing critical context, or working at cross-purposes across Slack, GitHub, Google Docs, and other collaboration tools.

## What This Project Does

An enterprise platform that ingests data from multiple collaboration tools to automatically identify coordination gaps including:

- **Duplicate Work** - Multiple teams solving the same problem independently
- **Missing Context** - Critical decisions made without stakeholder awareness
- **Stale Documentation** - Outdated docs contradicting current implementations
- **Knowledge Silos** - Critical knowledge trapped in individual teams
- **Missed Dependencies** - Work proceeding unaware of blocking issues

## Key Features

- **Multi-Source Integration** - Slack, GitHub, Google Docs, Jira, Confluence
- **Real-Time Detection** - Identify coordination gaps as they emerge
- **ML-Based Ranking** - Prioritize gaps by organizational impact
- **LLM Reasoning** - Claude-powered insights and recommendations
- **Search Quality** - Advanced IR with MRR, NDCG evaluation metrics
- **Distributed Architecture** - Kubernetes, Kafka, Elasticsearch for scale

## 🚀 Interactive Demo

**New to the project? Start here!**

Follow our [**Interactive Demo Guide**](./docs/DEMO.md) for a hands-on walkthrough:
- ⏱️ 15 minutes from zero to detecting your first coordination gap
- 🔍 Try semantic search, BM25, and hybrid ranking
- 🎯 See gap detection in action with realistic mock data
- 📊 Understand impact scoring and cost estimation
- 💡 Learn how the detection pipeline works

Perfect for first-time users, demos, and understanding the system's capabilities.

## Tech Stack

- **Python 3.11+** with FastAPI
- **UV** for dependency management
- **Claude API** for LLM reasoning
- **ChromaDB** for vector search
- **Elasticsearch** for full-text search
- **PostgreSQL** with pgvector
- **Redis** for caching
- **Kafka** for event streaming
- **Kubernetes** for orchestration

## Quick Start

### Prerequisites

- Python 3.11 or higher
- UV package manager ([installation guide](https://github.com/astral-sh/uv))
- Docker and Docker Compose (for infrastructure)

### Installation

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/timduly4/coordination_gap_detector.git
cd coordination_gap_detector

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Start Infrastructure

```bash
# Start required services (PostgreSQL, Redis, ChromaDB, Elasticsearch)
docker compose up -d

# Verify all services are running
docker compose ps

# Check service health
curl http://localhost:8000/health/detailed

# Run database migrations
docker compose exec api alembic upgrade head
```

**Infrastructure Services:**
- **PostgreSQL** (port 5432) - Primary database with pgvector extension
- **Redis** (port 6379) - Caching and real-time feature store
- **ChromaDB** (port 8001) - Vector database for semantic search
- **Elasticsearch** (port 9200) - Full-text search with BM25 ranking
- **API** (port 8000) - FastAPI application server

### Generate Mock Data (Development)

For development and testing, you can generate realistic Slack conversation data:

```bash
# Generate all mock conversation scenarios
docker compose exec api python scripts/generate_mock_data.py --scenarios all

# Generate specific scenarios
docker compose exec api python scripts/generate_mock_data.py --scenarios oauth_discussion decision_making

# Clear existing data before generating
docker compose exec api python scripts/generate_mock_data.py --scenarios all --clear
```

**Available Mock Scenarios:**

1. **oauth_discussion** (8 messages)
   - **Purpose**: Demonstrates potential duplicate work detection
   - **Content**: Two teams (platform and auth) independently working on OAuth2 integration
   - **Channels**: #platform, #auth-team
   - **Gap Type**: Duplicate Work
   - **Key Insights**: Shows parallel efforts without coordination, different approaches to same problem

2. **decision_making** (5 messages)
   - **Purpose**: Illustrates missing context and stakeholder exclusion
   - **Content**: Architecture decision made without security team input
   - **Channels**: #engineering
   - **Gap Type**: Missing Context
   - **Key Insights**: Critical decisions made without required stakeholders

3. **bug_report** (5 messages)
   - **Purpose**: Demonstrates effective coordination (positive example)
   - **Content**: Bug report, diagnosis, and resolution with proper handoffs
   - **Channels**: #frontend
   - **Gap Type**: None (baseline for comparison)
   - **Key Insights**: Shows what good coordination looks like

4. **feature_planning** (6 messages)
   - **Purpose**: Shows cross-team collaboration patterns
   - **Content**: Feature planning across platform, mobile, and backend teams
   - **Channels**: #product
   - **Gap Type**: Potential dependency issues
   - **Key Insights**: Multiple teams with dependencies, coordination required

**Verify Data:**
```bash
# Check message count in database
docker compose exec postgres psql -U coordination_user -d coordination -c "SELECT COUNT(*) FROM messages;"

# View messages by channel
docker compose exec postgres psql -U coordination_user -d coordination -c "SELECT channel, COUNT(*) FROM messages GROUP BY channel;"
```

### Run the Application

```bash
# Development mode with auto-reload
uv run uvicorn src.main:app --reload --port 8000

# Or use the main module directly
uv run python -m src.main
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Development

### Project Structure

```
coordination-gap-detector/
├── src/                    # Main application code
│   ├── api/               # FastAPI routes and endpoints
│   │   └── routes/        # API route modules (search, gaps, etc.)
│   ├── services/          # Business logic layer
│   ├── detection/         # Gap detection algorithms
│   ├── ranking/           # Search quality & ranking
│   ├── search/            # Multi-source search
│   ├── models/            # Data models and schemas
│   ├── ingestion/         # Data source integrations
│   ├── analysis/          # Content analysis
│   ├── db/                # Database operations
│   ├── infrastructure/    # Observability, rate limiting
│   └── utils/             # Utilities
├── tests/                 # Test suite
│   ├── test_api/          # API endpoint tests
│   ├── test_detection/    # Detection algorithm tests
│   └── test_integration/  # Integration tests
├── notebooks/             # Jupyter notebooks for analysis
├── scripts/               # Utility scripts
├── k8s/                   # Kubernetes manifests
└── terraform/             # Infrastructure as code
```

### Running Tests

```bash
# Run all tests (unit tests only, faster)
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test modules
uv run pytest tests/test_api/test_search.py -v          # Search API tests
uv run pytest tests/test_db/test_vector_store.py -v     # Vector store tests
uv run pytest tests/test_db/test_elasticsearch.py -v    # Elasticsearch tests
uv run pytest tests/test_embeddings.py -v               # Embedding tests

# Run tests by category
uv run pytest tests/test_api/ -v                        # All API tests
uv run pytest tests/test_db/ -v                         # All database tests
uv run pytest tests/test_ranking/ -v                    # All ranking tests

# Run integration tests (requires running services)
uv run pytest -m integration                            # Only integration tests
uv run pytest tests/test_integration/ -v                # All integration tests

# Skip integration tests (useful for CI without Docker)
uv run pytest -m "not integration"

# Run in Docker
docker compose exec api pytest

# Run with verbose output
docker compose exec api pytest -v

# Run specific test with Docker
docker compose exec api pytest tests/test_api/test_search.py::TestSearchEndpoint::test_search_basic_query -v
```

**Test Categories:**
- **Unit Tests** - Fast, mocked dependencies (default)
- **Integration Tests** - Require Docker services (Elasticsearch, PostgreSQL, etc.)
  - Marked with `@pytest.mark.integration`
  - Run with: `pytest -m integration`
  - Require: `docker compose up -d`

**Elasticsearch Integration Tests:**
```bash
# Start Elasticsearch
docker compose up -d elasticsearch

# Run Elasticsearch integration tests
uv run pytest tests/test_integration/test_elasticsearch_integration.py -v

# These tests verify:
# - Connection and cluster health
# - Index creation and deletion
# - Message indexing (single and bulk)
# - BM25-based search
# - Filtering by source and channel
# - Search relevance ordering
```

For comprehensive testing documentation including test structure, options, and troubleshooting, see **[TESTING.md](./docs/TESTING.md)**.

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Configuration

Configuration is managed through environment variables. See `.env.example` for all available options.

Key configuration areas:
- **API Keys** - Anthropic, Slack, GitHub, etc.
- **Databases** - PostgreSQL, Redis, Elasticsearch
- **Feature Flags** - Enable/disable specific detection types
- **Observability** - Prometheus, Grafana, Sentry

## API Documentation

Once running, visit http://localhost:8000/docs for interactive API documentation.

### Available Endpoints

#### Search API

**POST /api/v1/search/** - Advanced search with multiple ranking strategies

Search for messages using semantic similarity, BM25 keyword matching, or hybrid fusion for best results.

##### Example 1: Hybrid Search (Recommended)

Combines semantic and BM25 for best overall quality:

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation decisions",
    "ranking_strategy": "hybrid_rrf",
    "limit": 10,
    "threshold": 0.7
  }'
```

**Response with Ranking Details:**
```json
{
  "results": [
    {
      "content": "OAuth2 implementation decision: We've decided to use Auth0...",
      "source": "slack",
      "channel": "#architecture",
      "author": "alice@demo.com",
      "timestamp": "2024-12-01T09:00:00Z",
      "score": 0.92,
      "ranking_details": {
        "semantic_score": 0.94,
        "bm25_score": 8.5,
        "semantic_rank": 1,
        "bm25_rank": 2,
        "fusion_method": "rrf",
        "features": {
          "recency": 0.98,
          "thread_depth": 0.75,
          "term_coverage": 0.85
        }
      },
      "message_id": 123,
      "external_id": "slack_msg_abc"
    }
  ],
  "total": 1,
  "query": "OAuth implementation decisions",
  "query_time_ms": 142,
  "threshold": 0.7,
  "strategy": "hybrid_rrf"
}
```

##### Example 2: Semantic Search

For conceptual matching and paraphrasing:

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how to authenticate users",
    "ranking_strategy": "semantic",
    "limit": 5,
    "threshold": 0.75
  }'
```

##### Example 3: BM25 Keyword Search

For exact technical terms and keywords:

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth2 JWT token",
    "ranking_strategy": "bm25",
    "limit": 5,
    "source_types": ["slack"],
    "channels": ["#engineering"]
  }'
```

##### Example 4: Custom Weighted Hybrid

Fine-tune the balance between semantic and keyword matching:

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth security best practices",
    "ranking_strategy": "hybrid_weighted",
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    "limit": 10
  }'
```

**Parameters:**
- `query` (required): Search query text (1-1000 characters)
- `ranking_strategy` (optional): Ranking method - `"semantic"`, `"bm25"`, `"hybrid_rrf"` (default), or `"hybrid_weighted"`
- `limit` (optional): Maximum results to return (1-100, default: 10)
- `threshold` (optional): Minimum similarity score (0.0-1.0, default: 0.0)
- `source_types` (optional): Filter by source types (e.g., ["slack", "github"])
- `channels` (optional): Filter by specific channels
- `date_from` (optional): Filter messages from this date (ISO format)
- `date_to` (optional): Filter messages until this date (ISO format)
- `semantic_weight` (optional): Weight for semantic scores when using `hybrid_weighted` (0.0-1.0, default: 0.7)
- `keyword_weight` (optional): Weight for BM25 scores when using `hybrid_weighted` (0.0-1.0, default: 0.3)

**Ranking Strategy Guide:**

| Strategy | Best For | Use When |
|----------|----------|----------|
| `hybrid_rrf` | General-purpose search (recommended default) | You want best overall results without tuning |
| `semantic` | Conceptual matches, paraphrasing, questions | Query describes concepts in natural language |
| `bm25` | Exact keywords, technical terms, acronyms | Query contains specific technical terms (OAuth, API, JWT) |
| `hybrid_weighted` | Custom requirements | You need fine control over semantic vs keyword balance |

See [docs/RANKING.md](./docs/RANKING.md) for detailed ranking strategy documentation.

**GET /api/v1/search/health** - Search service health check

Check the health and status of the search service and its dependencies.

```bash
curl http://localhost:8000/api/v1/search/health
```

**Response:**
```json
{
  "status": "healthy",
  "vector_store": {
    "connected": true,
    "collection": "coordination_messages",
    "document_count": 24
  }
}
```

#### Ranking Evaluation API

**POST /api/v1/evaluate** - Evaluate ranking strategies offline

Evaluate search quality using test queries and relevance judgments.

```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "strategies": ["semantic", "bm25", "hybrid_rrf"],
    "metrics": ["mrr", "ndcg@10", "precision@5"],
    "test_queries": [
      {
        "query_id": "oauth_001",
        "query": "OAuth implementation decisions",
        "judgments": [
          {"document_id": "msg_1", "relevance": 3},
          {"document_id": "msg_2", "relevance": 2}
        ]
      }
    ]
  }'
```

**Response:**
```json
{
  "results": {
    "semantic": {
      "mrr": 0.742,
      "ndcg@10": 0.789,
      "precision@5": 0.680
    },
    "bm25": {
      "mrr": 0.685,
      "ndcg@10": 0.721,
      "precision@5": 0.620
    },
    "hybrid_rrf": {
      "mrr": 0.798,
      "ndcg@10": 0.841,
      "precision@5": 0.740
    }
  },
  "best_strategy": "hybrid_rrf",
  "statistical_significance": {
    "hybrid_rrf_vs_semantic": {
      "mrr_improvement": 0.056,
      "p_value": 0.003,
      "significant": true
    }
  }
}
```

See [docs/EVALUATION.md](./docs/EVALUATION.md) for comprehensive evaluation methodology.

#### Gap Detection API

**POST /api/v1/gaps/detect** - Detect coordination gaps

Run the detection pipeline to identify coordination failures across teams.

```bash
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "sources": ["slack"],
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.6,
    "include_evidence": true
  }'
```

**Response:**
```json
{
  "gaps": [
    {
      "id": "gap_abc123",
      "type": "DUPLICATE_WORK",
      "topic": "OAuth2 Implementation",
      "teams_involved": ["platform-team", "auth-team"],
      "impact_score": 0.89,
      "confidence": 0.87,
      "evidence": [
        {
          "source": "slack",
          "channel": "#platform",
          "author": "alice@company.com",
          "content": "Starting OAuth2 implementation...",
          "timestamp": "2024-12-01T10:30:00Z"
        }
      ],
      "recommendation": "Connect alice@company.com and bob@company.com immediately.",
      "detected_at": "2024-12-15T14:20:00Z"
    }
  ],
  "metadata": {
    "total_gaps": 1,
    "messages_analyzed": 24,
    "detection_time_ms": 3200
  }
}
```

**GET /api/v1/gaps** - List detected gaps with filtering

```bash
curl "http://localhost:8000/api/v1/gaps?gap_type=duplicate_work&min_impact_score=0.7&limit=10"
```

**GET /api/v1/gaps/{gap_id}** - Get specific gap details

```bash
curl http://localhost:8000/api/v1/gaps/gap_abc123
```

See [docs/API_EXAMPLES.md](./docs/API_EXAMPLES.md) for comprehensive API usage examples.

## Milestone 2: Ranking & Search Quality

Milestone 2 adds sophisticated ranking capabilities and evaluation frameworks to the system.

### Features Completed

✅ **Elasticsearch Integration** - Full-text search with BM25 scoring
✅ **BM25 Implementation** - Probabilistic ranking with configurable parameters (k1, b)
✅ **Hybrid Search** - Combines semantic and keyword search using Reciprocal Rank Fusion (RRF) and weighted fusion
✅ **Ranking Metrics** - MRR, NDCG, DCG, Precision@k, Recall@k calculations
✅ **Feature Engineering** - 40+ ranking features across query-doc similarity, temporal, engagement, and authority signals
✅ **Evaluation Framework** - Offline evaluation with test queries, relevance judgments, and strategy comparison
✅ **A/B Testing Support** - Statistical significance testing and experiment framework

### Quick Start with Milestone 2

```bash
# 1. Start all services (includes Elasticsearch)
docker compose up -d

# 2. Generate mock data with varied relevance
docker compose exec api python scripts/generate_mock_data.py --scenarios all

# 3. Try different ranking strategies
# Hybrid search (best overall)
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "OAuth implementation", "ranking_strategy": "hybrid_rrf"}'

# BM25 keyword search
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "OAuth implementation", "ranking_strategy": "bm25"}'

# 4. Run offline evaluation
python scripts/evaluate_ranking.py \
  --strategies semantic,bm25,hybrid_rrf \
  --metrics mrr,ndcg@10
```

### Key Documentation

- **[RANKING.md](./docs/RANKING.md)** - Comprehensive ranking strategy guide
  - When to use semantic vs BM25 vs hybrid
  - Parameter tuning (BM25 k1/b, fusion weights)
  - Feature descriptions and importance
  - Performance optimization
  - Troubleshooting ranking issues

- **[EVALUATION.md](./docs/EVALUATION.md)** - Evaluation methodology
  - Creating test query sets
  - Relevance judgment guidelines (0-3 scale)
  - Running offline evaluations
  - Interpreting metrics (MRR, NDCG, P@k, R@k)
  - A/B testing best practices
  - Continuous evaluation pipelines

### Ranking Performance

All performance targets from Milestone 2 breakdown:

| Operation | Target (p95) | Typical |
|-----------|--------------|---------|
| Hybrid search | <200ms | 100-150ms |
| Feature extraction | <50ms/doc | 10-20ms |
| BM25 scoring (1000 docs) | <100ms | 40-60ms |
| NDCG calculation (50 results) | <10ms | 3-5ms |

### Example: Comparing Strategies

```bash
# Evaluate all strategies on test queries
python scripts/evaluate_ranking.py \
  --queries data/test_queries/queries.jsonl \
  --judgments data/test_queries/judgments.jsonl \
  --strategies semantic,bm25,hybrid_rrf,hybrid_weighted \
  --metrics mrr,ndcg@10,precision@5 \
  --output results/strategy_comparison.json

# Output:
# Strategy Comparison
# ═══════════════════════════════════════
# Strategy         MRR    NDCG@10  P@5
# semantic        0.742   0.789   0.680
# bm25            0.685   0.721   0.620
# hybrid_rrf      0.798   0.841   0.740  ⭐ Best
# hybrid_weighted 0.776   0.823   0.720
#
# Best strategy: hybrid_rrf
# Improvement over semantic: +7.5% MRR (p<0.01)
```

## Milestone 3: Gap Detection & Testing

Milestone 3 implements the core coordination gap detection capabilities with comprehensive testing and documentation.

### Features Completed

✅ **Duplicate Work Detection** - Multi-stage pipeline to identify teams duplicating effort
✅ **Entity Extraction** - Extract teams, people, projects, and topics from messages
✅ **Impact Scoring** - Quantify organizational cost of coordination gaps
✅ **Confidence Scoring** - Multi-factor confidence calculation for gap detection
✅ **Claude Integration** - LLM-based verification and reasoning
✅ **Mock Data Scenarios** - 6 realistic gap detection test scenarios
✅ **End-to-End Testing** - Complete integration test suite
✅ **Comprehensive Documentation** - Gap detection methodology, entity extraction, and API guides
✅ **Interactive Demo** - Jupyter notebook for gap analysis exploration

### Gap Detection Pipeline

The system uses a 6-stage detection pipeline:

1. **Data Retrieval** - Fetch messages from specified timeframe and sources
2. **Semantic Clustering** - Group similar discussions using embeddings (>85% similarity)
3. **Entity Extraction** - Identify teams, people, projects involved
4. **Temporal Overlap** - Check if teams are working simultaneously (≥3 days overlap)
5. **LLM Verification** - Claude confirms actual duplication vs collaboration
6. **Gap Creation** - Store verified gaps with evidence and recommendations

### Mock Data Scenarios

The system includes 6 realistic scenarios for testing and development:

#### Positive Cases (Should Detect Gaps)
1. **OAuth Duplication** - Platform and Auth teams independently implementing OAuth2
   - 24 messages over 14 days, 2 teams
   - High impact: ~40 engineering hours duplicated

2. **API Redesign** - Mobile and Backend duplicating API restructuring
   - 18 messages over 10 days, 2 teams
   - Medium-high impact: API inconsistency risk

3. **Auth Migration** - Security and Platform duplicating JWT migration
   - 10 messages over 7 days, 2 teams
   - High impact: Security implications

#### Edge Cases (Should NOT Detect)
4. **Similar Topics, Different Scope** - User auth vs service auth
   - Semantically similar but different purposes
   - Tests false positive prevention

5. **Sequential Work** - Team B starts after Team A completes
   - No temporal overlap (60 days apart)
   - Tests temporal overlap detection

6. **Intentional Collaboration** - Teams explicitly coordinating
   - Cross-references present (@team mentions)
   - Tests collaboration vs duplication distinction

### Quick Start with Milestone 3

```bash
# 1. Start all services
docker compose up -d

# 2. Load gap detection scenarios
docker compose exec api python scripts/generate_mock_data.py \
  --scenarios oauth_duplication,api_redesign_duplication,auth_migration_duplication

# 3. Run gap detection
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.6,
    "include_evidence": true
  }'

# 4. List detected gaps
curl "http://localhost:8000/api/v1/gaps?limit=10"

# 5. Explore with Jupyter notebook
jupyter notebook notebooks/gap_analysis_demo.ipynb
```

### Testing Gap Detection

```bash
# Run all gap detection tests
docker compose exec api pytest tests/test_integration/test_duplicate_detection_scenarios.py -v

# Run end-to-end detection tests
docker compose exec api pytest tests/test_integration/test_e2e_gap_detection.py -v

# Test specific scenario
docker compose exec api pytest tests/test_integration/ -k oauth_duplication -v

# Run with performance testing
docker compose exec api pytest tests/test_integration/test_e2e_gap_detection.py::TestEndToEndGapDetection::test_detection_performance -v
```

### Key Documentation

- **[GAP_DETECTION.md](./docs/GAP_DETECTION.md)** - Comprehensive gap detection methodology
  - 6-stage detection pipeline explained
  - Detection criteria and exclusion rules
  - Confidence scoring formula (semantic + team separation + temporal + LLM)
  - Impact assessment methodology (team size, time, criticality)
  - Tuning parameters (similarity threshold, temporal overlap, confidence)
  - Best practices and troubleshooting
  - Performance targets (<5s detection, <20ms entity extraction)

- **[ENTITY_EXTRACTION.md](./docs/ENTITY_EXTRACTION.md)** - Entity extraction guide
  - Extracting teams, people, projects, topics
  - Pattern-based vs NLP-based approaches
  - Entity normalization and deduplication
  - Confidence scoring for extractions
  - Performance optimization techniques
  - Testing patterns and examples

- **[API_EXAMPLES.md](./docs/API_EXAMPLES.md)** - Complete API usage guide
  - Gap detection examples (basic, filtered, with evidence)
  - Listing and pagination patterns
  - Error handling and validation
  - Rate limiting guidelines
  - Python and TypeScript client implementations
  - Webhook integration examples
  - Batch detection and monitoring patterns

### Detection Performance

All performance targets from Milestone 3:

| Operation | Target (p95) | Typical |
|-----------|--------------|---------|
| Full gap detection (30 days) | <5s | 2-3s |
| Entity extraction per message | <20ms | 5-10ms |
| Clustering (1000 messages) | <500ms | 200-300ms |
| LLM verification per gap | <2s | 800ms-1.2s |
| Impact score calculation | <50ms | 10-20ms |

### Example: Detecting Duplicate Work

```bash
# Detect OAuth duplication scenario
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.6,
    "include_evidence": true
  }'

# Expected output:
# {
#   "gaps": [
#     {
#       "id": "gap_...",
#       "type": "DUPLICATE_WORK",
#       "topic": "OAuth2 Implementation",
#       "teams_involved": ["platform-team", "auth-team"],
#       "impact_score": 0.89,
#       "confidence": 0.87,
#       "evidence": [...],
#       "recommendation": "Connect alice@company.com and bob@company.com..."
#     }
#   ],
#   "metadata": {
#     "total_gaps": 1,
#     "messages_analyzed": 24,
#     "detection_time_ms": 2800
#   }
# }
```

### Interactive Analysis with Jupyter

Explore gap detection interactively:

```bash
# Start Jupyter notebook
jupyter notebook notebooks/gap_analysis_demo.ipynb
```

The notebook demonstrates:
- API client setup and usage
- Loading and analyzing mock scenarios
- Running gap detection pipeline
- Visualizing impact scores and confidence
- Entity extraction and team co-occurrence
- Cost estimation (engineering hours wasted)
- Temporal analysis of gaps
- Exporting results for reporting

### Impact Estimation

The system estimates organizational cost for each gap:

```python
# Example impact calculation
impact_score = (
    0.25 * team_size_score +           # How many people affected?
    0.25 * time_investment_score +     # How much time wasted?
    0.20 * project_criticality_score + # How important is this?
    0.15 * velocity_impact_score +     # What's blocked?
    0.15 * duplicate_effort_score      # How much actual duplication?
)

# Impact tiers:
# 0.8-1.0: Critical (100+ hours, multiple large teams)
# 0.6-0.8: High (40-100 hours, 5-10 people)
# 0.4-0.6: Medium (10-40 hours, 2-5 people)
# 0.0-0.4: Low (<10 hours, small scope)
```

## Troubleshooting

### Common Issues and Solutions

#### Docker Services Not Starting

**Problem**: `docker compose up` fails or services don't start

**Solutions**:
```bash
# Check if ports are already in use
lsof -i :8000  # API port
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis
lsof -i :8001  # ChromaDB

# Stop conflicting services or change ports in docker-compose.yml

# Remove old containers and volumes
docker compose down -v
docker compose up -d

# Check logs for specific service
docker compose logs api
docker compose logs postgres
```

#### Database Connection Errors

**Problem**: `asyncpg.exceptions.InvalidCatalogNameError` or connection refused

**Solutions**:
```bash
# Verify PostgreSQL is running
docker compose ps postgres

# Check database exists
docker compose exec postgres psql -U coordination_user -l

# Recreate database
docker compose exec postgres psql -U coordination_user -c "DROP DATABASE IF EXISTS coordination;"
docker compose exec postgres psql -U coordination_user -c "CREATE DATABASE coordination;"

# Run migrations
docker compose exec api alembic upgrade head
```

#### ChromaDB Connection Issues

**Problem**: Vector store operations fail or timeout

**Solutions**:
```bash
# Check ChromaDB service status
docker compose logs chromadb

# Verify ChromaDB is accessible
curl http://localhost:8001/api/v1/heartbeat

# Restart ChromaDB service
docker compose restart chromadb

# Clear and reinitialize ChromaDB data
rm -rf data/chroma/*
docker compose restart chromadb
```

#### Elasticsearch Connection Issues

**Problem**: Elasticsearch operations fail or cluster is unreachable

**Solutions**:
```bash
# Check Elasticsearch service status
docker compose logs elasticsearch

# Verify Elasticsearch is accessible
curl http://localhost:9200
curl http://localhost:9200/_cluster/health

# Check Elasticsearch in health endpoint
curl http://localhost:8000/health/detailed | jq '.services.elasticsearch'

# Restart Elasticsearch service
docker compose restart elasticsearch

# Clear Elasticsearch data and restart
docker compose down
docker volume rm coordination_gap_detector_elasticsearch_data
docker compose up -d elasticsearch

# Monitor Elasticsearch logs in real-time
docker compose logs elasticsearch --follow
```

**Check Elasticsearch Status:**
```bash
# Cluster info
curl http://localhost:9200

# Cluster health (should be green or yellow)
curl http://localhost:9200/_cluster/health?pretty

# List all indices
curl http://localhost:9200/_cat/indices?v

# Check specific index
curl http://localhost:9200/messages/_count
```

#### Search Returns No Results

**Problem**: Search queries return empty results even with mock data

**Solutions**:
```bash
# Verify messages in database
docker compose exec postgres psql -U coordination_user -d coordination -c "SELECT COUNT(*) FROM messages;"

# Check vector store document count
curl http://localhost:8000/api/v1/search/health

# Regenerate mock data and embeddings
docker compose exec api python scripts/generate_mock_data.py --scenarios all --clear

# Verify embeddings were created
curl http://localhost:8000/health/detailed
```

#### Import Errors or Module Not Found

**Problem**: `ModuleNotFoundError` when running tests or application

**Solutions**:
```bash
# Ensure dependencies are installed
uv pip install -e ".[dev]"

# Verify virtual environment is activated
which python  # Should point to .venv/bin/python

# Reinstall dependencies
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

#### Tests Failing

**Problem**: Tests fail with database or fixture errors

**Solutions**:
```bash
# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_api/test_search.py -v

# Clear pytest cache
rm -rf .pytest_cache
pytest --cache-clear

# Check test database connection (uses SQLite in-memory by default)
uv run pytest tests/ -v --tb=short
```

#### API Returns 500 Errors

**Problem**: API endpoints return internal server errors

**Solutions**:
```bash
# Check API logs
docker compose logs api --tail=50 --follow

# Verify all services are healthy
curl http://localhost:8000/health/detailed

# Restart API service
docker compose restart api

# Check for recent code changes that might have broken something
git diff HEAD~1
```

#### Slow Search Performance

**Problem**: Search queries take too long to complete

**Solutions**:
1. **Check document count**: Large vector stores may need optimization
   ```bash
   curl http://localhost:8000/api/v1/search/health
   ```

2. **Reduce search limit**: Use smaller limit values for faster results
   ```json
   {"query": "test", "limit": 5}
   ```

3. **Increase similarity threshold**: Filter out low-relevance results
   ```json
   {"query": "test", "threshold": 0.7}
   ```

4. **Monitor resource usage**:
   ```bash
   docker stats
   ```

#### Environment Variable Issues

**Problem**: Application can't find configuration or API keys

**Solutions**:
```bash
# Verify .env file exists
ls -la .env

# Check environment variables are loaded
docker compose exec api env | grep ANTHROPIC

# Recreate .env from template
cp .env.example .env
# Edit .env with your actual values

# Restart services to pick up new env vars
docker compose down
docker compose up -d
```

#### BM25 Scores All Zero (Milestone 2)

**Problem**: BM25 search returns all documents with score 0.0

**Causes**:
- Query terms not found in documents
- Case mismatch between query and documents
- Elasticsearch index not populated

**Solutions**:
```bash
# 1. Check if Elasticsearch index exists and has documents
curl http://localhost:9200/messages/_count

# 2. Verify documents are indexed
curl http://localhost:9200/messages/_search?pretty

# 3. Reindex messages (regenerate mock data)
docker compose exec api python scripts/generate_mock_data.py --scenarios all --clear

# 4. Test with known query
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "OAuth", "ranking_strategy": "bm25"}'

# 5. Check query terms (case-insensitive matching should work)
# If still failing, check Elasticsearch analyzer settings
curl http://localhost:9200/messages/_mapping?pretty
```

#### Hybrid Search Not Improving Results (Milestone 2)

**Problem**: Hybrid search performs worse than single methods

**Causes**:
- Score normalization issues
- One method dominating
- Inappropriate fusion weights

**Solutions**:
```python
# Try RRF instead of weighted (more robust)
{
  "query": "OAuth implementation",
  "ranking_strategy": "hybrid_rrf"  # Instead of hybrid_weighted
}

# Check score distributions
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth",
    "ranking_strategy": "hybrid_rrf",
    "limit": 5
  }' | jq '.results[].ranking_details'

# Look for:
# - semantic_score and bm25_score values
# - semantic_rank vs bm25_rank
# - Both should contribute to final ranking
```

#### Low Ranking Metrics (Milestone 2)

**Problem**: MRR < 0.5, NDCG@10 < 0.55 in offline evaluation

**Causes**:
- Insufficient training data
- Test queries too difficult
- Relevance judgments too strict
- System underperforming

**Solutions**:
```bash
# 1. Check test query difficulty
python scripts/analyze_query_difficulty.py \
  --queries data/test_queries/queries.jsonl

# 2. Review relevance judgment distribution
python scripts/check_judgments.py \
  --judgments data/test_queries/judgments.jsonl
# Should have mix of 0, 1, 2, 3 scores

# 3. Evaluate per-category
python scripts/evaluate_ranking.py \
  --queries data/test_queries/queries.jsonl \
  --judgments data/test_queries/judgments.jsonl \
  --strategies hybrid_rrf \
  --group-by category

# 4. Identify failure cases
python scripts/find_failure_cases.py \
  --results results/eval.json \
  --threshold-mrr 0.3
# Shows queries with poor performance

# 5. Try different strategies
python scripts/evaluate_ranking.py \
  --strategies semantic,bm25,hybrid_rrf,hybrid_weighted \
  --metrics mrr,ndcg@10
# Compare to find best for your use case
```

#### Feature Extraction Slow (Milestone 2)

**Problem**: Feature extraction takes too long per document

**Solutions**:
```python
# Use minimal feature set
from src.ranking.feature_config import get_minimal_config

config = get_minimal_config()
extractor = FeatureExtractor(config=config)

# Only extract for top-k results (not all results)
# Extract for top 20 instead of top 100

# Profile to find bottleneck
python -m cProfile scripts/profile_feature_extraction.py
```

#### No Gaps Detected (Expected Some) (Milestone 3)

**Problem**: Gap detection returns empty results even with duplicate work scenarios loaded

**Causes**:
- Detection algorithm not fully implemented
- Thresholds too strict (similarity, confidence, impact)
- Temporal overlap window too narrow
- Mock data not loaded correctly

**Solutions**:
```bash
# 1. Verify mock data scenarios are loaded
docker compose exec postgres psql -U coordination_user -d coordination -c \
  "SELECT channel, COUNT(*) FROM messages GROUP BY channel;"

# Expected: Should see #platform, #auth-team, #mobile, #backend, etc.

# 2. Check detection with relaxed thresholds
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 90,
    "min_impact_score": 0.0,
    "include_evidence": true
  }'

# 3. Verify specific scenario is loaded
docker compose exec api python -c "
from src.ingestion.slack.mock_client import MockSlackClient
client = MockSlackClient()
msgs = client.get_scenario_messages('oauth_duplication')
print(f'OAuth scenario: {len(msgs)} messages')
"

# 4. Reload scenarios
docker compose exec api python scripts/generate_mock_data.py \
  --scenarios oauth_duplication,api_redesign_duplication --clear

# 5. Check logs for detection errors
docker compose logs api | grep -i "gap detection"
docker compose logs api | grep -i "error"
```

#### Too Many False Positives (Milestone 3)

**Problem**: Detection flags collaboration as duplication

**Causes**:
- Similarity threshold too low
- LLM not detecting cross-references
- Collaboration markers not recognized

**Solutions**:
```bash
# 1. Increase similarity threshold (more strict)
# In detection request:
{
  "similarity_threshold": 0.90,  # Default: 0.85
  "min_impact_score": 0.7
}

# 2. Check for cross-references in false positives
docker compose exec postgres psql -U coordination_user -d coordination -c \
  "SELECT content FROM messages WHERE content LIKE '%@%team%';"

# 3. Review LLM verification prompt
# Check src/detection/duplicate_work.py for collaboration detection logic

# 4. Test with collaboration scenario (should NOT detect)
docker compose exec api python scripts/generate_mock_data.py \
  --scenarios intentional_collaboration --clear

curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{"timeframe_days": 30, "min_impact_score": 0.0}'

# Should return 0 gaps for collaboration scenario
```

#### Entity Extraction Missing Teams/People (Milestone 3)

**Problem**: Gaps don't show correct teams_involved or evidence is incomplete

**Causes**:
- Team metadata not set in messages
- Entity extraction patterns not matching
- Channel-to-team mapping missing

**Solutions**:
```bash
# 1. Verify messages have team metadata
docker compose exec postgres psql -U coordination_user -d coordination -c \
  "SELECT channel, metadata->'team' as team, COUNT(*)
   FROM messages
   GROUP BY channel, metadata->'team';"

# Expected: Each channel should have team metadata

# 2. Check entity extraction on sample message
docker compose exec api python -c "
from src.analysis.entity_extraction import extract_entities

message = {
    'content': 'We are implementing OAuth in @platform-team',
    'channel': '#platform',
    'metadata': {'team': 'platform-team'}
}

entities = extract_entities(message)
print(f'Extracted: {entities}')
"

# 3. Review extraction patterns
# See docs/ENTITY_EXTRACTION.md for supported patterns

# 4. Add team mapping if needed
# In src/analysis/entity_extraction.py, add channel-to-team mapping:
CHANNEL_TEAM_MAP = {
    '#platform': 'platform-team',
    '#auth-team': 'auth-team',
    # ...
}
```

#### Impact Scores All Similar (Milestone 3)

**Problem**: All detected gaps have similar impact scores

**Causes**:
- Impact scoring not fully implemented
- Mock data lacks variety (same team sizes, similar message counts)
- Features not diverse enough

**Solutions**:
```bash
# 1. Check impact score calculation
docker compose exec api python -c "
from src.detection.impact_scoring import calculate_impact

gap = {
    'teams_involved': ['platform-team', 'auth-team'],
    'evidence': [{'content': 'test'}] * 10,  # 10 messages
    'topic': 'OAuth'
}

score = calculate_impact(gap)
print(f'Impact score: {score}')
print(f'Breakdown: team_size={score.team_size}, time={score.time}')
"

# 2. Verify different scenarios have different characteristics
docker compose exec api python -c "
from src.ingestion.slack.mock_client import MockSlackClient

client = MockSlackClient()
scenarios = ['oauth_duplication', 'api_redesign_duplication', 'auth_migration_duplication']

for name in scenarios:
    msgs = client.get_scenario_messages(name)
    teams = set(m.metadata.get('team') for m in msgs if m.metadata)
    print(f'{name}: {len(msgs)} messages, {len(teams)} teams')
"

# 3. Review impact scoring formula
# See docs/GAP_DETECTION.md "Impact Assessment" section
```

#### Detection Takes Too Long (Milestone 3)

**Problem**: Gap detection exceeds 5s target (p95)

**Performance Targets**:
- Full detection (30 days): <5s (p95)
- Entity extraction: <20ms per message
- Clustering: <500ms for 1000 messages
- LLM verification: <2s per gap

**Solutions**:
```bash
# 1. Check detection time in response metadata
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{"timeframe_days": 30}' | jq '.metadata.detection_time_ms'

# 2. Profile detection pipeline
docker compose exec api python -m cProfile -s cumtime scripts/profile_detection.py

# 3. Reduce timeframe
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 7,
    "min_impact_score": 0.7
  }'

# 4. Enable caching (if implemented)
# Check .env for ENABLE_EMBEDDING_CACHE=true

# 5. Batch LLM calls
# Verify src/detection/duplicate_work.py batches verification calls

# 6. Monitor resource usage
docker stats coordination_gap_detector-api-1

# If memory/CPU high, consider scaling:
docker compose up -d --scale api=2
```

#### Claude API Errors (Milestone 3)

**Problem**: LLM verification fails with 429, 401, or timeout errors

**Common Errors**:
- **429 Too Many Requests**: Rate limit exceeded
- **401 Unauthorized**: Invalid API key
- **408 Timeout**: LLM call took too long
- **500 Internal Error**: Claude API issue

**Solutions**:
```bash
# 1. Check API key is set
docker compose exec api env | grep ANTHROPIC_API_KEY

# 2. Verify API key is valid
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-3-haiku-20240307","messages":[{"role":"user","content":"test"}],"max_tokens":10}'

# 3. Check rate limits and retry logic
docker compose logs api | grep -i "rate limit\|retry\|429"

# 4. Reduce LLM calls (disable verification for testing)
# In .env:
ENABLE_LLM_VERIFICATION=false

# Then detection will use heuristics only

# 5. Increase timeout
# In src/config.py, set:
LLM_TIMEOUT_SECONDS = 30  # Default: 10

# 6. Use faster model
# In src/config.py:
LLM_MODEL = "claude-3-haiku-20240307"  # Faster than sonnet
```

#### Gaps Missing Evidence (Milestone 3)

**Problem**: Detected gaps have empty or incomplete evidence lists

**Causes**:
- Evidence collection not implemented
- Messages not associated with gaps
- Evidence limit too low

**Solutions**:
```bash
# 1. Verify include_evidence parameter
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "include_evidence": true
  }' | jq '.gaps[0].evidence | length'

# Should return > 0

# 2. Check message retrieval
docker compose exec postgres psql -U coordination_user -d coordination -c \
  "SELECT COUNT(*) FROM messages WHERE timestamp > NOW() - INTERVAL '30 days';"

# 3. Verify gap-evidence association
# Check src/services/detection_service.py collect_evidence() function

# 4. Increase evidence limit if needed
# In detection request:
{
  "max_evidence_per_gap": 50  # Default: 20
}
```

### Getting Help

If you encounter issues not covered here:

1. **Check logs**: `docker compose logs <service-name>`
2. **Review documentation**: See [CLAUDE.md](./CLAUDE.md) and [docs/API.md](./docs/API.md)
3. **Run health checks**: `curl http://localhost:8000/health/detailed`
4. **Check GitHub issues**: [github.com/timduly4/coordination_gap_detector/issues](https://github.com/timduly4/coordination_gap_detector/issues)
5. **Verify prerequisites**: Python 3.11+, Docker, UV installed correctly

### Debug Mode

Enable debug logging for more detailed output:

```bash
# In .env file
LOG_LEVEL=DEBUG

# Restart services
docker compose restart api
```

## Deployment

### Docker

```bash
# Build image
docker build -t coordination-detector:latest .

# Run container
docker run -p 8000:8000 --env-file .env coordination-detector:latest
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check status
kubectl get pods -n coordination-prod
```

### Terraform

```bash
# Infrastructure as code
cd terraform
terraform init
terraform plan
terraform apply
```

## Architecture

The system uses a multi-stage detection pipeline:

1. **Data Ingestion** - Stream events from all sources
2. **Entity Extraction** - Identify people, teams, projects
3. **Semantic Indexing** - Embed content for similarity search
4. **Pattern Detection** - Apply gap detection algorithms
5. **Impact Scoring** - Rank gaps by organizational cost
6. **LLM Reasoning** - Generate insights with Claude
7. **Alert Routing** - Notify relevant stakeholders

For detailed architecture documentation, see [CLAUDE.md](./CLAUDE.md).

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Workflow

- Follow PEP 8 style guidelines
- Write tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

### Continuous Integration

This project uses GitHub Actions to automatically run tests on every pull request.

**Automated Testing:**
- All tests run automatically when you open or update a PR
- PRs are blocked from merging if tests fail
- Check the "Actions" tab to see test results
- View detailed logs if tests fail

**Branch Protection:**
The `main` branch is protected and requires:
- All tests to pass before merging
- Pull request reviews (recommended for teams)
- Up-to-date branches

See [.github/BRANCH_PROTECTION.md](.github/BRANCH_PROTECTION.md) for setup instructions.

**Before Pushing:**
```bash
# Run tests locally first
docker compose exec api pytest

# Run specific tests
docker compose exec api pytest tests/test_detection/ -v

# With coverage
docker compose exec api pytest --cov=src
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Resources

- [Information Retrieval Fundamentals](https://nlp.stanford.edu/IR-book/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Claude API Reference](https://docs.anthropic.com/)
- [UV Package Manager](https://github.com/astral-sh/uv)

## Acknowledgments

Built with [Claude Code](https://claude.com/claude-code) for AI-native development.

---

**Status**: Alpha - Active Development
**Python Version**: 3.11+
**Last Updated**: December 2024
