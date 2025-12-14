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

For comprehensive testing documentation including test structure, options, and troubleshooting, see **[TESTING.md](./TESTING.md)**.

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

**POST /api/v1/search/** - Semantic search across messages

Search for messages using semantic similarity with flexible filtering options.

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation",
    "limit": 5,
    "threshold": 0.7,
    "source_types": ["slack"],
    "channels": ["#engineering"]
  }'
```

**Response:**
```json
{
  "results": [
    {
      "content": "Starting OAuth2 integration...",
      "source": "slack",
      "channel": "#engineering",
      "author": "alice@demo.com",
      "timestamp": "2024-12-01T09:00:00Z",
      "score": 0.89,
      "message_id": 123,
      "external_id": "slack_msg_abc",
      "message_metadata": {}
    }
  ],
  "total": 1,
  "query": "OAuth implementation",
  "query_time_ms": 45,
  "threshold": 0.7
}
```

**Parameters:**
- `query` (required): Search query text (1-1000 characters)
- `limit` (optional): Maximum results to return (1-100, default: 10)
- `threshold` (optional): Minimum similarity score (0.0-1.0, default: 0.0)
- `source_types` (optional): Filter by source types (e.g., ["slack", "github"])
- `channels` (optional): Filter by specific channels
- `date_from` (optional): Filter messages from this date (ISO format)
- `date_to` (optional): Filter messages until this date (ISO format)

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

#### Gap Detection API (Coming Soon)

**POST /api/v1/gaps/detect** - Detect coordination gaps

```bash
POST /api/v1/gaps/detect
{
  "timeframe_days": 30,
  "sources": ["slack", "github", "google_docs"],
  "gap_types": ["duplicate_work", "missing_context"],
  "min_impact_score": 0.7
}
```

*Note: Gap detection endpoints are planned for future milestones.*

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
