# Coordination Gap Detector - Quick Start Guide

Get up and running with the Coordination Gap Detector in under 5 minutes.

## Prerequisites

- **Docker** and **Docker Compose** installed
- **Python 3.11+** (for local development without Docker)
- **UV** (optional, for faster dependency management)

## Quick Start with Docker (Recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/timduly4/coordination_gap_detector.git
cd coordination_gap_detector
```

### 2. Create Environment File

```bash
cp .env.example .env
```

Edit `.env` and add your API keys (optional for initial testing):
```bash
ANTHROPIC_API_KEY=your_key_here  # Optional for basic testing
SLACK_BOT_TOKEN=your_token       # Not needed yet
GITHUB_TOKEN=your_token           # Not needed yet
```

### 3. Start All Services

```bash
docker-compose up
```

This will start:
- **PostgreSQL** (port 5432) - Database with pgvector
- **Redis** (port 6379) - Cache
- **ChromaDB** (port 8001) - Vector store
- **FastAPI** (port 8000) - API server

Wait for all services to show healthy status (30-60 seconds).

### 4. Verify Everything is Running

**Check API health:**
```bash
curl http://localhost:8000/health
```

**Detailed health check:**
```bash
curl http://localhost:8000/health/detailed
```

**Open API docs:**
```bash
open http://localhost:8000/docs
```

### 5. Run Database Migrations

```bash
docker-compose exec api alembic upgrade head
```

This creates the `sources` and `messages` tables in PostgreSQL.

### 6. Verify Database Tables

```bash
docker-compose exec postgres psql -U coordination_user -d coordination -c "\dt"
```

You should see:
```
          List of relations
 Schema |  Name    | Type  |      Owner
--------+----------+-------+------------------
 public | messages | table | coordination_user
 public | sources  | table | coordination_user
```

## Local Development (Without Docker)

### 1. Install UV (Fast Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 3. Start External Services

You'll still need PostgreSQL, Redis, and ChromaDB. The easiest way is to run them via Docker:

```bash
docker-compose up postgres redis chromadb
```

### 4. Run Migrations

```bash
alembic upgrade head
```

### 5. Start Development Server

```bash
uvicorn src.main:app --reload --port 8000
```

## Project Structure

```
coordination-gap-detector/
├── src/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Settings & environment variables
│   ├── api/
│   │   ├── dependencies.py  # FastAPI dependency injection
│   │   └── routes/          # API endpoints (coming soon)
│   ├── db/
│   │   ├── models.py        # SQLAlchemy models (Message, Source)
│   │   ├── postgres.py      # Database connection management
│   │   └── vector_store.py  # ChromaDB client
│   ├── detection/           # Gap detection algorithms
│   ├── ranking/             # Search quality & ranking
│   ├── search/              # Cross-source search
│   └── ingestion/           # Data source integrations
├── alembic/                 # Database migrations
├── tests/                   # Test suite
├── docker-compose.yml       # Service orchestration
├── Dockerfile               # Application container
└── pyproject.toml          # Project dependencies

```

## Available API Endpoints

### Health Checks
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed service status (Postgres, Redis, ChromaDB)

### Documentation
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

### Root
- `GET /` - API information

## Database Schema

### `sources` table
Represents data sources (Slack, GitHub, Google Docs, etc.):
- `id` (PK) - Unique identifier
- `type` - Source type (slack, github, google_docs)
- `name` - Source name
- `config` - JSON configuration
- `created_at`, `updated_at` - Timestamps

### `messages` table
Stores content from all sources:
- `id` (PK) - Unique identifier
- `source_id` (FK) - References sources.id
- `content` - Message/document content
- `external_id` - ID in source system
- `author` - Author email/username
- `channel` - Channel/repo/doc name
- `thread_id` - Thread identifier
- `timestamp` - When message was created
- `metadata` - JSON additional data
- `embedding_id` - Reference to vector store
- `created_at`, `updated_at` - Timestamps

## Useful Commands

### Docker Commands

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# Stop and remove volumes (fresh start)
docker-compose down -v

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f api

# Rebuild after code changes
docker-compose up --build
```

### Database Commands

```bash
# Access PostgreSQL shell
docker-compose exec postgres psql -U coordination_user -d coordination

# List tables
docker-compose exec postgres psql -U coordination_user -d coordination -c "\dt"

# Check migrations status
docker-compose exec api alembic current

# Create new migration
docker-compose exec api alembic revision --autogenerate -m "description"

# Run migrations
docker-compose exec api alembic upgrade head

# Rollback migration
docker-compose exec api alembic downgrade -1
```

### ChromaDB Commands

```bash
# Check ChromaDB status
curl http://localhost:8001/api/v1/heartbeat

# List collections
curl http://localhost:8001/api/v1/collections

# Get collection info
curl http://localhost:8001/api/v1/collections/coordination_messages
```

### Redis Commands

```bash
# Access Redis CLI
docker-compose exec redis redis-cli

# Ping Redis
docker-compose exec redis redis-cli ping

# Check Redis keys
docker-compose exec redis redis-cli KEYS '*'
```

## Development Workflow

### Hot Reload

Docker Compose mounts source code as volumes, so changes to Python files automatically reload the API server:

1. Edit files in `src/`
2. Save changes
3. API automatically reloads (watch logs)
4. Test at http://localhost:8000

### Running Tests

```bash
# Inside container
docker-compose exec api pytest

# With coverage
docker-compose exec api pytest --cov=src --cov-report=html

# Specific test file
docker-compose exec api pytest tests/test_db/

# Local development
pytest
pytest --cov=src
```

## Troubleshooting

### Services won't start

**Error:** `Cannot connect to Docker daemon`
```bash
# Start Docker Desktop or Docker daemon
```

**Error:** `port is already allocated`
```bash
# Stop conflicting services or change ports in docker-compose.yml
docker-compose down
```

### Database connection fails

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check PostgreSQL logs
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres
```

### ChromaDB connection fails

```bash
# Check ChromaDB is running
docker-compose ps chromadb

# Check ChromaDB logs
docker-compose logs chromadb

# Test ChromaDB directly
curl http://localhost:8001/api/v1/heartbeat
```

### Migrations fail

```bash
# Check current migration status
docker-compose exec api alembic current

# Force migration to specific version
docker-compose exec api alembic stamp head

# Drop all tables and start fresh (DESTRUCTIVE)
docker-compose down -v
docker-compose up -d
docker-compose exec api alembic upgrade head
```

### Code changes not reflecting

```bash
# Rebuild Docker image
docker-compose up --build

# Or restart API service
docker-compose restart api
```

## Next Steps

1. **Add Mock Data** (Milestone 1D)
   - Generate realistic Slack conversation data
   - Seed database with example scenarios

2. **Implement Search** (Milestone 1E-1F)
   - Add vector embeddings
   - Build semantic search API
   - Implement search endpoint

3. **Gap Detection** (Milestone 2-3)
   - Duplicate work detection
   - Missing context identification
   - Stale documentation detection

## Resources

- **Project Documentation**: [README.md](README.md)
- **Detailed Architecture**: [CLAUDE.md](CLAUDE.md)
- **Milestones**: [MILESTONES.md](MILESTONES.md)
- **API Docs**: http://localhost:8000/docs
- **GitHub Issues**: https://github.com/timduly4/coordination_gap_detector/issues

## Getting Help

- Check [GitHub Issues](https://github.com/timduly4/coordination_gap_detector/issues)
- Review logs: `docker-compose logs -f`
- Verify health: `curl http://localhost:8000/health/detailed`

---

**Happy Coding!** 🚀

Built with [Claude Code](https://claude.com/claude-code)
